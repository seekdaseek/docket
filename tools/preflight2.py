#!/usr/bin/env python3
"""Verify every Day 2 query shape against a live node before trusting any of it.

Day 1 shipped with three gates measured by preflight rather than assumed, and
two of them could have corrupted the whole graph silently. The retrieval layer
has the same exposure in reverse: a pattern this server rejects comes back as
an empty result, and an empty result reads exactly like "the memory does not
contain that". A rejected query and an honest absence must never be confused,
so each shape is exercised here and reported as one of three verdicts.

    python3 tools/preflight2.py            # against a running node
    python3 tools/preflight2.py --json      # machine-readable

Exit codes: 0 all shapes supported · 4 a shape the answerer depends on is
unsupported · 5 could not reach the node.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docket.hydra import HydraClient, HydraError, HydraUnsupported  # noqa: E402
from docket.retrieve import QUERY_SHAPES, Retriever  # noqa: E402

# Shapes the answerer cannot work without. A failure in any of these is fatal;
# the rest degrade a receipt rather than an answer.
REQUIRED = {"claims_all", "claim_by_key", "tip_by_pred", "statement_for_claim",
            "statement_text"}


def connect(args) -> HydraClient:
    """Exactly how ingest_run and extract_run build a client.

    The token default matters: with no token the node answers 401 and the
    first preflight run reported every shape as unmeasured for a reason that
    had nothing to do with the shapes. Same env names, same fallbacks, one
    behaviour across every tool that talks to this node.
    """
    return HydraClient(
        base_url=args.base or os.environ.get("HYDRA_URL", "http://127.0.0.1:8443"),
        admin_url=os.environ.get("HYDRA_ADMIN", "http://127.0.0.1:9090"),
        token=args.token or os.environ.get("HYDRA_TOKEN",
                                           "local-development-token-32-bytes"),
        graph=os.environ.get("HYDRA_GRAPH", "default"),
        namespace=os.environ.get("HYDRA_NAMESPACE", "default"),
        cell=os.environ.get("HYDRA_CELL", "cell-0"),
        consistency="causal",
    )


def run_shape(name, fn) -> dict:
    """supported / unsupported / unmeasured -- never a bare pass or fail."""
    try:
        value = fn()
    except HydraUnsupported as exc:
        return {"shape": name, "verdict": "unsupported", "detail": str(exc)[:200]}
    except HydraError as exc:
        detail = str(exc)[:200]
        low = detail.lower()
        # A parse or shape rejection is the server saying no. A transport or
        # timeout failure says nothing about the shape at all.
        if any(w in low for w in ("parse", "expected", "unsupported", "requires",
                                  "invalid", "only")):
            return {"shape": name, "verdict": "unsupported", "detail": detail}
        return {"shape": name, "verdict": "unmeasured", "detail": detail}
    except Exception as exc:  # noqa: BLE001
        return {"shape": name, "verdict": "unmeasured",
                "detail": f"{type(exc).__name__}: {exc}"[:200]}
    return {"shape": name, "verdict": "supported", "sample": value}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="")
    ap.add_argument("--token", default="")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    client = connect(a)
    try:
        client.wait_ready(seconds=20)
    except Exception as exc:  # noqa: BLE001
        print(f"cannot reach the node: {exc}", file=sys.stderr)
        return 5

    r = Retriever(client)
    results = []

    # 1. the bulk read the gate is built from.
    #
    # The value comes back through run_shape's return rather than being
    # captured in the lambda. An earlier version wrote
    #     lambda: len(claims := r.all_claims()) or 0
    # and a walrus inside a lambda binds in the LAMBDA's scope, so `claims`
    # out here stayed empty. The read had actually returned 5,938 rows; every
    # later shape then reported "no claim in the graph to probe with" and the
    # count printed 0. A tool that diagnoses an empty graph when the graph is
    # full is worse than no tool.
    res = run_shape("claims_all", r.all_claims)
    claims = res.get("sample") if res["verdict"] == "supported" else {}
    claims = claims if isinstance(claims, dict) else {}
    if res["verdict"] == "supported":
        res["sample"] = f"{len(claims)} claims"
    results.append(res)

    sample_key = next(iter(claims), None)
    sample = claims.get(sample_key) if sample_key else None

    if sample_key is None:
        for name in ("claim_by_key", "tip_by_pred", "tip_by_pred_subj",
                     "supersedes", "statement_for_claim", "statement_text",
                     "sspaths"):
            results.append({"shape": name, "verdict": "unmeasured",
                            "detail": "no claim in the graph to probe with"})
    else:
        ts = int(sample.get("ts") or 0) + 1
        results.append(run_shape(
            "claim_by_key", lambda: (r.claim(sample_key) or {}).get("pred")))
        results.append(run_shape(
            "tip_by_pred", lambda: len(r.tip(sample["pred"], ts))))
        results.append(run_shape(
            "tip_by_pred_subj",
            lambda: len(r.tip(sample["pred"], ts, subject=sample["subj"]))))
        # Probe the chain with a claim that actually has one. Only ~189 of
        # 5,938 claims carry a SUPERSEDES edge, so a random sample returns 0
        # and "supported, 0 rows" says nothing about whether the walk works.
        chained = None
        try:
            rows = client.query(
                "MATCH (a:Claim)-[:SUPERSEDES]->(b:Claim) RETURN a.nkey").rows
            for row in rows:
                if row.get("a.nkey"):
                    chained = row["a.nkey"]
                    break
        except Exception:  # noqa: BLE001
            chained = None
        walk_key = chained or sample_key
        res_walk = run_shape("supersedes",
                             lambda: len(r.superseded_by(walk_key)))
        if chained is None and res_walk["verdict"] == "supported":
            res_walk["sample"] = (f"{res_walk['sample']} (no chained claim "
                                  f"found to probe with)")
        results.append(res_walk)
        stmt_res = run_shape("statement_for_claim",
                             lambda: r.statement_for(sample_key))
        stmt = stmt_res.get("sample") or {}
        stmt = stmt if isinstance(stmt, dict) else {}
        if stmt_res["verdict"] == "supported":
            stmt_res["sample"] = stmt.get("nkey")
        results.append(stmt_res)
        if stmt.get("nkey"):
            results.append(run_shape(
                "statement_text",
                lambda: len(r.statement_text(stmt["nkey"]) or "")))
        else:
            results.append({"shape": "statement_text", "verdict": "unmeasured",
                            "detail": "no statement found for the sample claim"})
        results.append(run_shape(
            "sspaths", lambda: len(r.evidence_path(walk_key))))

    if a.json:
        print(json.dumps({"results": results,
                          "queries": client.queries}, indent=2, default=str))
    else:
        width = max(len(x["shape"]) for x in results)
        for x in results:
            mark = {"supported": "OK  ", "unsupported": "NO  ",
                    "unmeasured": "??  "}[x["verdict"]]
            extra = x.get("sample", x.get("detail", ""))
            print(f"{mark}{x['shape']:<{width}}  {extra}")
            print(f"      {QUERY_SHAPES.get(x['shape'], '')}")
        print(f"\n{client.queries} queries")

    broken = [x["shape"] for x in results
              if x["shape"] in REQUIRED and x["verdict"] != "supported"]
    if broken:
        print(f"\nREQUIRED shapes not supported: {', '.join(broken)}. "
              f"The answerer would return empty results that look like "
              f"honest absences. Fix these before running the harness.",
              file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

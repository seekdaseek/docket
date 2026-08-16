#!/usr/bin/env python3
"""Why did retrieval miss? Three answers, three different fixes.

    python3 tools/diagnose_misses.py
    python3 tools/diagnose_misses.py --limit 10 --depth 200

For every question whose evidence contained no gold session, this decides
between:

  EXTRACTION GAP   the gold sessions produced no claim at all, or none that
                   mentions the answer. The graph never held the fact. No
                   amount of gate tuning helps; this costs a re-extraction.

  RANKING GAP      the gold claim exists and the gate scored it, but below the
                   candidate cut. Free to fix -- weights, stoplist, candidate
                   count -- and the cheapest possible win.

  FILTER GAP       the gold claim was a candidate and something downstream
                   dropped it: said after the question date, or missing from
                   the graph. Neither is a retrieval problem.

Runs entirely offline against ~/docket/state/claims.jsonl. No model, no
database, no spend -- which is the point: knowing which of the three you have
is worth more than guessing at the one that costs $10 to try.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docket.extract import claim_key  # noqa: E402
from docket.gate import build_gate  # noqa: E402

STATE = Path(os.environ.get("DOCKET_STATE", Path.home() / "docket" / "state"))

EXTRACTION = "extraction gap: the graph never held it"
RANKING = "ranking gap: the claim exists but scored too low"
FILTER = "filter gap: it was a candidate and got dropped"
UNKNOWN = "unclear"


def load_claims(path: Path) -> tuple[dict, dict]:
    """{claim_key: props}, {sid: [claims]} from the current prompt version only.

    The file is append-only and holds every generation, so the last row per
    session wins and older prompt versions are skipped -- mixing them silently
    changes any count taken from this file.
    """
    rows: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rows[row.get("sid")] = row
    claims: dict[str, dict] = {}
    by_session: dict[str, list] = {}
    for sid, row in rows.items():
        if row.get("prompt_version", 0) < 2 or row.get("status") != "measured":
            continue
        ts = int(row.get("ts") or 0)
        kept = []
        for c in row.get("claims") or []:
            key = claim_key(sid, c["turn"], c["subject"], c["predicate"], c["object"])
            props = {"subj": c["subject"], "pred": c["predicate"], "obj": c["object"],
                     "kind": c["kind"], "card": c.get("cardinality", "many"),
                     "sid": sid, "turn": c["turn"], "ts": ts}
            claims[key] = props
            kept.append(props)
        by_session[sid] = kept
    return claims, by_session


def load_answers(path: Path) -> list[dict]:
    rows: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rows[row.get("question_id")] = row
    return list(rows.values())


def classify(row: dict, gate, by_session: dict, depth: int) -> dict:
    gold = set(row.get("gold_sessions") or [])
    gold_claims = [c for sid in gold for c in by_session.get(sid, [])]
    retrieval = row.get("retrieval") or {}

    out = {
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "gold": row.get("gold"),
        "gold_sessions": sorted(gold),
        "gold_claims": len(gold_claims),
        "dropped_future": retrieval.get("dropped_future"),
        "dropped_missing": retrieval.get("dropped_missing"),
        "hits": retrieval.get("hits"),
        "kept": retrieval.get("kept"),
    }

    if not gold_claims:
        out["verdict"] = EXTRACTION
        out["detail"] = ("the gold sessions produced no claims at this prompt "
                         "version")
        return out

    # Where would the gold claims rank if the gate looked much deeper?
    deep = gate.search(row.get("question") or "", limit=depth)
    ranks = [i for i, h in enumerate(deep, 1) if h.claim.get("sid") in gold]
    out["best_rank"] = ranks[0] if ranks else None
    out["gold_in_top_depth"] = len(ranks)

    cut = int(retrieval.get("candidates") or 0)
    # A claim ranked below the candidate cut was NEVER a candidate, so a
    # dropped_future count alongside it is coincidence, not cause. The first
    # version of this classifier called those FILTER and reported 51 of them
    # when the true number was near zero -- the misattribution sent the whole
    # diagnosis at the as-of rule instead of at ranking.
    if ranks and cut and ranks[0] > cut:
        out["verdict"] = RANKING
        out["detail"] = (f"gold claim ranked {ranks[0]} of {depth}; the "
                         f"candidate cut is {cut}, so it was never a candidate")
    elif ranks and out["dropped_future"]:
        out["verdict"] = FILTER
        out["detail"] = (f"gold claim ranked {ranks[0]}, inside the cut of "
                         f"{cut}, and {out['dropped_future']} candidates were "
                         f"dropped as said-after-the-question")
    elif ranks:
        out["verdict"] = RANKING
        out["detail"] = (f"gold claim ranked {ranks[0]} of {depth}, inside "
                         f"the cut, but did not reach the evidence")
    else:
        out["verdict"] = EXTRACTION
        out["detail"] = (f"{len(gold_claims)} claims exist for the gold "
                         f"sessions but none scores at all on this question: "
                         f"the fact is not in the words that were extracted")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=str(STATE / "answers.jsonl"))
    ap.add_argument("--claims", default=str(STATE / "claims.jsonl"))
    ap.add_argument("--depth", type=int, default=200,
                    help="how deep to look for the gold claim before calling "
                         "it an extraction gap")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    for p in (a.answers, a.claims):
        if not Path(p).exists():
            print(f"missing {p}", file=sys.stderr)
            return 4

    claims, by_session = load_claims(Path(a.claims))
    gate = build_gate(claims)
    answers = load_answers(Path(a.answers))

    misses = []
    for row in answers:
        if row.get("is_abstention"):
            continue
        gold = set(row.get("gold_sessions") or [])
        cited = {e.get("session") for e in (row.get("evidence") or [])}
        if gold & cited:
            continue
        misses.append(classify(row, gate, by_session, a.depth))

    tally = Counter(m["verdict"] for m in misses)
    if a.json:
        print(json.dumps({"misses": misses, "tally": dict(tally),
                          "indexed_claims": gate.size}, indent=2))
        return 0

    print(f"indexed {gate.size} claims from {len(by_session)} sessions")
    print(f"{len(misses)} misses out of "
          f"{sum(1 for r in answers if not r.get('is_abstention'))} scoreable\n")
    for verdict, n in tally.most_common():
        print(f"  {n:>3}  {verdict}")
    print()
    for m in misses[:a.limit]:
        print(f"[{m['verdict'].split(':')[0].upper()}] {m['question']}")
        print(f"    gold          : {m['gold']}")
        print(f"    gold sessions : {len(m['gold_sessions'])}, "
              f"holding {m['gold_claims']} claims")
        if m.get("best_rank"):
            print(f"    best gold rank: {m['best_rank']} "
                  f"({m['gold_in_top_depth']} gold claims in top {a.depth})")
        print(f"    retrieval     : hits={m['hits']} kept={m['kept']} "
              f"dropped_future={m['dropped_future']} "
              f"dropped_missing={m['dropped_missing']}")
        print(f"    -> {m['detail']}\n")

    if tally.get(EXTRACTION):
        print("An extraction gap is the only one that costs money to close. "
              "Everything else is a free tuning change, so exhaust those first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

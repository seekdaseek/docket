#!/usr/bin/env python3
"""As-of time travel: one fact, three dates, three correct answers.

    python3 tools/timetravel.py                    # pick the best chain itself
    python3 tools/timetravel.py --pred lives_in    # a specific predicate
    python3 tools/timetravel.py --list 20          # what changed, ranked

No model, no gate, no spend. Every line printed comes back from HydraDB, and
the Cypher that produced it is printed above the result. That is the whole
demonstration: the current answer is the tip of a chain, the chain is still in
the graph, and asking the same question with an earlier as-of date returns the
earlier value -- which was correct then and is not wrong now, only superseded.

Written for the ≤3 minute video, where the alternative is describing this in
words while a judge takes it on trust.

WHY THIS IS HYDRADB DOING THE WORK RATHER THAN PYTHON. The as-of read is a
range predicate plus an ordering plus a limit, executed by the node:

    MATCH (c:Claim {pred: $p}) WHERE c.ts <= <t> RETURN ... ORDER BY c.ts DESC LIMIT 1

Nothing is filtered in this process. Change the literal `t` and the database
returns a different tip. The supersede chain is a variable-length traversal
from a fixed id, which is the shape the probes measured as the only one the
engine accepts.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docket.hydra import HydraClient  # noqa: E402
from docket.retrieve import Retriever  # noqa: E402

BAR = "=" * 72


def connect(args) -> HydraClient:
    """Same construction as every other tool here, same token fallback."""
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


def when(ts):
    if ts is None:
        return "unknown"
    try:
        return datetime.datetime.fromtimestamp(
            int(ts), datetime.timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OSError, OverflowError):
        return str(ts)


def changed_facts(claims, max_values: int = 8, max_mentions: int = 40):
    """Predicates that REPLACE their value over time, best first.

    MEASURED Aug 17 and it broke the first version of this tool: ranking by
    the NUMBER of distinct values picks exactly the wrong thing. `user owns`
    took 523 values across 538 mentions -- because the user owns 523 different
    objects, not because anything changed. A predicate like that ACCUMULATES,
    and `ORDER BY ts DESC LIMIT 1` over it returns "the most recent thing
    mentioned", which is not a current value and not time travel.

    A fact worth demonstrating REPLACES itself: `lives_in` goes Seattle ->
    Las Vegas -> Boston -> LA and only one of those is true at a time. Those
    have FEW values, not many. Hence the caps, and hence ranking by the
    FEWEST values rather than the most, with the time span breaking ties.

    A predicate restated with the same object is still not a change; that was
    right in the first version and is kept.
    """
    by_pred = defaultdict(list)
    for c in claims.values():
        pred, obj, ts = c.get("pred"), c.get("obj"), c.get("ts")
        if pred and obj is not None and ts is not None:
            by_pred[(pred, c.get("subj"))].append(c)

    out = []
    for (pred, subj), rows in by_pred.items():
        rows.sort(key=lambda c: c.get("ts") or 0)
        distinct = []
        for r in rows:
            v = str(r.get("obj"))
            if not distinct or distinct[-1] != v:
                distinct.append(v)
        unique = len(set(distinct))
        if unique < 2:
            continue
        # An accumulating list, not a fact that changed.
        if unique > max_values or len(rows) > max_mentions:
            continue
        span = (rows[-1].get("ts") or 0) - (rows[0].get("ts") or 0)
        out.append({"pred": pred, "subj": subj, "rows": rows,
                    "distinct": distinct, "unique": unique, "span": span,
                    "chain": 0})
    out.sort(key=lambda d: (d["unique"], -d["span"]))
    return out


def rank_by_chain(ret, facts, probe: int = 40):
    """Promote facts the SYSTEM ITSELF marked as superseding.

    The ingest writes a SUPERSEDES edge when it judges one claim replaces
    another. Those are the facts where the chain beat actually works, and
    using the graph's own judgement beats any heuristic here -- a demo picked
    by my rule and a demo picked by the pipeline's rule are not equally
    honest.

    Chains are sparse in this data (roughly 189 links over 5,687 claims), so
    most facts will come back with nothing. That is reported, not hidden.
    """
    for f in facts[:probe]:
        try:
            f["chain"] = len(ret.superseded_by(f["rows"][-1].get("nkey")))
        except Exception:
            f["chain"] = 0
    facts.sort(key=lambda d: (-d["chain"], d["unique"], -d["span"]))
    return facts


def marks_for(rows, cap: int = 6):
    """The timestamps to ask at: one per CHANGE of value, deduped, capped.

    The first version emitted one mark per value change and printed 500 lines
    for a predicate with 500 values. It also emitted duplicates, because many
    claims share a timestamp -- same conversation, same second.
    """
    out, seen_value, seen_ts = [], None, set()
    for r in rows:
        v, ts = str(r.get("obj")), r.get("ts")
        if v != seen_value and ts not in seen_ts:
            out.append(ts)
            seen_ts.add(ts)
            seen_value = v
    if len(out) > cap:
        # Keep the first and last change, sample the middle evenly.
        step = (len(out) - 1) / float(cap - 1)
        out = [out[int(round(i * step))] for i in range(cap)]
    return [rows[0].get("ts", 0) - 86_400] + out


def show(ret, fact) -> None:
    pred, subj, rows = fact["pred"], fact["subj"], fact["rows"]
    print(BAR)
    print(f"FACT   {subj} {pred}")
    print(f"       took {fact['unique']} values across {len(rows)} mentions"
          + (f", {fact['chain']} linked by SUPERSEDES" if fact.get("chain")
             else ", none linked by SUPERSEDES"))
    print(BAR)
    print()
    print("What the graph holds, oldest first:")
    shown, last = 0, None
    for r in rows:
        v = str(r.get("obj"))
        if v == last:
            continue
        last = v
        shown += 1
        if shown > 12:
            print(f"  ... and {len(rows) - shown} more mentions")
            break
        print(f"  {when(r.get('ts'))}   {r.get('obj')}")
    print()

    # Ask the same question at each point the value changed, plus one date
    # before anything was known -- the answer there must be nothing.
    marks = marks_for(rows)

    print("The same read, executed by the node, with only the timestamp changed:")
    print()
    for t in marks:
        cypher = (f"MATCH (c:Claim {{pred: $p, subj: $s}}) WHERE c.ts <= {int(t)} "
                  f"RETURN c.obj, c.ts ORDER BY c.ts DESC LIMIT 1")
        tip = ret.tip(pred, int(t), subject=subj, limit=1)
        print(f"  as of {when(t)}")
        print(f"    {cypher}")
        if tip:
            print(f"    -> {tip[0].get('obj')}   "
                  f"(said {when(tip[0].get('ts'))}, claim {tip[0].get('nkey')})")
        else:
            print("    -> nothing. The graph held no claim about this yet, and "
                  "says so rather than guessing.")
        print()

    # The chain itself, walked in the database from the newest claim.
    newest = rows[-1]
    chain = ret.superseded_by(newest.get("nkey"))
    print("The chain behind the current value, walked from a fixed id:")
    print(f"  MATCH (a:Claim {{id: <{newest.get('nkey')}>}})"
          f"-[:SUPERSEDES*1..5]->(b:Claim) RETURN b.obj, b.ts")
    if chain:
        for c in chain:
            print(f"    <- {c.get('obj')}   (said {when(c.get('ts'))})")
        print()
        print("  Nothing was overwritten. The earlier values are still there, "
              "which is what makes the reads above possible.")
    else:
        print("    none. The extractor did not link these claims, so the "
              "as-of reads above stand on the timestamps alone.")
        print("    Chains are sparse in this data by design -- SUPERSEDES is "
              "the audit trail, not the retrieval path.")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="show this predicate instead of choosing")
    ap.add_argument("--subj")
    ap.add_argument("--max-values", type=int, default=8,
                    help="above this many distinct values a predicate is an "
                         "accumulating list, not a fact that changed")
    ap.add_argument("--list", type=int, metavar="N",
                    help="list the N facts that changed most, then stop")
    ap.add_argument("--base", default="")
    ap.add_argument("--token", default="")
    a = ap.parse_args()

    client = connect(a)
    client.wait_ready(seconds=30)
    ret = Retriever(client)
    claims = ret.all_claims()
    if not claims:
        print("the graph holds no Claim nodes", file=sys.stderr)
        return 4
    print(f"{len(claims)} claims in the graph\n")

    facts = changed_facts(claims, max_values=a.max_values)
    if facts and not a.pred:
        facts = rank_by_chain(ret, facts)
    if not facts:
        print("no predicate in this graph took more than one value", file=sys.stderr)
        return 4

    if a.list:
        print(f"{len(facts)} facts changed value. Top {a.list}:\n")
        for f in facts[:a.list]:
            days = int(f["span"] / 86_400)
            print(f"  {f['unique']} values over {days:>4}d  "
                  f"chain={f['chain']}   "
                  f"{f['subj']} {f['pred']}")
            print(f"      {' -> '.join(f['distinct'][:5])}")
        return 0

    if a.pred:
        picked = [f for f in facts if f["pred"] == a.pred
                  and (a.subj is None or f["subj"] == a.subj)]
        if not picked:
            print(f"no changing fact for predicate {a.pred!r}", file=sys.stderr)
            print("try: python3 tools/timetravel.py --list 20", file=sys.stderr)
            return 4
        # Probe the chain for THIS fact even though the ranking was skipped.
        # Without it the header printed "none linked by SUPERSEDES" and the
        # walk below then printed the chain -- the tool contradicting itself
        # on screen.
        rank_by_chain(ret, picked[:1])
        show(ret, picked[0])
        return 0

    show(ret, facts[0])
    print(f"({len(facts)} other facts in this graph changed value the same way; "
          f"`--list 20` shows them.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

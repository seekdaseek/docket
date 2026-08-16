#!/usr/bin/env python3
"""Does LongMemEval's `question_date` actually bound its evidence?

    python3 tools/date_audit.py --data ~/docket/data/longmemeval_oracle.json

Found Aug 16, on a question whose retrieval returned nothing: both gold
sessions were dated AFTER the question. Asked 06:04, evidence at 06:19 and
23:31 the same day. A strict `ts <= question_date` filter drops the answer.

That filter is the project's own idea -- as-of retrieval is the differentiator
-- so before loosening it, measure how the dataset actually behaves. This
reports, over every instance:

  how many have evidence dated after the question
  by how much, as a distribution rather than an average
  what tolerance would keep every instance's evidence

and separately the same for NON-evidence sessions, because a tolerance wide
enough to admit the answer must not also admit the whole haystack. If the two
distributions overlap, no tolerance separates them and the honest design is to
keep as-of as a demonstrated capability rather than a hard benchmark filter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docket import dataset  # noqa: E402

HOUR = 3600
DAY = 86400
TOLERANCES = [0, HOUR, 6 * HOUR, DAY, 2 * DAY, 7 * DAY, 30 * DAY]


def label(seconds: int) -> str:
    if seconds == 0:
        return "strict"
    if seconds < DAY:
        return f"+{seconds // HOUR}h"
    return f"+{seconds // DAY}d"


def buckets(offsets: list[float]) -> dict:
    """Offsets in seconds -> a readable distribution."""
    out = {"<=0 (before the question)": 0, "0-1h": 0, "1-6h": 0, "6-24h": 0,
           "1-2d": 0, "2-7d": 0, ">7d": 0}
    for o in offsets:
        if o <= 0:
            out["<=0 (before the question)"] += 1
        elif o <= HOUR:
            out["0-1h"] += 1
        elif o <= 6 * HOUR:
            out["1-6h"] += 1
        elif o <= DAY:
            out["6-24h"] += 1
        elif o <= 2 * DAY:
            out["1-2d"] += 1
        elif o <= 7 * DAY:
            out["2-7d"] += 1
        else:
            out[">7d"] += 1
    return out


def audit(instances) -> dict:
    ev_offsets: list[float] = []      # per evidence session
    ev_worst: list[float] = []        # per instance, the latest evidence
    other_offsets: list[float] = []   # per non-evidence session
    instances_with_late_evidence = 0
    no_evidence = 0

    for inst in instances:
        asked = inst.asked_at.timestamp()
        gold = set(inst.evidence_session_ids or [])
        ev, other = [], []
        for s in inst.sessions:
            delta = s.when.timestamp() - asked
            (ev if s.session_id in gold else other).append(delta)
        if not ev:
            no_evidence += 1
            continue
        ev_offsets.extend(ev)
        other_offsets.extend(other)
        worst = max(ev)
        ev_worst.append(worst)
        if worst > 0:
            instances_with_late_evidence += 1

    keeps = {}
    for tol in TOLERANCES:
        kept = sum(1 for w in ev_worst if w <= tol)
        admitted = sum(1 for o in other_offsets if 0 < o <= tol)
        keeps[label(tol)] = {
            "instances_whose_evidence_all_fits": kept,
            "pct": round(kept / len(ev_worst) * 100, 1) if ev_worst else None,
            "non_evidence_sessions_admitted": admitted,
        }

    return {
        "instances": len(instances),
        "instances_without_evidence_sessions": no_evidence,
        "instances_with_evidence_after_the_question": instances_with_late_evidence,
        "pct_with_late_evidence": round(
            instances_with_late_evidence / len(ev_worst) * 100, 1) if ev_worst else None,
        "evidence_sessions": len(ev_offsets),
        "non_evidence_sessions": len(other_offsets),
        "evidence_offset_distribution": buckets(ev_offsets),
        "non_evidence_offset_distribution": buckets(other_offsets),
        "tolerance_table": keeps,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser(
        "~/docket/data/longmemeval_oracle.json"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if not Path(a.data).exists():
        print(f"missing {a.data}", file=sys.stderr)
        return 4
    out = audit(dataset.load(a.data, limit=a.limit))

    if a.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"instances                    {out['instances']}")
    print(f"with evidence AFTER the question  "
          f"{out['instances_with_evidence_after_the_question']} "
          f"({out['pct_with_late_evidence']}%)")
    print(f"\nevidence sessions, offset from the question date")
    for k, v in out["evidence_offset_distribution"].items():
        print(f"  {k:<28} {v}")
    print(f"\nnon-evidence sessions, same measure")
    for k, v in out["non_evidence_offset_distribution"].items():
        print(f"  {k:<28} {v}")
    print(f"\ntolerance    evidence kept        non-evidence admitted")
    for k, v in out["tolerance_table"].items():
        print(f"  {k:<9}  {v['instances_whose_evidence_all_fits']:>4} "
              f"({v['pct']}%)          {v['non_evidence_sessions_admitted']}")
    print("\nPick the smallest tolerance that keeps the evidence. If none does "
          "without admitting most of the haystack, question_date is not an "
          "as-of boundary in this dataset and the filter must not be one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

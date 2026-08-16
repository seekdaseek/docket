#!/usr/bin/env python3
"""Score the retrieval, before anything is spent on answering.

    python3 tools/score_retrieval.py                       # ~/docket/state/answers.jsonl
    python3 tools/score_retrieval.py --answers path --json

LongMemEval ships `answer_session_ids`: the sessions that actually contain the
answer. Every row written by answer_run carries them as `gold_sessions`, so
recall is checkable offline with no model and no database.

Two numbers, and they mean different things:

  RECALL   did the evidence include at least one gold session? If this is low
           the answerer cannot succeed however good the model is, because the
           right sentence was never put in front of it.
  DENSITY  what fraction of the cited claims came from a gold session. Low
           density with high recall means the model will be reasoning over
           mostly-irrelevant evidence, which is a prompt problem rather than a
           retrieval one.

Abstention questions (`_abs`) are counted separately and never averaged in:
they have no gold sessions by construction, and the only thing worth measuring
there is whether the system refused.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

DEFAULT = Path(os.environ.get("DOCKET_STATE",
                              Path.home() / "docket" / "state")) / "answers.jsonl"


def load(path: Path) -> list[dict]:
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
            # append-only: the last row for a question id wins
            rows[row.get("question_id")] = row
    return list(rows.values())


def score(rows: list[dict]) -> dict:
    real = [r for r in rows if not r.get("is_abstention")]
    absts = [r for r in rows if r.get("is_abstention")]

    by_type: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "hit": 0, "cited": 0, "gold_cited": 0, "empty": 0})
    hits = cited_total = gold_total = empty = 0

    for row in real:
        gold = set(row.get("gold_sessions") or [])
        ev = row.get("evidence") or []
        sessions = [e.get("session") for e in ev]
        hit = bool(gold & set(sessions))
        t = by_type[row.get("question_type") or "unknown"]
        t["n"] += 1
        t["cited"] += len(ev)
        t["gold_cited"] += sum(1 for s in sessions if s in gold)
        if hit:
            t["hit"] += 1
            hits += 1
        if not ev:
            t["empty"] += 1
            empty += 1
        cited_total += len(ev)
        gold_total += sum(1 for s in sessions if s in gold)

    refused = sum(1 for r in absts if r.get("status") == "absent")
    drops = {"dropped_future": 0, "dropped_missing": 0, "gate_hits": 0,
             "admitted_by_tolerance": 0}
    for row in rows:
        ret = row.get("retrieval") or {}
        drops["dropped_future"] += int(ret.get("dropped_future") or 0)
        drops["dropped_missing"] += int(ret.get("dropped_missing") or 0)
        drops["gate_hits"] += int(ret.get("hits") or 0)
        drops["admitted_by_tolerance"] += int(ret.get("admitted_by_tolerance") or 0)
    return {
        "questions": len(real),
        "recall": round(hits / len(real), 4) if real else None,
        "density": round(gold_total / cited_total, 4) if cited_total else None,
        "no_evidence": empty,
        "cited_total": cited_total,
        "gold_cited": gold_total,
        "drops": drops,
        "abstention": {"n": len(absts), "refused": refused,
                       "rate": round(refused / len(absts), 4) if absts else None},
        "by_type": {k: {**v,
                        "recall": round(v["hit"] / v["n"], 4) if v["n"] else None}
                    for k, v in sorted(by_type.items())},
        "status_counts": {s: sum(1 for r in rows if r.get("status") == s)
                          for s in sorted({r.get("status") for r in rows})},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=str(DEFAULT))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--misses", type=int, default=0,
                    help="print N questions whose evidence missed every gold session")
    a = ap.parse_args()

    path = Path(a.answers)
    if not path.exists():
        print(f"no answers file at {path}; run tools/answer_run.py first")
        return 4
    rows = load(path)
    if not rows:
        print(f"{path} holds no rows")
        return 4
    out = score(rows)

    if a.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"answers          {len(rows)}  ({out['questions']} scoreable, "
              f"{out['abstention']['n']} abstention)")
        print(f"status           {out['status_counts']}")
        print(f"RECALL           {out['recall']}   "
              f"a gold session was cited for this fraction of questions")
        print(f"density          {out['density']}   "
              f"({out['gold_cited']}/{out['cited_total']} cited claims from gold sessions)")
        print(f"no evidence      {out['no_evidence']}")
        d = out["drops"]
        print(f"gate/drops       {d['gate_hits']} candidates, "
              f"{d['dropped_future']} said after the question, "
              f"{d['dropped_missing']} not in the graph, "
              f"{d['admitted_by_tolerance']} kept by the as-of tolerance")
        if out["abstention"]["n"]:
            print(f"abstention       {out['abstention']['refused']}/"
                  f"{out['abstention']['n']} refused")
        print("\nby question type")
        width = max(len(k) for k in out["by_type"]) if out["by_type"] else 10
        for name, v in out["by_type"].items():
            print(f"  {name:<{width}}  n={v['n']:<4} recall={v['recall']}  "
                  f"cited={v['cited']}  from_gold={v['gold_cited']}  "
                  f"no_evidence={v['empty']}")

    if a.misses:
        print("\nmisses (evidence cited, no gold session among them)")
        shown = 0
        for row in rows:
            if row.get("is_abstention"):
                continue
            gold = set(row.get("gold_sessions") or [])
            ev = row.get("evidence") or []
            if gold & {e.get("session") for e in ev}:
                continue
            shown += 1
            print(f"\n  [{row.get('question_type')}] {row.get('question')}")
            print(f"    gold answer   : {row.get('gold')}")
            print(f"    gold sessions : {sorted(gold)}")
            for e in ev[:4]:
                print(f"    cited         : {e.get('triple')}  "
                      f"session={e.get('session')} terms={list((e.get('matched_terms') or {}))[:3]}")
            if not ev:
                print(f"    cited         : nothing. {row.get('reason')}")
            if shown >= a.misses:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

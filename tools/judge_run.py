#!/usr/bin/env python3
"""Grade answers.jsonl with a disclosed model judge.

    python3 tools/judge_run.py                                  # ~/docket/state/answers.jsonl
    python3 tools/judge_run.py --limit 25                       # pilot, measure the spend first
    python3 tools/judge_run.py --answers state/baseline.jsonl --out state/judged-baseline.jsonl
    python3 tools/judge_run.py --free                           # no spend: abstention + refusals only

Resumable in the same way as answer_run: verdicts append one row per question
and an id already judged is skipped, so a closed lid costs nothing.

WHAT THIS COSTS. One judge call per ANSWERED answerable question, and nothing
at all for abstention questions, refusals, or unmeasured rows -- those are
decided from `status`. Run `--limit 25` first: the tool prints its real token
usage, so the full run is priced from a measurement rather than an estimate.

WHICH MODEL JUDGES. Defaults to the same model string the project already uses
(docket.llm.DEFAULT_MODEL) because that string is verified present in this
codebase. If you want a stronger judge, pass --model with the exact id from
your console. Never guess a model string. The id used is written onto every
judged row and printed in the header, because an undisclosed judge is not a
method.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docket import judge as J  # noqa: E402
from docket.llm import Anthropic, load_env  # noqa: E402

STATE = Path(os.environ.get("DOCKET_STATE", Path.home() / "docket" / "state"))
ANSWERS = STATE / "answers.jsonl"
JUDGED = STATE / "judged.jsonl"
ENV = Path(os.environ.get("DOCKET_ENV", Path.home() / "docket" / ".env"))


def load_rows(path: Path) -> list[dict]:
    """Append-only file: the last row for a question id wins."""
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
            qid = row.get("question_id")
            if qid:
                rows[qid] = row
    return list(rows.values())


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r.get("question_id") for r in load_rows(path)}


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def make_llm(model: str | None):
    env = load_env(str(ENV))
    key = env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit(f"no ANTHROPIC_API_KEY in {ENV}")
    return Anthropic(key, model=model) if model else Anthropic(key)


def report(out: dict) -> None:
    acc = out["accuracy"]
    print(f"\nanswerable       {out['answerable']}  "
          f"({out['graded']} graded, {out['unjudged']} unjudged)")
    print(f"ACCURACY         {acc if acc is not None else 'n/a'}   "
          f"{out['correct']} correct / {out['graded']} graded")
    a = out["abstention"]
    print(f"abstention       {a['refused']}/{a['n']} refused, "
          f"{a['hallucinated']} answered anyway, {a['unjudged']} unjudged")
    c = out["calibration"]
    denom = out["graded"] - out["retrieval_misses"]
    print(f"refusals         {c['false_refusals']} with gold in evidence "
          f"(calibration)  +  {out['retrieval_misses']} with gold never "
          f"retrieved (retrieval)")
    print(f"CALIBRATION      false refusals {c['false_refusals']}/{denom} "
          f"({c['false_refusal_rate']})  vs  abstention refusal rate "
          f"{c['abstention_refusal_rate']}")
    print("                 neither number means anything alone: refuse "
          "everything and the right column is 1.0")
    if out["by_type"]:
        print("\nby question type")
        width = max(len(k) for k in out["by_type"])
        for name, v in out["by_type"].items():
            print(f"  {name:<{width}}  n={v['n']:<4} acc={v['accuracy']}  "
                  f"correct={v['correct']}  wrong={v['incorrect']}  "
                  f"false_ref={v['false_refusal']}  "
                  f"ret_miss={v['retrieval_miss']}  unjudged={v['unjudged']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=str(ANSWERS))
    ap.add_argument("--out", default=str(JUDGED))
    ap.add_argument("--limit", type=int, help="judge at most N new rows")
    ap.add_argument("--model", help="exact judge model id; do not guess one")
    ap.add_argument("--free", action="store_true",
                    help="no model: grades only what status already decides "
                         "(abstention, refusals). Everything else is unjudged.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--wrong", type=int, default=0,
                    help="print N graded-wrong answers with the judge's reason")
    a = ap.parse_args()

    src = Path(a.answers)
    if not src.exists():
        print(f"no answers file at {src}; run tools/answer_run.py first")
        return 4
    rows = load_rows(src)
    if not rows:
        print(f"{src} holds no rows")
        return 4

    out_path = Path(a.out)
    already = done_ids(out_path)
    llm = None if a.free else make_llm(a.model)
    judge = J.Judge(llm)
    print(f"judging {len(rows)} rows from {src}")
    print(f"judge model      {judge.model or 'none (--free)'}")

    todo = [r for r in rows if r.get("question_id") not in already]
    if a.limit:
        # Rows needing a model call come first, so a pilot spends its budget on
        # the questions that actually cost something rather than on refusals.
        todo.sort(key=lambda r: (r.get("is_abstention") or r.get("status") != "answered"))
        todo = todo[:a.limit]
    started = time.time()
    for i, row in enumerate(todo, 1):
        verdict = judge.judge(row)
        append(out_path, {
            "question_id": row.get("question_id"),
            "question_type": row.get("question_type"),
            "is_abstention": row.get("is_abstention"),
            "question": row.get("question"),
            "gold": row.get("gold"),
            "answer": row.get("answer"),
            "status": row.get("status"),
            **verdict,
            "at": time.time(),
        })
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}  {judge.calls} judge calls  "
                  f"{i / max(time.time() - started, 1e-9):.2f}/s", flush=True)

    if llm is not None:
        print(f"\nspend            {llm.calls} calls, {llm.retries} retries, "
              f"{llm.input_tokens} in / {llm.output_tokens} out tokens")
        print("                 price the full run off THESE numbers, not an estimate")

    judged = load_rows(out_path)
    result = J.tally(judged)
    if a.json:
        print(json.dumps(result, indent=2))
    else:
        report(result)

    if a.wrong:
        print("\nwrong answers")
        shown = 0
        for r in judged:
            if r.get("verdict") != J.INCORRECT or r.get("is_abstention"):
                continue
            shown += 1
            print(f"\n  [{r.get('question_type')}] {r.get('question')}")
            print(f"    gold     : {r.get('gold')}")
            print(f"    produced : {r.get('answer')}")
            print(f"    judge    : {r.get('kind')} - {r.get('why')}")
            if shown >= a.wrong:
                break
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The no-memory baseline: the same questions, the same model, no evidence.

    python3 tools/baseline_run.py --data data/longmemeval_oracle.json --limit 25
    python3 tools/judge_run.py --answers ~/docket/state/baseline.jsonl \
                               --out ~/docket/state/judged-baseline.jsonl

This is the control, and without it the headline accuracy is unreadable. Some
LongMemEval questions are answerable from world knowledge or from the question's
own wording, and a system that scores 0.55 against a 0.05 baseline has done
something very different from one that scores 0.55 against a 0.45 baseline.
Reporting the first number without the second is the most common way a memory
result overstates itself.

It touches neither HydraDB nor the gate on purpose: no graph client is built,
so this also runs on a machine where the database is not up.

The prompt is deliberately the same contract as the real answerer -- answer, or
say NOT_IN_MEMORY -- so the difference measured is the presence of evidence and
nothing else. Rows are written in the answers.jsonl shape with empty evidence,
so judge_run grades both files with identical code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docket import dataset  # noqa: E402
from docket.answer import ABSENT, ANSWERED, NOT_IN_MEMORY, UNMEASURED  # noqa: E402
from docket.llm import Anthropic, LLMError, load_env  # noqa: E402

STATE = Path(os.environ.get("DOCKET_STATE", Path.home() / "docket" / "state"))
BASELINE = STATE / "baseline.jsonl"
ENV = Path(os.environ.get("DOCKET_ENV", Path.home() / "docket" / ".env"))

SYSTEM = f"""You answer a question about a user's past conversations. You have
no access to those conversations.

If you cannot know the answer without them, reply with exactly {NOT_IN_MEMORY}.
Do not guess, and do not reason about what is likely.

Reply with JSON and nothing else:
{{"answer": "<the answer, or {NOT_IN_MEMORY}>"}}"""


def done_ids(path: Path, retry_unmeasured: bool = False) -> set[str]:
    """Question ids already recorded, so a stopped run resumes for free.

    `retry_unmeasured` exists because resumption had a hole: a row written as
    `unmeasured` (a dead socket after the lid closed, a model that returned
    nothing) counts as done and is skipped forever on re-run, leaving a
    permanent gap in a benchmark that reports its own denominator. With the
    flag those ids are NOT treated as done, so a second pass retries exactly
    the failures and nothing else.
    """
    if not path.exists():
        return set()
    out = set()
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
                qid = row["question_id"]
            except (ValueError, KeyError):
                continue
            if retry_unmeasured and row.get("status") == UNMEASURED:
                out.discard(qid)
                continue
            out.add(qid)
    return out


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def answer_without_memory(llm, question: str, asked_at: int) -> tuple[str, str, str]:
    """Returns (status, answer, reason). A model failure is UNMEASURED."""
    user = f"Question: {question}\nAsked at: {asked_at}"
    try:
        reply = llm.complete_json(SYSTEM, user)
    except (LLMError, ValueError) as exc:
        return UNMEASURED, None, f"model call failed: {exc}"
    if not isinstance(reply, dict) or "answer" not in reply:
        return UNMEASURED, None, f"unusable shape: {json.dumps(reply)[:160]}"
    text = str(reply.get("answer") or "").strip()
    if not text or text.upper().replace(" ", "_") == NOT_IN_MEMORY:
        return ABSENT, None, "said it cannot answer without the history"
    return ANSWERED, text, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model")
    ap.add_argument("--retry-unmeasured", action="store_true",
                    help="re-run only the rows that failed to measure")
    ap.add_argument("--out", default=str(BASELINE))
    a = ap.parse_args()

    env = load_env(str(ENV))
    key = env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit(f"no ANTHROPIC_API_KEY in {ENV}")
    llm = Anthropic(key, model=a.model) if a.model else Anthropic(key)

    out_path = Path(a.out)
    already = done_ids(out_path, a.retry_unmeasured)
    instances = dataset.load(a.data, limit=a.limit)
    print(f"no-memory baseline over {len(instances)} questions, model {llm.model}")

    tally = {ANSWERED: 0, ABSENT: 0, UNMEASURED: 0, "skipped": 0}
    started = time.time()
    for inst in instances:
        if inst.question_id in already:
            tally["skipped"] += 1
            continue
        asked = int(inst.asked_at.timestamp())
        status, text, reason = answer_without_memory(llm, inst.question, asked)
        tally[status] = tally.get(status, 0) + 1
        append(out_path, {
            "question_id": inst.question_id,
            "question_type": inst.question_type,
            "is_abstention": inst.is_abstention,
            "question": inst.question,
            "asked_at": asked,
            "gold": inst.answer,
            "gold_sessions": inst.evidence_session_ids,
            "status": status,
            "answer": text,
            "reason": reason,
            "evidence": [],
            "retrieval": {"mode": "none", "note": "no-memory baseline"},
            "at": time.time(),
        })
        done = sum(v for k, v in tally.items() if k != "skipped")
        if done % 10 == 0:
            print(f"  {done}  {done / max(time.time() - started, 1e-9):.2f}/s  "
                  f"{tally}", flush=True)

    print(f"\n{tally}  in {time.time() - started:.1f}s")
    print(f"spend            {llm.calls} calls, {llm.retries} retries, "
          f"{llm.input_tokens} in / {llm.output_tokens} out tokens")
    print(f"wrote {out_path}")
    print("judge this file with the SAME judge, then report both numbers "
          "together or neither.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

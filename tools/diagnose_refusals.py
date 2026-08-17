#!/usr/bin/env python3
"""Why did the answerer refuse? Offline, free, no model, no database.

A false refusal has two completely different causes and they need opposite
fixes, so they must be separated before anything is changed:

  GOLD PRESENT  the evidence contained a session the benchmark says holds the
                answer, and the model declined anyway. That is the answerer's
                problem -- prompt, or evidence the model cannot read.

  GOLD ABSENT   retrieval never put the answer in front of it. Refusing was
                the correct behaviour on what it was shown.

Timestamps are printed as both epoch and human date, because the evidence block
currently renders them as raw epochs and that is the leading suspect for why a
TEMPORAL question gets refused.
"""
import argparse
import datetime
import json
import os
import sys

STATE = os.environ.get("DOCKET_STATE",
                       os.path.expanduser("~/docket/state"))
ANSWERS = os.path.join(STATE, "answers-model.jsonl")
JUDGED = os.path.join(STATE, "judged-model.jsonl")


def load(path):
    if not os.path.exists(path):
        sys.exit("missing file: " + path)
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def when(ts):
    if ts is None:
        return "None"
    try:
        stamp = datetime.datetime.fromtimestamp(int(ts),
                                                datetime.timezone.utc)
        return "%s  (%s)" % (ts, stamp.strftime("%Y-%m-%d %H:%M"))
    except (ValueError, OSError, OverflowError):
        return str(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=ANSWERS)
    ap.add_argument("--judged", default=JUDGED)
    a = ap.parse_args()
    answers = {r["question_id"]: r for r in load(a.answers)}
    judged = load(a.judged)

    refusals = [j for j in judged
                if j.get("kind") in ("false_refusal", "retrieval_miss")]
    wrong = [j for j in judged if j.get("kind") == "incorrect"]
    correct = [j for j in judged if j.get("verdict") == "correct"]

    print("=" * 68)
    print("%d correct   %d wrong answers   %d false refusals"
          % (len(correct), len(wrong), len(refusals)))
    print("=" * 68)

    gold_present = 0
    print("\n--- FALSE REFUSALS ---")
    for j in refusals:
        row = answers.get(j["question_id"])
        if not row:
            print("  (no answers row for %s)" % j["question_id"])
            continue
        gold = set(row.get("gold_sessions") or [])
        cited = set()
        for e in (row.get("evidence") or []):
            if e.get("session"):
                cited.add(e["session"])
        hit = bool(gold & cited)
        gold_present += 1 if hit else 0
        print("  gold_in_evidence=%-5s ev=%-2d  %s"
              % (hit, len(row.get("evidence") or []), row["question"][:62]))

    if refusals:
        print("\n>>> %d/%d refusals HAD gold evidence in front of them"
              % (gold_present, len(refusals)))
        print(">>> %d/%d were refused with nothing useful retrieved"
              % (len(refusals) - gold_present, len(refusals)))

    # Show one refusal in full, preferring one where the gold WAS present --
    # that is the interesting case.
    show = None
    for j in refusals:
        row = answers.get(j["question_id"])
        if not row:
            continue
        gold = set(row.get("gold_sessions") or [])
        cited = {e.get("session") for e in (row.get("evidence") or [])}
        if gold & cited:
            show = row
            break
    if show is None and refusals:
        show = answers.get(refusals[0]["question_id"])

    if show:
        print("\n--- WHAT THE MODEL ACTUALLY SAW (one refusal) ---")
        print("Q        : %s" % show["question"])
        print("asked_at : %s" % when(show.get("asked_at")))
        print("gold     : %s" % show.get("gold"))
        print("gold sess: %s" % (show.get("gold_sessions") or []))
        for e in (show.get("evidence") or []):
            print("  [%s] %s" % (e.get("n"), e.get("triple")))
            print("       session=%s  said=%s" % (e.get("session"),
                                                  when(e.get("said_at"))))
            sup = e.get("superseded") or []
            if sup:
                print("       superseded: %s" % sup)

    print("\n--- WRONG ANSWERS (first 5) ---")
    for j in wrong[:5]:
        print("  gold : %s" % str(j.get("gold"))[:70])
        print("  got  : %s" % str(j.get("answer"))[:70])
        print("  judge: %s" % str(j.get("why"))[:90])
        print()


if __name__ == "__main__":
    main()

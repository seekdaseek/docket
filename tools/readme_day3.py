#!/usr/bin/env python3
"""Patch README.md: correct one false claim, append the Day 3 section.

    python3 tools/readme_day3.py            # apply
    python3 tools/readme_day3.py --check    # report without writing

Idempotent: running it twice changes nothing the second time.

THE CORRECTION. The opening section said embeddings nominate the candidate
subjects. They do not. `EmbeddingGate.__init__` raises NotImplementedError and
the gate that runs is BM25 -- which the README itself says correctly ninety
lines further down, so the file contradicted itself in the section a reader
opens first. Day 1 text, corrected on Day 2 lower down, never fixed at the top.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")

WRONG = """HydraDB has no vector index and none is claimed. Embeddings are computed
outside the database and used only to nominate candidate subjects. Every
statement that decides an answer is a traversal."""

RIGHT = """HydraDB has no vector index and none is claimed, and no embeddings are used:
candidates are nominated by a BM25 gate built in the standard library, and
every statement that decides an answer is a traversal executed by the node.
The gate is lexical on purpose -- see "the gate is lexical" below -- and
`EmbeddingGate` raises rather than silently degrading, so no number in this
repository can be reported for a component that did not run."""

MARKER = "## Day 3 -- the eval, and what it found"

DAY3 = """

## Day 3 -- the eval, and what it found

Full numbers, generated from the run files rather than typed, are in
[RESULTS.md](RESULTS.md). Regenerate them with `python3 tools/report.py`.

**Accuracy 0.3511 (165/470) against a no-memory baseline of 0.000 (0/470).**
The baseline is the same 500 questions and the same model with no evidence at
all. It answered none of them, so nothing in the figure above is world
knowledge leaking in. Stated precisely, because it is a weaker claim than it
looks: the baseline was instructed not to guess, so it shows that a model which
declines rather than guesses gets none of these questions. It does not show
they are unguessable.

Judge is `claude-sonnet-5`, deliberately a stronger model than the
`claude-haiku-4-5` answerer so the system is not grading itself. The judge is
shown the question, the reference answer and the produced answer, and never the
retrieved evidence -- a judge that can see the claims ends up grading the
retrieval a second time. Its prompt is in `docket/judge.py`. 470 questions were
graded and none came back unjudged.

### The average hides three different systems

| category | n | retrieval recall | accuracy | conversion |
| --- | --- | --- | --- | --- |
| knowledge-update | 72 | 0.9306 | 0.7222 | 0.776 |
| single-session-user | 64 | 0.7812 | 0.5781 | 0.740 |
| temporal-reasoning | 127 | 0.8189 | 0.3386 | 0.413 |
| multi-session | 121 | 0.8099 | 0.2231 | 0.275 |
| single-session-assistant | 56 | 0.4821 | 0.0893 | 0.185 |
| single-session-preference | 30 | 0.3000 | 0.0333 | 0.111 |

**conversion** is accuracy divided by retrieval recall: of the questions whose
evidence was found, the share that also produced a right answer.

Read the last column rather than the aggregate. Where retrieval is strong the
system works -- knowledge-update finds the evidence 93 percent of the time and
converts three quarters of that into correct answers. Where retrieval is weak
the system collapses, and the collapse is honest: preference finds the evidence
on 30 percent of questions, and 19 of its 30 refusals happened with nothing
relevant retrieved. Refusing there was the correct behaviour.

The interesting rows are the middle two. single-session-user and
temporal-reasoning have almost the same retrieval recall, 0.7812 and 0.8189,
and convert at 0.740 and 0.413. multi-session has recall 0.8099 and converts at
0.275. Whatever is costing those answers, it is not the gate.

### The failure mode is over-refusal, not hallucination

Of 470 graded questions: 165 correct, 69 answered and wrong, 133 refused with a
gold session sitting in the evidence, and 103 refused with the gold session
never retrieved.

Those last two are counted separately everywhere in this repository because
they need opposite fixes, and merging them pointed the diagnosis at the wrong
layer once already during this build. A refusal with the evidence present is
the answerer's failure. A refusal with nothing retrieved is the gate's, and the
refusal itself was right.

    false refusal rate         0.3624   (133 of 367)
    abstention refusal rate    0.9333   (28 of 30)

Neither number means anything alone. A system that refuses everything scores
1.000 on the right-hand number and 0.000 accuracy -- `tests/test_day3.py` has a
test named for exactly that case, and the no-memory baseline is a live example
of it. Reported together they say what is actually wrong: the refusal machinery
works, and it is tuned too hot.

Two of the 30 unanswerable questions were answered anyway. That is a 2/30
hallucination rate, not zero, and it is written here rather than rounded away.

**When docket does answer, it is right 165 times out of 234 -- 70.5 percent.**
Converting the 133 calibration failures at that same precision would land near
0.55. That is an estimate and not a result, but it sizes the gap: over-refusal
is worth roughly twenty accuracy points.

### Limitations, stated

**A claim records when a fact was mentioned, not when the event happened.** Two
claims extracted from one conversation carry the same timestamp even when they
describe events months apart, and the event's own date survives only if the
extractor captured it inside the claim. Ordering and duration questions are
answerable only in that case. This is visible in the table: temporal-reasoning
and multi-session carry 92 of the 133 calibration failures. Fixing it properly
means extracting event dates as a first-class field and re-running extraction,
which did not fit the six days.

**The judge is a model.** Its prompt is in the repository and its id is written
onto every judged row, so the grading can be re-run or disputed, but it is not
a human rubric.

**The as-of tolerance was measured on the oracle file, which contains only
evidence sessions.** The audit's "non-evidence admitted" column reads zero at
every tolerance and proves nothing, because there are no distractors present to
admit. On `longmemeval_s` a 24-hour window will let some in, and the same audit
has to be re-run there.

**Answering was measured on the oracle split, not the full haystack.**

### Changes that were measured worse and reverted

The answerer prompt was rewritten to explain the mention-date problem and to
soften the refusal rule. On an identical 25-question slice it took answers from
15 to 7, accuracy from 0.3333 to 0.2381, and the false refusal rate from 0.375
to 0.6667. It is reverted, and `SYSTEM` is the original text. The rewrite
bundled three changes at once, so it cannot say which of them did the damage --
that is a fault in the experiment, not a finding.

Rendering evidence timestamps as dates rather than raw Unix epochs was then
measured on its own against the same slice: answer count identical, one answer
flipped from wrong to correct. **Measured neutral.** It is kept because it does
no harm and because `said 2023-04-10 (Mon)` is legible where `said 1681138020`
is not. `--epoch-dates` restores the earlier rendering exactly.

### A parser bug worth naming

The first paid run recorded 14 of 25 answers as measurement failures. The model
had been emitting correct JSON and then continuing in prose, and `json.loads`
over the whole reply rejected it. Seven of those 14 were correct abstentions --
the one behaviour this project exists to demonstrate, discarded as noise.

`extract_json` now reads the first complete JSON value with `raw_decode` and
reports whatever followed; `trailing_prose` counts how often the model kept
talking. On the full run that was 168 of 493 replies. The project's rule is
unchanged: output with no JSON in it still raises and is still recorded as
unmeasured. What changed is that valid JSON followed by commentary stopped
being counted as a failure to answer. Those were two different facts being
merged into one.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(README):
        raise SystemExit("no README.md at " + README)
    text = open(README).read()

    fix_needed = WRONG in text
    fix_done = RIGHT in text
    day3_needed = MARKER not in text

    if not fix_needed and not fix_done:
        print("WARNING: the embeddings paragraph was not found in either form.")
        print("The README has changed; correct it by hand and re-check.")
        if not a.check:
            return 4

    print("embeddings correction : %s" % ("needed" if fix_needed
                                          else ("already applied" if fix_done
                                                else "NOT FOUND")))
    print("day 3 section         : %s" % ("needed" if day3_needed
                                          else "already applied"))
    if a.check:
        return 0

    if fix_needed:
        text = text.replace(WRONG, RIGHT, 1)
    if day3_needed:
        text = text.rstrip("\n") + DAY3
    with open(README, "w") as fh:
        fh.write(text)

    after = open(README).read()
    ok = (RIGHT in after) and (MARKER in after) and (WRONG not in after)
    print("verified              : %s" % ("yes" if ok else "NO"))
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())

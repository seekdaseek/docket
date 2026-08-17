"""Judge an answer against the benchmark's gold answer.

LongMemEval's own evaluation asks a model whether the produced answer matches
the reference. That is what happens here, with three rules that are not
optional:

  THE JUDGE NEVER SEES THE EVIDENCE. It is given the question, the reference
  answer and the produced answer, and nothing else. A judge shown the retrieved
  claims can be argued into "the evidence supports this", which grades the
  retrieval a second time instead of grading the answer.

  A JUDGE THAT FAILS PRODUCES `unjudged`, NEVER `incorrect`. A model that times
  out, returns prose instead of JSON, or answers in a shape we cannot read is a
  measurement failure. Folding those into the wrong column would quietly lower
  the score for reasons that have nothing to do with the system under test --
  the same rule llm.complete_json already enforces one layer down.

  ABSTENTION IS NOT JUDGED BY A MODEL AT ALL. For an `_abs` question the only
  correct behaviour is a refusal, and `status` already records whether the
  system refused. Asking a model to grade that would spend money to re-derive a
  boolean we hold, and would introduce judge noise into the one number the
  Hack Hydra brief singles out.

The judge model is recorded on every row. A score produced by an undisclosed
judge is not a result.
"""
from __future__ import annotations

from docket.answer import ABSENT, ANSWERED, UNMEASURED
from docket.llm import LLMError

CORRECT = "correct"
INCORRECT = "incorrect"
UNJUDGED = "unjudged"

# Why "hallucinated" rather than "incorrect" for abstention: on a question the
# history cannot answer, producing any answer at all is the failure. Naming it
# separately keeps it visible instead of averaging it away.
HALLUCINATED = "hallucinated"
REFUSED = "refused"
FALSE_REFUSAL = "false_refusal"
# A refusal where retrieval never supplied the gold session. Still a wrong
# answer, but NOT a calibration failure: refusing when nothing relevant was
# retrieved is the behaviour this project argues for. Merging the two hides
# which layer to fix -- measured Aug 17, single-session-assistant showed 6/6
# "false refusals" in a category whose retrieval recall is 0.48.
RETRIEVAL_MISS = "retrieval_miss"

JUDGE_SYSTEM = """You grade one answer against a reference answer.

Say correct when the produced answer conveys the same fact as the reference.
Wording, extra detail, rounding of a spelled-out number, and different phrasing
do not matter. A produced answer that contains the reference fact plus harmless
context is correct.

Say incorrect when the produced answer states a different fact, contradicts the
reference, omits the specific thing that was asked for, or hedges so heavily
that no fact is asserted.

Judge only the produced answer against the reference. You are not given the
source material and must not speculate about what it contained.

Reply with JSON and nothing else:
{"correct": true or false, "why": "<one short sentence>"}"""


def build_prompt(question, gold, produced) -> str:
    return (f"Question: {question}\n"
            f"Reference answer: {gold}\n"
            f"Produced answer: {produced}")


def read_verdict(reply) -> tuple[str, str]:
    """Turn the model's reply into (verdict, why). Unusable shape -> unjudged."""
    if not isinstance(reply, dict) or "correct" not in reply:
        return UNJUDGED, f"judge returned an unusable shape: {str(reply)[:160]}"
    value = reply.get("correct")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "correct"):
            value = True
        elif lowered in ("false", "no", "incorrect"):
            value = False
        else:
            return UNJUDGED, f"judge said correct={value!r}, which is neither"
    if not isinstance(value, bool):
        return UNJUDGED, f"judge said correct={value!r}, which is not a boolean"
    why = str(reply.get("why") or "").strip()
    return (CORRECT if value else INCORRECT), why


class Judge:
    """Grades one answers.jsonl row. `llm` may be None for the free paths."""

    def __init__(self, llm=None, model: str | None = None):
        self.llm = llm
        self.model = model or getattr(llm, "model", None)
        self.calls = 0

    def judge(self, row: dict) -> dict:
        status = row.get("status")

        # -- abstention: decided by status, no spend -----------------------
        if row.get("is_abstention"):
            if status == ABSENT:
                return self._out(CORRECT, REFUSED,
                                 "refused, which is the correct behaviour here",
                                 model=None)
            if status == ANSWERED:
                return self._out(INCORRECT, HALLUCINATED,
                                 "answered a question the history cannot answer",
                                 model=None)
            return self._out(UNJUDGED, UNMEASURED,
                             f"status {status}: nothing was decided", model=None)

        # -- answerable ----------------------------------------------------
        if status == ABSENT:
            # Both are wrong answers and both belong in the accuracy
            # denominator, but they point at opposite layers, so they are
            # never merged: FALSE_REFUSAL means the gold session WAS in the
            # evidence and the answerer declined anyway (calibration), while
            # RETRIEVAL_MISS means it was never retrieved (gate/ranking).
            gold = set(row.get("gold_sessions") or [])
            cited = {e.get("session") for e in (row.get("evidence") or [])}
            if gold and not (gold & cited):
                return self._out(INCORRECT, RETRIEVAL_MISS,
                                 "refused, and the gold session was never "
                                 "retrieved -- refusing was correct on what "
                                 "it was shown",
                                 model=None)
            return self._out(INCORRECT, FALSE_REFUSAL,
                             "refused with a gold session in the evidence",
                             model=None)
        if status != ANSWERED:
            return self._out(UNJUDGED, UNMEASURED,
                             row.get("reason") or f"status {status}", model=None)

        # str() before strip(): LongMemEval gold answers are not all strings.
        # "how many charities" has gold 4, an int, and the judge died on it
        # 60 rows into the 500-question run. Coerce every field that reaches
        # the prompt rather than trusting the dataset's types.
        produced = str(row.get("answer") or "").strip()
        gold = str(row.get("gold") or "").strip()
        if not produced:
            return self._out(UNJUDGED, UNMEASURED,
                             "status was answered but the answer is empty",
                             model=None)
        if not gold:
            return self._out(UNJUDGED, UNMEASURED,
                             "the row carries no reference answer to grade against",
                             model=None)
        if self.llm is None:
            return self._out(UNJUDGED, UNMEASURED,
                             "no judge model configured", model=None)

        prompt = build_prompt(str(row.get("question") or ""), gold, produced)
        try:
            reply = self.llm.complete_json(JUDGE_SYSTEM, prompt)
        except (LLMError, ValueError) as exc:
            return self._out(UNJUDGED, UNMEASURED, f"judge call failed: {exc}",
                             model=self.model)
        self.calls += 1
        verdict, why = read_verdict(reply)
        kind = {CORRECT: CORRECT, INCORRECT: INCORRECT}.get(verdict, UNMEASURED)
        return self._out(verdict, kind, why, model=self.model)

    def _out(self, verdict: str, kind: str, why: str, model) -> dict:
        return {"verdict": verdict, "kind": kind, "why": why,
                "judge_model": model}


def tally(rows: list[dict]) -> dict:
    """Aggregate judged rows. Answerable and abstention never mix."""
    real = [r for r in rows if not r.get("is_abstention")]
    absts = [r for r in rows if r.get("is_abstention")]

    by_type: dict[str, dict] = {}
    correct = incorrect = unjudged = false_refusals = 0
    retrieval_misses = 0
    for r in real:
        v, kind = r.get("verdict"), r.get("kind")
        bucket = by_type.setdefault(r.get("question_type") or "unknown",
                                    {"n": 0, "correct": 0, "incorrect": 0,
                                     "unjudged": 0, "false_refusal": 0,
                                     "retrieval_miss": 0})
        bucket["n"] += 1
        if v == CORRECT:
            correct += 1
            bucket["correct"] += 1
        elif v == INCORRECT:
            incorrect += 1
            bucket["incorrect"] += 1
            if kind == FALSE_REFUSAL:
                false_refusals += 1
                bucket["false_refusal"] += 1
            elif kind == RETRIEVAL_MISS:
                retrieval_misses += 1
                bucket["retrieval_miss"] += 1
        else:
            unjudged += 1
            bucket["unjudged"] += 1

    for bucket in by_type.values():
        graded = bucket["correct"] + bucket["incorrect"]
        bucket["accuracy"] = round(bucket["correct"] / graded, 4) if graded else None

    graded = correct + incorrect
    refused = sum(1 for r in absts if r.get("kind") == REFUSED)
    hallucinated = sum(1 for r in absts if r.get("kind") == HALLUCINATED)
    abst_unjudged = len(absts) - refused - hallucinated
    abst_graded = refused + hallucinated

    return {
        "answerable": len(real),
        "graded": graded,
        "correct": correct,
        "incorrect": incorrect,
        "unjudged": unjudged,
        # Accuracy is over what was actually graded. `unjudged` is printed
        # beside it every time so the denominator is never silently smaller
        # than it looks.
        "accuracy": round(correct / graded, 4) if graded else None,
        "by_type": dict(sorted(by_type.items())),
        "abstention": {
            "n": len(absts),
            "refused": refused,
            "hallucinated": hallucinated,
            "unjudged": abst_unjudged,
            "refusal_rate": round(refused / abst_graded, 4) if abst_graded else None,
        },
        # The calibration pair. One of these alone says nothing: a system that
        # refuses everything scores 1.0 on abstention and is useless, and a
        # system that never refuses scores 0.0 on false refusals and lies on
        # every unanswerable question. They are only meaningful together.
        "retrieval_misses": retrieval_misses,
        "calibration": {
            "false_refusals": false_refusals,
            # Denominator excludes retrieval misses: this rate is about the
            # answerer's judgement, and a question whose evidence never arrived
            # gave it no judgement to make.
            "false_refusal_rate": (
                round(false_refusals / (graded - retrieval_misses), 4)
                if (graded - retrieval_misses) else None),
            "retrieval_misses": retrieval_misses,
            "abstention_refusal_rate": (
                round(refused / abst_graded, 4) if abst_graded else None),
        },
    }

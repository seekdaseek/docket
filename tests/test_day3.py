"""Day 3: the judge, the abstention rule, the calibration pair.

Every test here is offline. The judge model is a scripted stand-in, so the
grading logic is pinned without spending anything, and the failure modes that
would quietly move the headline number each have a regression.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docket import judge as J
from docket.answer import ABSENT, ANSWERED, UNMEASURED
from docket.llm import LLMError


class ScriptedJudge:
    """Returns replies in order. Raise an LLMError by scripting an exception."""

    model = "scripted-judge-1"

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def complete_json(self, system, user):
        self.seen.append((system, user))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def row(**kw):
    base = {"question_id": "q1", "question_type": "temporal-reasoning",
            "is_abstention": False, "question": "Where do I live?",
            "gold": "Lisbon", "answer": "Lisbon", "status": ANSWERED}
    base.update(kw)
    return base


class JudgeVerdicts(unittest.TestCase):
    def test_correct_answer_is_correct(self):
        j = J.Judge(ScriptedJudge([{"correct": True, "why": "same city"}]))
        out = j.judge(row())
        self.assertEqual(J.CORRECT, out["verdict"])
        self.assertEqual("scripted-judge-1", out["judge_model"])

    def test_wrong_answer_is_incorrect(self):
        j = J.Judge(ScriptedJudge([{"correct": False, "why": "different city"}]))
        self.assertEqual(J.INCORRECT, j.judge(row(answer="Porto"))["verdict"])

    def test_judge_never_sees_the_evidence(self):
        # The whole point of the separation. If the retrieved claims reach the
        # judge it can be argued into grading the retrieval instead.
        llm = ScriptedJudge([{"correct": True, "why": "ok"}])
        J.Judge(llm).judge(row(evidence=[{"triple": ["user", "lives_in", "Lisbon"]}]))
        _, user = llm.seen[0]
        self.assertNotIn("lives_in", user)
        self.assertIn("Reference answer: Lisbon", user)
        self.assertIn("Produced answer: Lisbon", user)

    def test_string_booleans_are_accepted(self):
        j = J.Judge(ScriptedJudge([{"correct": "true", "why": "y"}]))
        self.assertEqual(J.CORRECT, j.judge(row())["verdict"])

    def test_unusable_shape_is_unjudged_not_incorrect(self):
        j = J.Judge(ScriptedJudge([{"verdict": "yes"}]))
        out = j.judge(row())
        self.assertEqual(J.UNJUDGED, out["verdict"])

    def test_judge_failure_is_unjudged_not_incorrect(self):
        # A timeout must never be recorded as a wrong answer: that would lower
        # the score for a reason unrelated to the system under test.
        j = J.Judge(ScriptedJudge([LLMError("http 529: overloaded")]))
        out = j.judge(row())
        self.assertEqual(J.UNJUDGED, out["verdict"])
        self.assertIn("judge call failed", out["why"])

    def test_non_boolean_correct_is_unjudged(self):
        j = J.Judge(ScriptedJudge([{"correct": "maybe", "why": "unsure"}]))
        self.assertEqual(J.UNJUDGED, j.judge(row())["verdict"])


class AbstentionIsFree(unittest.TestCase):
    def test_refusal_on_abstention_is_correct_without_a_call(self):
        llm = ScriptedJudge([])  # popping from this would IndexError
        out = J.Judge(llm).judge(row(is_abstention=True, status=ABSENT,
                                     answer=None))
        self.assertEqual(J.CORRECT, out["verdict"])
        self.assertEqual(J.REFUSED, out["kind"])
        self.assertEqual([], llm.seen)
        self.assertIsNone(out["judge_model"])

    def test_answering_an_abstention_question_is_hallucination(self):
        llm = ScriptedJudge([])
        out = J.Judge(llm).judge(row(is_abstention=True, status=ANSWERED,
                                     answer="Lisbon"))
        self.assertEqual(J.INCORRECT, out["verdict"])
        self.assertEqual(J.HALLUCINATED, out["kind"])
        self.assertEqual([], llm.seen)

    def test_unmeasured_abstention_is_unjudged(self):
        out = J.Judge(None).judge(row(is_abstention=True, status=UNMEASURED))
        self.assertEqual(J.UNJUDGED, out["verdict"])


class AnswerableRefusals(unittest.TestCase):
    def test_refusing_an_answerable_question_is_a_labelled_miss(self):
        llm = ScriptedJudge([])
        out = J.Judge(llm).judge(row(status=ABSENT, answer=None))
        self.assertEqual(J.INCORRECT, out["verdict"])
        self.assertEqual(J.FALSE_REFUSAL, out["kind"])
        self.assertEqual([], llm.seen)

    def test_unmeasured_is_not_graded(self):
        out = J.Judge(None).judge(row(status=UNMEASURED,
                                      reason="model call failed"))
        self.assertEqual(J.UNJUDGED, out["verdict"])

    def test_empty_answer_with_answered_status_is_unjudged(self):
        out = J.Judge(ScriptedJudge([])).judge(row(answer="   "))
        self.assertEqual(J.UNJUDGED, out["verdict"])

    def test_missing_gold_is_unjudged_not_wrong(self):
        out = J.Judge(ScriptedJudge([])).judge(row(gold=""))
        self.assertEqual(J.UNJUDGED, out["verdict"])


class Tally(unittest.TestCase):
    def judged(self, **kw):
        base = {"question_type": "multi-session", "is_abstention": False,
                "verdict": J.CORRECT, "kind": J.CORRECT}
        base.update(kw)
        return base

    def test_accuracy_excludes_unjudged_from_the_denominator(self):
        out = J.tally([
            self.judged(),
            self.judged(verdict=J.INCORRECT, kind=J.INCORRECT),
            self.judged(verdict=J.UNJUDGED, kind=UNMEASURED),
        ])
        self.assertEqual(2, out["graded"])
        self.assertEqual(0.5, out["accuracy"])
        self.assertEqual(1, out["unjudged"])

    def test_abstention_never_enters_accuracy(self):
        out = J.tally([
            self.judged(),
            self.judged(is_abstention=True, verdict=J.CORRECT, kind=J.REFUSED),
            self.judged(is_abstention=True, verdict=J.INCORRECT,
                        kind=J.HALLUCINATED),
        ])
        self.assertEqual(1, out["answerable"])
        self.assertEqual(1.0, out["accuracy"])
        self.assertEqual(2, out["abstention"]["n"])
        self.assertEqual(0.5, out["abstention"]["refusal_rate"])

    def test_false_refusals_are_counted_and_still_wrong(self):
        out = J.tally([
            self.judged(),
            self.judged(verdict=J.INCORRECT, kind=J.FALSE_REFUSAL),
        ])
        self.assertEqual(0.5, out["accuracy"])
        self.assertEqual(1, out["calibration"]["false_refusals"])
        self.assertEqual(0.5, out["calibration"]["false_refusal_rate"])

    def test_the_refuse_everything_system_is_visibly_useless(self):
        # A system that refuses every question scores a perfect abstention rate
        # and 0.0 accuracy. If the pair is reported together that is obvious;
        # if only the abstention number is quoted it looks like a result.
        rows = [self.judged(verdict=J.INCORRECT, kind=J.FALSE_REFUSAL)
                for _ in range(10)]
        rows += [self.judged(is_abstention=True, verdict=J.CORRECT,
                             kind=J.REFUSED) for _ in range(10)]
        out = J.tally(rows)
        self.assertEqual(0.0, out["accuracy"])
        self.assertEqual(1.0, out["calibration"]["abstention_refusal_rate"])
        self.assertEqual(1.0, out["calibration"]["false_refusal_rate"])

    def test_per_type_accuracy(self):
        out = J.tally([
            self.judged(question_type="knowledge-update"),
            self.judged(question_type="knowledge-update",
                        verdict=J.INCORRECT, kind=J.INCORRECT),
            self.judged(question_type="single-session-preference"),
        ])
        self.assertEqual(0.5, out["by_type"]["knowledge-update"]["accuracy"])
        self.assertEqual(1.0, out["by_type"]["single-session-preference"]["accuracy"])

    def test_nothing_graded_reports_none_not_zero(self):
        # A run where every judge call failed must not read as 0% accuracy.
        out = J.tally([self.judged(verdict=J.UNJUDGED, kind=UNMEASURED)])
        self.assertIsNone(out["accuracy"])


if __name__ == "__main__":
    unittest.main()


class RefusalSplit(unittest.TestCase):
    """Refusing with the answer in front of you and refusing with nothing in
    front of you are opposite problems. Measured Aug 17: a stratified slice
    reported 27 'false refusals', but single-session-assistant contributed 6/6
    in a category whose retrieval recall is 0.4821 -- those were retrieval
    misses being scored as calibration failures."""

    def refusal(self, gold_sessions, cited_sessions):
        return row(status=ABSENT, answer=None,
                   gold_sessions=gold_sessions,
                   evidence=[{"session": s} for s in cited_sessions])

    def test_gold_in_evidence_is_a_calibration_failure(self):
        out = J.Judge(ScriptedJudge([])).judge(self.refusal(["s1"], ["s1", "s9"]))
        self.assertEqual(J.INCORRECT, out["verdict"])
        self.assertEqual(J.FALSE_REFUSAL, out["kind"])

    def test_gold_never_retrieved_is_a_retrieval_miss(self):
        out = J.Judge(ScriptedJudge([])).judge(self.refusal(["s1"], ["s7", "s9"]))
        self.assertEqual(J.INCORRECT, out["verdict"])
        self.assertEqual(J.RETRIEVAL_MISS, out["kind"])

    def test_both_are_still_wrong_answers(self):
        out = J.tally([
            {"question_type": "t", "is_abstention": False,
             "verdict": J.INCORRECT, "kind": J.FALSE_REFUSAL},
            {"question_type": "t", "is_abstention": False,
             "verdict": J.INCORRECT, "kind": J.RETRIEVAL_MISS},
        ])
        self.assertEqual(0.0, out["accuracy"])
        self.assertEqual(2, out["incorrect"])

    def test_calibration_rate_excludes_retrieval_misses(self):
        # 1 calibration failure and 1 correct answer among the questions the
        # answerer actually got a chance to judge -> 0.5, not 0.3333.
        out = J.tally([
            {"question_type": "t", "is_abstention": False,
             "verdict": J.CORRECT, "kind": J.CORRECT},
            {"question_type": "t", "is_abstention": False,
             "verdict": J.INCORRECT, "kind": J.FALSE_REFUSAL},
            {"question_type": "t", "is_abstention": False,
             "verdict": J.INCORRECT, "kind": J.RETRIEVAL_MISS},
        ])
        self.assertEqual(1, out["retrieval_misses"])
        self.assertEqual(0.5, out["calibration"]["false_refusal_rate"])
        self.assertAlmostEqual(0.3333, out["accuracy"], places=3)

    def test_no_gold_sessions_recorded_does_not_excuse_the_refusal(self):
        # Absence of ground truth is not evidence the retrieval failed.
        out = J.Judge(ScriptedJudge([])).judge(self.refusal([], ["s7"]))
        self.assertEqual(J.FALSE_REFUSAL, out["kind"])

    def test_neither_costs_a_model_call(self):
        llm = ScriptedJudge([])
        J.Judge(llm).judge(self.refusal(["s1"], ["s1"]))
        J.Judge(llm).judge(self.refusal(["s1"], ["s2"]))
        self.assertEqual([], llm.seen)


class ResumeRetriesFailures(unittest.TestCase):
    """A stopped run resumes for free, but a row recorded as `unmeasured` used
    to count as done and be skipped forever -- a permanent hole in a benchmark
    that reports its own denominator. `--retry-unmeasured` retries exactly the
    failures and nothing else."""

    def setUp(self):
        import tempfile
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import answer_run
        self.done_ids = answer_run.done_ids
        self.dir = tempfile.mkdtemp()

    def write(self, rows):
        import json as _json
        p = Path(self.dir) / "rows.jsonl"
        with p.open("w") as fh:
            for r in rows:
                fh.write(_json.dumps(r) + "\n")
        return p

    def test_all_ids_count_as_done_by_default(self):
        p = self.write([{"question_id": "a", "status": "answered"},
                        {"question_id": "b", "status": "unmeasured"}])
        self.assertEqual({"a", "b"}, self.done_ids(p))

    def test_retry_flag_excludes_unmeasured_rows(self):
        p = self.write([{"question_id": "a", "status": "answered"},
                        {"question_id": "b", "status": "unmeasured"}])
        self.assertEqual({"a"}, self.done_ids(p, retry_unmeasured=True))

    def test_a_later_success_supersedes_an_earlier_failure(self):
        # Append-only file: the retry writes a second row for the same id.
        p = self.write([{"question_id": "b", "status": "unmeasured"},
                        {"question_id": "b", "status": "answered"}])
        self.assertEqual({"b"}, self.done_ids(p, retry_unmeasured=True))

    def test_a_later_failure_reopens_an_earlier_success(self):
        p = self.write([{"question_id": "b", "status": "answered"},
                        {"question_id": "b", "status": "unmeasured"}])
        self.assertEqual(set(), self.done_ids(p, retry_unmeasured=True))

    def test_refusals_are_not_retried(self):
        # ABSENT is a real result, not a failure to measure.
        p = self.write([{"question_id": "a", "status": "absent"}])
        self.assertEqual({"a"}, self.done_ids(p, retry_unmeasured=True))

    def test_missing_file_is_empty_not_an_error(self):
        self.assertEqual(set(), self.done_ids(Path(self.dir) / "nope.jsonl", True))


class CapabilitiesDoc(unittest.TestCase):
    """The capability map is generated from the probe reports, so it cannot
    drift from what was measured. Its most important job is reconciling the
    probes against each other: probe1 reported almost everything unsupported,
    and probes 2-6 overturned eight of those verdicts by finding the working
    syntax. A superseded verdict must never appear as a live one."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import capabilities
        self.cap = capabilities

    def test_every_probe_report_is_loaded(self):
        checks = self.cap.load()
        self.assertGreater(len(checks), 150)
        self.assertEqual(6, len({c["probe"] for c in checks}))

    def test_every_overturned_pair_is_real(self):
        # A wrong mapping would print a correction that never happened.
        checks = self.cap.load()
        first = {c["name"]: c for c in checks if c["probe"] == "probe1"}
        later = {c["name"]: c for c in checks
                 if c["probe"] != "probe1" and c["verdict"] == self.cap.WORKS}
        for old, new in self.cap.OVERTURNED:
            self.assertIn(old, first, f"{old} not in probe1")
            self.assertEqual(self.cap.FAILS, first[old]["verdict"],
                             f"{old} did not fail in probe1")
            self.assertIn(new, later, f"{new} never measured working")

    def test_verdicts_are_only_the_three_kinds(self):
        kinds = {c["verdict"] for c in self.cap.load()}
        self.assertTrue(kinds <= {self.cap.WORKS, self.cap.FAILS,
                                  self.cap.SKIPPED}, kinds)


class NonStringGold(unittest.TestCase):
    """LongMemEval gold answers are not all strings. `how many charities` has
    gold 4, an int, and it killed the judge 60 rows into the 500-question run
    with `'int' object has no attribute 'strip'`. Every field that reaches the
    prompt is coerced now."""

    def test_int_gold_is_judged_not_crashed(self):
        llm = ScriptedJudge([{"correct": True, "why": "same number"}])
        out = J.Judge(llm).judge(row(gold=4, answer="4"))
        self.assertEqual(J.CORRECT, out["verdict"])
        self.assertIn("Reference answer: 4", llm.seen[0][1])

    def test_int_answer_is_judged_not_crashed(self):
        llm = ScriptedJudge([{"correct": False, "why": "different"}])
        out = J.Judge(llm).judge(row(gold="four", answer=7))
        self.assertEqual(J.INCORRECT, out["verdict"])

    def test_float_and_bool_survive(self):
        llm = ScriptedJudge([{"correct": True, "why": "y"},
                             {"correct": True, "why": "y"}])
        j = J.Judge(llm)
        self.assertEqual(J.CORRECT, j.judge(row(gold=3.5, answer="3.5"))["verdict"])
        self.assertEqual(J.CORRECT, j.judge(row(gold=True, answer="yes"))["verdict"])

    def test_zero_gold_is_not_treated_as_missing(self):
        # `or ""` turns 0 into "", which would read as "no reference answer"
        # and silently mark a real question unjudged.
        out = J.Judge(ScriptedJudge([])).judge(row(gold=0, answer="0"))
        self.assertEqual(J.UNJUDGED, out["verdict"])
        self.assertIn("no reference answer", out["why"])

    def test_non_string_question_survives(self):
        llm = ScriptedJudge([{"correct": True, "why": "y"}])
        J.Judge(llm).judge(row(question=12345, gold="x", answer="x"))
        self.assertIn("12345", llm.seen[0][1])

"""Day 3b: the answerer prompt, human dates, and a slice that can measure both
halves of the calibration pair.

Aug 17 measurement that forced all of this: 9 of 9 false refusals on the
temporal-reasoning slice HAD their gold claim in the evidence, and every one was
an ordering or duration question. The evidence rendered every timestamp as a raw
Unix epoch and labelled it `said`, when it is really the date the user MENTIONED
the thing -- two claims from one conversation share it even when they describe
events months apart.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docket.answer import SYSTEM, fmt, human_date, render_evidence


# The real refusal, verbatim from state/answers-model.jsonl.
CAR_ISSUE = [
    {"when": 1681138020, "claim": {"subj": "user", "pred": "car_issue_type",
                                   "obj": "GPS system malfunction",
                                   "kind": "event"}},
    {"when": 1681138020, "claim": {"subj": "user", "pred": "had_car_issue_on",
                                   "obj": "2023-03-22", "kind": "event"}},
    {"when": 1679976900, "claim": {"subj": "user", "pred": "has_streaming_service",
                                   "obj": "new service", "kind": "state"}},
]


class HumanDates(unittest.TestCase):
    def test_epoch_becomes_a_readable_date(self):
        self.assertEqual("2023-04-10 (Mon)", human_date(1681138020))

    def test_none_is_named_not_crashed(self):
        self.assertEqual("unknown", human_date(None))

    def test_garbage_falls_back_to_the_raw_value(self):
        self.assertEqual("not-a-time", human_date("not-a-time"))

    def test_no_bare_epoch_survives_into_the_prompt(self):
        block = render_evidence(CAR_ISSUE)
        self.assertNotIn("1681138020", block)
        self.assertNotIn("1679976900", block)
        self.assertIn("2023-04-10 (Mon)", block)

    def test_event_date_inside_the_claim_is_left_alone(self):
        # The date the thing HAPPENED lives in the claim object and must not be
        # rewritten -- it is the only usable date for ordering questions.
        self.assertIn("2023-03-22", render_evidence(CAR_ISSUE))

    def test_two_claims_from_one_conversation_share_a_date(self):
        # The shape that makes mention dates useless for ordering. The prompt
        # no longer explains it (that rewrite measured worse), but the fact is
        # still true and belongs in the README's limitations.
        self.assertEqual(2, render_evidence(CAR_ISSUE).count("said 2023-04-10 (Mon)"))

    def test_superseded_values_also_get_human_dates(self):
        items = [{"when": 1681138020,
                  "claim": {"subj": "user", "pred": "p", "obj": "new"},
                  "superseded": [{"obj": "old", "ts": 1679976900}]}]
        block = render_evidence(items)
        self.assertIn("replaced an earlier value: old (said 2023-03-28", block)
        self.assertNotIn("1679976900", block)


class SystemPrompt(unittest.TestCase):
    """The prompt is the Run A text, restored by measurement.

    Aug 17, identical 25-question slice, one variable:
        Run A  (this prompt)  answered 15  accuracy 0.3333  false refusal 0.375
        rewrite               answered  7  accuracy 0.2381  false refusal 0.6667
    The rewrite added a mention-vs-event-date block and softened the refusal
    rule. It halved the answers and pushed trailing prose from 13/25 to 21/25.
    Reverted. Do not reintroduce it without a measurement that beats Run A.
    """

    def test_refusal_rule_is_the_original_strict_one(self):
        self.assertIn("If the evidence does not contain the answer, reply with "
                      "exactly NOT_IN_MEMORY", SYSTEM)

    def test_the_rewrite_is_gone(self):
        # The two sentences that cost half the answers.
        self.assertNotIn("Do NOT refuse merely because", SYSTEM)
        self.assertNotIn("NOT necessarily the date the thing happened", SYSTEM)

    def test_supersede_rule_survives(self):
        self.assertIn("the most recent one is the current", SYSTEM)

    def test_citation_rule_survives(self):
        self.assertIn("Cite the evidence numbers you used", SYSTEM)

    def test_prompt_is_short_again(self):
        # Length is the suspected mechanism: the longer prompt was followed
        # less literally, not more.
        self.assertLess(len(SYSTEM), 1100)


class DateFormatIsTheOnlyRemainingVariable(unittest.TestCase):
    def test_human_is_the_default(self):
        self.assertIn("said 2023-04-10 (Mon)", render_evidence(CAR_ISSUE))

    def test_epoch_mode_restores_the_exact_old_rendering(self):
        block = render_evidence(CAR_ISSUE, dates="epoch")
        self.assertIn("said 1681138020", block)
        self.assertNotIn("2023-04-10 (Mon)", block)

    def test_label_is_back_to_said(self):
        # `mentioned` shipped with the rewrite and is reverted too, so the
        # only diff from Run A is the date format itself.
        self.assertNotIn("mentioned", render_evidence(CAR_ISSUE))

    def test_event_date_inside_the_claim_is_untouched_in_both_modes(self):
        for mode in ("human", "epoch"):
            self.assertIn("2023-03-22", render_evidence(CAR_ISSUE, dates=mode))


class Sampler(unittest.TestCase):
    """`--limit` slices, `--sample` samples. The difference is measurable."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import answer_run
        self.select = answer_run.select

    def instances(self):
        class I:
            def __init__(self, qid, qtype, abst):
                self.question_id = qid
                self.question_type = qtype
                self.is_abstention = abst
        out = []
        # mirrors the real oracle mix closely enough to test the shape
        for n, t in (("temporal-reasoning", 133), ("multi-session", 133),
                     ("knowledge-update", 78), ("single-session-user", 70),
                     ("single-session-assistant", 56),
                     ("single-session-preference", 30)):
            for i in range(t):
                out.append(I(f"{n}_{i}", n, False))
        for i in range(30):
            out.append(I(f"abs_{i}", "temporal-reasoning", True))
        return out

    def test_no_sample_returns_everything(self):
        rows = self.instances()
        got, forced = self.select(rows, None, 7, 0)
        self.assertEqual(len(rows), len(got))
        self.assertFalse(forced)

    def test_sample_is_deterministic_for_a_seed(self):
        rows = self.instances()
        a, _ = self.select(rows, 60, 7, 0)
        b, _ = self.select(rows, 60, 7, 0)
        self.assertEqual([i.question_id for i in a], [i.question_id for i in b])

    def test_a_different_seed_gives_a_different_slice(self):
        rows = self.instances()
        a, _ = self.select(rows, 60, 7, 0)
        b, _ = self.select(rows, 60, 8, 0)
        self.assertNotEqual([i.question_id for i in a], [i.question_id for i in b])

    def test_sample_size_is_respected(self):
        got, _ = self.select(self.instances(), 60, 7, 0)
        self.assertEqual(60, len(got))

    def test_sample_spans_several_categories_unlike_limit(self):
        got, _ = self.select(self.instances(), 60, 7, 0)
        kinds = {i.question_type for i in got}
        self.assertGreater(len(kinds), 3)

    def test_abstention_floor_is_honoured(self):
        got, forced = self.select(self.instances(), 60, 7, 10)
        self.assertGreaterEqual(sum(1 for i in got if i.is_abstention), 10)
        self.assertTrue(forced)

    def test_floor_below_proportional_share_is_not_flagged_as_forced(self):
        # 30 abstention in 530 is ~5.7%, so ~3 of 60. A floor of 1 changes
        # nothing and must not print the not-proportional warning.
        _, forced = self.select(self.instances(), 60, 7, 1)
        self.assertFalse(forced)

    def test_asking_for_more_than_exists_returns_everything(self):
        rows = self.instances()
        got, _ = self.select(rows, 99999, 7, 0)
        self.assertEqual(len(rows), len(got))


if __name__ == "__main__":
    unittest.main()

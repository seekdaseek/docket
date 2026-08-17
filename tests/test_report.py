"""The results generator. Its arithmetic is small and every piece of it has
already been got wrong by hand at least once during this build, which is the
reason the report is generated at all."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import report


def write(dirpath, name, rows):
    p = Path(dirpath) / name
    with p.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return str(p)


class LoadIsAppendOnlyAware(unittest.TestCase):
    def test_last_row_for_an_id_wins(self):
        d = tempfile.mkdtemp()
        p = write(d, "j.jsonl", [{"question_id": "a", "verdict": "incorrect"},
                                 {"question_id": "a", "verdict": "correct"}])
        rows = report.load(p)
        self.assertEqual(1, len(rows))
        self.assertEqual("correct", rows["a"]["verdict"])

    def test_blank_and_broken_lines_are_skipped(self):
        d = tempfile.mkdtemp()
        p = Path(d) / "j.jsonl"
        p.write_text('{"question_id": "a"}\n\nnot json\n{"question_id": "b"}\n')
        self.assertEqual({"a", "b"}, set(report.load(str(p))))


class Recall(unittest.TestCase):
    def rows(self, gold, cited, qtype="t", abst=False):
        return {"q": {"question_id": "q", "question_type": qtype,
                      "is_abstention": abst, "gold_sessions": gold,
                      "evidence": [{"session": s} for s in cited]}}

    def test_hit_when_a_gold_session_is_cited(self):
        out = report.recall_by_type(self.rows(["g"], ["g", "x"]))
        self.assertEqual({"n": 1, "hit": 1}, out["t"])

    def test_miss_when_no_gold_session_is_cited(self):
        out = report.recall_by_type(self.rows(["g"], ["x"]))
        self.assertEqual({"n": 1, "hit": 0}, out["t"])

    def test_abstention_rows_are_excluded(self):
        # They have no gold sessions by construction; counting them would
        # drag every recall figure down.
        self.assertEqual({}, report.recall_by_type(
            self.rows([], [], abst=True)))

    def test_rows_without_ground_truth_are_excluded(self):
        self.assertEqual({}, report.recall_by_type(self.rows([], ["x"])))


class Arithmetic(unittest.TestCase):
    """`answered` and `wrong_answers` are derived by subtraction, and a wrong
    subtraction here would misreport precision — the number most likely to be
    quoted out of context."""

    def test_answered_excludes_both_refusal_kinds(self):
        # 10 graded: 4 correct, 2 answered-wrong, 3 false refusals, 1 miss.
        correct, wrong, false_ref, ret_miss = 4, 6, 3, 1
        graded = correct + wrong
        answered = graded - false_ref - ret_miss
        wrong_answers = wrong - false_ref - ret_miss
        self.assertEqual(6, answered)
        self.assertEqual(2, wrong_answers)
        self.assertEqual(correct + wrong_answers, answered)

    def test_precision_is_over_answered_not_over_graded(self):
        # 165/234, not 165/470. Conflating them overstates by 2x here.
        self.assertEqual(0.7051, report.pct(165, 234))
        self.assertEqual(0.3511, report.pct(165, 470))

    def test_false_refusal_rate_excludes_retrieval_misses(self):
        self.assertEqual(0.3624, report.pct(133, 470 - 103))

    def test_pct_of_zero_denominator_is_none_not_zero(self):
        # A run where nothing was graded must not read as 0% accuracy.
        self.assertIsNone(report.pct(0, 0))

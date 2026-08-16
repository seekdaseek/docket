"""Retrieval scoring is checkable offline against LongMemEval's own
`answer_session_ids`, so it must be right before any model spend is justified
by it. Abstention questions have no gold sessions and are never averaged in."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "score_retrieval", ROOT / "tools" / "score_retrieval.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


def row(qid, qtype="temporal-reasoning", gold=("s1",), sessions=(), status="answered"):
    return {"question_id": qid, "question_type": qtype,
            "is_abstention": qid.endswith("_abs"),
            "question": "q", "gold": "a", "gold_sessions": list(gold),
            "status": status,
            "evidence": [{"session": s, "triple": ["user", "p", "o"],
                          "matched_terms": {"p": 1.0}} for s in sessions]}


class Scoring(unittest.TestCase):
    def test_recall_counts_a_question_once_however_many_gold_claims_it_cited(self):
        out = sr.score([row("a", sessions=["s1", "s1", "s9"]),
                        row("b", sessions=["s7"])])
        self.assertEqual(out["recall"], 0.5)
        self.assertEqual(out["density"], round(2 / 4, 4))

    def test_a_question_with_no_evidence_is_counted_not_skipped(self):
        out = sr.score([row("a", sessions=[]), row("b", sessions=["s1"])])
        self.assertEqual(out["no_evidence"], 1)
        self.assertEqual(out["recall"], 0.5)

    def test_abstention_questions_never_enter_recall(self):
        out = sr.score([row("a", sessions=["s1"]),
                        row("x_abs", gold=(), sessions=[], status="absent")])
        self.assertEqual(out["questions"], 1)
        self.assertEqual(out["recall"], 1.0)
        self.assertEqual(out["abstention"], {"n": 1, "refused": 1, "rate": 1.0})

    def test_an_abstention_that_answered_anyway_is_not_counted_as_refused(self):
        out = sr.score([row("x_abs", gold=(), sessions=["s4"], status="answered")])
        self.assertEqual(out["abstention"]["refused"], 0)

    def test_per_type_recall_is_reported_separately(self):
        out = sr.score([row("a", "knowledge-update", sessions=["s1"]),
                        row("b", "knowledge-update", sessions=["s9"]),
                        row("c", "multi-session", sessions=["s1"])])
        self.assertEqual(out["by_type"]["knowledge-update"]["recall"], 0.5)
        self.assertEqual(out["by_type"]["multi-session"]["recall"], 1.0)

    def test_the_last_row_for_a_question_wins_because_the_file_is_append_only(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps(row("a", sessions=["s9"])) + "\n")
            fh.write(json.dumps(row("a", sessions=["s1"])) + "\n")
            path = fh.name
        rows = sr.load(Path(path))
        self.assertEqual(len(rows), 1)
        self.assertEqual(sr.score(rows)["recall"], 1.0)

    def test_a_corrupt_line_is_skipped_rather_than_killing_the_report(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write("{not json\n")
            fh.write(json.dumps(row("a", sessions=["s1"])) + "\n")
            path = fh.name
        self.assertEqual(len(sr.load(Path(path))), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class Diagnose(unittest.TestCase):
    """The three misses need three different fixes and only one costs money,
    so the classifier must not collapse them."""

    def setUp(self):
        spec2 = importlib.util.spec_from_file_location(
            "diagnose_misses", ROOT / "tools" / "diagnose_misses.py")
        self.dm = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(self.dm)
        sys.path.insert(0, str(ROOT))
        from docket.gate import build_gate
        self.claims = {
            "g1": {"subj": "user", "pred": "attended", "obj": "Data Analysis using Python webinar",
                   "kind": "event", "card": "many", "sid": "gold", "turn": 1, "ts": 100},
            "n1": {"subj": "user", "pred": "attended", "obj": "blues workshop",
                   "kind": "event", "card": "many", "sid": "other", "turn": 1, "ts": 100},
        }
        self.by_session = {"gold": [self.claims["g1"]], "other": [self.claims["n1"]]}
        self.gate = build_gate(self.claims)

    def _row(self, **over):
        row = {"question_id": "q", "question": "which webinar did I attend",
               "gold": "Data Analysis using Python webinar",
               "gold_sessions": ["gold"], "evidence": [],
               "retrieval": {"hits": 5, "kept": 5, "dropped_future": 0,
                             "dropped_missing": 0}}
        row.update(over)
        return row

    def test_no_claims_from_the_gold_session_is_an_extraction_gap(self):
        out = self.dm.classify(self._row(gold_sessions=["never_extracted"]),
                               self.gate, self.by_session, 50)
        self.assertEqual(out["verdict"], self.dm.EXTRACTION)
        self.assertEqual(out["gold_claims"], 0)

    def test_a_gold_claim_that_ranks_is_a_ranking_gap_not_an_extraction_one(self):
        out = self.dm.classify(self._row(), self.gate, self.by_session, 50)
        self.assertEqual(out["verdict"], self.dm.RANKING)
        self.assertIsNotNone(out["best_rank"])

    def test_a_claim_below_the_candidate_cut_is_ranking_even_with_drops(self):
        # The Aug 16 misattribution: gold at rank 177 against a cut of 12 was
        # never a candidate, so dropped_future beside it is coincidence. The
        # first classifier called 51 of these FILTER and pointed the whole
        # diagnosis at the wrong layer.
        row = self._row(retrieval={"hits": 12, "kept": 4, "dropped_future": 8,
                                   "dropped_missing": 0, "candidates": 12})
        gate = self.gate
        by = dict(self.by_session)
        # bury the gold claim behind many better matches
        from docket.gate import build_gate
        claims = dict(self.claims)
        for i in range(30):
            claims[f"pad{i}"] = {"subj": "user", "pred": "attended",
                                 "obj": "webinar webinar webinar",
                                 "kind": "event", "card": "many",
                                 "sid": f"pad{i}", "turn": 1, "ts": 1}
        out = self.dm.classify(row, build_gate(claims), by, 200)
        self.assertEqual(out["verdict"], self.dm.RANKING)
        self.assertIn("never a candidate", out["detail"])

    def test_dropped_future_makes_it_a_filter_gap(self):
        row = self._row(retrieval={"hits": 5, "kept": 0, "dropped_future": 5,
                                   "dropped_missing": 0})
        out = self.dm.classify(row, self.gate, self.by_session, 50)
        self.assertEqual(out["verdict"], self.dm.FILTER)

    def test_a_gold_claim_that_scores_nothing_is_an_extraction_gap(self):
        row = self._row(question="quantum chromodynamics lattice")
        out = self.dm.classify(row, self.gate, self.by_session, 50)
        self.assertEqual(out["verdict"], self.dm.EXTRACTION)
        self.assertGreater(out["gold_claims"], 0)


class DateAudit(unittest.TestCase):
    """The as-of filter is the project's own idea, so loosening it has to be
    driven by what the dataset does, not by the one instance that failed."""

    def setUp(self):
        spec3 = importlib.util.spec_from_file_location(
            "date_audit", ROOT / "tools" / "date_audit.py")
        self.da = importlib.util.module_from_spec(spec3)
        spec3.loader.exec_module(self.da)

    def _inst(self, offsets, gold_idx=(0,)):
        from datetime import datetime, timedelta, timezone

        class S:
            def __init__(self, sid, when):
                self.session_id, self.when = sid, when

        class I:
            pass

        base = datetime(2023, 5, 20, 6, 0, tzinfo=timezone.utc)
        inst = I()
        inst.asked_at = base
        inst.sessions = [S(f"s{i}", base + timedelta(seconds=o))
                         for i, o in enumerate(offsets)]
        inst.evidence_session_ids = [f"s{i}" for i in gold_idx]
        return inst

    def test_evidence_after_the_question_is_counted(self):
        out = self.da.audit([self._inst([900])])          # 15 minutes late
        self.assertEqual(out["instances_with_evidence_after_the_question"], 1)
        self.assertEqual(out["evidence_offset_distribution"]["0-1h"], 1)

    def test_a_strict_filter_drops_it_and_a_day_of_tolerance_keeps_it(self):
        out = self.da.audit([self._inst([900])])
        self.assertEqual(out["tolerance_table"]["strict"]["instances_whose_evidence_all_fits"], 0)
        self.assertEqual(out["tolerance_table"]["+1d"]["instances_whose_evidence_all_fits"], 1)

    def test_the_tolerance_table_also_counts_what_it_lets_in(self):
        # evidence 15 min late, an unrelated session 5 hours late
        out = self.da.audit([self._inst([900, 5 * 3600], gold_idx=(0,))])
        self.assertEqual(out["tolerance_table"]["+1h"]["non_evidence_sessions_admitted"], 0)
        self.assertEqual(out["tolerance_table"]["+6h"]["non_evidence_sessions_admitted"], 1)

    def test_evidence_before_the_question_needs_no_tolerance(self):
        out = self.da.audit([self._inst([-3600])])
        self.assertEqual(out["instances_with_evidence_after_the_question"], 0)
        self.assertEqual(out["tolerance_table"]["strict"]["pct"], 100.0)

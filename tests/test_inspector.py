"""The inspector is a single static file, so its risks are escaping and shape.

It is served with no process behind it on purpose: a link that a judge opens
weeks later must not depend on anything still running.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import inspector


def answers(**kw):
    base = {"question_id": "q1", "question_type": "temporal-reasoning",
            "is_abstention": False, "question": "Which came first?",
            "asked_at": 1681168020, "gold": "the GPS fault",
            "gold_sessions": ["gsess"], "status": "absent", "answer": None,
            "reason": "the model was given evidence and said it does not contain it",
            "evidence": [{"n": 1, "session": "gsess", "said_at": 1681138020,
                          "triple": ["user", "car_issue_type", "GPS fault"],
                          "matched_terms": {"car": 3.1}, "used": None},
                         {"n": 2, "session": "other", "said_at": 1679976900,
                          "triple": ["user", "owns", "shoes"],
                          "matched_terms": {}, "used": None}],
            "retrieval": {"hits": 12, "candidates": 12, "kept": 6,
                          "dropped_future": 4, "admitted_by_tolerance": 1}}
    base.update(kw)
    return {base["question_id"]: base}


def judged(**kw):
    base = {"question_id": "q1", "verdict": "incorrect", "kind": "false_refusal",
            "why": "refused with a gold session in the evidence",
            "judge_model": "claude-sonnet-5"}
    base.update(kw)
    return {base["question_id"]: base}


class Build(unittest.TestCase):
    def test_gold_sessions_are_flagged_in_the_evidence(self):
        # The whole point of the page: a refusal with the answer present must
        # be visible, not asserted.
        row = inspector.build(answers(), judged())[0]
        self.assertEqual(1, row["ev"][0]["g"])
        self.assertEqual(0, row["ev"][1]["g"])

    def test_verdict_and_kind_are_joined_from_the_judged_file(self):
        row = inspector.build(answers(), judged())[0]
        self.assertEqual("incorrect", row["v"])
        self.assertEqual("false_refusal", row["k"])

    def test_a_question_with_no_judged_row_is_unjudged_not_dropped(self):
        rows = inspector.build(answers(), {})
        self.assertEqual(1, len(rows))
        self.assertEqual("unjudged", rows[0]["v"])

    def test_abstention_rows_carry_no_reference_answer(self):
        rows = inspector.build(answers(is_abstention=True), judged())
        self.assertEqual("", rows[0]["gold"])
        self.assertEqual(1, rows[0]["abs"])

    def test_epochs_are_rendered_as_dates(self):
        row = inspector.build(answers(), judged())[0]
        self.assertEqual("2023-04-10", row["at"])
        self.assertEqual("2023-04-10", row["ev"][0]["d"])

    def test_retrieval_report_is_carried_through(self):
        row = inspector.build(answers(), judged())[0]
        self.assertEqual(4, row["r"]["dropped_future"])
        self.assertEqual(1, row["r"]["admitted_by_tolerance"])


class Escaping(unittest.TestCase):
    """Transcripts are user text. Anything can be in them."""

    def render(self, answers_map, judged_map):
        d = tempfile.mkdtemp()
        pa, pj = Path(d) / "a.jsonl", Path(d) / "j.jsonl"
        for p, m in ((pa, answers_map), (pj, judged_map)):
            with p.open("w") as fh:
                for r in m.values():
                    fh.write(json.dumps(r) + "\n")
        rows = inspector.build(inspector.load(str(pa)), inspector.load(str(pj)))
        blob = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
        return inspector.PAGE.replace("__DATA__", blob.replace("</", "<\\/"))

    def test_a_closing_script_tag_in_the_data_cannot_break_out(self):
        page = self.render(answers(question="what about </script><img src=x>?"),
                           judged())
        body = page.split('type="application/json">')[1].split("</script>")[0]
        self.assertNotIn("</script", body)
        self.assertIn("<\\/script", body)

    def test_the_payload_still_parses_as_json_after_escaping(self):
        page = self.render(answers(question="</script> & <b>bold</b>"), judged())
        body = page.split('type="application/json">')[1].split("</script>")[0]
        rows = json.loads(body.replace("<\\/", "</"))
        self.assertIn("</script>", rows[0]["q"])

    def test_unicode_survives_without_escaping_to_mojibake(self):
        page = self.render(answers(question="caf\u00e9 \u2014 it's fine"), judged())
        self.assertIn("caf\u00e9", page)


class Page(unittest.TestCase):
    def test_no_external_asset_is_referenced(self):
        # It has to work from file:// and from any static host, forever.
        for bad in ("http://", "https://", "cdn", "<link", "src=\"//"):
            self.assertNotIn(bad, inspector.PAGE)

    def test_the_data_placeholder_exists_exactly_once(self):
        self.assertEqual(1, inspector.PAGE.count("__DATA__"))


class FilteredCount(unittest.TestCase):
    """A filtered view has to state its own size. Otherwise the headline claim
    -- 133 refusals with the answer present -- is taken on trust, which is the
    one thing this project does not ask of a reader."""

    def test_the_page_renders_a_count_element(self):
        self.assertIn('id=count', inspector.PAGE)
        self.assertIn("showing <b>'+rows.length+'</b> of ", inspector.PAGE)

    def test_the_count_names_abstention_rows_separately(self):
        # Abstention questions keep their underlying question_type, so a
        # category filter mixes them in. Saying so beats leaving it inferred.
        self.assertIn("unanswerable by construction", inspector.PAGE)

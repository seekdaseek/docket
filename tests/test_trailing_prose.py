"""The trailing-prose parse bug, pinned by the real strings that caused it.

Aug 17, first paid run: 14 of 25 answers were recorded as measurement failures
because the model emitted correct JSON and then explained itself. Seven of those
were correct abstentions. Every string below is copied from that run.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docket.llm import Anthropic, LLMError, extract_json


# Verbatim from state/answers-model.jsonl, Aug 17.
REAL_ABSTENTION = (
    '{"answer": "NOT_IN_MEMORY", "used": []}\n\nThe evidence mentions a GPS '
    "system malfunction on 3/22, but does not specify that this occurred "
    "after the car's first service."
)
REAL_ANSWER_WITH_FENCE = (
    '{"answer": "Effective Time Management workshop", "used": [2, 3]}\n```\n\n'
    "The Effective Time Management workshop was attended on 1685307840."
)
REAL_PRETTY_PRINTED = (
    '{\n  "answer": "NOT_IN_MEMORY",\n  "used": []\n}\n```\n\nThe evidence only '
    "mentions bike repairs in mid-February 2023."
)


class ExtractJson(unittest.TestCase):
    def test_the_seven_lost_abstentions_now_parse(self):
        value, trailing = extract_json(REAL_ABSTENTION)
        self.assertEqual("NOT_IN_MEMORY", value["answer"])
        self.assertEqual([], value["used"])
        self.assertTrue(trailing)

    def test_stray_closing_fence_with_no_opening_one(self):
        # strip_fences only strips a fence that OPENS the text, so this shape
        # is invisible to it.
        value, _ = extract_json(REAL_ANSWER_WITH_FENCE)
        self.assertEqual("Effective Time Management workshop", value["answer"])
        self.assertEqual([2, 3], value["used"])

    def test_pretty_printed_json_then_prose(self):
        value, _ = extract_json(REAL_PRETTY_PRINTED)
        self.assertEqual("NOT_IN_MEMORY", value["answer"])

    def test_trailing_prose_containing_braces_does_not_extend_the_parse(self):
        # The reason for raw_decode over a first-brace-to-last-brace slice.
        # backbone's tolerant parser sliced to the LAST brace and would swallow
        # the commentary here, then fail.
        text = '{"answer": "x", "used": [1]}\n\nThe {workshop} came first, {not} the other.'
        value, trailing = extract_json(text)
        self.assertEqual({"answer": "x", "used": [1]}, value)
        self.assertIn("workshop", trailing)

    def test_prose_before_the_json_with_a_stray_brace(self):
        text = 'Here is the result {as requested}:\n{"answer": "y", "used": []}'
        value, _ = extract_json(text)
        self.assertEqual("y", value["answer"])

    def test_clean_json_reports_no_trailing(self):
        value, trailing = extract_json('{"answer": "clean", "used": [1]}')
        self.assertEqual("clean", value["answer"])
        self.assertEqual("", trailing)

    def test_a_leading_fence_still_works(self):
        value, trailing = extract_json('```json\n{"answer": "z", "used": []}\n```')
        self.assertEqual("z", value["answer"])

    def test_a_bare_list_still_works(self):
        # extract.py expects a list; it must not regress.
        value, _ = extract_json('[{"turn": 0}, {"turn": 1}]')
        self.assertEqual([{"turn": 0}, {"turn": 1}], value)

    def test_genuinely_unparseable_output_still_raises(self):
        # The project's rule is intact: no JSON means a measurement failure,
        # not a quietly empty result.
        with self.assertRaises(LLMError):
            extract_json("I am sorry, I cannot help with that request.")

    def test_empty_text_raises(self):
        with self.assertRaises(LLMError):
            extract_json("   ")

    def test_truncated_json_raises(self):
        with self.assertRaises(LLMError):
            extract_json('{"answer": "half a th')


class ClientCountsTrailingProse(unittest.TestCase):
    """The counter exists so a chatty model is visible, not merely tolerated."""

    def client(self, text):
        payload = {"content": [{"type": "text", "text": text}],
                   "usage": {"input_tokens": 10, "output_tokens": 5},
                   "stop_reason": "end_turn"}

        class Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def read(self_inner):
                return json.dumps(payload).encode()

        return Anthropic("k", opener=lambda req, timeout=None: Resp())

    def test_counter_rises_on_trailing_prose(self):
        a = self.client(REAL_ABSTENTION)
        a.complete_json("s", "u")
        self.assertEqual(1, a.trailing_prose)

    def test_counter_stays_at_zero_on_clean_json(self):
        a = self.client('{"answer": "clean", "used": []}')
        out = a.complete_json("s", "u")
        self.assertEqual({"answer": "clean", "used": []}, out)
        self.assertEqual(0, a.trailing_prose)

    def test_unparseable_still_raises_through_complete_json(self):
        a = self.client("no json at all here")
        with self.assertRaises(LLMError):
            a.complete_json("s", "u")


if __name__ == "__main__":
    unittest.main()

"""Offline tests.

The HydraDB stub here REPLAYS recorded response envelopes. It deliberately
does not interpret Cypher: a stub that answers queries by simulating a graph
proves only that the simulation agrees with itself. What is asserted is what
the client does with bytes the real server actually produced, plus how it
behaves when the server does something awkward.

Graph semantics are verified against the live node by tools/probe.py.
"""
from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.error
from datetime import datetime, timezone

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from docket.dataset import Instance, Session, Turn, load_instance, stats, unique_sessions
from docket.hydra import (HydraClient, HydraError, HydraTruncated, cell_value,
                          lit)
from docket.timeparse import BadTimestamp, order_sessions, parse

# Recorded verbatim from a live node on 13 Aug 2026.
READ_ENVELOPE = {
    "query_id": "http-query-2", "columns": ["id"],
    "rows": [[{"type": "vertex_id", "value": 2}]],
    "read_epoch": 1, "next_cursor": None,
    "bookmark": "sgk:1:64656661756c74:64656661756c74:63656c6c2d30:1",
}
WRITE_ENVELOPE = {
    "query_id": "http-query-1", "columns": [], "rows": [],
    "read_epoch": None, "next_cursor": None,
    "bookmark": "sgk:1:64656661756c74:64656661756c74:63656c6c2d30:1",
}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def opener_for(pages, record=None, fail_after=None):
    """Serve the given payloads in order; record the request bodies."""
    state = {"i": 0}

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        if record is not None:
            record.append({"body": body, "headers": dict(req.headers)})
        i = state["i"]
        state["i"] += 1
        if fail_after is not None and i >= fail_after:
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request",
                                         {}, io.BytesIO(b'{"error":"bad shape"}'))
        payload = pages[min(i, len(pages) - 1)]
        return FakeResponse(json.dumps(payload).encode())

    return _open


class TestCells(unittest.TestCase):
    def test_typed_cell_unwrapped(self):
        self.assertEqual(cell_value({"type": "vertex_id", "value": 2}), 2)

    def test_unknown_shape_is_returned_intact_not_guessed(self):
        weird = {"type": "node", "labels": ["X"]}
        self.assertEqual(cell_value(weird), weird)

    def test_plain_value_passes_through(self):
        self.assertEqual(cell_value(7), 7)


class TestLiterals(unittest.TestCase):
    def test_quote_is_escaped(self):
        self.assertEqual(lit("it's"), "'it\\'s'")

    def test_backslash_escaped_before_quote(self):
        self.assertEqual(lit("a\\b"), "'a\\\\b'")

    def test_newline_escaped(self):
        self.assertEqual(lit("a\nb"), "'a\\nb'")

    def test_none_and_bools(self):
        self.assertEqual(lit(None), "null")
        self.assertEqual(lit(True), "true")
        self.assertEqual(lit(False), "false")

    def test_bool_not_rendered_as_int(self):
        self.assertNotEqual(lit(True), "1")

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            lit({"a": 1})

    def test_nan_raises_rather_than_writing_nan(self):
        with self.assertRaises(ValueError):
            lit(float("nan"))


class TestClientParsing(unittest.TestCase):
    def test_rows_are_column_keyed_dicts(self):
        c = HydraClient(opener=opener_for([READ_ENVELOPE]))
        self.assertEqual(c.query("RETURN 1").rows, [{"id": 2}])

    def test_scalar(self):
        c = HydraClient(opener=opener_for([READ_ENVELOPE]))
        self.assertEqual(c.query("RETURN 1").scalar(), 2)

    def test_scalar_on_empty_write_result_is_default(self):
        c = HydraClient(opener=opener_for([WRITE_ENVELOPE]))
        self.assertIsNone(c.query("CREATE (a)").scalar())

    def test_scalar_refuses_multi_column(self):
        env = dict(READ_ENVELOPE, columns=["a", "b"],
                   rows=[[{"value": 1}, {"value": 2}]])
        c = HydraClient(opener=opener_for([env]))
        with self.assertRaises(HydraError):
            c.query("RETURN 1,2").scalar()

    def test_write_envelope_has_no_rows(self):
        c = HydraClient(opener=opener_for([WRITE_ENVELOPE]))
        self.assertEqual(len(c.query("CREATE (a)")), 0)


class TestBookmark(unittest.TestCase):
    def test_bookmark_is_captured_then_sent_on_next_request(self):
        rec = []
        c = HydraClient(opener=opener_for([WRITE_ENVELOPE, READ_ENVELOPE], rec))
        c.query("CREATE (a)")
        c.query("MATCH (a) RETURN a")
        self.assertNotIn("bookmark", rec[0]["body"])
        self.assertEqual(rec[1]["body"]["bookmark"], WRITE_ENVELOPE["bookmark"])

    def test_bookmark_can_be_disabled_for_servers_that_reject_it(self):
        rec = []
        c = HydraClient(opener=opener_for([WRITE_ENVELOPE, READ_ENVELOPE], rec),
                        send_bookmark=False)
        c.query("CREATE (a)")
        c.query("MATCH (a) RETURN a")
        self.assertNotIn("bookmark", rec[1]["body"])

    def test_headers_and_cell_are_sent(self):
        rec = []
        c = HydraClient(token="tok", opener=opener_for([READ_ENVELOPE], rec))
        c.query("RETURN 1")
        self.assertEqual(rec[0]["headers"]["Authorization"], "Bearer tok")
        self.assertEqual(rec[0]["headers"]["X-graph-namespace"], "default")
        self.assertEqual(rec[0]["body"]["cell_id"], "cell-0")


class TestPagination(unittest.TestCase):
    def test_pages_are_followed_and_concatenated(self):
        p1 = dict(READ_ENVELOPE, next_cursor="c1")
        p2 = dict(READ_ENVELOPE, rows=[[{"type": "vertex_id", "value": 3}]],
                  next_cursor=None)
        c = HydraClient(opener=opener_for([p1, p2]))
        r = c.query("MATCH (n) RETURN n")
        self.assertEqual([x["id"] for x in r.rows], [2, 3])
        self.assertEqual(r.pages, 2)

    def test_cursor_is_sent_on_the_second_request(self):
        rec = []
        p1 = dict(READ_ENVELOPE, next_cursor="c1")
        p2 = dict(READ_ENVELOPE, next_cursor=None)
        HydraClient(opener=opener_for([p1, p2], rec)).query("MATCH (n) RETURN n")
        self.assertEqual(rec[1]["body"]["cursor"], "c1")

    def test_rejected_page_two_raises_instead_of_truncating(self):
        p1 = dict(READ_ENVELOPE, next_cursor="c1")
        c = HydraClient(opener=opener_for([p1], fail_after=1))
        with self.assertRaises(HydraTruncated) as ctx:
            c.query("MATCH (n) RETURN n")
        self.assertIn("NOT the whole answer", str(ctx.exception))

    def test_follow_cursor_off_still_refuses_to_return_a_partial(self):
        p1 = dict(READ_ENVELOPE, next_cursor="c1")
        c = HydraClient(opener=opener_for([p1]))
        with self.assertRaises(HydraTruncated):
            c.query("MATCH (n) RETURN n", follow_cursor=False)


class TestClientErrors(unittest.TestCase):
    def test_http_error_carries_code_and_body(self):
        c = HydraClient(opener=opener_for([READ_ENVELOPE], fail_after=0))
        with self.assertRaises(HydraError) as ctx:
            c.query("RETURN 1")
        self.assertIn("http 400", str(ctx.exception))

    def test_non_json_body_named_as_such(self):
        def _open(req, timeout=None):
            return FakeResponse(b"<html>nope</html>")
        c = HydraClient(opener=_open)
        with self.assertRaises(HydraError) as ctx:
            c.query("RETURN 1")
        self.assertIn("not JSON", str(ctx.exception))

    def test_empty_body_named_as_such(self):
        def _open(req, timeout=None):
            return FakeResponse(b"")
        c = HydraClient(opener=_open)
        with self.assertRaises(HydraError) as ctx:
            c.query("RETURN 1")
        self.assertIn("empty response", str(ctx.exception))

    def test_a_400_means_the_node_is_up_and_answering(self):
        """The old test asserted the opposite and the CODE was right.

        A 400 carrying a JSON error body is a live node rejecting a query.
        Treating it as "not ready" burned sixty seconds waiting for a node that
        was already serving.
        """
        c = HydraClient(opener=opener_for([READ_ENVELOPE], fail_after=0))
        self.assertIsInstance(c.wait_ready(seconds=0.2, interval=0.05), float)

    def test_wait_ready_gives_up_when_nothing_is_listening(self):
        def _open(req, timeout=None):
            raise urllib.error.URLError("Connection refused")
        c = HydraClient(opener=_open)
        with self.assertRaises(HydraError) as ctx:
            c.wait_ready(seconds=0.2, interval=0.05)
        self.assertIn("not ready", str(ctx.exception))

    def test_wait_ready_returns_once_the_node_answers(self):
        c = HydraClient(opener=opener_for([READ_ENVELOPE]))
        self.assertIsInstance(c.wait_ready(seconds=1), float)


class TestTimestamps(unittest.TestCase):
    def test_parses_the_real_format(self):
        dt = parse("2023/04/10 (Mon) 23:07")
        self.assertEqual((dt.year, dt.month, dt.day, dt.hour, dt.minute),
                         (2023, 4, 10, 23, 7))
        self.assertEqual(dt.tzinfo, timezone.utc)

    def test_fromisoformat_really_cannot_read_it(self):
        with self.assertRaises(ValueError):
            datetime.fromisoformat("2023/04/10 (Mon) 23:07")

    def test_wrong_weekday_is_rejected(self):
        with self.assertRaises(BadTimestamp):
            parse("2023/04/10 (Thu) 23:07")

    def test_garbage_is_rejected(self):
        for bad in ["", "2023-04-10T23:07", "not a date", None, 17]:
            with self.assertRaises(BadTimestamp):
                parse(bad)

    def test_ordering_fixes_the_unsorted_oracle_case(self):
        ids = ["a_2", "a_3", "a_1"]
        dates = ["2023/04/10 (Mon) 17:50", "2023/04/10 (Mon) 14:47",
                 "2023/04/10 (Mon) 17:15"]
        self.assertEqual([sid for _, sid, _ in order_sessions(ids, dates)],
                         ["a_3", "a_1", "a_2"])

    def test_length_mismatch_raises(self):
        with self.assertRaises(BadTimestamp):
            order_sessions(["a"], ["2023/04/10 (Mon) 17:50",
                                   "2023/04/10 (Mon) 14:47"])


def raw_instance(qid="gpt4_x", **over):
    base = {
        "question_id": qid, "question_type": "temporal-reasoning",
        "question": "What was the first issue?", "answer": "GPS",
        "question_date": "2023/04/10 (Mon) 23:07",
        "haystack_dates": ["2023/04/10 (Mon) 17:50", "2023/04/10 (Mon) 14:47"],
        "haystack_session_ids": ["s2", "s1"],
        "haystack_sessions": [
            [{"role": "user", "content": "later"}],
            [{"role": "user", "content": "earlier"}],
        ],
        "answer_session_ids": ["s1"],
    }
    base.update(over)
    return base


class TestDataset(unittest.TestCase):
    def test_sessions_come_back_in_time_order_not_file_order(self):
        inst = load_instance(raw_instance())
        self.assertEqual(inst.session_ids, ["s1", "s2"])
        self.assertEqual(inst.sessions[0].turns[0].content, "earlier")

    def test_evidence_ids_are_kept_for_citation_scoring(self):
        self.assertEqual(load_instance(raw_instance()).evidence_session_ids, ["s1"])

    def test_abstention_flag(self):
        self.assertTrue(load_instance(raw_instance("gpt4_x_abs")).is_abstention)
        self.assertFalse(load_instance(raw_instance("gpt4_x")).is_abstention)

    def test_session_count_mismatch_raises(self):
        bad = raw_instance(haystack_sessions=[[{"role": "user", "content": "x"}]])
        with self.assertRaises(ValueError):
            load_instance(bad)

    def test_malformed_turn_raises_rather_than_being_dropped(self):
        bad = raw_instance(haystack_sessions=[[{"role": "user"}],
                                              [{"role": "user", "content": "b"}]])
        with self.assertRaises(ValueError):
            load_instance(bad)

    def test_dedupe_counts_shared_sessions_once(self):
        a = load_instance(raw_instance("q1"))
        b = load_instance(raw_instance("q2"))
        uniq, collisions = unique_sessions([a, b])
        self.assertEqual(len(uniq), 2)
        self.assertEqual(collisions, [])

    def test_same_id_different_text_is_reported_not_silently_merged(self):
        a = load_instance(raw_instance("q1"))
        b = load_instance(raw_instance("q2", haystack_sessions=[
            [{"role": "user", "content": "DIFFERENT"}],
            [{"role": "user", "content": "earlier"}]]))
        _, collisions = unique_sessions([a, b])
        self.assertEqual(collisions, ["s2"])

    def test_stats_reports_reuse(self):
        a = load_instance(raw_instance("q1"))
        b = load_instance(raw_instance("q2"))
        s = stats([a, b])
        self.assertEqual((s["instances"], s["session_slots"],
                          s["unique_sessions"], s["reuse_factor"]), (2, 4, 2, 2.0))


if __name__ == "__main__":
    unittest.main(verbosity=1)

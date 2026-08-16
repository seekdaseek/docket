"""Offline tests for the write path: ids, schema, ingest, and the client rules
that came out of the six probes.

The stub server here does not simulate a graph. It records the exact request
bodies the client produces and replays recorded envelopes, because what needs
asserting is the SHAPE of what goes on the wire -- that is where every measured
constraint lives, and a simulator would only ever agree with itself.

The one exception is the interruption test, which needs a stub that fails at a
chosen write so the checkpoint's behaviour on resume can be observed.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from datetime import datetime, timezone

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from docket.dataset import Session, Turn, load_instance
from docket.hydra import (MAX_PARAM_CHARS, HydraClient, HydraError,
                          HydraUnsupported, cell_value, lit)
from docket.ids import IdCollision, IdRegistry, stable_id
from docket.ingest import Checkpoint, ingest, ingest_session
from docket.schema import (CHUNK_CHARS, Writer, chunk_text, join_text,
                           render_props, session_props, statement_key,
                           statement_props, text_props)

WRITE_ENVELOPE = {
    "query_id": "http-query-1", "columns": [], "rows": [],
    "read_epoch": None, "next_cursor": None,
    "bookmark": "sgk:1:64656661756c74:64656661756c74:63656c6c2d30:1",
}
COUNT_ZERO = {
    "query_id": "q", "columns": ["count(*)"],
    "rows": [[{"type": "integer", "value": 0}]],
    "read_epoch": 1, "next_cursor": None, "bookmark": None,
}
# Recorded verbatim: an id-MATCH against a node that was never written.
NULL_CELL_ROW = {
    "query_id": "q", "columns": ["n.kind"],
    "rows": [[{"type": "null"}]],
    "read_epoch": 1, "next_cursor": None, "bookmark": None,
}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def recording_opener(record, payload=None, fail_at=None):
    """Record every request body; optionally fail on the Nth write."""
    payload = payload or WRITE_ENVELOPE
    state = {"i": 0}

    def _open(req, timeout=None):
        i = state["i"]
        state["i"] += 1
        if fail_at is not None and i == fail_at:
            raise urllib.error.HTTPError(
                req.full_url, 500, "boom", {},
                io.BytesIO(b'{"error":{"message":"internal"}}'))
        record.append(json.loads(req.data.decode()))
        return FakeResponse(json.dumps(payload).encode())

    return _open


def client_for(record, payload=None, fail_at=None):
    return HydraClient(opener=recording_opener(record, payload, fail_at))


class TestParameterRules(unittest.TestCase):
    def test_parameters_key_is_named_parameters(self):
        rec = []
        client_for(rec).query("MATCH (n:X {k: $v}) RETURN count(*)", {"v": "a"})
        self.assertIn("parameters", rec[0])
        self.assertNotIn("params", rec[0])

    def test_no_parameters_key_when_none_are_passed(self):
        rec = []
        client_for(rec).query("MATCH (n:X) RETURN count(*)")
        self.assertNotIn("parameters", rec[0])

    def test_consistency_is_sent_when_set(self):
        rec = []
        c = HydraClient(opener=recording_opener(rec), consistency="causal")
        c.query("MATCH (n:X) RETURN count(*)")
        self.assertEqual(rec[0]["consistency"], "causal")

    def test_unknown_consistency_is_refused(self):
        with self.assertRaises(ValueError):
            HydraClient(consistency="eventual")

    def test_oversized_parameter_raises_before_the_request(self):
        rec = []
        c = client_for(rec)
        with self.assertRaises(HydraUnsupported):
            c.query("CREATE (a {id: 1, b: $b})-[:R]->(c {id: 2})",
                    {"b": "z" * (MAX_PARAM_CHARS + 1)})
        self.assertEqual(rec, [], "nothing should have been sent")

    def test_oversized_parameter_inside_a_list_is_caught_too(self):
        c = client_for([])
        with self.assertRaises(HydraUnsupported):
            c.query("X", {"rows": [{"body": "z" * (MAX_PARAM_CHARS + 1)}]})


class TestLiteralLimit(unittest.TestCase):
    def test_long_string_literal_is_refused_with_the_real_reason(self):
        with self.assertRaises(HydraUnsupported) as ctx:
            lit("z" * 300)
        self.assertIn("parameters", str(ctx.exception))

    def test_short_string_still_renders(self):
        self.assertEqual(lit("ok"), "'ok'")


class TestAbsence(unittest.TestCase):
    def test_null_cell_becomes_none_not_a_dict(self):
        self.assertIsNone(cell_value({"type": "null"}))

    def test_scalar_of_a_null_cell_row_is_the_default(self):
        """The absence trap, asserted on the recorded envelope.

        An id-MATCH on a node that does not exist returns ONE ROW whose cell is
        null. Anything reading truthiness off row count concludes the node is
        there.
        """
        c = client_for([], payload=NULL_CELL_ROW)
        result = c.query("MATCH (n {id: 1}) RETURN n.kind")
        self.assertEqual(len(result), 1)
        self.assertIsNone(result.scalar())
        self.assertEqual(result.scalar("missing"), "missing")

    def test_count_by_property_refuses_an_id_predicate(self):
        c = client_for([], payload=COUNT_ZERO)
        with self.assertRaises(HydraUnsupported):
            c.count_by_property("Session", "id", 5)

    def test_count_by_property_uses_a_non_id_predicate(self):
        rec = []
        c = client_for(rec, payload=COUNT_ZERO)
        self.assertEqual(c.count_by_property("Session", "sid", "s1"), 0)
        self.assertIn("MATCH (n:Session {sid: $_v}) RETURN count(*)",
                      rec[0]["query"])


class TestIds(unittest.TestCase):
    def test_stable_across_calls(self):
        self.assertEqual(stable_id("Session", "s1"), stable_id("Session", "s1"))

    def test_kind_is_part_of_the_key(self):
        self.assertNotEqual(stable_id("Session", "x"), stable_id("Statement", "x"))

    def test_width_is_respected(self):
        self.assertLess(stable_id("Session", "s1", bits=40), 1 << 40)

    def test_never_zero(self):
        for i in range(2000):
            self.assertNotEqual(stable_id("Session", f"s{i}", bits=32), 0)

    def test_registry_allows_the_same_key_twice(self):
        r = IdRegistry(bits=40)
        self.assertEqual(r.issue("Session", "s1"), r.issue("Session", "s1"))
        self.assertEqual(r.issued, 1)

    def test_registry_raises_on_a_real_collision(self):
        """Forced with a narrow id space, because the point is detection.

        A wide hash makes a collision unlikely. The registry makes an
        UNDETECTED one impossible, which is the stronger claim and the one
        worth testing.
        """
        r = IdRegistry(bits=32)
        seen = {}
        with self.assertRaises(IdCollision):
            for i in range(400000):
                key = f"s{i}"
                node_id = r.issue("Session", key)
                if node_id in seen and seen[node_id] != key:
                    self.fail("registry should have raised first")
                seen[node_id] = key


class TestTextChunking(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        props = text_props("hello")
        self.assertEqual(props["nchunks"], 1)
        self.assertEqual(props["t0"], "hello")

    def test_long_text_round_trips_exactly(self):
        text = "".join(chr(97 + i % 26) for i in range(30000))
        props = text_props(text)
        self.assertGreater(props["nchunks"], 1)
        self.assertEqual(join_text(props), text)

    def test_every_chunk_is_within_the_parameter_cap(self):
        for part in chunk_text("z" * 40000):
            self.assertLessEqual(len(part), MAX_PARAM_CHARS)

    def test_a_missing_chunk_raises_rather_than_returning_a_prefix(self):
        props = text_props("z" * 30000)
        del props["t1"]
        with self.assertRaises(ValueError):
            join_text(props)

    def test_no_nchunks_is_an_error_not_an_empty_string(self):
        with self.assertRaises(ValueError):
            join_text({"t0": "partial"})

    def test_empty_text_survives(self):
        self.assertEqual(join_text(text_props("")), "")


class TestPropertyRendering(unittest.TestCase):
    def test_strings_become_parameters_and_numbers_become_literals(self):
        text, params = render_props({"sid": "s1", "ts": 7}, "a")
        self.assertIn("sid: $a_sid", text)
        self.assertIn("ts: 7", text)
        self.assertEqual(params, {"a_sid": "s1"})

    def test_none_is_dropped(self):
        text, params = render_props({"a": None, "b": 1}, "a")
        self.assertNotIn("a:", text)
        self.assertEqual(params, {})

    def test_unsupported_type_raises(self):
        with self.assertRaises(HydraUnsupported):
            render_props({"a": ["x"]}, "a")

    def test_bad_property_name_is_refused(self):
        with self.assertRaises(ValueError):
            render_props({"drop table": 1}, "a")


class TestWriter(unittest.TestCase):
    def test_write_is_a_single_one_hop_create_with_integer_ids(self):
        rec = []
        w = Writer(client_for(rec), IdRegistry(bits=44))
        w.edge(src_label="Statement", src_key="s1|0", src_props={"role": "user"},
               rel="IN", dst_label="Session", dst_key="s1", dst_props={"ts": 5})
        q = rec[0]["query"]
        self.assertEqual(q.count("CREATE"), 1)
        self.assertIn("-[:IN]->", q)
        self.assertIn("(a:Statement {id: ", q)
        self.assertIn("(b:Session {id: ", q)

    def test_text_never_appears_in_the_query_string(self):
        rec = []
        w = Writer(client_for(rec), IdRegistry(bits=44))
        secret = "the quick brown fox " * 100
        w.edge(src_label="Statement", src_key="s1|0",
               src_props=statement_props(session_id="s1", index=0, role="user",
                                         ts=1, text=secret),
               rel="IN", dst_label="Session", dst_key="s1", dst_props={})
        self.assertNotIn("quick brown", rec[0]["query"])
        self.assertIn(secret, json.dumps(rec[0]["parameters"]))

    def test_natural_key_is_stored_so_absence_can_be_checked(self):
        rec = []
        w = Writer(client_for(rec), IdRegistry(bits=44))
        w.edge(src_label="Statement", src_key="s1|0", src_props={},
               rel="IN", dst_label="Session", dst_key="s1", dst_props={})
        self.assertEqual(rec[0]["parameters"]["a_nkey"], "s1|0")
        self.assertEqual(rec[0]["parameters"]["b_nkey"], "s1")

    def test_same_key_gives_the_same_id_across_writes(self):
        rec = []
        w = Writer(client_for(rec), IdRegistry(bits=44))
        for i in (0, 1):
            w.edge(src_label="Statement", src_key=f"s1|{i}", src_props={},
                   rel="IN", dst_label="Session", dst_key="s1", dst_props={})
        first = rec[0]["query"].split("(b:Session {id: ")[1].split(",")[0]
        second = rec[1]["query"].split("(b:Session {id: ")[1].split(",")[0]
        self.assertEqual(first, second)

    def test_exactly_once_deletes_before_it_creates(self):
        rec = []
        w = Writer(client_for(rec), IdRegistry(bits=44))
        w.exactly_once(src_label="Statement", src_key="s1|0", src_props={},
                       rel="IN", dst_label="Session", dst_key="s1", dst_props={})
        self.assertIn("DELETE r", rec[0]["query"])
        self.assertIn("CREATE", rec[1]["query"])

    def test_the_registry_it_is_handed_is_the_one_it_uses(self):
        """Regression: a fresh IdRegistry used to be discarded silently.

        IdRegistry defined __len__, so an empty one was falsy, and
        `registry or IdRegistry()` swapped a caller's 44-bit registry for a
        default 52-bit one. Ids stayed deterministic and plausible, so nothing
        looked wrong -- the whole graph would simply have been written in a
        different id space than the preflight measured as safe.
        """
        w = Writer(client_for([]), IdRegistry(bits=44))
        self.assertEqual(w.ids.bits, 44)
        self.assertLess(w.node_id("Statement", "s1|0"), 1 << 44)

    def test_injection_through_a_label_is_refused(self):
        w = Writer(client_for([]), IdRegistry(bits=44))
        with self.assertRaises(ValueError):
            w.edge(src_label="Statement) DELETE (x", src_key="k", src_props={},
                   rel="IN", dst_label="Session", dst_key="s", dst_props={})


class TestStatementProps(unittest.TestCase):
    def test_has_answer_never_becomes_a_node_property(self):
        """The ground-truth leak, closed by an allowlist rather than a comment.

        LongMemEval marks the turns that hold the evidence. If that flag
        reached the graph, retrieval could rank by the answer key.
        """
        turn = Turn(role="user", content="I got my car serviced", has_answer=True)
        props = statement_props(session_id="s1", index=0, role=turn.role,
                                ts=1, text=turn.content)
        self.assertNotIn("has_answer", props)
        for key, value in props.items():
            self.assertNotIsInstance(value, bool, f"{key} looks like a flag")

    def test_unexpected_role_raises(self):
        with self.assertRaises(ValueError):
            statement_props(session_id="s1", index=0, role="system", ts=1,
                            text="x")

    def test_statement_key_is_session_and_index(self):
        self.assertEqual(statement_key("s1", 3), "s1|3")


def make_session(sid="s1", n=3, when=None):
    return Session(
        session_id=sid,
        when=when or datetime(2023, 4, 10, 17, 50, tzinfo=timezone.utc),
        turns=[Turn(role="user" if i % 2 == 0 else "assistant",
                    content=f"turn {i}") for i in range(n)])


class TestIngestSession(unittest.TestCase):
    def test_session_properties_ride_the_first_statement_only(self):
        """There is no node-only CREATE, so the session is born on a write.

        Later statements name only its id, which merges. Restating its
        properties every time would be wasted bytes on 10,866 writes.
        """
        rec = []
        w = Writer(client_for(rec), IdRegistry(bits=44))
        ingest_session(w, make_session(n=3))
        self.assertIn("turns: 3", rec[0]["query"])
        self.assertNotIn("turns", rec[1]["query"])
        self.assertNotIn("turns", rec[2]["query"])

    def test_one_write_per_turn(self):
        rec = []
        w = Writer(client_for(rec), IdRegistry(bits=44))
        self.assertEqual(ingest_session(w, make_session(n=5)), 5)
        self.assertEqual(len(rec), 5)

    def test_every_statement_carries_its_session_timestamp(self):
        rec = []
        w = Writer(client_for(rec), IdRegistry(bits=44))
        ingest_session(w, make_session(n=2))
        for body in rec:
            self.assertIn("ts: ", body["query"])


class TestCheckpointAndResume(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "ingest.jsonl")

    def test_completed_sessions_are_skipped_on_the_second_run(self):
        sessions = {"s1": make_session("s1", 3)}
        rec = []
        first = ingest(client_for(rec), sessions, checkpoint_path=self.path,
                       id_bits=44)
        self.assertEqual(first["statements_written"], 3)
        rec2 = []
        second = ingest(client_for(rec2), sessions, checkpoint_path=self.path,
                        id_bits=44)
        self.assertEqual(second["statements_written"], 0)
        self.assertEqual(second["sessions_skipped_already_done"], 1)
        self.assertEqual(rec2, [], "a completed session must not be rewritten")

    def test_an_interrupted_session_is_rewritten_with_edges_deleted_first(self):
        """Edges duplicate on repeat, so a resumed session must clear its own.

        Node properties need no such care -- a repeat write to the same id
        merges -- but a doubled IN edge would inflate every per-session count
        for the rest of the project.
        """
        sessions = {"s1": make_session("s1", 3)}
        rec = []
        with self.assertRaises(HydraError):
            ingest(client_for(rec, fail_at=1), sessions,
                   checkpoint_path=self.path, id_bits=44)
        ckpt = Checkpoint(self.path)
        self.assertEqual(ckpt.interrupted, {"s1"})

        rec2 = []
        summary = ingest(client_for(rec2), sessions, checkpoint_path=self.path,
                         id_bits=44)
        self.assertEqual(summary["statements_written"], 3)
        self.assertEqual(summary["deletes"], 3)
        self.assertIn("DELETE r", rec2[0]["query"])

    def test_a_half_written_checkpoint_line_does_not_break_resume(self):
        with open(self.path, "w") as fh:
            fh.write(json.dumps({"event": "done", "sid": "s1"}) + "\n")
            fh.write('{"event": "start", "sid": "s2"')  # killed mid-write
        ckpt = Checkpoint(self.path)
        self.assertEqual(ckpt.done, {"s1"})

    def test_empty_sessions_are_reported_not_silently_absent(self):
        """A session with no turns cannot exist: there is no node-only CREATE.

        Reported by id so the count is explainable, rather than leaving a hole
        in the graph that looks like a bug later.
        """
        sessions = {"s1": make_session("s1", 0), "s2": make_session("s2", 2)}
        summary = ingest(client_for([]), sessions, checkpoint_path=self.path,
                         id_bits=44)
        self.assertEqual(summary["sessions_empty"], ["s1"])
        self.assertEqual(summary["statements_written"], 2)

    def test_sessions_are_written_oldest_first(self):
        late = make_session("late", 1,
                            datetime(2023, 5, 1, tzinfo=timezone.utc))
        early = make_session("early", 1,
                             datetime(2023, 1, 1, tzinfo=timezone.utc))
        rec = []
        ingest(client_for(rec), {"late": late, "early": early},
               checkpoint_path=self.path, id_bits=44)
        self.assertEqual(rec[0]["parameters"]["b_nkey"], "early")


class TestEndToEndShape(unittest.TestCase):
    def test_a_real_instance_becomes_writes_with_no_ground_truth_in_them(self):
        raw = {
            "question_id": "gpt4_2655b836",
            "question_type": "temporal-reasoning",
            "question": "What was the first issue?",
            "answer": "GPS system not functioning correctly",
            "question_date": "2023/04/10 (Mon) 23:07",
            "haystack_dates": ["2023/04/10 (Mon) 17:50"],
            "haystack_session_ids": ["sess_a"],
            "haystack_sessions": [[
                {"role": "user", "content": "serviced on March 15th",
                 "has_answer": True},
                {"role": "assistant", "content": "noted", "has_answer": False},
            ]],
            "answer_session_ids": ["sess_a"],
        }
        instance = load_instance(raw)
        self.assertTrue(instance.sessions[0].turns[0].has_answer)
        rec = []
        ingest(client_for(rec), {"sess_a": instance.sessions[0]},
               checkpoint_path=os.path.join(tempfile.mkdtemp(), "c.jsonl"),
               id_bits=44)
        blob = json.dumps(rec)
        self.assertNotIn("has_answer", blob)
        self.assertNotIn("GPS system", blob, "the answer must not be in memory")
        self.assertNotIn("answer_session_ids", blob)
        self.assertIn("serviced on March 15th", blob)


if __name__ == "__main__":
    unittest.main()

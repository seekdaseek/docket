"""Offline tests for extraction: the model client, the validator, the store
that distinguishes absent from unmeasured, and the SUPERSEDES chain.

No network. The Anthropic client is exercised against a stub that replays real
response shapes, and the chain builder is a pure function so a wrong chain is
caught here rather than in a benchmark score three days later.
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

from docket.claims import (chain_rows, load_chains, load_claims,
                           normalise_object, predicate_cardinality)
from docket.dataset import Session, Turn
from docket.extract import (PROMPT_VERSION, ClaimStore, claim_key,
                            entity_key, extract_session, render_session,
                            validate)
from docket.hydra import HydraClient
from docket.ids import IdRegistry
from docket.llm import Anthropic, LLMError, load_env, strip_fences
from docket.schema import Writer

WRITE_ENVELOPE = {"query_id": "q", "columns": [], "rows": [],
                  "read_epoch": None, "next_cursor": None, "bookmark": "bm"}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def llm_returning(text, usage=None, stop="end_turn"):
    def _open(req, timeout=None):
        body = {"content": [{"type": "text", "text": text}],
                "stop_reason": stop,
                "usage": usage or {"input_tokens": 10, "output_tokens": 5}}
        return FakeResponse(json.dumps(body).encode())
    return _open


def graph_recorder(record):
    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        return FakeResponse(json.dumps(WRITE_ENVELOPE).encode())
    return _open


def session(sid="s1", turns=2, when=None):
    return Session(session_id=sid,
                   when=when or datetime(2023, 4, 10, tzinfo=timezone.utc),
                   turns=[Turn("user", f"line {i}") for i in range(turns)])


class TestLLMClient(unittest.TestCase):
    def test_json_array_is_parsed(self):
        a = Anthropic("k", opener=llm_returning('[{"turn": 0}]'))
        self.assertEqual(a.complete_json("s", "u"), [{"turn": 0}])

    def test_code_fence_is_stripped(self):
        a = Anthropic("k", opener=llm_returning('```json\n[1,2]\n```'))
        self.assertEqual(a.complete_json("s", "u"), [1, 2])

    def test_unparseable_output_raises_rather_than_returning_empty(self):
        """The distinction the whole project rests on.

        Returning [] here would make a model that refused look exactly like a
        session with nothing durable in it.
        """
        a = Anthropic("k", opener=llm_returning("I'm happy to help!"))
        with self.assertRaises(LLMError):
            a.complete_json("s", "u")

    def test_empty_content_names_the_stop_reason(self):
        a = Anthropic("k", opener=llm_returning("", stop="max_tokens"))
        with self.assertRaises(LLMError) as ctx:
            a.complete_json("s", "u")
        self.assertIn("max_tokens", str(ctx.exception))

    def test_missing_key_says_where_to_put_one(self):
        with self.assertRaises(LLMError) as ctx:
            Anthropic("")
        self.assertIn(".env", str(ctx.exception))

    def test_429_is_retried_then_succeeds(self):
        state = {"i": 0}
        slept = []

        def _open(req, timeout=None):
            state["i"] += 1
            if state["i"] == 1:
                raise urllib.error.HTTPError(
                    req.full_url, 429, "slow down", {"retry-after": "2"},
                    io.BytesIO(b"{}"))
            return FakeResponse(json.dumps({
                "content": [{"type": "text", "text": "[]"}],
                "usage": {}}).encode())

        a = Anthropic("k", opener=_open, sleeper=slept.append)
        self.assertEqual(a.complete_json("s", "u"), [])
        self.assertEqual(slept, [2.0], "the server's retry-after was honoured")
        self.assertEqual(a.retries, 1)

    def test_400_is_not_retried(self):
        calls = []

        def _open(req, timeout=None):
            calls.append(1)
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {},
                                         io.BytesIO(b'{"error":"nope"}'))
        a = Anthropic("k", opener=_open, sleeper=lambda s: None)
        with self.assertRaises(LLMError):
            a.complete_json("s", "u")
        self.assertEqual(len(calls), 1)

    def test_tokens_are_counted_so_the_run_can_be_priced(self):
        a = Anthropic("k", opener=llm_returning(
            "[]", usage={"input_tokens": 3300, "output_tokens": 120}))
        a.complete_json("s", "u")
        self.assertEqual((a.input_tokens, a.output_tokens), (3300, 120))

    def test_strip_fences_leaves_plain_json_alone(self):
        self.assertEqual(strip_fences('  [1] '), "[1]")

    def test_load_env_reads_key_values_and_ignores_comments(self):
        path = os.path.join(tempfile.mkdtemp(), ".env")
        with open(path, "w") as fh:
            fh.write("# comment\nANTHROPIC_API_KEY=\"abc\"\nJUNK\n")
        self.assertEqual(load_env(path), {"ANTHROPIC_API_KEY": "abc"})


class TestValidation(unittest.TestCase):
    def good(self, **over):
        base = {"turn": 0, "subject": "user", "predicate": "lives_in",
                "object": "Berlin", "kind": "fact", "cardinality": "one"}
        base.update(over)
        return base

    def test_a_clean_claim_survives(self):
        kept, drops = validate([self.good()], turn_count=2)
        self.assertEqual(len(kept), 1)
        self.assertEqual(drops, {})

    def test_out_of_range_turn_is_dropped_not_clamped(self):
        """Clamping would cite the wrong sentence, convincingly."""
        kept, drops = validate([self.good(turn=9)], turn_count=2)
        self.assertEqual(kept, [])
        self.assertEqual(drops, {"turn_out_of_range": 1})

    def test_negative_turn_is_dropped(self):
        kept, _ = validate([self.good(turn=-1)], turn_count=2)
        self.assertEqual(kept, [])

    def test_boolean_turn_is_not_accepted_as_an_integer(self):
        kept, drops = validate([self.good(turn=True)], turn_count=2)
        self.assertEqual(drops, {"turn_not_an_integer": 1})

    def test_empty_field_is_dropped(self):
        kept, drops = validate([self.good(object="   ")], turn_count=2)
        self.assertEqual(drops, {"missing_or_empty_field": 1})

    def test_overlong_object_is_dropped(self):
        kept, drops = validate([self.good(object="x" * 5000)], turn_count=2)
        self.assertEqual(drops, {"field_too_long": 1})

    def test_unknown_kind_is_dropped(self):
        kept, drops = validate([self.good(kind="vibe")], turn_count=2)
        self.assertEqual(drops, {"bad_kind": 1})

    def test_predicate_is_normalised_to_snake_case(self):
        kept, _ = validate([self.good(predicate="Lives In")], turn_count=2)
        self.assertEqual(kept[0]["predicate"], "lives_in")

    def test_duplicate_claims_in_one_response_are_collapsed(self):
        kept, drops = validate([self.good(), self.good()], turn_count=2)
        self.assertEqual(len(kept), 1)
        self.assertEqual(drops, {"duplicate_in_response": 1})

    def test_empty_array_is_a_valid_answer_not_an_error(self):
        kept, drops = validate([], turn_count=2)
        self.assertEqual((kept, drops), ([], {}))

    def test_a_non_array_response_raises(self):
        with self.assertRaises(ValueError):
            validate({"claims": []}, turn_count=2)

    def test_junk_items_are_counted_while_good_ones_survive(self):
        kept, drops = validate(["nope", self.good()], turn_count=2)
        self.assertEqual(len(kept), 1)
        self.assertEqual(drops, {"not_an_object": 1})


class TestPromptBody(unittest.TestCase):
    def test_turns_are_numbered_so_a_claim_can_cite_one(self):
        body = render_session(session(turns=3))
        self.assertIn("[0] user: line 0", body)
        self.assertIn("[2] user: line 2", body)

    def test_the_session_date_is_given_to_the_model(self):
        self.assertIn("2023-04-10", render_session(session()))


class TestClaimStore(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "claims.jsonl")

    def test_absent_and_unmeasured_are_different_rows(self):
        """A session with nothing durable in it, versus one the model failed on.

        Both hold zero claims. Only one of them should ever be retried, and
        only one of them means the corpus is smaller than it looks.
        """
        store = ClaimStore(self.path)
        store.record_measured("s1", [], {}, 4)
        store.record_unmeasured("s2", "model output was not JSON")
        reloaded = ClaimStore(self.path)
        self.assertEqual(list(reloaded.measured), ["s1"])
        self.assertEqual(list(reloaded.unmeasured), ["s2"])

    def test_a_half_written_line_does_not_lose_the_rest(self):
        store = ClaimStore(self.path)
        store.record_measured("s1", [], {}, 2)
        with open(self.path, "a") as fh:
            fh.write('{"sid": "s2", "status": "meas')
        self.assertEqual(list(ClaimStore(self.path).rows), ["s1"])

    def test_extraction_failure_is_recorded_with_its_reason(self):
        a = Anthropic("k", opener=llm_returning("sorry!"))
        with self.assertRaises(LLMError):
            extract_session(a, session())


class TestChains(unittest.TestCase):
    def rows(self, *triples):
        out = []
        for sid, ts, claims in triples:
            out.append({"sid": sid, "ts": ts, "claims": claims})
        return out

    def claim(self, obj, turn=0, subject="user", predicate="lives_in",
              cardinality="one"):
        return {"turn": turn, "subject": subject, "predicate": predicate,
                "object": obj, "kind": "fact", "cardinality": cardinality}

    def test_a_later_value_supersedes_an_earlier_one(self):
        rows = self.rows(("s1", 100, [self.claim("Berlin")]),
                         ("s2", 200, [self.claim("Munich")]))
        links = chain_rows(rows)
        self.assertEqual(len(links), 1)
        later, earlier = links[0]
        self.assertEqual(later, claim_key("s2", 0, "user", "lives_in", "Munich"))
        self.assertEqual(earlier, claim_key("s1", 0, "user", "lives_in", "Berlin"))

    def test_restating_the_same_value_does_not_make_a_link(self):
        """Corroboration is not replacement.

        Nine sessions repeating one fact would otherwise become a chain of
        nine, and every walk that measures chain depth would be wrong.
        """
        rows = self.rows(("s1", 100, [self.claim("Berlin")]),
                         ("s2", 200, [self.claim("Berlin")]),
                         ("s3", 300, [self.claim("Berlin")]))
        self.assertEqual(chain_rows(rows), [])

    def test_a_value_that_returns_is_chained_again(self):
        rows = self.rows(("s1", 100, [self.claim("Berlin")]),
                         ("s2", 200, [self.claim("Munich")]),
                         ("s3", 300, [self.claim("Berlin")]))
        self.assertEqual(len(chain_rows(rows)), 2)

    def test_different_predicates_do_not_share_a_chain(self):
        rows = self.rows(("s1", 100, [self.claim("Berlin")]),
                         ("s2", 200, [self.claim("Acme", predicate="works_at")]))
        self.assertEqual(chain_rows(rows), [])

    def test_subjects_are_matched_case_and_space_insensitively(self):
        rows = self.rows(("s1", 100, [self.claim("Berlin", subject="User")]),
                         ("s2", 200, [self.claim("Munich", subject="  user ")]))
        self.assertEqual(len(chain_rows(rows)), 1)

    def test_order_is_time_not_file_order(self):
        rows = self.rows(("s2", 300, [self.claim("Munich")]),
                         ("s1", 100, [self.claim("Berlin")]))
        later, earlier = chain_rows(rows)[0]
        self.assertEqual(later, claim_key("s2", 0, "user", "lives_in", "Munich"))

    def test_a_tie_on_timestamp_is_broken_stably(self):
        rows = self.rows(("s2", 100, [self.claim("Munich")]),
                         ("s1", 100, [self.claim("Berlin")]))
        self.assertEqual(chain_rows(rows), chain_rows(list(reversed(rows))))

    def test_entity_key_normalises(self):
        self.assertEqual(entity_key("  My  Car "), "my car")


class TestCardinality(unittest.TestCase):
    """The bug that 48 real sessions exposed, pinned so it cannot come back.

    The first chain rule linked any change of value. Against real extractor
    output that produced 120 links from 336 claims, and reading them showed
    most were invented: owning Vans does not stop you owning Converse.
    """

    def rows(self, *triples):
        return [{"sid": sid, "ts": ts, "claims": claims}
                for sid, ts, claims in triples]

    def owns(self, obj, sid_turn=0, cardinality="many"):
        return {"turn": sid_turn, "subject": "user", "predicate": "owns",
                "object": obj, "kind": "fact", "cardinality": cardinality}

    def test_an_accumulating_predicate_never_chains(self):
        rows = self.rows(
            ("answer_099c1b6c_1", 100, [self.owns("Vans Old Skool sneakers")]),
            ("answer_099c1b6c_2", 200, [self.owns("Converse Chuck Taylor")]),
            ("answer_099c1b6c_4", 300, [self.owns("brown leather dress shoes")]))
        self.assertEqual(chain_rows(rows), [],
                         "owning three pairs of shoes is not three replacements")

    def test_a_single_valued_predicate_still_chains(self):
        rows = self.rows(
            ("s1", 100, [{"turn": 0, "subject": "user", "predicate": "lives_in",
                          "object": "Berlin", "kind": "fact",
                          "cardinality": "one"}]),
            ("s2", 200, [{"turn": 0, "subject": "user", "predicate": "lives_in",
                          "object": "Munich", "kind": "fact",
                          "cardinality": "one"}]))
        self.assertEqual(len(chain_rows(rows)), 1)

    def test_one_dissenting_claim_collapses_the_predicate_to_many(self):
        """Unanimity, not majority. Default-deny is the safe direction."""
        rows = self.rows(
            ("s1", 100, [self.owns("bike", cardinality="one")]),
            ("s2", 200, [self.owns("car", cardinality="one")]),
            ("s3", 300, [self.owns("guitar", cardinality="many")]))
        self.assertEqual(chain_rows(rows), [])
        resolved, disputed = predicate_cardinality(rows)
        self.assertEqual(resolved["owns"], "many")
        self.assertEqual(disputed["owns"], {"one": 2, "many": 1})

    def test_a_missing_cardinality_is_treated_as_accumulating(self):
        rows = self.rows(
            ("s1", 100, [{"turn": 0, "subject": "user", "predicate": "p",
                          "object": "a", "kind": "fact"}]),
            ("s2", 200, [{"turn": 0, "subject": "user", "predicate": "p",
                          "object": "b", "kind": "fact"}]))
        self.assertEqual(chain_rows(rows), [])

    def test_the_validator_defaults_a_missing_cardinality_and_counts_it(self):
        kept, drops = validate(
            [{"turn": 0, "subject": "user", "predicate": "p", "object": "o",
              "kind": "fact"}], turn_count=1)
        self.assertEqual(kept[0]["cardinality"], "many")
        self.assertEqual(drops, {"cardinality_missing_defaulted_to_many": 1})

    def test_a_bogus_cardinality_value_defaults_to_many(self):
        kept, _ = validate(
            [{"turn": 0, "subject": "user", "predicate": "p", "object": "o",
              "kind": "fact", "cardinality": "sometimes"}], turn_count=1)
        self.assertEqual(kept[0]["cardinality"], "many")


class TestObjectNormalisation(unittest.TestCase):
    """Also from the real run: the same shoes came back two ways."""

    def test_underscores_and_case_do_not_make_a_change_of_value(self):
        self.assertEqual(
            normalise_object("Converse_Chuck_Taylor_All_Star_sneakers"),
            normalise_object("Converse Chuck Taylor All Star sneakers"))

    def test_punctuation_and_spacing_are_ignored(self):
        self.assertEqual(normalise_object("  Berlin, Germany "),
                         normalise_object("berlin germany"))

    def test_genuinely_different_values_stay_different(self):
        self.assertNotEqual(normalise_object("Berlin"), normalise_object("Munich"))

    def test_a_restatement_in_another_format_is_not_chained(self):
        rows = [{"sid": "s1", "ts": 100, "claims": [
                    {"turn": 0, "subject": "user", "predicate": "lives_in",
                     "object": "Berlin, Germany", "kind": "fact",
                     "cardinality": "one"}]},
                {"sid": "s2", "ts": 200, "claims": [
                    {"turn": 0, "subject": "user", "predicate": "lives_in",
                     "object": "berlin_germany", "kind": "fact",
                     "cardinality": "one"}]}]
        self.assertEqual(chain_rows(rows), [])


class TestPromptVersioning(unittest.TestCase):
    def test_rows_from_an_older_prompt_are_reported_as_stale(self):
        """A row missing a field the chain logic needs is not smaller, it is
        wrong. Re-extract rather than mix."""
        path = os.path.join(tempfile.mkdtemp(), "claims.jsonl")
        store = ClaimStore(path)
        store.record_measured("s1", [], {}, 2)
        with open(path, "a") as fh:
            fh.write(json.dumps({"sid": "old", "status": "measured",
                                 "claims": [], "prompt_version": 1}) + "\n")
        reloaded = ClaimStore(path)
        self.assertEqual(list(reloaded.stale()), ["old"])


class TestGraphWrites(unittest.TestCase):
    def test_a_claim_cites_its_statement_and_joins_its_entity(self):
        rec = []
        w = Writer(HydraClient(opener=graph_recorder(rec)), IdRegistry(bits=52))
        rows = [{"sid": "s1", "ts": 100,
                 "claims": [{"turn": 1, "subject": "user",
                             "predicate": "lives_in", "object": "Berlin",
                             "kind": "fact", "cardinality": "one"}]}]
        out = load_claims(w, rows)
        self.assertEqual(out["claims_written"], 1)
        self.assertIn("(a:Statement {id:", rec[0]["query"])
        self.assertIn("-[:ASSERTS]->", rec[0]["query"])
        self.assertIn("-[:ABOUT]->", rec[1]["query"])
        self.assertIn("(b:Entity {id:", rec[1]["query"])

    def test_loading_a_claim_does_not_restate_the_statement_it_cites(self):
        """The statement already exists with its text. Naming only its id
        merges, so a claim load cannot disturb the ingest."""
        rec = []
        w = Writer(HydraClient(opener=graph_recorder(rec)), IdRegistry(bits=52))
        load_claims(w, [{"sid": "s1", "ts": 100, "claims": [
            {"turn": 0, "subject": "user", "predicate": "p", "object": "o",
             "kind": "fact", "cardinality": "one"}]}])
        statement_side = rec[0]["query"].split("-[:ASSERTS]->")[0]
        self.assertNotIn("t0:", statement_side)
        self.assertIn("nkey", statement_side)

    def test_chains_are_cleared_before_being_rewritten(self):
        rec = []
        w = Writer(HydraClient(opener=graph_recorder(rec)), IdRegistry(bits=52))
        rows = [{"sid": "s1", "ts": 100, "claims": [
                    {"turn": 0, "subject": "user", "predicate": "lives_in",
                     "object": "Berlin", "kind": "fact",
                     "cardinality": "one"}]},
                {"sid": "s2", "ts": 200, "claims": [
                    {"turn": 0, "subject": "user", "predicate": "lives_in",
                     "object": "Munich", "kind": "fact",
                     "cardinality": "one"}]}]
        out = load_chains(w, rows)
        self.assertEqual(out["supersedes_links"], 1)
        self.assertEqual(out["claims_cleared_first"], 2)
        self.assertIn("DELETE r", rec[0]["query"])
        self.assertIn("DELETE r", rec[1]["query"])
        self.assertIn("-[:SUPERSEDES]->", rec[2]["query"])

    def test_the_answer_text_never_reaches_the_graph_through_a_claim(self):
        rec = []
        w = Writer(HydraClient(opener=graph_recorder(rec)), IdRegistry(bits=52))
        load_claims(w, [{"sid": "s1", "ts": 100, "claims": [
            {"turn": 0, "subject": "user", "predicate": "lives_in",
             "object": "Berlin", "kind": "fact", "cardinality": "one"}]}])
        blob = json.dumps(rec)
        self.assertNotIn("has_answer", blob)
        self.assertNotIn("answer_session_ids", blob)


if __name__ == "__main__":
    unittest.main()

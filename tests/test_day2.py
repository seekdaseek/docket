"""Day 2 offline suite. No network, no HydraDB, no model.

The wire-level assertions matter more than the return values here: this layer's
whole job is to emit Cypher the measured surface actually accepts, so the tests
read the request bodies and check the shapes rather than trusting a stub to
answer plausibly.
"""
from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docket.answer import ABSENT, ANSWERED, UNMEASURED, Answerer, render_evidence
from docket.gate import MIN_STEM, LexicalGate, build_gate, stem, tokenise
from docket.hydra import HydraClient, HydraError
from docket.ids import stable_id
from docket.retrieve import CLAIM_PROPS, Retriever


# --------------------------------------------------------------------------
# stub transport, same shape as the day1 suite
# --------------------------------------------------------------------------
class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def scripted_opener(record, replies):
    """Answer each request from a list of payloads; record every body sent."""
    state = {"i": 0}

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        i = state["i"]
        state["i"] += 1
        payload = replies[i] if i < len(replies) else {"columns": [], "rows": []}
        if isinstance(payload, Exception):
            raise payload
        return FakeResponse(json.dumps(payload).encode())

    return _open


def client_for(record, replies):
    return HydraClient(opener=scripted_opener(record, replies))


def cells(*values):
    out = []
    for v in values:
        if v is None:
            out.append({"type": "null"})
        elif isinstance(v, bool):
            out.append({"type": "boolean", "value": v})
        elif isinstance(v, int):
            out.append({"type": "integer", "value": v})
        else:
            out.append({"type": "string", "value": v})
    return out


def claim_reply(alias="c", **over):
    claim = {"nkey": "s1|3|user|lives_in|abcd1234", "subj": "user",
             "pred": "lives_in", "obj": "Tokyo", "kind": "fact", "card": "one",
             "sid": "s1", "turn": 3, "ts": 1_700_000_000}
    claim.update(over)
    return {"columns": [f"{alias}.{p}" for p in CLAIM_PROPS],
            "rows": [cells(*[claim[p] for p in CLAIM_PROPS])]}


CLAIMS = {
    "k_tokyo": {"subj": "user", "pred": "lives_in", "obj": "Tokyo",
                "kind": "fact", "card": "one", "sid": "s3", "turn": 1,
                "ts": 1_700_002_000},
    "k_chicago": {"subj": "user", "pred": "lives_in", "obj": "Chicago",
                  "kind": "fact", "card": "one", "sid": "s1", "turn": 2,
                  "ts": 1_700_000_000},
    "k_shoes": {"subj": "user", "pred": "owns", "obj": "Brooks running shoes",
                "kind": "fact", "card": "many", "sid": "s2", "turn": 5,
                "ts": 1_700_001_000},
}


# --------------------------------------------------------------------------
class Tokenising(unittest.TestCase):
    def test_snake_case_predicates_split_into_their_words(self):
        self.assertEqual(tokenise("planning_hiking_trip", stemming=False),
                         ["planning", "hiking", "trip"])

    def test_stemming_makes_the_tenses_of_a_plan_meet(self):
        forms = {stem(w) for w in ("plans", "planning", "planned")}
        self.assertEqual(len(forms), 1, f"expected one stem, got {forms}")

    def test_the_stem_floor_is_a_known_limitation_and_stays_pinned(self):
        # MIN_STEM=3 for EVERY suffix was measured and REJECTED: `shoes` then
        # stems to `sho` via `es`, and `lives_in` fell out of the top three for
        # "where do I live" because collapsed terms gain df and lose idf.
        self.assertEqual(MIN_STEM, 4)
        self.assertNotEqual(stem("buying"), stem("buy"))
        self.assertEqual(stem("shoes"), "shoe")

    def test_the_plain_plural_is_allowed_one_character_shorter(self):
        # A floor of 4 meant three-letter nouns never took a plural, so a
        # question about "model kits" could not match a claim about a "kit".
        for singular in ("kit", "job", "car", "bag", "app"):
            self.assertEqual(stem(singular + "s"), singular)
        # and the rejected case stays rejected: `es` still stops at 4
        self.assertEqual(stem("shoes"), "shoe")
        self.assertEqual(stem("passes"), "pass")
        # two-letter stems are still left alone
        self.assertEqual(stem("gas"), "gas")
        self.assertEqual(stem("bus"), "bus")

    def test_stemming_never_shortens_below_the_floor(self):
        for word in ("is", "was", "sees", "ties"):
            self.assertGreaterEqual(len(stem(word)), min(len(word), 4) - 1)

    def test_question_words_are_dropped_on_both_sides(self):
        # `did` was scoring 7.6 against did_system_update_on before the
        # stoplist covered interrogatives: rare in the corpus, so high idf,
        # and meaningless in the question.
        for word in ("what", "when", "where", "did", "do", "how", "type"):
            self.assertEqual(tokenise(word), [], f"{word} survived the stoplist")

    def test_a_question_of_only_stopwords_yields_no_search(self):
        self.assertEqual(LexicalGate().search("what did I do?"), [])


class Gate(unittest.TestCase):
    def setUp(self):
        self.gate = build_gate(CLAIMS)

    def test_it_finds_the_claim_a_question_is_about(self):
        hits = self.gate.search("Where do I live?", limit=3)
        self.assertTrue(hits)
        self.assertTrue(all(h.claim["pred"] == "lives_in" for h in hits))

    def test_every_hit_can_show_which_terms_earned_it(self):
        hit = self.gate.search("what shoes do I own?", limit=1)[0]
        self.assertEqual(hit.claim["obj"], "Brooks running shoes")
        self.assertIn("shoe", hit.terms)
        self.assertTrue(all(v > 0 for v in hit.terms.values()))
        # ordered by contribution so the top reason is first
        self.assertEqual(list(hit.terms), sorted(hit.terms, key=lambda t: -hit.terms[t]))

    def test_scores_are_deterministic_and_ties_break_by_key(self):
        a = [(h.key, h.score) for h in self.gate.search("live", limit=5)]
        b = [(h.key, h.score) for h in self.gate.search("live", limit=5)]
        self.assertEqual(a, b)

    def test_readding_a_key_does_not_double_count_its_terms(self):
        gate = LexicalGate()
        gate.add("x", CLAIMS["k_tokyo"])
        gate.add("x", CLAIMS["k_tokyo"])
        gate.finalise()
        self.assertEqual(gate.size, 1)
        self.assertEqual(gate.df["tokyo"], 1)

    def test_an_unmatchable_question_returns_nothing_rather_than_the_least_bad(self):
        self.assertEqual(self.gate.search("quantum chromodynamics"), [])

    def test_session_mode_returns_every_claim_in_the_best_sessions(self):
        # The aggregation shape: the answer to "how many" is spread across
        # claims, some of which do not match the question at all.
        claims = {
            "a1": {"subj": "user", "pred": "built", "obj": "Spitfire model kit",
                   "kind": "event", "sid": "S", "turn": 1, "ts": 1},
            "a2": {"subj": "user", "pred": "bought", "obj": "Tiger I tank kit",
                   "kind": "event", "sid": "S", "turn": 2, "ts": 2},
            "a3": {"subj": "user", "pred": "prefers", "obj": "black coffee",
                   "kind": "preference", "sid": "S", "turn": 3, "ts": 3},
            "b1": {"subj": "user", "pred": "walked", "obj": "the dog",
                   "kind": "event", "sid": "T", "turn": 1, "ts": 4},
        }
        gate = build_gate(claims)
        hits = gate.search_sessions("how many model kits do I have", sessions=1)
        self.assertEqual({h.claim["sid"] for h in hits}, {"S"})
        # the non-matching claim comes too: it is one of the things being counted
        self.assertEqual(len(hits), 3)
        self.assertEqual([h.key for h in hits if h.score == 0], ["a3"])
        self.assertGreater([h for h in hits if h.key == "a2"][0].score, 0)

    def test_session_mode_returns_nothing_when_nothing_matches(self):
        gate = build_gate(CLAIMS)
        self.assertEqual(gate.search_sessions("quantum chromodynamics"), [])

    def test_a_session_of_several_weak_matches_beats_one_strong_match(self):
        claims = {
            "m1": {"subj": "user", "pred": "attended", "obj": "wedding",
                   "kind": "event", "sid": "MANY", "turn": 1, "ts": 1},
            "m2": {"subj": "user", "pred": "attended", "obj": "wedding",
                   "kind": "event", "sid": "MANY", "turn": 2, "ts": 2},
            "m3": {"subj": "user", "pred": "attended", "obj": "wedding",
                   "kind": "event", "sid": "MANY", "turn": 3, "ts": 3},
            "o1": {"subj": "user", "pred": "attended", "obj": "wedding",
                   "kind": "event", "sid": "ONE", "turn": 1, "ts": 4},
        }
        gate = build_gate(claims)
        hits = gate.search_sessions("how many weddings did I attend", sessions=1)
        self.assertEqual({h.claim["sid"] for h in hits}, {"MANY"})

    def test_the_embedding_gate_refuses_instead_of_degrading(self):
        from docket.gate import EmbeddingGate
        with self.assertRaises(NotImplementedError):
            EmbeddingGate()


class Reads(unittest.TestCase):
    def test_all_claims_uses_a_label_only_match(self):
        record = []
        r = Retriever(client_for(record, [claim_reply()]))
        got = r.all_claims()
        self.assertIn("MATCH (c:Claim) RETURN", record[0]["query"])
        self.assertEqual(list(got)[0], "s1|3|user|lives_in|abcd1234")

    def test_a_missing_claim_reads_as_none_not_as_a_null_row(self):
        # The absence trap: an id-MATCH on a node that was never written
        # returns one row of nulls. Same defence applied to nkey lookups.
        record = []
        nulls = {"columns": [f"c.{p}" for p in CLAIM_PROPS],
                 "rows": [cells(*[None] * len(CLAIM_PROPS))]}
        r = Retriever(client_for(record, [nulls]))
        self.assertIsNone(r.claim("nope"))

    def test_tip_orders_in_the_database_and_bounds_by_the_question_date(self):
        record = []
        r = Retriever(client_for(record, [claim_reply()]))
        r.tip("lives_in", asked_at=1_700_005_000, limit=3)
        q = record[0]["query"]
        self.assertIn("WHERE c.ts <= 1700005000", q)
        self.assertIn("ORDER BY c.ts DESC", q)
        self.assertIn("LIMIT 3", q)
        self.assertEqual(record[0]["parameters"], {"p": "lives_in"})

    def test_tip_with_a_subject_uses_two_inline_properties(self):
        record = []
        r = Retriever(client_for(record, [claim_reply()]))
        r.tip("lives_in", 1_700_005_000, subject="user")
        self.assertIn("{pred: $p, subj: $s}", record[0]["query"])
        self.assertEqual(record[0]["parameters"]["s"], "user")

    def test_the_chain_walk_starts_from_a_fixed_id_not_a_property(self):
        # MEASURED Aug 16: matching the source of a variable-length pattern by
        # an inline property returns "variable-length MATCH requires a fixed
        # source id". The id is derivable, so there is no cost to obeying it.
        record = []
        r = Retriever(client_for(record, [claim_reply(alias="b")]))
        r.superseded_by("k", hops=4)
        q = record[0]["query"]
        self.assertIn(f"(a:Claim {{id: {stable_id('Claim', 'k', 52)}}})", q)
        self.assertNotIn("{nkey:", q)   # no inline property on the source
        self.assertNotIn("$k", q)       # and nothing left to parameterise
        self.assertIn("[:SUPERSEDES*1..4]", q)
        self.assertNotIn("*1..]", q)
        self.assertNotIn("[:SUPERSEDES*]", q)

    def test_the_chain_is_ordered_in_python_not_in_cypher(self):
        # ORDER BY on a variable-length pattern is not in the measured surface.
        record = []
        cols = [f"b.{p}" for p in CLAIM_PROPS]
        rows = {"columns": cols, "rows": [
            cells("old", "user", "lives_in", "Boston", "fact", "one", "s0", 1, 100),
            cells("new", "user", "lives_in", "Chicago", "fact", "one", "s1", 1, 900),
        ]}
        r = Retriever(client_for(record, [rows]))
        got = r.superseded_by("k")
        self.assertNotIn("ORDER BY", record[0]["query"])
        self.assertEqual([c["obj"] for c in got], ["Chicago", "Boston"])

    def test_statement_text_is_rejoined_and_never_returned_in_part(self):
        record = []
        replies = [
            {"columns": ["s.nchunks"], "rows": [cells(2)]},
            {"columns": ["s.t0", "s.t1", "s.nchunks"],
             "rows": [cells("hello ", "world", 2)]},
        ]
        r = Retriever(client_for(record, replies))
        self.assertEqual(r.statement_text("sk"), "hello world")

    def test_a_missing_chunk_raises_rather_than_truncating(self):
        record = []
        replies = [
            {"columns": ["s.nchunks"], "rows": [cells(2)]},
            {"columns": ["s.t0", "s.t1", "s.nchunks"],
             "rows": [cells("hello ", None, 2)]},
        ]
        r = Retriever(client_for(record, replies))
        with self.assertRaises(ValueError):
            r.statement_text("sk")

    def test_the_evidence_path_asks_for_the_claims_own_id(self):
        record = []
        path = {"columns": ["path"],
                "rows": [[{"type": "path", "value": {"nodes": [{"id": 1}]}}]]}
        r = Retriever(client_for(record, [path]))
        got = r.evidence_path("k_tokyo")
        self.assertEqual(record[0]["parameters"]["sourceNode"],
                         stable_id("Claim", "k_tokyo", 52))
        self.assertIn("algo.SSpaths", record[0]["query"])
        self.assertEqual(len(got), 1)

    def test_a_rejected_path_call_is_no_receipt_not_a_wrong_answer(self):
        record = []
        err = urllib.error.HTTPError(
            "u", 400, "bad", {}, io.BytesIO(b'{"error":{"message":"nope"}}'))
        r = Retriever(client_for(record, [err]))
        self.assertEqual(r.evidence_path("k_tokyo"), [])


# --------------------------------------------------------------------------
class StubRetriever:
    def __init__(self, evidence: dict):
        self.evidence = evidence
        self.asked: list[str] = []

    def evidence_for(self, key, with_text=True):
        self.asked.append(key)
        return self.evidence.get(
            key, {"key": key, "found": False, "reason": "not in graph"})


class StubLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system, user):
        self.calls.append((system, user))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def evidence(key, ts, obj, superseded=()):
    return {"key": key, "found": True,
            "claim": {"nkey": key, "subj": "user", "pred": "lives_in",
                      "obj": obj, "kind": "fact", "sid": "s", "turn": 1,
                      "ts": ts},
            "statement": {"nkey": f"s|1", "sid": "s", "idx": 1},
            "text": f"I live in {obj}",
            "superseded": [{"obj": o, "ts": ts - 100} for o in superseded]}


class Answering(unittest.TestCase):
    def setUp(self):
        self.gate = build_gate(CLAIMS)

    def _answerer(self, ev, reply=None):
        llm = StubLLM(reply) if reply is not None else None
        return Answerer(StubRetriever(ev), self.gate, llm)

    def test_the_tolerance_keeps_late_evidence_and_counts_that_it_did(self):
        # Measured over all 500 oracle instances: 43 have evidence dated after
        # the question, none by more than 24h. A strict filter drops those.
        ev = {"k_tokyo": evidence("k_tokyo", 1_700_000_000 + 3600, "Tokyo")}
        a = self._answerer(ev, {"answer": "Tokyo", "used": [1]})
        out = a.answer("Where do I live?", asked_at=1_700_000_000)
        self.assertEqual(out["status"], ANSWERED)
        self.assertEqual(out["retrieval"]["dropped_future"], 0)
        self.assertEqual(out["retrieval"]["admitted_by_tolerance"], 1)
        self.assertEqual(out["retrieval"]["as_of_tolerance"], 86_400)

    def test_beyond_the_tolerance_is_still_dropped(self):
        ev = {"k_tokyo": evidence("k_tokyo", 1_700_000_000 + 90_000, "Tokyo")}
        a = self._answerer(ev, {"answer": "Tokyo", "used": [1]})
        out = a.answer("Where do I live?", asked_at=1_700_000_000)
        self.assertEqual(out["status"], ABSENT)
        self.assertEqual(out["retrieval"]["dropped_future"], 1)
        self.assertEqual(out["retrieval"]["admitted_by_tolerance"], 0)

    def test_a_zero_tolerance_restores_the_strict_rule(self):
        ev = {"k_tokyo": evidence("k_tokyo", 1_700_000_000 + 60, "Tokyo")}
        a = Answerer(StubRetriever(ev), self.gate,
                     StubLLM({"answer": "Tokyo", "used": [1]}),
                     as_of_tolerance=0)
        out = a.answer("Where do I live?", asked_at=1_700_000_000)
        self.assertEqual(out["status"], ABSENT)
        self.assertEqual(out["retrieval"]["dropped_future"], 1)

    def test_evidence_before_the_question_is_not_counted_as_admitted(self):
        ev = {"k_chicago": evidence("k_chicago", 1_700_000_000 - 60, "Chicago")}
        a = self._answerer(ev, {"answer": "Chicago", "used": [1]})
        out = a.answer("Where do I live?", asked_at=1_700_000_000)
        self.assertEqual(out["retrieval"]["admitted_by_tolerance"], 0)

    def test_a_claim_said_well_after_the_question_is_never_used(self):
        ev = {"k_tokyo": evidence("k_tokyo", 1_700_001_000 + 200_000, "Tokyo"),
              "k_chicago": evidence("k_chicago", 1_700_000_000, "Chicago")}
        a = self._answerer(ev, {"answer": "Chicago", "used": [1]})
        out = a.answer("Where do I live?", asked_at=1_700_001_000)
        self.assertEqual(out["status"], ANSWERED)
        self.assertEqual(out["retrieval"]["dropped_future"], 1)
        keys = [c["claim_key"] for c in out["evidence"]]
        self.assertNotIn("k_tokyo", keys)

    def test_evidence_is_selected_by_relevance_not_by_recency(self):
        # The Aug 16 regression: sorting the whole pool newest-first and then
        # truncating discarded high-scoring old claims in favour of recent
        # irrelevant ones. Session mode made it visible -- recall fell 0.734 to
        # 0.549 -- but the bug was there in claim mode too.
        ev = {}
        for i in range(10):
            ev[f"pad{i}"] = evidence(f"pad{i}", 1_700_000_000 + i, f"noise{i}")
        ev["k_chicago"] = evidence("k_chicago", 1_600_000_000, "Chicago")

        class RankedGate:
            name = "ranked"

            def search(self, question, limit=10, min_score=0.0):
                from docket.gate import Hit
                out = [Hit("k_chicago", 99.0, {"live": 99.0},
                           {"sid": "s", "pred": "lives_in"})]
                out += [Hit(f"pad{i}", 0.1, {"x": 0.1}, {"sid": "s"})
                        for i in range(10)]
                return out[:limit]

        a = Answerer(StubRetriever(ev), RankedGate(),
                     StubLLM({"answer": "Chicago", "used": [1]}),
                     candidates=11, evidence=2)
        out = a.answer("Where do I live?", asked_at=1_800_000_000)
        keys = [c["claim_key"] for c in out["evidence"]]
        self.assertIn("k_chicago", keys,
                      "the highest-scoring claim was dropped for newer noise")

    def test_evidence_is_newest_first_so_the_tip_leads(self):
        ev = {"k_tokyo": evidence("k_tokyo", 1_700_002_000, "Tokyo"),
              "k_chicago": evidence("k_chicago", 1_700_000_000, "Chicago")}
        a = self._answerer(ev, {"answer": "Tokyo", "used": [1]})
        out = a.answer("Where do I live?", asked_at=1_700_009_000)
        self.assertEqual(out["evidence"][0]["triple"][2], "Tokyo")

    def test_the_answerer_can_be_put_in_session_mode(self):
        ev = {"k_chicago": evidence("k_chicago", 1_700_000_000, "Chicago")}
        a = Answerer(StubRetriever(ev), self.gate,
                     StubLLM({"answer": "Chicago", "used": [1]}), sessions=2)
        out = a.answer("Where do I live?", asked_at=1_700_009_000)
        self.assertEqual(out["retrieval"]["mode"], "sessions")

    def test_claim_mode_is_still_the_default_and_is_reported(self):
        ev = {"k_chicago": evidence("k_chicago", 1_700_000_000, "Chicago")}
        a = self._answerer(ev, {"answer": "Chicago", "used": [1]})
        out = a.answer("Where do I live?", asked_at=1_700_009_000)
        self.assertEqual(out["retrieval"]["mode"], "claims")
        self.assertEqual(out["retrieval"]["candidates"], 12)

    def test_nothing_matching_is_absent_and_never_calls_the_model(self):
        llm = StubLLM({"answer": "should not be reached"})
        a = Answerer(StubRetriever({}), self.gate, llm)
        out = a.answer("quantum chromodynamics", asked_at=1_700_009_000)
        self.assertEqual(out["status"], ABSENT)
        self.assertEqual(llm.calls, [])

    def test_claims_that_all_postdate_the_question_are_absent_not_answered(self):
        ev = {"k_tokyo": evidence("k_tokyo", 1_700_002_000, "Tokyo")}  # +400,002,000s
        a = self._answerer(ev, {"answer": "Tokyo", "used": [1]})
        out = a.answer("Where do I live?", asked_at=1_600_000_000)
        self.assertEqual(out["status"], ABSENT)
        self.assertEqual(out["retrieval"]["dropped_future"], 1)

    def test_the_model_saying_not_in_memory_is_absent_not_an_answer(self):
        ev = {"k_chicago": evidence("k_chicago", 1_700_000_000, "Chicago")}
        a = self._answerer(ev, {"answer": "NOT_IN_MEMORY", "used": []})
        out = a.answer("Where do I live?", asked_at=1_700_009_000)
        self.assertEqual(out["status"], ABSENT)
        self.assertIn("does not contain", out["reason"])

    def test_a_failed_model_call_is_unmeasured_never_absent(self):
        from docket.llm import LLMError
        ev = {"k_chicago": evidence("k_chicago", 1_700_000_000, "Chicago")}
        a = self._answerer(ev, LLMError("429 forever"))
        out = a.answer("Where do I live?", asked_at=1_700_009_000)
        self.assertEqual(out["status"], UNMEASURED)
        self.assertIn("model call failed", out["reason"])

    def test_an_unusable_model_shape_is_unmeasured(self):
        ev = {"k_chicago": evidence("k_chicago", 1_700_000_000, "Chicago")}
        a = self._answerer(ev, ["not", "a", "dict"])
        out = a.answer("Where do I live?", asked_at=1_700_009_000)
        self.assertEqual(out["status"], UNMEASURED)

    def test_no_model_configured_is_unmeasured_with_the_evidence_kept(self):
        ev = {"k_chicago": evidence("k_chicago", 1_700_000_000, "Chicago")}
        a = self._answerer(ev)
        out = a.answer("Where do I live?", asked_at=1_700_009_000)
        self.assertEqual(out["status"], UNMEASURED)
        self.assertTrue(out["evidence"])

    def test_every_answer_carries_citations_and_the_terms_that_found_them(self):
        ev = {"k_chicago": evidence("k_chicago", 1_700_000_000, "Chicago",
                                    superseded=["Boston"])}
        a = self._answerer(ev, {"answer": "Chicago", "used": [1]})
        out = a.answer("Where do I live?", asked_at=1_700_009_000)
        cite = out["evidence"][0]
        self.assertEqual(cite["session"], "s")
        self.assertEqual(cite["triple"], ["user", "lives_in", "Chicago"])
        self.assertTrue(cite["matched_terms"])
        self.assertEqual(cite["superseded"], ["Boston"])
        self.assertTrue(cite["used"])

    def test_the_prompt_shows_what_a_value_replaced(self):
        block = render_evidence([evidence("k", 1, "Chicago", superseded=["Boston"])])
        self.assertIn("replaced an earlier value: Boston", block)

    def test_a_candidate_missing_from_the_graph_is_counted_not_silently_lost(self):
        a = self._answerer({}, {"answer": "x", "used": []})
        out = a.answer("Where do I live?", asked_at=1_700_009_000)
        self.assertEqual(out["status"], ABSENT)
        self.assertGreater(out["retrieval"]["dropped_missing"], 0)


class Pagination(unittest.TestCase):
    """MEASURED Aug 16: HydraDB 0.1.0 rejects a continuation body that carries
    only cell_id and cursor with `422 missing field query`. The first real
    paged read in this project was the Day 2 bulk claim load, which is why a
    Day 1 client shipped with a page-2 shape that had never been exercised."""

    def _client(self, record, replies):
        return client_for(record, replies)

    def test_the_continuation_carries_the_original_query_and_parameters(self):
        record = []
        page1 = {"columns": ["c.nkey"], "rows": [cells("a")], "next_cursor": "c1"}
        page2 = {"columns": ["c.nkey"], "rows": [cells("b")], "next_cursor": None}
        client = self._client(record, [page1, page2])
        result = client.query("MATCH (c:Claim {pred: $p}) RETURN c.nkey",
                              {"p": "lives_in"})
        self.assertEqual(len(result.raw_rows), 2)
        self.assertEqual(record[1]["query"], record[0]["query"])
        self.assertEqual(record[1]["parameters"], {"p": "lives_in"})
        self.assertEqual(record[1]["cursor"], "c1")

    def test_a_cursor_that_does_not_advance_raises_instead_of_spinning(self):
        record = []
        same = {"columns": ["c.nkey"], "rows": [cells("a")], "next_cursor": "c1"}
        client = self._client(record, [same, same])
        with self.assertRaises(HydraError) as ctx:
            client.query("MATCH (c:Claim) RETURN c.nkey")
        self.assertIn("same cursor twice", str(ctx.exception))

    def test_paged_rows_walks_skip_until_a_short_page_ends_it(self):
        record = []
        col = {"columns": ["c.nkey"]}
        replies = [
            dict(col, rows=[cells("a"), cells("b")]),
            dict(col, rows=[cells("c"), cells("d")]),
            dict(col, rows=[cells("e")]),
        ]
        r = Retriever(client_for(record, replies))
        rows = r.paged_rows("MATCH (c:Claim) RETURN c.nkey", "c.nkey",
                            page_size=2)
        self.assertEqual([x["c.nkey"] for x in rows], list("abcde"))
        self.assertIn("SKIP 0 LIMIT 2", record[0]["query"])
        self.assertIn("SKIP 2 LIMIT 2", record[1]["query"])
        self.assertIn("SKIP 4 LIMIT 2", record[2]["query"])
        self.assertIn("ORDER BY c.nkey", record[0]["query"])

    def test_a_server_cursor_halves_the_page_and_retries_that_page(self):
        # The server caps the page itself and hands back a cursor it cannot
        # serve. Rather than guess its cap, come down to meet it.
        record = []
        col = {"columns": ["c.nkey"]}
        replies = [
            dict(col, rows=[cells("a")], next_cursor="x"),   # capped
            dict(col, rows=[cells("a")]),                    # fits at half
        ]
        r = Retriever(client_for(record, replies))
        rows = r.paged_rows("MATCH (c:Claim) RETURN c.nkey", "c.nkey",
                            page_size=100)
        self.assertEqual(len(rows), 1)
        self.assertIn("LIMIT 100", record[0]["query"])
        self.assertIn("LIMIT 50", record[1]["query"])
        self.assertIn("SKIP 0", record[1]["query"])

    def test_it_refuses_rather_than_return_part_when_no_page_is_small_enough(self):
        record = []
        cursored = {"columns": ["c.nkey"], "rows": [cells("a")],
                    "next_cursor": "x"}
        r = Retriever(client_for(record, [cursored] * 40))
        with self.assertRaises(HydraError) as ctx:
            r.paged_rows("MATCH (c:Claim) RETURN c.nkey", "c.nkey",
                         page_size=100, min_page=25)
        self.assertIn("no way to read this result whole", str(ctx.exception))

    def test_a_paged_bulk_read_returns_every_claim(self):
        record = []
        cols = [f"c.{p}" for p in CLAIM_PROPS]
        page = lambda *keys: {
            "columns": cols,
            "rows": [cells(k, "user", "lives_in", "Tokyo", "fact", "one",
                           "s1", 1, 100) for k in keys]}
        r = Retriever(client_for(record, [page("k1", "k2"), page("k3")]))
        claims = r.all_claims(page_size=2)
        self.assertEqual(sorted(claims), ["k1", "k2", "k3"])
        self.assertIn("SKIP 2 LIMIT 2", record[1]["query"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

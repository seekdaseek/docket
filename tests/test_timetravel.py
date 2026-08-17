"""Selecting which fact to demonstrate is where a time-travel demo lies.

Two failures are guarded here, and the second one shipped before it was caught:

  A predicate merely RESTATED -- same value, mentioned twice -- is not a fact
  that changed, and presenting it as one would be invisible in the output.

  A predicate that ACCUMULATES is not a fact that changed either. `user owns`
  took 523 values across 538 mentions because the user owns 523 objects. The
  first version of this tool ranked by number of distinct values and therefore
  picked exactly that, then printed 500 lines of "the most recent thing
  mentioned" and called it time travel.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import timetravel

DAY = 86_400


def claims(*rows):
    out = {}
    for i, (subj, pred, obj, ts) in enumerate(rows):
        out[f"k{i}"] = {"nkey": f"k{i}", "subj": subj, "pred": pred,
                        "obj": obj, "ts": ts}
    return out


def series(pred, values, start=0, step=DAY):
    return claims(*[("user", pred, v, start + i * step)
                    for i, v in enumerate(values)])


class RejectsThingsThatAreNotChanges(unittest.TestCase):
    def test_a_fact_restated_unchanged_is_not_a_change(self):
        c = claims(("user", "lives_in", "Lisbon", 100),
                   ("user", "lives_in", "Lisbon", 200))
        self.assertEqual([], timetravel.changed_facts(c))

    def test_an_accumulating_list_is_rejected(self):
        # The `user owns` case, in miniature.
        c = series("owns", [f"thing{i}" for i in range(30)])
        self.assertEqual([], timetravel.changed_facts(c))

    def test_the_cap_is_adjustable_and_enforced(self):
        c = series("owns", ["a", "b", "c", "d", "e"])
        self.assertEqual([], timetravel.changed_facts(c, max_values=4))
        self.assertEqual(1, len(timetravel.changed_facts(c, max_values=5)))

    def test_many_mentions_of_few_values_is_still_a_change(self):
        # Restating the current value repeatedly must not disqualify a fact.
        c = series("lives_in", ["Lisbon"] * 10 + ["Porto"] * 10)
        got = timetravel.changed_facts(c)
        self.assertEqual(1, len(got))
        self.assertEqual(2, got[0]["unique"])

    def test_the_same_predicate_for_different_subjects_is_not_one_fact(self):
        c = claims(("alice", "lives_in", "Lisbon", 100),
                   ("bob", "lives_in", "Porto", 200))
        self.assertEqual([], timetravel.changed_facts(c))

    def test_claims_without_a_timestamp_are_ignored(self):
        c = {"a": {"nkey": "a", "subj": "u", "pred": "p", "obj": "A", "ts": None},
             "b": {"nkey": "b", "subj": "u", "pred": "p", "obj": "B", "ts": 200}}
        self.assertEqual([], timetravel.changed_facts(c))


class Ranking(unittest.TestCase):
    def test_fewer_values_ranks_first_not_more(self):
        # The inversion that caused the bug.
        c = {}
        c.update(series("clean", ["A", "B"], start=0))
        c.update({f"m{i}": v for i, v in enumerate(
            series("messy", ["A", "B", "C", "D", "E", "F"], start=10 * DAY).values())})
        self.assertEqual("clean", timetravel.changed_facts(c)[0]["pred"])

    def test_a_longer_span_breaks_a_tie(self):
        c = {}
        c.update(series("near", ["A", "B"], start=0, step=DAY))
        c.update({f"f{i}": v for i, v in enumerate(
            series("far", ["X", "Y"], start=0, step=400 * DAY).values())})
        self.assertEqual("far", timetravel.changed_facts(c)[0]["pred"])

    def test_a_fact_with_a_supersedes_chain_is_promoted(self):
        """The graph's own judgement beats the heuristic.

        `chained` has MORE values, so the heuristic ranks it second. The
        SUPERSEDES edges the pipeline itself wrote pull it to the front.
        """
        c = {}
        for i, v in enumerate(["A", "B", "C"]):
            c[f"ch{i}"] = {"nkey": f"ch{i}", "subj": "user", "pred": "chained",
                           "obj": v, "ts": i * DAY}
        for i, v in enumerate(["X", "Y"]):
            c[f"pl{i}"] = {"nkey": f"pl{i}", "subj": "user", "pred": "plain",
                           "obj": v, "ts": i * DAY}
        facts = timetravel.changed_facts(c)
        self.assertEqual("plain", facts[0]["pred"])  # fewest values first

        class Ret:
            def superseded_by(self, key):
                return [{}, {}] if key.startswith("ch") else []

        ranked = timetravel.rank_by_chain(Ret(), facts)
        self.assertEqual("chained", ranked[0]["pred"])
        self.assertEqual(2, ranked[0]["chain"])
        self.assertEqual(0, ranked[1]["chain"])

    def test_a_chain_probe_that_raises_does_not_kill_the_run(self):
        facts = timetravel.changed_facts(series("p", ["A", "B"]))

        class Ret:
            def superseded_by(self, key):
                raise RuntimeError("variable-length MATCH requires a fixed source id")

        self.assertEqual(0, timetravel.rank_by_chain(Ret(), facts)[0]["chain"])


class Marks(unittest.TestCase):
    """One mark per CHANGE, deduped, capped. The first version emitted one per
    mention and printed 500 near-identical blocks."""

    def rows(self, pairs):
        return [{"obj": v, "ts": t} for v, t in pairs]

    def test_one_mark_per_change_plus_a_before_date(self):
        m = timetravel.marks_for(self.rows([("A", 100), ("A", 200), ("B", 300)]))
        self.assertEqual([100 - DAY, 100, 300], m)

    def test_the_first_mark_predates_everything(self):
        m = timetravel.marks_for(self.rows([("A", 5 * DAY), ("B", 6 * DAY)]))
        self.assertLess(m[0], 5 * DAY)

    def test_duplicate_timestamps_are_not_asked_twice(self):
        # Many claims share a timestamp: same conversation, same second.
        m = timetravel.marks_for(self.rows([("A", 100), ("B", 100), ("C", 100)]))
        self.assertEqual(len(m), len(set(m)))

    def test_output_is_capped(self):
        rows = self.rows([(f"v{i}", i * DAY) for i in range(50)])
        m = timetravel.marks_for(rows, cap=6)
        self.assertEqual(7, len(m))  # the before-date plus six

    def test_the_cap_keeps_the_first_and_last_change(self):
        rows = self.rows([(f"v{i}", i * DAY) for i in range(50)])
        m = timetravel.marks_for(rows, cap=6)
        self.assertEqual(0, m[1])
        self.assertEqual(49 * DAY, m[-1])


class Dates(unittest.TestCase):
    def test_epoch_renders_as_a_date(self):
        self.assertEqual("2023-04-10", timetravel.when(1681138020))

    def test_missing_is_named_not_crashed(self):
        self.assertEqual("unknown", timetravel.when(None))


class HeaderMatchesTheWalk(unittest.TestCase):
    """The header reports `chain`, and the walk below prints the chain. If the
    ranking is skipped the counter stays zero while the walk still finds edges,
    and the tool contradicts itself in the middle of a demo. That shipped once."""

    def test_probing_one_fact_sets_its_chain_count(self):
        facts = timetravel.changed_facts(series("p", ["A", "B"]))
        self.assertEqual(0, facts[0]["chain"])

        class Ret:
            def superseded_by(self, key):
                return [{"obj": "A"}]

        timetravel.rank_by_chain(Ret(), facts[:1])
        self.assertEqual(1, facts[0]["chain"])

    def test_probe_limit_does_not_skip_the_first_fact(self):
        facts = timetravel.changed_facts(series("p", ["A", "B"]))

        class Ret:
            def superseded_by(self, key):
                return [{}]

        timetravel.rank_by_chain(Ret(), facts, probe=1)
        self.assertEqual(1, facts[0]["chain"])

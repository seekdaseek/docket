"""Claims into the graph, and the chains that make time answerable.

Three writes per claim at most, all in the one form this server has:

    (:Statement)-[:ASSERTS]->(:Claim)      the claim cites the sentence
    (:Claim)-[:ABOUT]->(:Entity)           the claim joins its subject
    (:Claim)-[:SUPERSEDES]->(:Claim)       the later replaces the earlier

The Statement and Entity ends name only their ids and merge, so loading claims
cannot disturb the statements already ingested -- measured, not hoped.

WHAT A CHAIN IS FOR, and the mistake this file exists to prevent.

The first version of this code chained any change of value within
(subject, predicate). Run against 48 real sessions it produced 120 links, and
reading them showed most were fiction: the user owns Vans AND Converse AND
brown dress shoes, has finished six different books, attended several events.
Nothing there replaced anything. A chain that says otherwise does not merely
lose information -- it answers a temporal question confidently and wrongly,
which is the exact failure this project claims to prevent.

So cardinality is DECLARED per claim by the extractor, and defaults to deny:

  one   a new value replaces the old one. lives_in, works_at, current_phone.
  many  values accumulate. owns, attended, purchased, completed.

A predicate chains only when EVERY claim carrying it was declared "one". One
dissenting claim collapses the whole predicate to "many", and the disagreement
is reported rather than settled by majority, because a predicate the extractor
cannot classify consistently is one a reader should look at.

Missing a chain costs a temporal answer. Inventing one costs the truth.
"""
from __future__ import annotations

import re

from .extract import claim_key, entity_key
from .schema import (CLAIM, ENTITY, REL_ABOUT, REL_ASSERTS, REL_SUPERSEDES,
                     STATEMENT, statement_key)

PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)


def normalise_object(value: str) -> str:
    """The form two objects must share to count as the same value.

    Real extractor output made this necessary: the same shoes came back as
    "Converse Chuck Taylor All Star sneakers" in one session and
    "Converse_Chuck_Taylor_All_Star_sneakers" in another. Compared raw, that
    reads as a change of value and would have been chained as one.
    """
    text = value.replace("_", " ").lower()
    text = PUNCT.sub(" ", text)
    return " ".join(text.split())


def claim_props(*, sid: str, turn: int, subject: str, predicate: str,
                obj: str, kind: str, ts: int, cardinality: str = "many") -> dict:
    return {
        "sid": sid,
        "turn": int(turn),
        "subj": subject,
        "pred": predicate,
        "obj": obj,
        "kind": kind,
        "ts": int(ts),
        "card": cardinality,
    }


def load_claims(writer, rows: list[dict]) -> dict:
    """Write every claim with its statement citation and its entity link."""
    written = 0
    entities: set[str] = set()
    for row in rows:
        sid = row["sid"]
        ts = int(row["ts"])
        for claim in row["claims"]:
            key = claim_key(sid, claim["turn"], claim["subject"],
                            claim["predicate"], claim["object"])
            props = claim_props(sid=sid, turn=claim["turn"],
                                subject=claim["subject"],
                                predicate=claim["predicate"],
                                obj=claim["object"], kind=claim["kind"],
                                ts=ts,
                                cardinality=claim.get("cardinality", "many"))
            writer.edge(
                src_label=STATEMENT,
                src_key=statement_key(sid, claim["turn"]),
                src_props={},
                rel=REL_ASSERTS,
                dst_label=CLAIM, dst_key=key, dst_props=props)
            ekey = entity_key(claim["subject"])
            writer.edge(
                src_label=CLAIM, src_key=key, src_props={},
                rel=REL_ABOUT,
                dst_label=ENTITY, dst_key=ekey,
                dst_props={"name": claim["subject"], "ekey": ekey})
            entities.add(ekey)
            written += 1
    return {"claims_written": written, "entities": len(entities)}


def predicate_cardinality(rows: list[dict]) -> tuple[dict, dict]:
    """Resolve each predicate to one/many, and report every disagreement.

    Unanimity, not majority. A predicate is single-valued only when nothing
    ever said otherwise.
    """
    votes: dict = {}
    for row in rows:
        for claim in row["claims"]:
            card = claim.get("cardinality", "many")
            tally = votes.setdefault(claim["predicate"], {"one": 0, "many": 0})
            tally["one" if card == "one" else "many"] += 1
    resolved = {pred: ("one" if tally["many"] == 0 and tally["one"] > 0 else "many")
                for pred, tally in votes.items()}
    disputed = {pred: tally for pred, tally in votes.items()
                if tally["one"] and tally["many"]}
    return resolved, disputed


def chain_rows(rows: list[dict]) -> list[tuple[str, str]]:
    """(later_key, earlier_key) pairs, one per link in every chain.

    Pure: no client, no network. A wrong chain is a wrong answer to every
    temporal question at once, so it is caught here rather than in a benchmark
    score three days later.
    """
    cardinality, _ = predicate_cardinality(rows)

    groups: dict = {}
    for row in rows:
        sid = row["sid"]
        ts = int(row["ts"])
        for claim in row["claims"]:
            if cardinality.get(claim["predicate"]) != "one":
                continue  # accumulating predicate: nothing here replaces
            key = claim_key(sid, claim["turn"], claim["subject"],
                            claim["predicate"], claim["object"])
            group = (entity_key(claim["subject"]), claim["predicate"])
            groups.setdefault(group, []).append(
                (ts, sid, claim["turn"], key, normalise_object(claim["object"])))

    links = []
    for group in sorted(groups):
        items = sorted(groups[group], key=lambda x: (x[0], x[1], x[2]))
        previous = None
        for ts, sid, turn, key, obj in items:
            if previous is not None and previous[4] == obj:
                # Restating the same value is corroboration, not replacement.
                continue
            if previous is not None:
                links.append((key, previous[3]))
            previous = (ts, sid, turn, key, obj)
    return links


def load_chains(writer, rows: list[dict]) -> dict:
    """Write the SUPERSEDES edges, and clear any that no longer apply.

    Every claim in `rows` has its outgoing SUPERSEDES deleted first, then the
    current links are written. That makes chain loading self-correcting: a
    reload after a rule change removes yesterday's wrong links instead of
    leaving them beside the right ones, and it is the only way a graph already
    holding a bad chain can be repaired without a rebuild.
    """
    links = chain_rows(rows)
    cleared = 0
    for row in rows:
        for claim in row["claims"]:
            key = claim_key(row["sid"], claim["turn"], claim["subject"],
                            claim["predicate"], claim["object"])
            writer.delete_edges(CLAIM, writer.node_id(CLAIM, key),
                                REL_SUPERSEDES)
            cleared += 1
    for later, earlier in links:
        writer.edge(
            src_label=CLAIM, src_key=later, src_props={},
            rel=REL_SUPERSEDES,
            dst_label=CLAIM, dst_key=earlier, dst_props={})
    resolved, disputed = predicate_cardinality(rows)
    return {
        "supersedes_links": len(links),
        "claims_cleared_first": cleared,
        "predicates_single_valued": sum(1 for v in resolved.values() if v == "one"),
        "predicates_accumulating": sum(1 for v in resolved.values() if v == "many"),
        "predicates_disputed": disputed,
    }

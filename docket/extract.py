"""Turning what was said into claims that can be superseded.

A claim here is deliberately small: subject, predicate, object, and the turn it
came from. Small claims are what make a SUPERSEDES chain meaningful -- "user /
lives_in / Berlin" can be replaced by "user / lives_in / Munich" six months
later, and the pair is the whole knowledge-update question.

The validation is the interesting part, and all of it exists because a model
that is mostly right is the dangerous kind:

  A claim whose `turn` is out of range is DROPPED, never clamped. Clamping it
  to 0 would attach real text to the wrong statement and produce a citation
  that looks perfect and points at the wrong sentence.

  A session whose response could not be parsed is recorded as UNMEASURED with
  the reason. A session the model read and found nothing in is recorded as
  ABSENT with zero claims. Collapsing those two is the failure this project is
  named after, and it would silently shrink the corpus.

  Every drop is counted by reason, so the extraction can be audited rather
  than trusted.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

MAX_SUBJECT = 120
MAX_PREDICATE = 60
MAX_OBJECT = 400
KINDS = ("fact", "preference", "event", "plan")
CARDINALITIES = ("one", "many")

# Bumped whenever the prompt or the validator changes in a way that makes
# older stored rows unusable. Rows carrying a lower version are re-extracted
# rather than silently mixed with new ones.
PROMPT_VERSION = 2

SYSTEM = """You extract durable claims from one conversation session.

A claim is a single small statement about the user or their world that could
still be true, or could be contradicted, weeks later. Write each one as a
subject, a predicate and an object.

Rules:
- Only claims supported by the text. Never infer, never generalise.
- The subject is who or what the claim is about. Use "user" for the person.
- The predicate is a short lowercase snake_case relation, like lives_in,
  works_at, owns, prefers, allergic_to, scheduled, completed.
- The object is the value, kept short.
- Give the index of the turn the claim came from, exactly as numbered below.
- Assistant turns rarely contain claims about the user. Extract from them only
  when the assistant is recording something the user stated.
- Skip pleasantries, small talk, and anything hypothetical or hedged.
- kind is one of: fact, preference, event, plan.

cardinality is the most important field and the easiest to get wrong. It asks
whether the subject can have only ONE current value for this predicate at a
time, or MANY at once.

- "one" when a new value REPLACES the old one. Someone lives in one city,
  works at one employer, has one current phone number. If you learn a new
  value, the previous one stopped being true.
- "many" when values ACCUMULATE. Someone can own many pairs of shoes, attend
  many events, finish many books, buy many things. Learning a new value does
  not make the earlier one false.

If you are unsure, answer "many". Saying "one" about something that
accumulates would record a contradiction that never happened.

Return ONLY a JSON array, no prose and no code fence. An empty array is a
valid and useful answer when the session contains nothing durable.

[{"turn": 0, "subject": "user", "predicate": "lives_in", "object": "Berlin",
  "kind": "fact", "cardinality": "one"},
 {"turn": 3, "subject": "user", "predicate": "owns", "object": "a red bicycle",
  "kind": "fact", "cardinality": "many"}]"""


class Dropped(Exception):
    pass


def render_session(session) -> str:
    """The prompt body: numbered turns, so a claim can name where it came from."""
    lines = [f"Session date: {session.when.date().isoformat()}", ""]
    for index, turn in enumerate(session.turns):
        lines.append(f"[{index}] {turn.role}: {turn.content}")
    return "\n".join(lines)


def claim_key(session_id: str, turn: int, subject: str, predicate: str,
              obj: str) -> str:
    """Deterministic, so re-loading the same claims writes the same nodes."""
    digest = hashlib.blake2b(obj.encode(), digest_size=4).hexdigest()
    return f"{session_id}|{turn}|{subject}|{predicate}|{digest}"


def entity_key(subject: str) -> str:
    """The normalised form two claims must share to be about one thing."""
    return " ".join(subject.lower().split())


def validate(raw, turn_count: int) -> tuple[list[dict], dict]:
    """Keep the claims that are usable; count every drop by reason."""
    drops: dict = {}

    def drop(reason):
        drops[reason] = drops.get(reason, 0) + 1

    if not isinstance(raw, list):
        raise ValueError(f"expected a JSON array, got {type(raw).__name__}")

    kept = []
    for item in raw:
        if not isinstance(item, dict):
            drop("not_an_object")
            continue
        turn = item.get("turn")
        if not isinstance(turn, int) or isinstance(turn, bool):
            drop("turn_not_an_integer")
            continue
        if not 0 <= turn < turn_count:
            # Never clamped. A clamped index cites the wrong sentence
            # convincingly, which is worse than losing the claim.
            drop("turn_out_of_range")
            continue
        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")
        if not all(isinstance(x, str) and x.strip()
                   for x in (subject, predicate, obj)):
            drop("missing_or_empty_field")
            continue
        if (len(subject) > MAX_SUBJECT or len(predicate) > MAX_PREDICATE
                or len(obj) > MAX_OBJECT):
            drop("field_too_long")
            continue
        kind = item.get("kind")
        if kind not in KINDS:
            drop("bad_kind")
            continue
        cardinality = item.get("cardinality")
        if cardinality not in CARDINALITIES:
            # Default-deny rather than drop: an unlabelled predicate is treated
            # as accumulating, which loses a chain at worst. Guessing "one"
            # would invent a contradiction, which is worse.
            drop("cardinality_missing_defaulted_to_many")
            cardinality = "many"
        kept.append({
            "turn": turn,
            "subject": subject.strip(),
            "predicate": " ".join(predicate.lower().split()).replace(" ", "_"),
            "object": obj.strip(),
            "kind": kind,
            "cardinality": cardinality,
        })

    seen = set()
    deduped = []
    for claim in kept:
        signature = (claim["turn"], entity_key(claim["subject"]),
                     claim["predicate"], claim["object"].lower())
        if signature in seen:
            drop("duplicate_in_response")
            continue
        seen.add(signature)
        deduped.append(claim)
    return deduped, drops


class ClaimStore:
    """Append-only record of extraction, one line per session.

    Written as it happens rather than at the end, because the end is the part
    of a long run that may not arrive. Holding the claims here means a failed
    graph write never costs a second call to the model.
    """

    def __init__(self, path: str):
        self.path = path
        self.rows: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue  # a half-written final line after a kill
                    if row.get("sid"):
                        self.rows[row["sid"]] = row

    @property
    def measured(self) -> dict[str, dict]:
        return {sid: row for sid, row in self.rows.items()
                if row.get("status") == "measured"}

    @property
    def unmeasured(self) -> dict[str, dict]:
        return {sid: row for sid, row in self.rows.items()
                if row.get("status") != "measured"}

    def _append(self, row: dict) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.rows[row["sid"]] = row

    def record_measured(self, sid: str, claims: list, drops: dict,
                        turn_count: int) -> None:
        self._append({"sid": sid, "status": "measured", "claims": claims,
                      "drops": drops, "turns": turn_count,
                      "prompt_version": PROMPT_VERSION, "at": time.time()})

    def stale(self, version: int = PROMPT_VERSION) -> dict:
        """Measured rows written by an older prompt.

        Re-extracted rather than mixed in. A row missing a field the current
        chain logic depends on is not a smaller row, it is a wrong one.
        """
        return {sid: row for sid, row in self.measured.items()
                if int(row.get("prompt_version", 1)) < version}

    def record_unmeasured(self, sid: str, reason: str) -> None:
        """The model did not answer usably. NOT the same as finding nothing."""
        self._append({"sid": sid, "status": "unmeasured", "reason": reason,
                      "claims": [], "at": time.time()})


def extract_session(llm, session) -> tuple[list[dict], dict]:
    raw = llm.complete_json(SYSTEM, render_session(session))
    return validate(raw, len(session.turns))

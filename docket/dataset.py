"""Loading LongMemEval, with the traps handled once so callers cannot hit them.

The file is a JSON list of instances with these fields, observed directly:

    question_id, question_type, question, answer, question_date,
    haystack_dates, haystack_session_ids, haystack_sessions, answer_session_ids

Four things this module refuses to leave to the caller:

  Chronology comes from `haystack_dates`, never from list position. The oracle
  file's sessions arrive unsorted.

  A question ending in `_abs` is an abstention question: the history does not
  contain the answer and the only correct response is to say so. It gets a
  first-class flag rather than being detected by string suffix at three
  different call sites.

  `answer_session_ids` is ground truth for WHICH sessions hold the evidence.
  Keeping it on the instance is what makes citation precision measurable, not
  just answer accuracy.

  The same session id appears in many instances' haystacks. Ingesting it once
  per instance would multiply both the extraction bill and the graph. Sessions
  are therefore keyed globally by id, and `content_hash` proves that two
  instances sharing an id really do share the same text rather than merely
  the same label.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime

from .timeparse import order_sessions, parse


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    has_answer: bool = False
    """Ground truth: this turn holds the evidence for its instance's question.

    Kept so the evaluation harness can score citations at turn level, and
    deliberately NOT part of `content_hash`, so it cannot affect dedupe.

    It must never reach the graph. `schema.statement_props` is an allowlist and
    the offline suite asserts that a turn with has_answer=True produces no node
    property carrying it: a retriever that can see which turn holds the answer
    is scoring itself against the answer key.
    """


@dataclass
class Session:
    session_id: str
    when: datetime
    turns: list[Turn]

    @property
    def content_hash(self) -> str:
        h = hashlib.sha256()
        for t in self.turns:
            h.update(t.role.encode())
            h.update(b"\x00")
            h.update(t.content.encode())
            h.update(b"\x01")
        return h.hexdigest()[:16]

    @property
    def text(self) -> str:
        return "\n".join(f"{t.role}: {t.content}" for t in self.turns)


@dataclass
class Instance:
    question_id: str
    question_type: str
    question: str
    answer: str
    asked_at: datetime
    sessions: list[Session]
    evidence_session_ids: list[str] = field(default_factory=list)

    @property
    def is_abstention(self) -> bool:
        """True when the history does not contain the answer.

        The benchmark marks these with an `_abs` suffix on the question id and
        its own retrieval evaluation skips them as unanswerable. Here they are
        the point: a memory system that cannot say "I do not have this" fails
        the only question where confidence is guaranteed to be wrong.
        """
        return self.question_id.endswith("_abs")

    @property
    def session_ids(self) -> list[str]:
        return [s.session_id for s in self.sessions]


def _turns(raw_session) -> list[Turn]:
    out = []
    for turn in raw_session:
        if not isinstance(turn, dict):
            raise ValueError(f"turn is {type(turn).__name__}, expected object")
        role = turn.get("role")
        content = turn.get("content")
        if role is None or content is None:
            raise ValueError(f"turn missing role or content: {list(turn)}")
        out.append(Turn(role=str(role), content=str(content),
                        has_answer=bool(turn.get("has_answer", False))))
    return out


def load_instance(raw: dict) -> Instance:
    ids = raw["haystack_session_ids"]
    dates = raw["haystack_dates"]
    sessions_raw = raw["haystack_sessions"]
    if len(sessions_raw) != len(ids):
        raise ValueError(
            f"{raw.get('question_id')}: {len(sessions_raw)} sessions but "
            f"{len(ids)} ids")
    ordered = order_sessions(ids, dates)
    sessions = [Session(session_id=sid, when=when, turns=_turns(sessions_raw[i]))
                for i, sid, when in ordered]
    return Instance(
        question_id=raw["question_id"],
        question_type=raw.get("question_type", "unknown"),
        question=raw["question"],
        answer=raw.get("answer", ""),
        asked_at=parse(raw["question_date"]),
        sessions=sessions,
        evidence_session_ids=list(raw.get("answer_session_ids") or []),
    )


def load(path: str, limit: int | None = None) -> list[Instance]:
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} is {type(data).__name__}, expected a list")
    if limit is not None:
        data = data[:limit]
    return [load_instance(x) for x in data]


def unique_sessions(instances: list[Instance]) -> tuple[dict[str, Session], list[str]]:
    """Every distinct session across the instances, plus any id collisions.

    A collision is one session id carrying two different texts. It is returned
    rather than raised so the caller can report it as a finding; if it ever
    fires, ingesting by id alone would overwrite one conversation with another.
    """
    seen: dict[str, Session] = {}
    hashes: dict[str, str] = {}
    collisions: list[str] = []
    for inst in instances:
        for s in inst.sessions:
            h = s.content_hash
            if s.session_id in hashes:
                if hashes[s.session_id] != h:
                    collisions.append(s.session_id)
                continue
            hashes[s.session_id] = h
            seen[s.session_id] = s
    return seen, collisions


def stats(instances: list[Instance]) -> dict:
    total_sessions = sum(len(i.sessions) for i in instances)
    uniq, collisions = unique_sessions(instances)
    types: dict[str, int] = {}
    for i in instances:
        types[i.question_type] = types.get(i.question_type, 0) + 1
    turns = sum(len(s.turns) for s in uniq.values())
    chars = sum(len(s.text) for s in uniq.values())
    return {
        "instances": len(instances),
        "abstention": sum(1 for i in instances if i.is_abstention),
        "types": dict(sorted(types.items(), key=lambda kv: -kv[1])),
        "session_slots": total_sessions,
        "unique_sessions": len(uniq),
        "reuse_factor": round(total_sessions / len(uniq), 2) if uniq else 0,
        "id_collisions": collisions,
        "turns_in_unique": turns,
        "chars_in_unique": chars,
    }

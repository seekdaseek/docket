"""Structural ingest: conversations become Statements attached to Sessions.

No model runs here. This pass writes only what was actually said and when,
which is the part of memory that must be exactly right before anything is
extracted from it.

Resumability is the design constraint, not a nicety. The Mac is available in
the evening and the lid closes; a run that loses its place is a run that never
finishes. So:

  Progress is appended to a JSONL file, one line per session, as it happens.
  Nothing is buffered until the end, because the end is the thing that may
  not arrive.

  A session marked `done` is skipped on the next run.

  A session marked `start` with no `done` was interrupted mid-write. Its IN
  edges are deleted before it is written again, because edges duplicate on
  repeat and a doubled edge would inflate every per-session count. Node
  properties need no such care: a repeat write to the same id merges.

  Ids are deterministic, so a resumed run produces the same graph as an
  uninterrupted one.
"""
from __future__ import annotations

import json
import os
import time

from .dataset import Session
from .hydra import HydraError
from .ids import IdRegistry
from .schema import (REL_IN, SESSION, STATEMENT, Writer, session_props,
                     statement_key, statement_props)


class Checkpoint:
    """An append-only record of which sessions are written."""

    def __init__(self, path: str):
        self.path = path
        self.done: set[str] = set()
        self.started: set[str] = set()
        if os.path.exists(path):
            self._load()

    def _load(self) -> None:
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    # A half-written final line is expected after a kill.
                    continue
                sid = row.get("sid")
                if not sid:
                    continue
                if row.get("event") == "done":
                    self.done.add(sid)
                elif row.get("event") == "start":
                    self.started.add(sid)

    @property
    def interrupted(self) -> set[str]:
        """Sessions that began and never finished."""
        return self.started - self.done

    def _append(self, row: dict) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def start(self, sid: str) -> None:
        self.started.add(sid)
        self._append({"event": "start", "sid": sid, "at": time.time()})

    def finish(self, sid: str, statements: int) -> None:
        self.done.add(sid)
        self._append({"event": "done", "sid": sid, "statements": statements,
                      "at": time.time()})


def ingest_session(writer: Writer, session: Session, *,
                   rewrite_edges: bool = False) -> int:
    """Write one session's statements. Returns how many were written.

    The session node is born on the FIRST statement's write, carrying its own
    properties on the destination end. Later statements name only its id, which
    merges and cannot wipe it -- verified by preflight before any ingest runs.
    """
    ts = int(session.when.timestamp())
    written = 0
    for index, turn in enumerate(session.turns):
        first = index == 0
        dst_props = (session_props(session_id=session.session_id, ts=ts,
                                   turns=len(session.turns))
                     if first else {})
        kw = dict(
            src_label=STATEMENT,
            src_key=statement_key(session.session_id, index),
            src_props=statement_props(session_id=session.session_id,
                                      index=index, role=turn.role, ts=ts,
                                      text=turn.content,
                                      chunk_chars=writer.chunk_chars),
            rel=REL_IN,
            dst_label=SESSION,
            dst_key=session.session_id,
            dst_props=dst_props,
        )
        if rewrite_edges:
            writer.exactly_once(**kw)
        else:
            writer.edge(**kw)
        written += 1
    return written


def ingest(client, sessions: dict[str, Session], *, checkpoint_path: str,
           id_bits: int = 52, chunk_chars: int | None = None,
           progress=None) -> dict:
    """Write every session that is not already written.

    `sessions` is the globally deduped map from dataset.unique_sessions, so a
    session shared by forty questions is stored once.
    """
    ckpt = Checkpoint(checkpoint_path)
    writer = Writer(client, IdRegistry(bits=id_bits),
                    **({"chunk_chars": chunk_chars} if chunk_chars else {}))

    empty: list[str] = []
    skipped = 0
    statements = 0
    started = time.time()
    order = sorted(sessions, key=lambda s: (sessions[s].when, s))

    for position, sid in enumerate(order, 1):
        session = sessions[sid]
        if not session.turns:
            # A session with no turns cannot create a node, because there is no
            # node-only CREATE. Reported rather than silently absent.
            empty.append(sid)
            continue
        if sid in ckpt.done:
            skipped += 1
            continue
        rewrite = sid in ckpt.interrupted
        ckpt.start(sid)
        try:
            n = ingest_session(writer, session, rewrite_edges=rewrite)
        except HydraError as e:
            raise HydraError(
                f"session {sid} failed after {writer.writes} writes: {e}. "
                f"Rerun the same command; it resumes from the checkpoint and "
                f"rewrites this session's edges.") from None
        ckpt.finish(sid, n)
        statements += n
        if progress:
            progress(position, len(order), sid, n)

    return {
        "sessions_total": len(sessions),
        "sessions_written": len(order) - skipped - len(empty),
        "sessions_skipped_already_done": skipped,
        "sessions_empty": empty,
        "statements_written": statements,
        "writes": writer.writes,
        "deletes": writer.deletes,
        "ids_issued": writer.ids.issued,
        "seconds": round(time.time() - started, 1),
    }

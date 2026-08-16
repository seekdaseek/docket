"""The graph, expressed in the only write form this server has.

HydraDB 0.1.0 accepts exactly one shape of write:

    CREATE (a {id: <int>, ...})-[:REL]->(b {id: <int>, ...})

One hop, an integer id required on both ends, and no node-only CREATE at all.
Everything else about the schema falls out of that:

  A node cannot be created alone. Session nodes are therefore born on the
  first Statement that points at them, carrying their properties on the
  destination end of that write.

  A repeat write to the same id MERGES: named properties are overwritten,
  unnamed ones survive. That is what makes ingest idempotent and dedupe free,
  and it is why the second and later statements of a session name only the
  session's id and cannot wipe it. Measured, probe6 group 1.

  Edges DUPLICATE on repeat. Where multiplicity would corrupt a count,
  `exactly_once` deletes then creates, which measured 2 -> delete -> 1.

  Ids are rendered as integer literals and text goes through `parameters`,
  because that is the combination the probes actually exercised. A string
  literal breaks the parser near a thousand characters.

  Turn text longer than the parameter cap is CHUNKED across t0, t1, ... with
  an `nchunks` count, and rejoined on read. It is never truncated. A memory
  system that silently drops the end of what someone said is the failure this
  project is named after.
"""
from __future__ import annotations

import re

from .hydra import MAX_PARAM_CHARS, HydraUnsupported, lit
from .ids import IdRegistry

SESSION = "Session"
STATEMENT = "Statement"
CLAIM = "Claim"
ENTITY = "Entity"

REL_IN = "IN"
REL_ASSERTS = "ASSERTS"
REL_ABOUT = "ABOUT"
REL_SUPERSEDES = "SUPERSEDES"
REL_SAME_AS = "SAME_AS"

NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Chunk below the client cap so a multibyte-heavy turn cannot cross it.
CHUNK_CHARS = 12_000


def _check_name(kind: str, value: str) -> str:
    if not isinstance(value, str) or not NAME.match(value):
        raise ValueError(f"{kind} must match {NAME.pattern}, got {value!r}")
    return value


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split text into storable pieces. Empty text is one empty piece."""
    if size <= 0 or size > MAX_PARAM_CHARS:
        raise ValueError(f"chunk size must be 1..{MAX_PARAM_CHARS}")
    if not text:
        return [""]
    return [text[i:i + size] for i in range(0, len(text), size)]


def text_props(text: str, size: int = CHUNK_CHARS) -> dict:
    """Properties holding the whole text: t0..tN plus nchunks."""
    parts = chunk_text(text, size)
    props = {f"t{i}": part for i, part in enumerate(parts)}
    props["nchunks"] = len(parts)
    return props


def join_text(props: dict) -> str:
    """Rebuild text from t0..tN, refusing to return a partial string.

    A missing chunk raises. Returning what happened to be present would be a
    quiet truncation, which is exactly the class of bug being designed out.
    """
    n = props.get("nchunks")
    if n is None:
        raise ValueError("no nchunks property: cannot tell whole from partial")
    parts = []
    for i in range(int(n)):
        key = f"t{i}"
        if key not in props or props[key] is None:
            raise ValueError(
                f"chunk {key} missing of {n}; the stored text is incomplete "
                f"and returning the rest would be a silent truncation")
        parts.append(props[key])
    return "".join(parts)


def render_props(props: dict, prefix: str) -> tuple[str, dict]:
    """Render a property map as inline Cypher plus the parameters it needs.

    Strings become parameters (the parser cannot take long literals). Numbers
    and booleans become literals (measured working inline). None is dropped,
    because writing a null property and not writing it are indistinguishable
    on read and the shorter statement is cheaper.
    """
    parts: list[str] = []
    params: dict = {}
    for key in sorted(props):
        _check_name("property", key)
        value = props[key]
        if value is None:
            continue
        if isinstance(value, str):
            pname = f"{prefix}_{key}"
            parts.append(f"{key}: ${pname}")
            params[pname] = value
        elif isinstance(value, (int, float, bool)):
            parts.append(f"{key}: {lit(value)}")
        else:
            raise HydraUnsupported(
                f"property {key} is {type(value).__name__}; only strings, "
                f"numbers and booleans can be stored on a node here")
    return ", ".join(parts), params


class Writer:
    """Issues ids and turns them into the one write form the server accepts."""

    def __init__(self, client, registry: IdRegistry | None = None,
                 chunk_chars: int = CHUNK_CHARS):
        self.client = client
        self.ids = registry if registry is not None else IdRegistry()
        self.chunk_chars = chunk_chars
        self.writes = 0
        self.deletes = 0

    def node_id(self, label: str, natural_key: str) -> int:
        return self.ids.issue(label, natural_key)

    def edge(self, *, src_label: str, src_key: str, src_props: dict,
             rel: str, dst_label: str, dst_key: str, dst_props: dict,
             rel_props: dict | None = None) -> tuple[int, int]:
        """Write one edge, creating or merging both endpoints.

        `src_props` and `dst_props` are what this write asserts. Properties not
        named here survive on an existing node.
        """
        _check_name("label", src_label)
        _check_name("label", dst_label)
        _check_name("relationship", rel)
        src_id = self.node_id(src_label, src_key)
        dst_id = self.node_id(dst_label, dst_key)

        a_text, a_params = render_props(dict(src_props, nkey=src_key), "a")
        b_text, b_params = render_props(dict(dst_props, nkey=dst_key), "b")
        r_text, r_params = render_props(rel_props or {}, "r")

        a_body = f"id: {src_id}" + (f", {a_text}" if a_text else "")
        b_body = f"id: {dst_id}" + (f", {b_text}" if b_text else "")
        rel_body = f":{rel}" + (f" {{{r_text}}}" if r_text else "")

        cypher = (f"CREATE (a:{src_label} {{{a_body}}})"
                  f"-[{rel_body}]->"
                  f"(b:{dst_label} {{{b_body}}})")
        self.client.query(cypher, {**a_params, **b_params, **r_params})
        self.writes += 1
        return src_id, dst_id

    def delete_edges(self, src_label: str, src_id: int, rel: str) -> None:
        """Remove every `rel` edge leaving that node.

        DELETE removes ALL matching edges, which is what makes the
        delete-then-create pair exactly-once rather than merely idempotent.
        """
        _check_name("label", src_label)
        _check_name("relationship", rel)
        self.client.query(
            f"MATCH (a:{src_label} {{id: {int(src_id)}}})-[r:{rel}]->(b) DELETE r")
        self.deletes += 1

    def exactly_once(self, **kw) -> tuple[int, int]:
        """An edge that exists once however many times this is called."""
        src_id = self.node_id(kw["src_label"], kw["src_key"])
        self.delete_edges(kw["src_label"], src_id, kw["rel"])
        return self.edge(**kw)


def statement_props(*, session_id: str, index: int, role: str, ts: int,
                    text: str, chunk_chars: int = CHUNK_CHARS) -> dict:
    """Everything a Statement node stores.

    This is an ALLOWLIST on purpose. LongMemEval turns carry `has_answer`,
    which is ground truth about where the evidence lives; writing it into the
    graph would let retrieval score itself against a label it can see. It is
    read only by the evaluation harness, and the test suite asserts it never
    appears here.
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"unexpected role {role!r}")
    props = {
        "sid": session_id,
        "idx": int(index),
        "role": role,
        "ts": int(ts),
    }
    props.update(text_props(text, chunk_chars))
    return props


def session_props(*, session_id: str, ts: int, turns: int) -> dict:
    return {"sid": session_id, "ts": int(ts), "turns": int(turns)}


def statement_key(session_id: str, index: int) -> str:
    return f"{session_id}|{index}"

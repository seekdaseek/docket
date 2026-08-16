"""Retrieval: the gate proposes, the graph decides.

Every claim that reaches an answer is re-read from HydraDB, filtered by the
question's own timestamp, ordered in the database, and carried back with the
statement it came from. The local index is a cache of the graph, never a
substitute for it -- if the two ever disagree the graph is right, and the
`--verify` path in the runner exists to catch that.

Why the tip is chosen by ORDER BY rather than by walking SUPERSEDES: chains in
this corpus are sparse. 5,687 claims produced 189 SUPERSEDES links whatever
grouping was tried, because the same fact is rarely restated with a changed
value under the same predicate. So SUPERSEDES is the AUDITABLE OVERLAY -- it
shows what a value replaced, where a replacement exists -- and the retrieval
path is `ts <= asked_at`, ORDER BY ts DESC, take the tip. That was measured
before it was decided.

Every query shape here is named in QUERY_SHAPES so tools/preflight2.py can run
each one against a live node and report supported / unsupported / unmeasured
rather than letting a rejected pattern surface as an empty answer.
"""
from __future__ import annotations

from .hydra import HydraError, HydraTruncated
from .ids import stable_id
from .schema import CLAIM, ENTITY, STATEMENT, join_text

DEFAULT_BITS = 52
MAX_CHAIN_HOPS = 5

# name -> one-line description, used by preflight2 and by the README table.
QUERY_SHAPES = {
    "claims_all": "MATCH (c:Claim) RETURN c.<props> ORDER BY c.nkey SKIP n LIMIT m",
    "claim_by_key": "MATCH (c:Claim {nkey: $k}) RETURN c.<props>",
    "tip_by_pred": "MATCH (c:Claim {pred: $p}) WHERE c.ts <= T RETURN ... ORDER BY c.ts DESC LIMIT n",
    "tip_by_pred_subj": "MATCH (c:Claim {pred: $p, subj: $s}) WHERE ... -- two inline properties",
    "supersedes": "MATCH (a:Claim {id: <int>})-[:SUPERSEDES*1..N]->(b:Claim) RETURN b.<props>",
    "statement_for_claim": "MATCH (s:Statement)-[:ASSERTS]->(c:Claim {nkey: $k}) RETURN s.<props>",
    "statement_text": "MATCH (s:Statement {nkey: $k}) RETURN s.t0, ... , s.nchunks",
    "sspaths": "CALL algo.SSpaths({sourceNode: $id, ...}) YIELD path RETURN path",
}

CLAIM_PROPS = ("nkey", "subj", "pred", "obj", "kind", "card", "sid", "turn", "ts")
STATEMENT_PROPS = ("nkey", "sid", "idx", "role", "ts", "nchunks")


def _projection(alias: str, props) -> str:
    return ", ".join(f"{alias}.{p}" for p in props)


def _row_to(props, row: dict, alias: str) -> dict:
    return {p: row.get(f"{alias}.{p}") for p in props}


class Retriever:
    """Reads the graph. Holds no answer logic and never calls a model."""

    def __init__(self, client, bits: int = DEFAULT_BITS):
        self.client = client
        self.bits = bits

    # -- bulk read for the gate --------------------------------------------
    def paged_rows(self, base_cypher: str, order_by: str,
                   parameters: dict | None = None, page_size: int = 500,
                   min_page: int = 25, max_pages: int = 2000) -> list[dict]:
        """Read a large result with SKIP/LIMIT, because cursors do not work.

        MEASURED Aug 16 against HydraDB 0.1.0: a result large enough to page
        comes back with a `next_cursor`, and asking for it returns

            400 invalid_request: ClientProtocol query is not supported yet:
                result cursor ...

        so the server advertises a continuation it cannot serve. probe1 had
        already measured `skip` as supported and noted it as the "pagination
        fallback if cursors are awkward", which is exactly the situation.

        `follow_cursor=False` makes the client raise rather than attempt the
        unsupported continuation. If a page still returns a cursor the request
        was above the server's own cap, so the page size halves and that page
        is retried -- adaptively finding the cap beats guessing it.

        The ORDER BY must be a total order or SKIP will drop and repeat rows
        between pages. Callers pass a unique property.
        """
        out: list[dict] = []
        skip = 0
        size = int(page_size)
        pages = 0
        while True:
            pages += 1
            if pages > max_pages:
                raise HydraTruncated(
                    f"stopped after {max_pages} pages at {len(out)} rows; "
                    f"the read is not terminating")
            cypher = (f"{base_cypher} ORDER BY {order_by} "
                      f"SKIP {int(skip)} LIMIT {int(size)}")
            try:
                result = self.client.query(cypher, parameters,
                                           follow_cursor=False)
            except HydraTruncated:
                if size <= min_page:
                    raise HydraTruncated(
                        f"the server paged even a {size}-row request and "
                        f"cannot serve the cursor it returned; there is no "
                        f"way to read this result whole") from None
                size = max(min_page, size // 2)
                continue
            rows = result.rows
            out.extend(rows)
            if len(rows) < size:
                return out
            skip += len(rows)

    def all_claims(self, page_size: int = 500) -> dict[str, dict]:
        """Every Claim node, keyed by nkey.

        A label-only MATCH is legal: the server's own message is "node-only
        MATCH requires an id, LABEL, or property predicate". Ordered by nkey,
        which is unique by construction, so SKIP paging neither drops nor
        repeats a claim.
        """
        base = f"MATCH (c:{CLAIM}) RETURN {_projection('c', CLAIM_PROPS)}"
        out: dict[str, dict] = {}
        for row in self.paged_rows(base, "c.nkey", page_size=page_size):
            claim = _row_to(CLAIM_PROPS, row, "c")
            key = claim.get("nkey")
            if key:
                out[key] = claim
        return out

    def claim(self, key: str) -> dict | None:
        cypher = (f"MATCH (c:{CLAIM} {{nkey: $k}}) "
                  f"RETURN {_projection('c', CLAIM_PROPS)}")
        rows = self.client.query(cypher, {"k": key}).rows
        if not rows:
            return None
        claim = _row_to(CLAIM_PROPS, rows[0], "c")
        return claim if claim.get("nkey") else None

    # -- the as-of read that decides the answer ----------------------------
    def tip(self, predicate: str, asked_at: int, subject: str | None = None,
            limit: int = 5) -> list[dict]:
        """Claims for a predicate as of a timestamp, newest first.

        `asked_at` is rendered as an integer literal, not a parameter: numbers
        are literals everywhere in this codebase because that is the
        combination the probes exercised. The LIMIT is a literal for the same
        reason.
        """
        where = f"WHERE c.ts <= {int(asked_at)}"
        params = {"p": predicate}
        if subject is None:
            pattern = f"MATCH (c:{CLAIM} {{pred: $p}})"
        else:
            pattern = f"MATCH (c:{CLAIM} {{pred: $p, subj: $s}})"
            params["s"] = subject
        cypher = (f"{pattern} {where} RETURN {_projection('c', CLAIM_PROPS)} "
                  f"ORDER BY c.ts DESC LIMIT {int(limit)}")
        rows = self.client.query(cypher, params).rows
        return [_row_to(CLAIM_PROPS, r, "c") for r in rows
                if r.get("c.nkey") is not None]

    def superseded_by(self, key: str, hops: int = MAX_CHAIN_HOPS) -> list[dict]:
        """What this claim replaced, walking the chain backwards in time.

        MEASURED Aug 16: a variable-length pattern must start from a FIXED ID.
        Matching the source by an inline property returns

            400 invalid_request: OpenCypher query is not supported yet:
                variable-length MATCH requires a fixed source id

        The id is a deterministic hash of the claim key, so this costs nothing
        -- but the source must be `{id: <literal>}`, never `{nkey: $k}`.

        The maximum hop count is also mandatory; `*1..` and `*` are rejected.
        A bounded walk is a design statement rather than a workaround: an
        unbounded chain walk over a memory graph is how you get an answer
        nobody can check.
        """
        node_id = stable_id(CLAIM, key, self.bits)
        cypher = (f"MATCH (a:{CLAIM} {{id: {int(node_id)}}})"
                  f"-[:SUPERSEDES*1..{int(hops)}]->"
                  f"(b:{CLAIM}) RETURN {_projection('b', CLAIM_PROPS)}")
        rows = self.client.query(cypher).rows
        out = [_row_to(CLAIM_PROPS, r, "b") for r in rows
               if r.get("b.nkey") is not None]
        # Sorted here rather than in Cypher: ORDER BY on a variable-length
        # pattern is not in the measured surface, and a chain is at most `hops`
        # long. One unverified feature per query is the budget.
        out.sort(key=lambda c: -(c.get("ts") or 0))
        return out

    # -- citations ----------------------------------------------------------
    def statement_for(self, claim_key: str) -> dict | None:
        cypher = (f"MATCH (s:{STATEMENT})-[:ASSERTS]->(c:{CLAIM} {{nkey: $k}}) "
                  f"RETURN {_projection('s', STATEMENT_PROPS)}")
        rows = self.client.query(cypher, {"k": claim_key}).rows
        if not rows:
            return None
        stmt = _row_to(STATEMENT_PROPS, rows[0], "s")
        return stmt if stmt.get("nkey") else None

    def statement_text(self, statement_key: str) -> str | None:
        """The whole turn, rejoined from its chunks.

        `join_text` raises on a missing chunk rather than returning the part it
        found. A citation that is quietly half a sentence is worse than none.
        """
        head = self.client.query(
            f"MATCH (s:{STATEMENT} {{nkey: $k}}) RETURN s.nchunks",
            {"k": statement_key}).scalar(None)
        if head is None:
            return None
        n = int(head)
        cols = ", ".join(f"s.t{i}" for i in range(n))
        rows = self.client.query(
            f"MATCH (s:{STATEMENT} {{nkey: $k}}) RETURN {cols}, s.nchunks",
            {"k": statement_key}).rows
        if not rows:
            return None
        props = {f"t{i}": rows[0].get(f"s.t{i}") for i in range(n)}
        props["nchunks"] = n
        return join_text(props)

    def evidence_path(self, claim_key: str, max_len: int = 3,
                      path_count: int = 3) -> list[dict]:
        """The native path payload for a claim, via algo.SSpaths.

        `RETURN path` is the one projection on this server that hands back
        whole nodes with their properties, which is exactly what a citation
        needs. Failures are returned as an empty list with the reason attached
        by the caller -- a missing path is not a wrong answer, it is a missing
        receipt, and the two must not be conflated.
        """
        node_id = stable_id(CLAIM, claim_key, self.bits)
        cypher = ("CALL algo.SSpaths({sourceNode: $sourceNode, "
                  "relTypes: ['SUPERSEDES'], relDirection: 'outgoing', "
                  f"maxLen: {int(max_len)}, pathCount: {int(path_count)}, "
                  f"resultLimit: {int(path_count)}}}) YIELD path RETURN path")
        try:
            result = self.client.query(cypher, {"sourceNode": node_id})
        except HydraError:
            return []
        paths = []
        for row in result.rows:
            value = row.get("path")
            if isinstance(value, dict) and "nodes" in value:
                paths.append(value)
        return paths

    # -- assembly -----------------------------------------------------------
    def evidence_for(self, claim_key: str, with_text: bool = True) -> dict:
        """One claim plus everything needed to check it."""
        claim = self.claim(claim_key)
        if claim is None:
            return {"key": claim_key, "found": False,
                    "reason": "no claim with that key in the graph"}
        stmt = self.statement_for(claim_key)
        text = None
        if stmt and with_text:
            try:
                text = self.statement_text(stmt["nkey"])
            except (ValueError, HydraError) as exc:
                text = None
                stmt = dict(stmt, text_error=str(exc))
        return {
            "key": claim_key,
            "found": True,
            "claim": claim,
            "statement": stmt,
            "text": text,
            "superseded": self.superseded_by(claim_key),
        }

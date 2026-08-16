"""HydraDB client, written against the surface six probes MEASURED on 14 Aug 2026.

Nothing here is inferred from Neo4j. Every constraint below was observed as a
server error and is enforced client side, so it fails in Python with a sentence
you can act on rather than as an HTTP 400 halfway through an ingest.

  parameters   The request key is `parameters`. `params` and `query_parameters`
               both come back "missing OpenCypher query parameter $x".

  literals     A string literal longer than about a thousand characters is a
               parse error ("Invalid input 'x': expected XOR"). The same string
               sent through `parameters` round-trips intact at 16,000 chars and
               fails with an internal error at 64,000. So text goes through
               parameters, and `lit()` refuses to render a long string at all.

  RETURN       Plain Cypher may return `<binding>.<property>` or `count(*)`.
               `RETURN n`, `RETURN id(n)` and path projections are rejected.
               The one exception is a native path procedure, where `RETURN path`
               is legal and yields whole nodes with their properties.

  MATCH        A node-only MATCH needs an id, a label, or a property predicate.
               `MATCH (n)` alone is rejected.

  absence      An id-MATCH against a node that does not exist returns ONE ROW
               with a null cell, not zero rows. Absence is therefore never
               inferred from row count on an id-MATCH; `count_by_property` uses
               a non-id predicate, where count(*) really does come back 0.

  readiness    A 400 carrying a JSON error body is a live node answering. The
               previous client treated any 4xx as "not ready" and burned sixty
               seconds waiting for a node that was already up.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:8443"
DEFAULT_ADMIN = "http://127.0.0.1:9090"
DEFAULT_GRAPH = "default"
DEFAULT_NAMESPACE = "default"
DEFAULT_CELL = "cell-0"

# The node logs runtime_limit_ms 30000 on its own spans. Keep the socket
# timeout above it so a server-side deadline arrives as a server error with a
# reason rather than as a client timeout with none.
SERVER_DEADLINE_SECONDS = 30
DEFAULT_TIMEOUT = 35

# Measured: 16,000 chars through `parameters` round-trips byte for byte,
# 64,000 returns "internal query execution error". The cap sits below the
# failure point rather than at it.
MAX_PARAM_CHARS = 15_000

# Measured: 1,000 chars as a literal is already a parse error. Literals are
# for ids and short tokens; anything longer is a bug in the caller.
MAX_LITERAL_CHARS = 256

CONSISTENCY_LEVELS = ("causal", "strong")


class HydraError(RuntimeError):
    """Any failure that came from the server or the transport."""


class HydraTruncated(HydraError):
    """The server had more rows and the client could not fetch them.

    Never downgraded to a warning. A partial answer presented as a whole
    answer is worse than an error.
    """


class HydraUnsupported(HydraError):
    """The caller asked for something this server is known not to support.

    Raised before the request leaves Python, so the message names the measured
    constraint instead of echoing a parse error.
    """


def lit(value) -> str:
    """Render a Python value as a Cypher literal.

    Only for ids, numbers and short tokens. A long string raises rather than
    being rendered, because the server's parser breaks on it and the resulting
    error ("expected XOR") says nothing about the real cause.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"cannot render {value!r} as a Cypher literal")
        return repr(value)
    if isinstance(value, str):
        if len(value) > MAX_LITERAL_CHARS:
            raise HydraUnsupported(
                f"string literal of {len(value)} chars: this server's parser "
                f"fails above roughly 1,000. Pass it through `parameters` "
                f"instead (measured intact to 16,000).")
        out = (value.replace("\\", "\\\\")
                    .replace("'", "\\'")
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t"))
        return f"'{out}'"
    raise TypeError(f"no Cypher literal for {type(value).__name__}")


def check_parameters(parameters: dict | None) -> None:
    """Reject a parameter payload the server will choke on, before sending it."""
    if parameters is None:
        return
    if not isinstance(parameters, dict):
        raise HydraUnsupported(
            f"parameters must be an object, got {type(parameters).__name__}")
    for key, value in parameters.items():
        _check_param_value(key, value)


def _check_param_value(key, value) -> None:
    if isinstance(value, str) and len(value) > MAX_PARAM_CHARS:
        raise HydraUnsupported(
            f"parameter ${key} is {len(value)} chars; this server stores "
            f"{MAX_PARAM_CHARS} reliably and errors at 64,000. Chunk it "
            f"rather than sending a value that may be refused.")
    if isinstance(value, list):
        for item in value:
            _check_param_value(key, item)
    elif isinstance(value, dict):
        for sub, item in value.items():
            _check_param_value(f"{key}.{sub}", item)


def cell_value(cell):
    """Unwrap one typed cell object to a plain Python value.

    A cell tagged `null` becomes None. Unknown type tags are NOT guessed at:
    the wrapper is returned intact so a caller can see what arrived instead of
    receiving a plausible-looking None.
    """
    if isinstance(cell, dict) and "value" in cell:
        return cell["value"]
    if isinstance(cell, dict) and cell.get("type") == "null":
        return None
    return cell


class QueryResult:
    def __init__(self, payload: dict, pages: int = 1):
        self.raw = payload
        self.columns: list[str] = payload.get("columns") or []
        self.raw_rows: list[list] = payload.get("rows") or []
        self.bookmark = payload.get("bookmark")
        self.read_epoch = payload.get("read_epoch")
        self.next_cursor = payload.get("next_cursor")
        self.query_id = payload.get("query_id")
        self.pages = pages

    @property
    def rows(self) -> list[dict]:
        """Rows as column-keyed dicts of plain values."""
        out = []
        for row in self.raw_rows:
            out.append({c: cell_value(v) for c, v in zip(self.columns, row)})
        return out

    def scalar(self, default=None):
        """The single value of a one-row one-column result.

        Returns `default` when there are no rows, and also when the single cell
        is null -- which is what an id-MATCH against a missing node produces.
        Raises on any other shape, because silently taking the first of several
        columns is how the wrong number ends up in a report.
        """
        if not self.raw_rows:
            return default
        if len(self.columns) != 1:
            raise HydraError(
                f"scalar() needs exactly one column, got {self.columns}")
        value = cell_value(self.raw_rows[0][0])
        return default if value is None else value

    def __len__(self) -> int:
        return len(self.raw_rows)

    def __repr__(self) -> str:
        return (f"<QueryResult {len(self.raw_rows)} rows "
                f"cols={self.columns} pages={self.pages}>")


class HydraClient:
    def __init__(self, base_url: str = DEFAULT_BASE, token: str = "",
                 graph: str = DEFAULT_GRAPH,
                 namespace: str = DEFAULT_NAMESPACE,
                 cell: str = DEFAULT_CELL,
                 admin_url: str = DEFAULT_ADMIN,
                 timeout: float = DEFAULT_TIMEOUT,
                 consistency: str | None = None,
                 send_bookmark: bool = True,
                 cursor_key: str = "cursor",
                 opener=None):
        self.base_url = base_url.rstrip("/")
        self.admin_url = admin_url.rstrip("/")
        self.token = token
        self.graph = graph
        self.namespace = namespace
        self.cell = cell
        self.timeout = timeout
        if consistency is not None and consistency not in CONSISTENCY_LEVELS:
            raise ValueError(
                f"consistency must be one of {CONSISTENCY_LEVELS}, "
                f"got {consistency!r}")
        self.consistency = consistency
        self.send_bookmark = send_bookmark
        self.cursor_key = cursor_key
        self.bookmark: str | None = None
        self.queries = 0
        self.pages_fetched = 0
        self.seconds_in_server = 0.0
        self._opener = opener or urllib.request.urlopen

    @property
    def url(self) -> str:
        return f"{self.base_url}/v1/graphs/{self.graph}/query"

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.url, data=data, method="POST", headers={
            "Authorization": f"Bearer {self.token}",
            "X-Graph-Namespace": self.namespace,
            "Content-Type": "application/json",
        })
        started = time.time()
        try:
            with self._opener(req, timeout=self.timeout) as r:
                text = r.read().decode()
        except urllib.error.HTTPError as e:
            detail = e.read()[:400]
            raise HydraError(f"http {e.code}: {detail!r}") from None
        except urllib.error.URLError as e:
            raise HydraError(f"transport: {e.reason}") from None
        finally:
            self.seconds_in_server += time.time() - started
        if not text.strip():
            raise HydraError("empty response body")
        try:
            return json.loads(text)
        except ValueError:
            raise HydraError(f"response was not JSON: {text[:200]!r}") from None

    def query(self, cypher: str, parameters: dict | None = None,
              consistency: str | None = None,
              follow_cursor: bool = True) -> QueryResult:
        """Run one Cypher statement and return the whole answer.

        One statement per request is enforced by the server ("query transport
        requires exactly one Cypher statement"), so there is deliberately no
        multi-statement helper here.
        """
        check_parameters(parameters)
        body = {"cell_id": self.cell, "query": cypher}
        if parameters is not None:
            body["parameters"] = parameters
        level = consistency if consistency is not None else self.consistency
        if level is not None:
            if level not in CONSISTENCY_LEVELS:
                raise ValueError(f"consistency must be one of {CONSISTENCY_LEVELS}")
            body["consistency"] = level
        if self.send_bookmark and self.bookmark:
            body["bookmark"] = self.bookmark
        payload = self._post(body)
        self.queries += 1
        self.pages_fetched += 1
        result = QueryResult(payload)
        if result.bookmark:
            self.bookmark = result.bookmark

        pages = 1
        while result.next_cursor:
            if not follow_cursor:
                raise HydraTruncated(
                    f"server returned next_cursor after {len(result.raw_rows)} "
                    f"rows and follow_cursor is off; the answer is incomplete")
            cursor = result.next_cursor
            page = self._next_page(cursor, cypher, parameters, level)
            pages += 1
            self.pages_fetched += 1
            result.raw_rows.extend(page.raw_rows)
            if page.next_cursor is not None and page.next_cursor == cursor:
                # The server handed back the same cursor it was given. Looping
                # on it would spin until the page guard trips and then report a
                # short answer as a whole one.
                raise HydraTruncated(
                    f"the server returned the same cursor twice after "
                    f"{len(result.raw_rows)} rows; pagination is not advancing "
                    f"and the answer is incomplete")
            result.next_cursor = page.next_cursor
            if page.bookmark:
                self.bookmark = page.bookmark
            if pages > 10000:
                raise HydraTruncated("pagination did not terminate")
        result.pages = pages
        return result

    def _next_page(self, cursor, cypher: str = "", parameters: dict | None = None,
                   consistency: str | None = None) -> QueryResult:
        """Fetch a continuation page.

        MEASURED Aug 16 against HydraDB 0.1.0: the continuation body must carry
        the ORIGINAL `query` as well as the cursor. Sending cell_id and cursor
        alone returns

            422 Failed to deserialize the JSON body into the target type:
                missing field `query`

        which surfaces as a truncated read -- and a truncated read of a memory
        graph is indistinguishable from a memory that does not contain the
        answer. The parameters go with it for the same reason: a continuation
        of a parameterised query without its parameters is a different query.
        """
        body = {"cell_id": self.cell, "query": cypher, self.cursor_key: cursor}
        if parameters is not None:
            body["parameters"] = parameters
        if consistency is not None:
            body["consistency"] = consistency
        if self.send_bookmark and self.bookmark:
            body["bookmark"] = self.bookmark
        try:
            payload = self._post(body)
        except HydraError as e:
            hint = ""
            if "not supported yet" in str(e) or "cursor" in str(e).lower():
                hint = (" This server advertises a cursor it cannot serve: "
                        "page the read with ORDER BY + SKIP/LIMIT instead "
                        "(see Retriever.paged_rows).")
            raise HydraTruncated(
                f"the server paged the result but the request shape for "
                f"fetching page 2 was rejected ({e}); rows already received "
                f"are NOT the whole answer.{hint}") from None
        return QueryResult(payload)

    # -- reads whose shape the measured surface actually permits -------------

    def count_by_property(self, label: str, prop: str, value,
                          extra: dict | None = None) -> int:
        """count(*) over a NON-ID property predicate.

        This is the only reliable way to ask whether something exists. An
        id-MATCH returns a null-cell row for a node that was never written, so
        row count on one of those is 1 whether or not the node is there.
        """
        if prop == "id":
            raise HydraUnsupported(
                "existence cannot be decided by an id-MATCH on this server: a "
                "missing node returns one row with a null cell. Match on a "
                "non-id property instead.")
        params = dict(extra or {})
        params["_v"] = value
        cypher = f"MATCH (n:{label} {{{prop}: $_v}}) RETURN count(*)"
        return int(self.query(cypher, params).scalar(0) or 0)

    def property_of(self, label: str, prop: str, value, want: str,
                    default=None):
        """One property of the single node matching a non-id predicate."""
        if prop == "id":
            raise HydraUnsupported(
                "read by a non-id predicate; see count_by_property")
        cypher = f"MATCH (n:{label} {{{prop}: $_v}}) RETURN n.{want}"
        return self.query(cypher, {"_v": value}).scalar(default)

    # -- readiness ----------------------------------------------------------

    def readyz(self) -> bool:
        """True when the admin port answers /readyz.

        A non-2xx status still means the process is listening; only a transport
        failure counts as not up.
        """
        req = urllib.request.Request(f"{self.admin_url}/readyz", method="GET")
        try:
            with self._opener(req, timeout=self.timeout) as r:
                r.read()
                return True
        except urllib.error.HTTPError:
            return True
        except urllib.error.URLError:
            return False

    def wait_ready(self, seconds: float = 60, interval: float = 1.0) -> float:
        """Block until the node answers, and return how long that took.

        A freshly started container opens its listeners before SlateDB and the
        writer lease are ready, so the first query after `docker start` can
        fail on a node that is merely still waking up.

        An HTTP error status counts as READY on purpose. A 400 with a JSON
        error body is a live node rejecting a query, which is exactly what a
        live node does. Only a transport failure means nothing is listening.
        """
        deadline = time.time() + seconds
        started = time.time()
        last = None
        while True:
            try:
                self.query("MATCH (n:DocketReadyCheck) RETURN count(*)")
                return time.time() - started
            except HydraError as e:
                last = e
                if not str(e).startswith("transport:"):
                    return time.time() - started
            if time.time() >= deadline:
                break
            time.sleep(interval)
        raise HydraError(f"node not ready after {seconds}s: {last}")

    def capabilities(self) -> dict:
        """What this client believes about the server, and on what basis.

        Everything marked measured was observed against a live node by
        tools/probe*.py on 14 Aug 2026.
        """
        return {
            "response_envelope": "measured",
            "typed_cells": "measured",
            "bookmark_returned": "measured",
            "parameters_key": "measured: `parameters`",
            "consistency_field": "measured: causal, strong",
            "param_string_chars": f"measured intact at 16000, capped at {MAX_PARAM_CHARS}",
            "literal_string_chars": "measured to fail near 1000",
            "return_forms": "measured: <binding>.<property>, count(*), procedure path",
            "node_only_match": "measured: needs id, label or property predicate",
            "id_match_absence": "measured: returns a null-cell row, not zero rows",
            "variable_length": "measured: explicit max hop required",
            "bookmark_accepted_in_request": "assumed",
            "cursor_request_shape": "assumed",
            "server_deadline_seconds": SERVER_DEADLINE_SECONDS,
        }

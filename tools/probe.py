"""Measure what this HydraDB node can actually do.

The repo went public a day before this was written and its README documents
one write and one read. Everything else a graph memory layer needs, from MERGE
to indexes to parameters to how a paged result is fetched, is unknown. Guessing
and finding out during ingest costs a day; asking the server costs a minute.

Every check is a real statement sent to a real node. A check that fails
records the server's own error text, because the error usually names the
supported alternative. Nothing here is inferred from Neo4j's behaviour: this
server is OpenCypher-compatible, which is not the same as Neo4j-complete.

    python3 tools/probe.py

Writes probe-report.json next to the repo and prints a summary.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docket.hydra import HydraClient, HydraError  # noqa: E402


class Unmeasured(Exception):
    """The check could not be performed, so it has no verdict.

    Distinct from a failure. A check that examined nothing must never report
    support, and must never report absence either.
    """

TAG = f"Probe{random.randint(100000, 999999)}"
TOKEN = os.environ.get("HYDRA_TOKEN", "local-development-token-32-bytes")
BASE = os.environ.get("HYDRA_URL", "http://127.0.0.1:8443")

results: list[dict] = []


def check(name: str, note: str = ""):
    def wrap(fn):
        results.append({"name": name, "fn": fn, "note": note})
        return fn
    return wrap


def run_all(client: HydraClient) -> list[dict]:
    out = []
    for spec in results:
        started = time.time()
        try:
            detail = spec["fn"](client) or ""
            status, err = "supported", ""
        except Unmeasured as e:
            status, err, detail = "unmeasured", str(e)[:300], ""
        except HydraError as e:
            status, err, detail = "unsupported", str(e)[:300], ""
        except Exception as e:  # a bug in the probe itself must not read as a server verdict
            status, err, detail = "probe_bug", f"{type(e).__name__}: {e}"[:300], ""
        out.append({"check": spec["name"], "status": status,
                    "ok": status == "supported", "detail": str(detail)[:200],
                    "error": err, "note": spec["note"],
                    "ms": int((time.time() - started) * 1000)})
    return out


# --- the checks -------------------------------------------------------------

@check("return_literal", "baseline: the node answers at all")
def _(c):
    return f"1 -> {c.query('RETURN 1 AS n').scalar()}"


@check("create_with_label_and_props", "can a typed node be written")
def _(c):
    c.query(f"CREATE (:{TAG} {{key: 'k1', n: 1, flag: true}})")
    return "created"


@check("match_by_property", "property lookup, the read path for exact keys")
def _(c):
    r = c.query(f"MATCH (n:{TAG} {{key: 'k1'}}) RETURN n.n AS n")
    return f"rows={len(r)} first={r.rows[0] if r.rows else None}"


@check("return_whole_node", "what a node looks like as a cell")
def _(c):
    r = c.query(f"MATCH (n:{TAG} {{key: 'k1'}}) RETURN n LIMIT 1")
    return json.dumps(r.raw_rows[:1])[:200]


@check("parameters_params_key", "does the API accept params, avoiding literal escaping")
def _(c):
    body = {"cell_id": c.cell, "query": "RETURN $x AS x", "params": {"x": 42}}
    return f"got {c._post(body).get('rows')}"


@check("parameters_parameters_key", "the other plausible spelling")
def _(c):
    body = {"cell_id": c.cell, "query": "RETURN $x AS x", "parameters": {"x": 42}}
    return f"got {c._post(body).get('rows')}"


@check("merge", "idempotent upsert; without it ingest needs match-then-create")
def _(c):
    c.query(f"MERGE (n:{TAG} {{key: 'm1'}}) RETURN n")
    c.query(f"MERGE (n:{TAG} {{key: 'm1'}}) RETURN n")
    r = c.query(f"MATCH (n:{TAG} {{key: 'm1'}}) RETURN count(n) AS c")
    return f"count after two merges = {r.scalar()} (must be 1)"


@check("set_property", "supersede marking needs SET")
def _(c):
    c.query(f"MATCH (n:{TAG} {{key: 'm1'}}) SET n.status = 'active'")
    return str(c.query(f"MATCH (n:{TAG} {{key: 'm1'}}) RETURN n.status AS s").scalar())


@check("relationship_create_with_props", "edges carry the meaning here")
def _(c):
    c.query(f"MATCH (a:{TAG} {{key:'k1'}}), (b:{TAG} {{key:'m1'}}) "
            f"CREATE (a)-[:SUPERSEDES {{at: 100}}]->(b)")
    r = c.query(f"MATCH (:{TAG})-[r:SUPERSEDES]->(:{TAG}) RETURN r.at AS at")
    return f"rows={len(r)} at={r.rows[0]['at'] if r.rows else None}"


@check("variable_length_path", "walking a supersede chain to its tip")
def _(c):
    r = c.query(f"MATCH (a:{TAG} {{key:'k1'}})-[:SUPERSEDES*1..3]->(b) "
                f"RETURN count(b) AS c")
    return f"reachable={r.scalar()}"


@check("optional_match", "absence has to be expressible, not inferred")
def _(c):
    r = c.query(f"MATCH (a:{TAG} {{key:'m1'}}) "
                f"OPTIONAL MATCH (a)-[:SUPERSEDES]->(b) RETURN b IS NULL AS tip")
    return f"tip={r.rows[0] if r.rows else None}"


@check("where_comparison", "as-of filtering is a range predicate")
def _(c):
    r = c.query(f"MATCH (n:{TAG}) WHERE n.n >= 1 RETURN count(n) AS c")
    return f"count={r.scalar()}"


@check("order_by_limit", "newest-first reads")
def _(c):
    r = c.query(f"MATCH (n:{TAG}) RETURN n.key AS k ORDER BY n.key DESC LIMIT 2")
    return str([x["k"] for x in r.rows])


@check("skip", "pagination fallback if cursors are awkward")
def _(c):
    r = c.query(f"MATCH (n:{TAG}) RETURN n.key AS k ORDER BY n.key SKIP 1 LIMIT 1")
    return str([x["k"] for x in r.rows])


@check("aggregation_collect", "one row per subject instead of per edge")
def _(c):
    r = c.query(f"MATCH (n:{TAG}) RETURN collect(n.key) AS keys")
    return str(r.rows[:1])[:200]


@check("unwind_inline_list", "batch writes without one round trip per row")
def _(c):
    c.query(f"UNWIND [1,2,3] AS i CREATE (:{TAG} {{key: 'u' + toString(i), n: i}})")
    r = c.query(f"MATCH (n:{TAG}) WHERE n.key STARTS WITH 'u' RETURN count(n) AS c")
    return f"created={r.scalar()}"


@check("string_functions", "toLower/split for entity resolution")
def _(c):
    r = c.query("RETURN toLower('AbC') AS a, size(split('a,b,c', ',')) AS b")
    return str(r.rows)


@check("coalesce", "defaults without a second query")
def _(c):
    return str(c.query("RETURN coalesce(null, 7) AS x").scalar())


@check("case_expression", "branching inside the query")
def _(c):
    r = c.query("RETURN CASE WHEN 1 < 2 THEN 'yes' ELSE 'no' END AS x")
    return str(r.rows)


@check("timestamp_function", "server-side clock, if any")
def _(c):
    return str(c.query("RETURN timestamp() AS t").scalar())


@check("multi_statement_semicolon", "schema setup in one call")
def _(c):
    c.query(f"CREATE (:{TAG} {{key:'ms1'}}); CREATE (:{TAG} {{key:'ms2'}})")
    r = c.query(f"MATCH (n:{TAG}) WHERE n.key STARTS WITH 'ms' RETURN count(n) AS c")
    return f"count={r.scalar()}"


@check("create_index", "property index DDL; the README says the planner uses them")
def _(c):
    c.query(f"CREATE INDEX FOR (n:{TAG}) ON (n.key)")
    return "accepted"


@check("create_constraint_unique", "uniqueness enforced by the database, not by code")
def _(c):
    c.query(f"CREATE CONSTRAINT FOR (n:{TAG}) REQUIRE n.key IS UNIQUE")
    return "accepted"


@check("show_indexes", "can the schema be inspected")
def _(c):
    r = c.query("SHOW INDEXES")
    return f"rows={len(r)} cols={r.columns}"


@check("long_string_property", "a full session transcript as one property")
def _(c):
    blob = "x" * 20000
    c.query(f"CREATE (:{TAG} {{key: 'blob', text: '{blob}'}})")
    r = c.query(f"MATCH (n:{TAG} {{key:'blob'}}) RETURN size(n.text) AS s")
    return f"stored size={r.scalar()} of 20000"


@check("unicode_and_quotes", "real transcripts contain both")
def _(c):
    from docket.hydra import lit
    val = lit("it's a “quote” — ü 日本 \\ end")
    c.query(f"CREATE (:{TAG} {{key: 'uni', text: {val}}})")
    r = c.query(f"MATCH (n:{TAG} {{key:'uni'}}) RETURN n.text AS t")
    return f"roundtrip={r.rows[0]['t'][:40]!r}"


@check("list_property", "can a node hold an array")
def _(c):
    c.query(f"CREATE (:{TAG} {{key: 'lst', tags: ['a','b']}})")
    r = c.query(f"MATCH (n:{TAG} {{key:'lst'}}) RETURN n.tags AS t")
    return f"tags={r.rows[0] if r.rows else None}"


@check("delete", "cleanup path; also tells us whether history can be erased")
def _(c):
    c.query(f"CREATE (:{TAG}Tmp {{key:'del'}})")
    c.query(f"MATCH (n:{TAG}Tmp) DELETE n")
    r = c.query(f"MATCH (n:{TAG}Tmp) RETURN count(n) AS c")
    return f"remaining={r.scalar()}"


@check("pagination_appears", "do large results page, and what does the cursor look like")
def _(c):
    c.query(f"UNWIND range(1, 2000) AS i CREATE (:{TAG}Big {{n: i}})")
    body = {"cell_id": c.cell, "query": f"MATCH (n:{TAG}Big) RETURN n.n AS n"}
    payload = c._post(body)
    cur = payload.get("next_cursor")
    return (f"rows_in_page_1={len(payload.get('rows') or [])} "
            f"next_cursor={json.dumps(cur)[:80]}")


@check("bookmark_accepted", "is the bookmark valid on the way back in")
def _(c):
    if not c.bookmark:
        raise Unmeasured("no bookmark was returned by any earlier check, "
                         "so nothing could be sent back")
    body = {"cell_id": c.cell, "query": "RETURN 1 AS n", "bookmark": c.bookmark}
    c._post(body)
    return "accepted"


def main() -> int:
    client = HydraClient(base_url=BASE, token=TOKEN, send_bookmark=False)
    print(f"probing {BASE} with label {TAG}")
    try:
        waited = client.wait_ready(seconds=60)
    except HydraError as e:
        print(f"node never became ready: {e}")
        return 2
    print(f"node ready after {waited:.1f}s\n")

    report = run_all(client)
    width = max(len(r["check"]) for r in report)
    marks = {"supported": "ok   ", "unsupported": "NO   ",
             "unmeasured": "?    ", "probe_bug": "BUG  "}
    for r in report:
        tail = r["detail"] if r["ok"] else r["error"]
        print(f"{marks[r['status']]}{r['check']:<{width}}  {tail}")

    counts = {}
    for r in report:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"of {len(report)} checks. unmeasured is not the same as unsupported.")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "probe-report.json")
    with open(out, "w") as fh:
        json.dump({"label": TAG, "base": BASE, "report": report}, fh, indent=2)
    print(f"full report written to {out}")
    print(f"\nleft behind in the graph: nodes labelled {TAG} and {TAG}Big. "
          f"They are inert and namespaced; the real schema uses its own labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

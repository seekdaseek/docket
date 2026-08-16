#!/usr/bin/env python3
"""probe2 - second-pass capability probe for HydraDB 0.1.0.

Probe 1 reported 25 unsupported of 30. Its own error text showed that many of
those verdicts were artefacts of how the check was written, not limits of the
server: a bare RETURN is rejected, the modern index DDL is rejected while the
legacy form was never tried, UNWIND was handed an inline list when the server
said it wants a parameter, and four checks failed on a setup CREATE rather than
on the thing they claimed to measure.

This probe fixes all four classes and answers the questions the schema actually
depends on, in dependency order:

  A  which CREATE form the server accepts        (nothing else can be measured
                                                  until one of them works)
  B  which relationship form it accepts
  C  what RETURN can project
  D  whether traversal works, and how far        (the SUPERSEDES chain lives here)
  E  whether WHERE can filter and compare        (as-of interval queries live here)
  F  whether parameters work, and under which key
  G  whether index/constraint DDL exists under legacy syntax
  H  strings, ordering, deletion

Rules carried over, and they are the point of the exercise:
  - a check that could not be performed reports SKIP, never NO
  - a check never inherits another check's setup failure
  - the server's own error text is recorded verbatim, because it usually names
    the supported alternative

  python3 probe2.py

Talks to the node over its raw HTTP API with the stdlib only. It deliberately
does not import the project client, so a bug in that client cannot colour the
measurements.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("HYDRA_URL", "http://127.0.0.1:8443")
TOKEN = os.environ.get("HYDRA_TOKEN", "local-development-token-32-bytes")
GRAPH = os.environ.get("HYDRA_GRAPH", "default")
NS = os.environ.get("HYDRA_NAMESPACE", "default")
CELL = os.environ.get("HYDRA_CELL", "cell-0")
TIMEOUT = float(os.environ.get("HYDRA_TIMEOUT", "30"))

TAG = f"P2{random.randint(100000, 999999)}"
OTHER = TAG + "B"

OK, NO, SKIP = "ok", "NO", "SKIP"


def post(query, params=None, params_key="params"):
    """Send one Cypher statement. Returns (status, payload_or_message).

    status 200 -> payload is the decoded response envelope
    status 4xx/5xx -> payload is the server's error message text
    status 0 -> transport failure, payload is the exception text
    """
    body = {"cell_id": CELL, "query": query}
    if params is not None:
        body[params_key] = params
    req = urllib.request.Request(
        f"{BASE}/v1/graphs/{GRAPH}/query",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
            "X-Graph-Namespace": NS,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw = r.read().decode("utf-8", errors="replace")
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw[:400]
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)["error"]["message"]
        except Exception:
            return e.code, raw[:400]
    except Exception as e:  # transport: refused, timeout, dns
        return 0, f"{type(e).__name__}: {e}"


def rows_of(payload):
    if isinstance(payload, dict):
        return payload.get("rows") or []
    return []


def brief(payload, limit=110):
    if isinstance(payload, dict):
        cols = payload.get("columns")
        rs = payload.get("rows")
        return f"columns={cols} rows={rs}"[:limit]
    return str(payload)[:limit]


REPORT = []


def record(group, name, verdict, detail, query=""):
    REPORT.append(
        {
            "group": group,
            "name": name,
            "verdict": verdict,
            "detail": str(detail)[:400],
            "query": query,
        }
    )
    print(f"{verdict:<4} {group}  {name:<32} {str(detail)[:150]}")


def run(group, name, query, params=None, params_key="params", want_rows=False):
    """Send one statement and record a verdict from the response alone."""
    status, payload = post(query, params, params_key)
    if status == 200:
        if want_rows and not rows_of(payload):
            record(group, name, OK, f"accepted but 0 rows | {brief(payload)}", query)
        else:
            record(group, name, OK, brief(payload), query)
        return True, payload
    if status == 0:
        record(group, name, SKIP, f"transport failure, nothing measured: {payload}", query)
        return False, payload
    record(group, name, NO, f"http {status}: {payload}", query)
    return False, payload


def skip(group, name, why, query=""):
    record(group, name, SKIP, why, query)


def main():
    print(f"probe2 -> {BASE}/v1/graphs/{GRAPH}/query   label {TAG}")

    # readiness: any structured answer from the node means it is up. a 4xx is
    # an answer. only transport failure means it is still waking.
    for attempt in range(30):
        status, payload = post(f"MATCH (n:{TAG}) RETURN n.k")
        if status != 0:
            print(f"node answering after {attempt}s (http {status})\n")
            break
        time.sleep(1)
    else:
        print("node never answered - transport failure for 30s. nothing measured.")
        sys.exit(1)

    # ---------------------------------------------------------------- group A
    # find a CREATE form that works. everything downstream needs one.
    creates = [
        ("create_bare_node", f"CREATE (n:{TAG})"),
        ("create_one_int_prop", f"CREATE (n:{TAG} {{k: 1}})"),
        ("create_one_string_prop", f"CREATE (n:{TAG} {{k: 'a'}})"),
        ("create_two_props", f"CREATE (n:{TAG} {{k: 'b', num: 2}})"),
        ("create_three_props", f"CREATE (n:{TAG} {{k: 'c', num: 3, flag: true}})"),
        ("create_with_return", f"CREATE (n:{TAG} {{k: 'd'}}) RETURN n.k"),
    ]
    working_create = None
    for name, q in creates:
        ok, _ = run("A", name, q)
        if ok and working_create is None and "RETURN" not in q:
            working_create = q

    def setup(statements):
        """Run setup statements. Returns None on success, else the error text."""
        for s in statements:
            status, payload = post(s)
            if status != 200:
                return f"setup failed ({s[:60]}...): http {status}: {payload}"
        return None

    if working_create is None:
        blocked = "no CREATE form was accepted, so nothing that needs data could be measured"

    # seed a small known graph using whichever create form worked
    seeded = False
    if working_create is not None:
        tmpl = None
        for name, q in creates:
            if q == working_create:
                tmpl = q
                break
        # prefer the richest accepted form for seeding
        rich = [q for n, q in creates if "num" in q and "RETURN" not in q]
        seed_stmts = []
        for i in (1, 2, 3):
            if rich:
                seed_stmts.append(f"CREATE (n:{OTHER} {{k: 'n{i}', num: {i}}})")
            else:
                seed_stmts.append(f"CREATE (n:{OTHER})")
        err = setup(seed_stmts)
        seeded = err is None
        if err:
            print(f"\n! seeding failed: {err}\n")

    # ---------------------------------------------------------------- group B
    if not seeded:
        for n in ("edge_no_props", "edge_with_props", "edge_two_hop_one_statement"):
            skip("B", n, "no seed data, nothing measured")
    else:
        run(
            "B",
            "edge_no_props",
            f"MATCH (a:{OTHER} {{k: 'n1'}}), (b:{OTHER} {{k: 'n2'}}) CREATE (a)-[:LINK]->(b)",
        )
        run(
            "B",
            "edge_with_props",
            f"MATCH (a:{OTHER} {{k: 'n2'}}), (b:{OTHER} {{k: 'n3'}}) CREATE (a)-[:LINK {{w: 1}}]->(b)",
        )
        run(
            "B",
            "edge_two_hop_one_statement",
            f"MATCH (a:{OTHER} {{k: 'n1'}}), (b:{OTHER} {{k: 'n2'}}), (c:{OTHER} {{k: 'n3'}}) "
            f"CREATE (a)-[:CHAIN]->(b)-[:CHAIN]->(c)",
        )

    # ---------------------------------------------------------------- group C
    if not seeded:
        for n in (
            "return_property",
            "return_count_star",
            "return_two_properties",
            "return_alias",
            "return_distinct",
        ):
            skip("C", n, "no seed data, nothing measured")
    else:
        run("C", "return_property", f"MATCH (n:{OTHER}) RETURN n.k", want_rows=True)
        run("C", "return_count_star", f"MATCH (n:{OTHER}) RETURN count(*)", want_rows=True)
        run("C", "return_two_properties", f"MATCH (n:{OTHER}) RETURN n.k, n.num", want_rows=True)
        run("C", "return_alias", f"MATCH (n:{OTHER}) RETURN n.k AS key", want_rows=True)
        run("C", "return_distinct", f"MATCH (n:{OTHER}) RETURN DISTINCT n.k", want_rows=True)

    # ---------------------------------------------------------------- group D
    # the SUPERSEDES chain lives or dies here.
    if not seeded:
        for n in ("hop_one", "hop_two_fixed", "var_length_1_3", "var_length_unbounded", "hop_reverse"):
            skip("D", n, "no seed data, nothing measured")
    else:
        run("D", "hop_one", f"MATCH (a:{OTHER})-[:LINK]->(b:{OTHER}) RETURN b.k")
        run("D", "hop_two_fixed", f"MATCH (a:{OTHER})-[:LINK]->()-[:LINK]->(c:{OTHER}) RETURN c.k")
        run("D", "var_length_1_3", f"MATCH (a:{OTHER})-[:LINK*1..3]->(c:{OTHER}) RETURN c.k")
        run("D", "var_length_unbounded", f"MATCH (a:{OTHER})-[:LINK*]->(c:{OTHER}) RETURN c.k")
        run("D", "hop_reverse", f"MATCH (a:{OTHER})<-[:LINK]-(b:{OTHER}) RETURN b.k")

    # ---------------------------------------------------------------- group E
    if not seeded:
        for n in (
            "where_equals_string",
            "where_greater_than",
            "where_interval_and",
            "where_starts_with",
            "where_in_list",
            "inline_property_match",
        ):
            skip("E", n, "no seed data, nothing measured")
    else:
        run("E", "inline_property_match", f"MATCH (n:{OTHER} {{k: 'n1'}}) RETURN n.k", want_rows=True)
        run("E", "where_equals_string", f"MATCH (n:{OTHER}) WHERE n.k = 'n1' RETURN n.k", want_rows=True)
        run("E", "where_greater_than", f"MATCH (n:{OTHER}) WHERE n.num > 1 RETURN n.num", want_rows=True)
        run(
            "E",
            "where_interval_and",
            f"MATCH (n:{OTHER}) WHERE n.num >= 1 AND n.num <= 2 RETURN n.num",
            want_rows=True,
        )
        run("E", "where_starts_with", f"MATCH (n:{OTHER}) WHERE n.k STARTS WITH 'n' RETURN n.k")
        run("E", "where_in_list", f"MATCH (n:{OTHER}) WHERE n.k IN ['n1','n2'] RETURN n.k")

    # ---------------------------------------------------------------- group F
    # the server said UNWIND batch input must be a parameter, so parameters
    # exist in some shape. find the key it accepts.
    if not seeded:
        for n in ("param_key_params", "param_key_parameters", "unwind_param_batch"):
            skip("F", n, "no seed data, nothing measured")
    else:
        run(
            "F",
            "param_key_params",
            f"MATCH (n:{OTHER}) WHERE n.k = $k RETURN n.k",
            params={"k": "n1"},
            params_key="params",
        )
        run(
            "F",
            "param_key_parameters",
            f"MATCH (n:{OTHER}) WHERE n.k = $k RETURN n.k",
            params={"k": "n1"},
            params_key="parameters",
        )
        run(
            "F",
            "unwind_param_batch",
            f"UNWIND $batch AS row MATCH (n:{OTHER}) WHERE n.k = row RETURN n.k",
            params={"batch": ["n1", "n2"]},
            params_key="params",
        )

    # ---------------------------------------------------------------- group G
    # probe 1 used the modern DDL. the parse errors named the legacy form.
    run("G", "index_legacy_syntax", f"CREATE INDEX ON :{OTHER}(k)")
    run("G", "index_modern_syntax", f"CREATE INDEX idx{TAG} FOR (n:{OTHER}) ON (n.k)")
    run("G", "constraint_legacy_syntax", f"CREATE CONSTRAINT ON (n:{OTHER}) ASSERT n.k IS UNIQUE")
    run("G", "start_clause_parses", f"MATCH (n:{OTHER}) RETURN count(*)")

    # ---------------------------------------------------------------- group H
    if not seeded:
        for n in ("long_string_1k", "long_string_16k", "unicode_and_quote", "order_by_property", "set_then_read", "delete_node"):
            skip("H", n, "no seed data, nothing measured")
    else:
        run("H", "long_string_1k", f"CREATE (n:{OTHER} {{k: 's1k', body: '{'x' * 1000}'}})")
        run("H", "long_string_16k", f"CREATE (n:{OTHER} {{k: 's16k', body: '{'x' * 16000}'}})")
        run("H", "unicode_and_quote", f"CREATE (n:{OTHER} {{k: 'uni', body: 'caf\\u00e9 \\u2014 it\\'s fine'}})")
        run("H", "order_by_property", f"MATCH (n:{OTHER}) RETURN n.num ORDER BY n.num DESC", want_rows=True)
        err = setup([f"MATCH (n:{OTHER}) WHERE n.k = 'n1' SET n.touched = 1"])
        if err:
            skip("H", "set_then_read", err)
        else:
            run("H", "set_then_read", f"MATCH (n:{OTHER}) WHERE n.touched = 1 RETURN n.k", want_rows=True)
        run("H", "delete_node", f"MATCH (n:{OTHER}) WHERE n.k = 'n3' DELETE n")

    # ---------------------------------------------------------------- summary
    counts = {OK: 0, NO: 0, SKIP: 0}
    for r in REPORT:
        counts[r["verdict"]] += 1
    print(
        f"\nsupported={counts[OK]}  unsupported={counts[NO]}  unmeasured={counts[SKIP]}"
        f"  of {len(REPORT)} checks."
    )
    print("unmeasured is not the same as unsupported.")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe2-report.json")
    with open(out, "w") as f:
        json.dump(
            {"base": BASE, "graph": GRAPH, "label": TAG, "checks": REPORT}, f, indent=1
        )
    print(f"full report written to {out}")
    print(f"left behind in the graph: nodes labelled {TAG} and {OTHER}. inert, namespaced.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""probe3 - capability probe built from HydraDB's own documentation.

Probes 1 and 2 both failed the same way: they asked the server Neo4j-shaped
questions. The README's only write example is a one-hop edge pattern

    CREATE (a {id: 1})-[:FOLLOWS]->(b {id: 2})

which is exactly what the server's error text kept saying, and exactly the form
neither earlier probe tried. Standalone node creation appears not to exist.

Everything here is either a form the README shows, a capability the README
claims ("bounded variable-length paths, OPTIONAL MATCH, UNION, batched UNWIND
writes", the algo.*paths procedures, causal/strong consistency), or the
specific unknown the schema depends on. Claimed is not measured; that is the
whole point of running it.

  python3 probe3.py

Rules unchanged: a check that could not be performed reports SKIP, never NO,
and no check inherits another check's setup failure.
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
ADMIN = os.environ.get("HYDRA_ADMIN", "http://127.0.0.1:9090")
TOKEN = os.environ.get("HYDRA_TOKEN", "local-development-token-32-bytes")
GRAPH = os.environ.get("HYDRA_GRAPH", "default")
NS = os.environ.get("HYDRA_NAMESPACE", "default")
CELL = os.environ.get("HYDRA_CELL", "cell-0")
TIMEOUT = float(os.environ.get("HYDRA_TIMEOUT", "30"))

TAG = f"P3{random.randint(100000, 999999)}"
OK, NO, SKIP = "ok", "NO", "SKIP"
REPORT = []


def post(query, params=None, params_key="params", consistency=None):
    body = {"cell_id": CELL, "query": query}
    if params is not None:
        body[params_key] = params
    if consistency is not None:
        body["consistency"] = consistency
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
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def brief(payload, limit=120):
    if isinstance(payload, dict):
        return f"columns={payload.get('columns')} rows={payload.get('rows')}"[:limit]
    return str(payload)[:limit]


def record(group, name, verdict, detail, query=""):
    REPORT.append({"group": group, "name": name, "verdict": verdict,
                   "detail": str(detail)[:400], "query": query})
    print(f"{verdict:<4} {group}  {name:<30} {str(detail)[:145]}")


def run(group, name, query, params=None, params_key="params", consistency=None):
    status, payload = post(query, params, params_key, consistency)
    if status == 200:
        record(group, name, OK, brief(payload), query)
        return True, payload
    if status == 0:
        record(group, name, SKIP, f"transport failure, nothing measured: {payload}", query)
        return False, payload
    record(group, name, NO, f"http {status}: {payload}", query)
    return False, payload


def skip(group, name, why):
    record(group, name, SKIP, why)


def main():
    print(f"probe3 -> {BASE}/v1/graphs/{GRAPH}/query   label {TAG}")

    # readiness: the admin port has a real endpoint for this. fall back to the
    # query port, where any structured answer - including a 4xx - means up.
    ready = False
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"{ADMIN}/readyz", timeout=5) as r:
                if r.status == 200:
                    print("node ready (/readyz on the admin port)\n")
                    ready = True
                    break
        except Exception:
            pass
        status, _ = post(f"MATCH (n:{TAG}) RETURN n.uid")
        if status != 0:
            print(f"node answering on the query port (http {status})\n")
            ready = True
            break
        time.sleep(1)
    if not ready:
        print("node never answered. nothing measured.")
        sys.exit(1)

    # ------------------------------------------------------------- group A
    # writes. the README's form goes first because it is the only one the
    # project itself documents.
    a1, _ = run("A", "doc_form_anonymous_nodes",
                "CREATE (a {id: 90001})-[:LINK]->(b {id: 90002})")
    a2, _ = run("A", "edge_pattern_with_labels",
                f"CREATE (a:{TAG} {{uid: 'a1'}})-[:LINK]->(b:{TAG} {{uid: 'b1'}})")
    run("A", "edge_pattern_with_edge_props",
        f"CREATE (a:{TAG} {{uid: 'c1'}})-[:LINK {{w: 1}}]->(b:{TAG} {{uid: 'c2'}})")
    run("A", "node_only_create_confirm",
        f"CREATE (n:{TAG} {{uid: 'lonely'}})")
    run("A", "two_hop_create_confirm",
        f"CREATE (a:{TAG} {{uid: 'x1'}})-[:LINK]->(b:{TAG} {{uid: 'x2'}})-[:LINK]->(c:{TAG} {{uid: 'x3'}})")

    seeded = a2
    if not seeded:
        skip("A", "match_then_create", "no labelled seed, nothing measured")
        skip("A", "match_then_create_edge_props", "no labelled seed, nothing measured")
        skip("A", "unwind_batch_create", "no labelled seed, nothing measured")
    else:
        # the question the whole ingest depends on: can a second edge be
        # attached to a node that already exists, or does every CREATE make
        # new nodes?
        run("A", "match_then_create",
            f"MATCH (a:{TAG} {{uid: 'a1'}}), (b:{TAG} {{uid: 'b1'}}) CREATE (a)-[:CHAIN]->(b)")
        run("A", "match_then_create_edge_props",
            f"MATCH (a:{TAG} {{uid: 'a1'}}), (b:{TAG} {{uid: 'b1'}}) CREATE (a)-[:WEIGHTED {{w: 2}}]->(b)")
        run("A", "unwind_batch_create",
            f"UNWIND $rows AS r CREATE (a:{TAG} {{uid: r.a}})-[:LINK]->(b:{TAG} {{uid: r.b}})",
            params={"rows": [{"a": "u1", "b": "u2"}, {"a": "u2", "b": "u3"}]})

    # seed a chain for traversal, one edge per statement
    chain = False
    if seeded:
        errs = []
        for i in (1, 2, 3):
            st, pl = post(
                f"CREATE (a:{TAG} {{uid: 'k{i}', num: {i}, name: 'node{i}'}})"
                f"-[:NEXT]->(b:{TAG} {{uid: 'k{i + 1}', num: {i + 1}, name: 'node{i + 1}'}})"
            )
            if st != 200:
                errs.append(f"http {st}: {pl}")
        chain = not errs
        if errs:
            print(f"\n! chain seeding failed: {errs[0]}\n")

    # ------------------------------------------------------------- group B
    if not seeded:
        for n in ("return_property", "return_count_star", "return_two_props",
                  "return_alias", "return_distinct", "order_by", "skip_limit"):
            skip("B", n, "no seed data, nothing measured")
    else:
        run("B", "return_property", f"MATCH (n:{TAG}) RETURN n.uid")
        run("B", "return_count_star", f"MATCH (n:{TAG}) RETURN count(*)")
        run("B", "return_two_props", f"MATCH (n:{TAG}) RETURN n.uid, n.num")
        run("B", "return_alias", f"MATCH (n:{TAG}) RETURN n.uid AS u")
        run("B", "return_distinct", f"MATCH (n:{TAG}) RETURN DISTINCT n.uid")
        run("B", "order_by", f"MATCH (n:{TAG}) RETURN n.num ORDER BY n.num DESC")
        run("B", "skip_limit", f"MATCH (n:{TAG}) RETURN n.uid SKIP 1 LIMIT 2")

    # ------------------------------------------------------------- group C
    # traversal. the SUPERSEDES chain lives here. the README claims bounded
    # variable-length paths; this measures them.
    if not chain:
        for n in ("hop_one", "hop_two_fixed", "var_len_1_3", "var_len_1_open",
                  "var_len_unbounded", "hop_reverse", "optional_match"):
            skip("C", n, "no chain seeded, nothing measured")
    else:
        run("C", "hop_one", f"MATCH (a:{TAG})-[:NEXT]->(b:{TAG}) RETURN b.uid")
        run("C", "hop_two_fixed", f"MATCH (a:{TAG})-[:NEXT]->()-[:NEXT]->(c:{TAG}) RETURN c.uid")
        run("C", "var_len_1_3", f"MATCH (a:{TAG} {{uid: 'k1'}})-[:NEXT*1..3]->(c:{TAG}) RETURN c.uid")
        run("C", "var_len_1_open", f"MATCH (a:{TAG} {{uid: 'k1'}})-[:NEXT*1..]->(c:{TAG}) RETURN c.uid")
        run("C", "var_len_unbounded", f"MATCH (a:{TAG} {{uid: 'k1'}})-[:NEXT*]->(c:{TAG}) RETURN c.uid")
        run("C", "hop_reverse", f"MATCH (a:{TAG})<-[:NEXT]-(b:{TAG}) RETURN b.uid")
        run("C", "optional_match",
            f"MATCH (a:{TAG} {{uid: 'k1'}}) OPTIONAL MATCH (a)-[:NEXT]->(b:{TAG}) RETURN b.uid")

    # ------------------------------------------------------------- group D
    if not seeded:
        for n in ("where_equals", "where_gt", "where_interval", "where_starts_with",
                  "where_in_list", "inline_prop_match", "union"):
            skip("D", n, "no seed data, nothing measured")
    else:
        run("D", "inline_prop_match", f"MATCH (n:{TAG} {{uid: 'a1'}}) RETURN n.uid")
        run("D", "where_equals", f"MATCH (n:{TAG}) WHERE n.uid = 'a1' RETURN n.uid")
        run("D", "where_gt", f"MATCH (n:{TAG}) WHERE n.num > 1 RETURN n.num")
        run("D", "where_interval", f"MATCH (n:{TAG}) WHERE n.num >= 1 AND n.num <= 2 RETURN n.num")
        run("D", "where_starts_with", f"MATCH (n:{TAG}) WHERE n.name STARTS WITH 'node' RETURN n.name")
        run("D", "where_in_list", f"MATCH (n:{TAG}) WHERE n.uid IN ['a1','b1'] RETURN n.uid")
        run("D", "union",
            f"MATCH (n:{TAG} {{uid: 'a1'}}) RETURN n.uid UNION MATCH (n:{TAG} {{uid: 'b1'}}) RETURN n.uid")

    # ------------------------------------------------------------- group E
    if not seeded:
        for n in ("param_key_params", "param_key_parameters", "unwind_param_read"):
            skip("E", n, "no seed data, nothing measured")
    else:
        run("E", "param_key_params", f"MATCH (n:{TAG}) WHERE n.uid = $uid RETURN n.uid",
            params={"uid": "a1"}, params_key="params")
        run("E", "param_key_parameters", f"MATCH (n:{TAG}) WHERE n.uid = $uid RETURN n.uid",
            params={"uid": "a1"}, params_key="parameters")
        run("E", "unwind_param_read",
            f"UNWIND $ids AS i MATCH (n:{TAG}) WHERE n.uid = i RETURN n.uid",
            params={"ids": ["a1", "b1"]})

    # ------------------------------------------------------------- group F
    # native path procedures. if these work they are the strongest possible
    # answer to "HydraDB has to do real work in your project".
    if not chain:
        for n in ("sspaths_yield_path", "sspaths_return_length", "sppaths", "mspaths"):
            skip("F", n, "no chain seeded, nothing measured")
    else:
        run("F", "sspaths_yield_path",
            f"CALL algo.SSpaths({{sourceLabel: '{TAG}', sourceProperty: 'uid', "
            f"sourceValues: ['k1'], relTypes: ['NEXT'], relDirection: 'out', "
            f"maxLen: 3, pathCount: 5, resultLimit: 10}}) YIELD path RETURN path")
        run("F", "sspaths_return_length",
            f"CALL algo.SSpaths({{sourceLabel: '{TAG}', sourceProperty: 'uid', "
            f"sourceValues: ['k1'], relTypes: ['NEXT'], relDirection: 'out', "
            f"maxLen: 3, pathCount: 5, resultLimit: 10}}) YIELD path RETURN length(path)")
        run("F", "sppaths",
            f"CALL algo.SPpaths({{sourceLabel: '{TAG}', sourceProperty: 'uid', "
            f"sourceValues: ['k1'], targetValues: ['k4'], relTypes: ['NEXT'], "
            f"relDirection: 'out', maxLen: 4, pathCount: 5, resultLimit: 10}}) "
            f"YIELD path RETURN path")
        run("F", "mspaths",
            f"CALL algo.MSpaths({{sourceLabel: '{TAG}', sourceProperty: 'uid', "
            f"sourceValues: ['k1','k2'], targetValues: ['k3','k4'], pairwise: true, "
            f"relTypes: ['NEXT'], relDirection: 'out', maxLen: 3, pathCount: 5, "
            f"resultLimit: 10}}) YIELD path RETURN path")

    # ------------------------------------------------------------- group G
    if not seeded:
        for n in ("consistency_causal", "consistency_strong"):
            skip("G", n, "no seed data, nothing measured")
    else:
        run("G", "consistency_causal", f"MATCH (n:{TAG}) RETURN count(*)", consistency="causal")
        run("G", "consistency_strong", f"MATCH (n:{TAG}) RETURN count(*)", consistency="strong")

    # ------------------------------------------------------------- group H
    if not seeded:
        for n in ("set_then_read", "delete_edge", "long_string_16k", "unicode_and_quote"):
            skip("H", n, "no seed data, nothing measured")
    else:
        st, pl = post(f"MATCH (n:{TAG}) WHERE n.uid = 'a1' SET n.touched = 1")
        if st != 200:
            skip("H", "set_then_read", f"SET failed: http {st}: {pl}")
        else:
            run("H", "set_then_read", f"MATCH (n:{TAG}) WHERE n.touched = 1 RETURN n.uid")
        run("H", "delete_edge", f"MATCH (a:{TAG})-[r:WEIGHTED]->(b:{TAG}) DELETE r")
        run("H", "long_string_16k",
            f"CREATE (a:{TAG} {{uid: 's16k', body: '{'x' * 16000}'}})-[:LINK]->(b:{TAG} {{uid: 's16kb'}})")
        run("H", "unicode_and_quote",
            f"CREATE (a:{TAG} {{uid: 'uni', body: 'caf\\u00e9 \\u2014 it\\'s fine'}})"
            f"-[:LINK]->(b:{TAG} {{uid: 'unib'}})")

    counts = {OK: 0, NO: 0, SKIP: 0}
    for r in REPORT:
        counts[r["verdict"]] += 1
    print(f"\nsupported={counts[OK]}  unsupported={counts[NO]}  unmeasured={counts[SKIP]}"
          f"  of {len(REPORT)} checks.")
    print("unmeasured is not the same as unsupported.")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe3-report.json")
    with open(out, "w") as f:
        json.dump({"base": BASE, "graph": GRAPH, "label": TAG, "checks": REPORT}, f, indent=1)
    print(f"full report written to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""probe4 - the CREATE matrix, one variable at a time, then everything downstream.

probe3 established that the README's own form works:

    CREATE (a {id: 1})-[:LINK]->(b {id: 2})        ok

and that this fails:

    CREATE (a:LABEL {uid: 'a1'})-[:LINK]->(b:LABEL {uid: 'b1'})
    -> CREATE requires source id

Two things changed between them at once: a label was added, and the property
was renamed from `id` to `uid`. The error names an id, which points at the
property, but pointing is not measuring. Group A below changes exactly one
thing per check so the cause is isolated rather than inferred.

Group B is the question the whole ingest design rests on: when the same `id`
is written twice, does the node get reused or duplicated? If ids are the
identity key, dedupe is free and the client computes deterministic ids. If not,
every ingest needs a read-before-write round trip.

  python3 probe4.py

A check that could not be performed reports SKIP, never NO.
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

# a fresh id range per run: earlier probes left nodes behind, and a count that
# starts at an unknown number cannot answer the identity question.
B = random.randint(1000000, 9000000)
LBL = f"P4{random.randint(100000, 999999)}"
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
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {TOKEN}",
                 "X-Graph-Namespace": NS},
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


def brief(p, limit=120):
    if isinstance(p, dict):
        return f"columns={p.get('columns')} rows={p.get('rows')}"[:limit]
    return str(p)[:limit]


def first_value(payload):
    """First cell of the first row, unwrapped. None if there is no row."""
    if not isinstance(payload, dict):
        return None
    rows = payload.get("rows") or []
    if not rows or not rows[0]:
        return None
    cell = rows[0][0]
    return cell.get("value") if isinstance(cell, dict) else cell


def record(group, name, verdict, detail, query=""):
    REPORT.append({"group": group, "name": name, "verdict": verdict,
                   "detail": str(detail)[:400], "query": query})
    print(f"{verdict:<4} {group}  {name:<32} {str(detail)[:140]}")


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


def count_where(pattern):
    """count(*) for a MATCH pattern. Returns (ok, count_or_error)."""
    status, payload = post(f"MATCH {pattern} RETURN count(*)")
    if status != 200:
        return False, f"http {status}: {payload}"
    return True, first_value(payload)


def main():
    print(f"probe4 -> {BASE}/v1/graphs/{GRAPH}/query   id base {B}   label {LBL}")
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"{ADMIN}/readyz", timeout=5) as r:
                if r.status == 200:
                    print("node ready (/readyz on the admin port)\n")
                    break
        except Exception:
            pass
        if post("MATCH (n {id: -1}) RETURN n.id")[0] != 0:
            print("node answering on the query port\n")
            break
        time.sleep(1)
    else:
        print("node never answered. nothing measured.")
        sys.exit(1)

    # ---------------------------------------------------------------- group A
    # the matrix. exactly one difference per line from the known-good baseline.
    run("A", "baseline_anon_int_id",
        f"CREATE (a {{id: {B}}})-[:LINK]->(b {{id: {B + 1}}})")
    run("A", "plus_label_only",
        f"CREATE (a:{LBL} {{id: {B + 2}}})-[:LINK]->(b:{LBL} {{id: {B + 3}}})")
    run("A", "id_renamed_to_uid_no_label",
        f"CREATE (a {{uid: 'u{B}'}})-[:LINK]->(b {{uid: 'u{B + 1}'}})")
    run("A", "id_is_a_string",
        f"CREATE (a {{id: 's{B}'}})-[:LINK]->(b {{id: 's{B + 1}'}})")
    run("A", "id_plus_extra_props",
        f"CREATE (a {{id: {B + 4}, name: 'alpha', num: 1}})-[:LINK]->(b {{id: {B + 5}, name: 'beta', num: 2}})")
    run("A", "label_and_id_and_props",
        f"CREATE (a:{LBL} {{id: {B + 6}, name: 'gamma'}})-[:LINK]->(b:{LBL} {{id: {B + 7}, name: 'delta'}})")
    run("A", "edge_props_with_id",
        f"CREATE (a {{id: {B + 8}}})-[:WEIGHTED {{w: 1}}]->(b {{id: {B + 9}}})")
    run("A", "target_has_no_id",
        f"CREATE (a {{id: {B + 10}}})-[:LINK]->(b {{name: 'no id here'}})")
    run("A", "typed_rel_and_id_reverse_direction",
        f"CREATE (a {{id: {B + 11}}})<-[:LINK]-(b {{id: {B + 12}}})")

    # ---------------------------------------------------------------- group B
    # identity. does writing the same id twice reuse the node or duplicate it?
    ok_before, before = count_where(f"(n {{id: {B}}})")
    if not ok_before:
        skip("B", "id_is_the_identity_key", f"count failed, nothing measured: {before}")
        skip("B", "edge_count_after_repeat", "count failed, nothing measured")
    else:
        st, pl = post(f"CREATE (a {{id: {B}}})-[:LINK]->(b {{id: {B + 1}}})")
        if st != 200:
            skip("B", "id_is_the_identity_key", f"repeat write failed: http {st}: {pl}")
            skip("B", "edge_count_after_repeat", "repeat write failed")
        else:
            ok_after, after = count_where(f"(n {{id: {B}}})")
            if not ok_after:
                skip("B", "id_is_the_identity_key", f"second count failed: {after}")
            elif before == after:
                record("B", "id_is_the_identity_key", OK,
                       f"node count unchanged at {after} across two identical CREATEs "
                       f"-> id IS the node key, client-side dedupe is free")
            else:
                record("B", "id_is_the_identity_key", NO,
                       f"node count went {before} -> {after} -> ids DUPLICATE, "
                       f"every ingest write needs a read-before-write")
            ok_e, edges = count_where(f"(a {{id: {B}}})-[:LINK]->(b)")
            record("B", "edge_count_after_repeat", OK if ok_e else SKIP,
                   f"edges from that node: {edges}")

    seeded, _ = run("B", "match_then_create_by_id",
                    f"MATCH (a {{id: {B}}}), (b {{id: {B + 1}}}) CREATE (a)-[:CHAIN]->(b)")
    run("B", "unwind_batch_create",
        f"UNWIND $rows AS r CREATE (a {{id: r.a}})-[:LINK]->(b {{id: r.b}})",
        params={"rows": [{"a": B + 20, "b": B + 21}, {"a": B + 21, "b": B + 22}]})

    # a chain to traverse: 1 -> 2 -> 3 -> 4
    chain_errs = []
    for i in range(3):
        st, pl = post(f"CREATE (a {{id: {B + 30 + i}, num: {i + 1}, name: 'node{i + 1}'}})"
                      f"-[:NEXT]->(b {{id: {B + 31 + i}, num: {i + 2}, name: 'node{i + 2}'}})")
        if st != 200:
            chain_errs.append(f"http {st}: {pl}")
    chain = not chain_errs
    if chain_errs:
        print(f"\n! chain seeding failed: {chain_errs[0]}\n")

    # ---------------------------------------------------------------- group C
    run("C", "match_by_property", f"MATCH (n {{id: {B}}}) RETURN n.id")
    run("C", "match_by_label", f"MATCH (n:{LBL}) RETURN n.id")
    run("C", "return_count_star", f"MATCH (n {{id: {B}}}) RETURN count(*)")
    run("C", "return_two_props", f"MATCH (n {{id: {B + 4}}}) RETURN n.name, n.num")
    run("C", "return_alias", f"MATCH (n {{id: {B}}}) RETURN n.id AS ident")
    run("C", "order_by_limit", f"MATCH (n {{name: 'node1'}}) RETURN n.num ORDER BY n.num DESC LIMIT 2")

    # ---------------------------------------------------------------- group D
    if not chain:
        for n in ("hop_one", "hop_two_fixed", "var_len_1_3", "var_len_1_open",
                  "var_len_unbounded", "hop_reverse", "optional_match"):
            skip("D", n, "no chain seeded, nothing measured")
    else:
        run("D", "hop_one", f"MATCH (a {{id: {B + 30}}})-[:NEXT]->(b) RETURN b.id")
        run("D", "hop_two_fixed", f"MATCH (a {{id: {B + 30}}})-[:NEXT]->()-[:NEXT]->(c) RETURN c.id")
        run("D", "var_len_1_3", f"MATCH (a {{id: {B + 30}}})-[:NEXT*1..3]->(c) RETURN c.id")
        run("D", "var_len_1_open", f"MATCH (a {{id: {B + 30}}})-[:NEXT*1..]->(c) RETURN c.id")
        run("D", "var_len_unbounded", f"MATCH (a {{id: {B + 30}}})-[:NEXT*]->(c) RETURN c.id")
        run("D", "hop_reverse", f"MATCH (a {{id: {B + 33}}})<-[:NEXT]-(b) RETURN b.id")
        run("D", "optional_match",
            f"MATCH (a {{id: {B + 30}}}) OPTIONAL MATCH (a)-[:NEXT]->(b) RETURN b.id")

    # ---------------------------------------------------------------- group E
    run("E", "where_equals", f"MATCH (n) WHERE n.id = {B} RETURN n.id")
    run("E", "where_gt", f"MATCH (n) WHERE n.num > 1 RETURN n.num")
    run("E", "where_interval", f"MATCH (n) WHERE n.num >= 1 AND n.num <= 2 RETURN n.num")
    run("E", "where_starts_with", f"MATCH (n) WHERE n.name STARTS WITH 'node' RETURN n.name")
    run("E", "where_in_list", f"MATCH (n) WHERE n.id IN [{B}, {B + 1}] RETURN n.id")
    run("E", "union", f"MATCH (n {{id: {B}}}) RETURN n.id UNION MATCH (n {{id: {B + 1}}}) RETURN n.id")

    # ---------------------------------------------------------------- group F
    run("F", "param_key_params", "MATCH (n) WHERE n.id = $wanted RETURN n.id",
        params={"wanted": B}, params_key="params")
    run("F", "param_key_parameters", "MATCH (n) WHERE n.id = $wanted RETURN n.id",
        params={"wanted": B}, params_key="parameters")
    run("F", "unwind_param_read", "UNWIND $ids AS i MATCH (n) WHERE n.id = i RETURN n.id",
        params={"ids": [B, B + 1]})

    # ---------------------------------------------------------------- group G
    # the native path procedures. these are the strongest use of the engine.
    if not chain:
        for n in ("sspaths_yield_path", "sspaths_return_length", "sppaths", "mspaths"):
            skip("G", n, "no chain seeded, nothing measured")
    else:
        src = f"sourceProperty: 'id', sourceValues: [{B + 30}]"
        run("G", "sspaths_yield_path",
            f"CALL algo.SSpaths({{{src}, relTypes: ['NEXT'], relDirection: 'out', "
            f"maxLen: 3, pathCount: 5, resultLimit: 10}}) YIELD path RETURN path")
        run("G", "sspaths_return_length",
            f"CALL algo.SSpaths({{{src}, relTypes: ['NEXT'], relDirection: 'out', "
            f"maxLen: 3, pathCount: 5, resultLimit: 10}}) YIELD path RETURN length(path)")
        run("G", "sppaths",
            f"CALL algo.SPpaths({{{src}, targetValues: [{B + 33}], relTypes: ['NEXT'], "
            f"relDirection: 'out', maxLen: 4, pathCount: 5, resultLimit: 10}}) "
            f"YIELD path RETURN path")
        run("G", "mspaths",
            f"CALL algo.MSpaths({{sourceProperty: 'id', sourceValues: [{B + 30}, {B + 31}], "
            f"targetValues: [{B + 32}, {B + 33}], pairwise: true, relTypes: ['NEXT'], "
            f"relDirection: 'out', maxLen: 3, pathCount: 5, resultLimit: 10}}) "
            f"YIELD path RETURN path")

    # ---------------------------------------------------------------- group H
    run("H", "consistency_causal", f"MATCH (n {{id: {B}}}) RETURN count(*)", consistency="causal")
    run("H", "consistency_strong", f"MATCH (n {{id: {B}}}) RETURN count(*)", consistency="strong")
    st, pl = post(f"MATCH (n {{id: {B}}}) SET n.touched = 1")
    if st != 200:
        skip("H", "set_then_read", f"SET failed: http {st}: {pl}")
    else:
        run("H", "set_then_read", f"MATCH (n) WHERE n.touched = 1 RETURN n.id")
    run("H", "delete_edge", f"MATCH (a {{id: {B + 8}}})-[r:WEIGHTED]->(b) DELETE r")
    run("H", "long_string_16k",
        f"CREATE (a {{id: {B + 40}, body: '{'x' * 16000}'}})-[:LINK]->(b {{id: {B + 41}}})")
    run("H", "unicode_and_quote",
        f"CREATE (a {{id: {B + 42}, body: 'caf\\u00e9 \\u2014 it\\'s fine'}})-[:LINK]->(b {{id: {B + 43}}})")

    counts = {OK: 0, NO: 0, SKIP: 0}
    for r in REPORT:
        counts[r["verdict"]] += 1
    print(f"\nsupported={counts[OK]}  unsupported={counts[NO]}  unmeasured={counts[SKIP]}"
          f"  of {len(REPORT)} checks.")
    print("unmeasured is not the same as unsupported.")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe4-report.json")
    with open(out, "w") as f:
        json.dump({"base": BASE, "graph": GRAPH, "id_base": B, "label": LBL,
                   "checks": REPORT}, f, indent=1)
    print(f"full report written to {out}")


if __name__ == "__main__":
    main()

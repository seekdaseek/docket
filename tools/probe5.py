#!/usr/bin/env python3
"""probe5 - the last unknowns before the schema is written.

probe4 measured the surface. Six things it left open decide how the ingest and
the as-of query are actually written:

  A  the parameter key. "params" produced "missing parameter"; "parameters"
     got past binding and failed later on the MATCH shape. Indicated, not
     proven. Everything batched depends on it.
  B  interval filtering. WHERE n.num >= x AND n.num <= y failed only because
     the pattern was a bare MATCH (n), which the server forbids. With a label
     it is unmeasured, and the as-of query is exactly this shape.
  C  property updates on repeat CREATE. Nodes are keyed by id, so writing the
     same id twice reuses the node - but does it overwrite the properties,
     ignore them, or merge them? Ingest correctness depends on the answer.
  D  string literal size. A 16K literal is a parse error and sessions average
     about 14K characters. Bisect the ceiling, then try a parameter instead.
  E  the path procedures. They want $sourceNode as a parameter, MSpaths wants
     string source values, and length(path) is not a valid projection.
  F  edge duplication. Edges append on repeat writes. Measure whether there is
     any server-side way to avoid it before building client-side bookkeeping.

  python3 probe5.py
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

B = random.randint(1000000, 9000000)
LBL = f"P5{random.randint(100000, 999999)}"
OK, NO, SKIP = "ok", "NO", "SKIP"
REPORT = []


def post(query, params=None, params_key="parameters", consistency=None):
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


def brief(p, limit=130):
    if isinstance(p, dict):
        return f"columns={p.get('columns')} rows={p.get('rows')}"[:limit]
    return str(p)[:limit]


def first_value(payload):
    if not isinstance(payload, dict):
        return None
    rows = payload.get("rows") or []
    if not rows or not rows[0]:
        return None
    cell = rows[0][0]
    return cell.get("value") if isinstance(cell, dict) else cell


def record(group, name, verdict, detail, query=""):
    REPORT.append({"group": group, "name": name, "verdict": verdict,
                   "detail": str(detail)[:400], "query": query[:600]})
    print(f"{verdict:<4} {group}  {name:<30} {str(detail)[:140]}")


def run(group, name, query, params=None, params_key="parameters", consistency=None):
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


def seed(stmts):
    for s in stmts:
        st, pl = post(s)
        if st != 200:
            return f"http {st}: {pl}"
    return None


def main():
    print(f"probe5 -> {BASE}   id base {B}   label {LBL}")
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"{ADMIN}/readyz", timeout=5) as r:
                if r.status == 200:
                    print("node ready\n")
                    break
        except Exception:
            pass
        time.sleep(1)
    else:
        print("node never became ready. nothing measured.")
        sys.exit(1)

    err = seed([
        f"CREATE (a:{LBL} {{id: {B}, name: 'alpha', num: 1}})-[:NEXT]->(b:{LBL} {{id: {B + 1}, name: 'beta', num: 2}})",
        f"CREATE (a:{LBL} {{id: {B + 1}, name: 'beta', num: 2}})-[:NEXT]->(b:{LBL} {{id: {B + 2}, name: 'gamma', num: 3}})",
        f"CREATE (a:{LBL} {{id: {B + 2}, name: 'gamma', num: 3}})-[:NEXT]->(b:{LBL} {{id: {B + 3}, name: 'delta', num: 4}})",
    ])
    if err:
        print(f"seeding failed, nothing downstream can be measured: {err}")
        sys.exit(1)

    # ---------------------------------------------------------------- A params
    q = f"MATCH (n:{LBL}) WHERE n.name = $wanted RETURN n.name"
    run("A", "key_parameters", q, params={"wanted": "alpha"}, params_key="parameters")
    run("A", "key_params", q, params={"wanted": "alpha"}, params_key="params")
    run("A", "key_query_parameters", q, params={"wanted": "alpha"}, params_key="query_parameters")
    run("A", "param_in_inline_pattern",
        f"MATCH (n:{LBL} {{name: $wanted}}) RETURN n.name", params={"wanted": "alpha"})
    run("A", "param_int_id_inline",
        f"MATCH (n:{LBL} {{id: $wanted}}) RETURN n.id", params={"wanted": B})

    # ---------------------------------------------------------------- B as-of
    # the interval filter the whole as-of design rests on.
    run("B", "labelled_gt", f"MATCH (n:{LBL}) WHERE n.num > 1 RETURN n.num")
    run("B", "labelled_interval",
        f"MATCH (n:{LBL}) WHERE n.num >= 2 AND n.num <= 3 RETURN n.num, n.name")
    run("B", "labelled_interval_desc_limit",
        f"MATCH (n:{LBL}) WHERE n.num <= 3 RETURN n.num ORDER BY n.num DESC LIMIT 1")
    run("B", "interval_on_traversal",
        f"MATCH (a:{LBL} {{id: {B}}})-[:NEXT*1..3]->(c:{LBL}) WHERE c.num <= 3 RETURN c.num")
    run("B", "not_equal", f"MATCH (n:{LBL}) WHERE n.num <> 2 RETURN n.num")
    run("B", "or_combination",
        f"MATCH (n:{LBL}) WHERE n.num = 1 OR n.num = 3 RETURN n.num")

    # ---------------------------------------------------------------- C upsert
    ok1, p1 = run("C", "read_prop_before", f"MATCH (n:{LBL} {{id: {B}}}) RETURN n.name")
    before = first_value(p1) if ok1 else None
    st, pl = post(f"CREATE (a:{LBL} {{id: {B}, name: 'REWRITTEN', extra: 'added'}})"
                  f"-[:NEXT]->(b:{LBL} {{id: {B + 1}}})")
    if st != 200:
        skip("C", "repeat_create_overwrites_props", f"repeat write failed: http {st}: {pl}")
        skip("C", "repeat_create_adds_new_prop", "repeat write failed")
    else:
        ok2, p2 = post(f"MATCH (n:{LBL} {{id: {B}}}) RETURN n.name")
        after = first_value(p2) if ok2 == 200 else None
        record("C", "repeat_create_overwrites_props", OK,
               f"name was {before!r}, now {after!r} -> "
               f"{'OVERWRITES' if after == 'REWRITTEN' else 'KEEPS FIRST WRITE' if after == before else 'unclear'}")
        st3, p3 = post(f"MATCH (n:{LBL} {{id: {B}}}) RETURN n.extra")
        record("C", "repeat_create_adds_new_prop", OK if st3 == 200 else NO,
               f"extra reads back as {first_value(p3)!r}" if st3 == 200 else f"http {st3}: {p3}")

    # ---------------------------------------------------------------- D strings
    for size in (1000, 4000, 8000, 12000, 16000):
        run("D", f"literal_{size}",
            f"CREATE (a:{LBL} {{id: {B + 100 + size // 1000}, body: '{'x' * size}'}})"
            f"-[:LINK]->(b:{LBL} {{id: {B + 200 + size // 1000}}})")
    run("D", "long_string_as_parameter",
        f"CREATE (a:{LBL} {{id: {B + 300}, body: $body}})-[:LINK]->(b:{LBL} {{id: {B + 301}}})",
        params={"body": "y" * 16000})
    run("D", "read_back_long", f"MATCH (n:{LBL} {{id: {B + 101}}}) RETURN n.body")

    # ---------------------------------------------------------------- E procs
    src = f"sourceLabel: '{LBL}', sourceProperty: 'name', sourceValues: ['alpha']"
    run("E", "sspaths_inline_string_values",
        f"CALL algo.SSpaths({{{src}, relTypes: ['NEXT'], relDirection: 'out', "
        f"maxLen: 3, pathCount: 5, resultLimit: 10}}) YIELD path RETURN path")
    run("E", "sspaths_with_sourcenode_param",
        "CALL algo.SSpaths($sourceNode) YIELD path RETURN path",
        params={"sourceNode": {"sourceLabel": LBL, "sourceProperty": "name",
                               "sourceValues": ["alpha"], "relTypes": ["NEXT"],
                               "relDirection": "out", "maxLen": 3, "pathCount": 5,
                               "resultLimit": 10}})
    run("E", "mspaths_string_values",
        f"CALL algo.MSpaths({{sourceLabel: '{LBL}', sourceProperty: 'name', "
        f"sourceValues: ['alpha','beta'], targetValues: ['gamma','delta'], pairwise: true, "
        f"relTypes: ['NEXT'], relDirection: 'out', maxLen: 3, pathCount: 5, "
        f"resultLimit: 10}}) YIELD path RETURN path")
    for proj in ("path.length", "path.nodes", "path.ids", "path.start", "nodes(path)", "path"):
        run("E", f"projection_{proj.replace('(', '_').replace(')', '')}",
            f"CALL algo.MSpaths({{sourceLabel: '{LBL}', sourceProperty: 'name', "
            f"sourceValues: ['alpha'], targetValues: ['delta'], pairwise: true, "
            f"relTypes: ['NEXT'], relDirection: 'out', maxLen: 3, pathCount: 5, "
            f"resultLimit: 10}}) YIELD path RETURN {proj}")

    # ---------------------------------------------------------------- F edges
    ok_c, pc = post(f"MATCH (a:{LBL} {{id: {B}}})-[:NEXT]->(b:{LBL}) RETURN count(*)")
    n_before = first_value(pc) if ok_c == 200 else None
    post(f"CREATE (a:{LBL} {{id: {B}}})-[:NEXT]->(b:{LBL} {{id: {B + 1}}})")
    ok_d, pd = post(f"MATCH (a:{LBL} {{id: {B}}})-[:NEXT]->(b:{LBL}) RETURN count(*)")
    n_after = first_value(pd) if ok_d == 200 else None
    if n_before is None or n_after is None:
        skip("F", "edges_duplicate_on_repeat", "edge count failed, nothing measured")
    else:
        record("F", "edges_duplicate_on_repeat", OK,
               f"{n_before} -> {n_after} on an identical edge write "
               f"({'DUPLICATES, client must dedupe' if n_after != n_before else 'deduped by the server'})")
    run("F", "delete_then_recreate_edge",
        f"MATCH (a:{LBL} {{id: {B}}})-[r:NEXT]->(b:{LBL}) DELETE r")
    run("F", "count_after_delete",
        f"MATCH (a:{LBL} {{id: {B}}})-[:NEXT]->(b:{LBL}) RETURN count(*)")
    run("F", "unwind_batch_create_right_key",
        f"UNWIND $rows AS r CREATE (a:{LBL} {{id: r.a}})-[:BATCH]->(b:{LBL} {{id: r.b}})",
        params={"rows": [{"a": B + 400, "b": B + 401}, {"a": B + 401, "b": B + 402}]})
    run("F", "unwind_batch_read_inline",
        f"UNWIND $ids AS i MATCH (n:{LBL} {{id: i}}) RETURN n.id",
        params={"ids": [B, B + 1]})

    counts = {OK: 0, NO: 0, SKIP: 0}
    for r in REPORT:
        counts[r["verdict"]] += 1
    print(f"\nsupported={counts[OK]}  unsupported={counts[NO]}  unmeasured={counts[SKIP]}"
          f"  of {len(REPORT)} checks.")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe5-report.json")
    with open(out, "w") as f:
        json.dump({"base": BASE, "id_base": B, "label": LBL, "checks": REPORT}, f, indent=1)
    print(f"full report written to {out}")


if __name__ == "__main__":
    main()

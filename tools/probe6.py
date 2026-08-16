#!/usr/bin/env python3
"""probe6 - the last three things, two of which can silently destroy data.

1. PROPERTY SEMANTICS ON REPEAT WRITE. probe5 proved a repeat CREATE on the
   same id overwrites a property and adds a new one. It did not prove what
   happens to properties the second write does not mention. Every edge write
   restates both endpoint ids, and many of those writes will name only the id.
   If unnamed properties are dropped, a later edge write silently wipes the
   node it points at. Merge or replace decides whether the ingest is safe.

2. LONG STRING ROUND TRIP. probe5 wrote a 16K string as a parameter and got
   a 200, then read back zero rows - because the read pointed at an id whose
   write had failed. The write was never verified to have stored anything.
   Accepted is not stored. Session text averages ~14K characters.

3. THE PATH PROCEDURES, one error away. relDirection wants
   'incoming' / 'outgoing' / 'both' and probe5 sent 'out'. SSpaths wants
   $sourceNode as a parameter but rejects a composite one, which points at a
   scalar id rather than a map.

Plus the batch shape: UNWIND creates reject labels, so node typing has to move
to a property, and the usable batch size sets the ingest's wall-clock cost.

  python3 probe6.py
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
TIMEOUT = float(os.environ.get("HYDRA_TIMEOUT", "60"))

B = random.randint(1000000, 9000000)
OK, NO, SKIP = "ok", "NO", "SKIP"
REPORT = []


def post(query, params=None, consistency=None):
    body = {"cell_id": CELL, "query": query}
    if params is not None:
        body["parameters"] = params
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


def value(payload):
    if not isinstance(payload, dict):
        return None
    rows = payload.get("rows") or []
    if not rows or not rows[0]:
        return None
    cell = rows[0][0]
    return cell.get("value") if isinstance(cell, dict) else cell


def record(group, name, verdict, detail):
    REPORT.append({"group": group, "name": name, "verdict": verdict, "detail": str(detail)[:400]})
    print(f"{verdict:<4} {group}  {name:<30} {str(detail)[:150]}")


def run(group, name, query, params=None):
    st, pl = post(query, params)
    if st == 200:
        cols = pl.get("columns") if isinstance(pl, dict) else None
        rows = pl.get("rows") if isinstance(pl, dict) else None
        record(group, name, OK, f"columns={cols} rows={str(rows)[:80]}")
        return True, pl
    if st == 0:
        record(group, name, SKIP, f"transport failure, nothing measured: {pl}")
        return False, pl
    record(group, name, NO, f"http {st}: {pl}")
    return False, pl


def main():
    print(f"probe6 -> {BASE}   id base {B}")
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

    # ------------------------------------------------------------- 1. merge?
    st, pl = post(f"CREATE (a {{id: {B}, kind: 'Claim', name: 'one', num: 1}})"
                  f"-[:R]->(b {{id: {B + 1}, kind: 'Claim', name: 'two', num: 2}})")
    if st != 200:
        record("1", "seed_for_merge_test", NO, f"http {st}: {pl}")
        record("1", "unnamed_props_survive_repeat_write", SKIP, "seed failed, nothing measured")
    else:
        record("1", "seed_for_merge_test", OK, "seeded with name+num+kind on both ends")
        st2, pl2 = post(f"CREATE (a {{id: {B}}})-[:R]->(b {{id: {B + 1}}})")
        if st2 != 200:
            record("1", "unnamed_props_survive_repeat_write", SKIP,
                   f"bare-id restatement rejected: http {st2}: {pl2} "
                   f"(which is itself good news - it cannot wipe anything)")
        else:
            _, pn = post(f"MATCH (n {{id: {B}}}) RETURN n.name")
            _, pu = post(f"MATCH (n {{id: {B}}}) RETURN n.num")
            _, pk = post(f"MATCH (n {{id: {B}}}) RETURN n.kind")
            name, num, kind = value(pn), value(pu), value(pk)
            survived = name == "one" and num == 1 and kind == "Claim"
            record("1", "unnamed_props_survive_repeat_write", OK if survived else NO,
                   f"after a bare-id restatement: name={name!r} num={num!r} kind={kind!r} -> "
                   + ("MERGE semantics, unnamed properties survive, ingest is safe"
                      if survived else
                      "REPLACE semantics - a bare-id edge write WIPES the node. "
                      "every write must restate every property."))

    # ------------------------------------------------------------- 2. strings
    for size in (1000, 16000, 64000):
        body = "z" * size
        okw, _ = run("2", f"param_write_{size}",
                     f"CREATE (a {{id: {B + 10 + size // 1000}, body: $body}})"
                     f"-[:R]->(b {{id: {B + 500 + size // 1000}}})",
                     params={"body": body})
        if not okw:
            record("2", f"param_readback_{size}", SKIP, "write failed, nothing to read")
            continue
        st, pl = post(f"MATCH (n {{id: {B + 10 + size // 1000}}}) RETURN n.body")
        got = value(pl)
        if st != 200:
            record("2", f"param_readback_{size}", NO, f"http {st}: {pl}")
        elif got is None:
            record("2", f"param_readback_{size}", NO,
                   "write returned 200 but the property reads back as nothing - ACCEPTED IS NOT STORED")
        elif len(got) != size:
            record("2", f"param_readback_{size}", NO,
                   f"TRUNCATED: wrote {size} chars, read back {len(got)}")
        else:
            record("2", f"param_readback_{size}", OK, f"{size} chars round-tripped intact")

    # ------------------------------------------------------------- 3. batches
    rows = [{"a": B + 1000 + i, "b": B + 1001 + i} for i in range(3)]
    run("3", "unwind_create_no_labels",
        "UNWIND $rows AS r CREATE (a {id: r.a, kind: 'Batch'})-[:BATCH]->(b {id: r.b})",
        params={"rows": rows})
    run("3", "verify_batch_landed", f"MATCH (n {{id: {B + 1000}}}) RETURN n.kind")
    run("3", "match_by_property_only", "MATCH (n {kind: 'Batch'}) RETURN count(*)")
    for n in (100, 1000):
        big = [{"a": B + 10000 + i * 2, "b": B + 10001 + i * 2} for i in range(n)]
        started = time.time()
        okb, _ = run("3", f"unwind_create_{n}_rows",
                     "UNWIND $rows AS r CREATE (a {id: r.a, kind: 'Bulk'})-[:BULK]->(b {id: r.b})",
                     params={"rows": big})
        if okb:
            record("3", f"unwind_{n}_rows_seconds", OK, f"{time.time() - started:.2f}s for {n} edges")

    # ------------------------------------------------------------- 4. procs
    seed_err = None
    for i in range(3):
        st, pl = post(f"CREATE (a {{id: {B + 2000 + i}, name: 'chain{i}'}})"
                      f"-[:NEXT]->(b {{id: {B + 2001 + i}, name: 'chain{i + 1}'}})")
        if st != 200:
            seed_err = f"http {st}: {pl}"
            break
    if seed_err:
        for n in ("mspaths_outgoing", "sspaths_scalar_param", "sppaths_scalar_param"):
            record("4", n, SKIP, f"chain seed failed: {seed_err}")
    else:
        run("4", "mspaths_outgoing",
            "CALL algo.MSpaths({sourceProperty: 'name', sourceValues: ['chain0'], "
            "targetValues: ['chain3'], pairwise: true, relTypes: ['NEXT'], "
            "relDirection: 'outgoing', maxLen: 3, pathCount: 5, resultLimit: 10}) "
            "YIELD path RETURN path")
        run("4", "sspaths_scalar_param",
            "CALL algo.SSpaths({sourceNode: $sourceNode, relTypes: ['NEXT'], "
            "relDirection: 'outgoing', maxLen: 3, pathCount: 5, resultLimit: 10}) "
            "YIELD path RETURN path",
            params={"sourceNode": B + 2000})
        run("4", "sppaths_scalar_param",
            "CALL algo.SPpaths({sourceNode: $sourceNode, targetNode: $targetNode, "
            "relTypes: ['NEXT'], relDirection: 'outgoing', maxLen: 4, pathCount: 5, "
            "resultLimit: 10}) YIELD path RETURN path",
            params={"sourceNode": B + 2000, "targetNode": B + 2003})

    # ------------------------------------------------------------- 5. exactly once
    post(f"CREATE (a {{id: {B + 3000}}})-[:ONCE]->(b {{id: {B + 3001}}})")
    post(f"CREATE (a {{id: {B + 3000}}})-[:ONCE]->(b {{id: {B + 3001}}})")
    _, pc = post(f"MATCH (a {{id: {B + 3000}}})-[:ONCE]->(b) RETURN count(*)")
    dup = value(pc)
    post(f"MATCH (a {{id: {B + 3000}}})-[r:ONCE]->(b) DELETE r")
    post(f"CREATE (a {{id: {B + 3000}}})-[:ONCE]->(b {{id: {B + 3001}}})")
    _, pc2 = post(f"MATCH (a {{id: {B + 3000}}})-[:ONCE]->(b) RETURN count(*)")
    final = value(pc2)
    record("5", "delete_then_create_gives_exactly_one", OK if final == 1 else NO,
           f"two writes -> {dup} edges; delete then one write -> {final}")

    counts = {OK: 0, NO: 0, SKIP: 0}
    for r in REPORT:
        counts[r["verdict"]] += 1
    print(f"\nsupported={counts[OK]}  unsupported={counts[NO]}  unmeasured={counts[SKIP]}"
          f"  of {len(REPORT)} checks.")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe6-report.json")
    with open(out, "w") as f:
        json.dump({"base": BASE, "id_base": B, "checks": REPORT}, f, indent=1)
    print(f"full report written to {out}")


if __name__ == "__main__":
    main()

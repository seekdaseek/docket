"""Measured-before-use checks that run at the start of every ingest.

Three things the probes did not settle, each of which would corrupt an ingest
quietly rather than loudly:

1. HOW WIDE AN ID THIS SERVER ACCEPTS. Every probe used ids around a few
   million. Node ids here are hash prefixes, and the hash width decides the
   collision odds, so the widest accepted id is worth knowing rather than
   guessing. Measured by writing an id and reading it back, not by whether the
   write returned 200.

2. WHETHER RESTATING A LABEL WIPES PROPERTIES. probe6 proved a bare-id
   restatement merges, but it restated no label. Every statement after the
   first names its session as `(b:Session {id: ...})`. If that drops the
   session's properties, the session's timestamp disappears halfway through
   ingest and every temporal answer is wrong.

3. WHAT A WRITE COSTS. Property-bearing writes cannot be batched on this
   server, so wall-clock is one HTTP request per statement. The rate decides
   whether tonight's ingest is minutes or hours, and it is cheaper to measure
   it on thirty rows than to discover it on eleven thousand.

Each check reports supported / unsupported / unmeasured. Unmeasured is not a
polite word for unsupported: it means the check could not run and therefore
has no opinion.
"""
from __future__ import annotations

import random
import time

from .hydra import HydraError
from .schema import Writer
from .ids import IdRegistry

OK, NO, SKIP = "ok", "NO", "SKIP"

LABEL = "DocketPreflight"
CANDIDATE_BITS = (32, 40, 47, 52, 56, 62)


def _mark(report, name, verdict, detail):
    report.append({"name": name, "verdict": verdict, "detail": detail})
    return verdict == OK


def measure_id_width(client, tag: str, report: list) -> int | None:
    """The widest id width whose value survives a write and read back exactly."""
    widest = None
    for bits in CANDIDATE_BITS:
        node_id = (1 << bits) - 3
        key = f"{tag}-w{bits}"
        try:
            client.query(
                f"CREATE (a:{LABEL} {{id: {node_id}, nkey: $k}})"
                f"-[:PREFLIGHT]->(b:{LABEL} {{id: {node_id + 1}, nkey: $k2}})",
                {"k": key, "k2": key + "-dst"})
        except HydraError as e:
            _mark(report, f"id_width_{bits}", NO, f"write rejected: {e}")
            break
        try:
            got = client.property_of(LABEL, "nkey", key, "id")
        except HydraError as e:
            _mark(report, f"id_width_{bits}", SKIP, f"read failed: {e}")
            break
        if got == node_id:
            widest = bits
            _mark(report, f"id_width_{bits}", OK, f"{node_id} stored and read back")
        else:
            _mark(report, f"id_width_{bits}", NO,
                  f"wrote {node_id}, read back {got!r} -- accepted is not stored")
            break
    return widest


def check_label_restatement(client, tag: str, report: list) -> bool:
    """Does naming a label with only an id preserve the node's properties?"""
    key = f"{tag}-label"
    # Seed a labelled node carrying properties, via the only write form there is.
    node_id = random.randint(1_000_000, 9_000_000)
    base = node_id
    try:
        client.query(
            f"CREATE (a:{LABEL} {{id: {base}, nkey: $k, keepme: $v}})"
            f"-[:PREFLIGHT]->(b:{LABEL} {{id: {base + 1}, nkey: $k2}})",
            {"k": key, "v": "survives", "k2": key + "-dst"})
    except HydraError as e:
        return _mark(report, "label_restatement_preserves_props", SKIP,
                     f"seed failed: {e}")
    try:
        client.query(
            f"CREATE (x:{LABEL} {{id: {node_id + 2}, nkey: $k3}})"
            f"-[:PREFLIGHT]->(a:{LABEL} {{id: {node_id}}})",
            {"k3": key + "-src2"})
    except HydraError as e:
        return _mark(report, "label_restatement_preserves_props", SKIP,
                     f"restatement rejected: {e}")
    got = client.property_of(LABEL, "nkey", key, "keepme")
    return _mark(
        report, "label_restatement_preserves_props", OK if got == "survives" else NO,
        "properties survive a label+id restatement, so later statements cannot "
        "wipe their session" if got == "survives" else
        f"keepme reads back {got!r}: a label+id restatement WIPES properties, "
        f"so every write must restate every property")


def check_absence(client, tag: str, report: list) -> bool:
    """count(*) on a non-id predicate must be 0 for something never written."""
    try:
        n = client.count_by_property(LABEL, "nkey", f"{tag}-never-written")
    except HydraError as e:
        return _mark(report, "absence_is_zero_on_non_id_predicate", SKIP, str(e))
    return _mark(report, "absence_is_zero_on_non_id_predicate",
                 OK if n == 0 else NO,
                 f"count for a key never written is {n}")


def measure_write_rate(client, tag: str, report: list, n: int = 30) -> float | None:
    """Property-bearing one-hop writes per second, measured not assumed."""
    writer = Writer(client, IdRegistry(bits=52))
    started = time.time()
    try:
        for i in range(n):
            writer.edge(
                src_label=LABEL, src_key=f"{tag}-rate-{i}",
                src_props={"body": "x" * 1200, "idx": i},
                rel="PREFLIGHT",
                dst_label=LABEL, dst_key=f"{tag}-rate-dst",
                dst_props={})
    except HydraError as e:
        _mark(report, "write_rate", SKIP, f"failed after {writer.writes}: {e}")
        return None
    elapsed = time.time() - started
    rate = n / elapsed if elapsed else float("inf")
    _mark(report, "write_rate", OK,
          f"{n} property-bearing writes in {elapsed:.1f}s = {rate:.1f}/s")
    return rate


def run(client, statements_expected: int = 0) -> dict:
    """Every check, plus the id width the ingest should use."""
    tag = f"pf{random.randint(100000, 999999)}"
    report: list = []
    bits = measure_id_width(client, tag, report)
    label_safe = check_label_restatement(client, tag, report)
    check_absence(client, tag, report)
    rate = measure_write_rate(client, tag, report)

    estimate = None
    if rate and statements_expected:
        estimate = statements_expected / rate

    return {
        "tag": tag,
        "id_bits": bits,
        "label_restatement_safe": label_safe,
        "write_rate_per_second": rate,
        "estimated_seconds": estimate,
        "checks": report,
    }


def summarise(result: dict) -> str:
    lines = []
    for check in result["checks"]:
        lines.append(f"  {check['verdict']:<4} {check['name']}: {check['detail']}")
    lines.append(f"  id width chosen: {result['id_bits']} bits")
    if result.get("write_rate_per_second"):
        lines.append(f"  write rate: {result['write_rate_per_second']:.1f}/s")
    if result.get("estimated_seconds"):
        lines.append(f"  estimated ingest: {result['estimated_seconds'] / 60:.1f} min")
    return "\n".join(lines)

#!/usr/bin/env python3
"""Add a map of the repository to README.md.

    python3 tools/readme_map.py --check
    python3 tools/readme_map.py

Idempotent.

WHY. Nine of the tools in this repository were not mentioned in the README at
all, including the three most worth finding: the generated capability map, the
browsable register, and the as-of demo. A reader of the front door had no way to
learn they existed. `tests/test_readme.py` now fails if a file in `tools/` is
undocumented, so this cannot happen again by accident.
"""
from __future__ import annotations

import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
MARKER = "## What is in here"

SECTION = """

## What is in here

Three artifacts are worth opening before any of the code.

**[HYDRADB-CAPABILITIES.md](HYDRADB-CAPABILITIES.md)** — what HydraDB 0.1.0's
query surface actually does, generated from the probe reports rather than
written from notes. 207 checks across six runs. It also reconciles the probes
against each other: the first one reported almost everything as unsupported and
the later ones overturned eight of its verdicts by finding the working syntax.
Regenerate with `python3 tools/capabilities.py > HYDRADB-CAPABILITIES.md`.

**[RESULTS.md](RESULTS.md)** — every number, computed from the run files.
`python3 tools/report.py > RESULTS.md`.

**inspector.html** — one self-contained page holding all five hundred questions,
the evidence put in front of the model, and what it did with it. Gold sessions
are marked, so filtering to the refusals where the answer was present and opening
one shows the failure rather than asserting it. No server and no process behind
it on purpose: `python3 tools/inspector.py > inspector.html` and open the file.

### The tools

Measuring the database:

- `tools/probe.py`, `probe2.py`, `probe3.py`, `probe4.py`, `probe5.py` and
  `probe6.py` — what the node accepts, one group at a time. Each writes its own
  JSON report beside it, and the later ones correct the earlier ones.
- `tools/capabilities.py` — consolidates those six reports into the capability
  map, including the corrections the later probes made to the first.
- `tools/preflight2.py` — exercises every query shape the answering path uses,
  against a live node, before anything expensive runs.

Building the graph:

- `tools/inspect_data.py` — counts the benchmark before loading it: session
  reuse, turns, characters, and how many instances are stored out of order.
- `tools/ingest_run.py` — sessions, statements and claims into HydraDB.
- `tools/extract_run.py` — turns statements into claims. The only step that
  costs real money.
- `tools/date_audit.py` — measures how far the benchmark's own evidence falls
  after its own question dates, which is what set the as-of tolerance.

Answering and scoring:

- `tools/answer_run.py` — retrieval and answering over a slice or the whole set.
  `--no-model` runs the retrieval path for free. `--sample N --min-abstention K`
  gives a stratified slice; `--limit` does not sample and cannot test
  calibration.
- `tools/score_retrieval.py` — recall and citation density, no model needed.
- `tools/baseline_run.py` — the same questions with no evidence at all. The
  control that stops the headline number being read as world knowledge.
- `tools/judge_run.py` — grades answers with a disclosed model judge.
  `--free` grades only what `status` already decides, at no cost.
- `tools/report.py` — the results table, generated.

Diagnosing:

- `tools/diagnose_misses.py` — splits retrieval misses into extraction,
  ranking and filter gaps, which need opposite fixes.
- `tools/diagnose_refusals.py` — splits refusals by whether the gold session
  was in the evidence. Refusing with the answer present and refusing with
  nothing retrieved are different failures.
- `tools/inspector.py` — the register, above.

Demonstrating:

- `tools/timetravel.py` — one fact, several dates, the node returning a
  different tip each time with only the timestamp literal changed. `--list 20`
  shows which facts in the graph changed value at all.

Housekeeping:

- `tools/readme_day3.py`, `tools/readme_map.py` — idempotent README edits, kept
  as scripts so the change is reviewable rather than a silent hand edit.

Everything runs on the standard library. `bash run_tests.sh` runs the suite
offline: no network, no database, no keys.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(README):
        raise SystemExit("no README.md at " + README)
    text = open(README).read()
    needed = MARKER not in text
    print("repository map : %s" % ("needed" if needed else "already applied"))

    missing = []
    for name in sorted(os.listdir(os.path.join(ROOT, "tools"))):
        if name.endswith(".py") and name not in (SECTION + text):
            missing.append(name)
    if missing:
        print("STILL UNDOCUMENTED: " + ", ".join(missing))

    if a.check:
        return 0
    if needed:
        with open(README, "w") as fh:
            fh.write(text.rstrip("\n") + SECTION)
    after = open(README).read()
    print("verified       : %s" % ("yes" if MARKER in after else "NO"))
    return 0 if MARKER in after else 4


if __name__ == "__main__":
    raise SystemExit(main())

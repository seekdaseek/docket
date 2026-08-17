# docket

Agent memory that can be cross-examined.

Built for Hack Hydra, track 3, on HydraDB.

Most memory systems answer. Ask one a question it has no record of and it will
still produce a sentence, because producing sentences is what the model does.
LongMemEval marks 30 of its 500 questions as unanswerable from the history, and
its own retrieval evaluation skips them, because there is no evidence to find.
Those are the questions docket is built around.

Four rules, each enforced somewhere you can point at rather than described in
a paragraph.

**A claim is never separated from where it came from.** Every remembered fact
carries the session it was said in and when that session happened. An answer
returns the claims it rests on and the traversal that found them.

**Chronology comes from timestamps, never from order of arrival.** The
benchmark stores sessions out of order on purpose. Anything that treats list
position as time gets the 133 temporal-reasoning questions wrong while looking
like it works.

**A later claim supersedes an earlier one along an edge, rather than
overwriting it.** The current answer is the tip of a chain. The chain is still
there, so the same question asked with a different as-of date returns a
different, correct answer.

**Nothing is answered from an empty traversal.** When the graph holds no claim
about the subject, docket says so and shows what it looked for.

Of the 500 questions in LongMemEval_S, 211 are temporal-reasoning or
knowledge-update: 42 percent of the benchmark is about time and about facts
that were later replaced.

## Where HydraDB does the work

The graph is not a store to be read into memory and processed in Python. The
chronology, the supersede chains, the multi-hop joins between sessions and
entities, and the as-of filtering are all expressed as OpenCypher and executed
by the node. Without HydraDB there is no chain to walk.

HydraDB has no vector index and none is claimed, and no embeddings are used:
candidates are nominated by a BM25 gate built in the standard library, and
every statement that decides an answer is a traversal executed by the node.
The gate is lexical on purpose -- see "the gate is lexical" below -- and
`EmbeddingGate` raises rather than silently degrading, so no number in this
repository can be reported for a component that did not run.

## Status

Built and tested offline:

- `docket/hydra.py`, the HTTP client. Written against a response envelope
  recorded from a live node, not from documentation: typed cells matched
  positionally to columns, causal bookmarks threaded write to read, and a
  paged result that the client refuses to return as if it were whole.
- `docket/timeparse.py`, the benchmark's timestamp format, with the weekday
  verified rather than skipped.
- `docket/dataset.py`, loading with chronological ordering, global session
  dedupe, and the evidence session ids kept so citations can be scored.
- `docket/ids.py`, deterministic node ids with exact collision detection.
  HydraDB requires an integer id on both ends of every write and that id is the
  identity key, so the client assigns them. A wide hash makes a collision
  unlikely; the registry makes an undetected one impossible.
- `docket/schema.py`, the write vocabulary. This server has one write form, a
  one-hop CREATE, and no node-only CREATE at all, so a Session node is born on
  the first Statement that points at it. Turn text goes through parameters and
  is chunked, never truncated, because a string literal breaks the parser near
  a thousand characters.
- `docket/preflight.py`, which measures three things before an ingest writes
  anything real: the widest id this node accepts and reads back, whether
  restating a label preserves properties, and what a write costs.
- `docket/ingest.py`, resumable structural ingest. Progress is appended per
  session as it happens; an interrupted session has its edges deleted before
  being rewritten, because edges duplicate on repeat while node properties
  merge.
- `tools/probe.py` through `probe6.py`, which measured what this node supports
  instead of assuming it behaves like Neo4j.
- `tools/inspect_data.py`, which counts the session reuse factor before any
  extraction is paid for.

- `docket/llm.py`, a standard-library Anthropic client. A model that answers
  in a shape we cannot use raises, rather than returning an empty result: a
  session the model refused and a session with nothing durable in it must
  never look the same.
- `docket/extract.py`, claims with strict validation. A claim whose turn index
  is out of range is dropped, never clamped, because a clamped index cites the
  wrong sentence convincingly. Every drop is counted by reason.
- `docket/claims.py`, claims into the graph and the SUPERSEDES chains. A chain
  is only meaningful where a new value REPLACES the old one, so cardinality is
  declared per claim and defaults to deny. The first version of this chained
  any change of value; run against 48 real sessions it produced 120 links and
  most were fiction, because owning one pair of shoes does not stop you owning
  another. A predicate chains only when every claim carrying it said the
  relation is single-valued, and disagreements are reported rather than voted
  on.

Not built yet: entity resolution, retrieval, the answerer, the evaluation
harness, the inspector.

139 offline tests, no network and no keys required.

    ./run_tests.sh

## Running the probe

Start a node as described in the HydraDB README, then

    python3 tools/probe.py

It writes probe-report.json and prints three verdicts per check: supported,
unsupported, or unmeasured. Unmeasured is not a polite word for unsupported.
It means the check could not be performed and therefore has no opinion.

## Attribution

HydraDB is by the HydraDB team, AGPL-3.0, at github.com/hydra-db/hydradb.
This repository is MIT and speaks to a HydraDB node over its HTTP API.

LongMemEval is by Di Wu and colleagues. BEAM is by Mohammad Tavakoli and
colleagues.

Written with the help of an AI coding assistant, as the rules permit.

## Day 2 -- retrieval and answering

The gate proposes, the graph decides. `docket/gate.py` narrows 5,687 claims to
a handful; every claim that reaches an answer is then re-read from HydraDB,
filtered by the question's own timestamp, ordered in the database, and returned
with the statement it came from.

**The gate is lexical, not embeddings, and that is a decision rather than a
constraint.** This project's claim is memory that can be cross-examined, and a
BM25 hit can show exactly which terms earned it -- every result carries
`matched_terms`. An embedding hit can only assert a distance. The cost is
paraphrase: a question about "footwear" will not surface a claim that says
"sneakers". `EmbeddingGate` is the slot that fixes it, and it raises rather
than silently degrading, so no number can ever be reported for a component
that did not run.

**SUPERSEDES is the audit trail, not the retrieval path.** 5,687 claims
produced 189 chain links whatever grouping was tried, because the same fact is
rarely restated with a changed value under the same predicate. That was
measured before it was decided. Retrieval is `ts <= asked_at`, ORDER BY ts
DESC, take the tip; the chain is carried alongside to show what a value
replaced, where a replacement exists.

**Three outcomes, never two.** `answered` -- evidence existed and the model
used it. `absent` -- the graph was searched and holds nothing, which is the
right answer to the 30 abstention questions and the only one where confidence
is guaranteed wrong. `unmeasured` -- something failed: the model, the
transport, an unparseable reply. A rejected query and an honest absence look
identical from the outside, which is why `tools/preflight2.py` exercises every
query shape against a live node first and reports supported / unsupported /
unmeasured per shape.

A claim said *after* the question date is dropped and counted, not used. On a
benchmark built from dated sessions that leak is the easiest way to score well
and mean nothing.

    python3 tools/preflight2.py
    python3 tools/answer_run.py --data data/longmemeval_oracle.json --limit 20 --no-model
    python3 tools/answer_run.py --data data/longmemeval_oracle.json --limit 20

`--no-model` runs the whole retrieval path for free and writes the evidence it
would have reasoned over. Results append to `state/answers.jsonl`, resumable.

### HydraDB 0.1.0 finding: the cursor is advertised but not implemented

Measured Aug 16, in two steps because the first error hid the second.

A read large enough to page returns a `next_cursor`. Asking for it with
`{cell_id, cursor}` gives

    422 Failed to deserialize the JSON body into the target type:
        missing field `query`

so the continuation must repeat the original `query` and its `parameters`.
Doing that gets the real answer:

    400 invalid_request: ClientProtocol query is not supported yet:
        result cursor ...

The server hands out a continuation token it cannot serve. Any client that
trusts `next_cursor` therefore reads exactly one page and, unless it checks,
reports it as the whole result. For a memory system that is the worst possible
failure: a truncated read is indistinguishable from a memory that does not
contain the answer.

The workaround was already in probe1, which measured `skip` as supported and
filed it under "pagination fallback if cursors are awkward". `paged_rows()`
walks ORDER BY + SKIP + LIMIT over a unique property, with `follow_cursor` off
so a cursor raises instead of being followed, and halves the page size and
retries when the server caps a page itself -- which finds the cap rather than
guessing it. Two things it will not do: return a short result as a whole one,
or page over a non-unique ordering, because SKIP over an unstable sort drops
and repeats rows between pages.

### HydraDB 0.1.0 finding: a variable-length walk needs a fixed source id

Measured Aug 16. Matching the source of a `*1..N` pattern by an inline
property is rejected:

    400 invalid_request: OpenCypher query is not supported yet:
        variable-length MATCH requires a fixed source id

So `MATCH (a:Claim {nkey: $k})-[:SUPERSEDES*1..5]->(b:Claim)` fails while
`MATCH (a:Claim {id: 3831337985243655})-[:SUPERSEDES*1..5]->(b:Claim)`
succeeds. Node ids here are a deterministic hash of the natural key, so the
constraint costs nothing -- but it means every traversal entry point must be
an id, and any design that plans to walk from a looked-up property has to
resolve to an id first. Ordering is done in Python rather than with ORDER BY,
because ORDER BY over a variable-length pattern is not in the measured surface
and a chain is at most N hops long. One unverified feature per query.

## The as-of filter, and why it is 24 hours rather than zero

Retrieval refuses any claim said after the question was asked. On a benchmark
built from dated sessions that rule is the difference between recalling a fact
and reading the answer sheet, so it is not negotiable in principle.

In practice a strict version of it is wrong for LongMemEval, and the data says
so. `tools/date_audit.py` over all 500 oracle instances:

| | |
|---|---|
| instances whose evidence postdates the question | 43 (8.6%) |
| evidence sessions 0-1h late | 8 |
| 1-6h late | 24 |
| 6-24h late | 33 |
| more than 24h late | 0 |

| tolerance | instances whose evidence survives |
|---|---|
| strict | 457 (91.4%) |
| +1h | 461 (92.2%) |
| +6h | 476 (95.2%) |
| **+1d** | **500 (100%)** |

24 hours is the smallest window that keeps every instance's evidence, and
nothing in the dataset lands beyond it. That is the rule, and it is
`as_of_tolerance` rather than a magic number: `--as-of-tolerance 0` restores
the strict behaviour, and every run reports `admitted_by_tolerance` -- how many
claims were used *because* of the window -- so its cost appears on the report
instead of hiding inside the score.

One honest limit. The audit's "non-evidence admitted" column reads zero at
every tolerance, and that proves nothing: the oracle file contains only
evidence sessions, so there are no distractors to admit. On `longmemeval_s`,
where the haystack is real, a 24-hour window will let some in and the same
audit has to be re-run there. Until it is, this number is measured on the
oracle and claimed for the oracle.

## Retrieval, measured over all 500 oracle questions

No model, no spend -- `answer_run --no-model` records what was retrieved and
`score_retrieval` scores it against LongMemEval's own `answer_session_ids`.

| | recall | density |
|---|---|---|
| claim mode (default) | **0.7553** | 0.2986 |
| session mode | 0.5553 | 0.2611 |

Per category, claim mode: knowledge-update **0.93** · temporal-reasoning 0.82 ·
multi-session 0.81 · single-session-user 0.78 · single-session-assistant 0.48 ·
single-session-preference 0.30.

The two categories the design is aimed at -- knowledge-update and
temporal-reasoning, 42% of the benchmark -- are the two it does best on. That
is the claim this project makes, and it is the claim the numbers support.

### Two things that were tried and measured worse

**Ranking sessions instead of claims.** Half the remaining misses are
aggregation questions: how many model kits, how many hours driving in total,
which store did I spend the most at. The answer is not in one claim, it is
spread across fourteen to thirty-six claims in three or four sessions, and the
best gold claim for those questions sat at rank 19, 36, 80, even 177. So the
gate learned to score whole sessions and return everything in the best few.
It cost twenty points of recall. The reason is worth stating: session mode
narrows the candidate pool to three sessions, while twelve top-ranked claims
can come from twelve different sessions. Coverage was worth more than depth.
The mode is kept, off by default, because a measured negative is a result.

**Raising the candidate cut.** 12 -> 30 moved recall 0.734 -> 0.747 while
density fell 0.29 -> 0.22 and runtime doubled. Gold at rank 80 is out of reach
either way; a bigger cut buys noise.

### One bug worth naming

Evidence was being SELECTED by recency and then truncated. With twelve
candidates for six slots that was survivable, because the pool was already
relevance-ranked. It was not survivable once the pool grew. The fix is to
select by relevance and only then present by recency, so the tip still leads
the prompt -- and it is worth two points of recall on its own. The regression
test for it was verified to fail against the old ordering before being kept.

## Day 3 -- the eval, and what it found

Full numbers, generated from the run files rather than typed, are in
[RESULTS.md](RESULTS.md). Regenerate them with `python3 tools/report.py`.

**Accuracy 0.3511 (165/470) against a no-memory baseline of 0.000 (0/470).**
The baseline is the same 500 questions and the same model with no evidence at
all. It answered none of them, so nothing in the figure above is world
knowledge leaking in. Stated precisely, because it is a weaker claim than it
looks: the baseline was instructed not to guess, so it shows that a model which
declines rather than guesses gets none of these questions. It does not show
they are unguessable.

Judge is `claude-sonnet-5`, deliberately a stronger model than the
`claude-haiku-4-5` answerer so the system is not grading itself. The judge is
shown the question, the reference answer and the produced answer, and never the
retrieved evidence -- a judge that can see the claims ends up grading the
retrieval a second time. Its prompt is in `docket/judge.py`. 470 questions were
graded and none came back unjudged.

### The average hides three different systems

| category | n | retrieval recall | accuracy | conversion |
| --- | --- | --- | --- | --- |
| knowledge-update | 72 | 0.9306 | 0.7222 | 0.776 |
| single-session-user | 64 | 0.7812 | 0.5781 | 0.740 |
| temporal-reasoning | 127 | 0.8189 | 0.3386 | 0.413 |
| multi-session | 121 | 0.8099 | 0.2231 | 0.275 |
| single-session-assistant | 56 | 0.4821 | 0.0893 | 0.185 |
| single-session-preference | 30 | 0.3000 | 0.0333 | 0.111 |

**conversion** is accuracy divided by retrieval recall: of the questions whose
evidence was found, the share that also produced a right answer.

Read the last column rather than the aggregate. Where retrieval is strong the
system works -- knowledge-update finds the evidence 93 percent of the time and
converts three quarters of that into correct answers. Where retrieval is weak
the system collapses, and the collapse is honest: preference finds the evidence
on 30 percent of questions, and 19 of its 30 refusals happened with nothing
relevant retrieved. Refusing there was the correct behaviour.

The interesting rows are the middle two. single-session-user and
temporal-reasoning have almost the same retrieval recall, 0.7812 and 0.8189,
and convert at 0.740 and 0.413. multi-session has recall 0.8099 and converts at
0.275. Whatever is costing those answers, it is not the gate.

### The failure mode is over-refusal, not hallucination

Of 470 graded questions: 165 correct, 69 answered and wrong, 133 refused with a
gold session sitting in the evidence, and 103 refused with the gold session
never retrieved.

Those last two are counted separately everywhere in this repository because
they need opposite fixes, and merging them pointed the diagnosis at the wrong
layer once already during this build. A refusal with the evidence present is
the answerer's failure. A refusal with nothing retrieved is the gate's, and the
refusal itself was right.

    false refusal rate         0.3624   (133 of 367)
    abstention refusal rate    0.9333   (28 of 30)

Neither number means anything alone. A system that refuses everything scores
1.000 on the right-hand number and 0.000 accuracy -- `tests/test_day3.py` has a
test named for exactly that case, and the no-memory baseline is a live example
of it. Reported together they say what is actually wrong: the refusal machinery
works, and it is tuned too hot.

Two of the 30 unanswerable questions were answered anyway. That is a 2/30
hallucination rate, not zero, and it is written here rather than rounded away.

**When docket does answer, it is right 165 times out of 234 -- 70.5 percent.**
Converting the 133 calibration failures at that same precision would land near
0.55. That is an estimate and not a result, but it sizes the gap: over-refusal
is worth roughly twenty accuracy points.

### Limitations, stated

**A claim records when a fact was mentioned, not when the event happened.** Two
claims extracted from one conversation carry the same timestamp even when they
describe events months apart, and the event's own date survives only if the
extractor captured it inside the claim. Ordering and duration questions are
answerable only in that case. This is visible in the table: temporal-reasoning
and multi-session carry 92 of the 133 calibration failures. Fixing it properly
means extracting event dates as a first-class field and re-running extraction,
which did not fit the six days.

**The judge is a model.** Its prompt is in the repository and its id is written
onto every judged row, so the grading can be re-run or disputed, but it is not
a human rubric.

**The as-of tolerance was measured on the oracle file, which contains only
evidence sessions.** The audit's "non-evidence admitted" column reads zero at
every tolerance and proves nothing, because there are no distractors present to
admit. On `longmemeval_s` a 24-hour window will let some in, and the same audit
has to be re-run there.

**Answering was measured on the oracle split, not the full haystack.**

### Changes that were measured worse and reverted

The answerer prompt was rewritten to explain the mention-date problem and to
soften the refusal rule. On an identical 25-question slice it took answers from
15 to 7, accuracy from 0.3333 to 0.2381, and the false refusal rate from 0.375
to 0.6667. It is reverted, and `SYSTEM` is the original text. The rewrite
bundled three changes at once, so it cannot say which of them did the damage --
that is a fault in the experiment, not a finding.

Rendering evidence timestamps as dates rather than raw Unix epochs was then
measured on its own against the same slice: answer count identical, one answer
flipped from wrong to correct. **Measured neutral.** It is kept because it does
no harm and because `said 2023-04-10 (Mon)` is legible where `said 1681138020`
is not. `--epoch-dates` restores the earlier rendering exactly.

### A parser bug worth naming

The first paid run recorded 14 of 25 answers as measurement failures. The model
had been emitting correct JSON and then continuing in prose, and `json.loads`
over the whole reply rejected it. Seven of those 14 were correct abstentions --
the one behaviour this project exists to demonstrate, discarded as noise.

`extract_json` now reads the first complete JSON value with `raw_decode` and
reports whatever followed; `trailing_prose` counts how often the model kept
talking. On the full run that was 168 of 493 replies. The project's rule is
unchanged: output with no JSON in it still raises and is still recorded as
unmeasured. What changed is that valid JSON followed by commentary stopped
being counted as a failure to answer. Those were two different facts being
merged into one.

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

# How blis-vllm is checked against vLLM

blis-vllm's claim is behavioural: given the same configuration and the same inputs, it makes the
same scheduling, KV-cache, connector and metric decisions as vLLM v0.32.0. This page says how
that claim is tested, and which part of it each test supports. Nothing is called proven beyond
its tier.

## The oracle

The oracle is vLLM itself, at the pinned release, running on CPU with no GPU. The harness under
`oracle/` builds vLLM's **real** classes and drives them through a scenario:

- tier 2 builds the `Scheduler` with vLLM's own CPU test helper,
  `tests/v1/core/utils.py:54@5df63a72` (`create_scheduler`), the same setup vLLM's `cpu_test`
  suites use in its own CI (`.buildkite/test_areas/misc.yaml:216-219@5df63a72`);
- it calls `schedule()` and `update_from_output()` as `EngineCore` does, one step at a time.

**Patch the environment, never the decisions.** The oracle may replace only the GPU, the clock,
transports and external services. Every replacement is listed by symbol in `oracle/mocks.json`,
and `oracle/check_mocks.py` fails CI if an entry names scheduler, KV-cache, EngineCore or
connector code. Where vLLM reads the clock to make a decision (tiered-offload backpressure,
P2P deadlines), the oracle replaces the clock with the scenario's virtual time; the decision code
runs unmodified.

## The shared GPU mock

A GPU decides three things a scheduler cannot: which tokens are sampled, how many speculative
drafts the target model accepts, and when an asynchronous KV transfer finishes. The mock supplies
all three, identically to both sides, from the scenario:

- **sampled tokens:** each request's assigned output, one token per step that completes the
  request's known tokens (read from vLLM's own `is_prefill_chunk`, set in
  `_update_after_schedule`, `vllm/v1/core/sched/scheduler.py:1634@5df63a72`). The last assigned
  token is sampled as EOS (`tests/v1/core/utils.py:43@5df63a72`), so the output length, which is
  the simulator's input, ends the request through vLLM's own `check_stop`;
- **draft acceptance:** a per-request, per-step accepted count from the scenario (acceptance is an
  input, #2021 B3; the drafting and rejection algorithms are outside parity);
- **transfer outcomes:** a completion step and success flag per transfer.

Both sides read the same scenario file, so the Go side never re-implements Python's random
number generator. The mock has a version (`mock_version` in `oracle/mocks.json`), recorded in
every result.

## Separate decisions from cost

Behavioural diff tests run with a **fixed, explicit KV capacity** (`num_blocks`), fixed step
durations and fixed transfer outcomes, identical on both sides, so a cost-model difference cannot
hide a scheduler bug. Runs with real latency-kernel pricing are a separate integrated tier, and
are reported as priced by the kernel's vLLM 0.31.0 behaviour.

## Tiers

| Tier | Proves | Oracle | Status |
|---|---|---|---|
| T1 Configuration | the `engine:` block resolves and validates as vLLM's startup does | rules packs, `MatchSource` against the pinned source | 0.29/0.31 packs checked; 0.32 pack in progress |
| T2 Scheduler and KV | per-step decisions with fixed capacity: tokens per request, preemptions, finishes, block contents by hash and ref count, prefix hits, request counts | `oracle/run_scheduler.py` (`create_scheduler`) | full attention, FCFS/priority, chunked prefill, prefix caching: 14 scenarios + 2000 fuzz seeds at zero mismatches; speculative decoding, LoRA and hybrid KV pending |
| T3 EngineCore | async scheduling and the PP batch queue (`vllm/v1/engine/core.py:698-810@5df63a72`), DP waves and prefill throttling | EngineCore batch queue with a mocked executor | planned |
| T4 Connectors and offload | connector scheduler-side state machines under virtual time and controlled transport | real connector scheduler classes, transports mocked | planned |
| T5 Metrics | names, labels, histogram buckets and enablement of every computable `vllm:*` family | the generated manifest and vLLM's Prometheus loggers | planned |

## Kinds of test

- **Differential:** the same scenario through the oracle and through blis-vllm; every step's
  record is compared field by field, and the report names the **earliest** step and field that
  differ, not only the last.
- **Property:** no lost requests, KV block conservation (used + free = total), budgets never
  exceeded, at most `max_num_seqs` running.
- **Metamorphic, with step duration fixed:** both of the issue's claims are false *for vLLM
  itself*, not only under real pricing, and the counterexamples are kept as scenarios checked
  against vLLM v0.32.0 (`oracle/scenarios/meta-*.json`): with FCFS, no prefix caching and every
  request at step 0, 33 blocks preempt nothing and 40 blocks preempt once (the larger pool admits
  a request early that later outgrows it); with priority scheduling, doubling the budget moves a
  request's first token from step 7 to 9 (a faster prefill lets a later, more urgent request in
  first). What holds, and is tested: more KV never adds preemptions when requests run one at a
  time in a fixed order; a larger budget never delays a lone request's first token.
- **Fuzz:** seeded random request streams through both sides. A failing seed is shrunk to its
  smallest scenario and kept under `oracle/scenarios/` as a permanent test.

## Provenance

Every oracle run records the vLLM version and commit, the Python version and the lockfile hash,
the scenario's sha256, the mock version and the list of mocks; every Go run records blis-vllm's
module version. A result can be reproduced from these alone.

## Golden traces

A golden trace (`testdata/golden/`) is the oracle's output for a scenario, committed so the Go
tests can run without Python. It is never overwritten automatically: regenerating one needs
`BLIS_VLLM_ALLOW_GOLDEN_UPDATE=1`, and a CI check fails a change to `testdata/golden/` whose
commit message has no `golden-reviewed:` trailer naming the reviewer.

## Target

Zero mismatches. A known mismatch is listed with its earliest divergence in the repository's
status, never silently tolerated.

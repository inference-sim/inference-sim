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
| T1 Configuration | the `engine:` block resolves and validates as vLLM's startup does | rules packs, `MatchSource` against the pinned source | 0.29, 0.31 and 0.32 packs each checked against their own release in CI (`packs-against-vllm`); `rules/v0_32` passes all 35 source checks at `5df63a72`, and mutants of its new data, rule and parser are caught |
| T2 Scheduler and KV | per-step decisions with fixed capacity: tokens per request, preemptions, finishes, block contents by hash and ref count, prefix hits, request counts | `oracle/run_scheduler.py` (`create_scheduler`) | full attention, FCFS/priority, chunked prefill, long-prefill cap (and its adaptive floor), preemption, prefix caching, speculative decoding (EAGLE3, P-EAGLE, draft_model, PARD, MTP single and multi-module, DSpark; acceptance supplied), max_loras, watermark, max_num_active_seqs, max_num_scheduled_tokens, KV events, hybrid KV (full attention + sliding windows) and disable_hybrid_kv_cache_manager: 35 scenarios + 4000 fuzz seeds at zero mismatches, all mutants caught (`oracle/mutate.py`); Mamba groups, prefix_match_unit, the retention interval, bounded replay (blis-vllm#7) and DCP (blis-vllm#8) are follow-ups |
| T3 EngineCore | async scheduling and the PP batch queue (`vllm/v1/engine/core.py:698-810@5df63a72`), DP waves and prefill throttling | `oracle/run_enginecore.py`: vLLM's own `step_with_batch_queue` on an EngineCore that holds the real (Async)Scheduler, only the executor mocked, the EngineCore built without its model and GPU setup (`oracle/mocks.json`) | batch queue, async scheduling and PP: 8 scenarios + 1000 fuzz seeds at zero mismatches, queue sizes 1-5, 10 of 10 mutants caught (`oracle/mutate_t3.py`). Limitation: the V2 model runner cannot start on the CPU wheel (no Triton, blis-vllm#3), so its effect on the scheduler and on the queue size reaches them through the scheduler's own `use_v2_model_runner` and `max_concurrent_batches`' own code, as vLLM's CPU tests do. Data parallel (`oracle/run_dp.py`): vLLM's own DP engine-core loop (prefill throttling, waves, dummy batches) and internal load balancer across 2-4 ranks, the all-reduce and the coordinator's stats transport mocked: 5 scenarios + 1000 fuzz seeds at zero mismatches; with the batch queue, 21 of 21 tier-3 mutants caught |
| T4 Connectors and offload | connector scheduler-side state machines under virtual time and controlled transport | real connector scheduler classes, transports mocked | NIXL pull mode and `kv_load_failure_policy`: 21 scenarios + 1000 fuzz seeds at zero mismatches, 20 of 20 non-equivalent mutants caught (`core/connector/nixl`, `oracle/run_connector.py`); the connector registry (`registry/`): an entry is earned only by one of these diff tests, mutants R1-R6 caught; the other connectors are follow-ups (see the feature matrix) |
| T5 Metrics | names, labels, histogram buckets and enablement of every computable `vllm:*` family | the generated manifest and vLLM's Prometheus loggers | inventory done: `docs/metrics-manifest.json`, 130 families from the v0.32.0 source (45 exact, 52 derived, 33 unavailable), regenerated in CI; the 45 families vLLM's own logger registers on CPU agree in type and labels, and their 19 histograms' bucket bounds are recorded from the live objects (`docs/metrics-buckets.json`); values: the 23 scheduler-decided exact families match vLLM's own logger sample for sample after every step (14 scenarios, 9,906 samples; 1,000 fuzz seeds, 1.39M samples; 0 mismatches); the 22 connector/offload exact families wait on tier 4 |

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
- **Mutation:** a test that cannot fail proves nothing, so each supported feature has a
  deliberate one-line bug (a mutant) that the differential and fuzz tests must catch
  (`oracle/mutate.py`, a CI step). The sweep runs the mutants in `core/sched` and `core/kv`
  (speculative decoding, KV events and hybrid KV among them) against the 35 scenarios and 4000
  fuzz seeds; every one is caught, and none is listed as a known gap. Two candidate mutants are
  left out as equivalent in a synchronous scheduler, each with its reason in `oracle/mutate.py`
  (freeing window blocks on the processed-token basis, which needs tokens in flight, and
  aligning the hybrid cacheable count, which cache_blocks floors anyway). Two that once survived, the
  `max_model_len - num_computed - 1` input cap on running requests
  (`vllm/v1/core/sched/scheduler.py:713-720@5df63a72`) and skipping rather than stopping at a
  running request scheduled zero tokens (`:755-773`), bind only when a request runs past its
  known tokens; `spec-model-len-cap` and the speculative fuzz seeds now catch both.

## Provenance

Every oracle run records the vLLM version and commit, the Python version and platform, the
hash of the environment's frozen package list (`lockfile_sha256`; CI writes it with `pip
freeze`, a local run passes it as `ORACLE_LOCKFILE`), the installed vLLM distribution (the
release CPU wheel in CI), the helper model's pinned revision, the
scenario's sha256, the mock version and the list of mocks. A Go build records blis-vllm's module
version (`blisvllm.ProvenanceFor`). A result can be reproduced from these alone.

## Golden traces

A golden trace (`core/sched/testdata/golden/`) is the oracle's output for a scenario, committed
so the Go tests can run without Python. It is never overwritten without review:
`oracle/run_scheduler.py` refuses to write into `testdata/golden/` unless
`BLIS_VLLM_ALLOW_GOLDEN_UPDATE=1`, and CI (`goldens-reviewed`, `oracle/check_goldens.py`) fails a
push or pull request with a commit that changes the decisions an existing golden records (its
`steps` or `finished`), or deletes one, without a `golden-reviewed: <reviewer>` trailer naming
the person who checked it. A new golden for a new scenario, or a refresh of only the
`provenance` block, needs no trailer; the check verifies the decisions are byte-for-byte
unchanged. The differential test also refuses a golden whose recorded
scenario hash, vLLM version and commit, or mock version does not match the scenario file and the
pins it runs against, so a trace can only count as evidence for the scenario it came from.
CI re-runs the pinned vLLM release on every scenario and fails if any committed golden's
decisions differ from that fresh trace (`oracle/compare_goldens.py`), so a golden made on a
developer's machine counts only while the release itself still produces it.

## Target

Zero mismatches. A known mismatch is listed with its earliest divergence in the repository's
status, never silently tolerated.

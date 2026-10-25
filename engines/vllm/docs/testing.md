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
| T1 Configuration | the rules packs' accepted values, defaults and refusals are the release's own, read from its source | rules packs, `MatchSource` against the pinned source (the source text, not vLLM's config classes; running `EngineArgs`/`VllmConfig` on the same blocks is https://github.com/inference-sim/blis-vllm/issues/27) | 0.29, 0.31 and 0.32 packs each checked against their own release in CI (`packs-against-vllm`, results and provenance uploaded as `tier1-<pack>`); `rules/v0_32` passes all 35 source checks at `5df63a72`, and mutants of its new data, rule and parser are caught |
| T2 Scheduler and KV | per-step decisions with fixed capacity: tokens per request, preemptions, finishes, block contents by hash and ref count, prefix hits, request counts | `oracle/run_scheduler.py` (`create_scheduler`) | full attention, FCFS/priority, chunked prefill, long-prefill cap (and its adaptive floor), preemption, prefix caching, speculative decoding (EAGLE3, P-EAGLE, draft_model, PARD, MTP single and multi-module, DSpark; acceptance supplied), max_loras, watermark, max_num_active_seqs, max_num_scheduled_tokens, KV events, hybrid KV (full attention + sliding windows, a sliding-window-only model, layers sharing one spec merged into one group, and the extra tokens a multi-module MTP drafter keeps below the window) and disable_hybrid_kv_cache_manager: 40 scenarios + 4000 fuzz seeds at zero mismatches, all mutants caught (`oracle/mutate.py`); Mamba groups, prefix_match_unit, the retention interval, bounded replay (blis-vllm#7) and DCP (blis-vllm#8) are follow-ups |
| T3 EngineCore | async scheduling and the PP batch queue (`vllm/v1/engine/core.py:698-810@5df63a72`), DP waves and prefill throttling | `oracle/run_enginecore.py`: vLLM's own `step_with_batch_queue` on an EngineCore that holds the real (Async)Scheduler, only the executor mocked, the EngineCore built without its model and GPU setup (`oracle/mocks.json`) | batch queue, async scheduling and PP: 8 scenarios + 1000 fuzz seeds at zero mismatches, queue sizes 1-5, 10 of 10 mutants caught (`oracle/mutate_t3.py`). Limitation: the V2 model runner cannot start on the CPU wheel (no Triton, blis-vllm#3), so its effect on the scheduler and on the queue size reaches them through the scheduler's own `use_v2_model_runner` and `max_concurrent_batches`' own code, as vLLM's CPU tests do. Data parallel (`oracle/run_dp.py`): vLLM's own prefill throttle and wave sync (`DPEngineCoreProc._should_throttle_prefills`, `_has_global_unfinished_reqs`) and internal load balancer (`DPLBAsyncMPClient.get_core_engine_for_request`) across 2-4 ranks, the all-reduce and the coordinator's stats transport mocked: 5 scenarios + 1000 fuzz seeds at zero mismatches. The loop that steps them is the harness's transcription of `run_busy_loop` (dummy batches, the wave's end, a request starting the next wave), listed in `oracle/mocks.json` as such: those decisions are shown to match the transcription, not vLLM's loop itself (blis-vllm#3); with the batch queue, 21 of 21 tier-3 mutants caught |
| T4 Connectors and offload | connector scheduler-side state machines under virtual time and controlled transport | real connector scheduler classes, transports mocked | NIXL pull mode and `kv_load_failure_policy`: 22 scenarios + 1300 fuzz seeds (300 of them into `max_model_len`) at zero mismatches, at each scenario's own `max_model_len`, 20 of 20 non-equivalent mutants caught (`core/connector/nixl`, `oracle/run_connector.py`); the connector registry (`registry/`): an entry is earned only by one of these diff tests, checked on its golden traces' content and provenance, mutants R1-R10 caught; the other connectors are follow-ups (see the feature matrix) |
| T5 Metrics | names, labels, histogram buckets and enablement of every computable `vllm:*` family | the generated manifest and vLLM's Prometheus loggers | inventory done: `docs/metrics-manifest.json`, 130 families from the v0.32.0 source (45 exact, 52 derived, 33 unavailable), regenerated in CI; the 45 families vLLM's own logger registers on CPU agree in type and labels, and their 19 histograms' bucket bounds are recorded from the live objects (`docs/metrics-buckets.json`); values: the 23 scheduler-decided exact families match vLLM's own logger sample for sample after every step (14 scenarios, 9,906 samples; 1,000 fuzz seeds, 1.39M samples; 0 mismatches); the 22 connector/offload exact families wait on tier 4 |

The engine interface (`core/vllm`) is checked in two ways. blis-schemas' `enginetest` checks the contract's laws on the v0.32.0 engine. A diff test drives tier-2 scenarios through `engine.Engine` (Add, then Schedule and Complete one batch at a time) and compares every step with vLLM's own Scheduler trace (`core/sched/testdata/golden`). Its mutants are in `oracle/mutate_e24.py`, which CI runs with the other sweeps. The factory builds one batch in flight only: with async scheduling or PP, vLLM's own loop decides which requests share a batch (`vllm/v1/engine/core.py:724-768@5df63a72`), and the adapter does not reproduce that loop yet, so it refuses those configurations (https://github.com/inference-sim/blis-vllm/issues/21). Tier 3 proves `core/engine`'s port of the loop, not the adapter.

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
  (speculative decoding, KV events and hybrid KV among them) against the 40 scenarios and 4000
  fuzz seeds; every one is caught, and none is listed as a known gap. Two candidate mutants are
  left out as equivalent in a synchronous scheduler, each with its reason in `oracle/mutate.py`
  (freeing window blocks on the processed-token basis, which needs tokens in flight, and
  aligning the hybrid cacheable count, which cache_blocks floors anyway). Two that once survived, the
  `max_model_len - num_computed - 1` input cap on running requests
  (`vllm/v1/core/sched/scheduler.py:713-720@5df63a72`) and skipping rather than stopping at a
  running request scheduled zero tokens (`:755-773`), bind only when a request runs past its
  known tokens; `spec-model-len-cap` and the speculative fuzz seeds now catch both. Each
  matrix row cites a mutant of its own feature: the scheduler knobs `watermark`,
  `long_prefill_token_threshold_adaptive` and `max_num_scheduled_tokens` have M52-M54, which
  touch only that knob. The tier-3, tier-4 and engine-interface sweeps are
  `oracle/mutate_t3.py`, `mutate_t4.py` and `mutate_e24.py`; the metrics model's is
  `oracle/mutate_metrics.py` (X1-X7, run against the 35 scenarios' metric goldens and the 4000
  metric fuzz traces).

## Provenance

Every oracle trace records, in its `provenance` block:
- the vLLM version and commit, and the installed vLLM distribution (the release CPU wheel in CI);
- the Python version and the platform;
- `lockfile_sha256`, the hash of the environment's frozen package list. CI installs
  `oracle/requirements.lock` and records the hash of what `pip freeze` then lists; a local run
  passes its own list as `ORACLE_LOCKFILE`;
- `requirements_lock_sha256`, the hash of the committed `oracle/requirements.lock`, which CI
  installs and `oracle/check_lock.py` proves was installed exactly: the environment a CI trace
  ran in is named by a file in the repository;
- the helper model and its pinned revision;
- the scenario's sha256, and the seed for a generated (fuzz) scenario (`seed`; `null` for a
  hand-written one);
- the blis-vllm commit the run checks (`blis_vllm_commit`: CI's `GITHUB_SHA`, else the checkout's
  `HEAD`, marked `+dirty` when it has uncommitted changes);
- the mock version and the list of mocks.

The Go side of a CI run is recorded once per run, in the artifact's `oracle/out/provenance.json`,
by `go run ./cmd/provenance`. It holds blis-vllm's identity as the build records it
(`blisvllm.ProvenanceFor`), the commit, the Go toolchain and the platform. The property and
metamorphic tests' verbose results are in the same artifact (`property-results.txt`), and each
tier-1 job uploads its source-check results with their provenance (`tier1-<pack>`). A result can
be reproduced from these alone.

The mock version is `gpu-mock/<n>+<digest>`, where the digest is taken over the mock list
(`oracle/check_mocks.py`, `mocks_digest`). A list that changes without a new version fails CI, so
a trace's recorded version names the exact mock list it ran under.

## Golden traces

A golden trace (`core/sched/testdata/golden/`) is the oracle's output for a scenario, committed
so the Go tests can run without Python. It is never overwritten without review:
`oracle/run_scheduler.py` refuses to write into `testdata/golden/` unless
`BLIS_VLLM_ALLOW_GOLDEN_UPDATE=1`, and CI (`goldens-reviewed`, `oracle/check_goldens.py`) fails a
push or pull request with a commit that changes the decisions an existing golden records (its
`steps` or `finished`), or deletes one, without a `golden-reviewed: <reviewer>` trailer naming
the person who checked it. A new golden for a new scenario, or a refresh of only the
`provenance` block, needs no trailer; the check verifies the decisions are byte-for-byte
unchanged. It also fails a golden whose `blis_vllm_commit` is not a commit in the pushed
history, so a golden's provenance always names a commit anyone can check out: a refresh takes
its traces from the `oracle-traces` workflow run on main (`oracle/refresh_provenance.py`). The differential test also refuses a golden whose recorded
scenario hash, vLLM version and commit, or mock version does not match the scenario file and the
pins it runs against, so a trace can only count as evidence for the scenario it came from.
CI re-runs the pinned vLLM release on every scenario and fails if any committed golden's
decisions differ from that fresh trace (`oracle/compare_goldens.py`), so a golden made on a
developer's machine counts only while the release itself still produces it.

## The compatibility engine's tests

`compat/` is inference-sim's engine as it ran before the extraction (#2021 B12 keeps the
existing 0.29.0 and 0.31.0 scenarios byte-identical, D7). It is not diff-tested against vLLM
v0.32.0, which is `core/`'s job: its correctness claim is that it is unchanged. Its own unit,
property and metamorphic tests moved with it (verbatim, with their source commit in each
header), and inference-sim's byte gate shows the simulator's output is identical before and
after the move. `compat/kernelmodel`'s tests read the copies under `testdata/`. The tests that
drive a whole simulator (event loop and engine together) stay in inference-sim. `compat/build`
(the configuration code that moved out of inference-sim's `cmd/`) keeps the moved unit tests of
its pure functions; its wrappers, which `cmd/` reached only through whole `blis run` and
`blis replay` invocations, gained direct tests here, each mutation-checked.

## Target

Zero mismatches. A known mismatch is listed with its earliest divergence in the repository's
status, never silently tolerated.

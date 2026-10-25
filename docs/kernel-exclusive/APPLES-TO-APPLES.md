# What the accuracy comparison reproduces, and what it does not

Each point the scorers simulate is one InferenceX benchmark run at one concurrency. Its ground
truth is that run's `mean_ttft` and `mean_tpot` as SemiAnalysis's database dump records them
(`inferencex_absolutes.json`, `db-dump/2026-09-28`). AISimulate's and AIC's predictions come from
NVIDIA's artifact. BLIS's are simulated by `blis replay` on the point's main phase. This document
says exactly what that replay reproduces, how each claim is checked, and what remains approximate.

## Sources

Citations are `file:line@commit` in [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX).
The measured runs are dated 2026-02-25 to 2026-08-07 (`source_run_date` in
`inferencex_engine_settings.json`). In that window the client was
`utils/bench_serving/benchmark_serving.py`, launched by `benchmarks/benchmark_lib.sh`
`run_benchmark_serving`. (It moved to `inferencex-e2e/infx/bench_serving/` in `bee3405c`,
2026-09-26.) Every commit that touched the client from `d053ae24` (2026-01-21, which added the
warm-up) to `8b4dcfae` (2026-07-09) was read. None changed any line cited below other than moving
it. Line numbers are at `08d7a9cd` (2026-06-11), and `d053ae24` and HEAD (`9fa76586`) agree.

| What | Where |
|---|---|
| seed | `benchmark_serving.py:808-809@08d7a9cd`: `random.seed(args.seed); np.random.seed(args.seed)`; `:1058`: `--seed` default `0` |
| the launcher passes no `--seed`, `--random-prefix-len` or `--use-chat-template` for these recipes | `benchmarks/benchmark_lib.sh:352-371@f4a002ef`. The recipes (`gptoss_fp4_h200.sh`, `minimaxm2.5_fp8_h200.sh`, `kimik2.5_fp4_b200.sh`, `dsv4_fp4_b300_vllm.sh`, ...) pass `--backend vllm`, `--random-range-ratio "$RANDOM_RANGE_RATIO"`, `--num-prompts $((CONC*10))` and `--max-concurrency "$CONC"` |
| range ratio | `RANDOM_RANGE_RATIO: 0.8` in the `.github/workflows/benchmark-tmpl.yml` job env at every run date (`:67@f267602f` ... `:153@8d7c906a`) |
| one process per point | the workflow runs one job per `conc` (`benchmark-tmpl.yml`, `CONC: ${{ inputs.conc }}`), and each job runs the client once |
| prefix draw | `benchmark_serving.py:217@08d7a9cd`: `np.random.randint(0, vocab_size, size=prefix_len)`, with `prefix_len` defaulting to 0 |
| length draws | `:228-235@08d7a9cd`: `lower = int(seq_len * range_ratio)`, then `np.random.randint(lower, upper+1, size=num_prompts)`; `input_lens` first, then `output_lens` |
| prompt construction | `:179-196@08d7a9cd` (parallel path) and `:298-320` (serial): token ids `(offsets[i] + i + j) % vocab_size`, then decode, re-encode, and pad or truncate up to 10 times |
| warm-up | `:522-556@08d7a9cd`: `test_input` from `input_requests[0]`; `num_warmups` tasks of it under a semaphore of `max_concurrency`; `_ = await asyncio.gather(*warmup_tasks)` |
| main phase | `:595-632@08d7a9cd`: a fresh `asyncio.Semaphore(max_concurrency)`; one task per request of `input_requests`, in order (`get_request` yields without waiting at `--request-rate inf`); `await asyncio.gather(*tasks)` |
| timing | `backend_request_func.py:270-319@08d7a9cd` (the `vllm` backend is `async_request_openai_completions`): `st` is taken after the semaphore is acquired; `ttft` at the first content chunk; `latency = most_recent_timestamp - st`, i.e. to the LAST content chunk; `max_tokens = output_len` and `ignore_eos` are set |
| metrics | `benchmark_serving.py:420-422, 468-474@08d7a9cd`: `tpot = (latency - ttft) / (output_len - 1)` for `output_len > 1`; `mean_ttft = mean(ttfts)` and `mean_tpot = mean(tpots)` over the successful main-phase requests |

One assumption underlies the recipe-level rows: that each run used the `main` recipes of its date.
The database dump records the run date, not the commit, and a workflow can be dispatched from
another ref.

## The workload: the benchmark's seed-0 target vectors

Nothing in `benchmark_serving.py` draws from numpy's global generator between `np.random.seed(0)`
and the length draws; the only code that runs in between is the tokenizer load. A size-0 prefix
draw consumes no generator state (checked with numpy 2.5.4: the state is unchanged). So each
point's target lengths are

```
np.random.seed(0)
input_lens  = np.random.randint(int(ISL*0.8), ISL+1, size=10*CONC)
output_lens = np.random.randint(int(OSL*0.8), OSL+1, size=10*CONC)
```

`scripts/gen_inferencex_workloads.py` makes exactly these calls for every scored
`(ISL, OSL, CONC)` and writes the vectors, in request order, to
`testdata/inferencex-workloads/seed0.json`; its `--check` flag regenerates and diffs.
`sim/kernelmodel/harness/fixture_test.go` recomputes every vector of every fixture with an
independent Go implementation of MT19937 and numpy's legacy bounded `randint` (masked rejection on
32-bit outputs). That implementation is itself checked against numpy over five ranges and three
seeds, so the fixture is verified in CI without Python. The legacy `RandomState` stream is frozen
by [NEP 19](https://numpy.org/neps/nep-0019-rng-policy.html), so the numpy version does not matter.
No scored point uses a chat template, which would shorten its targets (next section but one).

## The sent prompt lengths

The client does not send the target length directly. It builds a prompt from token ids, decodes
it to text, and re-encodes it until it reaches the target, then sends what it reached.
`scripts/inferencex_prompt_lengths.py` reproduces that construction offline for every logged vLLM
point:

- **Client version.** It uses the version live on the run's date. Before `1780296b` (2026-04-21),
  the serial loop pads from the global numpy stream. From `1780296b`, padding comes from
  `local_rng` and the parallel path's per-chunk `RandomState`, at `min(cpu_count, 8) = 8` workers;
  the dump does not record the worker count.
- **Tokenizer.** It uses the point's tokenizer at two pinned Hugging Face revisions: the one
  current at the run date, and today's.

It writes `testdata/inferencex-workloads/prompt-lengths.json`. That file records, per point, the
largest `|sent - target|` and the tokens a server adds with `add_special_tokens=True` (the
completions default). In its header it records every revision with the sha256 of every file read,
and the numpy, transformers and tokenizers versions.

| tokenizer repository | pinned revisions (role) | points | max \|sent - target\| | server-added tokens |
|---|---|---|---|---|
| `MiniMaxAI/MiniMax-M2.5` | `1248334b` (run dates 2026-03-01)<br>`f710177d` (run dates 2026-03-28, 2026-05-17, 2026-05-31, 2026-06-05; today) | 69 | 0 | 0 |
| `deepseek-ai/DeepSeek-V4-Pro` | `b5968e91` (run dates 2026-07-12; today) | 28 | 0 | 0 |
| `nvidia/Kimi-K2.5-NVFP4` | `0fd0a5e6` (run dates 2026-06-07, 2026-08-07; today) | 44 | 0 | 0 |
| `nvidia/MiniMax-M2.5-NVFP4` | `b6220d65` (run dates 2026-06-04; today) | 26 | 0 | 0 |
| `nvidia/MiniMax-M3-NVFP4` | `90146408` (run dates 2026-06-26, 2026-07-02, 2026-07-06, 2026-08-04; today) | 50 | 0 | 0 |
| `openai/gpt-oss-120b` | `b5c939de` (run dates 2026-02-25, 2026-03-27, 2026-05-17, 2026-05-30, 2026-05-31; today) | 143 | 0 | 0 |
| `meta-llama/Llama-3.1-70B-Instruct` | not checked: gated, and its runs have no engine log (no run date, no client version) | 0 | -- | -- |

Computed with numpy 2.2.6, transformers 4.57.1, tokenizers 0.22.2; 360 points, 7 pinned revisions. The run-date and today's revision of a repository are listed separately where they differ; where they hold byte-identical tokenizer files (every sha256 equal) the construction ran once.

A point is **verified** when every pinned revision gives zero on both counts. For a verified
point, the sent prompt length is exactly the target the fixture replays.
`TestThePromptLengthEvidenceIsPinnedAndReplayed` checks the file's provenance, and that every
verified point's vectors are the replayed fixture's. `TestEveryMeasuredOrResolvedPointSentItsTarget`
checks that every MEASURED-OR-RESOLVED point is verified.

A point the file does not hold is not verified and carries the gap
`sent prompt lengths not verified equal to the replayed targets`, which keeps it out of STRICT and
MEASURED-OR-RESOLVED. That is every Llama-3.1-70B point: its runs captured no engine log, so it has
no run date or client version, and its tokenizer is gated.

Two assumptions remain. The client's transformers is taken to be v4 (the runs here use the version
in the file's header). `backend_request_func._fix_tokenizer_for_sglang` exists because v5 rebuilds
some tokenizers' pre-tokenizer, and it is a no-op on v4. And a run that served a local checkpoint
path is taken to have the tokenizer of the Hugging Face repository of the same name.

## The main phase: order, concurrency and count

Each point runs
`blis replay --session-mode closed-loop --concurrent-sessions CONC --total-sessions 10*CONC` on a
TraceV2 whose record `i` is benchmark request `i`: a one-round session at exactly the fixture's
lengths, with no prefix group and no deadline. The pool injects records `0..CONC-1` at t=0, and
admits record `k`, in trace order, at the completion that frees a slot
(`workload.BuildSessionPool`, `SessionPoolDriver.OnComplete`). That is the client's semaphore at
`--request-rate inf`. There is no `--shuffle-corpus`, no clone (the total equals the trace length),
and no think time.

`TestThePoolIsTheClientsSemaphore` checks this on the code `blis replay` runs, and
`TestAReplayedPointIsTheBenchmarksMainPhase` checks it end to end through the binary.

**Output length.** The client sets `--ignore-eos` and `max_tokens = output_len`, so every request
emits exactly its target. Each simulated request is given exactly that many output tokens, and the
scorer refuses a run in which any request ran other lengths or fewer than `10*CONC` completed.

**Metrics.** TTFT is the mean of BLIS's `ttft_ms` over the `10*CONC` main-phase requests. TPOT is
the mean of BLIS's per-request `itl_ms` over the requests with more than one output token. The
two line up with the client's as follows:

- BLIS's request clock starts when the pool admits the request, just as the client's `st` is taken
  when its task acquires the semaphore.
- `ttft_ms` runs to the emission of the first token.
- `itl_ms` is the sum of the request's inter-token intervals divided by `output_len - 1`
  (`sim/engine.go`, `recordRequestCompletion`). That is the client's
  `(latency - ttft) / (output_len - 1)`, because the client's `latency` stops at the last content
  chunk.
- BLIS's `e2e_ms` also includes `PostDecodeFixedOverhead` after the last token, so TPOT is not
  derived from it.
- Both means are taken over requests, not pooled over tokens.

## The warm-up

The client sends `2*CONC` copies of request 0, waits for all of them to finish, and only then
starts the main phase. No warm-up request is in any metric. What matters is the state a drained
warm-up leaves in the simulator. From the code paths every point runs (single-tier GPU KV cache,
placement by requests in flight through the embedded llm-d router, FCFS, the router's default
admission, no offload tier):

- **KV cache.** It keeps released blocks' hashes (`sim/kv/cache.go`, `ReleaseKVBlocks`).
  Cross-request reuse reads them only in `sim/batch_formation.go`, which skips `GetCachedBlocks`
  when prefix caching is disabled. The other readers are the offload tiers and a router
  configuration with a prefix-cache producer, and these runs use neither. Request 0's prompt is the only main-phase
  prompt the warm-up cached, because no other main prompt shares a block with it.
- **Router.** Nothing: every warm-up copy has completed, so every replica has zero requests in
  flight when the main phase starts (`sim/kernelmodel/harness/vllm-dp-lb.epp.yaml`).
- **Everything else.** The FCFS scheduler, batch formation and the kernel adapter hold no
  per-request state, no step price reads the clock, and a drained engine holds no KV block.

**With prefix caching disabled**, the warm-up leaves nothing the main phase reads. The point is
scored on the main phase from an empty engine. This is every point whose run passed
`--no-enable-prefix-caching`.

**With prefix caching enabled**, the point is scored on the main phase run **after** the warm-up's
state, built with existing trace features only. The trace prepends `dp` warm-up sessions:

- Each session's round 0 is a copy of request 0 at t=0: its prompt as a shared prefix group, and
  its output length.
- Each session's round 1 is a one-token holder far in the future. It keeps the session's pool slot,
  so a copy's completion admits no main request early.
- Main request 0 is in the same prefix group, so its prompt is token-for-token the copy's.
- Every main request is recorded at 10^4 s. The pool (`CONC + dp`) therefore starts the first
  `CONC` together and in order, on an idle engine whose replicas each hold request 0's prompt. The
  router counter has moved by `dp`, so every main request lands on the replica it would land on
  from a fresh counter.

This is the drained warm-up's state, with `dp` copies standing in for `2*CONC`. Three tests check
that, rather than assume it:

- `TestCheckPrimedRefusesAnUndrainedWarmup`: the scorer refuses a run in which a copy had not
  completed before the main phase, main request 0 did not start with it, or a holder round entered
  the engine before the main phase ended.
- `TestThePrimedMainPhaseDoesNotDependOnHowManyCopiesRan`: one copy and the benchmark's `2*CONC`
  copies give identical main-phase observations. So the copy count leaves no other state behind.
- `TestADrainedCopyLeavesOnlyRequestZerosPromptReadable`: on the KV cache itself, a drained copy
  leaves no block in use, request 0's prompt readable in full blocks, and no other main prompt
  readable.

The empty-engine run of a prefix-caching point is printed beside it as information only and is
never scored. A drain-gated two-phase trace format, which would replay the warm-up itself, is
tracked as blis-schemas#56 and inference-sim#1926. What the warm-up does on real GPUs beyond the
prefix cache (first-call compilation, allocator growth) is outside any simulator; the client
discards the warm-up precisely so that none of it reaches the measured phase.

## The run settings

Each point's scenario variant carries what the measured engine was launched with, from its own
engine-args log: `max_num_seqs`, `max_num_batched_tokens`, `block_size`, `enable_prefix_caching`,
`max_model_len`, `gpu_memory_utilization` and `parallel.enable_expert_parallel`.

A setting the run did not pass takes vLLM's own default, and the coverage report says which:

- `enable_expert_parallel` and `gpu_memory_utilization` have unconditional defaults: expert
  parallelism off, and memory utilisation 0.9 before vLLM v0.20.0 and 0.92 from it, chosen by the
  engine version the log reports. An engine that was not passed them ran exactly these values, so
  this is no gap.
- `max_num_seqs`, `max_num_batched_tokens` and `block_size` resolve by rules this harness restates
  (`vllm_defaults.go`, block size 16). These are reported as **resolved**.

`max_cudagraph_capture_size`, passed on 298 of the 360 logged points, has no field in the
blis-schemas deployment, so it cannot be simulated as passed. It is a gap on every such point.

## What else is not reproduced

**The client's send gap.** The simulator admits the next request at the engine step that emitted
the previous request's last token. The client sends it after the response stream closes and the
asyncio semaphore wakes the next task. That interval is not modelled.

**Points with no engine log.** Llama-3.1-70B's runs captured no server log, so what they passed is
unknown. Their settings are vLLM's resolution, their prefix caching is vLLM's default (on), and
their sent lengths are unverified. They are scored only in the approximate tier.

**Other engines.** sglang and TensorRT-LLM points are not simulated, because BLIS models vLLM.

**Speculative decoding and chat templates.** No scored point uses either. Every scored vLLM sweep
has `spec_method: none`. At the last `main` commit on or before each of the 16 run dates,
`git grep -lE 'use-chat-template|--dsv4' <commit> -- benchmarks/single_node` finds the flag (or
`--dsv4`, which implies it) only in these recipes, none of them a scored vLLM sweep:

- speculative `*_mtp.sh` recipes;
- agentic recipes;
- `qwen3.5_fp8_b300.sh`;
- from 2026-06-26, two `minimaxm3_*_mi355x_atom.sh`.

## Tiers

The coverage report (`-gaps`) is computed before scoring. Each sweep is put in a tier by its worst
point. A sweep is scored whole or not at all, because its shape error divides every point by the
anchor's simulation.

| Tier | Contains |
|---|---|
| STRICT | Every point has an engine log, and every setting it passed is carried into the variant. The four admission settings and `max_model_len` were measured. Expert parallelism and `gpu_memory_utilization` were measured or are vLLM's unconditional default. The sent prompt lengths are verified equal to the targets. No other gap. |
| MEASURED-OR-RESOLVED | As STRICT, except that an unpassed admission setting may take the value vLLM resolves by a rule this harness restates. |
| ALL (approximate) | Every scoreable vLLM sweep. This includes sweeps with a setting not carried or not expressible, a scenario fallback, no engine log, or unverified sent lengths. |

Each table includes the stricter tiers, and within a table every arm (BLIS, AISimulate, AIC) is
scored on the same points. A figure labelled ALL is not a statement about the measured runs as they
ran. `metricscore` prints each tier's N, the sweeps in the two strict tiers with the resolved
settings that keep each out of STRICT, and the figures; `RESULTS.md` records them.

## The error definition

Shape error is AISimulate's own definition (`scripts/build_e2e_accuracy_overview.py:150-181`): each
side is divided by its own value at the sweep's lowest concurrency, that anchor is excluded, and
the errors are pooled over sweeps. Applied to AISimulate's own per-point data over the whole
1,137-point snapshot, it returns the 10.05% `tpot_shape_error_pct` that AISimulate publishes; the
anchor-inclusive variant does not (`TestAISimulatesPublishedShapeErrorIsReproducible`, in
`parity_test.go`). mape compares absolute latency at every point against
`inferencex_absolutes.json`. Both errors are computed from the same scored observation per point
(`TestThePrimedObservationEntersBothErrors`, in `cmd/metricscore`).

## The draw

`cmd/sensitivity` re-scores the sweeps with the vectors numpy draws at seeds 1 to 4 from the same
intervals. Those are other workloads, not repeated observations of the benchmark's seed 0. Their
spread shows how far a figure scored against some other draw could move for that reason alone,
which is the uncertainty that replaying the exact seed-0 vectors removes. They relax nothing.

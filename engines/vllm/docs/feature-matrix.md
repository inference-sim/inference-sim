# Feature matrix

Every vLLM feature, knob and configuration our sources use, with where vLLM v0.32.0 defines it,
which repository owns it, how it is proven, and its status. This is the finite list #2021 B9
asks for: a row is **supported** only when its oracle tier passes at zero mismatches, **excluded**
only with a written reason it cannot change a simulated result for our models, hardware and
settings, and **follow-up** only with an issue link. Until then it is **planned** (in this run,
not yet proven) or **in progress**.

A **supported** row names its proof: the oracle tier, the scenarios and fuzz seeds it passed at
zero mismatches against vLLM v0.32.0, and the mutants (deliberate one-line bugs, `docs/testing.md`)
the differential and fuzz tests caught for it, so a row cannot be marked supported by a test that
cannot fail.

## How to read a row

- **Sources.** `IX` = the InferenceX run index (`inferencex-e2e/configs/nvidia-master.yaml` and
  `amd-master.yaml` at InferenceX `7afd3b85`, 54 vLLM-family run definitions, and the 63 recipes
  they reference); the number is how many of the 54 run definitions use the feature. `IR` = our
  internal deployment and campaign reports (30 pages; not public). `AD33` =
  [accuracy-dashboard#33](https://github.com/inference-sim/accuracy-dashboard/discussions/33)
  (scoring gaps; its number is blocked rows). `BL` = inference-sim models it today. In the
  last table `SN/` and `MN/` abbreviate InferenceX's
  `inferencex-e2e/benchmarks/{single,multi}_node/srt-slurm-recipes/`.
- **v0.32.0 citation.** `vllm/<path>:<line>@5df63a72`, where the field, flag or the code acting on
  it is defined. No citation means the feature is not in v0.32.0, which makes it a follow-up.
- **Owner.** `vllm` = this repository; `schemas` = blis-schemas (a field or contract);
  `kernel` = blis-latency-kernel (pricing); `sim` = inference-sim (time, cluster, reports);
  `router` = the embedded llm-d router; `data` = blis-catalog or blis-registry (out of scope:
  "needs data", follow-up).
- **Tier.** The oracle tier that proves the row (`docs/testing.md`): T1 configuration, T2
  scheduler and KV, T3 EngineCore, T4 connectors and offload, T5 metrics; `—` for a row no
  behavioural tier applies to (pricing, routing, or an excluded row).
- **Compared.** The fields the differential test compares for the row.

## Scheduler and batch formation

| Feature | Sources | v0.32.0 citation | Owner | Tier | Compared | Status |
|---|---|---|---|---|---|---|
| Continuous batching, `max_num_seqs` | IX 51, IR, BL | `vllm/config/scheduler.py:62@5df63a72` | vllm | T2 | tokens per request, running set | supported (T2: 14 scenarios + 2000 fuzz seeds, 0 mismatches; mutant M1 caught) |
| `max_num_batched_tokens` (token budget) | IX 51, IR, BL | `vllm/config/scheduler.py:48@5df63a72` | vllm | T2 | tokens per request | supported (T2: 14 scenarios + 2000 fuzz seeds, 0 mismatches; mutant M2 caught) |
| `max_num_scheduled_tokens` | IX via defaults | `vllm/config/scheduler.py:55@5df63a72` | vllm | T2 | tokens per request | planned |
| Chunked prefill (`enable_chunked_prefill`) | IX 4 (explicit), IR, BL | `vllm/config/scheduler.py:134@5df63a72` | vllm | T2 | tokens per request | supported (T2: 14 scenarios + 2000 fuzz seeds, 0 mismatches; mutant M3 caught) |
| `long_prefill_token_threshold` | IX 8, IR, BL | `vllm/config/scheduler.py:79@5df63a72` | vllm | T2 | tokens per request | supported (T2: 14 scenarios + 2000 fuzz seeds, 0 mismatches; mutant M4 caught) |
| `long_prefill_token_threshold_adaptive` | IX via defaults | `vllm/config/scheduler.py:86@5df63a72` | vllm | T2 | tokens per request | planned |
| Lone-request long-prefill exemption | BL | `vllm/v1/core/sched/scheduler.py:2608@5df63a72` (`get_request_counts`) | vllm | T2 | tokens per request | supported (T2: 14 scenarios + 2000 fuzz seeds, 0 mismatches; mutant M5 caught) |
| Preemption (recompute), FCFS victim | IR, BL | `vllm/v1/core/sched/scheduler.py:1571@5df63a72` | vllm | T2 | preempted ids, order | supported (T2: 14 scenarios + 2000 fuzz seeds, 0 mismatches; mutant M6, M12 caught) |
| Scheduling policy `fcfs` / `priority` | IR (router priority bands), BL | `vllm/config/scheduler.py:21,159@5df63a72`; `vllm/v1/core/sched/request_queue.py:75,131@5df63a72` | vllm | T2 | admission order, preempted ids | supported (T2: 14 scenarios + 2000 fuzz seeds, 0 mismatches; mutant M8 caught) |
| Priority preemption victim | BL | `vllm/v1/core/sched/scheduler.py:796-808@5df63a72` | vllm | T2 | preempted ids | supported (T2: 14 scenarios + 2000 fuzz seeds, 0 mismatches; mutant M7, M22 caught) |
| `max_num_active_seqs` | (defaults) | `vllm/config/scheduler.py:69@5df63a72` | vllm | T2 | running set | planned |
| `watermark` (free-block reserve) | (defaults, 0.0) | `vllm/config/scheduler.py:196@5df63a72` | vllm | T2 | admission | planned |
| `max_model_len` cap and `check_stop` | IX 33, IR, BL | `vllm/v1/core/sched/scheduler.py:2465@5df63a72` (`_handle_stopped_request`) | vllm | T2 | finished ids, reasons, output tokens | supported (T2: 14 scenarios + 2000 fuzz seeds, 0 mismatches; mutant M10, M11 caught) |
| DP prefill throttling, `prefill_schedule_interval` | IX 8, AD33 46 | `vllm/config/scheduler.py:203@5df63a72`; `vllm/v1/engine/core.py:2318-2325@5df63a72` | vllm | T3 | tokens per request per rank | planned |
| `stream_interval` | IX 22 | `vllm/config/scheduler.py:213@5df63a72`; acts in `vllm/v1/engine/output_processor.py:198,253-255@5df63a72` | sim | — | client-side token delivery | follow-up (https://github.com/inference-sim/inference-sim/issues/2035): an API-server delivery rule (vllm/v1/engine/output_processor.py:324-343@5df63a72), not an engine decision; inference-sim, which owns client-side delivery, applies it |
| `max_num_queued_reqs` (HTTP 503) | — | `vllm/config/scheduler.py:92@5df63a72` | — | — | — | excluded: an API-server admission limit (vllm/config/scheduler.py:92@5df63a72), not a scheduling decision; no source sets it, so it cannot change a simulated result for our deployments |
| `scheduler_cls` (custom scheduler) | IX 1 (`AsyncScheduler` named explicitly) | `vllm/config/scheduler.py:177@5df63a72` | vllm | T3 | as async scheduling | follow-up (https://github.com/inference-sim/blis-vllm/issues/3): the one value a source uses is vLLM's own AsyncScheduler, which is the async-scheduling path of tier 3; a custom class is excluded by #2021 B10 ("Custom scheduler: Not now") |
| SJF and reverse-priority queues | BL only | not in v0.32.0 (`vllm/config/scheduler.py:21@5df63a72` lists `fcfs`, `priority`) | — | — | — | excluded: invented by inference-sim, not vLLM (#2021 B9 "Drop"); vLLM v0.32.0 accepts only fcfs and priority (vllm/config/scheduler.py:21@5df63a72; rules/v0_32 SchedulerPolicies, T1 MatchSource); blis-vllm has neither |

## KV cache and prefix caching

| Feature | Sources | v0.32.0 citation | Owner | Tier | Compared | Status |
|---|---|---|---|---|---|---|
| Block pool, free queue, null block | BL | `vllm/v1/core/block_pool.py:187@5df63a72`; `vllm/v1/core/kv_cache_utils.py:246@5df63a72` | vllm | T2 | free block count, block contents by hash and ref count | supported (T2: 14 scenarios + 2000 fuzz seeds, 0 mismatches; mutant M16, M17, M19, M20 caught) |
| `block_size` | IX 31, IR, BL | `vllm/config/cache.py:74@5df63a72` | vllm | T2 | block contents | supported (T2: block sizes 1-16 across 14 scenarios + 2000 fuzz seeds, 0 mismatches) |
| Prefix caching (`enable_prefix_caching`) | IX 30, IR, BL | `vllm/config/cache.py:141@5df63a72` | vllm | T2 | prefix-hit tokens, block contents | supported (T2: 14 scenarios + 2000 fuzz seeds, 0 mismatches; mutant M15, M18 caught) |
| Block hash with LoRA and cache-salt keys | BL (hash lacks them) | `vllm/v1/core/kv_cache_utils.py:592-595,647,695@5df63a72` | vllm | T2 | prefix-hit tokens | supported (T2: byte-identical to hashes vLLM v0.32.0 computed, `core/kv/testdata/block_hashes.json`, 6 cases incl. LoRA and cache salt; fixes inference-sim's LoRA-blind hash) |
| `prefix_caching_hash_algo` | (defaults, sha256) | `vllm/config/cache.py:143@5df63a72` | vllm | T2 | block hashes | supported for `sha256` (the default, which every scenario and the oracle use); other values (`sha256_cbor`, `xxhash`, `xxhash_cbor`, `vllm/config/cache.py:64@5df63a72`) excluded: no source sets one, and a hash function changes block identity only, never which blocks hit |
| `prefix_match_unit` | IX 9 | `vllm/config/cache.py:90@5df63a72`; `vllm/v1/core/kv_cache_utils.py:817-834@5df63a72` | vllm | T2 | prefix-hit tokens | planned |
| `prefix_cache_retention_interval` | IX 1 (flag; 21 set an env var absent from v0.32.0) | `vllm/config/cache.py:159@5df63a72`; `vllm/v1/core/kv_cache_coordinator.py:185@5df63a72` | vllm | T2 | prefix-hit tokens | planned |
| `kv_cache_memory_bytes` (fixed KV size) | IX 3, AD33 18 | `vllm/config/cache.py:257@5df63a72` | vllm + kernel | T1 | KV capacity | follow-up (https://github.com/inference-sim/blis-latency-kernel/issues/86): the KV budget comes from the kernel (#2021 B5), which does not read the field yet; blis-schemas#102 carries it |
| `gpu_memory_utilization` | IX 45, IR, BL | `vllm/config/cache.py` (field) | kernel | — | KV capacity (priced) | follow-up (https://github.com/inference-sim/blis-latency-kernel/issues/83, https://github.com/inference-sim/blis-latency-kernel/issues/84): priced by the kernel, which sizes the KV budget from it at its vLLM 0.31.0 behaviour; not a blis-vllm oracle tier (#2021 B1: performance belongs to the kernel) |
| `cache_dtype` | IX 46, IR, BL | `vllm/config/cache.py:122@5df63a72` | kernel | — | bytes per block | follow-up (https://github.com/inference-sim/blis-latency-kernel/issues/83, https://github.com/inference-sim/blis-latency-kernel/issues/84): the kernel prices bytes per block per dtype; the 0.32 rules pack validates the values (rules/v0_32 CacheDTypes, T1 MatchSource at 5df63a72) |
| Hybrid KV cache manager (full + sliding window + mamba groups) | IX 14, IR, AD33 29 | `vllm/v1/core/kv_cache_coordinator.py:609@5df63a72`; `vllm/v1/core/single_type_kv_cache_manager.py:729,933,1415@5df63a72` | vllm | T2 | block contents per group | planned |
| `disable_hybrid_kv_cache_manager` | IX 6, AD33 29 | `vllm/config/scheduler.py:182@5df63a72` | vllm | T2 | block contents | planned |
| `mamba_cache_mode`, `mamba_ssm_cache_dtype` | IX 6, IR | `vllm/config/cache.py:185,189@5df63a72`; `vllm/v1/core/sched/scheduler.py:377@5df63a72` (aligned split) | vllm + kernel | T2 | tokens per request (aligned split) | planned |
| `swa_bounded_replay` | IX 2, BL | `vllm/config/cache.py:250@5df63a72` | vllm | T2 | prefix-hit tokens | planned |
| KV events (`kv_events_config`) | IX 2, IR | `vllm/config/kv_events.py:11-14@5df63a72` | vllm → router | T2 | KV event stream | planned |

## Speculative decoding (acceptance is an input, #2021 B3)

| Feature | Sources | v0.32.0 citation | Owner | Tier | Compared | Status |
|---|---|---|---|---|---|---|
| DSpark slots | IX 39, IR, BL | `vllm/config/speculative.py:69@5df63a72` | vllm | T2 | tokens per request (K+1 verify), draft slots | planned |
| MTP slots | IX 4, IR, BL | `vllm/config/speculative.py:38@5df63a72` | vllm | T2 | tokens per request | planned |
| EAGLE3 with an external draft model (`speculative_config.model`) | IX 9, AD33 332 | `vllm/config/speculative.py:71,396@5df63a72` | vllm + schemas (#97) + data | T2 | tokens per request | planned; drafter graph needs data (follow-up) |
| `num_speculative_tokens` | IX 52, IR, BL | `vllm/config/speculative.py:150@5df63a72` | vllm | T2 | tokens per request | planned |
| Draft slots (`max_num_new_slots_for_drafting`, parallel drafting) | BL | `vllm/v1/core/sched/scheduler.py:613@5df63a72`; `vllm/config/speculative.py:478@5df63a72` | vllm | T2 | input budget | planned |
| Probabilistic drafting, block rejection sampling | IX 39 / 30 | `vllm/config/speculative.py:529,532@5df63a72` | vllm (accept knobs) | — | — | excluded from parity: acceptance is supplied (B3); the knobs are accepted and recorded as "acceptance supplied" |
| DSpark adaptive verification | IX 13, IR | `vllm/config/speculative.py:551@5df63a72`; acts in attention backends, `vllm/v1/attention/selector.py:143-151@5df63a72` | vllm (accept knob) + kernel | — | verify width priced | planned: the verifier's trimmed query length is a GPU-side decision; scheduling still budgets K+1 |
| Drafter KV dtype (`speculative_config.kv_cache_dtype`) | IX 8, AD33 29 | `vllm/config/speculative.py:382@5df63a72` | schemas + kernel | — | bytes | planned |
| Acceptance-dependent metrics | IR | `vllm/v1/spec_decode/metrics.py@5df63a72` | vllm | T5 | spec-decode counters given supplied acceptance | planned |

## EngineCore: async scheduling, batch queue, PP, DP

| Feature | Sources | v0.32.0 citation | Owner | Tier | Compared | Status |
|---|---|---|---|---|---|---|
| Async scheduling (on by default) | IX (default; 2 turn it off), IR | `vllm/config/vllm.py:1585-1644@5df63a72`; `vllm/v1/core/sched/async_scheduler.py:14@5df63a72` | vllm | T3 | tokens per request per step, in-flight batches | planned |
| Batch queue, `max_concurrent_batches` | (all async/PP runs) | `vllm/config/vllm.py:577-588@5df63a72`; `vllm/v1/engine/core.py:698-810@5df63a72` | vllm | T3 | batch order, in-flight count | planned |
| Pipeline parallel (PP2) | IX 2, IR | `vllm/config/parallel.py:139@5df63a72` | vllm + kernel | T3 | batch queue | planned |
| Data parallel lockstep, waves, dummy batches | IX 26, IR, BL | `vllm/v1/engine/core.py:2143,2168@5df63a72` | vllm | T3 | per-rank steps, waves | planned |
| DP load balancing internal / hybrid / external | IX 2 (external), IR, BL | `vllm/config/parallel.py:168,175@5df63a72` | vllm (internal DPLB) + router | T3 | rank choice | planned |
| Decode context parallel (DCP 8/16, `dcp_comm_backend: a2a`) | IX 10 | `vllm/config/parallel.py:385,397@5df63a72` | vllm + kernel | T2 | block contents (interleave) | planned |
| Expert parallel, EPLB, DBO, all2all backend | IX 25 / 1 / 1 / 4, IR | `vllm/config/parallel.py:189,205,257@5df63a72` | kernel | — | priced | follow-up (https://github.com/inference-sim/blis-latency-kernel/issues/83, https://github.com/inference-sim/blis-latency-kernel/issues/84): priced by the kernel (#2021 B9 "Elsewhere"); the startup refusals and resolution are T1: rules/v0_32 checks the all2all backends, DBO thresholds, EPLB divisibility and the new passthrough binding against the v0.32.0 source |
| `VLLM_USE_V2_MODEL_RUNNER` | IX 31 | `vllm/envs.py:311@5df63a72` | vllm | T3 | `max_concurrent_batches` with PP | follow-up (https://github.com/inference-sim/blis-vllm/issues/3): it changes max_concurrent_batches (vllm/config/vllm.py:577-588@5df63a72) and spaces PP decodes (vllm/v1/core/sched/async_scheduler.py:52-58@5df63a72), both EngineCore (T3); the V2 runner needs Triton, which the CPU oracle lacks, recorded in the issue |

## KV connectors and offload (state machines under virtual time)

| Feature | Sources | v0.32.0 citation | Owner | Tier | Compared | Status |
|---|---|---|---|---|---|---|
| NIXL P/D (`NixlConnector` = `NixlPullConnector`, `NixlPushConnector`) | IX 14, IR, BL | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:414@5df63a72`; `.../factory.py:177-189@5df63a72` | vllm | T4 | waiting-for-remote-KV set, admission, held blocks | planned |
| `MultiConnector` | IX 11, IR, BL | `vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:135@5df63a72` | vllm | T4 | which connector serves a load | planned |
| `OffloadingConnector` (CPU, `TieringOffloadingSpec`, `cpu_bytes_to_use`, `blocks_per_chunk`, LRU) | IX 1, IR, BL | `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:53@5df63a72`; `vllm/v1/kv_offload/tiering/spec.py:9,82,451@5df63a72`; `vllm/v1/kv_offload/cpu/policies/factory.py:87@5df63a72` | vllm | T4 | offload hits, loads, evictions | planned |
| Tiered-offload backpressure (decay, cooldown) | (Tiering spec) | `vllm/v1/kv_offload/tiering/backpressure.py:136-142,295@5df63a72` | vllm | T4 | admission under backpressure (virtual time) | planned |
| P2P sessions (handshake, idle deadlines) | (Tiering p2p) | `vllm/v1/kv_offload/tiering/p2p/session/session.py:133,195,277@5df63a72` | vllm | T4 | session state (virtual time) | planned |
| `SimpleCPUOffloadConnector` (`lazy_offload`) | IX 13 | `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:66,104@5df63a72` | vllm | T4 | offload hits | planned |
| `MooncakeStoreConnector` (`load_async`, `lookup_async`, `enable_cross_layers_blocks`) | IX 15, AD33 25 | `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:127@5df63a72`; `.../store/worker.py:1397,1553@5df63a72` | vllm | T4 | loads, lookups | planned |
| `MoRIIOConnector` (AMD P/D) | IX 1 | `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py:225@5df63a72` | vllm | T4 | as NIXL | planned |
| LMCache (`LMCacheConnectorV1`, `LMCacheMPConnector`) | IX 1 | `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:68@5df63a72`; `.../lmcache_mp_connector.py:461@5df63a72` | vllm | T4 | loads | planned |
| `kv_load_failure_policy` (`recompute` / `fail`) | IX 12 | `vllm/config/kv_transfer.py:88@5df63a72` | vllm | T4 | failed-load handling | planned |
| Connector registry (`behaves_like`) | BL (hard-coded lists) | `vllm/distributed/kv_transfer/kv_connector/factory.py:31@5df63a72` | schemas + vllm | T4 | — | planned |
| Eviction policies beyond `lru` (`arc`) | — | `vllm/v1/kv_offload/cpu/policies/arc.py@5df63a72` | vllm | — | — | follow-up (#2021 B10: "others are a follow-up") |

## LoRA

| Feature | Sources | v0.32.0 citation | Owner | Tier | Compared | Status |
|---|---|---|---|---|---|---|
| `max_loras` per-batch skip | BL | `vllm/config/lora.py:35@5df63a72`; `vllm/v1/core/sched/scheduler.py:897-946@5df63a72` | vllm | T2 | tokens per request, skip order | planned (replaces inference-sim's blocking gate) |
| `max_cpu_loras` (LRU of loaded adapters) | BL | `vllm/config/lora.py:42@5df63a72` | vllm | T2 | residency | planned |

## Metrics

| Feature | Sources | v0.32.0 citation | Owner | Tier | Compared | Status |
|---|---|---|---|---|---|---|
| Every computable `vllm:*` family (generated manifest) | IR (router reads them), BL (13 today) | `vllm/v1/metrics/loggers.py:528-1120@5df63a72`; `vllm/v1/spec_decode/metrics.py:217-258@5df63a72`; `vllm/v1/kv_offload/tiering/base.py:43-64@5df63a72` | vllm | T5 | names, labels, buckets, values, enablement | supported (T5, family inventory): docs/metrics-manifest.json lists all 130 families v0.32.0 defines (45 exact, 52 derived, 33 unavailable), generated from the source by cmd/metricsmanifest and regenerated in CI (job metrics-manifest); the 45 families vLLM's own PrometheusStatLogger registers on CPU match it in type and labels (oracle/check_manifest.py). Emitting and diff-testing each exact family's values: see the next row |
| `kv_cache_metrics`, `cudagraph_metrics` | IX 3 / 3 | `vllm/config/observability.py:62,82@5df63a72` | vllm | T5 | enablement | supported (T5, enablement): kv_cache_metrics registers the three kv_block_* histograms (vllm/v1/metrics/loggers.py:1036@5df63a72), recorded as their enabled_by in the manifest and registered in the live cross-check; cudagraph_metrics registers no vllm:* family (it gates logging only, vllm/v1/metrics/loggers.py:136@5df63a72) |
| Series needing hardware or processes (e.g. `corrupted_requests`, `engine_sleep_state`) | — | (manifest) | — | — | — | excluded: the manifest's 33 unavailable families, each with its written reason (real numerics, sleep mode, watermarking, multimodal, diffusion, API-server endpoints, connectors no source uses); none changes a scheduling, KV or timing result for our deployments |

## Elsewhere (not blis-vllm behaviour)

| Feature | Sources | v0.32.0 citation | Owner | Tier | Compared | Status |
|---|---|---|---|---|---|---|
| CUDA graph modes and capture sizes | IX 41, IR, BL | `vllm/config/compilation.py` | kernel | — | — | follow-up (https://github.com/inference-sim/blis-latency-kernel/issues/83, https://github.com/inference-sim/blis-latency-kernel/issues/84): priced by the kernel (#2021 B9 "Elsewhere"); the capture sizes a deployment resolves to are T1: rules/v0_32's cudagraph-capture-sizes rule and CUDAGraphCaptureSizes pass their v0.32.0 source check (vllm/config/vllm.py _set_cudagraph_sizes) |
| `decoder_replay_cudagraph_capture_sizes` | IX 6, AD33 18 | `vllm/config/compilation.py:565@5df63a72` | schemas + kernel | — | — | follow-up (https://github.com/inference-sim/blis-latency-kernel/issues/85): the field is in blis-schemas#102; pricing its graphs needs launch-cost coefficients (needs data) |
| `custom_ops` | IX 3, AD33 123 | `vllm/config/compilation.py:480@5df63a72` | schemas + kernel | — | — | follow-up (https://github.com/inference-sim/blis-latency-kernel/issues/85): the field is in blis-schemas#102; pricing needs per-chip rates (needs data) |
| `use_trtllm_attention`, `mla_prefill_backend`, attention backend | IX 5 / 11 / 23, AD33 84 / 78 | `vllm/config/attention.py:74,81@5df63a72` | schemas + kernel | — | — | follow-up (https://github.com/inference-sim/blis-latency-kernel/issues/85): the fields are in blis-schemas#102; pricing needs per-backend rates (needs data) |
| `linear_backend` | IX 1, AD33 75 | `vllm/config/kernel.py:350@5df63a72` | schemas + kernel | — | — | follow-up (https://github.com/inference-sim/blis-latency-kernel/issues/85): the field is in blis-schemas#102; pricing needs per-backend GEMM rates (needs data) |
| `moe_backend`, quantization, fp8/fp4/nvfp4 | IX 30, IR | (`vllm/config/kernel.py`, quantization) | kernel | — | — | follow-up (https://github.com/inference-sim/blis-latency-kernel/issues/62, https://github.com/inference-sim/blis-latency-kernel/issues/83): priced by the kernel; the accepted values are T1 (rules/v0_32 MoEBackends, Quantizations, CacheDTypes, checked against the v0.32.0 source) |
| Kernel env vars (`VLLM_FLASHINFER_ALLREDUCE_BACKEND`, `VLLM_MARLIN_USE_ATOMIC_ADD`) | IX 11 / 3, AD33 | `vllm/envs.py:225,192@5df63a72` | schemas + kernel | — | — | follow-up (https://github.com/inference-sim/blis-latency-kernel/issues/85): carried as env in blis-schemas#102; pricing needs data |
| Prefix scorer, P/D decider, flow control | IR | — | router | — | — | excluded from blis-vllm: routing and admission decisions of the embedded llm-d router (#2021 B9 "Elsewhere"), which inference-sim owns; blis-vllm supplies the vllm:* metrics and KV events the router reads |
| Dynamo and vLLM routers | IX 17 / 11 | — | router | — | — | follow-up (https://github.com/inference-sim/inference-sim/issues/2028) |
| Multimodal encoder scheduling, structured output | — | (`vllm/v1/core/encoder_cache_manager.py`) | — | — | — | excluded: no source serves multimodal or constrained-decoding requests; every source is a text agentic workload |
| Tool-call / reasoning parsers, `language_model_only`, tokenizer mode | IX 43 / 40 / 25 | — | — | — | — | excluded: parse responses after decoding; no scheduling, KV or timing effect in a token-level simulation |

## Used by a source but not in v0.32.0

Each gets a follow-up issue labelled "upstream after 0.32.0" or "custom-image only" (#2021 B9).

| Item | Sources | Where seen | Label | Status |
|---|---|---|---|---|
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` (env) | IX 21 | `SN/dsv4/vllm/b200-fp4-mtp/agentic.yaml:46@7afd3b85` | custom-image only | follow-up (https://github.com/inference-sim/inference-sim/issues/2026) |
| `VLLM_RPC_TIMEOUT` | IX 20 | `SN/dsv4/vllm/b200-fp4-mtp/agentic.yaml:50@7afd3b85` | custom-image only | follow-up (https://github.com/inference-sim/inference-sim/issues/2026) |
| `--monokernel` | IX 1 | `SN/dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml:52@7afd3b85` | upstream after 0.32.0 | follow-up (https://github.com/inference-sim/inference-sim/issues/2025) |
| `attention_config.use_fp4_indexer_cache` | IX 7 | `MN/dsv4/vllm/gb200-fp4/agentx/agg-llm-d.yaml:129@7afd3b85` | custom-image only | follow-up (https://github.com/inference-sim/inference-sim/issues/2026) |
| `moe_backend: deep_gemm_amxf4_mega_moe` | IX 4 | `SN/dsv4/vllm/b200-fp4-mtp/agentic.yaml:180@7afd3b85` | custom-image only | follow-up (https://github.com/inference-sim/inference-sim/issues/2026) |
| `additional_config.decode_backend: latency_specialized` | IX 2 | `SN/dsv41flash/vllm/b300-fp4-mtp/agentic.yaml:65@7afd3b85` | custom-image only | follow-up (https://github.com/inference-sim/inference-sim/issues/2026) |
| Other absent `VLLM_*` env vars and connector keys (11 more) | IX | survey list | custom-image only | follow-up (https://github.com/inference-sim/inference-sim/issues/2026, one issue listing every item) |

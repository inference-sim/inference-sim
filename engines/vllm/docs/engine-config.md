<!-- Moved from inference-sim/blis-schemas/docs/reference/deployment.md at 70246460a379f8b1898fbf6c36a3091364118950 (the engine section). -->

# The vLLM engine block

A deployment pool's `engine:` block, as `rules/engineconfig` decodes it (`Decode`, strictly) and
checks it (`Validate`). blis-schemas holds the block unopened; the keys below are vLLM's.


Every engine key is optional. Some are requests the engine may not honor:
`disable_custom_all_reduce`, `allreduce_backend`, `async_scheduling`,
`disable_cascade_attn` and the decode-context keys. A cost model reports what it resolved
in its `Resolution`. Which values each string key accepts depends
on the engine release and is checked by its rules pack (`rules/v0_*`).

Communication:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `all2all_backend` | string | no | How a mixture-of-experts (MoE) layer sends tokens to their experts and back. |
| `disable_custom_all_reduce` | boolean | no | Disable the custom all-reduce GPU kernel, which runs on the streaming multiprocessors (SMs) rather than the network. Omitted means false, so the kernel is used where the layout allows. |
| `allreduce_backend` | string | no | `custom` or `nccl`, stated directly instead of through `disable_custom_all_reduce`. Must not contradict it. |
| `dbo` | mapping | no | Dual-batch overlap, below. |

Scheduling and memory:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `block_size` | integer | no | KV block size in tokens. |
| `max_num_batched_tokens` | integer | no | Token budget per step. |
| `max_num_seqs` | integer | no | Sequence cap per step. |
| `long_prefill_token_threshold` | integer | no | The most tokens of one prefill scheduled in a step, so a long prompt cannot take the whole token budget from the decodes beside it. Not negative; omitted or 0 leaves prefills uncapped. The engine refuses one above `max_model_len`. 0.31.0 waives it for a request alone in the scheduler; 0.29.0 does not. |
| `max_model_len` | integer or `auto` | no | Context length. `auto` (or `-1`) asks the engine to fit the window to the KV memory it has left, which is a different request from omitting the key: omitted, the engine uses the model's whole window and refuses to start if it does not fit. |
| `gpu_memory_utilization` | number | no | Fraction of GPU memory the engine may use, in [0, 1]. Zero means the engine's default. |
| `scheduling_policy` | string | no | `fcfs` or `priority` in vLLM 0.29.0. |
| `async_scheduling` | boolean | no | Omitted means the engine decides through its own disqualifying conditions. |
| `frontend` | string | no | The API server that renders and tokenizes each prompt: one of the [frontends](vocabularies.md#frontends). Omitted, `python`. `rust` is vLLM's vllm-rs frontend, selected by the environment variable `VLLM_USE_RUST_FRONTEND=1`, not a flag. It changes how long a request waits for admission, per prompt token. |
| `cudagraph_mode` | string | no | CUDA graph capture strategy. |
| `max_cudagraph_capture_size` | integer | no | `compilation_config.max_cudagraph_capture_size`: the largest step, in tokens, a CUDA graph is captured for; a larger step runs eager. At least 1. Omitted, the release default, min(`max_num_seqs` x decode query length x 2, 512), or 1024 on SM100. The engine caps it at `max_num_batched_tokens` and truncates it to the largest size it generates. `cudagraph_mode: NONE`, or `--enforce-eager`, which has no field here, captures nothing. |
| `cudagraph_capture_sizes` | list of integers | no | `compilation_config.cudagraph_capture_sizes`: the exact step sizes to capture, each at least 1, not empty. Sizes above `max_num_batched_tokens` are dropped. With `max_cudagraph_capture_size` too, that must be the largest kept, or the engine refuses the pair. |
| `decoder_replay_cudagraph_capture_sizes` | list of integers | no | `compilation_config.decoder_replay_cudagraph_capture_sizes`, the decoder-replay graph sizes of a YOCO model such as DeepSeek-V4.1 (new in vLLM 0.32.0). Each at least 1. Omitted, generated from the sliding window. |
| `custom_ops` | list of strings | no | `compilation_config.custom_ops`: which ops run as custom kernels, `all`, `none`, or `+name` / `-name` entries. No entry may be empty. Omitted, the engine default. |
| `long_prefill_token_threshold_adaptive` | boolean | no | Floor the `long_prefill_token_threshold` cap at a fair share of the token budget (`max_num_batched_tokens` over the queued and running requests). Omitted means false. |
| `max_num_scheduled_tokens` | integer | no | The most tokens the scheduler issues in a step, where the model appends tokens to a batch, as speculative decoding does. Not negative; omitted or 0 uses `max_num_batched_tokens`. |
| `max_num_active_seqs` | integer | no | Caps the requests admitted to running without shrinking `max_num_seqs`. Not negative, and not above `max_num_seqs`; omitted or 0 means `max_num_seqs`. |
| `watermark` | number | no | Fraction of KV blocks kept free when admitting waiting or preempted requests, in [0, 1). Omitted means 0, the engine default. |
| `prefill_schedule_interval` | integer | no | Admit new prefills only once every N engine steps, aligned across data-parallel ranks. Not negative; omitted or 0 means the engine default, 1 (no throttling). |
| `stream_interval` | integer | no | Tokens the API server buffers before streaming them to the client. It changes when a client sees tokens, not what the scheduler does. Not negative; omitted or 0 means 1. |
| `disable_hybrid_kv_cache_manager` | boolean | no | Give every attention layer the same KV size even when they differ (full and sliding-window). Omitted lets the engine decide. |
| `kv_cache_memory_bytes` | integer | no | The KV cache's size per GPU in bytes, instead of deriving it from `gpu_memory_utilization`. Positive. |
| `prefix_match_unit` | integer | no | The finest token boundary a prefix-cache hit lands on; may be finer than the block size where every group's block size divides by it. Not negative; omitted or 0 means the block size. |
| `prefix_cache_retention_interval` | integer | no | Token interval between retained sliding-window and Mamba prefix-cache checkpoints; 0 keeps only semantic checkpoints. Not negative. |
| `kv_events` | mapping | no | KV-event publishing, below. |
| `max_loras` | integer | no | The most distinct LoRA adapters in one batch; a request whose adapter does not fit waits for a later step while those behind it run. Not negative; omitted or 0 means 1. |
| `max_cpu_loras` | integer | no | LoRA adapters kept loaded in CPU memory (an LRU set). Not below `max_loras`; omitted or 0 means `max_loras`. |

Weights and caches:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `quantization` | string | no | Weight format the engine serves, where it differs from the checkpoint's. Omitted means the checkpoint's own format. |
| `moe_backend` | string | no | The expert-kernel backend, `--moe-backend`. Omitted means `auto`. A mega-MoE backend is one of the conditions under which DeepSeek-V4.1 turns on its own sequence parallelism. |
| `cache_dtype` | string | no | KV cache format. |
| `enable_prefix_caching` | boolean | no | Omitted means true, vLLM's default. `false` describes a deployment without prefix caching. |
| `swa_bounded_replay` | boolean | no | DeepSeek-V4.1 SWA bounded replay (`--swa-bounded-replay` / `--no-swa-bounded-replay`). `true` keeps the sliding-window cache out of prefix caching and recomputes a hit's last sliding window; `false` lets the window cache take part in prefix caching (retained window tails). Omitted, the engine's resolution: on in vLLM 0.31.0 under the V2 model runner, without prefill context parallelism and not on ROCm. Only DeepSeek-V4.1 reads it. |
| `mamba_cache_dtype` | string | no | Format of a recurrent layer's state. |
| `mamba_ssm_cache_dtype` | string | no | Format of the SSM state alone, where it differs. |
| `mamba_cache_mode` | string | no | Whether recurrent state is fixed per sequence or grows with context. |
| `disable_cascade_attn` | boolean | no | Cascade attention reads a prefix shared by a batch once rather than once per request. Omitted means `true`, so it is disabled unless this is set to `false`. |
| `use_trtllm_attention` | boolean | no | `attention_config.use_trtllm_attention`: use or skip FlashInfer's TRT-LLM attention backend. Omitted auto-detects. |
| `mla_prefill_backend` | string | no | `attention_config.mla_prefill_backend`: `FLASH_ATTN`, `FLASHINFER` or `TRTLLM_RAGGED` in vLLM 0.32.0. Omitted selects automatically. |
| `linear_backend` | string | no | `kernel_config.linear_backend`, the GEMM backend of linear layers. Omitted is `auto`. |
| `env` | mapping | no | Engine environment variables the launch set that change what the engine runs, such as `VLLM_FLASHINFER_ALLREDUCE_BACKEND`, name to value. Each name must be an environment-variable name. A consumer refuses, or discloses as an assumption, every key it cannot honour. |

Speculation and experts:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `speculative` | mapping | no | Speculative decoding, below. |
| `eplb` | mapping | no | Expert-parallel load balancing, below. |
| `engram_config` | mapping | no | Where DeepSeek-V4.1's Engram tables live, below. Decides the residency of every model `Gather` whose `residency_control` is `engram_cpu_offload`. |

Decode-context parallelism. These have no effect when `dcp` is 0 or 1:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `dcp_comm_backend` | string | no | The collectives a decode step runs: `ag_rs` (the default) or `a2a` in vLLM 0.29.0. |
| `dcp_q_replicate` | boolean | no | Replicate the query projection of multi-head latent attention (MLA) on every rank, to skip a query all-gather. |
| `cp_kv_cache_interleave_size` | integer | no | Consecutive tokens one rank holds before the next takes over. Zero means unstated, which is a different request from 1. |

#### `engine.dbo`

<!-- fields: deployment.DBO -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `enable_dbo` | boolean | no | Split a batch so one half's communication overlaps the other's compute. |
| `dbo_decode_token_threshold` | integer | no | Batch size in tokens at or above which a uniform-decode batch is split. Omitted takes the engine's default. With `enable_dbo` on, vLLM 0.31.0 refuses a stated value below the two microbatches, 0 or 1. Not negative. |
| `dbo_prefill_token_threshold` | integer | no | Batch size in tokens at or above which any other batch is split. As for the decode threshold. |

#### `engine.eplb`

<!-- fields: deployment.EPLB -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `enable_eplb` | boolean | no | Enable expert-parallel load balancing. |
| `num_redundant_experts` | integer | no | Extra expert replicas. With load balancing on, the expert-parallel width must divide the experts plus these. |
| `window_size` | integer | no | Load-balancing window. |
| `step_interval` | integer | no | Steps between rebalances. |

#### `engine.speculative`

<!-- fields: deployment.Speculative -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `method` | string | yes | The draft method. Which names are valid depends on the engine release. |
| `num_spec_tokens` | integer | yes | Draft tokens per step, at least 1. |
| `draft_model` | string | no | The catalog model whose graph is the drafter, where the drafter is a separate checkpoint, as an EAGLE3 head is (vLLM's `speculative_config.model`). Omitted, the target graph's own speculator. Not blank. |
| `kv_cache_dtype` | string | no | The drafter's KV cache format where it differs from the target's. |
| `draft_sample_method` | string | no | `greedy` or `probabilistic`. Recorded: acceptance is supplied, so it does not change which drafts are accepted. |
| `rejection_sample_method` | string | no | `standard`, `synthetic` or `block`. Recorded, as `draft_sample_method`. |
| `enable_adaptive_verification` | boolean | no | Size the verification budget from per-request confidence (DSpark only). Omitted means false. |

#### `engine.engram_config`

The cost-bearing keys of vLLM's `--engram-config`. All three are optional and take the engine's
default when omitted. The rules pack for 0.31.0 refuses the block on a model with no Engram
lookups.

<!-- fields: deployment.EngramConfig -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `cpu_offload` | boolean | no | Keep the tables in pinned host memory, read over the host link. Omitted means true. `false` puts them in GPU memory. |
| `dp_shared_memory` | boolean | no | Keep one host copy per node, shared by its data-parallel replicas. Omitted lets the engine decide: it enables it when `cpu_offload` is on, `use_thp` is off, `dp` exceeds 1 and elastic EP is off. Without sharing, the node's co-located replicas split each table between them instead. |
| `use_thp` | boolean | no | Back a per-rank host table with transparent huge pages. Refused with `dp_shared_memory`. |

#### `engine.kv_events`

vLLM's `--kv-events-config`: whether the engine publishes the changes to the set of KV blocks
its prefix cache can serve. A router's precise prefix-cache scorer consumes the stream.

<!-- fields: deployment.KVEvents -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `enable_kv_cache_events` | boolean | yes | Publish KV-cache events. |
| `publisher` | string | no | `null` or `zmq`. Omitted, the engine's resolution: `zmq` when events are enabled. |
| `topic` | string | no | The topic consumers subscribe to. |


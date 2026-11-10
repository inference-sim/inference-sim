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
| `enable_dbo`, `dbo_decode_token_threshold`, `dbo_prefill_token_threshold` | | no | Dual-batch overlap, below. |

Scheduling and memory:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `block_size` | integer | no | KV block size in tokens. |
| `max_num_batched_tokens` | integer | no | Token budget per step. |
| `max_num_seqs` | integer | no | Sequence cap per step. |
| `long_prefill_token_threshold` | integer | no | The most tokens of one prefill scheduled in a step, so a long prompt cannot take the whole token budget from the decodes beside it. Not negative; omitted or 0 leaves prefills uncapped. The engine refuses one above `max_model_len`. 0.31.0 waives it for a request alone in the scheduler; 0.29.0 does not. |
| `max_model_len` | integer or `auto` | no | Context length. `auto` (or `-1`) asks the engine to fit the window to the KV memory it has left, which is a different request from omitting the key: omitted, the engine uses the model's whole window and refuses to start if it does not fit. |
| `gpu_memory_utilization` | number | no | Fraction of GPU memory the engine may use, in [0, 1]. Zero or unstated is vLLM's default, 0.92 in each release the engine models (`vllm/config/cache.py` at v0.29.0:111, v0.31.0:103, v0.32.0:102); `Model.EngineSources()` reports it. |
| `scheduling_policy` | string | no | `fcfs` or `priority` in vLLM 0.29.0. |
| `async_scheduling` | boolean | no | Omitted means the engine decides through its own disqualifying conditions. |
| `env.VLLM_USE_RUST_FRONTEND` | string | no | The API server that renders and tokenizes each prompt: `python` or `rust` (`engineconfig.Frontend`, rules/engineconfig/engine.go). Omitted, `python`. `rust` is vLLM's vllm-rs frontend, selected by the environment variable `VLLM_USE_RUST_FRONTEND=1`, not a flag. It changes how long a request waits for admission, per prompt token. |
| `compilation_config.cudagraph_mode` | string | no | CUDA graph capture strategy. |
| `max_cudagraph_capture_size` | integer | no | `compilation_config.max_cudagraph_capture_size`: the largest step, in tokens, a CUDA graph is captured for; a larger step runs eager. At least 1. Omitted, the release default, min(`max_num_seqs` x decode query length x 2, 512), or 1024 on SM100. The engine caps it at `max_num_batched_tokens` and truncates it to the largest size it generates. `cudagraph_mode: NONE`, or `--enforce-eager`, which has no field here, captures nothing. |
| `cudagraph_capture_sizes` | list of integers | no | `compilation_config.cudagraph_capture_sizes`: the exact step sizes to capture, each at least 1, not empty. Sizes above `max_num_batched_tokens` are dropped. With `max_cudagraph_capture_size` too, that must be the largest kept, or the engine refuses the pair. |
| `compilation_config.decoder_replay_cudagraph_capture_sizes` | list of integers | no | `compilation_config.decoder_replay_cudagraph_capture_sizes`, the decoder-replay graph sizes of a YOCO model such as DeepSeek-V4.1 (new in vLLM 0.32.0). Each at least 1. Omitted, generated from the sliding window. |
| `compilation_config.custom_ops` | list of strings | no | `compilation_config.custom_ops`: which ops run as custom kernels, `all`, `none`, or `+name` / `-name` entries. No entry may be empty. Omitted, the engine default. |
| `long_prefill_token_threshold_adaptive` | boolean | no | Floor the `long_prefill_token_threshold` cap at a fair share of the token budget (`max_num_batched_tokens` over the queued and running requests). Omitted means false. |
| `enable_chunked_prefill` | boolean | no | Let a prefill be split across steps. Off, a waiting request whose whole prompt does not fit the step's remaining token budget waits for a step that has room, and an unstated `max_num_batched_tokens` is raised to `max_model_len`. Omitted means true. vLLM refuses it off with `max_num_batched_tokens` below `max_model_len`, and with prefix caching on a Mamba model; the v0.32.0 engine also refuses it off beside speculative decoding, `max_loras`, a KV connector or offload tier, data or pipeline parallelism, hybrid KV cache groups, `long_prefill_token_threshold_adaptive`, `max_num_scheduled_tokens`, a nonzero `watermark` and `max_num_active_seqs`, because no oracle scenario runs them together (`core/vllm/chunked.go`). Prefix caching, KV events, preemption, a long-prefill cap, priority scheduling and async scheduling at one pipeline stage are recorded. |
| `max_num_scheduled_tokens` | integer | no | The most tokens the scheduler issues in a step, where the model appends tokens to a batch, as speculative decoding does. Not negative; omitted or 0 uses `max_num_batched_tokens`. |
| `max_num_active_seqs` | integer | no | Caps the requests admitted to running without shrinking `max_num_seqs`. Not negative, and not above `max_num_seqs`; omitted or 0 means `max_num_seqs`. |
| `watermark` | number | no | Fraction of KV blocks kept free when admitting waiting or preempted requests, in [0, 1). Omitted means 0, the engine default. |
| `dp_sync_interval` | integer | no | Steps between a data-parallel group's unfinished-request all-reduces (vLLM's `parallel_config.dp_sync_interval`, default 16). A deployment's `parallel` block has no key for it, so it is stated here. Not negative; omitted or 0 means 16. |
| `prefill_schedule_interval` | integer | no | Admit new prefills only once every N engine steps, aligned across data-parallel ranks. Not negative; omitted or 0 means the engine default, 1 (no throttling). |
| `stream_interval` | integer | no | Tokens the API server buffers before streaming them to the client. It changes when a client sees tokens, not what the scheduler does (the inertness proof `oracle/inertness/stream_interval.json`). The simulator's delivery stage does not read it yet, so a stated value is accepted and has no effect on the result until that stage does (inference-sim#2035). Not negative; omitted or 0 means 1. |
| `disable_hybrid_kv_cache_manager` | boolean | no | Give every attention layer the same KV size even when they differ (full and sliding-window). Omitted lets the engine decide. |
| `language_model_only` | boolean | no | vLLM's `--language-model-only`: a multimodal model serves text only and builds no vision tower or projector, so their weights are not resident. Omitted or `false` keeps them. For a text-only model it changes nothing. The cost model prices only the text decoder of the fourteen catalog models whose `config.json` has a `vision_config` (DeepSeek-V4.1-Flash, GLM-5.3-Flash, Inkling, Kimi-K2.5, Kimi-K2.6, Kimi-K3, Llama-4-Scout, MiniMax-M3 in both checkpoints, Muse-Spark-Open in both checkpoints, Qwen3.5-397B in both checkpoints, Qwen3.6-35B), so `true` is what it prices; omitted or `false` is accepted for those models, and the engine reports the assumption (`Assumptions()`). That the catalog graph is the text decoder is a reading of its node names (none names a vision tower, projector or patch merge); it is not compared with the weights vLLM keeps resident under `true`. |
| `kv_cache_memory_bytes` | integer | no | The KV cache's size per GPU in bytes, instead of deriving it from `gpu_memory_utilization`. Positive. |
| `prefix_match_unit` | integer | no | The finest token boundary a prefix-cache hit lands on; may be finer than the block size where every group's block size divides by it. Not negative; omitted or 0 means the block size. |
| `prefix_cache_retention_interval` | integer | no | Token interval between retained sliding-window and Mamba prefix-cache checkpoints; 0 keeps only semantic checkpoints. Not negative. |
| `kv_events_config` | mapping | no | KV-event publishing, below. |
| `max_loras` | integer | no | The most distinct LoRA adapters in one batch; a request whose adapter does not fit waits for a later step while those behind it run. Not negative; omitted or 0 means 1. |
| `max_cpu_loras` | integer | no | LoRA adapters kept loaded in CPU memory (an LRU set). Not below `max_loras`; omitted or 0 means `max_loras`. |

Weights and caches:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `quantization` | string | no | Weight format the engine serves, where it differs from the checkpoint's. Omitted means the checkpoint's own format. |
| `moe_backend` | string | no | The expert-kernel backend, `--moe-backend`. Omitted means `auto`. A mega-MoE backend is one of the conditions under which DeepSeek-V4.1 turns on its own sequence parallelism. |
| `kv_cache_dtype` | string | no | KV cache format. |
| `enable_prefix_caching` | boolean | no | Omitted means true, vLLM's default. `false` describes a deployment without prefix caching. |
| `swa_bounded_replay` | boolean | no | DeepSeek-V4.1 SWA bounded replay (`--swa-bounded-replay` / `--no-swa-bounded-replay`). `true` keeps the sliding-window cache out of prefix caching and recomputes a hit's last sliding window; `false` lets the window cache take part in prefix caching (retained window tails). Omitted, the engine's resolution: on in vLLM 0.31.0 under the V2 model runner, without prefill context parallelism and not on ROCm. Only DeepSeek-V4.1 reads it. |
| `mamba_cache_dtype` | string | no | Format of a recurrent layer's state. |
| `mamba_ssm_cache_dtype` | string | no | Format of the SSM state alone, where it differs. |
| `mamba_cache_mode` | string | no | Whether recurrent state is fixed per sequence or grows with context. |
| `disable_cascade_attn` | boolean | no | Cascade attention reads a prefix shared by a batch once rather than once per request. Omitted means `true`, so it is disabled unless this is set to `false`. |
| `attention_config.use_trtllm_attention` | boolean | no | `attention_config.use_trtllm_attention`: use or skip FlashInfer's TRT-LLM attention backend. Omitted auto-detects. |
| `attention_config.mla_prefill_backend` | string | no | `attention_config.mla_prefill_backend`: `FLASH_ATTN`, `FLASHINFER` or `TRTLLM_RAGGED` in vLLM 0.32.0. Omitted selects automatically. |
| `linear_backend` | string | no | `kernel_config.linear_backend`, the GEMM backend of linear layers. Omitted is `auto`. |
| `env` | mapping | no | Engine environment variables the launch set that change what the engine runs, such as `VLLM_FLASHINFER_ALLREDUCE_BACKEND`, name to value. Each name must be an environment-variable name. A consumer refuses, or discloses as an assumption, every key it cannot honour. |

Speculation and experts:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `speculative_config` | mapping | no | Speculative decoding, below. |
| `enable_eplb`, `eplb_config` | | no | Expert-parallel load balancing, below. |
| `engram_config` | mapping | no | Where DeepSeek-V4.1's Engram tables live, below. Decides the residency of every model `Gather` whose `residency_control` is `engram_cpu_offload`. |

Decode-context parallelism. These have no effect when `dcp` is 0 or 1:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `dcp_comm_backend` | string | no | The collectives a decode step runs: `ag_rs` (the default) or `a2a` in vLLM 0.29.0. |
| `dcp_q_replicate` | boolean | no | Replicate the query projection of multi-head latent attention (MLA) on every rank, to skip a query all-gather. |
| `cp_kv_cache_interleave_size` | integer | no | Consecutive tokens one rank holds before the next takes over. Zero means unstated, which is a different request from 1. |

#### Dual-batch overlap (top-level keys)

<!-- fields: deployment.DBO -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `enable_dbo` | boolean | no | Split a batch so one half's communication overlaps the other's compute. |
| `dbo_decode_token_threshold` | integer | no | Batch size in tokens at or above which a uniform-decode batch is split. Omitted takes the engine's default. With `enable_dbo` on, vLLM 0.31.0 refuses a stated value below the two microbatches, 0 or 1. Not negative. |
| `dbo_prefill_token_threshold` | integer | no | Batch size in tokens at or above which any other batch is split. As for the decode threshold. |

#### `eplb_config` (and `enable_eplb`)

<!-- fields: deployment.EPLB -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `enable_eplb` | boolean | no | Enable expert-parallel load balancing. |
| `num_redundant_experts` | integer | no | Extra expert replicas. With load balancing on, the expert-parallel width must divide the experts plus these. |
| `window_size` | integer | no | Load-balancing window. |
| `step_interval` | integer | no | Steps between rebalances. |

#### `speculative_config`

<!-- fields: deployment.Speculative -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `method` | string | yes | The draft method. Which names are valid depends on the engine release. |
| `num_speculative_tokens` | integer | yes | Draft tokens per step, at least 1. |
| `model` | string | no | The catalog model whose graph is the drafter, where the drafter is a separate checkpoint, as an EAGLE3 head is (vLLM's `speculative_config.model`). Omitted, the target graph's own speculator. Not blank. |
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


## Names: vLLM's own, and the migration

The block takes vLLM's `EngineArgs` names (design X1: what you hand vLLM is what you hand BLIS), so a
name in `fields.json` is a name in the block. A setting that vLLM keeps inside a config object is
written inside it: `compilation_config: {cudagraph_mode: FULL}`, `attention_config: {...}`,
`speculative_config: {...}`, `eplb_config: {...}`, `kv_events_config: {...}`; the dotted name in
`fields.json` is the path. The block's own keys that are not `EngineArgs` names are `env` (a mapping
of environment variables, named as vLLM reads them) and `allreduce_backend` (the resolved choice,
which the engine derives and `EngineArgs` has no field for).

A block written in the earlier names is refused, and the error names the key to use. These changed:

| Earlier | Now |
|---|---|
| `cache_dtype` | `kv_cache_dtype` |
| `cudagraph_mode`, `decoder_replay_cudagraph_capture_sizes`, `custom_ops` | the same keys inside `compilation_config` |
| `use_trtllm_attention`, `mla_prefill_backend`, `indexer_kv_dtype` | the same keys inside `attention_config` |
| `dbo: {enable_dbo, dbo_*_token_threshold}` | those keys at the top level |
| `eplb: {enable_eplb, ...}` | `enable_eplb` at the top level; the rest inside `eplb_config` |
| `kv_events: {...}` | `kv_events_config: {...}` |
| `speculative: {num_spec_tokens, draft_model, ...}` | `speculative_config: {num_speculative_tokens, model, ...}` |
| `frontend: rust` | `env: {VLLM_USE_RUST_FRONTEND: "1"}` |

To rewrite a repository's YAML, run `python3 -I tools/migrateengineblock/migrate.py <file-or-dir>...`
(`--check` writes nothing and exits 1 if a file would change). It edits only the lines of an
`engine:` mapping, keeps comments and layout, and is safe to run twice. A value written as a block
list (`custom_ops:` with `- x` lines) is refused; write it on one line and run again. For YAML held in
Go test strings, `tools/migrateengineblock/migrate_go.py` does the same for string literals.

A tool that writes the block from `fields.json` names (blis-config-search) needs only this rule:
a config name `a.b` is the nested mapping `a: {b: ...}`, a name without a dot is a top-level key, and
an environment variable is a key of `env`. `VLLMKeys` in `rules/engineconfig/names.go` lists the names the block
takes.

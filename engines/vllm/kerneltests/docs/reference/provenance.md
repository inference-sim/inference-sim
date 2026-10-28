# Provenance, Evidence, Resolved

<p class="lede">Three methods say what a kernel rests on. <code>Provenance</code> lists
every value in scope and where each came from. <code>Evidence</code> counts how many rest on
evidence. <code>Resolved</code> reports the configuration it priced, including every request
the engine would not honor. This page lists what each can contain.</p>

## Provenance

Each entry is a `kernel.CoefficientOrigin`: a name, the set it came from, the method that
produced it, and its scope. Most entries are registry coefficients: every entry of the
scenario's sets whose scope covers this chip and model, whether or not this deployment's
steps use it. Their set is the registry set that supplied them, such as
`cost-model-primitives`. Their scope names the deployment's part, the model where the entry is
scoped to one, and, for a `copied` entry, the scope it was copied from: DeepSeek-V4.1's eager
launch cost on h100 reads `hardware: h100, model: deepseek-v4.1-flash; copied from
hardware=h200,model=deepseek-v4.1-flash`. The registry's
documentation describes each set and method.

Entries the kernel supplied itself carry the set `blis-latency-kernel` (the constant
`KernelAssumptionSet`) and the method `assumed`. Their scope holds the value used and the
reason. They are of two kinds.

### Engine settings the kernel filled in

<!-- assumptions -->

| Name | Recorded when |
|---|---|
| `attention_prefill_<kind>` | A measured prefill family for an attention kind (`sparse_mla`) or a scorer's role (`block_index_scores`) is only partly in the scenario's sets, so it is not used, and prefill is priced as without it. Value `fallback`. |
| `async_scheduling_spec` | The deployment speculates and the engine rules carry no async-compatible draft list, so the draft method is taken as leaving async scheduling on. |
| `block_size` | The deployment states no block size. 64 for a DSA sparse-attention model, whose indexer accepts only 64; for a DeepSeek-V4-family model the one kernel block its sparse backends accept on the part (for DeepSeek-V4.1 64 on SM90 and 128 on SM100; for DeepSeek-V4 256 on every part); otherwise vLLM's default of 16. |
| `cache_record` | A graph states a cache with no record listed under the cache data type that runs, so the cache is sized at its base record. |
| `compute_rate_<kind>` | A GEMM's weights are in a format whose rate the catalog does not state: FP32 weights run at the TF32 rate, taken as half the dense BF16 peak. |
| `compute_capability` | The catalog states no compute capability for the chip, so its generation is inferred from its native-format peaks and name where it decides kernel selection. |
| `cudagraph_mode` | No CUDA graph mode is stated. vLLM's default, `FULL_AND_PIECEWISE`. |
| `deepep_v2_mode` | The MoE dispatch runs DeepEP v2, whose path vLLM picks from `enforce_eager`, which a deployment cannot state: a resolved `cudagraph_mode` of `NONE` is read as it. Value `deepep_v2_eager` or `deepep_v2_graph`. |
| `dcp_comm_backend` | Decode-context parallelism is on and no combine backend is stated: the stock `ag_rs`. Also when one is stated but the engine rules give no list to check it against. |
| `engine_version` | The scenario declares a vLLM release other than the one whose behavior the kernel prices (v0.31.0). |
| `engram_cpu_offload` | A graph states an Engram table and the deployment does not say where it lives: the release's `cpu_offload` default. |
| `engram_dp_shared_memory` | An Engram table under data parallelism, with `dp_shared_memory` unstated: resolved as the engine resolves it. |
| `gemm_eps_max_<kind>` | A weight-only kernel (Marlin W8A16 MXFP8, W4A16 grouped) has no efficiency ramp in the registry for the part, so it is priced on the BF16 ramp at the BF16 peak it computes at. |
| `host_gather_rate_uva` | A table lookup reads a host-resident table (an Engram table under `cpu_offload`) and the registry has no rate for that gather, so its rows are read at the host link's rate. |
| `hybrid_kv_cache_manager` | A graph states window caches and the deployment runs a KV connector, which can turn the hybrid KV cache manager off and the window rings into full-length caches. |
| `kv_cache_grouping` | A graph states its caches and the rules pack cannot group them into the engine's shared block pool, so each cache is sized at its own page rather than in whole pool blocks. The reason names what the pack does not model. |
| `mamba_cache_mode` | A model with recurrent layers states no recurrent cache mode: `align` with prefix caching on, `none` with it off. |
| `max_cudagraph_capture_size` | Any CUDA graph is captured and the deployment states neither `max_cudagraph_capture_size` nor `cudagraph_capture_sizes`: vLLM's default ceiling, above which a step runs with no graph. |
| `max_model_len` | The deployment states no context length, or states `auto`: the bound is the model's context window from the graph (for `auto`, the most the engine can fit it to). |
| `max_num_batched_tokens` | No token budget per step is stated. vLLM's default for the chip's memory size. |
| `max_num_seqs` | No limit on concurrent requests is stated. vLLM's default for the chip's memory size. |
| `modality` | The graph prices the text decoder of a multimodal model; its vision or audio tower is not priced. |
| `mhc` | Only one of the mHC pre and post forms resolves, so neither is used. Value `fallback`. |
| `moe_a2a_<kind>` | A DeepEP backend (`deepep_high_throughput`, `deepep_low_latency`, `deepep_v2_eager`, `deepep_v2_graph`) runs the MoE dispatch but has no measured price for the part and expert width, or the expert group spans nodes, so it is priced as the generic routed all-to-all. |
| `mhc_<kind>` | An mHC component's rate (`pre` or `post`) is in the scenario's sets without its floor, so the measured mHC form is not used. Value `fallback`. |
| `mla_context_chunk` | Decode-context parallelism is on for a latent-attention model and no context length is stated, so the chunk in which a prefill gathers its cached context is taken at its largest. |
| `num_spec_tokens` | The deployment drafts a different number of tokens a step from the graph's speculator, so what the graph sizes at its own count is taken as stated. |
| `recurrent_prefill_rate_<kind>` | A recurrent mixer has no measured prefill form, so its prefill tokens are priced as the chunked scan's FLOPs on the BF16 ramp rather than at the decode scan's rate. |
| `sparse_mla_backend` | A DSA sparse-attention model on Hopper selects FlashInfer's sparse MLA backend, which needs a FlashInfer release the kernel cannot see installed. |
| `sparse_read_per_query` | A speculating deployment of a sparse-attention model: each verify token is charged its own window-and-top-k read, as vLLM gathers them, though the registry's rate was fitted on single-token decode and the sharing between a request's tokens was never measured. |
| `speculator` | The deployment speculates and the graph states no speculator, so draft passes are not priced; or the deployment names a different method from the graph's speculator, which is the one priced. |

### Optional coefficients that were missing

When a coefficient the kernel can price without is absent from the scenario's sets, and the
deployment needs it, the entry is named for the coefficient, its value is `absent`, and its
reason says what was priced without it. These are the coefficients it can apply to; a
per-kind name ends in the kind it covers, such as `_mla` or `_kda`, and is needed only where
the model has a layer of that kind.

| Coefficient | Without it |
|---|---|
| `attention_decode_floor`, `attention_decode_rate` | No measured decode attention: the KV read is charged at memory bandwidth and the per-layer floor is not charged |
| `attention_decode_rate_<kind>` | The chip-wide attention pair is used for that kind; for a scorer named by its role, its attention kind's pair |
| `attention_decode_floor_<kind>` | A scorer with its own rate inherits the part-wide attention floor |
| `attention_prefill_floor`, `attention_prefill_work_scale` | Prefill attention is charged on the plain efficiency ramp, with no floor |
| `attention_prefill_rate_<kind>` | Prefill of that attention kind (`sparse_mla`) or scorer role (`block_index_scores`) is priced with the part-wide prefill pair on its FLOPs, not with its own measured floor, per-request, per-token and byte terms |
| `mhc_rate_<kind>` | The mHC nodes running that component (`pre`, `post`) are charged their stated bytes at memory bandwidth, and, without the `pre` form, the `hc_*fn` GEMMs on the dense ramp |
| `recurrent_decode_floor_<kind>`, `recurrent_decode_rate_<kind>` | The recurrent mixer's decode term is not charged |
| `recurrent_prefill_floor_<kind>` | A recurrent mixer with a measured prefill rate is charged no prefill floor |
| `moe_routing_imbalance_median` | Experts are priced as evenly loaded |
| `host_link_bandwidth` | An offload transfer is not capped at the host link's rate |
| `host_step_per_request` | The scheduler's per-request CPU cost is charged at zero; with it, a step without async scheduling carries it in series and one with async scheduling shows only its excess over the device work |
| `host_admission_per_token`, `host_admission_per_request`, `host_output_token`, `host_completion`, `host_launch_eager_per_layer`, `host_launch_per_kernel`, `host_replay_graph_per_step` | That host cost is charged at zero |

The eager launch cost is read as `host_launch_eager_per_kernel` where a set carries it, and as
`host_launch_eager_per_layer` otherwise; both are charged per kernel. A model-scoped entry
under either name is taken over a part-wide one.

## Evidence

`Evidence()` returns three things: how many entries of `Provenance` rest on evidence (method
`measured`, `literature` or `vendor_spec`), how many there are in all, and the names of those
whose method is `assumed`. The last includes the kernel's own entries above and any registry
coefficient recorded as assumed. Entries with the methods `copied` and `not_charged` are in
the total and in neither of the other two.
[Refuse, assume, override](../concepts/resolution.md#the-evidence-summary) shows a worked
example.

## Resolved

`Resolved()` returns a `kernel.Resolution`: the tensor-, data- and expert-parallel widths the
layout resolved to, the all-reduce backend that will run, whether asynchronous scheduling,
cascade attention and sequence-parallel MoE are on, and a list of overrides. Each
override names the field, what was requested, what was resolved, and why. The kernel records
these:

| Field | When |
|---|---|
| `all2all_backend` | An MoE dispatch backend is stated for an expert-parallel layout that runs no dispatch: dp 1, pcp 1 and no sequence-parallel MoE input, where vLLM's `use_all2all_kernels` is false. The experts run on the replicated tokens and the tensor-parallel all-reduce combines them, so the backend is resolved as `none` and every backend prices the same step. |
| `allreduce_backend` | The custom all-reduce was requested at a tensor-parallel width it does not support, or across nodes on a chip without multi-node NVLink. NCCL runs instead. |
| `cache_dtype` | A sparse-attention model's KV cache is stored in a packed format its backend chooses, such as `fp8_ds_mla` for a stated `fp8` on Hopper, or DeepSeek-V4.1's `nvfp4_ds_mla` for a stated `fp8` on SM100. |
| `cp_kv_cache_interleave_size` | Decode-context parallelism is on, the interleave is unstated, the block size is above one, and a NIXL connector is configured, so vLLM sets the interleave to the block size. |
| `cudagraph_capture_sizes` | Stated capture sizes above `max_num_batched_tokens`, which vLLM drops. |
| `cudagraph_mode` | DeepEP high-throughput with dp above one, or MoonEP where its all-to-all runs: the engine cannot capture the dispatch and runs with no CUDA graphs. |
| `mamba_cache_mode` | A hybrid model states `none` with prefix caching on, which vLLM runs as `align`. |
| `max_cudagraph_capture_size` | A stated maximum that vLLM lowers: capped at `max_num_batched_tokens`, or off the sizes it captures (1, 2, 4, every 8 below 256, every 16 from it). The largest size that runs is the ceiling. |

The two context-parallel widths are not yet fields of `kernel.Resolution`. Until blis-schemas
adds them, read them from `DecodeContextParallelWidth()` and `PrefillContextParallelWidth()`.

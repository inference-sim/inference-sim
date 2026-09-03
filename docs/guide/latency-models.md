# Latency Models

The `LatencyModel` interface determines how BLIS estimates GPU step time for each batch iteration. BLIS ships two backends -- **trained-physics** (default, physics-informed roofline with MoE-aware corrections) and **roofline** (pure analytical) -- and the pluggable architecture supports adding custom backends.

**Migration note:** Three legacy backends have been removed (`blackbox`, `crossmodel`, `trained-roofline`). Use `--latency-model trained-physics` instead, which supersedes all three with improved accuracy and MoE support.

```bash
# Trained-physics mode (default) — roofline × architecture-aware basis functions × learned corrections
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --rate 100 --num-requests 500

# Roofline mode — pure analytical estimation from model architecture (explicit flag)
./blis run --model qwen/qwen3-14b \
  --latency-model roofline --hardware H100 --tp 1 \
  --num-instances 4 --rate 100 --num-requests 500
```

## Trained-Physics Mode (Default)

Trained-physics mode combines roofline basis functions with learned correction coefficients. It provides better out-of-box accuracy than pure roofline by capturing architecture-specific overheads (MoE routing, memory access patterns) that analytical models miss.

**Benefits:**
- Better generalization across model architectures and TP configurations
- Lower MAPE in practice compared to pure roofline
- No per-model calibration needed

Use this for capacity planning and what-if analysis unless you specifically need pure analytical estimates.

## Roofline Mode

Roofline mode computes step time analytically from model architecture (FLOPs, parameter count) and hardware specifications (compute throughput, memory bandwidth). It does not require pre-trained coefficients, making it suitable for new models.

### The `--latency-model roofline` Flag

The simplest way to use roofline mode:

```bash
./blis run --model qwen/qwen3-14b \
  --latency-model roofline --hardware H100 --tp 1
```

This auto-resolves both required inputs:

1. **Model config** -- checks `model_configs/` for a cached `config.json`, fetches from HuggingFace on miss
2. **Hardware config** -- uses the bundled `hardware_config.json`

**Supported hardware:** The bundled `hardware_config.json` includes specs for **H100** (80 GB HBM3, 989.5 TFLOPS BF16, 3.35 TB/s), **A100-SXM** (80 GB HBM2e, 312 TFLOPS BF16, 2.04 TB/s), and **A100-80** (alias for A100-SXM). To use a different GPU, add an entry to `hardware_config.json` with the required fields (`TFlopsPeak`, `BwPeakTBs`, `mfuPrefill`, `mfuDecode`, `MemoryGiB`), plus `IntraNodeBwGBps`/`InterNodeBwGBps` if instances on that GPU may span nodes (see [Inter-Node Network Cost](#inter-node-network-cost-trained-physics-only)) and reference it via `--hardware <name>`.

**Validated models:** Any dense or MoE transformer with a HuggingFace `config.json` works. The following have been validated end-to-end:

- [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) / [Llama-2-70B](https://huggingface.co/meta-llama/Llama-2-70b-hf)
- [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)
- [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) (MoE)
- [CodeLlama-34B](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf)

Set `HF_TOKEN` to access gated models (e.g., [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf)) and avoid rate limits:

```bash
export HF_TOKEN=your_token_here
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --latency-model roofline --hardware H100 --tp 1
```

### Manual Configuration

For full control, provide configs explicitly:

```bash
./blis run --model my-custom-model \
  --model-config-folder ./my-model-configs/ \
  --hardware-config ./my-hardware-config.json \
  --hardware H100 --tp 4
```

### Adding Support for New Models

Any model with a HuggingFace `config.json` can use roofline mode:

1. Download `config.json` from HuggingFace
2. Place it in `model_configs/<model-name>/config.json`
3. Run with `--latency-model roofline --hardware <GPU> --tp <N>`

Or let BLIS fetch it automatically with `--latency-model roofline`.

### Tensor Parallelism and Roofline

The `--tp` flag divides FLOPs and memory traffic across TP ranks:

- Higher TP reduces per-GPU step time (more parallelism)
- Higher TP reduces KV blocks per GPU (memory split across ranks)

When choosing between TP and replication (more instances): TP reduces per-request latency, replication increases throughput. For capacity planning, simulate both configurations.

!!! note "Automatic KV block calculation"
    For both latency backends (roofline, trained-physics), `--total-kv-blocks` is automatically derived from model architecture and GPU memory if not explicitly set. The auto-calculated value accounts for TP (KV heads are sharded across ranks; total GPU memory scales with GPU count). Override with `--total-kv-blocks <N>` for non-standard deployments. The auto-calculation uses reference constants (90% GPU utilization, standard activation/overhead budgets matching the llm-d-benchmark capacity planner) and requires SwiGLU-family activations.

!!! note "Automatic MaxModelLen derivation"
    When using roofline or trained-physics mode and `--max-model-len` is not explicitly set, BLIS auto-derives it from `max_position_embeddings` in the HuggingFace `config.json`. For models with `rope_scaling`, the scaling factor is applied based on vLLM's blacklist approach: types `linear`, `dynamic`, `yarn`, `default`, and `mrope` apply the factor; types `su`, `longrope`, and `llama3` are excluded (these encode the full context in `max_position_embeddings`). For `yarn`, `original_max_position_embeddings` is used as the base when present. `gemma3` models skip `rope_scaling` entirely (`max_position_embeddings` is pre-scaled). The derived value is then capped at the KV-feasible maximum (`total_kv_blocks * block_size`) to prevent context windows from exceeding GPU memory capacity. Override with `--max-model-len` <N>` when needed.

## How Trained-Physics Works

Trained-physics mode applies **learned correction factors** to analytical roofline basis functions, combining the physical grounding of roofline with the accuracy of data-driven fitting. Coefficients are fitted from real vLLM measurements and generalize across model architectures, workloads, and TP configurations.

**StepTime formula** (10 beta coefficients in bundled defaults):

```
StepTime = β₁ₐ × T_pf_compute                  # prefill compute only
         + β₁ᵦ × T_pf_kv                       # prefill memory (typically ~0)
         + β₂ₐ × T_dc_compute                  # decode compute (typically ~0)
         + β₂ᵦ × T_dc_kv                       # decode memory only
         + β₃ × T_weight                       # weight loading × correction
         + β₄ × T_tp                           # TP communication × correction
         + β₅ × L                              # per-layer overhead (µs/layer)
         + β₆ × batch_size                     # per-request scheduling (µs/req)
         + β₇                                  # per-step fixed overhead (µs)
         + β₈ × nMoE                           # per-MoE-layer overhead (µs/layer)
```

The model supports 7-11 beta coefficients. Bundled defaults use 11 coefficients (prefill/decode split + the MoE expert-parallel dispatch correction β_EP).

**Beta coefficients:**

- **β₁ₐ** (prefill compute, ~0.15): Corrects analytical FlashAttention + MLP FLOP estimates for kernel efficiency, memory access patterns.
- **β₁ᵦ** (prefill memory, ~0): Prefill KV cache write bandwidth correction (typically near zero).
- **β₂ₐ** (decode compute, ~0): Decode compute correction (typically near zero, decode is memory-bound).
- **β₂ᵦ** (decode memory, ~1.9): Corrects KV cache read bandwidth. Primary decode bottleneck.
- **β₃** (weight loading, ~1.4): Corrects model weight bandwidth for cache effects, prefetching, HBM contention.
- **β₄** (TP communication, ~0.75): Corrects tensor-parallel All-Reduce overhead.
- **β₅** (per-layer, ~32 µs/layer): Fixed overhead per transformer layer: kernel launch, CUDA graph, residual connections.
- **β₆** (per-request, ~4 µs/request): Scheduling overhead per request: queue management, attention mask construction.
- **β₇** (per-step, ~126 µs/step): Fixed overhead per step: CUDA synchronization, sampler invocation.
- **β₈** (MoE-layer, ~482 µs/layer): Per-MoE-layer overhead for router gating, token permutation. Architecture-aware: applies only to interleaved MoE architectures (InterleaveMoELayerStep > 0). Zero for uniform MoE and dense models.
- **β_EP** (MoE dispatch/combine, defaults to β₄): Corrects MoE expert-/data-parallel dispatch+combine all-to-all communication. Active only when `ModelHardwareConfig.DP > 1` on an MoE model — the gate is `m.isMoE && m.dp > 1` (`sim/latency/trained_physics_model.go:493`), where `m.dp` is `hw.EffectiveDP()` (same file, ~line 794). This is **not** reachable by passing `--dp > 1` on the CLI: under DP-as-placement (#1531 / #1556) every replica is configured at `DP=1`, so `EffectiveDP() == 1` and this term is inert — a config with `DP > 1` now comes only from constructing one directly (e.g. tests). With the term inert, an MoE model at `TP > 1` instead pays the MoE-FFN all-reduce over the TP group (the `m.dp == 1 && m.tp > 1` branch, ~line 484); at `TP == 1` there is no collective at all. See the DP-as-placement note below. Defaults to β₄ because both comm-backend families run over the same NVLink fabric and share the ring-collective per-phase efficiency β₄ captures (the per-family *volume* difference is in the basis, not the coefficient). The 11th coefficient overrides the default.

**Alpha coefficients** (3 terms, API/framework overheads in µs):

- **α₀** (QueueingTime, ~15,563 µs): Fixed per-request API processing (HTTP parsing, request validation, queue insertion).
- **α₁** (PostDecodeFixedOverhead, ~777 µs): Fixed per-request post-decode overhead (detokenization setup, finish reason determination).
- **α₂** (OutputTokenProcessingTime, ~46 µs/token): Per-output-token overhead (streaming token transmission, incremental detokenization).

**Pre-trained coefficients** are stored in `trained_physics_coefficients` in `defaults.yaml`. No per-model calibration needed -- the model generalizes across architectures, workloads, and TP configurations.

### Generalization Scope

The trained-physics model is designed to generalize without per-model calibration:

**Supported hardware:**

- **H100** (80 GB HBM3, 989.5 TFLOPS BF16 / 1979 TFLOPS FP8, 3.35 TB/s)
- **A100-SXM** (80 GB HBM2e, 312 TFLOPS BF16, 2.04 TB/s)
- **A100-80** (alias for A100-SXM)
- **L40S** (48 GB GDDR6, 362 TFLOPS BF16 / 1466 TFLOPS FP8, 0.864 TB/s)

**Coefficients were trained on H100 traces** but the roofline basis functions automatically scale to each GPU's compute/bandwidth specifications via hardware config. This enables the model to generalize across hardware without GPU-specific calibration.

**Model architectures:**

- **Dense transformers** (Llama-2, Qwen3, GPT, etc.): Standard attention + MLP layers
- **Uniform MoE** (Mixtral): All layers are MoE with top-k expert routing
- **Interleaved MoE** (Scout): Alternating MoE and dense layers with architecture-specific β₈ overhead

The model automatically detects MoE configuration from `config.json` (`num_local_experts`, `num_experts_per_tok`, `interleave_moe_layer_step`) and adjusts basis functions accordingly.

**Workload types:**

- **Prefill-heavy** (large input, short output): Chatbot prompts, document Q&A
- **Decode-heavy** (small input, long output): Content generation, code completion
- **Mixed batches** (concurrent prefill/decode): Production serving with heterogeneous requests
- **TP configurations**: TP=1, TP=2, TP=4, TP=8 (All-Reduce overhead scales via β₄)

**Why trained-physics over roofline:**

Trained-physics uses up to **14 coefficients** (11 beta: prefill compute/memory split, decode compute/memory split, weight, TP, layer overhead, batch overhead, step overhead, MoE overhead, and the MoE expert-parallel dispatch correction β_EP; 3 alpha: queueing, post-decode, per-token) that capture more architectural detail than pure roofline (no learned corrections). The prefill/decode split (β₁ₐ/β₁ᵦ, β₂ₐ/β₂ᵦ) and MoE-specific overhead (β₈) enable better generalization to unseen model architectures (especially interleaved MoE) and batch compositions (mixed prefill/decode).

!!! note "MoE architecture detection"
    β₈ applies conditionally based on `InterleaveMoELayerStep` from the model's `config.json`: 0 = uniform MoE (β₈ skipped), 1 = alternating MoE/dense (β₈ × 24 layers for Scout's 48 total), 2 = every 3rd layer is MoE, etc. This prevents over-penalizing uniform MoE models like Mixtral where expert routing overhead is amortized across all layers.

### Data + Expert Parallelism for MoE (trained-physics only)

!!! note "DP-as-placement (#1531, #1556): `blis run` and `blis replay` feed DP=1 per replica"
    Since #1531 (`blis run`) and #1556 (`blis replay`), BLIS models MoE `--dp N` as **N real single-node engine replicas** (DP-as-placement), each configured at **DP=1**. So the per-replica step time uses `moeGroup = TP` (experts replicated per DP rank — expert-parallel-OFF physics) and the DP>1 dispatch term below does **not** fire per replica (the all-reduce at `DP=1, TP>1` does). The `moeGroup = TP·DP` / `/dp` / DP>1-dispatch math described in this section is the per-instance latency model's response to a DP>1 `ModelHardwareConfig` — the basis for EP-on placement (#1548), and still reached when a config sets DP>1 directly (e.g. tests). `--enable-expert-parallel` + `--dp>1` (MoE) currently fails fast on both `blis run` and `blis replay`, deferring true EP placement to #1548.

For MoE deployments, trained-physics models data parallelism (`--dp`) and expert parallelism (`--enable-expert-parallel`) the way vLLM does (mirrors `vllm-project/vllm`):

- **Routed-expert weight/compute** are scoped to the flattened MoE group `moeGroup = TP·DP` via the `ExpertPlacement` seam: each GPU holds `numExperts/moeGroup` full-expert-equivalents. This replaces a batch-dependent heuristic, so MoE step time at `DP=1` intentionally differs from pre-DP/EP BLIS (a deliberate fidelity fix). Dense models at `DP=1` are byte-identical (INV-BC-DP1). **Step time uses `TP·DP` in both EP modes, and that matches vLLM**: `FusedMoEParallelConfig.make` flattens TP across DP for MoE layers *unconditionally* (at TP=2/DP=2 the MoE `tp_size` is 4 whether or not `--enable-expert-parallel` is set), so per-GPU routed-expert bytes are `numExperts/(TP·DP)` in both modes — EP changes the sharding *style* (whole experts vs tensor slices) and the collective, not the footprint. The **capacity** model's EP-off baseline differs: it follows BLIS's own DP model (`--dp N` = N independent engine replicas, #1531, each holding a full tensor-sharded copy), so at `DP>1` with EP *off* capacity charges `numExperts/TP` and is conservative relative to both vLLM and this step-time term. Unreachable today — `blis run` gives every DP replica `DP=1` (#1531, making `moeGroup = TP`) and `blis replay` rejects MoE `--dp>1` (#1556) — and tracked in [#1666](https://github.com/inference-sim/inference-sim/issues/1666), to be settled with #1548 (which owns the EP-mode step-time toggle).
- **Sequence-split terms** (attention/dense-FFN compute, KV read/write) gain a `/dp` factor — each DP rank processes ~`1/dp` of the tokens. Weights stay `/tp` (replicated across DP groups).
- **Shared experts** (DeepSeek/Qwen-style) are charged for every token when the model exposes a shared-expert FFN dim; a no-op otherwise (including Llama-4 Scout until its shared-expert dim — `config.intermediate_size`, not `intermediate_size_mlp` which is the dense-layer FFN — is mapped).
- **MoE-FFN communication** partitions on the `DP` boundary: at `DP=1, TP>1` an all-reduce over the TP group; at `DP>1` a dispatch/combine all-to-all (β_EP).

**`--moe-comm-backend`** selects the dispatch/combine cost model (mirrors vLLM `VLLM_ALL2ALL_BACKEND`). The seven names map to two physical volume families:

| Family | Backends | Per-rank dispatch volume |
|--------|----------|--------------------------|
| all-gather | `naive`, `allgather_reducescatter` (default) | dense hidden states, **no top_k** |
| modular all-to-all | `pplx`, `deepep_high_throughput`, `deepep_low_latency`, `mori`, `flashinfer_all2allv` | top_k-routed tokens (carries `kEff`) |

DP/EP and `--moe-comm-backend` require `--latency-model trained-physics` (roofline is DP/EP-blind for step time) and are rejected on dense models for `--dp > 1` (dense data parallelism is the router-replica mechanism — use `--num-instances`). Absolute MoE communication magnitudes are physics-estimated with β_EP defaulted to β₄; an empirical re-fit is future work.

#### Calibrating β_EP

The β₄ default assumes the dispatch/combine collective runs at the same per-byte efficiency as the TP all-reduce — true when both share one NVLink fabric, but not when EP spans nodes (e.g. inter-node InfiniBand for EP while TP stays intra-node NVLink). To fit β_EP for such a deployment:

1. Collect real per-step latencies for a **MoE model at `--dp > 1`** with a fixed `--moe-comm-backend`, holding everything else constant.
2. Freeze the other 10 β coefficients (and the α coefficients) at their bundled values.
3. Fit only β_EP to the residual between observed step time and the model's prediction with the dispatch term zeroed — i.e. attribute the leftover to `β_EP · tMoEDispatch`.

Because the dispatch term is the *only* term gated on `DP > 1`, the residual isolates it cleanly. Fit per comm-backend *family* (all-gather vs all-to-all), not per backend name; see the PR #1433 discussion for why per-backend scalars are the wrong granularity (the within-family differences are prefill/decode shape effects a single scalar cannot represent).

## Inter-Node Network Cost (trained-physics only)

By default a tensor-parallel all-reduce and a MoE expert dispatch/combine are both
priced at the GPU's on-package bandwidth, with the NVLink/HBM ratio folded into the
learned coefficient β₄. That is right for an instance living inside one node — and
*free* for one that spans nodes. Since [#1529](https://github.com/inference-sim/inference-sim/issues/1529)
an instance can occupy whole nodes across a pool (GLM-5.2 at TP=16 on 2×8 H100), so
BLIS now charges the crossing.

### How it works

The two communication bases divide byte volume by an **effective** link bandwidth:
`bwHbmUs` when the collective fits inside one node, and `bwHbmUs / spanScale` when it
does not. This is a re-scale of the existing term, not an extra one — an additive
cross-node term would double-charge a `DP>1` MoE instance whose all-to-all is already
priced by the dispatch basis.

Let `G` be the collective's group size, `p` the size of the node(s) the instance was
placed on, `n = ceil(G/p)` the nodes spanned, `g = min(G, p)` the group members per
node, and `r = IntraNodeBwGBps / InterNodeBwGBps`. Two penalty shapes apply:

| Collective | Applies to | Penalty |
|------------|-----------|---------|
| Ring | TP all-reduce (`G = tp`: attention, dense FFN, and the `DP=1` MoE-FFN reduce) and the `allgather_reducescatter` / `naive` MoE family, whose volume basis is ring-shaped | `1 + (r-1)·(n-1)/(G-1)` |
| All-to-all | `deepep_*` / `pplx` / `mori` / `flashinfer_all2allv` MoE dispatch (`G = TP·DP`) | `1 + (r-1)·(G-g)/(G-1)` |

The ring form is derived from the **hierarchical (two-level)** algorithm NCCL uses
across nodes: an intra-node reduce-scatter + all-gather over the `g` ranks on a node,
then an inter-node all-reduce of the *reduced* `S/g` chunk across the `n` nodes.
Normalized by the flat single-node baseline this simplifies exactly to the expression
above. The all-to-all form is a per-peer split: a rank's egress goes to `G-1` peers,
`G-g` of which are on other nodes — far more of the traffic leaves the node than in a
ring, which is why the expert all-to-all rather than the TP all-reduce is the dominant
cross-node cost for wide expert parallelism.

Both penalties are exactly `1.0` when nothing crosses a boundary (`n = 1`) or when the
fabric is no slower than the on-node link (`r ≤ 1`), and both are monotone in `r`: a
worse fabric never lowers the cost.

### The second, size-independent half

Bandwidth is only half the story. Every cross-node collective also pays a fixed cost —
NCCL launch, fabric round-trip, and the synchronization a two-level collective imposes —
that does not shrink with the message. For the small messages a decode step produces,
that fixed cost can exceed the bandwidth half by an order of magnitude, and it is the
mechanism behind vLLM's guidance to prefer pipeline parallelism across nodes and tensor
parallelism within a node: per-layer all-reduce means *many small* collectives, not a few
large ones.

`InterNodeLatencyUs` supplies it. It is charged once per comm unit that crosses a node
boundary, so a step running `L` layers × 2 phases pays it `2L` times, and it is skipped
entirely for a step that communicates no tokens (no collective runs, so nothing launches).

**It is 0 — not charged — in the bundled hardware config, deliberately.** BLIS has no
measured per-collective latency to ship, and a guessed constant would sit in front of
every multi-node estimate. So out of the box the cross-node cost is bandwidth-only, and
the size-independent half is available but off. Supply a measured value to model it; see
[#1661](https://github.com/inference-sim/inference-sim/issues/1661), which also records
the calibration-evidence bar. Like the bandwidth half, it rides the learned communication
coefficient (β₄, or β_EP for MoE dispatch), so calibrate it in that frame — the charge is
`β · units · InterNodeLatencyUs`, not a raw wall-clock number.

### Where the inputs come from

**Topology is derived from placement, not declared.** There is no CLI flag for it: the
placement manager reports the size of the node(s) an instance's GPUs actually occupy,
and that is stamped onto the instance's configuration at every placement site
(startup, deferred node-ready, autoscaler scale-up). A declared "GPUs per node" knob
could contradict the real `node_pools` placement, charging for a boundary that was
never crossed or missing one that was.

**Fabric speeds are hardware calibration**, in the file `--hardware-config` already
points at:

```json
"H100": {
  "TFlopsPeak": 989.5, "BwPeakTBs": 3.35, "MemoryGiB": 80.0,
  "IntraNodeBwGBps": 450,
  "InterNodeBwGBps": 50
}
```

Both are per-GPU **effective unidirectional** GB/s. Only their ratio is used, so the
absolute scale cancels — but the *convention* must match on both fields (mixing a
bidirectional NVLink figure with a unidirectional NIC figure doubles the penalty).
Committed values: H100 450/50 (NVLink 4 against one 400 Gb/s ConnectX-7 per GPU),
A100 300/25 (NVLink 3 against HDR-200 per GPU), L40S 32/12.5 (PCIe Gen4 — no NVLink —
against 100 GbE). Set both or neither; a half-calibration is a hard error.

To compare fabrics (InfiniBand vs RoCE vs a single uplink), run the same workload twice
with different `InterNodeBwGBps` values in that file. (Giving two pools distinct
`gpu_type` entries models a *mixed-fabric* fleet in one run, which is a different
question — and it leans on the `gpu_type` keying that
[#1662](https://github.com/inference-sim/inference-sim/issues/1662) tracks.)

!!! warning "Check the shape you are comparing actually spans nodes"
    A cost that is only charged when a collective crosses a node boundary is zero for a
    deployment where none does. In particular the shape GLM-5.2 is really served with —
    TP=1, DP=16, EP=16, tensor parallelism kept inside the node — charges **nothing**
    today: at TP=1 there is no TP collective, and the expert all-to-all leg is not yet
    reachable (see the last of the known approximations below). Comparing fabrics is
    meaningful for a multi-node **TP** shape now, and for wide expert parallelism once
    [#1548](https://github.com/inference-sim/inference-sim/issues/1548) lands.

#### Worked example: InfiniBand vs a single 100 GbE uplink, TP=16

```bash
# A pool of 8-GPU H100 nodes; TP=16 forces the instance across two of them.
cat > pools.yaml <<'YAML'
node_pools:
  - name: h100
    gpu_type: H100
    gpus_per_node: 8
    gpu_memory_gib: 80
    initial_nodes: 4
    max_nodes: 4
    cost_per_hour: 30.0
YAML

# Run A — the bundled H100 entry: 450/50 GB/s (one 400 Gb/s NIC per GPU), ratio 9x.
./blis run --model <your-model> --tp 16 --hardware H100   --latency-model trained-physics --policy-config pools.yaml   --num-requests 500 --rate 8

# Run B — copy hardware_config.json, drop the H100 entry's InterNodeBwGBps to 12.5
# (a single 100 GbE uplink shared by the node's 8 GPUs), then:
./blis run --model <your-model> --tp 16 --hardware H100   --latency-model trained-physics --policy-config pools.yaml   --hardware-config ./hardware_config.roce.json   --num-requests 500 --rate 8
```

Compare `ttft_p50_ms` / `itl_mean_ms` between the two. Run B's ratio is 36× rather than
9×, so its communication term is larger; the difference is the fabric's contribution.
Both runs warn once on stderr that an instance spans nodes, and if either run reports
that the cross-node cost is *unpriced*, the calibration or the backend is the reason —
the message says which.

### Inert unless a boundary is actually crossed

Three independent gates each make the cost exactly zero, so every configuration that
existed before this feature produces bit-identical step times:

1. no `node_pools` ⇒ no placement ⇒ no topology;
2. the collective fits inside one node ⇒ `n = 1`;
3. the hardware declares no interconnect bandwidths ⇒ `r = 1`.

Multi-node placement is `blis run` only — `blis replay` rejects `node_pools` outright
and `blis observe` takes its timing from a real server — so a cross-node cost cannot
arise off the run path.

That leaves one hole, which is fenced explicitly. A trace exported from a multi-node run
could be replayed *without* the `node_pools` section, and replay would then reproduce the
workload at single-node speed — faster than the run that produced the trace, with nothing
to indicate it. So `blis run` records the widest instance node span in the trace header
(`max_nodes_spanned`) and `blis replay` refuses any trace that reports more than one node.
Traces from runs without multi-node placement omit the field entirely and replay exactly
as before.

If a spanning placement will *not* be charged (uncalibrated fabric, or a backend with
no communication term), BLIS says so once on stderr rather than silently returning an
optimistic number. Watch for this if you use a policy bundle's `hw_config_by_gpu`
override: it replaces the whole hardware calibration, so an entry that omits the two
fabric fields drops them.

### How large is the effect, and what is still missing

For TP=16 on 2×8 H100 at `r = 9`, the TP communication term rises about **1.53×**,
which is roughly **+10%** on total step time for a mid-size dense model. The penalty
is modest by design: hierarchical all-reduce moves the same bytes as a flat ring, and
only the reduced `S/g` chunk crosses the fabric. A *flat* multi-node ring would instead
be throttled to ≈`r` (9×) — an order of magnitude more. Measured two-node H100
all-reduce bus-bandwidth degradation (~1.3–1.5×) is why the hierarchical model is the
one used here; treat that as the assumption to revisit if a deployment's collectives
are known not to be hierarchical.

Read `IntraNodeBwGBps` as *the on-node link speed β₄ was calibrated against*, not as a
free-standing hardware spec: β₄ already absorbs the NVLink/HBM ratio, so the
cross-node cost inherits β₄'s calibration as its baseline.

Known approximations, each tracked:

- **Per-collective launch + round-trip cost is not modeled** — only bandwidth is. At
  decode message sizes that fixed cost is plausibly the *dominant* cross-node effect
  ([#1661](https://github.com/inference-sim/inference-sim/issues/1661)).
- The fabric is keyed by GPU type rather than by pool, which is only equivalent while
  #1529's "one `gpu_type` per pool" rule holds
  ([#1662](https://github.com/inference-sim/inference-sim/issues/1662)).
- The **roofline** backend models no communication at all, so a spanning placement is
  unpriced there ([#1663](https://github.com/inference-sim/inference-sim/issues/1663)).
- The all-to-all penalty sums the on-node and off-node portions rather than overlapping
  them, and ignores DeepEP's per-node RDMA coalescing. Both are pessimistic.
- The lumped `TP·DP` MoE group's span is extrapolated from the placed node size, since
  BLIS places a TP group. Combined with `node_pools` + `--dp>1` being a fail-fast
  today, the expert-all-to-all leg ships **inert in every reachable configuration**; it
  becomes reachable with expert-parallel placement
  ([#1548](https://github.com/inference-sim/inference-sim/issues/1548)).

## When to Use Which

| Aspect | Roofline | Trained-Physics (default) |
|--------|----------|---------------------------|
| **When to use** | Quick analytical estimate | Default (generalizes across architectures, workloads, TP) |
| **Data required** | HF `config.json` + `--hardware` + `--tp` | HF `config.json` + `--hardware` + `--tp` (global coefficients bundled) |
| **GPU step time accuracy** | Good (analytical) | Better (13 global params, physics-informed basis functions) |
| **MoE support** | Yes (per-expert FLOPs + effective expert count) | Yes (per-expert FLOPs + effective expert count + β₈ per-MoE-layer overhead) |
| **Alpha model** | α₀ + α₁·inputLen (constant + per-token queueing) | α₀ (constant), α₁ (post-decode fixed), α₂ (per-token) |
| **PostDecodeFixedOverhead** | 0 | α₁ (~777µs) |

!!! tip "Choosing the right mode"
    **Trained-physics** is the default for any model with a HuggingFace `config.json` (generalizes across architectures, workloads, and TP configurations without per-model calibration). **Roofline** for pure analytical estimates when no learned corrections are desired.

!!! warning "Current limitations"
    All analytical latency models support tensor parallelism (TP). MoE data parallelism (`--dp`) is a trained-physics step-time term (#1419) **and** real placement on both `blis run` (#1531) and `blis replay` (#1556) — see the DP-as-placement note above. Expert parallelism (EP) is not yet a *step-time* term (#1548), though trained-physics does model the DP/EP-mode MoE terms described above; `--enable-expert-parallel` with `--dp > 1` fails fast on both commands (#1548). EP **does** change **KV-capacity sizing**: routed-expert weights are charged to the `TP·DP` EP group rather than to each rank's TP group (#1656), which is what lets a large MoE be sized on its real EP topology. That is a memory-footprint effect only — per-token KV bytes are EP-independent. Quantized weight precision (GPTQ, AWQ, FP8, compressed-tensors) is auto-detected from `quantization_config`, model name conventions (e.g., `w4a16`, `FP8`), or `torch_dtype` fallback, and is used for weight bandwidth and model-weight memory. KV-cache storage precision is configured **independently** via `--kv-cache-dtype` (vLLM parity, #1565): `auto` (default) follows the compute dtype, while `fp8` stores the KV cache at 1 byte/element — roughly doubling KV-block capacity — regardless of the weight precision. MFU calibration values are still derived from FP16/BF16 measurements.

## Speculative Decoding / MTP (#1528)

Both backends model the decode-throughput effect of speculative decoding / Multi-Token Prediction (GLM-5.2's 5-token MTP, DeepSeek-V3, EAGLE, Medusa). Enable it with `--num-speculative-tokens K` (draft tokens per step, `0` = off) and the **required** `--speculative-acceptance-rate α` (mean fraction accepted, `[0,1]`); `--speculative-method` optionally labels the scheme.

The model splits the effect into two decoupled quantities:

- **Verify width `w = K+1`** — the target verifies `K` drafts plus 1 bonus token in a single forward pass. This drives the per-step **cost**: the decode compute-FLOPs and KV-bandwidth terms scale by `w`, while the once-per-step weight-load, TP/EP communication, and constant overhead terms do **not**. Cost is therefore *sublinear* in `w` — the physics that makes speculative decoding a net win. Verifying drafts is not free (a config with more drafts has a strictly higher per-step time), but it is cheap relative to the tokens it can produce.
- **Accepted tokens `g = 1 + α·K`** — the sequence advances by `g` tokens per step (mean). This raises throughput and lowers the decode-step count by ≈`g`. It is applied deterministically via a per-request fractional carry (no RNG), so runs stay byte-identical for a seed (INV-6) and every metric is the expectation.

**Progress stops at the completion boundary.** A step is granted only as many tokens as the request still needs to reach the point where it completes, so the final verified block lands exactly on the target instead of 1..`K` past it — the same thing vLLM does (it appends the accepted tokens one at a time and trims the tail once `check_stop` fires). Step time and step count are unaffected (both backends size a decode step by verify width `w`, not by the granted count), so this is not a compute discount: it means a request's output-token count — and a closed-loop `accumulate` session's context growth — is identical to a `K=0` run. The one timing effect is per-token detokenization: the clamped final step charges `OutputTokenProcessingTime` only for the tokens it actually emitted, so a request's total detokenization overhead is exactly `(L−1)×OTPT` as at `K=0` (it was over-charged by up to `K×OTPT` before). Spec-decode E2E/ITL therefore shift slightly downward with this fix; `K=0` runs are byte-identical. Speculative decoding buys fewer, wider decode steps; it never changes token accounting. (Before [#1657](https://github.com/inference-sim/inference-sim/issues/1657) an overshoot was read back as output-accounting corruption and cancelled the entire closed-loop session after round 0.)

`K=0` (the default) leaves step time and progress byte-identical to a pre-feature build. The feature is model-level and supplied by identical flags to `blis run` and `blis replay`, so traces round-trip under INV-13 with no schema change. `α` is user-supplied — BLIS does not predict acceptance (it does not run a real draft model). Under spec-decode the raw ITL percentiles are per-verification-step; use TPOT for per-token latency.

**Known limitation — occupancy contention under saturation ([#1627](https://github.com/inference-sim/inference-sim/issues/1627)).** This model captures spec-decode's *latency* (verify width) and *throughput* (accepted tokens), but accounts KV/token-budget *occupancy* by the accepted count `g`, not the speculative footprint. Real vLLM reserves `K` lookahead KV slots per running request per step *unconditionally* (before acceptance is known) and consumes ~`K+1` of `max_num_batched_tokens`. So below KV saturation the two agree (the extra reservation never binds), but **above** saturation BLIS over-predicts max batch size and under-predicts preemption — it shows the MTP speedup without the contention penalty that erodes it. The stated use case (mean throughput / capacity planning below saturation) is faithful; the saturated regime is optimistic. Tracked in #1627. The boundary clamp above sharpens this in one place: BLIS caps the *scheduler grant*, whereas vLLM grants `K+1`, reserves the lookahead KV, and trims only afterwards — so on a request's final step BLIS's KV reservation and `max_num_batched_tokens` debit are further below vLLM's than they were before. Same divergence class, same tracking issue.

The `--speculative-method` values (`mtp`, `eagle`, `medusa`, `ngram`, `draft`) are BLIS labels, not verbatim vLLM method strings — the first four match vLLM's literals exactly, while `draft` is BLIS shorthand for vLLM's `draft_model`. The label is informational provenance today (it does not change the step-time math, which is driven entirely by `K` and `α`).

**Backend note — verify-width attention.** The two backends scale the decode **attention** term differently under `w`, and both are physically defensible:

- **trained-physics** keeps decode attention on `sumCtx` (per active sequence) and deliberately does **not** scale it by `w`: the `w` verified positions are contiguous and attend the same shared KV context, read once per step. This is the more accurate model for MTP.
- **roofline** routes `w` through its per-request FLOPs helper, so the decode attention-score ops scale with `w` (and its `effectiveCtx` picks up a small prefill-style `(w-1)/2` term).

Both preserve `K=0` byte-identity and monotonicity in `K`; they differ only in the *shape* of the cost-vs-`w` curve, not in correctness. Keep this in mind when calibrating verify-width cost against a specific backend.

## Pluggable Architecture

The `LatencyModel` interface (defined in `sim/latency_model.go`) has four methods:

| Method | Purpose |
|--------|---------|
| `StepTime(batch)` | Duration of one batch step given the running batch |
| `QueueingTime(req)` | Arrival-to-queue delay for a request |
| `OutputTokenProcessingTime()` | Per-token post-processing time |
| `PostDecodeFixedOverhead()` | Fixed per-request overhead at completion (0 for roofline, non-zero for trained-physics) |

All time estimates are in microseconds (ticks).

New backends register via the `NewLatencyModelFunc` variable in `sim/latency_model.go`. The `sim/latency/register.go` file uses `init()` to wire the factory, breaking the import cycle between `sim/` (interface owner) and `sim/latency/` (implementation). To add a custom backend, implement the four methods and register your factory via `init()` in a sub-package. See [Extension Recipes](../contributing/extension-recipes.md) for a step-by-step guide.

## Further Reading

- [Roofline Estimation](../concepts/roofline.md) -- the mathematical model behind roofline step time calculation
- [Configuration Reference](../reference/configuration.md#roofline-mode) -- all roofline-related CLI flags

# Concepts & Glossary

This page defines terminology used throughout BLIS documentation. Terms are listed alphabetically with cross-references to the relevant design pages.

---

### Admission Policy

A cluster-level gate that decides whether an incoming request enters the routing pipeline or is rejected. Built-in policies: `always-admit` (accept all), `token-bucket` (rate-limiting), `reject-all` (testing only). See [Cluster Architecture](architecture.md#admission-pipeline).

### Alpha Coefficients

Three regression coefficients `[alpha0, alpha1, alpha2]` that model non-GPU overhead per request. `alpha0 + alpha1 * input_length` estimates queueing delay (tokenization, API serialization); `alpha2` estimates output token processing time. These overheads are added to per-request metrics but do not block the simulation clock. See [Core Engine: Latency Models](core-engine.md#latency-models).

### Batch Formation

The process of selecting which requests from the wait queue join the running batch for the next step. BLIS implements vLLM-style continuous batching with chunked prefill and preemption. See [Core Engine: Batch Formation](core-engine.md#batch-formation).

### Beta Coefficients

Three regression coefficients `[beta0, beta1, beta2]` that predict GPU step time: `beta0 + beta1 * cache_miss_tokens + beta2 * decode_tokens`. Trained offline via Bayesian optimization against real vLLM measurements. See [Core Engine: Latency Models](core-engine.md#latency-models).

### Block (KV Block)

The unit of KV cache allocation. Each block holds a fixed number of tokens (default: 16). Requests are allocated blocks proportional to their token count. Blocks are reference-counted and can be shared across requests via prefix caching. See [Core Engine: KV Cache](core-engine.md#kv-cache-management).

### Calibration Report

The JSON output of `blis calibrate` comparing real observed latencies against simulator predictions. Contains per-request TTFT and E2E deltas, aggregate error metrics (MAPE, Pearson correlation, percentile comparisons), bias assessment, and a quality rating. See [Observe / Replay / Calibrate](../guide/observe-replay-calibrate.md#blis-calibrate).

### Chunked Prefill

A vLLM optimization where long prefill sequences are split into chunks that fit within the per-step token budget (`max-num-scheduled-tokens`). Controlled by `--long-prefill-token-threshold`. See [Core Engine: Batch Formation](core-engine.md#batch-formation).

### Cohort

A workload primitive that models a population of similar clients sharing the same arrival process, token distributions, and SLO class, with an optional time-varying lifecycle modifier. A cohort specifies a `population` count and one of three lifecycle patterns (`diurnal`, `spike`, `drain`); BLIS expands it into individual `ClientSpec` entries at workload generation time, each seeded independently for staggered arrival times. Distinct from a single `ClientSpec` in that cohorts directly model populations of concurrent users rather than a single traffic source. See [Workload Specifications: Cohort Dynamics](../guide/workloads.md#cohort-dynamics) and [Workload Spec Schema](../reference/workload-spec.md).

### Continuous Batching

The serving strategy where new requests can join the running batch between decode steps, rather than waiting for the entire batch to complete. BLIS models this by re-evaluating the batch composition at every step. See [Core Engine: Batch Formation](core-engine.md#batch-formation).

### Counterfactual Regret

A trace-level metric that measures how much better an alternative routing decision could have been. For each routing decision, BLIS scores all candidate instances and computes `regret = best_score - chosen_score`. Useful for offline analysis of routing policy quality. See [Cluster Architecture: Counterfactual Regret](architecture.md#counterfactual-regret).

### Decode Phase

The token generation phase where the model produces output tokens one at a time (or in parallel within a step). Each decode token uses the KV cache from all prior tokens. Decode steps are typically memory-bandwidth-bound. Contrast with *Prefill Phase*.

### Discrete Event Simulation (DES)

A simulation paradigm where the system state changes only at discrete event times. BLIS maintains a priority queue of timestamped events and advances the simulation clock by jumping between events, rather than stepping through fixed time intervals. See [Core Engine: Event Queue](core-engine.md#event-queue).

### Distribution Synthesis

The `--rate` mode of `blis observe` that generates workload from statistical distributions (prompt/output token counts, arrival rate) instead of a workload spec YAML file. Useful for quick single-client experiments without crafting a full workload specification. See [Observe / Replay / Calibrate: Distribution Synthesis Flags](../guide/observe-replay-calibrate.md#distribution-synthesis-flags).

### E2E (End-to-End Latency)

Total time from request arrival to final token completion. Computed as `TTFT + sum(ITLs)`, where each ITL includes step time plus output processing overhead (alpha2). See [Core Engine: Metrics](core-engine.md#metrics).

### Effective Load

A routing signal computed as `QueueDepth + BatchSize + InFlightRequests`. Because `InFlightRequests` tracks the full dispatch-to-completion window, it overlaps with `QueueDepth` and `BatchSize` — this intentional double-counting combines stale Prometheus signals with the synchronous gateway counter. Used by least-loaded routing and load-balance scoring. See [Cluster Architecture: Routing Pipeline](architecture.md#routing-pipeline).

### Fitness Score

A single numeric value summarizing multi-objective simulation performance. Computed as a weighted combination of configurable metrics (TTFT percentiles, E2E percentiles, throughput). Latency metrics normalized via `1/(1 + value/1000)`; throughput metrics via `value/(value + reference)`. See [Configuration Reference](../reference/configuration.md#fitness-evaluation).

### GAIE (Gateway API Inference Extension)

The Kubernetes gateway component in the [llm-d](#llm-d) project responsible for admission control and request routing to LLM inference engines. BLIS's `gaie-legacy` admission policy replicates GAIE's production saturation-based shedding behavior: pool-average saturation is computed as `avg(max(qd/qdThreshold, kvUtil/kvThreshold))` across instances; non-sheddable requests always pass, sheddable requests (priority < 0) are rejected when saturation ≥ 1.0. Default thresholds match GAIE production: `qdThreshold=5`, `kvThreshold=0.8`. See [Admission Control: GAIE-Legacy Admission](../guide/admission.md#gaie-legacy-admission).

### Gateway Queue

An optional flow-control buffer between admission and routing, enabled with `--flow-control`. When active, admitted requests enter a priority-ordered queue instead of routing immediately. A saturation detector (`--saturation-detector`) evaluates cluster capacity on each step completion; when saturation falls below 1.0, the highest-priority waiting request is dequeued and routed with fresh routing snapshots (late-binding dispatch). Requests that exceed `--max-gateway-queue-depth` are shed and counted in the INV-1 conservation formula. See [Cluster Architecture](architecture.md#admission-pipeline).

### HOL Blocking (Head-of-Line Blocking)

A scheduling pathology where one long prefill monopolizes the GPU for an entire step, preventing shorter requests from entering decode until it completes. In BLIS, a 2,048-token prompt can occupy an entire step (~97 ms on Qwen3-14B / H100 / TP=1 in roofline mode), blocking all decode-phase requests during that step. Detected and counted as "HOL Blocking Events" in the anomaly counters. The mitigation is chunked prefill (`--long-prefill-token-threshold`), which splits long prefills so decode requests can be interleaved. See [KV Cache: Chunked Prefill](../guide/kv-cache.md#chunked-prefill) and [Chunked Prefill](#chunked-prefill).

### Horizon

The simulation time limit in ticks (microseconds). The simulation stops when the clock exceeds the horizon or all requests complete, whichever comes first. See [Configuration Reference](../reference/configuration.md#simulation-control).

### ITL (Inter-Token Latency)

The observed time between consecutive decode steps for a single request. A request generating N output tokens produces N-1 ITL entries (the number of inter-token gaps). ITL varies with batch composition changes between steps. Mean ITL is reported as TPOT (Time Per Output Token).

### KV Cache

GPU memory organized as blocks that store key-value tensors computed during attention. BLIS simulates block allocation, prefix sharing, LRU eviction, and optional CPU offloading without actual GPU memory. See [Core Engine: KV Cache](core-engine.md#kv-cache-management).

### Latency Model

The component that predicts GPU execution time for a batch step. Two modes: *Roofline* (default; analytical FLOPs/bandwidth estimation) and *Trained-Physics* (physics-informed basis functions with architecture-aware MoE scaling). See [Core Engine: Latency Models](core-engine.md#latency-models), [Roofline Estimation](roofline.md), and [Latency Models Guide](../guide/latency-models.md).

### llm-d

An open-source LLM inference routing and serving framework that BLIS is designed to simulate and validate against. The default weighted routing profile (`precise-prefix-cache:2, queue-depth:1, kv-utilization:1`) matches llm-d's production Endpoint Picker scoring pipeline, and several BLIS components directly mirror llm-d counterparts: `gaie-legacy` admission replicates [GAIE](#gaie-gateway-api-inference-extension) behavior, `vllm-dp` routing replicates `DPLBAsyncMPClient`, and scorer names map to llm-d scorer identifiers. See [Cluster Architecture: Routing Pipeline](architecture.md#routing-pipeline).

### MaxModelLen

Maximum total sequence length (input + output) for a single request, in tokens. Mirrors vLLM's `--max-model-len`. When set (> 0), requests whose input alone fills the context window (`input >= MaxModelLen`) or whose input + output budget exceeds it are dropped before entering the wait queue. A three-part proactive cap matches vLLM `scheduler.py:773-774`: FormBatch clamps token scheduling to `maxModelLen - 1 - ProgressIndex`, executeBatchStep skips decode when 0 tokens allocated, and processCompletions force-completes at the `maxModelLen - 1` boundary. Output per length-capped request: `maxModelLen - 1 - inputLen`. Set to 0 for unlimited. Auto-derived from `max_position_embeddings` in roofline and trained-physics modes, with [`rope_scaling`](#rope_scaling) factor application and KV-feasible capping. See [Configuration Reference](../reference/configuration.md#simulation-control).

### MFU (Model FLOPS Utilization)

The fraction of a GPU's theoretical peak TFLOPS actually achieved during model execution, accounting for kernel efficiency, memory-access patterns, and CUDA overhead. BLIS uses separate MFU values for the prefill phase (`mfuPrefill`, typically higher because prefill is compute-bound) and the decode phase (`mfuDecode`, lower because decode is memory-bandwidth-bound). Both are configured per GPU in `hardware_config.json`. In trained-physics mode, MFU-equivalent corrections are captured by the learned β₁–β₄ coefficients rather than explicit fractions. See [Roofline Estimation](roofline.md).

### MoE (Mixture of Experts)

A transformer architecture variant where each layer routes each token to a subset of specialist sub-networks (experts) rather than processing all tokens through shared weights. BLIS distinguishes two MoE layouts: *uniform MoE* (every layer is a MoE layer, e.g., Mixtral-8x7B) and *interleaved MoE* (alternating MoE and dense layers, e.g., Scout). The trained-physics latency model applies a per-MoE-layer overhead term (β₈) only to interleaved architectures, auto-detected from `interleave_moe_layer_step` in the HuggingFace `config.json`. See [Latency Models Guide](../guide/latency-models.md).

### Observe / Replay / Calibrate Pipeline

The end-to-end workflow of `blis observe` → `blis replay` → `blis calibrate` for validating simulator accuracy against real inference servers. Each stage is independently useful: observe collects latency baselines, replay tests simulator behavior on real traces, and calibrate compares results. See [Observe / Replay / Calibrate](../guide/observe-replay-calibrate.md).

### Oracle Knowledge Boundary (INV-9)

The principle that control-plane decisions (admission, routing, scheduling, priority) must not read `Request.OutputTokens`. The actual output token count is oracle knowledge — known only after generation completes. The control plane uses `MaxOutputLen` (client-declared output budget) or input-only checks instead. Only the execution engine (batch step processing, completion detection) may access `OutputTokens` for token generation and determining when a request finishes. See [Standards: Invariants](../contributing/standards/invariants.md).

### Pending Requests

Requests that have been routed to an instance but not yet enqueued (the queueing event hasn't fired). Tracked per-instance to prevent routing pile-on at high arrival rates. Decremented when the `QueuedEvent` fires. See [Cluster Architecture: Routing Pipeline](architecture.md#routing-pipeline).

### Policy Bundle

A YAML configuration file (`--policy-config`) that specifies admission, routing, priority, and scheduling policies in one place. CLI flags override bundle values when explicitly set. See [Configuration Reference](../reference/configuration.md#policy-bundle).

### Preemption

KV cache eviction under memory pressure. When the batch formation algorithm cannot allocate blocks for a continuing request, it evicts requests from the batch, frees their blocks, and re-enqueues them at the front of the wait queue. By default (`--preemption-policy fcfs`), BLIS evicts from the batch tail. With `--preemption-policy priority`, the least-urgent SLO tier is evicted first. See [Core Engine: Batch Formation](core-engine.md#batch-formation).

### Prefill Phase

The initial processing phase where the model computes attention over all input tokens. Prefill is compute-bound for large inputs. After prefill completes, the request transitions to decode. TTFT is recorded at this boundary.

### Prefix Caching

Reuse of KV blocks across requests that share a common input prefix. BLIS uses hierarchical block hashing: each block's hash chains with the prior block's hash, enabling semantic prefix matching. Shared blocks are reference-counted and exempt from eviction while in use. See [Core Engine: KV Cache](core-engine.md#kv-cache-management).

### Prefix-Affinity Scoring

A routing scorer that directs requests to instances likely to have their prefix cached. Uses a lightweight router-side cache index (not the actual per-instance KV cache) to estimate cache hit probability per instance. Score range [0, 1]. See [Cluster Architecture: Scorer Composition](architecture.md#scorer-composition).

### Priority Policy

A per-instance policy that assigns a numeric priority score to each request before batch formation. Used by priority-aware schedulers to reorder the wait queue. Built-in policies: `constant`, `slo-based`, `inverted-slo` (testing only). Note: despite its name, `slo-based` currently uses only request age (favoring older requests), not per-request SLO metadata. See [Core Engine: Scheduling](core-engine.md#scheduling-policies).

### Roofline Model

An analytical latency estimation technique that predicts step time as `max(FLOPs / peak_compute, bytes / peak_bandwidth)`. Requires only the model's HuggingFace `config.json` and GPU hardware specs. No training data needed. See [Roofline Estimation](roofline.md).

### rope_scaling

A field in a model's HuggingFace `config.json` that extends the effective context window beyond the base `max_position_embeddings` value by scaling the positional frequencies of RoPE (Rotary Position Embeddings). The object contains a `type` (or `rope_type`) field identifying the scaling method and a `factor` field specifying the multiplier.

BLIS applies the scaling factor when auto-deriving [`MaxModelLen`](#maxmodellen) in roofline and trained-physics modes, following vLLM's blacklist approach: types `linear`, `dynamic`, `yarn`, `default`, and `mrope` apply the factor; types `su`, `longrope`, and `llama3` are excluded because they encode the full extended context length directly in `max_position_embeddings`. For `yarn`, `original_max_position_embeddings` is used as the base when present. Models whose `model_type` contains `gemma3` skip `rope_scaling` entirely — their `max_position_embeddings` is already pre-scaled. Override the auto-derived value with `--max-model-len`. See [Latency Models Guide](../guide/latency-models.md#roofline-mode-default) and [Configuration Reference](../reference/configuration.md#simulation-control).

### Routing Policy

A cluster-level policy that selects which instance receives an admitted request. Simple policies (round-robin, least-loaded) use fixed rules. The weighted scoring policy composes multiple scorers with configurable weights. See [Cluster Architecture: Routing Pipeline](architecture.md#routing-pipeline).

### Routing Snapshot

A point-in-time view of instance state used for routing decisions. Contains queue depth, batch size, KV utilization, cache hit rate, and pending request count. Signals have different freshness tiers depending on how they're collected. See [Cluster Architecture: Signal Freshness](architecture.md#signal-freshness).

### Scheduling Delay

The elapsed time between a request's arrival and the moment it enters the running batch, reported as `scheduling_delay_ms` (per-request JSON) and `scheduling_delay_p99_ms` (aggregate). Includes the alpha queueing overhead (`alpha0 + alpha1 × inputLen`) plus wait queue residence time. Distinct from TTFT, which additionally includes prefill compute time. High scheduling delay with low preemption rate indicates queue saturation (add instances); low scheduling delay with high TTFT indicates compute saturation (reduce batch size or enable chunked prefill). See [Metrics & Results](../guide/results.md#scheduling-delay) and [Core Engine: Metrics](core-engine.md#metrics).

### Scorer

A component in the weighted scoring pipeline that produces a per-instance score in [0, 1] for a specific signal dimension. Built-in scorers: `precise-prefix-cache`, `prefix-affinity`, `no-hit-lru`, `queue-depth`, `kv-utilization`, `load-balance`, `active-requests`, `running-requests`, `load-aware`. Most scorers produce scores in [0, 1]; `load-aware` uses [0, 0.5] per llm-d semantics. Scores are multiplied by weights and summed. See [Cluster Architecture: Scorer Composition](architecture.md#scorer-composition).

### Seed

The random seed for deterministic simulation. Same seed produces byte-identical stdout across runs (INV-6). BLIS uses partitioned RNG to isolate randomness across subsystems. See [Configuration Reference](../reference/configuration.md#simulation-control).

### SLO Class

A label that categorizes each request by service-level objective sensitivity, governing admission decisions, scheduling priority, gateway queue dispatch order, and per-class metric reporting. BLIS uses five classes with GAIE-compatible integer priorities: `critical` (4), `standard` (3), `batch` (−1), `sheddable` (−2), `background` (−3). Classes with negative priority are *sheddable* — preferentially dropped under overload or when a tenant exceeds their capacity budget. Non-sheddable classes (`critical`, `standard`) are always protected. Set via `slo_class` in the workload spec YAML. See [Workload Specifications](../guide/workloads.md#slo-classes) and [Configuration Reference](../reference/configuration.md#slo-tier-priorities).

### Step

A single iteration of the inference engine. Each step processes one batch: prefill tokens for new/continuing requests and decode tokens for generating output. Step time is predicted by the latency model. See [Core Engine: Step Phases](core-engine.md#step-phases).

### Tensor Parallelism (TP)

A parallelism strategy that shards a model's weight tensors across multiple GPUs, reducing per-GPU compute load and KV memory requirements. Configured via `--tp`. Higher TP reduces per-request latency (fewer FLOPs per GPU, less KV memory per rank) but adds All-Reduce communication overhead per transformer layer (modeled by the β₄ coefficient in trained-physics mode). Increasing the instance count (replication) instead increases throughput without reducing per-request latency. BLIS does not model data parallelism (DP) or expert parallelism (EP). See [Latency Models Guide](../guide/latency-models.md#tensor-parallelism-and-roofline) and [Roofline Estimation](roofline.md).

### Tick

The fundamental time unit in BLIS, representing one microsecond. All timestamps, durations, and latencies are measured in ticks. The simulation clock advances in ticks.

### Tiered KV Cache

An extension of the KV cache with GPU and CPU tiers. When GPU utilization exceeds a threshold, blocks are offloaded to CPU memory. On cache miss, blocks can be reloaded from CPU with a transfer latency penalty. See [Core Engine: KV Cache](core-engine.md#kv-cache-management).

### TPOT (Time Per Output Token)

The mean inter-token latency across all decode steps for a request, computed as `mean(ITL)`. TPOT represents the steady-state token generation speed once prefill completes; lower TPOT means faster streaming throughput. Each ITL sample includes the step time plus the alpha2 output-processing overhead per token. Reported as the `itl_mean_ms` field in JSON output. See [ITL](#itl-inter-token-latency) and [Core Engine: Metrics](core-engine.md#metrics).

### TraceV2

The trace format used by the observe/replay/calibrate pipeline, consisting of two files: a header YAML file (recording model, server config, and observation metadata) and a data CSV file (recording per-request timing: arrival time, TTFT, E2E latency, token counts). See [Observe / Replay / Calibrate](../guide/observe-replay-calibrate.md).

### TTFT (Time To First Token)

Time from request arrival to completion of the prefill phase (first output token ready). Includes queueing delay, prefill step times, and output processing overhead (alpha2). A key latency SLO metric for interactive applications. See [Core Engine: Metrics](core-engine.md#metrics).

### Warmup Requests

Initial requests dispatched by `blis observe` that are excluded from trace output (`--warmup-requests N`). Warmup requests allow the server to populate its KV cache and reach steady-state scheduling before measurement begins, avoiding cold-start artifacts in the recorded trace. See [Observe / Replay / Calibrate: blis observe](../guide/observe-replay-calibrate.md#blis-observe).

### Work-Conserving

The property that the simulator never idles while requests are waiting. After every step completion, if the wait queue is non-empty, a new `StepEvent` is scheduled immediately (INV-8). See [Core Engine: Event Queue](core-engine.md#event-queue).

### Workload Specification

A YAML file (`--workload-spec`) defining multi-client workloads with per-client arrival distributions, token length distributions, prefix groups, and SLO classes. Supports `poisson`, `gamma`, `weibull`, and `constant` arrival processes and `gaussian`, `exponential`, `pareto_lognormal`, `constant`, and `empirical` token distributions. See [Configuration Reference](../reference/configuration.md#workload-modes).

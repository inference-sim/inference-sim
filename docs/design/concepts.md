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

### Chunked Prefill

A vLLM optimization where long prefill sequences are split into chunks that fit within the per-step token budget (`max-num-scheduled-tokens`). Controlled by `--long-prefill-token-threshold`. See [Core Engine: Batch Formation](core-engine.md#batch-formation).

### Continuous Batching

The serving strategy where new requests can join the running batch between decode steps, rather than waiting for the entire batch to complete. BLIS models this by re-evaluating the batch composition at every step. See [Core Engine: Batch Formation](core-engine.md#batch-formation).

### Counterfactual Regret

A trace-level metric that measures how much better an alternative routing decision could have been. For each routing decision, BLIS scores all candidate instances and computes `regret = best_score - chosen_score`. Useful for offline analysis of routing policy quality. See [Cluster Architecture: Counterfactual Regret](architecture.md#counterfactual-regret).

### Decode Phase

The token generation phase where the model produces output tokens one at a time (or in parallel within a step). Each decode token uses the KV cache from all prior tokens. Decode steps are typically memory-bandwidth-bound. Contrast with *Prefill Phase*.

### Discrete Event Simulation (DES)

A simulation paradigm where the system state changes only at discrete event times. BLIS maintains a priority queue of timestamped events and advances the simulation clock by jumping between events, rather than stepping through fixed time intervals. See [Core Engine: Event Queue](core-engine.md#event-queue).

### E2E (End-to-End Latency)

Total time from request arrival to final token completion. Computed as `TTFT + sum(ITLs)`, where each ITL includes step time plus output processing overhead (alpha2). See [Core Engine: Metrics](core-engine.md#metrics).

### Effective Load

A routing signal computed as `QueueDepth + BatchSize + PendingRequests`. Represents the total work visible at an instance, including requests that have been routed but not yet enqueued. Used by least-loaded routing and load-balance scoring. See [Cluster Architecture: Routing Pipeline](architecture.md#routing-pipeline).

### Fitness Score

A single numeric value summarizing multi-objective simulation performance. Computed as a weighted combination of configurable metrics (TTFT percentiles, E2E percentiles, throughput). Latency metrics normalized via `1/(1 + value/1000)`; throughput metrics via `value/(value + reference)`. See [Configuration Reference](configuration.md#fitness-evaluation).

### Horizon

The simulation time limit in ticks (microseconds). The simulation stops when the clock exceeds the horizon or all requests complete, whichever comes first. See [Configuration Reference](configuration.md#simulation-control).

### ITL (Inter-Token Latency)

The observed time between consecutive decode steps for a single request. ITL varies with batch composition changes between steps. Mean ITL is reported as TPOT (Time Per Output Token).

### KV Cache

GPU memory organized as blocks that store key-value tensors computed during attention. BLIS simulates block allocation, prefix sharing, LRU eviction, and optional CPU offloading without actual GPU memory. See [Core Engine: KV Cache](core-engine.md#kv-cache-management).

### Latency Model

The component that predicts GPU execution time for a batch step. Two modes: *Blackbox* (trained regression coefficients) and *Roofline* (analytical FLOPs/bandwidth estimation). See [Core Engine: Latency Models](core-engine.md#latency-models) and [Roofline Estimation](roofline.md).

### Pending Requests

Requests that have been routed to an instance but not yet enqueued (the queueing event hasn't fired). Tracked per-instance to prevent routing pile-on at high arrival rates. Decremented when the `QueuedEvent` fires. See [Cluster Architecture: Routing Pipeline](architecture.md#routing-pipeline).

### Policy Bundle

A YAML configuration file (`--policy-config`) that specifies admission, routing, priority, and scheduling policies in one place. CLI flags override bundle values when explicitly set. See [Configuration Reference](configuration.md#policy-bundle).

### Preemption

KV cache eviction under memory pressure. When the batch formation algorithm cannot allocate blocks for a continuing request, it evicts requests from the batch tail, frees their blocks, and re-enqueues them at the front of the wait queue. See [Core Engine: Batch Formation](core-engine.md#batch-formation).

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

### Routing Policy

A cluster-level policy that selects which instance receives an admitted request. Simple policies (round-robin, least-loaded) use fixed rules. The weighted scoring policy composes multiple scorers with configurable weights. See [Cluster Architecture: Routing Pipeline](architecture.md#routing-pipeline).

### Routing Snapshot

A point-in-time view of instance state used for routing decisions. Contains queue depth, batch size, KV utilization, cache hit rate, and pending request count. Signals have different freshness tiers depending on how they're collected. See [Cluster Architecture: Signal Freshness](architecture.md#signal-freshness).

### Scorer

A component in the weighted scoring pipeline that produces a per-instance score in [0, 1] for a specific signal dimension. Built-in scorers: `prefix-affinity`, `queue-depth`, `kv-utilization`, `load-balance`. Scores are multiplied by weights and summed. See [Cluster Architecture: Scorer Composition](architecture.md#scorer-composition).

### Seed

The random seed for deterministic simulation. Same seed produces byte-identical stdout across runs (INV-6). BLIS uses partitioned RNG to isolate randomness across subsystems. See [Configuration Reference](configuration.md#simulation-control).

### Step

A single iteration of the inference engine. Each step processes one batch: prefill tokens for new/continuing requests and decode tokens for generating output. Step time is predicted by the latency model. See [Core Engine: Step Phases](core-engine.md#step-phases).

### Tick

The fundamental time unit in BLIS, representing one microsecond. All timestamps, durations, and latencies are measured in ticks. The simulation clock advances in ticks.

### Tiered KV Cache

An extension of the KV cache with GPU and CPU tiers. When GPU utilization exceeds a threshold, blocks are offloaded to CPU memory. On cache miss, blocks can be reloaded from CPU with a transfer latency penalty. See [Core Engine: KV Cache](core-engine.md#kv-cache-management).

### TTFT (Time To First Token)

Time from request arrival to completion of the prefill phase (first output token ready). Includes queueing delay, prefill step times, and output processing overhead (alpha2). A key latency SLO metric for interactive applications. See [Core Engine: Metrics](core-engine.md#metrics).

### Workload Specification

A YAML file (`--workload-spec`) defining multi-client workloads with per-client arrival distributions, token length distributions, prefix groups, and SLO classes. Supports `poisson`, `gamma`, and `weibull` arrival processes and `gaussian`, `exponential`, `pareto_lognormal`, `constant`, and `empirical` token distributions. See [Configuration Reference](configuration.md#workload-modes).

### Work-Conserving

The property that the simulator never idles while requests are waiting. After every step completion, if the wait queue is non-empty, a new `StepEvent` is scheduled immediately (INV-8). See [Core Engine: Event Queue](core-engine.md#event-queue).

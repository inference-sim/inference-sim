# Routing Policies

This guide covers how BLIS distributes incoming requests across instances in cluster mode. For single-instance simulation, routing is not applicable. For instance-level request ordering, see [Scheduling & Priority](scheduling.md).

```bash
# Quick example: compare round-robin vs weighted routing
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml --scenarios testdata/scenarios \
  --registry $PWD/blis-registry \
  --num-instances 4 --rate 100 --num-requests 500 \
  --routing-policy weighted --trace-level decisions --summarize-trace
```

## Available Policies

| Policy | Flag Value | Strategy |
|--------|-----------|----------|
| **Round-robin** | `round-robin` | Cyclic assignment — request N goes to instance N % k |
| **Least-loaded** | `least-loaded` | Send to the instance with lowest `EffectiveLoad` |
| **Weighted** | `weighted` | Composable multi-scorer pipeline (default: llm-d parity) |
| **Always-busiest** | `always-busiest` | Pathological template — sends to the most loaded instance (for testing) |

## Weighted Scoring (Composable Pipeline)

The `weighted` routing policy is the most flexible. It combines multiple scoring dimensions, each evaluating instances on a `[0, 1]` scale:

```bash
--routing-policy weighted --routing-scorers "precise-prefix-cache:2,queue-depth:1,kv-utilization:1"
```

### Available Scorers

| Scorer | What It Measures | llm-d Equivalent |
|--------|-----------------|------------------|
| `prefix-affinity` | Proportional prefix match ratio via router-side block hash cache | prefix-scorer |
| `precise-prefix-cache` | Actual KV cache state query with min-max normalization; all-equal (including all-zero) → 1.0 (llm-d parity) | precise-prefix-cache-scorer |
| `no-hit-lru` | LRU positional scoring for cold requests (warm = 0.5) | no-hit-lru-scorer |
| `queue-depth` | Queue depth: `QueueDepth` only (min-max normalized) | queue-scorer |
| `kv-utilization` | Inverse KV utilization: `1 - KVUtilization` | kv-cache-utilization-scorer |
| `load-balance` | Inverse transform: `1 / (1 + effectiveLoad)` | BLIS-native (no llm-d equivalent) |
| `active-requests` | In-flight requests: `(maxCount - count) / maxCount` | active-request-scorer |
| `running-requests` | Batch size (min-max normalized) | running-requests-size-scorer (GIE) |
| `load-aware` | Queue depth (linear threshold-capped, range [0, 0.5]) | load-aware-scorer |
| `vllm-dp` | vLLM data-parallel routing: `waiting × 4 + running` (inverted min-max) | DPLBAsyncMPClient.get_core_engine_for_request |
| `prefix-cache` | `matched/total` prefix blocks from an llm-d prefix producer (optionally blended with a match-length term) | prefix-cache-scorer |
| `token-load` | `1 − min(inFlight + uncached incoming tokens, threshold)/threshold`; idle = 1.0 | token-load-scorer |

!!! note "Prefix-affinity is a scorer, not a standalone policy"
    The `prefix-affinity` scorer operates within the `weighted` routing pipeline, composed with load-balancing scorers. It uses a router-side `PrefixCacheIndex` with proportional block hash matching and LRU eviction. Always pair it with at least one load-aware scorer (queue-depth or kv-utilization) to prevent cold-start pile-on.

### Default Profile

When `--routing-scorers` is not specified, the default profile is:

```
precise-prefix-cache:2, queue-depth:1, kv-utilization:1
```

This matches the llm-d production scoring pipeline. Weights are relative — only ratios matter. `[2, 1, 1]` behaves identically to `[0.5, 0.25, 0.25]`.

## vLLM Data-Parallel Routing

BLIS supports vLLM's internal load-balancing algorithm via the `vllm-dp` scorer:

```bash
# Oracle mode (immediate signals — best for algorithmic studies):
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml --scenarios testdata/scenarios \
  --registry $PWD/blis-registry --routing-scorers "vllm-dp:1" \
  --snapshot-refresh-interval 0

# Staleness-matched mode (matches vLLM's 100ms coordinator interval):
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml --scenarios testdata/scenarios \
  --registry $PWD/blis-registry --routing-scorers "vllm-dp:1" \
  --snapshot-refresh-interval 100000
```

This replicates vLLM's `DPLBAsyncMPClient` formula: selecting the instance with the lowest `waiting × 4 + running` score. The scorer applies inverted min-max normalization to convert vLLM's argmin selection to BLIS's argmax routing framework.

**Signal staleness:** vLLM's coordinator publishes instance stats every 100ms by default (`min_stats_update_interval_ms`). Using `--snapshot-refresh-interval 100000` matches this staleness interval. The default 50ms (`--snapshot-refresh-interval 50000`) matches llm-d's `RefreshMetricsInterval`. Pass `--snapshot-refresh-interval 0` for oracle mode (immediate signals, no staleness) — the cleanest baseline for algorithmic comparisons.

**Known limitation:** vLLM's client speculatively increments its cached waiting count after each routing decision (`current_counts[eng_index][0] += client_count`, `core_client.py:1225`) to spread traffic within the 100ms window. BLIS does not model this because BLIS's DES is sequential — requests route one at a time, so the concurrent-routing scenario doesn't arise. In staleness-matched mode, BLIS will concentrate more traffic per snapshot window than real vLLM would.

**Note:** Do not compose `vllm-dp` with other scorers unless comparing hybrid strategies. For vLLM parity, use `vllm-dp:1` alone — composing with other scorers double-normalizes dimensions and breaks vLLM fidelity.

## Signal Freshness

> **Canonical source:** Signal freshness tiers are specified in [`docs/contributing/standards/invariants.md`](../contributing/standards/invariants.md) (INV-7). The descriptions below provide additional user-facing context; `invariants.md` is authoritative if they diverge.

!!! warning "Not all routing signals are equally fresh"
    In production inference serving systems (e.g., llm-d), the router is a separate process from the inference engines. Some signals are maintained at the router level, while others require periodic reporting from instances. BLIS models this asymmetry.

### Why Signals Have Different Staleness

Different signals originate from different places in the system:

- **Router-local signals** are maintained by the router itself — they're always current because the router controls them directly.
- **Instance-internal signals** live on the inference engine and must be communicated to the router — they're inherently stale by the reporting interval.

BLIS models three signal freshness tiers:

| Tier | Signals | Source | Freshness |
|------|---------|--------|-----------|
| **Router-local** | InFlightRequests, `prefix-affinity` router-side cache index | Router increments InFlightRequests at dispatch, decrements at completion; the `prefix-affinity` scorer's router-side LRU index is updated after each routing decision | Always fresh — router owns this state |
| **Instance-reported (Immediate/Periodic)** | QueueDepth, BatchSize, KVUtilization, FreeKVBlocks, CacheHitRate, PreemptionCount | Instance-internal state (scheduler queue, running batch, KV cache) | Default (`--snapshot-refresh-interval 50000`): Periodic at 50ms (llm-d parity). When `--snapshot-refresh-interval 0`: Immediate (read from instance at routing time). All Prometheus-sourced signals share the same refresh interval, matching real vLLM's single `/metrics` endpoint. |
| **Periodic (precise prefix-cache query)** | `precise-prefix-cache` / `no-hit-lru` cache-block hit counts | Actual instance KV cache state (via `CachedSnapshotProvider`) | Governed by `--cache-signal-delay` (default 50ms; set to 0 for synchronous ground-truth queries). Distinct from the router-local `prefix-affinity` index above. |

!!! info "DES semantics of 'Immediate' mode"
    "Immediate" means "re-read from the instance object at query time" — NOT "perfectly synchronized with the simulation clock." At the same clock tick, cluster events are processed before instance events (determinism rule). So a routing decision at time T sees QueueDepth that hasn't yet processed instance events at time T. This is a determinism mechanism (INV-6), not a freshness guarantee.

### Staleness Impact

At high request rates, many routing decisions occur between KV utilization updates (step time varies by model — ~6ms for Qwen3-14B / H100 / TP=1 at low load, longer under batch saturation). If using `kv-utilization:1` alone, all decisions within one step see the same stale utilization — this can cause severe load imbalance.

!!! tip "Safe zone for `--snapshot-refresh-interval`"
    Below **5ms** (~1 step time): no degradation. At 10ms: 14% TTFT p99 increase. At 100ms: +354% (measured with the prior default `prefix-affinity:3, queue-depth:2, kv-utilization:2`). The default composite profile (`precise-prefix-cache:2, queue-depth:1, kv-utilization:1`) is resilient — queue-depth's QueueDepth signal complements stale KV signals. The exact mitigation percentage may differ from the prior profile due to different weight ratios and scorer staleness characteristics (`precise-prefix-cache` has its own staleness model via `--cache-signal-delay`). For staleness-critical deployments, consider adding `load-balance` which reads EffectiveLoad (includes synchronous InFlightRequests).

## When to Use Which Policy

| Workload | Recommended Policy | Why |
|----------|-------------------|-----|
| Uniform traffic, no prefix sharing | `least-loaded` or `weighted` with `queue-depth:1` | Load balance is the only signal that matters |
| RAG with shared system prompts | `weighted` default or `precise-prefix-cache:3,queue-depth:1` | Prefix-aware scoring maximizes KV cache reuse |
| Mixed SLO classes | `weighted` default + [priority scheduling](scheduling.md) | Routing distributes load; scheduling prioritizes critical requests |
| Low traffic (< 10 req/s) | Any | All policies produce equivalent results within 5% |

## Example: Comparing Policies

BLIS includes a routing comparison script:

```bash
chmod +x examples/routing-comparison.sh
./examples/routing-comparison.sh
```

This runs 5 configurations and shows TTFT p99, target distribution, and throughput for each. See `examples/routing-comparison.sh` for the full script.

## llm-d EndpointPickerConfig (`--epp-config`)

`--epp-config <file>` ingests an llm-d router `EndpointPickerConfig` verbatim (YAML
comments allowed) on `blis run` and `blis replay`. It replaces `--routing-policy`,
`--routing-scorers`, the P/D routing flags (`--prefill-routing-scorers`,
`--decode-routing-scorers`, `--pd-decider`, `--pd-prefix-threshold`) and the
flow-control flags; setting any of them alongside it is refused. The ingested
config, its sha256 and the BLIS mapping of every plugin are written to the
`--metrics-path` file as the `epp` block (never to stdout).

```bash
blis run --scenario <file> --scenarios <dir> --registry <root> --rate 10 --num-requests 200 \
  --epp-config optimized-baseline-cpu-tier.yaml --epp-version v0.10.0 --metrics-path out.json
```

`--epp-version` selects the llm-d-router release whose plugin semantics BLIS
mirrors: `main@981acb8a` (default, alias `main`) or `v0.10.0` (the EPP image
`ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.10.0`, which curvebender deploys).
Use the release your router runs; the deltas are listed below.

### What runs

The profile runs llm-d's order: producers over all candidates, filters in profile
order, weighted scorers over the survivors (each clamped to [0, 1]), then the
max-score pick. If the filters remove every candidate, the request is rejected at
routing, as llm-d fails it. A `disagg-profile-handler` config runs BLIS's
decode-first P/D routing: the `decode` profile picks the decode pod, the prefill
decider decides, and the `prefill` profile picks the prefill pod. The scenario
must define prefill/decode pools exactly when the config is disaggregated.

| llm-d plugin | BLIS implementation |
|---|---|
| `prefix-cache-affinity-filter` | filter: sticky set (match ratio ≥ `affinityThreshold`), exploration (seeded), TTFT gate `inFlightTokens / peakPrefillThroughput · 1000` |
| `concurrency-detector` | filter (`maxConcurrency·(1+headroom)` per endpoint, requests/tokens/hybrid) and, as `flowControl.saturationDetector`, the gateway saturation |
| `token-load-scorer`, `prefix-cache-scorer` | `token-load`, `prefix-cache` scorers |
| `queue-scorer`, `kv-cache-utilization-scorer`, `active-request-scorer`, `running-requests-size-scorer`, `load-aware-scorer`, `lora-affinity-scorer` | the existing scorers (defaults only; parameters are refused) |
| `no-hit-lru-scorer` | `no-hit-lru`, bound to an llm-d prefix producer |
| `approx-prefix-cache-producer` | router-side LRU per instance: autoTune capacity = the instance's GPU KV blocks, block size = engine block size floored at 64, `maxPrefixTokensToMatch` cap; a trailing partial block counts |
| a second approx producer (e.g. the tiered guide's `cpu-prefix-cache-producer`) | exactly as llm-d: its own router-side LRU, sized by `lruCapacityPerServer` (or the GPU block count under `autoTune`), credited at PreRequest. `--epp-cpu-tier-view <name>` is a BLIS opt-in that makes the named approx producer report each instance's CPU-offload-tier-resident prefix instead (#1912); it is listed in the `epp` block's `deviations` |
| `precise-prefix-cache-producer` | the instance's KV view frozen for `--cache-signal-delay` (KV-event propagation): GPU blocks plus, with an offload tier, CPU-tier blocks (read as of the same snapshot time) weighted by `kvCacheBackendConfigs`; speculative indexing with `speculativeTTL` |
| `inflight-load-producer` | router-side in-flight tokens: uncached prompt tokens charged at routing, released at the first token (or, with `addEstimatedOutputTokens`, prefill at the first token and the rest at completion) |
| `max-score-picker`, `single-profile-handler` | argmax with a seeded uniform tie-break; one profile |
| `disagg-profile-handler`, `prefill-filter`, `decode-filter` | decode-first P/D routing; role filters = pool membership |
| `always-disagg-pd-decider`, `prefix-based-pd-decider` | `always`; llm-d's rule `input − cachedBlocks·blockSize ≥ nonCachedTokens` (with `promptTokens`), on the decode pod's match info |
| `flowControl` (`featureGates: [flowControl]`) | gateway queue: `maxRequests`, `maxBytes`, `defaultRequestTTL`, `priorityBands` (caps, round-robin fairness) |

Accepted as inert, and listed in the `epp` block: `token-producer`,
`endpoint-notification-source`, metrics/model data sources and extractors, request
parsers, `request-attribute-reporter`, `disaggregatedset-rollout-screener` (one
revision), `dataLayer`, `kvEventsConfig`, index storage sizing,
`syncCrossReplicaState`, other feature gates. An inert plugin keeps only the
parameters with no simulation meaning (endpoints, credentials, metric names,
headers, rollout scoping); any other parameter is refused. Two inert-plugin
parameters do carry meaning and are mapped: `metrics-data-source.interval` sets
`--snapshot-refresh-interval` (rounded to the EPP's 50 ms base tick; an explicit
conflicting flag is refused) and `static-usage-limit-policy.threshold` sets one
saturation ceiling for every band when that policy is in force. Any other plugin
type, and any parameter BLIS does not model (for example `ttftSource:
latencyPredictor`, `enableEviction: true`, `maxNumOfEndpoints > 1`, `stageOrder:
prefill-first`, E/P/D, zero scorer weights), is refused by name.

### Modeling choices

- **Tie-break.** llm-d's max-score picker orders candidates from a map, shuffles
  (v0.10.0) or rotates ties by a request counter (main); both are a uniform pick
  among the tied maxima. BLIS draws that pick from the seeded router RNG (INV-6).
- **Weights.** llm-d sums `weight · score`; BLIS divides the weights by their total,
  which preserves the argmax.
- **Hashing.** The approx producer chains SHA-256 block hashes where llm-d uses
  xxhash64; prefix semantics are identical.
- **Request bytes.** Flow-control byte limits use 4 bytes per prompt token (llm-d's
  `averageCharactersPerToken`); llm-d counts request body bytes.
- **Output estimate (main).** llm-d main estimates output tokens by an
  `outlen-bucket` request header; BLIS requests carry none, so main's unknown-bucket
  value (1000) applies.
- **P/D charges.** A disaggregated request's prefill charge is released at the
  decode stream's first token, as llm-d does; its prefill request counter follows
  BLIS's in-flight count (prefill completion), not llm-d's StartOfStream. A prefill
  sub-request that ends without completing (timeout, drop) ends the request and
  releases all of its charges, as llm-d frees every entry on stream end.
- **Scheduling timing.** Producers run once per request over every candidate of
  both pools before any profile; PreRequest work (index recording, in-flight
  charges, no-hit-lru moves) runs once both P/D picks are made and not at all if
  scheduling fails, as in llm-d's director. llm-d applies the approx index update
  asynchronously (a goroutine in PreRequest); BLIS applies it synchronously at
  PreRequest, so a request scheduled in the same instant sees it slightly sooner.
- **Rejections.** A request the filters cannot place is a routing rejection
  (INV-1) and is reported to the session driver, so closed-loop runs continue.
- **Tokenization.** Without a `token-producer` (or with its `estimate` backend)
  llm-d hashes byte-packed pseudo-tokens; BLIS uses the request's token IDs, so
  token counts and block boundaries only approximate llm-d's (noted in the `epp`
  block). With the vLLM render backend the two agree.
- **Prefix match unit.** llm-d main prefers an endpoint's `CachePrefixMatchUnit`
  metric; BLIS uses the engine block size.

### Release deltas (`--epp-version`)

| Plugin | v0.10.0 | main@981acb8a |
|---|---|---|
| concurrency-detector filter | no fail-open: all dropped ⇒ empty (`v0.10.0:…/concurrency/detector.go:187-209`) | `failOpen` (default true) returns all; tokens/hybrid count the request's own uncached tokens, an idle endpoint always passes (`main:…/detector.go:249-298`) |
| concurrency-detector saturation | in-flight only | + dispatch reservations between flow-control dispatch and routing (`main:…/detector.go:139-228`) |
| prefix-cache-affinity-filter | missing in-flight signal = zero load; negative threshold accepted | missing signal skips the TTFT gate; threshold must be in [0, 1] (`main:…/prefixcacheaffinity/plugin.go:141-250`) |
| inflight-load-producer | output = round(input·`outputRatio`=1.5); role-blind | output = 1000 (unknown bucket); `outputRatio` removed; prefill-role pods charged prompt only, decode-role pods output only (`main:…/inflightload/producer.go:508-545`) |
| approx-prefix-cache-producer | block 0 inserted first (LRU evicts a prompt's head), linear match | last block first (evicts the tail), galloping match (`main:…/approximateprefix/indexer.go:94-160`). The `blockSizeTokens` default (16 vs 64) has no effect: both releases floor the block size at 64 |
| precise producer tier weights | a tier missing from `kvCacheBackendConfigs` weighs 1.0 | weighs 0.0 (`main:pkg/kvcache/prefix_match.go:44-45`) |
| disagg-profile-handler | an empty prefill result falls back to decode-only (refused at load: a concurrency-detector in the prefill profile is not accepted under v0.10.0) | the request fails |
| config schema | `llm-d.ai/v1alpha1` only | also `llm-d.ai/v1`; `noEndpointRequestTTL` (inert), `bandSelectionPolicyPluginRef` (refused) |

### llm-d conformance

`test/llmd-conformance/{main,v0_10_0}` are separate Go modules that run
llm-d-router's own plugin code and BLIS's ports on the same randomized endpoint
states (rapid) and require identical outputs: filtered sets, per-endpoint scores,
saturation, the P/D decision, the full profile's pick (within BLIS's tied maxima)
and its empty-result failure, and the in-flight ledger over PreRequest /
ResponseBody sequences. Generators vary per-endpoint block size and totals,
attribute presence, tier-weighted match, role labels, output estimation and client
caps, exploration, and endpoint counts down to zero. Not compared differentially:
the approx producer's index (llm-d updates it asynchronously) and flow-control
queue internals, which are unit-tested against the cited semantics. Run locally
(needs network access for the modules and the Go toolchain they pin):

```bash
cd test/llmd-conformance/main && go test ./... -rapid.checks=5000
cd ../v0_10_0 && go test ./... -rapid.checks=5000
```

The `llm-d conformance` workflow runs both on every PR and push to `main`; it
needs network access (module and toolchain downloads), which GitHub-hosted runners
have, and is intended to be a required status check.

## Further Reading

- [Scheduling & Priority](scheduling.md) — instance-level request ordering
- [Admission Control](admission.md) — the gate before routing
- [Cluster Architecture](../concepts/architecture.md) — how the routing pipeline works internally
- [Configuration Reference](../reference/configuration.md#routing-policy) — all routing flags
- [Metrics & Results](results.md) — understanding trace summaries and regret analysis

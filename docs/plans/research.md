# Research Document

## Problem Statement

BLIS (Blackbox Inference Simulator) is a discrete-event simulator for LLM inference serving systems. It models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation.

We recently validated a powerful research methodology: **pose an intuitive hypothesis about system behavior, design experiments to test it, and use hypothesis failures to surface bugs and design limitations.** Specifically, we hypothesized that "prefix-aware routing should outperform load-only routing for prefix-heavy workloads." Testing this hypothesis uncovered 3 bugs in the multi-turn workload generator, led to a new configurable `prefix_length` field, and produced documented example YAMLs with clear TTFT improvements (2.3x better mean, 3.1x better p99).

We want to generate 20 more testable hypotheses that:
1. Are **intuitive** -- explainable to anyone without deep system knowledge
2. Are **testable** -- can be validated with BLIS experiments (existing or near-future capabilities)
3. Are **documentable** -- each becomes an example YAML + comparison script that users can run
4. **Achieve broad coverage** -- spanning routing, scheduling, KV cache, workload patterns, admission control, tiered storage, multi-instance behavior, and latency modeling
5. Follow the pattern: **hypothesis -> experiment design -> predicted outcome -> what bugs/limitations failure would surface**

### BLIS Capabilities (what can be tested today)

**Routing policies:** round-robin, least-loaded, weighted (composable scorer pipeline with prefix-affinity, queue-depth, kv-utilization, load-balance scorers), prefix-affinity (legacy), always-busiest (pathological)

**Scheduling:** FCFS, priority-FCFS, SJF (shortest-job-first), reverse-priority (pathological)

**Admission control:** always-admit, token-bucket, reject-all (pathological)

**KV cache:** Single-tier (GPU), tiered (GPU+CPU with offload/reload and transfer latency), prefix caching with block-level LRU eviction

**Workload generation:** Poisson/Gamma/Weibull arrivals, Gaussian/Exponential/ParetoLogNormal distributions, prefix groups with configurable prefix_length, multi-turn chat with context accumulation, multi-client with SLO classes (realtime/interactive/batch), multimodal tokens, reasoning with think time

**Metrics:** TTFT (mean/p90/p95/p99), E2E latency, ITL (inter-token latency), throughput (req/s, tok/s), scheduling delay, per-instance distribution, anomaly detection (priority inversions, HOL blocking), fitness evaluation with weighted multi-objective scoring

**Decision tracing:** Per-request routing decisions with counterfactual analysis (what-if top-k candidates)

**Cluster:** 1-64 instances, shared-clock event loop, online routing pipeline with admission/routing latency, pending request tracking

### What makes a good hypothesis

- **Intuitive claim:** "X should be better than Y for workload Z because of mechanism M"
- **Clear experiment:** Two configurations differing in exactly one dimension
- **Measurable outcome:** Specific metric (TTFT p99, throughput, distribution uniformity) that should differ
- **Diagnostic value:** If the hypothesis fails, it points to a specific code path or design assumption worth investigating
- **User value:** The experiment becomes a documented example that helps users understand when to use which configuration

---

# Background

## Repository Overview
BLIS is a Go-based discrete-event simulator (~19K LOC) for LLM inference serving. It models the full serving pipeline: request arrival -> admission control -> routing -> per-instance queueing -> batch formation -> step execution (prefill/decode) -> completion. The simulator supports both blackbox mode (trained alpha/beta coefficients) and roofline mode (analytical FLOPs/bandwidth estimation).

## Technology Stack

> See [`CLAUDE.md`](../../CLAUDE.md) for the full technology stack and build/test commands.

## Architecture Reference

> For the full architecture overview, file organization, and key data flow, see [`CLAUDE.md`](../../CLAUDE.md).

## Existing Patterns and Conventions

- **BDD testing:** `TestType_Scenario_Behavior` naming, behavioral assertions only
- **Antipattern rules:** 20 codified rules — see `docs/standards/rules.md`
- **Determinism invariant:** Same seed -> byte-identical output
- **Conservation invariants:** injected == completed + queued + running; allocated + free == total blocks

## Validated Methodology (PR18 Session)

The prefix-affinity hypothesis testing methodology:
1. **Pose hypothesis:** "Prefix-affinity routing should outperform load-only for prefix-heavy workloads"
2. **Run experiment:** Compare `prefix-affinity:3,queue-depth:2,kv-utilization:2` vs `queue-depth:2,kv-utilization:2`
3. **Observe failure:** Initial results showed no differentiation
4. **Debug:** Found 3 bugs (token sharing, session overlap, short prefix)
5. **Fix and re-validate:** 2.3x TTFT improvement confirmed
6. **Document:** Example YAMLs (`prefix-affinity-demo.yaml`, `multiturn-chat-demo.yaml`)

This pattern -- **hypothesis-driven testing as a debugging and documentation methodology** -- is what we want to replicate across 20 more hypotheses covering the full system.

---

# Idea 1: Hypotheses 1-7 (Routing and Scheduling)

## H1: SJF scheduling should reduce TTFT for mixed-length workloads

**Intuition:** If short requests get stuck behind long ones (head-of-line blocking), scheduling short jobs first should reduce average wait time — the classic SJF result from operating systems.

**Experiment:** Create a `workload-spec` YAML with two clients: "short-batch" (slo_class: batch, input mean 32 tokens, 50% rate) and "long-batch" (slo_class: interactive, input mean 1024 tokens, 50% rate). Use high enough rate (3000+ req/s, 4 instances) to ensure queues build up beyond a single batch. Compare `--scheduler fcfs` vs `--scheduler sjf`. Measure per-SLO-class TTFT.

**Precondition:** Verify queue depth > max_batch_size during the experiment. If queues never exceed one batch, SJF reordering is invisible (both schedulers produce identical batches).

**Predicted outcome:** SJF should produce significantly lower TTFT for short requests (the "batch" SLO class), at the cost of slightly higher E2E for long requests. The magnitude depends on queue depth — larger queues amplify the SJF advantage.

**If hypothesis fails:** Most likely: (b) batch formation absorbs everything regardless of queue order — this would be a design insight that BLIS's greedy batch formation masks scheduling effects. Also check: (a) SJF sorts by wrong field (not input token count), (c) per-SLO-class metric breakdown doesn't work with this YAML.

**Coverage:** Scheduling (SJF vs FCFS), batch formation interaction, per-class metrics

---

## H2: Priority-FCFS with SLO-based priority should reduce realtime TTFT at the cost of batch TTFT

**Intuition:** In a mixed SLO workload (realtime + batch), giving realtime requests higher priority should reduce their waiting time, while batch requests wait longer — a classic priority scheduling tradeoff.

**Experiment:** Use `ScenarioMixedSLO` (33% realtime, 34% interactive, 33% batch). Compare `--priority-policy constant --scheduler fcfs` vs `--priority-policy slo-based --scheduler priority-fcfs`. Measure per-SLO-class TTFT.

**Predicted outcome:** Realtime TTFT p99 should drop 2-3x. Batch TTFT p99 should increase. Overall throughput should remain similar (same total work).

**If hypothesis fails:** May indicate priority inversions (the anomaly counter should be zero with slo-based priority but nonzero with constant), or that the SLO-based priority formula doesn't create enough differentiation between classes. Check `sim/priority.go` SLOBasedPriority.Compute() logic.

**Coverage:** Priority policy, scheduler interaction, per-SLO metrics, anomaly detection

---

## H3: At high request rates, queue-depth scorer should distribute more evenly than kv-utilization scorer

**Intuition:** Queue-depth updates instantly (PendingRequests increments synchronously at routing time), while KV utilization only changes when batch formation allocates blocks (a lagging indicator). At high rates, the stale KV signal should cause pile-on.

**Experiment:** 4 instances, rate=5000, 1000 requests. Compare `--routing-scorers "queue-depth:1"` vs `--routing-scorers "kv-utilization:1"`. Measure distribution std_dev and TTFT p99.

**Predicted outcome:** Queue-depth should produce a significantly more uniform distribution (lower std_dev) and significantly lower TTFT p99 than kv-utilization. The kv-utilization scorer should produce a visibly skewed distribution because the KV signal is stale during the burst of routing decisions between batch formation steps. This is the signal-staleness effect from issue #229.

**If hypothesis fails:** May indicate snapshot caching is more/less aggressive than expected, or that KV allocation frequency changed. Check `sim/cluster/snapshot.go` CachedSnapshotProvider refresh logic.

**Coverage:** Routing scorers, signal freshness, snapshot observability

---

## H4: Round-robin should outperform least-loaded for uniform workloads at low rates

**Intuition:** For perfectly uniform request sizes at low utilization, round-robin distributes optimally with zero overhead. Least-loaded has the same distribution but with routing computation overhead and potential for minor oscillation due to PendingRequests tracking delays.

**Experiment:** Uniform workload (fixed 256 input, 128 output tokens), rate=100, 4 instances. Compare round-robin vs least-loaded. Measure TTFT mean and throughput.

**Predicted outcome:** Nearly identical metrics (within 1%), confirming that least-loaded's intelligence adds no value for uniform low-rate workloads. This validates the baseline.

**If hypothesis fails:** Would indicate either (a) round-robin has a subtle bug in cycle counting, or (b) least-loaded's PendingRequests tracking creates asymmetry even at low rates. Check `RoundRobin.Route()` counter logic and `pendingRequests` bookkeeping.

**Coverage:** Routing baselines, PendingRequests lifecycle, low-utilization behavior

---

## H5: Token-bucket admission control should improve tail latency under burst by shedding excess load

**Intuition:** During traffic bursts (Gamma arrivals with high CV), accepting all requests floods the queues and increases tail latency. A token bucket that rejects excess requests should cap queue depth, trading some rejected requests for much better latency for admitted ones.

**Experiment:** Create a `workload-spec` YAML with Gamma arrivals (`arrival: {process: gamma, cv: 3.5}`, aggregate_rate: 2000). **Important: Gamma arrivals are only available via workload-spec, not CLI `--rate` which produces Poisson.** Alternatively, use `ScenarioBurstyTraffic(seed, 2000)` as a built-in preset. Compare `--admission-policy always-admit` vs `--admission-policy token-bucket --token-bucket-capacity 500 --token-bucket-refill-rate 400`. Measure TTFT p99 and rejected request count (visible in trace summary as "Rejected: N", and in `RejectedRequests()` counter).

**Predicted outcome:** Token-bucket: lower TTFT p99 (burst smoothing), some rejected requests. Always-admit: higher TTFT p99 (queue buildup during bursts), zero rejects. The token-bucket refill rate (400 tokens/s) should be lower than the burst peak rate, causing rejections during bursts but admitting most requests during quiet periods.

**If hypothesis fails:** May indicate the token bucket refill rate is too high relative to the effective burst rate (never rejects) or too low (rejects everything). Could also reveal that the admission decision happens before routing, so rejecting doesn't actually reduce instance queue depth — check event ordering in `cluster_event.go`. Also check: does the Gamma sampler with CV=3.5 actually produce the expected burstiness? (verify in `sim/workload/arrival.go` Marsaglia-Tsang implementation).

**Coverage:** Admission control, bursty arrivals (Gamma process), event pipeline ordering, load shedding

---

## H6: Counterfactual regret should be higher for round-robin than weighted routing under variable load

**Intuition:** Round-robin ignores load entirely, so it should frequently route to suboptimal instances when load is asymmetric. The `weighted` policy with queue-depth scoring actively picks the best instance. The counterfactual analysis should quantify this difference.

**Experiment:** 4 instances, **variable workload** (use a workload-spec with two clients: one with short inputs at 70% rate, one with long inputs at 30% rate — creates load asymmetry). Rate=2000, `--trace-level decisions --counterfactual-k 3`. Compare mean regret for `--routing-policy round-robin` vs `--routing-policy weighted --routing-scorers "queue-depth:1"`.

**Note:** This is primarily an **integration test of the counterfactual machinery** (PR13). The hypothesis validates that the tracing pipeline correctly captures routing decisions, scores alternatives, and computes meaningful regret. The variable workload ensures load asymmetry exists to be detected.

**Predicted outcome:** Round-robin mean regret > 0 (routes to loaded instances when idle ones exist). Weighted/queue-depth mean regret should be significantly lower (actively avoids loaded instances).

**If hypothesis fails:** (a) If regret is zero for both: the workload doesn't create enough load asymmetry at the rate tested — increase rate or variability. (b) If regret is nonzero for both: check `sim/cluster/counterfactual.go` scoring formula — it may use a metric that weighted routing doesn't optimize for (e.g., KV utilization instead of effective load). (c) If regret is high for weighted: the counterfactual is scoring by a different criterion than what the scorer optimizes — a real discrepancy worth investigating.

**Coverage:** Decision tracing, counterfactual analysis, routing optimality, variable workload

---

## H7: Increasing instances from 4 to 8 should roughly halve TTFT p99 for saturated workloads

**Intuition:** If the workload saturates 4 instances (long queues, high utilization), adding 4 more should absorb the excess load — requests wait in shorter queues, reducing TTFT.

**Precondition:** The workload must saturate 4 instances. Verify by checking that TTFT p99 with 4 instances is significantly worse than TTFT at low rate (indicating queuing pressure). Use a high rate (5000+ req/s) with 2000 requests.

**Experiment:** Compare `--num-instances 4` vs `--num-instances 8` with round-robin routing at rate=5000. Measure TTFT p99 (primary metric — directly reflects queuing delay) and `responses_per_sec` (throughput). **Note:** `responses_per_sec` = completed_requests / (sim_ended_time / 1e6), so it reflects simulated service capacity, not wall-clock speed.

**Predicted outcome:** 8 instances should have significantly lower TTFT p99 (roughly halved — queues drain faster with more capacity). Throughput (`responses_per_sec`) should increase, approaching 2x if the workload saturates 4 instances.

**If hypothesis fails:** (a) If TTFT p99 is similar with 4 and 8: the workload doesn't saturate 4 instances (increase rate). (b) If throughput doesn't increase: workload generation may cap at the same total requests regardless of instances (check `numRequests` limit). (c) If throughput increases but TTFT doesn't improve: each instance gets `--total-kv-blocks` independently, so KV capacity scales automatically — but if blocks are undersized per-instance, preemptions may limit the benefit.

**Coverage:** Horizontal scaling, saturation behavior, throughput vs latency under capacity

## Reviews for Idea 1

### Reviewer Consensus (Claude, GPT-4o, Gemini)

**Highest diagnostic value:** H3 (scorer signal freshness) > H1 (SJF vs batch formation) > H5 (token-bucket event ordering).

**Key refinements needed:** (1) H1/H5 need workload-spec YAMLs, not CLI flags. (2) Soften numeric predictions to relational claims. (3) H6 needs high utilization to differentiate regret. (4) H7 needs throughput metric clarification.

**Major coverage gaps identified by all 3 reviewers:** KV cache behavior, tiered GPU+CPU offload, determinism/conservation invariants, batch formation mechanics, fitness evaluation, pathological templates, multi-scorer interactions. These are addressed in Idea 2 and Idea 3 below.

---

# Idea 2: Hypotheses 8-14 (KV Cache, Tiered Storage, Batch Formation)

*Addresses reviewer feedback: KV cache, tiered storage, batch formation, and conservation invariants were absent from Idea 1.*

## H8: Reducing total KV blocks should increase preemption frequency and worsen tail latency

**Intuition:** KV blocks are the memory currency — each running request holds blocks proportional to its token count. With fewer blocks, the cache fills up faster, forcing preemptions (evictions of running requests to make room). Preempted requests restart from scratch, increasing tail latency.

**Experiment:** Fixed workload (rate=500, 200 requests, 4 instances, `--block-size-in-tokens 16`). Compare 3 configurations: `--total-kv-blocks 2000` (abundant), `--total-kv-blocks 500` (constrained), `--total-kv-blocks 100` (severely constrained). Measure preemption count (`PreemptionCount` in JSON output), TTFT p99, and E2E p99.

**Precondition:** Verify that `PreemptionCount` appears in the CLI JSON output. Also verify admission is `always-admit` (not token-bucket) to ensure rejections don't mask preemptions.

**Predicted outcome:** With fewer blocks, preemption count should monotonically increase and TTFT p99 should worsen. The 100-block config may be degenerate (constant preemption); the 500-block config should show the interesting transition.

**If hypothesis fails:** May indicate (a) preemptions aren't happening despite low blocks (check `makeRunningBatch` allocation failure path in `simulator.go`), (b) preempted requests aren't properly re-queued (check `preempt()` in `simulator.go`), (c) preemption count isn't surfaced in CLI output (check `PreemptionCount` in `sim/metrics.go` and `SaveResults`), or (d) all requests are completing before the next arrives (rate too low — increase to 1000+).

**Coverage:** KV cache pressure, preemption mechanics, resource exhaustion

---

## H9: Prefix caching should reduce TTFT proportional to prefix length

**Intuition:** When two requests share a prefix, the second request can skip prefilling the cached tokens — its `numNewTokens` is reduced by `cachedBlocks × blockSize`. Longer shared prefixes = more tokens skipped = lower TTFT.

**Experiment:** Create 4 variants of `prefix-affinity-demo.yaml` with varying `prefix_length` (64, 128, 256, 512 tokens). **Hold total input length constant** at 768 tokens (so new tokens = 704, 640, 512, 256 respectively — this isolates the caching effect from the input-length effect). All requests share the same prefix group. Measure cluster TTFT mean with `--routing-scorers "prefix-affinity:5,queue-depth:1"`.

**Predicted outcome:** TTFT should decrease monotonically as prefix_length increases — longer shared prefixes mean more cached blocks reused, fewer tokens to prefill. The 512-token configuration should produce noticeably lower TTFT than the 64-token configuration. **Metric source:** Cluster-level `ttft_mean_ms` in the JSON output; `PreemptionCount` in metrics (should not increase as prefix grows).

**Precondition:** Ensure `total_kv_blocks × block_size_in_tokens > prefix_length × num_instances` so cached prefixes fit in memory. If prefix_length exceeds per-instance cache capacity, LRU eviction will destroy cached blocks between requests, inverting the expected monotonic decrease. (This is itself an interesting finding — document the inflection point if observed.)

**If hypothesis fails:** May indicate (a) `GetCachedBlocks()` isn't finding cached blocks despite shared prefix tokens (check hash matching in `kvcache.go:126-141`), (b) block eviction is clearing cached blocks between requests (LRU capacity too small — see precondition), or (c) the latency model (beta coefficients) doesn't weight prefill tokens enough for the savings to be measurable. This hypothesis is a direct test of the cache-hit-reduces-prefill mechanism in `batch_formation.go` (formerly `simulator.go`, extracted in #371).

**Coverage:** Prefix caching effectiveness on latency, GetCachedBlocks correctness, cache eviction pressure

---

## H10: Tiered KV cache (GPU+CPU offload) should reduce preemptions compared to single-tier at the cost of transfer latency

**Intuition:** When GPU KV blocks are exhausted, the single-tier cache preempts requests. With a CPU tier, blocks can be offloaded to CPU instead of being evicted entirely. When the request resumes, blocks are reloaded from CPU (faster than recomputing from scratch). The tradeoff: reload incurs transfer latency, but avoids full recomputation.

**Prerequisite:** Verify CLI flags exist: `--kv-cpu-blocks`, `--kv-offload-threshold`, `--kv-transfer-bandwidth`, `--kv-transfer-base-latency`. Run `./simulation_worker run --help | grep kv-cpu` to confirm. Also verify that `kvcache_tiered.go` exists and is compiled.

**Experiment:** Constrained GPU blocks (200, `--block-size-in-tokens 16`), rate=500, 200 requests, 4 instances. Compare single-tier (`--kv-cpu-blocks 0`) vs tiered (`--kv-cpu-blocks 500 --kv-offload-threshold 0.8 --kv-transfer-bandwidth 100 --kv-transfer-base-latency 10`). Transfer bandwidth is in blocks/tick (see `kvcache_tiered.go`). Measure preemption count, TTFT p99, E2E p99.

**Smoke test:** Before the full comparison, run the tiered configuration once and verify that offload events are nonzero (check logrus debug output for "offload" messages).

**Predicted outcome:** Tiered: fewer preemptions (blocks offloaded to CPU instead of evicted), comparable or slightly worse TTFT (transfer latency adds to step time). Single-tier: more preemptions, worse E2E (preempted requests recompute from scratch).

**If hypothesis fails:** May indicate (a) offload isn't triggering — threshold 0.8 means offload starts at 80% GPU utilization; with 200 blocks, that's 160 used blocks (check if workload reaches this), (b) transfer latency math uses wrong units (check `math.Ceil` in `kvcache_tiered.go`), (c) reload doesn't restore blocks (check `TieredKVCache` reload path), or (d) `ConsumePendingTransferLatency` isn't called during step execution.

**Coverage:** Tiered KV cache, offload/reload mechanics, transfer latency modeling

---

## H11: Batch formation with larger token budgets should improve throughput but worsen ITL (inter-token latency)

**Intuition:** Larger batches process more requests per step (higher throughput) but each step takes longer (higher per-token latency). This is the fundamental batching tradeoff: throughput vs latency.

**Experiment:** Fixed workload (rate=1000, 500 requests). Compare `--max-scheduled-tokens 1024` vs `--max-scheduled-tokens 8192`. Measure throughput (req/s), ITL mean, and TTFT p99.

**Predicted outcome:** Larger budget: higher throughput (more requests per batch), higher ITL mean (longer steps). Smaller budget: lower throughput (fewer per batch), lower ITL (shorter steps).

**If hypothesis fails:** May indicate (a) batch formation doesn't actually use the token budget (check `makeRunningBatch` budget logic), (b) step time estimation via beta coefficients doesn't scale with batch size as expected, or (c) the ITL metric doesn't capture per-step latency correctly.

**Coverage:** Batch formation, token budget mechanics, throughput-latency tradeoff

---

## H12: Request conservation invariant should hold across all policy configurations

**Intuition:** No matter what routing, scheduling, or admission policy is used, every injected request must end up completed, queued, or running at simulation end. `injected == completed + queued + running`. This is a fundamental correctness property.

**Experiment:** Run 10 different configurations (varying routing, scheduler, admission, KV blocks, instances) with the same workload. For each, verify `injected == completed + still_queued + still_running`.

**Predicted outcome:** Conservation holds for all configurations. Zero violations.

**If hypothesis fails:** This is a critical correctness bug. The failure would point to: (a) requests silently dropped in the admission-routing pipeline, (b) preemption not properly re-queuing requests, (c) KV allocation failure path losing requests, or (d) request count mismatch between cluster and instance metrics. Every failure is a real bug worth fixing (see issue #183 where a golden test perpetuated a dropped request for months).

**Coverage:** Fundamental invariant, correctness across all configurations, regression detection

---

## H13: Determinism invariant should hold: same seed produces identical output

**Intuition:** BLIS uses PartitionedRNG with a seed for deterministic simulation. Running the same configuration with the same seed twice should produce byte-identical output. This is critical for reproducible research.

**Experiment:** Run 5 different configurations twice each (same seed). At least one configuration MUST use `--routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:2"` with enough requests and prefix diversity to trigger LRU eviction in the PrefixCacheIndex (the most likely determinism violation source — `evictOldest` iterates a map). Compare TTFT mean, distribution, and per-request metrics. They must be identical.

**Predicted outcome:** Byte-identical output for all runs with the same seed.

**If hypothesis fails:** Non-determinism bug — likely from: (a) unguarded map iteration feeding output ordering, (b) floating-point accumulation order dependency, (c) time-dependent randomness (using wall clock instead of sim clock), or (d) stateful scorer (prefix-affinity) with non-deterministic internal state. The prefix-affinity scorer's LRU uses map iteration in `evictOldest` — the monotonic clock invariant must hold.

**Coverage:** Determinism, reproducibility, non-deterministic map iteration detection

---

## H14: Pathological templates should produce measurably worse behavior than their normal counterparts

**Intuition:** The pathological policies exist specifically to test anomaly detection. `always-busiest` should produce HOL blocking (routes to most loaded instance). `reverse-priority` should produce priority inversions. If anomaly counters don't detect these, the detection logic has a bug.

**Prerequisite:** Verify pathological policy names exist in CLI: `always-busiest`, `reverse-priority`, `inverted-slo`. Run `./simulation_worker run --help | grep -E "always-busiest|reverse-priority|inverted-slo"` or check `ValidRoutingPolicyNames()`, `ValidSchedulerNames()`, `ValidPriorityPolicyNames()`. **This hypothesis is a prerequisite for H24** — if individual pathological templates don't work, the combined test (H24) is meaningless.

**Experiment:** Mixed SLO workload with 4 instances. Compare normal configuration (`least-loaded` + `priority-fcfs` + `slo-based`) vs pathological (`always-busiest` + `priority-fcfs` + `inverted-slo`). Measure HOL blocking events, priority inversions, TTFT p99, distribution std_dev.

**Predicted outcome:** Pathological: high HOL blocking count, nonzero priority inversions, extreme load imbalance, much worse TTFT p99. Normal: zero anomalies, balanced distribution, good TTFT.

**If hypothesis fails:** May indicate (a) anomaly detection logic doesn't correctly identify HOL blocking or priority inversions (check `sim/cluster/metrics.go` detection code), (b) pathological templates don't actually produce the expected behavior (check `AlwaysBusiest.Route()` and `ReversePriority.OrderQueue()`), or (c) the `always-busiest` + `inverted-slo` combination creates unexpected interactions. Note: do NOT combine `inverted-slo` with `reverse-priority` -- the two inversions cancel out (see H14 FINDINGS.md BUG 3).

**Coverage:** Anomaly detection, pathological templates, HOL blocking, priority inversions

---

# Idea 3: Hypotheses 15-20 (Multi-Scorer Interactions, Fitness, Workload Patterns)

*Addresses reviewer feedback: multi-scorer interactions, fitness evaluation, and workload pattern diversity were absent.*

## H15: Fitness evaluation should rank prefix-affinity-aware routing higher than load-only for prefix-heavy workloads

**Intuition:** The fitness function is a weighted combination of metrics (throughput, TTFT, E2E). For prefix-heavy workloads, prefix-affinity routing produces better TTFT. Therefore, fitness scores weighted toward TTFT should rank prefix-affinity configurations higher.

**Experiment:** Run both configurations with `--fitness-weights "throughput:0.3,p99_ttft:0.5,mean_e2e:0.2"` on `prefix-affinity-demo.yaml`. Compare fitness scores.

**Predicted outcome:** Prefix-affinity config has higher fitness score (better TTFT dominates the weight).

**If hypothesis fails:** May indicate (a) fitness score computation normalizes metrics in unexpected ways (check `ComputeFitness` in `sim/cluster/metrics.go`), (b) the normalization formula (`1/(1+value)`) compresses differences too much, or (c) throughput differences counteract TTFT improvements. Could surface the PR9 fitness normalization bug (500,000x scale imbalance).

**Coverage:** Fitness evaluation, multi-objective scoring, configuration ranking

---

## H16: Bursty (Gamma) arrivals should produce worse tail latency than Poisson at the same average rate

**Intuition:** Gamma arrivals with high CV produce bursts — many requests arrive together, then quiet periods. Bursts create temporary queue buildups that Poisson (uniform random) doesn't. The queue buildup during bursts should increase TTFT p99.

**Experiment:** Same average rate (1000 req/s), 4 instances, 500 requests. Compare Poisson arrivals (`process: poisson`) vs Gamma (`process: gamma, cv: 3.5`). Use `--workload-spec` for both. Measure TTFT p99 and scheduling delay p99.

**Predicted outcome:** Gamma: significantly higher TTFT p99 (burst-induced queue buildup). Poisson: lower p99 (smoother arrivals).

**If hypothesis fails:** May indicate (a) the Gamma arrival sampler doesn't produce the expected burstiness (check `SampleIAT` in `sim/workload/arrival.go` for the Marsaglia-Tsang implementation), (b) the CV parameter doesn't translate to sufficient burstiness, or (c) at this rate the system has enough headroom to absorb bursts without queue buildup.

**Coverage:** Arrival process diversity, Gamma/Poisson comparison, bursty workload behavior

---

## H17: Multi-scorer weights should produce a Pareto frontier: no single configuration dominates all metrics

**Intuition:** Different scorer weight combinations optimize for different objectives. `prefix-affinity:5,queue-depth:1` maximizes cache locality (good TTFT for prefix-sharing requests). `queue-depth:5,kv-utilization:1` maximizes load balance (good tail latency). No single weight combination should be best on ALL metrics simultaneously.

**Experiment:** Sweep 5 weight configurations on `multiturn-chat-demo.yaml`. Measure TTFT mean, TTFT p99, E2E mean, throughput. Plot the Pareto frontier.

**Predicted outcome:** A clear Pareto frontier — configurations trading off TTFT (prefix-affinity helps) vs throughput (load-balance helps). The default llm-d profile should sit near the knee of the curve.

**If hypothesis fails:** If one configuration dominates all others on ALL metrics, it means the tradeoff doesn't exist — the scoring dimensions are redundant (echoing issue #229's finding about the old cache/load dimensions). This would question whether the composable scorer framework adds value beyond a single well-chosen scorer.

**Coverage:** Multi-scorer interactions, Pareto optimality, weight sensitivity

---

## H18: Unfair tenant workload with SLO-based priority should achieve Jain fairness index > 0.9

**Intuition:** `ScenarioUnfairTenants` has 90% low-priority bulk traffic and 10% high-priority realtime. Without priority scheduling, the realtime class gets drowned out. With SLO-based priority, realtime requests should get preferential treatment, improving cross-class fairness.

**Experiment:** Use `ScenarioUnfairTenants` at high rate. Compare `--priority-policy constant` vs `--priority-policy slo-based --scheduler priority-fcfs`. Measure Jain fairness index across SLO classes and per-class TTFT.

**Predicted outcome:** SLO-based: Jain index > 0.9 (good fairness). Constant: Jain index < 0.5 (realtime drowns in bulk traffic).

**If hypothesis fails:** First, verify what `JainFairnessIndex` in `sim/cluster/metrics.go` actually measures — is it throughput fairness (completion rates per class), latency fairness (TTFT per class), or SLO attainment? The threshold (0.9) assumes throughput fairness. Also check: (a) SLO-based priority doesn't create enough differentiation for 90/10 split, (b) per-SLO-class metrics aren't computed correctly for this scenario.

**Coverage:** Fairness metrics, SLO differentiation, unfair workload patterns

---

## H19: Roofline mode should produce different absolute latencies but same relative policy rankings as blackbox mode

**Intuition:** Roofline mode uses analytical FLOPs/bandwidth estimation; blackbox mode uses trained alpha/beta coefficients. The absolute numbers will differ, but the relative performance of routing/scheduling policies should be the same — a policy that is better under blackbox should also be better under roofline.

**Experiment:** Run 3 routing configurations (round-robin, least-loaded, weighted) under both modes for the same workload. Compare TTFT rankings.

**Predicted outcome:** Same policy ranking in both modes (e.g., if weighted > least-loaded > round-robin in blackbox mode, same ordering in roofline mode). Absolute TTFT values will differ.

**Precondition:** Verify whether roofline mode's step time computation accounts for `cachedBlocks` (i.e., do cache hits reduce roofline step time?). Check `roofline_step.go` for use of `numNewTokens` vs total tokens. If roofline doesn't model cache hit latency reduction, use non-prefix workloads for the cross-mode comparison to avoid confounding.

**If hypothesis fails:** May indicate (a) roofline step time estimation has a fundamentally different sensitivity to batch features than blackbox, (b) roofline mode has a bug in prefill/decode step computation (check `roofline_step.go`), or (c) the policy advantage depends on specific latency model characteristics (e.g., prefix caching benefit disappears in roofline mode because it doesn't model cache hit latency reduction — this is a known limitation, not a bug).

**Coverage:** Latency model comparison, model abstraction validity, roofline vs blackbox

---

## H20: Heavy-tailed input distributions (ParetoLogNormal) should produce more preemptions and HOL blocking than Gaussian

**Intuition:** ParetoLogNormal distributions have heavy tails — occasional very long requests. These long requests hold KV blocks for extended periods, starving short requests. Gaussian distributions have bounded variance, so extreme outliers are rare.

**Experiment:** Same average input length (256 tokens), 4 instances, 500 requests, rate=1000. Compare Gaussian (std_dev=50, min=32, max=512) vs ParetoLogNormal (alpha=1.5, xm=50, mu=5.5, sigma=1.2). Measure preemption count, HOL blocking events, TTFT p99, and distribution of per-request latency.

**Predicted outcome:** ParetoLogNormal: more preemptions (long requests exhaust KV blocks), higher TTFT p99 (short requests stuck behind long ones), higher HOL blocking count. Gaussian: fewer preemptions, lower p99, minimal HOL blocking.

**If hypothesis fails:** May indicate (a) ParetoLogNormal sampler doesn't produce the expected heavy tail (check `sim/workload/distribution.go` implementation), (b) the KV cache is large enough to absorb even very long requests without pressure, or (c) HOL blocking detection has a threshold issue (blocks short requests but doesn't count as HOL).

**Coverage:** Heavy-tailed distributions, resource exhaustion patterns, preemption under tail load

---

# Idea 4: Edge Cases and Robustness (Round 2 Feedback)

*Addresses Round 2 reviewer feedback: missing edge cases for extreme weights, resource exhaustion, underload, and pathological interactions.*

## H21: Extreme scorer weight (weight=100:1) should behave identically to single-scorer routing

**Intuition:** If prefix-affinity weight is 100 and queue-depth weight is 1, the queue-depth contribution is negligible (1/101 ≈ 1%). The behavior should be indistinguishable from `prefix-affinity:1` alone. This tests that the scorer aggregation math handles extreme weight ratios correctly (no overflow, no precision loss).

**Experiment:** Compare `--routing-scorers "prefix-affinity:100,queue-depth:1"` vs `--routing-scorers "prefix-affinity:1"` on `prefix-affinity-demo.yaml`. Measure TTFT and target distribution. They should be nearly identical.

**Predicted outcome:** Distribution and TTFT should match within measurement noise. **Metric source:** Compare target distributions from trace summary; compare `ttft_mean_ms` cluster values.

**If hypothesis fails:** May indicate (a) weight normalization has precision issues at extreme ratios (check `normalizeScorerWeights` in `routing_scorers.go`), (b) the queue-depth scorer produces NaN/Inf at some edge case that gets amplified by the small weight, or (c) the argmax tie-breaking behaves differently when scores have vastly different magnitudes.

**Coverage:** Scorer weight normalization, numeric precision, edge case robustness

---

## H22: Running with zero KV blocks should panic at CLI validation, not deep inside the simulation

**Intuition:** `--total-kv-blocks 0` is an invalid configuration. BLIS should catch it at the CLI boundary with a clear error message, not let it reach `kvcache.go` where it would cause a cryptic division-by-zero or empty-allocation panic. This tests the input validation chain.

**Experiment:** Run `./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --total-kv-blocks 0` and capture stderr. It should produce a `logrus.Fatalf` message from `cmd/root.go`, not a panic stack trace from `sim/`.

**Predicted outcome:** Clean error: `"--total-kv-blocks must be > 0, got 0"` from CLI boundary. **Metric source:** Exit code 1 + logrus error on stderr, no panic.

**If hypothesis fails:** A panic from `sim/kvcache.go` or `sim/simulator.go` means the CLI validation is missing or incomplete. Check `cmd/root.go` for the `totalKVBlocks` validation. Also test `--block-size-in-tokens 0` — same principle.

**Coverage:** Input validation chain, error handling boundaries (CLI vs library), antipattern rule 6

---

## H23: Under very low load (1 req/s, 4 instances), all routing policies should produce equivalent TTFT

**Intuition:** When the system is barely utilized, queues never build up. All instances are always idle. Every routing policy effectively picks an idle instance, so TTFT should be determined purely by the latency model (prefill time), not by queue waiting. This is a baseline sanity check.

**Experiment:** Rate=1, 50 requests, 4 instances. Run with round-robin, least-loaded, weighted (default), and prefix-affinity. Measure TTFT mean for each.

**Predicted outcome:** All four produce TTFT within 5% of each other. The TTFT should be close to the bare prefill time (determined by alpha/beta coefficients × input tokens).

**If hypothesis fails:** A policy showing significantly different TTFT at near-zero load indicates a bug in that policy's routing or snapshot logic — it's adding latency that shouldn't exist. Check whether PendingRequests is being incorrectly accumulated, or whether snapshot staleness creates phantom load differentials even when all instances are idle.

**Coverage:** Baseline behavior, low-utilization correctness, policy overhead measurement

---

## H24: Combining always-busiest routing with inverted-slo scheduling should produce maximum measurable anomalies

**Intuition:** `always-busiest` routes every request to the most loaded instance (maximizing HOL blocking). `inverted-slo` gives newer requests higher priority when used with `priority-fcfs`, starving older ones (maximizing priority inversions). Combining them should trigger both anomaly detectors simultaneously — the most pathological configuration possible. Warning: do NOT pair `inverted-slo` with `reverse-priority` -- the double inversion cancels out (see issue #295).

**Experiment:** Use `ScenarioMixedSLO` (33% realtime + 34% interactive + 33% batch), rate=2000, 4 instances. Compare: (A) normal (`least-loaded` + `priority-fcfs` + `slo-based`), (B) pathological (`always-busiest` + `priority-fcfs` + `inverted-slo`). Measure HOL blocking count, priority inversion count, TTFT p99, and distribution std_dev.

**Predicted outcome:** Pathological: HOL blocking count > 0, priority inversions > 0, distribution std_dev > 100 (severe imbalance), TTFT p99 dramatically worse. Normal: zero anomalies, std_dev < 5, good TTFT. **Metric source:** `RawMetrics.HOLBlockingEvents`, `RawMetrics.PriorityInversions` in `sim/cluster/metrics.go`.

**If hypothesis fails:** (a) If HOL blocking is zero: the `always-busiest` policy isn't producing load imbalance — check `AlwaysBusiest.Route()`. (b) If priority inversions are zero: verify `inverted-slo` is paired with `priority-fcfs` (not `reverse-priority`, which cancels the inversion — see #295) and check `detectPriorityInversions` in `sim/cluster/metrics.go`. (c) If both anomalies are zero: the anomaly detection thresholds may be too lenient for this configuration.

**Coverage:** Anomaly detection completeness, pathological template interaction, combined worst-case

---

## H25: Integration stress test — the full policy stack should maintain conservation invariants under combined load

**Intuition:** Individual policies are tested in isolation, but real deployments combine multiple dimensions. The system should remain correct (no dropped requests, no KV leaks, deterministic output) when ALL features are active simultaneously.

**Experiment:** Run with the most complex configuration possible: `--routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2" --admission-policy token-bucket --token-bucket-capacity 500 --token-bucket-refill-rate 300 --priority-policy slo-based --scheduler priority-fcfs --kv-cpu-blocks 200 --kv-offload-threshold 0.8 --kv-transfer-bandwidth 50 --trace-level decisions --counterfactual-k 3 --summarize-trace` with `multiturn-chat-demo.yaml` at high rate (2000 req/s). Verify: (a) conservation holds (completed + queued + running + rejected == injected), (b) output is deterministic (run twice with same seed), (c) no panics.

**Predicted outcome:** Conservation holds, deterministic output, no crashes. Some requests rejected by token-bucket, some preemptions from tiered KV, some priority inversions suppressed by slo-based priority.

**If hypothesis fails:** A conservation violation under combined load would indicate an interaction bug — e.g., token-bucket rejections not counted, preempted requests lost in the tiered KV handoff, or the trace recorder miscounting decisions. This is the highest-value integration test because it exercises ALL code paths simultaneously.

**Coverage:** Full system integration, cross-cutting policy interactions, conservation under stress

---

## H26: Adding admission latency should delay E2E by exactly that amount under low load

**Intuition:** The cluster event pipeline processes admission before routing. With `--admission-latency 10000` (10ms), every request should wait 10ms in the admission pipeline before routing. Under low load (no queuing), E2E latency should increase by exactly the admission latency.

**Precondition:** Verify TTFT measurement point — is TTFT measured from request arrival or from queue entry? Check `sim/metrics.go` for `FirstTokenTime - ArrivalTime`. If TTFT includes the admission pipeline delay, measure TTFT. If TTFT is measured from queue entry, use E2E instead (which should always include the full pipeline).

**Experiment:** Rate=10, 50 requests, 4 instances. Compare `--admission-latency 0` vs `--admission-latency 10000`. Measure both TTFT and E2E mean. At least one of them should show the 10ms delta.

**Predicted outcome:** The 10ms admission latency configuration should have E2E mean approximately 10ms higher than the zero-latency baseline. If TTFT includes pipeline delay, TTFT should also show the delta. This directly validates the event pipeline's causal ordering: Arrival → Admission (+ latency) → Routing → Queue → Batch → Step.

**If hypothesis fails:** (a) If neither TTFT nor E2E shows the delta: events are being processed out of order, or the admission latency is not correctly added to the event timestamp. Check `ClusterArrivalEvent.Execute` and `AdmissionDecisionEvent` timestamp computation in `cluster_event.go`. (b) If only E2E shows the delta but not TTFT: TTFT is measured from queue entry, not arrival — this is not a bug, just a measurement definition to document.

**Coverage:** Event pipeline causal ordering, admission/routing latency modeling, DES event timing, metric measurement points

---

# Executive Summary

## Ideas Overview

**Idea 1 (H1-H7):** Routing and scheduling hypotheses. Strongest: H3 (signal freshness), H1 (SJF vs batch formation), H5 (admission load shedding). These probe core routing/scheduling interactions with high diagnostic value.

**Idea 2 (H8-H14):** KV cache, tiered storage, and invariant testing. Addresses the biggest gaps from Idea 1 reviews. H12 (conservation) and H13 (determinism) are the most critical — they validate fundamental correctness properties. H9 (prefix caching effectiveness) directly tests the KV cache hit → reduced prefill mechanism.

**Idea 3 (H15-H20):** Multi-scorer interactions, fitness evaluation, and workload patterns. H17 (Pareto frontier) tests whether the composable scorer framework adds real value. H19 (roofline vs blackbox) validates latency model abstraction.

## Coverage Map (by Hypothesis Family)

Hypotheses are organized by **family** (what domain is tested) and **tier** (priority). See `docs/standards/experiments.md` for family definitions.

| Family | Hypotheses | Status | Gaps |
|--------|-----------|--------|------|
| **Scheduler invariants** | H12 ✅, H13 ✅, H-Liveness ✅, H25 ✅ | **4/4 done** | None — all planned hypotheses complete |
| **Structural model** | H3 ✅, H9 ✅, H10 ✅, H-Phase ✅, H-MMK ✅, H26 ✅, H-Step-Quantum ✅, H19 ✅ | **8/8 done** | None — H19 confirmed mean rankings preserved across latency modes; P99 diverges from alpha overhead |
| **Performance-regime** | H7 ✅, H8 ✅, H11 ✅, H-Reasoning-KV ✅ | **4/4 done** | None — H7 confirmed super-linear scaling (7.4x TTFT improvement 4→8 instances) |
| **Robustness/failure-mode** | H5 ✅, H14 ✅, H-Overload ✅, H-Overload-KV ✅, H21 ✅, H22 ✅, H24 ✅ | **7/7 done** | None — H21 refuted (cold-start cascade); H24 confirmed (4.9x TTFT, super-additive interaction) |
| **Workload/arrival** | H-Arrival ✅, H16 ✅, H20 ✅ | **3/3 done** | None — H20 refuted: ParetoLN produces fewer preemptions; distribution median drives KV pressure |
| **Cross-policy comparative** | Prefix-Affinity ✅, H1-SJF ✅, H2 ✅, H4 ✅, H6 ✅, H15 ✅, H17 ✅, H18, H23 ✅ | **8/9 done** | H18 (fairness — blocked by #348) |

### Legacy Coverage Map (by area)

| Area | Hypotheses | Components Tested |
|------|-----------|-------------------|
| **Routing** | H3 ✅, H4 ✅, H6 ✅ | Scorer freshness, round-robin (LL tie-breaking bias), counterfactual (structurally zero weighted regret) |
| **Scheduling** | H1 ✅, H2 ✅ | SJF, Priority-FCFS, batch interaction |
| **Admission** | H5 ✅ | Token-bucket, burst smoothing |
| **KV Cache** | H8 ✅, H9 ✅ | Preemption, prefix caching effectiveness |
| **Tiered Storage** | H10 ✅ | GPU+CPU offload/reload, transfer latency |
| **Batch Formation** | H11 ✅ | Token budget, throughput-latency tradeoff |
| **Invariants** | H12 ✅, H13 ✅ | Conservation, determinism |
| **Anomaly Detection** | H14 ✅ | Pathological templates, HOL blocking |
| **Multi-Scorer** | H17 ✅ | Cross-workload dominance, weight sensitivity, kv-heavy micro-bursting |
| **Fitness** | H15 ✅ | Multi-objective scoring (1/(1+x) normalization compresses raw 14-38% improvement to <10% fitness score diff) |
| **Workload Patterns** | H16 ✅, H18, H20 ✅ | Bursty (Gamma effect is load-duration dependent), unfair, heavy-tailed (median drives KV pressure, not tail) |
| **Latency Model** | H19 ✅, H-Step-Quantum ✅ | Roofline vs blackbox (mean rankings preserved, P99 diverges from alpha); alpha/beta service time split (#329) |
| **Scaling** | H7 ✅ | Instance count throughput (7.4x super-linear TTFT scaling 4→8; queue growth rate is non-linear in k) |
| **Prefix Affinity** | H9 ✅, H15 ✅, H17 ✅ | Cache hits, routing, weight tuning, fitness evaluation |
| **Edge Cases** | H21 ✅, H22 ✅, H23 ✅, H24 ✅ | Extreme weights (refuted — cold-start cascade), input validation, underload (all policies equivalent; uniform workload kills differentiation), pathological combos (4.9x TTFT) |
| **Integration** | H25 ✅, H26 ✅ | Full policy stack, event pipeline causal ordering |

## Reviewer Consensus

All 3 reviewers (Claude, GPT-4o, Gemini) agreed on:
1. **H3 has highest diagnostic value** — signal freshness probes deep architectural concerns
2. **KV cache and tiered storage were the biggest gaps** in Idea 1 — now addressed by H8-H10
3. **Invariant testing (H12, H13) should run first** as baseline sanity checks
4. **Numeric predictions should be relational** ("A should be significantly lower than B") not absolute ("30-50% improvement")
5. **Workload specs (YAMLs) needed** for experiments, not just CLI flags

## Statistical Rigor (Round 4 Feedback)

Each hypothesis experiment should follow these conventions:
- **Seeds:** Run each comparison with at least 3 seeds (e.g., 42, 123, 456) to distinguish real effects from seed-dependent noise
- **Effect size:** "Significantly better" means >20% improvement consistent across all seeds. <10% in any seed = inconclusive
- **Equivalence tests** (H4, H23): "Equivalent" means within 5% across all seeds
- **Pass/fail:** A hypothesis passes if the predicted directional outcome holds across all seeds. A hypothesis fails if any seed contradicts the direction

### Family-specific rigor

- **Scheduler invariants (family 2):** Single seed sufficient — determinism is the point. Pass/fail is exact.
- **Cross-policy comparative (family 6):** 3+ seeds mandatory. Must control confounding variables (ED-1, ED-6).
- **Performance-regime (family 3):** Sweep ≥3 values of the independent variable, not just pairwise comparison.
- **Workload/arrival (family 1):** Statistical tests on generated distributions. Long runs for accurate rate estimation.
- **Structural model (family 4):** Code-level verification (RCV-1, RCV-4) essential — these test implementation assumptions.
- **Robustness (family 5):** Must test BOTH defined behavior AND verify no undefined states (deadlock, panic, data loss).

## Cross-Validation Opportunities

The following analytical models can validate the DES under matching assumptions (see VV&UQ framework in `docs/standards/experiments.md`):

| Analytical model | DES configuration to match | What it validates |
|---|---|---|
| **M/M/k** (k servers, Poisson arrivals, exponential service) | Poisson rate, exponential input/output, k instances, FCFS | Queue length distribution, mean wait time, utilization |
| **Little's Law** (L = λW) | Any stable configuration | Fundamental consistency: avg queue length = arrival rate × avg wait time |
| **Phase structure** (prefill ∝ input, decode ∝ output) | Constant distributions, single instance | Linear relationship between token counts and latency components |

These cross-validations should be run as **Validation** category experiments per the VV&UQ framework. Divergence from analytical predictions indicates either a modeling error in the DES or a violated assumption.

## Tier 0: Measurement Audit (Run First)

Before running any hypothesis, verify which metrics are available in CLI JSON output:
1. Enumerate all metrics in the JSON output schema (run one experiment, inspect output)
2. Map each hypothesis to its required metrics (e.g., H8 needs `PreemptionCount`, H12 needs per-request counts)
3. Identify gaps where a hypothesis needs a metric not currently surfaced
4. Implement missing metric surfacing before running the hypothesis

This prevents the scenario where an experiment runs but the measurement doesn't exist.

## Recommendation

**Priority order for implementation:**

1. ~~**Tier 1 (run first — correctness baselines):** H12 (conservation), H13 (determinism)~~ **DONE**
2. ~~**Tier 2 (high diagnostic value):** H3 (signal freshness), H9 (prefix caching), H14 (pathological)~~ **DONE**
3. ~~**Tier 3 (system understanding):** H1 (SJF), H5 (token-bucket), H10 (tiered KV), H11 (batch formation)~~ **DONE**
4. ~~**Tier 4 (research questions):** H17 (Pareto frontier), H19 (roofline vs blackbox), H15 (fitness)~~ **DONE**
5. ~~**Tier 5 (workload diversity):** H2, H16, H20~~ **DONE**. H18 remaining (blocked by #348).

**Additional completed (not in original tiers):**
- H-Liveness, H-Overload, H-Overload-KV, H-MMK, H-Phase, H-Arrival (methodology validation experiments)
- H22 (zero blocks CLI validation, #343)
- H25 (integration stress test — full policy stack conservation)
- H26 (admission latency causal ordering)
- H-Step-Quantum (#329, refuted — discovered alpha/beta service time split)
- H16 (Gamma vs Poisson — confirmed with nuance: effect is load-duration dependent, #385)
- H19 (roofline vs blackbox — partially confirmed: mean rankings preserved, P99 diverges, #385)
- H21 (extreme scorer weights — refuted: cold-start prefix cascade, tiebreaker is binary, #385)
- H24 (combined pathological — confirmed: 4.9x TTFT, routing ~95% dominant, super-additive, #385)

**Remaining (1 hypothesis):**
- H18 (unfair tenant fairness — blocked by #348)

**Newly completed:**
- H4 (RR vs LL — confirmed with nuance: mean equivalent, LL p99 12-21% worse from tie-breaking)
- H6 (counterfactual regret — confirmed with wrong mechanism: weighted regret structurally zero by design)
- H7 (horizontal scaling — confirmed: 7.4x super-linear TTFT scaling due to non-linear queue growth rate)
- H15 (fitness evaluation — confirmed with nuance: correctly ranks prefix-affinity; normalization compresses)
- H20 (heavy-tailed — refuted: ParetoLN produces fewer preemptions; median drives KV pressure, not tail)
- H23 (low-load equivalence — confirmed with nuance: all policies within 5%; uniform workload eliminates differentiation)

## Next Steps

1. ~~Create workload-spec YAMLs for each hypothesis~~ — done for all completed experiments
2. ~~Start with Tier 1 (invariants) to establish correctness baselines~~ — done
3. ~~**Highest-ROI batch:** H24, H19, H21, H16~~ — done (#385). Used shared experiment harness (`hypotheses/lib/`) for timeout safety.
4. ~~**Remaining highest-ROI:** H7 (horizontal scaling), H15 (fitness evaluation), H20 (heavy-tailed distributions)~~ **DONE**
5. **Act on #329 findings:** file design issue for DES service time split; update H-MMK FINDINGS; propose calibration rule
6. **Act on H17 findings:** file design issue for kv-heavy micro-bursting; follow-up hypothesis at high utilization
7. **Act on #385 findings:**
   - H21: Document that single-scorer prefix-affinity creates degenerate concentration — users should always pair with a load-balancing scorer
   - H16: Gamma burst effect is load-duration dependent — Poisson may be sufficient for long-running workloads
   - H19: Consider adding explicit `--latency-mode roofline` flag independent of alpha coefficients (CLI design coupling)
8. **Promote confirmed hypotheses to Go tests:** H26 (event pipeline), H25 (full-stack conservation), H24 (anomaly detection completeness)
9. **Act on new findings:**
   - H4: LL tie-breaking positional bias (instance 0 preference) — consider random tie-breaking option
   - H6: Counterfactual regret is structurally zero for weighted policies — document limitation; regret only meaningful for non-score-based policies
   - H7: Super-linear scaling is a general queueing property — document for capacity planning guidance
   - H20: Distribution median (not mean/tail) drives KV pressure — document for workload configuration guidance
   - H23: Routing policy differentiation requires workload heterogeneity — document as experiment design guidance

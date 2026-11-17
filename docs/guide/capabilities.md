# Capabilities Guide

Each section shows one capability as a scenario you can run: the YAML that states it, the command, what it prices, and what is refused by name. Every example is a file under `examples/capabilities/` and a test (`cmd/capability_examples_test.go`) runs it through the CLI and checks its effect against a control run with the capability off, so an example cannot stop working unnoticed. Runs are small: a few seconds each, no GPU.

Run an example from the repository root:

```bash
blis run --scenarios examples/capabilities --scenario <file> ...
```

## Engine and release

A scenario names the engine it runs: `engine: {kind: vllm, version: "0.33.0"}`. This build runs vLLM 0.33.0 only.

```yaml
kind: Scenario
name: session-load-only
engine: {kind: vllm, version: "0.33.0"}
```

Refused by name: `version: "0.32.0"` ("this build runs vllm 0.33.0, and the newest inference-sim release that runs it is v0.17.0"), an engine kind this build does not register (`sglang`: "this build runs the kinds vllm"), and the retired `engine_version:` field. The Result records the engine under `engine`.

## KV tier: CPU

`examples/capabilities/cpu-tier.yaml` budgets HBM so tightly (`gpu_memory_utilization: 0.2`) that 160 distinct 2,000-token system prompts do not fit, and gives the deployment a 512 GiB CPU tier:

```yaml
offload:
  connector: OffloadingConnector
  spec: TieringOffloadingSpec
  eviction_policy: lru
  tiers:
    - {tier: cpu_dram, bytes: 549755813888}
```

```bash
blis run --scenarios examples/capabilities --scenario cpu-tier.yaml --num-instances 1 \
  --workload-spec examples/capabilities/spill-workload.yaml --horizon 25000000
```

The coefficient set `cost-model-offload-tiers` prices each tier transfer from the catalog's storage device. Blocks evicted from HBM are found in the CPU tier on the prompt's next request: the cache hit rate rises and TTFT falls against the same run without the `offload` block. Tier details: [KV Cache & Memory](kv-cache.md).

## Prefill/decode with pools of unequal width

`pd-unequal-tp.yaml` runs four TP4 prefill engines and one TP8 decode engine. Each pool is sized and priced by its own layout:

```yaml
pools:
  - role: prefill
    nodes: 2
    parallel: {tp: 4, pp: 1, dp: 1, enable_expert_parallel: false}
  - role: decode
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1, enable_expert_parallel: false}
pd_transfer:
  connector: NixlConnector
```

```bash
blis run --scenarios examples/capabilities --scenario pd-unequal-tp.yaml \
  --num-instances 5 --prefill-instances 4 --decode-instances 1 --rate 5 --num-requests 20
```

Every request is handed off once (`breakdowns.pd.kv_transfers`), and the TP8 decode engine steps faster than a TP4 one on the same model. Pools of different data-parallel width are in `pd-unequal-dp.yaml` once the engine pin carries blis-vllm#206; until then the run is refused ("their dp must match").

## Per-pool hardware

`node-pools.yaml` states two node pools of different chips; `node-pools-bundle.yaml` gives each its nodes, memory and price:

```yaml
cluster:
  hardware: h200
  pools:
    - {name: fast, hardware: h200}
    - {name: older, hardware: h100}
```

```bash
blis run --scenarios examples/capabilities --scenario node-pools.yaml --num-instances 4 \
  --policy-config examples/capabilities/node-pools-bundle.yaml --rate 4 --num-requests 40
```

Each instance is sized and priced by the chip of the pool it is placed in; the smallest KV budget in the cluster is the H100's. A bundle pool whose `gpu_type` differs from the scenario's chip for that pool, or that the scenario does not list, is refused.

## Session affinity through the hosted router

`session-affinity.yaml` routes with the `session-affinity-scorer` (strategy `session_id`). Every round of a session carries its `x-session-id`:

```yaml
router:
  epp_flags: [--pool-name=blis-pool, --pool-namespace=default, --allow-experimental-plugins]
  endpoint_picker_config: router/session-affinity-epp.yaml
  manifests: [router/inferencepool.yaml]
```

```bash
blis run --scenarios examples/capabilities --scenario session-affinity.yaml --num-instances 4 \
  --workload-spec examples/capabilities/multiturn-sessions.yaml
```

A session is served by one instance and the prefix cache keeps its context; `session-load-only.yaml` is the control, where the rounds spread.

## Workloads

**Drain-gated phases.** `phases-workload.yaml` sends a warm-up and then the measured requests; the measured phase starts when the warm-up has drained, the cache carries over, and the measured window holds the measured rows only.

```yaml
phases:
  - {name: warmup, num_requests: 12, concurrency: 4}
  - {name: measured, num_requests: 24, concurrency: 4}
```

```bash
blis run --scenarios examples/capabilities --scenario cpu-tier.yaml --num-instances 1 \
  --workload-spec examples/capabilities/phases-workload.yaml
```

**Constant-load replay.** `constant-load-workload.yaml` holds a fixed number of live root sessions over an agentic corpus, with a warm-up outside the measured window (`replay: {scenario: aiperf-constant-load, load: {warmup_duration_s: 10}}`); see [Trace Replay](replay.md).

**Pooled session replay.** Record a multi-turn run with `--trace-output`, then `blis replay --concurrent-sessions 8 --total-sessions 40` replays 40 sessions with at most 8 live at once.

**Open-loop accumulate replay.** `open-loop-sessions.yaml` (`closed_loop: false`, `context_growth: accumulate`) exports as per-round deltas, and `blis replay --session-mode fixed-accumulate` replays it row for row.

## Autoscaling and loading delay

`autoscale.yaml` with `autoscale-bundle.yaml` scales the model, the nodes and, after a silence, the model to zero:

```yaml
autoscaler:
  interval_us: 1000000
  analyzer: {kind: threshold, signal: in_flight, target: 4, min_replicas: 0}
  scale_to_zero_idle_us: 5000000
  scale_from_zero_hold_us: 30000000
cluster_autoscaler: {policy: pending-driven, scale_down: bin-pack}
instance_lifecycle:
  loading_delay: {mean: 2.0, stddev: 0}
  weight_source: nvme_gen4
  drain_policy: WAIT
```

```bash
blis run --scenarios examples/capabilities --scenario autoscale.yaml --num-instances 1 \
  --policy-config examples/capabilities/autoscale-bundle.yaml \
  --workload-spec examples/capabilities/bursts-workload.yaml --horizon 120000000
```

A replica the autoscaler adds loads in `loading_delay` plus the time to read its weights from `weight_source`, so the second burst, which finds no replica, waits seconds for the first token; a slower storage tier waits longer. Configuration: [Configuration reference](../reference/configuration.md).

## Measurement outputs

Every instance states, under `reports.engine.instances.<id>.measured_window`, its busy share, the prefill and decode parts of it, and the window it measured over (`window_start_us`, `window_end_us`). The cluster's `busy_share` is the instances' mean. Any example shows them; see [Metrics & Results](results.md).

## Not yet exercisable from a scenario

- A vision tower: image tokens are priced as prompt text. The tower is priced in the cost model but no run calls it (no encode stage reads the scenario).
- A KV connector or CPU offload on one pool only: the `offload` block applies to every engine (blis-substrate#307).
- Dynamo and vLLM router bounds: the router table is read when a measured point is graded, not by `blis run`.

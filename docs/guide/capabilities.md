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

## KV tier: file system (fs)

`fs-tier.yaml` is the same deployment with an 8 GiB CPU tier and an NVMe device below it. The device is a catalog storage device (`nvme_gen4`, `nvme_gen3`, `sata_ssd`, `nvme_gen4_raid0_8x`), priced by the registry's `offload_tier_<device>_*` entries:

```yaml
offload:
  connector: OffloadingConnector
  spec: TieringOffloadingSpec
  tiers:
    - {tier: cpu_dram, bytes: 8589934592}
    - {tier: nvme_gen4, bytes: 4398046511104}
```

```bash
blis run --scenarios examples/capabilities --scenario fs-tier.yaml --num-instances 1 \
  --workload-spec examples/capabilities/spill-workload.yaml --horizon 25000000
```

Blocks the CPU tier evicts spill to the device and are read back later: against the same run without the device the hit rate is higher and more requests complete by the horizon. Tier copies share the device's link with each other and with P/D pulls, and the Result states each link's peak concurrent copies under `breakdowns.pd.link_peak_concurrent_transfers_tier-<device>-<to|from>-node-<n>`. `cpu_dram` must come first. Engines of one model that name one `root_dir` share one file system.

## KV tier: object store (obj)

`obj-tier.yaml` is the same deployment with a bucket of the catalog's `s3` device class below the CPU tier; `bytes` is the bucket's size, and the registry's `offload_tier_s3_*` entries price it. Engines of one model share the bucket.

```yaml
    - {tier: cpu_dram, bytes: 8589934592}
    - {tier: s3, bytes: 1099511627776}
```

```bash
blis run --scenarios examples/capabilities --scenario obj-tier.yaml --num-instances 1 \
  --workload-spec examples/capabilities/spill-workload.yaml --horizon 25000000
```

The hit rate rises against the run without the bucket, and the Result states the bucket's link (`link_peak_concurrent_transfers_tier-s3-to-node-0`).

## KV tier: peer-to-peer (p2p) and P/D

`pd-p2p.yaml` replaces NIXL with vLLM's p2p tier. The tier has no device, so it is an option of the offload connector, as in vLLM's `kv_connector_extra_config.secondary_tiers`:

```yaml
pd_transfer:
  connector: OffloadingConnector
offload:
  connector: OffloadingConnector
  spec: TieringOffloadingSpec
  tiers:
    - {tier: cpu_dram, bytes: 549755813888}
  connector_options:
    secondary_tiers: [{type: p2p, port: 5710}]
```

```bash
blis run --scenarios examples/capabilities --scenario pd-p2p.yaml \
  --num-instances 3 --prefill-instances 2 --decode-instances 1 --rate 2 --num-requests 6
```

The decode engine loads each prompt from the prefill engine's CPU tier; the Result states `peer_messages_sent`, `peer_messages_delivered` (equal) and `peer_messages_dropped` (zero). The prefill nodes' WRITEs cross the fabric, and the Result states the NIC links they used (`link_peak_concurrent_transfers_nic-egress-node-<prefill>` and `nic-ingress-node-<decode>`) beside the CPU tiers' links. The example runs one TP8 engine per node. With prefill and decode pools of unequal TP the run still completes every request as disaggregated, but its Result shows no NIC link and no decode-side tier load, so the handoff's fabric cost is not stated; that is a known gap ([inference-sim#2214](https://github.com/inference-sim/inference-sim/issues/2214)), and the example keeps equal TP. The NIXL deployment sends no peer messages.

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

### Unequal data-parallel width

`pd-unequal-dp.yaml`: MiniMax-M2.5 (a routed MoE, so dp is placement) with one-rank prefill pods and decode pods of dp 2. The engine builds one instance per data-parallel group and sizes and prices each pool by its own layout.

```bash
blis run --scenarios examples/capabilities --scenario pd-unequal-dp.yaml \
  --num-instances 6 --prefill-instances 4 --decode-instances 2 --rate 5 --num-requests 20
```

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

Every instance also states `mfu`, the model-FLOPs utilization over the window (never above the busy share; more load per weight read raises it). Every instance states, under `reports.engine.instances.<id>.measured_window`, its busy share, the prefill and decode parts of it, and the window it measured over (`window_start_us`, `window_end_us`). The cluster's `busy_share` is the instances' mean. Any example shows them; see [Metrics & Results](results.md).

## Not yet exercisable from a scenario

- A vision tower: image tokens are priced as prompt text. The tower is priced in the cost model but no run calls it (no encode stage reads the scenario).
- A KV connector or CPU offload on one pool only: the `offload` block applies to every engine (blis-substrate#307).
- Dynamo and vLLM router bounds: the router table is read when a measured point is graded, not by `blis run`.

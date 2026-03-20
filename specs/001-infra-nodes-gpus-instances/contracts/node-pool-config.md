# Contract: Node Pool Configuration Schema

**Feature**: Phase 1A — Infrastructure: Nodes, GPUs, Instances
**Date**: 2026-03-13

---

## YAML Schema: `NodePoolConfig`

Node pool configuration is added to `DeploymentConfig` and expressed in YAML. Unknown fields cause parse errors (`yaml.KnownFields(true)`, R10).

```yaml
node_pools:
  - name: <string>              # Required. Non-empty pool identifier.
                                # Used as prefix in node/GPU IDs.

    gpu_type: <string>          # Required. GPU model (e.g., "H100", "A100", "L4").
                                # Must match a key in hardware_config.json for
                                # KV capacity auto-calculation.

    gpus_per_node: <int>        # Required. ≥1. Number of GPUs on each node in this pool.

    gpu_memory_gib: <float64>   # Required. >0. Memory capacity per GPU in GiB.

    initial_nodes: <int>        # Required. ≥0. Nodes created at simulation start.
                                # Zero is valid (empty pool).

    min_nodes: <int>            # Required. ≥0, ≤max_nodes.
                                # Autoscaler floor (Phase 1C).

    max_nodes: <int>            # Required. ≥initial_nodes.
                                # Autoscaler ceiling. Hard cap.

    provisioning_delay:         # Required. Duration distribution for VM spin-up.
      kind: <string>            # One of: constant, gaussian, exponential, weibull
      mean: <float64>           # Required for all kinds. Seconds. ≥0.
      stddev: <float64>         # Optional. Only used by gaussian. Default 0.
      shape: <float64>          # Optional. Only used by weibull.
      scale: <float64>          # Optional. Only used by weibull.
```

---

## YAML Schema: `InstanceLifecycleConfig`

```yaml
instance_lifecycle:
  loading_delay:                # Optional. Default: constant(0).
    kind: <string>              # One of: constant, gaussian, exponential, weibull
    mean: <float64>             # Seconds. ≥0.

  warm_up_request_count: <int>  # Optional. Default: 0. ≥0.
                                # Number of requests with TTFT penalty after Active.

  warm_up_ttft_factor: <float64> # Optional. Default: 1.0. ≥1.0.
                                 # Multiplier applied to TTFT during warm-up.

  drain_policy: <string>        # Optional. Default: "WAIT".
                                # One of: IMMEDIATE, WAIT, REDIRECT.
```

---

## Validation Rules

| Rule | Condition | Error |
|------|-----------|-------|
| Pool name unique | All pool names must be distinct | `"duplicate pool name: {name}"` |
| GPUs per node | `gpus_per_node ≥ 1` | `"gpus_per_node must be ≥1"` |
| GPU memory | `gpu_memory_gib > 0` | `"gpu_memory_gib must be >0"` |
| Initial ≤ max | `initial_nodes ≤ max_nodes` | `"initial_nodes exceeds max_nodes"` |
| Min ≤ max | `min_nodes ≤ max_nodes` | `"min_nodes exceeds max_nodes"` |
| Provisioning delay | valid DistSpec | DistSpec validation error |
| Loading delay | valid DistSpec | DistSpec validation error |
| Warm-up factor | `≥ 1.0` | `"warm_up_ttft_factor must be ≥1.0"` |
| Drain policy | in `{IMMEDIATE, WAIT, REDIRECT}` | `"unknown drain_policy: {val}"` |

---

## CLI Contract

The `--num-instances` flag retains its meaning as "number of instances per model" when node pools are configured. When node pools are absent, `--num-instances` instances are created without node/GPU tracking (backward-compatible mode).

When `--cluster-config <file.yaml>` is provided, the file is parsed with `yaml.KnownFields(true)`. Unknown top-level fields cause a fatal error.

---

## Output Contract: Per-Model Metrics

When the simulation completes, `per_model` appears in the JSON output alongside existing aggregate metrics:

```json
{
  "per_model": {
    "<model-name>": {
      "ttft_p50": 0.45,
      "ttft_p99": 1.23,
      "e2e_p50": 2.1,
      "e2e_p99": 4.56,
      "throughput_rps": 8.7,
      "tokens_per_sec": 1240.0,
      "total_requests": 500
    }
  }
}
```

`per_model` is always present when `Request.Model` is non-empty for any completed request. In single-model runs, `per_model` contains exactly one entry.

---

## GPU ID Traceability Contract

GPU IDs encode their pool and node of origin and are stable within a simulation run:

```text
Format: {pool-name}-{node-index}-gpu-{gpu-index}
Example: h100-pool-0-gpu-3

Node ID: {pool-name}-{node-index}
Example: h100-pool-0
```

`PlacementRecord` in the simulation trace captures: timestamp, instance ID, model, outcome (placed/pending/evicted), node ID, and GPU ID list.

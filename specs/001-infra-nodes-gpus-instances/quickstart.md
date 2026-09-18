# Quickstart: Phase 1A — Infrastructure: Nodes, GPUs, Instances

**Branch**: `001-infra-nodes-gpus-instances`

---

## What This Feature Adds

Phase 1A makes nodes and GPUs first-class entities in BLIS. You can now:
1. Define **node pools** with GPU type, count, and provisioning delays
2. **Place model instances** onto nodes via bin-packing (TP degree = GPU count)
3. Observe **instance lifecycle phases** (Loading → WarmingUp → Active → Draining → Terminated)
4. Run **multi-model clusters** with per-model metrics in the output JSON

---

## Configuration: Node Pool YAML

Add `node_pools` to your deployment configuration or workload spec:

```yaml
# example-cluster.yaml
node_pools:
  - name: h100-pool
    gpu_type: H100
    gpus_per_node: 8
    gpu_memory_gib: 80.0
    initial_nodes: 2
    min_nodes: 1
    max_nodes: 4
    provisioning_delay:
      kind: constant
      mean: 120.0   # seconds

  - name: a100-pool
    gpu_type: A100
    gpus_per_node: 4
    gpu_memory_gib: 40.0
    initial_nodes: 1
    min_nodes: 0
    max_nodes: 2
    provisioning_delay:
      kind: gaussian
      mean: 90.0
      stddev: 15.0

instance_lifecycle:
  loading_delay:
    kind: constant
    mean: 30.0     # 30 seconds to load model weights
  warm_up_request_count: 10
  warm_up_ttft_factor: 2.0   # 2× TTFT for first 10 requests
  drain_policy: WAIT         # finish in-flight before terminating
```

---

## Multi-Model Cluster

Requests carry a `model` field that routes them to the correct instances:

```yaml
# workload-spec.yaml
clients:
  - name: llama-traffic
    model: meta-llama/Llama-3.1-8B
    rate: 5.0
    # ... token distributions

  - name: qwen-traffic
    model: qwen/Qwen3-14B
    rate: 3.0
    # ... token distributions
```

The simulator creates separate instances for each model (based on `num_instances` per model) and routes each request only to instances serving its model.

---

## Running a Node-Pool Simulation

```bash
# Single-model cluster with node pools
./blis run \
  --model qwen/qwen3-14b \
  --hardware H100 \
  --tp 4 \
  --num-instances 2 \
  --cluster-config example-cluster.yaml

# Output includes per-model metrics (even for single-model clusters):
# {
#   "per_model": {
#     "qwen/qwen3-14b": {
#       "ttft_p99": 1.23,
#       "e2e_p99": 4.56,
#       "throughput_rps": 8.7
#     }
#   }
# }
```

---

## Verifying the GPU Conservation Invariant

The simulator automatically checks `allocated + free == total_gpus` per node after every placement/release. Violations are logged to stderr as warnings and counted in `RawMetrics.GPUConservationViolations`.

---

## Instance States and Routing

| State | Routable? | Notes |
|-------|-----------|-------|
| Scheduling | No | Waiting for node placement |
| Loading | No | Loading model weights |
| WarmingUp | Yes (with penalty) | First N requests get `TTFT × WarmUpTTFTFactor` |
| Active | Yes | Normal operation |
| Draining | No | WAIT/REDIRECT policies; no new requests |
| Terminated | No | Simulation complete |

---

## Per-Model Metrics JSON Schema

When using `--metrics-path`, the output JSON includes a `per_model` key (omitted if no requests carry a model tag):

```json
{
  "per_model": {
    "meta-llama/Llama-3.1-8B": {
      "model": "meta-llama/Llama-3.1-8B",
      "ttft": {
        "Mean": 0.45, "P50": 0.40, "P95": 0.90, "P99": 1.23,
        "Min": 0.10, "Max": 2.50, "Count": 500
      },
      "e2e": {
        "Mean": 2.1, "P50": 1.9, "P95": 3.8, "P99": 4.56,
        "Min": 0.5, "Max": 8.0, "Count": 500
      },
      "throughput_rps": 8.7,
      "tokens_per_sec": 1240.0,
      "total_requests": 500
    }
  }
}
```

**Fields:** All latency values are in simulation ticks (microseconds). `total_requests` counts completed requests only. `tokens_per_sec` is total output tokens / simulation duration. `per_model` is `omitempty` — absent for single-model runs where `Request.Model` is empty.

## Drain Policies

| Policy | Behavior |
|--------|----------|
| `IMMEDIATE` | Instance terminates immediately; in-flight requests receive no more steps |
| `WAIT` | New requests excluded from routing; existing in-flight requests complete normally |
| `REDIRECT` | Like WAIT, plus queued (not yet scheduled) requests are re-injected to other Active instances |

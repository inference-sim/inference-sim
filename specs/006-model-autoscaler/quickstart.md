# Quickstart: Model Autoscaler

**Branch**: `006-model-autoscaler`

This guide shows how to enable and configure the model autoscaler in a BLIS simulation after the 1C-1a through 1C-1d PRs are merged.

---

## Minimal Configuration (YAML)

```yaml
# Minimal viable pipeline: DefaultCollector → SaturationAnalyzer → UnlimitedEngine → DirectActuator
model_autoscaler_interval_us: 60000000   # 60s tick interval

# No actuation delay (default): ScaleActuationEvent fires in same tick as ScalingTickEvent
# actuation_delay:
#   mean: 0
#   stddev: 0

# Cooldown: prevent oscillation (optional; 0 = disabled by default)
scale_up_cooldown_us: 120000000    # 2 minutes
scale_down_cooldown_us: 300000000  # 5 minutes

# Node pools must have CostPerHour for cost-aware allocation
node_pools:
  - name: a100-pool
    gpu_type: A100-80GB
    gpus_per_node: 8
    gpu_memory_gib: 80
    initial_nodes: 4
    min_nodes: 1
    max_nodes: 10
    cost_per_hour: 12.0         # ← new field required for Engine cost-aware decisions
```

---

## Wiring in Go (Programmatic)

```go
// After 1C-1a/1C-1b are merged:
cfg := cluster.DeploymentConfig{
    // ... existing fields ...
    ModelAutoscalerIntervalUs: 60_000_000,  // 60s
    ActuationDelay:            cluster.DelaySpec{Mean: 30, Stddev: 10},  // 30s ± 10s (Mean/Stddev are in seconds)
    ScaleUpCooldownUs:         120_000_000,
    ScaleDownCooldownUs:       300_000_000,
}

// The simulator wires the pipeline internally. No explicit interface assignment needed
// for the default pipeline (DefaultCollector + SaturationAnalyzer + UnlimitedEngine + DirectActuator).
```

---

## Swapping Components

```go
// After 1C-1c (baseline analyzers):
// - Use UtilizationAnalyzer instead of SaturationAnalyzer
// - Configure via new field on DeploymentConfig (exact field TBD in 1C-1c micro-plan)

// After 1C-1d (engines):
// - Use GreedyEngine instead of UnlimitedEngine for GPU-inventory-aware allocation
```

---

## Testing the Pipeline

```bash
# Run all autoscaler tests
go test ./sim/cluster/... -run TestAutoscaler -v
go test ./sim/cluster/... -run TestSaturationAnalyzer -v
go test ./sim/cluster/... -run TestGreedyEngine -v

# Verify INV-6 (no regression with zero-interval autoscaler)
./blis run --model qwen/qwen3-14b > out-with-autoscaler.txt
# (run without autoscaler config for comparison)
diff out-baseline.txt out-with-autoscaler.txt   # must be empty

# Run all tests + lint
go test ./...
golangci-lint run ./...
```

---

## Observing Autoscaler Behavior

The autoscaler writes scaling decisions to stderr (diagnostic output). Simulation metrics (stdout) remain deterministic.

To observe scaling events in a simulation run, redirect stderr:
```bash
./blis run --model qwen/qwen3-14b 2>autoscaler.log
grep "scale" autoscaler.log
```

---

## Disabling the Autoscaler

Set `model_autoscaler_interval_us: 0` (or omit the field entirely — the zero value disables the autoscaler). When disabled, no `ScalingTickEvent` is ever scheduled, and the simulation is byte-identical to a run before Phase 1C was introduced (INV-6).

---

## Dependency Chain

| PR | Issues | What it adds | Required before |
|----|--------|-------------|-----------------|
| 1C-1a | #692 | Interfaces, types, events, wiring | Everything else |
| 1C-1b | #905 | SaturationAnalyzer, DefaultCollector, DirectActuator | Integration test |
| 1C-1c | #906 | UtilizationAnalyzer, QueueAnalyzer | Optional baseline experiments |
| 1C-1d | #918 | GreedyEngine, UnlimitedEngine | GPU-inventory-aware allocation |

The minimal viable pipeline (1C-1a + 1C-1b) is the validation target for the WVA/llm-d team.

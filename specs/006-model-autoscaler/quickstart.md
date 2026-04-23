# Quickstart: Model Autoscaler

**Branch**: `006-model-autoscaler`

This guide shows how to enable and configure the model autoscaler in a BLIS simulation after the 1C-1a through 1C-1d PRs are merged. 1C-1a is already merged (PR #934).

---

## Minimal Configuration (YAML)

```yaml
# Minimal viable pipeline: DefaultCollector → V2SaturationAnalyzer → UnlimitedEngine → DirectActuator
autoscaler:
  interval_us: 60000000   # 60s tick interval

  # HPA scrape lag (default 0): ScaleActuationEvent fires in same tick as ScalingTickEvent
  # hpa_scrape_delay:
  #   mean: 0
  #   stddev: 0

  # Stabilization windows: hold off on acting until signal is consistently present (0 = disabled by default)
  scale_up_stabilization_window_us: 120000000    # 2 minutes
  scale_down_stabilization_window_us: 300000000  # 5 minutes (matches Kubernetes HPA default)

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
    HPAScrapeDelay:                    cluster.DelaySpec{Mean: 30, Stddev: 10},  // 30s ± 10s (Mean/Stddev are in seconds)
    ScaleUpStabilizationWindowUs:      120_000_000,
    ScaleDownStabilizationWindowUs:    300_000_000,
}

// NOTE: Automatic pipeline wiring from `blis run` CLI flags is not yet implemented.
// The autoscaler pipeline is constructed internally by NewClusterSimulator when
// ModelAutoscalerIntervalUs > 0, but the four components (Collector, Analyzer,
// Engine, Actuator) are initialized to nil. cmd/ wiring to inject the default
// V2 pipeline (DefaultCollector + V2SaturationAnalyzer + UnlimitedEngine +
// DirectActuator) is planned for a follow-up.
//
// For now, the pipeline is testable via internal wiring in sim/cluster/ tests:
//   cs.autoscaler = newAutoscalerPipeline(collector, analyzer, engine, actuator, rng)
```

> **Limitation:** The autoscaler pipeline currently runs only in tests. CLI integration
> (e.g., `--autoscaler-analyzer v2-saturation`) will be added in a follow-up PR to
> expose the pipeline from `blis run` and `blis replay`.

---

## Swapping Components

```go
// After 1C-1d (GreedyEngine):
// - Use GreedyEngine instead of UnlimitedEngine for GPU-inventory-aware allocation

// Future (#954 — QueueingModelAnalyzer):
// - Use QueueingModelAnalyzer instead of V2SaturationAnalyzer
// - M/M/1/K-SD queueing model with online EKF parameter learning (WVA parity)
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

Set `autoscaler.interval_us: 0` (or omit the `autoscaler` block entirely — the zero value disables the autoscaler). When disabled, no `ScalingTickEvent` is ever scheduled, and the simulation is byte-identical to a run before Phase 1C was introduced (INV-6).

---

## Dependency Chain

| PR | Issue | What it adds | Status |
|----|-------|-------------|--------|
| 1C-1a | #692 | Interfaces, types, events, pipeline wiring, stabilization window gate | ✅ Merged (#934) |
| 1C-1b | #905 | V2SaturationAnalyzer, DefaultCollector, UnlimitedEngine, DirectActuator | Pending |
| 1C-1d | #918 | GreedyEngine (inventory-aware allocation) | Pending |
| Future | #954 | QueueingModelAnalyzer (M/M/1/K-SD with EKF) | Future |

The minimal viable pipeline (1C-1a + 1C-1b) is the validation target for the WVA/llm-d team.

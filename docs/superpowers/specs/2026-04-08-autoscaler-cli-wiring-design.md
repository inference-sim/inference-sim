# Design: Autoscaler CLI Wiring

**Date:** 2026-04-08
**Branch:** feat/autoscaler-cli-wiring
**Status:** Approved — ready for implementation

---

## 1. Problem

The Phase 1C-1b autoscaler pipeline (`DefaultCollector → V2SaturationAnalyzer → UnlimitedEngine → DirectActuator`) and Phase 1A node/GPU infrastructure (`NodePoolConfig`, `PlacementManager`) are fully implemented and unit-tested at the library level, but neither is reachable from the CLI.

`DeploymentConfig` has fields `ModelAutoscalerIntervalUs`, `ActuationDelay`, `ScaleUpCooldownUs`, `ScaleDownCooldownUs`, and `NodePools []NodePoolConfig`, but `cmd/root.go` never populates them and never injects the concrete pipeline components after `NewClusterSimulator`.

The result: autoscaling cannot be exercised via `blis run`, making it impossible to demo, validate, or experiment with the feature.

---

## 2. Design

### 2.1 Configuration surface

Extend the existing `--policy-config` YAML with two new top-level sections:

```yaml
node_pools:
  - name: h100-pool
    gpu_type: H100
    gpus_per_node: 8
    gpu_memory_gib: 80.0
    initial_nodes: 1
    min_nodes: 1
    max_nodes: 4
    cost_per_hour: 32.0
    provisioning_delay:
      mean: 30.0    # seconds
      stddev: 5.0

autoscaler:
  interval_us: 30000000          # 30s tick; 0 = disabled (default)
  scale_up_cooldown_us: 60000000
  scale_down_cooldown_us: 180000000
  actuation_delay:
    mean: 10.0                   # seconds
    stddev: 2.0
  analyzer:
    kv_cache_threshold: 0.8
    scale_up_threshold: 0.8
    scale_down_boundary: 0.4
    avg_input_tokens: 512.0
```

One CLI escape hatch: `--model-autoscaler-interval-us <N>` overrides `autoscaler.interval_us` when set (non-zero). This lets quick experiments enable/disable the autoscaler without editing the YAML file, consistent with how other CLI flags override `--policy-config` values.

Autoscaler is disabled when `interval_us == 0` (the zero value). No autoscaler section in the YAML means no autoscaling (backward-compatible).

### 2.2 Dependency constraint and solution

`PolicyBundle` lives in package `sim`. `NodePoolConfig` lives in `sim/cluster`, which already imports `sim`. Adding `sim → sim/cluster` would be circular.

**Solution:** Define a parallel `NodePoolBundleConfig` struct in `sim/bundle.go` with identical YAML tags and field types to `cluster.NodePoolConfig`. In `cmd/root.go`, convert each `NodePoolBundleConfig` to `cluster.NodePoolConfig` before building `DeploymentConfig`. The conversion is mechanical (field-by-field copy); no logic lives there.

`AutoscalerBundleConfig` and its nested `AnalyzerBundleConfig` use only primitive types (`float64`) and `DelaySpec` — but `DelaySpec` is also in `sim/cluster`. Apply the same solution: define a parallel `DelayBundleSpec` in `sim/bundle.go` (`Mean`, `Stddev float64`), convert in `cmd/`.

This avoids circular imports, keeps `sim/` free of cluster knowledge, and places the conversion responsibility squarely in `cmd/` where it belongs.

### 2.3 Pipeline injection — inside NewClusterSimulator

`autoscalerPipeline` fields (`collector`, `analyzer`, `engine`, `actuator`) are unexported; `cmd/` cannot write them directly. Tests replace the entire `cs.autoscaler` object from inside the package.

**Solution:** `NewClusterSimulator` wires the default pipeline itself when `ModelAutoscalerIntervalUs > 0`. `DeploymentConfig` gains a new embedded `AutoscalerAnalyzerConfig` (containing the V2SaturationAnalyzer thresholds), and `NewClusterSimulator` builds `DefaultCollector + V2SaturationAnalyzer(config) + UnlimitedEngine + DirectActuator(cs)` in the same block that currently creates the nil-component pipeline.

`cmd/root.go` only sets `DeploymentConfig` fields — no injection code needed there. This is consistent with every other module (router, scheduler, admission controller) where `cmd/` sets config and the constructor wires the implementation.

Tests that need custom components (stubs, nopActuator) continue to replace `cs.autoscaler` after construction — they are in the same package and have access.

**New fields added to `DeploymentConfig`:**

```
AutoscalerAnalyzerConfig  V2SaturationAnalyzerConfig  // thresholds for V2SaturationAnalyzer
```

(`V2SaturationAnalyzerConfig` already lives in `sim/cluster`, so no import issues.)

### 2.4 Files changed

| File | Change |
|------|--------|
| `sim/bundle.go` | Add `NodePoolBundleConfig`, `DelayBundleSpec`, `AutoscalerBundleConfig`, `AnalyzerBundleConfig`; add `NodePools` and `Autoscaler` fields to `PolicyBundle`; extend `Validate()` |
| `sim/bundle_test.go` | Tests for new YAML sections: round-trip, validation errors, zero-value safety |
| `sim/cluster/deployment.go` | Add `AutoscalerAnalyzerConfig V2SaturationAnalyzerConfig` field to `DeploymentConfig` |
| `sim/cluster/cluster.go` | Wire default pipeline in `NewClusterSimulator` when `ModelAutoscalerIntervalUs > 0` |
| `cmd/root.go` | Add `--model-autoscaler-interval-us` flag; convert bundle `autoscaler:` + `node_pools:` fields → `DeploymentConfig` |
| `examples/autoscaler-demo.yaml` | New self-contained demo workload+policy (see §2.5) |

No changes to `sim/cluster/` — all autoscaler types are already exported and injectable.

### 2.5 Demo artifact

`examples/autoscaler-demo.yaml` — a policy config that, combined with a spiked workload, demonstrates scale-up and scale-down. It includes:

- A node pool with spare capacity beyond `--num-instances`
- Autoscaler interval of 30s, cooldowns of 60s up / 180s down
- A 10s actuation delay modeling HPA scrape lag
- Comments explaining each parameter and what to observe

Demo command:

```bash
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 2 \
  --policy-config examples/autoscaler-demo.yaml \
  --workload-spec examples/regression_workload_load_spikes.yaml \
  --rate 50 --num-requests 500 \
  --horizon 300000000
```

---

## 3. Invariants and backward compatibility

- **INV-6 Determinism:** `interval_us: 0` (default) produces no `ScalingTickEvent`; existing runs are byte-identical.
- **INV-A2:** `DirectActuator.scaleUp` requires `PlacementManager`. With `node_pools:` configured, `placement` is non-nil and scale-up works. Without `node_pools:`, scale-up logs an error and continues (no panic).
- **Backward compat:** `--policy-config` files without `node_pools:` or `autoscaler:` sections parse cleanly (strict parser ignores absent keys; zero values are safe per existing `DeploymentConfig` guarantees).
- **R10 strict YAML:** `KnownFields(true)` already in `LoadPolicyBundle`; new structs inherit this.
- **R8 exported mutable maps:** `NodePoolBundleConfig` has no maps.

---

## 4. Out of scope

- `GreedyEngine` (Phase 1C-1d, issue #918) — still `UnlimitedEngine` only
- Cluster autoscaler / node provisioning (Phase 1C-2a) — deferred
- Per-model autoscaler config (different thresholds per model) — single global config for now
- Metrics/trace records for scale events (Phase 1C-5a, issue #743) — deferred

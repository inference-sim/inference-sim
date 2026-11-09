# Test Data

## Golden Dataset (`goldendataset.json`)

The golden dataset contains known-good simulation outputs for regression testing.
Tests in `sim/cluster/cluster_test.go` and `sim/cluster/instance_golden_test.go` compare
simulation output against these values.

Steps are priced by the deterministic fake latency model in `sim/internal/testutil`
(`testutil.DefaultFakeLatency`, adapted to the engine contract by
`fake_latency_engine_test.go` in `sim/cluster` and in `sim/llmd`) --
not by any real latency backend. The dataset therefore pins simulation behavior
(scheduling, batch formation, KV allocation, workload generation, metric aggregation),
never pricing; each case records `"approach": "fake-latency"`.

### When to regenerate

Regenerate after ANY deliberate change that affects simulation output:
- The fake latency model's formula or default coefficients
- Request scheduling or batch formation logic
- KV cache allocation or eviction
- Workload generation (RNG, distribution parameters)
- Metric collection or aggregation

### How to regenerate

```bash
REGEN_GOLDEN=1 go test ./sim/ -run TestRegenGoldenDataset -count=1
go test ./sim/... -run Golden -v
```

`TestRegenGoldenDataset` (sim/simulator_test.go) re-runs every case with the default
fake and rewrites the `metrics` section. (`go test ./sim/ -run TestSimulator_GoldenDataset
-update-golden` does the same while preserving each case's wall-clock
`simulation_duration_s`.) Read the diff before committing it: only the metrics the
change is meant to move should move.

### Companion invariant tests

Per R7 (docs/contributing/standards/rules.md), every golden test MUST have a companion
invariant test. The companions are:
- `TestSimulator_GoldenDataset` -> inline INV-1, INV-4, INV-5 checks (sim/simulator_test.go)
- `TestInstanceSimulator_GoldenDataset_Equivalence` -> `TestInstanceSimulator_GoldenDataset_Invariants` (sim/cluster/instance_test.go)
- `TestClusterSimulator_SingleInstance_GoldenEquivalence` -> `TestClusterSimulator_SingleInstance_GoldenInvariants` (sim/cluster/cluster_test.go)

---

## The cost model's scenario fixtures (`kernel/`)

`testdata/kernel/` holds the scenario+deployment fixtures the cost model is tested on: the
hand-written files directly under it, and the generated `aisimulate/` and `direct/`. They came
from blis-latency-kernel v0.5.0 (commit 560bda63998700b1745c84387008ce3c1ded1b2a, its
`testdata/`), because they carry engine blocks and need an engine to lower them, and only this
repository imports both the engine and the cost model (release plan §2).
`internal/kernelfixtures` locates them.

## The catalog and registry

Tests read the `catalog/` and `registry/` directories of the blis-substrate module that
`go.mod` selects (`internal/substratedata`). There is no copy here and no lock file: the
substrate requirement in `go.mod` is the one place that names the data.

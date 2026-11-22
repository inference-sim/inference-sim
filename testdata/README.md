# Test Data

## Golden Dataset (`goldendataset.json`)

The golden dataset contains known-good simulation outputs for regression testing.
Tests in `sim/cluster/cluster_test.go` and `sim/cluster/instance_golden_test.go` read
this file and compare simulation output against these values.

Steps are priced by the deterministic fake latency model in `sim/internal/testutil`
(`testutil.DefaultFakeLatency`, adapted to the engine contract by
`fake_latency_engine_test.go` in `sim/cluster` and in `sim/llmd`) --
not by any real latency backend. The dataset therefore pins simulation behavior
(scheduling, batch formation, KV allocation, workload generation, metric aggregation),
never pricing; each case records `"approach": "fake-latency"`.

### When it changes

The dataset's `metrics` section is read, never written, by the tests: nothing rewrites it. After a
deliberate change that affects simulation output (the fake latency model's formula or default
coefficients, request scheduling or batch formation, KV allocation or eviction, workload
generation, metric collection) the failing test prints the new value; read the diff, then edit
`goldendataset.json` by hand so that only the metrics the change is meant to move move.
(`testutil.UpdateGolden` / `-update-golden` is declared in `sim/internal/testutil/golden.go`, but
no test reads it.)

### Companion invariant tests

Per R7 (.github/agent-prompts/standards/rules.md), every golden test MUST have a companion
invariant test. The companions are:
- `TestClusterSimulator_SingleInstance_GoldenEquivalence` -> `TestClusterSimulator_SingleInstance_GoldenInvariants` (sim/cluster/cluster_test.go)
- `TestInstance_GoldenDataset_KeepsTheInstanceLaws` (sim/cluster/instance_golden_test.go) checks INV-1, INV-3 and INV-5 on the same workloads

---

## The cost model's scenario fixtures (`kernel/`)

`testdata/kernel/` holds the scenario+deployment fixtures the cost model is tested on: the
hand-written files directly under it, and the generated `aisimulate/` and `direct/`. They came
from blis-latency-kernel v0.5.0 (commit 560bda63998700b1745c84387008ce3c1ded1b2a, its
`testdata/`), because they carry engine blocks and need an engine to lower them, and only this
repository imports both the engine and the cost model (release plan §2).
`internal/kernelfixtures` locates them.

## The catalog and registry

Tests read the `catalog/` and `registry/` directories under `substrate/` (`internal/substratedata`).
There is no copy here and no lock file: `substrate/` is the one place that holds the data.

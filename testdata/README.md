# Test Data

## Golden Dataset (`goldendataset.json`)

The golden dataset contains known-good simulation outputs for regression testing.
Tests in `sim/simulator_test.go`, `sim/cluster/cluster_test.go` and
`sim/cluster/instance_test.go` compare simulation output against these values.

Steps are priced by the deterministic fake latency model in `sim/internal/testutil`
(`testutil.DefaultFakeLatency`, adapted to `sim.LatencyModel` by
`sim/internal/testutil/fakelatency` and by package sim's `fake_latency_test.go`) --
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

## Vendored upstream releases (`catalog/`, `registry/`)

`testdata/catalog/` is a verbatim subset of a tagged blis-catalog release and
`testdata/registry/` holds a tagged blis-registry release's coefficient sets. Both are pinned to
the releases blis-latency-kernel's `testdata/upstream.lock` names, and a test fails if they
drift. See `catalog/README.md` for what is included and how to update it. The kernel's scenario
fixtures are not copied: they are read from the blis-latency-kernel module `go.mod` pins.


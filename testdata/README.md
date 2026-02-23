# Test Data

## Golden Dataset (`goldendataset.json`)

The golden dataset contains known-good simulation outputs for regression testing.
Tests in `sim/simulator_test.go` and `sim/cluster/cluster_test.go` compare
simulation output against these values.

### When to regenerate

Regenerate after ANY change that affects simulation output:
- Latency model coefficients or formula
- Request scheduling or batch formation logic
- KV cache allocation or eviction
- Workload generation (RNG, distribution parameters)
- Metric collection or aggregation

### How to regenerate

```bash
# From the repository root:
go test ./sim/... -run TestSimulator_GoldenDataset -update-golden
```

If the `-update-golden` flag is not implemented, manually run the simulation
with the golden dataset parameters and capture the output:

```bash
# See sim/internal/testutil/golden.go for the dataset format
# Each test case specifies model, seed, coefficients, and expected metrics
```

### Companion invariant tests

Per R7 (docs/standards/rules.md), every golden test MUST have a companion
invariant test. The companions are:
- `TestSimulator_GoldenDataset` -> inline INV-1, INV-4, INV-5 checks (sim/simulator_test.go)
- `TestInstanceSimulator_GoldenDataset_Equivalence` -> `TestInstanceSimulator_GoldenDataset_Invariants` (sim/cluster/instance_test.go)
- `TestClusterSimulator_SingleInstance_GoldenEquivalence` -> `TestClusterSimulator_SingleInstance_GoldenInvariants` (sim/cluster/cluster_test.go)

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

Manually run the simulation with each golden dataset test case's parameters
and update the expected metrics in `goldendataset.json`:

1. Read `sim/internal/testutil/golden.go` for the dataset format
2. Each test case specifies model, seed, coefficients, and expected metrics
3. Run the simulator with those parameters and capture the output
4. Update the `metrics` section for each test case
5. Verify all tests pass: `go test ./sim/... -run Golden -v`

### Companion invariant tests

Per R7 (docs/contributing/standards/rules.md), every golden test MUST have a companion
invariant test. The companions are:
- `TestSimulator_GoldenDataset` -> inline INV-1, INV-4, INV-5 checks (sim/simulator_test.go)
- `TestInstanceSimulator_GoldenDataset_Equivalence` -> `TestInstanceSimulator_GoldenDataset_Invariants` (sim/cluster/instance_test.go)
- `TestClusterSimulator_SingleInstance_GoldenEquivalence` -> `TestClusterSimulator_SingleInstance_GoldenInvariants` (sim/cluster/cluster_test.go)

---

## Latency Backend Golden Datasets

These datasets validate latency backend predictions remain stable across code changes.

### Trained-Physics Dataset (`trained_physics_iter29.json`)

Contains 15 iter29 training experiments with expected TTFT, E2E, and ITL metrics from the `trained-physics` latency backend (iter29 alpha/beta coefficients, loss=34.5675%).

**Test:** `TestTrainedPhysics_GoldenDataset` in `sim/cluster/trained_physics_golden_test.go`

**When to regenerate:** NEVER update values in place. If trained-physics backend behavior must change intentionally, rename the backend (e.g., `trained-physics-v2`) and create a new dataset. This prevents silent regressions from accumulating.

**How to regenerate (new backend only):**
1. Generate golden values by running BLIS with the new backend across all 15 experiments
2. Create a new JSON file: `testdata/<new-backend>_iter29.json`
3. Update test to reference the new file
4. Keep old dataset for historical comparison

**Experiments:** 15 experiments across Scout (TP2), Llama-2-7B (TP1), Llama-3.1-70B (TP4), Mistral-Nemo (TP1/TP2), Qwen2.5-7B (TP1), Yi-34B (TP2) on H100 hardware.

**Invariants checked:** INV-1 (request conservation), token conservation, INV-5 (causality: TTFT > 0, TTFT < E2E)

### Roofline Dataset (`roofline_goldendataset.json`)

Contains the same 15 iter29 experiments with expected metrics from the analytical `roofline` latency backend (no learned coefficients).

**Test:** `TestRoofline_GoldenDataset` in `sim/cluster/roofline_golden_test.go`

**When to regenerate:** NEVER update values in place. If roofline backend calculations must change intentionally, rename the backend (e.g., `roofline-v2`) and create a new dataset.

**How to regenerate (new backend only):**
1. Manually run BLIS with `--latency-model roofline` for all 15 experiments
2. Capture TTFT, E2E, ITL metrics (mean/P90/P99)
3. Create a new JSON file: `testdata/roofline-v2_goldendataset.json`
4. Update test to reference the new file

**Experiments:** Identical to trained-physics dataset (15 experiments, same models/TP/hardware)

**Invariants checked:** INV-1 (request conservation), token conservation, INV-5 (causality)

**Regression protection:** Guards against unintended changes to roofline FLOPs/bandwidth calculations, Scout MoE interleaved architecture handling (issue #877), weight bandwidth calculations, and TP all-reduce modeling.

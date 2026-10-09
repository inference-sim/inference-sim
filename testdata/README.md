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

**When to regenerate:** <a id="golden-regeneration-policy"></a>What `iter29` names is the **alpha/beta coefficient set**, not a frozen output snapshot. That splits the rule in two, and only the first half is "never":

- **Re-fitting the coefficients is a different model, not a fix to this one.** Rename the backend (e.g. `trained-physics-v2`), create `testdata/<new-backend>_<iter>.json`, point a new test at it, and keep this dataset for historical comparison. Never overwrite these values with a different generation's.
- **A deliberate change to what the backend computes under the SAME coefficients regenerates in place** — that is how every expected-value change in this file's history was made (#965, #974, #985, #1098, #1099, #1146, and #1419, which rescoped exactly the same routed-expert weight term as #1849). Forking a backend name per structural fix would fork the `--latency-model` surface users type, and a v2/v3/v4 ladder of near-identical datasets protects nothing.

Regenerating in place is a **claim that simulator output changed on purpose**, so it is only legitimate when all three hold:

1. An issue-scoped behavior change motivates it — never to make a failing test go green, and never as an unexplained side effect.
2. The PR states which experiments move, in which direction, and why.
3. Experiments the change **cannot** affect stay byte-identical. For a MoE-only change that means all dense rows — the evidence INV-BC-DP1 rests on. Last-bit float churn in untouched rows is noise, not a result: revert it rather than committing it.

**How to regenerate (in place, same coefficients):**
```bash
go test ./sim/cluster/ -run TestTrainedPhysics_GoldenDataset -update-golden
```
Then read the diff against the three conditions above before committing it.

**How to regenerate (new backend, re-fitted coefficients):**
1. Generate golden values by running BLIS with the new backend across all 15 experiments
2. Create a new JSON file: `testdata/<new-backend>_iter29.json`
3. Update test to reference the new file
4. Keep old dataset for historical comparison

**Experiments:** 15 experiments across Scout (TP2), Llama-2-7B (TP1), Llama-3.1-70B (TP4), Mistral-Nemo (TP1/TP2), Qwen2.5-7B (TP1), Yi-34B (TP2) on H100 hardware.

**Invariants checked:** INV-1 (request conservation), token conservation, INV-5 (causality: TTFT > 0, TTFT < E2E)

### Roofline Dataset (`roofline_goldendataset.json`)

Contains the same 15 iter29 experiments with expected metrics from the analytical `roofline` latency backend (no learned coefficients).

**Test:** `TestRoofline_GoldenDataset` in `sim/cluster/roofline_golden_test.go`

**When to regenerate:** Same policy as the trained-physics dataset above — see [When to regenerate](#golden-regeneration-policy). A re-derivation that makes this a different analytical model renames the backend (e.g. `roofline-v2`) and starts a new dataset; a deliberate, issue-scoped fix to what the existing `roofline` backend computes regenerates in place under the same three conditions.

**How to regenerate (in place, same analytical model):**
```bash
go test ./sim/cluster/ -run TestRoofline_GoldenDataset -update-golden
```

**How to regenerate (new backend, re-derived model):**
1. Manually run BLIS with `--latency-model roofline` for all 15 experiments
2. Capture TTFT, E2E, ITL metrics (mean/P90/P99)
3. Create a new JSON file: `testdata/roofline-v2_goldendataset.json`
4. Update test to reference the new file

**Experiments:** Identical to trained-physics dataset (15 experiments, same models/TP/hardware)

**Invariants checked:** INV-1 (request conservation), token conservation, INV-5 (causality)

**Regression protection:** Guards against unintended changes to roofline FLOPs/bandwidth calculations, Scout MoE interleaved architecture handling (issue #877), weight bandwidth calculations, and TP all-reduce modeling.

---

## Vendored upstream releases (`catalog/`, `registry/`)

`testdata/catalog/` is a verbatim subset of a tagged blis-catalog release and
`testdata/registry/` holds a tagged blis-registry release's coefficient sets. Both are pinned to
the releases blis-latency-kernel's `testdata/upstream.lock` names, and a test fails if they
drift. See `catalog/README.md` for what is included and how to update it. The kernel's scenario
fixtures are not copied: they are read from the blis-latency-kernel module `go.mod` pins.


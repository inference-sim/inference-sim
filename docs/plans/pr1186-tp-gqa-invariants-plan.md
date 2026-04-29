# PR Plan: TP>1 and GQA Behavioral Invariants for trained-physics and roofline

**Goal:** Add three missing behavioral invariant tests to `sim/latency/` — TP=2 produces shorter step time than TP=1 (trained-physics only; roofline already covered), and GQA produces shorter step time than MHA (both backends).

**Source:** https://github.com/inference-sim/inference-sim/issues/1186

**Closes:** #1186

**Tier:** Small (2 test files, new test functions only, no production code changes)

**Source audit:** Issue #1186 is unambiguous. All referenced helpers (`trainedPhysicsTestModelConfig`, `testHardwareConfig`, `testCoeffs`, `makePrefillBatch`, `makeDecodeBatch`, `testModelConfig`, `testHardwareCalib`, `rooflineStepTime`) confirmed present in test files. No clarifications needed.

---

## Behavioral Contracts

**BC-1 (trained-physics TP scaling):**
GIVEN two `TrainedPhysicsModel` instances built from the same `ModelConfig`, `HardwareCalib`, and `LatencyCoeffs`
WHEN one is constructed with `TP=1` and the other with `TP=2`
AND both run the same mixed prefill+decode batch
THEN `stepTime(TP=2) < stepTime(TP=1)`

**BC-2 (trained-physics GQA):**
GIVEN two `TrainedPhysicsModel` instances with identical configs except `NumKVHeads`
WHEN one has `NumKVHeads = NumHeads` (MHA: full attention bandwidth) and the other has `NumKVHeads = 8` (GQA: reduced KV bandwidth)
AND both run the same decode-heavy batch (`makeDecodeBatch(8, 512)`)
THEN `stepTime(GQA) < stepTime(MHA)`

**BC-3 (roofline GQA):**
GIVEN two calls to `rooflineStepTime` with identical hardware and step config
WHEN one uses `NumKVHeads = NumHeads` (MHA) and the other uses `NumKVHeads = 8` (GQA)
AND the step is decode-only (memory-bound on H100)
THEN `stepTime(GQA) < stepTime(MHA)`

---

## Tasks

### Task 1: BC-1 — trained-physics TP scaling test

**File:** `sim/latency/trained_physics_model_test.go`

**Test (write first, run to see it fail — it won't compile yet if needed, but here we're just adding a new test):**

```go
// BC-1: TP=2 reduces step time vs TP=1 for trained-physics
func TestTrainedPhysicsModel_TPScaling_TP2LessThanTP1(t *testing.T) {
    mc := trainedPhysicsTestModelConfig()
    hw := testHardwareConfig()
    coeffs := testCoeffs()

    hwTP1 := sim.ModelHardwareConfig{
        Backend:     "trained-physics",
        TP:          1,
        ModelConfig: *mc,
        HWConfig:    hw,
    }
    hwTP2 := sim.ModelHardwareConfig{
        Backend:     "trained-physics",
        TP:          2,
        ModelConfig: *mc,
        HWConfig:    hw,
    }

    mTP1, err := NewTrainedPhysicsModel(*coeffs, hwTP1)
    require.NoError(t, err)
    mTP2, err := NewTrainedPhysicsModel(*coeffs, hwTP2)
    require.NoError(t, err)

    batch := append(makePrefillBatch(1, 128), makeDecodeBatch(4, 256)...)

    tp1Time := mTP1.StepTime(batch)
    tp2Time := mTP2.StepTime(batch)

    assert.Less(t, tp2Time, tp1Time,
        "TP=2 step time (%d µs) must be less than TP=1 (%d µs)", tp2Time, tp1Time)
    assert.Greater(t, tp2Time, int64(0), "TP=2 step time must be positive")
}
```

**Implement:** Add the function above to `sim/latency/trained_physics_model_test.go`.

**Verify passes:** `go test ./sim/latency/... -run TestTrainedPhysicsModel_TPScaling_TP2LessThanTP1 -v`

**Lint:** `golangci-lint run ./sim/latency/...`

**Commit:** `test(latency): BC-1 TP=2 reduces step time in trained-physics model`

---

### Task 2: BC-2 — trained-physics GQA test

**File:** `sim/latency/trained_physics_model_test.go`

**Test:**

```go
// BC-2: GQA (NumKVHeads < NumHeads) reduces step time vs MHA (NumKVHeads == NumHeads)
func TestTrainedPhysicsModel_GQA_ReducesKVBandwidth(t *testing.T) {
    hw := testHardwareConfig()
    coeffs := testCoeffs()

    // MHA: NumKVHeads == NumHeads (full attention bandwidth)
    mcMHA := trainedPhysicsTestModelConfig()
    mcMHA.NumKVHeads = mcMHA.NumHeads // 32

    // GQA: NumKVHeads = 8 (reduced KV bandwidth, 4x less than MHA)
    mcGQA := trainedPhysicsTestModelConfig()
    mcGQA.NumKVHeads = 8

    hwMHA := sim.ModelHardwareConfig{
        Backend: "trained-physics", TP: 1, ModelConfig: *mcMHA, HWConfig: hw,
    }
    hwGQA := sim.ModelHardwareConfig{
        Backend: "trained-physics", TP: 1, ModelConfig: *mcGQA, HWConfig: hw,
    }

    mMHA, err := NewTrainedPhysicsModel(*coeffs, hwMHA)
    require.NoError(t, err)
    mGQA, err := NewTrainedPhysicsModel(*coeffs, hwGQA)
    require.NoError(t, err)

    // Decode-heavy batch: long sequence history makes KV bandwidth the dominant term
    batch := makeDecodeBatch(8, 512)

    mhaTime := mMHA.StepTime(batch)
    gqaTime := mGQA.StepTime(batch)

    assert.Less(t, gqaTime, mhaTime,
        "GQA step time (%d µs) must be less than MHA (%d µs): fewer KV heads → lower bandwidth", gqaTime, mhaTime)
}
```

**Verify passes:** `go test ./sim/latency/... -run TestTrainedPhysicsModel_GQA_ReducesKVBandwidth -v`

**Lint:** `golangci-lint run ./sim/latency/...`

**Commit:** `test(latency): BC-2 GQA reduces step time in trained-physics model`

---

### Task 3: BC-3 — roofline GQA test

**File:** `sim/latency/roofline_test.go`

**Test:**

```go
// BC-3: GQA (NumKVHeads < NumHeads) reduces step time vs MHA for roofline backend
func TestRooflineStepTime_GQA_ReducesKVBandwidth(t *testing.T) {
    hc := testHardwareCalib()

    // MHA: NumKVHeads == NumHeads (full KV bandwidth)
    mcMHA := testModelConfig()
    mcMHA.NumKVHeads = mcMHA.NumHeads // 32

    // GQA: NumKVHeads = 8 (4x fewer KV heads → lower KV bandwidth)
    mcGQA := testModelConfig() // already has NumKVHeads: 8

    // Decode-only: memory-bound on H100, so KV bandwidth differences dominate
    step := StepConfig{
        DecodeRequests: []DecodeRequestConfig{
            {ProgressIndex: 512, NumNewDecodeTokens: 1},
        },
    }

    mhaTime := rooflineStepTime(mcMHA, hc, step, 1)
    gqaTime := rooflineStepTime(mcGQA, hc, step, 1)

    assert.Greater(t, mhaTime, int64(0), "MHA step time must be positive")
    assert.Greater(t, gqaTime, int64(0), "GQA step time must be positive")
    assert.Less(t, gqaTime, mhaTime,
        "GQA step time (%d µs) must be less than MHA (%d µs): fewer KV heads → lower KV bandwidth", gqaTime, mhaTime)
}
```

**Verify passes:** `go test ./sim/latency/... -run TestRooflineStepTime_GQA_ReducesKVBandwidth -v`

**Lint:** `golangci-lint run ./sim/latency/...`

**Commit:** `test(latency): BC-3 GQA reduces step time in roofline model`

---

## Sanity Checklist

- [ ] `go test ./sim/latency/... -count=1` passes with all three new tests
- [ ] `golangci-lint run ./sim/latency/...` reports zero issues
- [ ] No production code modified
- [ ] All three tests are behavioral (assert observable step time differences, not internal struct fields)
- [ ] Each test uses existing helpers only (no new helpers needed)
- [ ] TP=1 config in Task 1 uses `NumHeads: 32` which is divisible by both TP=1 and TP=2 ✓
- [ ] GQA `NumKVHeads=8` is divisible by `TP=1` ✓

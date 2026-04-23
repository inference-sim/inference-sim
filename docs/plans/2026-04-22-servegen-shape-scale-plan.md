# ServeGen Shape/Scale Parameter Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable BLIS ServeGen converter to read and use MLE-fitted shape/scale parameters from trace columns 5-6, ensuring parity with ServeGen's arrival pattern generation.

**Architecture:** Extend ArrivalSpec with optional Shape/Scale fields. ServeGen converter populates these from trace data. NewArrivalSampler prioritizes explicit parameters over CV-derived ones. Backward compatible: CV-only specs continue working unchanged.

**Tech Stack:** Go 1.22+, gopkg.in/yaml.v3 (strict parsing)

---

## File Structure

**Files to modify:**
- `sim/workload/spec.go` - Add Shape/Scale fields to ArrivalSpec
- `sim/workload/servegen.go` - Parse columns 5-6, store in ArrivalSpec
- `sim/workload/arrival.go` - Prioritize explicit params in NewArrivalSampler
- `sim/workload/servegen_test.go` - Test ServeGen converter with high-CV traces
- `sim/workload/arrival_test.go` - Test sampler priority logic

---

## Task 1: Extend ArrivalSpec with Shape/Scale Fields

**Files:**
- Modify: `sim/workload/spec.go` (ArrivalSpec struct definition)

- [ ] **Step 1: Add Shape and Scale fields to ArrivalSpec**

Locate the `ArrivalSpec` struct (around line 50-55) and add two new optional fields:

```go
type ArrivalSpec struct {
	Process string   `yaml:"process"`
	CV      *float64 `yaml:"cv,omitempty"`

	// Optional MLE-fitted distribution parameters (ServeGen compatibility).
	// When present, these override CV-based derivation in NewArrivalSampler.
	// Only populated by `blis convert servegen` from trace columns 5-6.
	Shape   *float64 `yaml:"shape,omitempty"`  // Gamma α or Weibull k
	Scale   *float64 `yaml:"scale,omitempty"`  // Gamma θ or Weibull λ (in microseconds)
}
```

- [ ] **Step 2: Verify code compiles**

Run: `go build ./sim/workload/...`
Expected: SUCCESS (no compilation errors)

- [ ] **Step 3: Commit**

```bash
git add sim/workload/spec.go
git commit -m "feat(workload): add optional Shape/Scale fields to ArrivalSpec

Extends ArrivalSpec to support MLE-fitted distribution parameters from
ServeGen traces. Fields are optional and only populated by ServeGen
converter. Backward compatible with existing CV-only specs.

Part of #1112"
```

---

## Task 2: Parse Shape/Scale from ServeGen Traces

**Files:**
- Modify: `sim/workload/servegen.go` (serveGenTraceRow struct, parseServeGenTrace function)

- [ ] **Step 1: Write failing test for 6-column parsing**

Add to `sim/workload/servegen_test.go`:

```go
func TestParseServeGenTrace_WithShapeScale(t *testing.T) {
	// GIVEN a trace CSV with 6 columns including shape/scale
	tmpfile, err := os.CreateTemp("", "trace-*.csv")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpfile.Name())

	// Write test data with high CV and fitted shape/scale
	content := "199200,22.46,173.81,Weibull,0.0575,0.000573\n"
	if _, err := tmpfile.Write([]byte(content)); err != nil {
		t.Fatal(err)
	}
	tmpfile.Close()

	// WHEN parsing the trace
	rows, err := parseServeGenTrace(tmpfile.Name())

	// THEN shape and scale are parsed correctly
	if err != nil {
		t.Fatalf("parseServeGenTrace failed: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}
	row := rows[0]
	if row.cv != 173.81 {
		t.Errorf("expected cv=173.81, got %f", row.cv)
	}
	if row.shapeParam != 0.0575 {
		t.Errorf("expected shapeParam=0.0575, got %f", row.shapeParam)
	}
	if row.scaleParam != 0.000573 {
		t.Errorf("expected scaleParam=0.000573, got %f", row.scaleParam)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestParseServeGenTrace_WithShapeScale -v`
Expected: FAIL (serveGenTraceRow has no shapeParam/scaleParam fields)

- [ ] **Step 3: Add shapeParam/scaleParam to serveGenTraceRow**

Locate `serveGenTraceRow` struct (around line 72-79) and add two fields:

```go
type serveGenTraceRow struct {
	startTimeSec float64
	rate         float64
	cv           float64
	pattern      string // "Gamma", "Weibull", or empty
	shapeParam   float64
	scaleParam   float64
}
```

- [ ] **Step 4: Parse columns 5-6 in parseServeGenTrace**

In `parseServeGenTrace` function (around line 191-241), after parsing the pattern (line 228), add parsing for columns 5-6:

```go
		pattern := strings.TrimSpace(record[3])

		// NEW: Parse shape and scale parameters (columns 5-6)
		var shapeParam, scaleParam float64
		if len(record) >= 6 {
			shape, err := strconv.ParseFloat(strings.TrimSpace(record[4]), 64)
			if err != nil {
				skippedRows++
				continue
			}
			scale, err := strconv.ParseFloat(strings.TrimSpace(record[5]), 64)
			if err != nil {
				skippedRows++
				continue
			}
			shapeParam = shape
			scaleParam = scale
		}

		rows = append(rows, serveGenTraceRow{
			startTimeSec: startTime,
			rate:         rate,
			cv:           cv,
			pattern:      pattern,
			shapeParam:   shapeParam,
			scaleParam:   scaleParam,
		})
```

- [ ] **Step 5: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestParseServeGenTrace_WithShapeScale -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sim/workload/servegen.go sim/workload/servegen_test.go
git commit -m "feat(workload): parse shape/scale from ServeGen trace columns 5-6

Extends parseServeGenTrace to read MLE-fitted distribution parameters.
Handles traces with 4 or 6 columns (backward compatible with existing
test data that may have only 4 columns).

Part of #1112"
```

---

## Task 3: Store Shape/Scale in ArrivalSpec

**Files:**
- Modify: `sim/workload/servegen.go` (loadServeGenChunk function)
- Modify: `sim/workload/servegen_test.go` (add test for ArrivalSpec population)

- [ ] **Step 1: Write failing test for ArrivalSpec with shape/scale**

Add to `sim/workload/servegen_test.go`:

```go
func TestLoadServeGenChunk_PopulatesShapeScale(t *testing.T) {
	// GIVEN a ServeGen chunk with high CV and fitted parameters
	traceDir, err := os.MkdirTemp("", "servegen-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(traceDir)

	// Write trace file
	tracePath := filepath.Join(traceDir, "chunk-0-trace.csv")
	traceContent := "0,22.46,173.81,Weibull,0.0575,0.000573\n"
	if err := os.WriteFile(tracePath, []byte(traceContent), 0644); err != nil {
		t.Fatal(err)
	}

	// Write dataset file
	datasetPath := filepath.Join(traceDir, "chunk-0-dataset.json")
	datasetContent := `{"0": {"input_tokens": "{256: 1.0}", "output_tokens": "{100: 1.0}"}}`
	if err := os.WriteFile(datasetPath, []byte(datasetContent), 0644); err != nil {
		t.Fatal(err)
	}

	sgConfig := &ServeGenDataSpec{}

	// WHEN loading the chunk
	client, err := loadServeGenChunk("0", tracePath, datasetPath, sgConfig)

	// THEN ArrivalSpec contains shape and scale
	if err != nil {
		t.Fatalf("loadServeGenChunk failed: %v", err)
	}
	if client == nil {
		t.Fatal("expected non-nil client")
	}
	if client.Arrival.Process != "weibull" {
		t.Errorf("expected process=weibull, got %s", client.Arrival.Process)
	}
	if client.Arrival.CV == nil || *client.Arrival.CV != 173.81 {
		t.Errorf("expected cv=173.81, got %v", client.Arrival.CV)
	}
	if client.Arrival.Shape == nil || *client.Arrival.Shape != 0.0575 {
		t.Errorf("expected shape=0.0575, got %v", client.Arrival.Shape)
	}
	if client.Arrival.Scale == nil || *client.Arrival.Scale != 0.000573 {
		t.Errorf("expected scale=0.000573, got %v", client.Arrival.Scale)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestLoadServeGenChunk_PopulatesShapeScale -v`
Expected: FAIL (ArrivalSpec.Shape and Scale are nil)

- [ ] **Step 3: Store shape/scale in loadServeGenChunk**

In `loadServeGenChunk` function (around line 123-181), after setting `arrivalSpec.CV` (line 161), add:

```go
		cv := bestRow.cv
		arrivalSpec.CV = &cv
		// NEW: Store MLE-fitted parameters
		shape := bestRow.shapeParam
		scale := bestRow.scaleParam
		arrivalSpec.Shape = &shape
		arrivalSpec.Scale = &scale
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestLoadServeGenChunk_PopulatesShapeScale -v`
Expected: PASS

- [ ] **Step 5: Run all ServeGen tests**

Run: `go test ./sim/workload/... -run ServeGen -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add sim/workload/servegen.go sim/workload/servegen_test.go
git commit -m "feat(workload): store shape/scale in ArrivalSpec from ServeGen

loadServeGenChunk now populates Shape and Scale fields when converting
ServeGen traces. These will be used by NewArrivalSampler to match
ServeGen's arrival pattern generation exactly.

Part of #1112"
```

---

## Task 4: Prioritize Explicit Parameters in NewArrivalSampler

**Files:**
- Modify: `sim/workload/arrival.go` (NewArrivalSampler function)
- Modify: `sim/workload/arrival_test.go` (add priority tests)

- [ ] **Step 1: Write failing test for Gamma with explicit shape/scale**

Add to `sim/workload/arrival_test.go`:

```go
func TestNewArrivalSampler_GammaExplicitParams(t *testing.T) {
	// GIVEN an ArrivalSpec with explicit shape/scale (ServeGen-style)
	shape := 0.5
	scale := 0.04
	spec := ArrivalSpec{
		Process: "gamma",
		Shape:   &shape,
		Scale:   &scale,
	}

	// WHEN creating a sampler
	sampler := NewArrivalSampler(spec, 0.00001) // 10 req/s

	// THEN sampler uses explicit parameters
	gammaSampler, ok := sampler.(*GammaSampler)
	if !ok {
		t.Fatalf("expected *GammaSampler, got %T", sampler)
	}
	if gammaSampler.shape != 0.5 {
		t.Errorf("expected shape=0.5, got %f", gammaSampler.shape)
	}
	if gammaSampler.scale != 0.04 {
		t.Errorf("expected scale=0.04, got %f", gammaSampler.scale)
	}
}

func TestNewArrivalSampler_WeibullExplicitParams(t *testing.T) {
	// GIVEN an ArrivalSpec with explicit shape/scale (ServeGen high-CV)
	shape := 0.0575
	scale := 0.000573
	spec := ArrivalSpec{
		Process: "weibull",
		Shape:   &shape,
		Scale:   &scale,
	}

	// WHEN creating a sampler
	sampler := NewArrivalSampler(spec, 0.00001) // 10 req/s

	// THEN sampler uses explicit parameters
	weibullSampler, ok := sampler.(*WeibullSampler)
	if !ok {
		t.Fatalf("expected *WeibullSampler, got %T", sampler)
	}
	if weibullSampler.shape != 0.0575 {
		t.Errorf("expected shape=0.0575, got %f", weibullSampler.shape)
	}
	if weibullSampler.scale != 0.000573 {
		t.Errorf("expected scale=0.000573, got %f", weibullSampler.scale)
	}
}

func TestNewArrivalSampler_GammaCVFallback(t *testing.T) {
	// GIVEN an ArrivalSpec with CV but no explicit params (existing behavior)
	cv := 2.5
	spec := ArrivalSpec{
		Process: "gamma",
		CV:      &cv,
	}

	// WHEN creating a sampler
	sampler := NewArrivalSampler(spec, 0.00001) // 10 req/s

	// THEN sampler derives shape/scale from CV
	gammaSampler, ok := sampler.(*GammaSampler)
	if !ok {
		t.Fatalf("expected *GammaSampler, got %T", sampler)
	}
	expectedShape := 1.0 / (2.5 * 2.5) // 0.16
	if gammaSampler.shape < 0.15 || gammaSampler.shape > 0.17 {
		t.Errorf("expected shape≈0.16 (from CV), got %f", gammaSampler.shape)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./sim/workload/... -run TestNewArrivalSampler_.*ExplicitParams -v`
Expected: FAIL (samplers don't check for explicit params yet)

- [ ] **Step 3: Add priority logic to Gamma case**

In `NewArrivalSampler` function (around line 212-266), modify the `case "gamma":` block:

```go
	case "gamma":
		// Priority 1: Use explicit MLE-fitted parameters if provided (ServeGen)
		if spec.Shape != nil && spec.Scale != nil {
			return &GammaSampler{shape: *spec.Shape, scale: *spec.Scale}
		}
		// Priority 2: Derive from CV (existing logic)
		cv := 1.0
		if spec.CV != nil {
			cv = *spec.CV
		}
		if cv <= 0 {
			cv = 1.0
		}
		// shape = 1/CV², scale = mean * CV² = (1/rate) * CV²
		shape := 1.0 / (cv * cv)
		mean := 1.0 / ratePerMicrosecond
		scale := mean * cv * cv
		if shape < 0.01 {
			logrus.Warnf("Gamma shape %.4f (CV=%.1f) is very small; falling back to Poisson", shape, cv)
			return &PoissonSampler{rateMicros: ratePerMicrosecond}
		}
		return &GammaSampler{shape: shape, scale: scale}
```

- [ ] **Step 4: Add priority logic to Weibull case**

In the same function, modify the `case "weibull":` block:

```go
	case "weibull":
		// Priority 1: Use explicit MLE-fitted parameters if provided (ServeGen)
		if spec.Shape != nil && spec.Scale != nil {
			return &WeibullSampler{shape: *spec.Shape, scale: *spec.Scale}
		}
		// Priority 2: Derive from CV (existing logic)
		cv := 1.0
		if spec.CV != nil {
			cv = *spec.CV
		}
		if cv <= 0 {
			cv = 1.0
		}
		mean := 1.0 / ratePerMicrosecond
		k := weibullShapeFromCV(cv)
		// scale = mean / Γ(1 + 1/k)
		scale := mean / math.Gamma(1.0+1.0/k)
		return &WeibullSampler{shape: k, scale: scale}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `go test ./sim/workload/... -run TestNewArrivalSampler -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add sim/workload/arrival.go sim/workload/arrival_test.go
git commit -m "feat(workload): prioritize explicit shape/scale in NewArrivalSampler

When ArrivalSpec contains Shape and Scale fields, use them directly
instead of deriving from CV. This ensures ServeGen parity for traces
with MLE-fitted parameters. Backward compatible: CV-only specs continue
using existing derivation logic.

Part of #1112"
```

---

## Task 5: Integration Test with Real ServeGen Data

**Files:**
- Modify: `sim/workload/servegen_test.go` (add end-to-end test)

- [ ] **Step 1: Write integration test**

Add to `sim/workload/servegen_test.go`:

```go
func TestServeGenConversion_HighCVTrace(t *testing.T) {
	// GIVEN a ServeGen trace with CV > 10.4 (previously failed validation)
	traceDir, err := os.MkdirTemp("", "servegen-highcv-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(traceDir)

	// Create chunk-0-trace.csv with high CV
	tracePath := filepath.Join(traceDir, "chunk-0-trace.csv")
	traceContent := "0,22.46,173.81,Weibull,0.0575,0.000573\n"
	if err := os.WriteFile(tracePath, []byte(traceContent), 0644); err != nil {
		t.Fatal(err)
	}

	// Create chunk-0-dataset.json
	datasetPath := filepath.Join(traceDir, "chunk-0-dataset.json")
	datasetContent := `{"0": {"input_tokens": "{256: 0.5, 512: 0.5}", "output_tokens": "{100: 1.0}"}}`
	if err := os.WriteFile(datasetPath, []byte(datasetContent), 0644); err != nil {
		t.Fatal(err)
	}

	// WHEN converting via loadServeGenData
	spec := &WorkloadSpec{
		ServeGenData: &ServeGenDataSpec{Path: traceDir},
	}
	err = loadServeGenData(spec)

	// THEN conversion succeeds
	if err != nil {
		t.Fatalf("loadServeGenData failed: %v", err)
	}
	if len(spec.Clients) != 1 {
		t.Fatalf("expected 1 client, got %d", len(spec.Clients))
	}

	client := spec.Clients[0]
	if client.Arrival.Process != "weibull" {
		t.Errorf("expected weibull, got %s", client.Arrival.Process)
	}

	// THEN Shape and Scale are present (not CV-derived)
	if client.Arrival.Shape == nil {
		t.Error("expected Shape to be set")
	}
	if client.Arrival.Scale == nil {
		t.Error("expected Scale to be set")
	}

	// THEN validation passes (no CV bounds error)
	if err := validateClientSpec(&client, 0); err != nil {
		t.Errorf("validation failed: %v", err)
	}

	// THEN sampler can be created without panic
	sampler := NewArrivalSampler(client.Arrival, 0.00001)
	if sampler == nil {
		t.Error("expected non-nil sampler")
	}

	// THEN sampler uses explicit parameters
	weibullSampler, ok := sampler.(*WeibullSampler)
	if !ok {
		t.Fatalf("expected *WeibullSampler, got %T", sampler)
	}
	if weibullSampler.shape != 0.0575 {
		t.Errorf("expected shape=0.0575, got %f", weibullSampler.shape)
	}
}
```

- [ ] **Step 2: Run integration test**

Run: `go test ./sim/workload/... -run TestServeGenConversion_HighCVTrace -v`
Expected: PASS

- [ ] **Step 3: Run all workload tests**

Run: `go test ./sim/workload/... -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add sim/workload/servegen_test.go
git commit -m "test(workload): integration test for high-CV ServeGen traces

Verifies end-to-end conversion of ServeGen traces with CV > 10.4
succeeds and produces samplers with correct MLE-fitted parameters.

Closes #1112"
```

---

## Task 6: Verify with Actual ServeGen Data

**Files:**
- None (manual verification)

- [ ] **Step 1: Test conversion with real ServeGen data**

Run: `go build -o blis main.go && ./blis convert servegen --path ServeGen/data/language/m-mid 2>&1 | head -50`
Expected: SUCCESS (no validation errors, YAML output produced)

- [ ] **Step 2: Verify YAML contains shape/scale fields**

Run: `./blis convert servegen --path ServeGen/data/language/m-mid 2>&1 | grep -A5 "arrival:" | head -20`
Expected: Output shows `shape:` and `scale:` fields for converted clients

- [ ] **Step 3: Test run with converted spec**

```bash
./blis convert servegen --path ServeGen/data/language/m-mid > /tmp/servegen-converted.yaml
./blis run --model qwen/qwen3-14b --workload-spec /tmp/servegen-converted.yaml --num-requests 100
```
Expected: SUCCESS (simulation runs without errors)

- [ ] **Step 4: Document verification in commit message**

(No commit for this task - verification only)

---

## Task 7: Update Documentation

**Files:**
- Modify: `SERVEGEN_TO_BLIS_TRANSLATION.md`

- [ ] **Step 1: Update translation guide**

In `SERVEGEN_TO_BLIS_TRANSLATION.md`, find the section "Key Translation Decisions" (around line 105-110) and update:

```markdown
### Key Translation Decisions:

1. **Picks highest-rate window:** Scans all 2016 windows, selects rate=22.46
2. **Uses MLE-fitted parameters:** Reads shape/scale from columns 5-6 and stores in ArrivalSpec (NEW as of #1112)
3. **Converts PDFs:** Parses Python dict strings → YAML params map
4. **Single client per chunk:** Each chunk becomes one BLIS client
```

Update the "Key Differences" table (around line 247):

```markdown
| Aspect | ServeGen | BLIS |
|--------|----------|------|
| **Reads columns** | All 6 columns | All 6 columns (updated #1112) |
| **Shape/scale** | Uses exact fitted values (columns 5-6) | Uses exact fitted values when present (updated #1112) |
| **Normalization** | Post-hoc (generate → scale sum to target) | Pre-hoc (parameterize sampler with rate) |
| **Result** | Exact reproduction of fitted distribution | Exact reproduction when shape/scale present (updated #1112) |
```

- [ ] **Step 2: Commit documentation update**

```bash
git add SERVEGEN_TO_BLIS_TRANSLATION.md
git commit -m "docs: update ServeGen translation guide for shape/scale support

Documents that BLIS now reads columns 5-6 and uses MLE-fitted parameters,
achieving parity with ServeGen's arrival pattern generation.

Part of #1112"
```

---

## Task 8: Final Verification

**Files:**
- None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `go test ./...`
Expected: All tests PASS

- [ ] **Step 2: Run linter**

Run: `golangci-lint run ./...`
Expected: No errors

- [ ] **Step 3: Build binary**

Run: `go build -o blis main.go`
Expected: SUCCESS

- [ ] **Step 4: Test with issue reproduction case**

```bash
# From issue #1112, this command previously failed:
./blis convert servegen --path ServeGen/data/language/m-mid
```
Expected: SUCCESS (no validation errors about CV bounds)

---

## Self-Review Checklist

**Spec Coverage:**
✅ Parse columns 5-6 from ServeGen traces (Task 2)
✅ Store shape/scale in ArrivalSpec (Task 3)
✅ Use shape/scale in NewArrivalSampler (Task 4)
✅ Handle high-CV traces without validation errors (Task 5)
✅ Backward compatibility with CV-only specs (Task 4, explicit tests)

**Placeholder Scan:**
✅ No TBD, TODO, "implement later"
✅ All code blocks are complete
✅ All test expectations are explicit

**Type Consistency:**
✅ ArrivalSpec fields: Shape *float64, Scale *float64 (consistent across all tasks)
✅ serveGenTraceRow fields: shapeParam float64, scaleParam float64 (consistent)
✅ Sampler types: *GammaSampler, *WeibullSampler (consistent)

**Issues Found:** None

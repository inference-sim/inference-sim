package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// baselineNoopGolden is the pre-feature no-op stdout golden captured in T002
// (./blis run --model qwen/qwen3-14b --seed 42). Path is relative to the cmd/ test cwd.
const baselineNoopGolden = "../specs/007-lora-control-plane/testdata/baseline_noop.json"

// noopFloatTolerance is the relative tolerance applied when comparing numeric
// metric fields against the golden. The golden was captured on one architecture
// (arm64) while CI runs on another (amd64); interpolated percentiles differ in
// the last ULP of their float64 representation (~1e-11 relative) purely from
// platform-dependent floating-point op ordering. That is NOT a behavioral change
// and NOT an INV-6 violation — INV-6 guarantees byte-identity across RUNS on a
// single platform, which still holds. Requiring byte-identity across
// ARCHITECTURES for float-derived JSON is unattainable and was never an
// invariant, so numeric fields are compared within tolerance while every
// non-numeric field (and the output structure) is still compared exactly.
const noopFloatTolerance = 1e-9

// TestNoOpByteIdentity_AdapterBlindRunMatchesBaseline is the load-bearing INV-6 / SC-001
// regression: an adapter-blind run (no --lora-config, no --lora-* flags) MUST produce
// stdout that matches the pre-feature baseline. This proves the LoRA subsystem is
// inert when unconfigured.
//
// The comparison is structural + numeric-tolerant rather than raw byte-identity: the
// non-JSON preamble and every non-numeric JSON field must match exactly, while numeric
// fields must match within noopFloatTolerance. This tolerates cross-architecture float64
// last-ULP drift (the golden is captured on one arch, CI runs on another) without
// weakening the inertness guarantee — any real behavioral change from the LoRA code
// path shifts a value far beyond the tolerance or changes a field's shape/name.
//
// The run is driven in a re-exec subprocess so the real cobra command tree executes
// (and os.Exit(0) suppresses the test framework's own stdout, leaving only the metrics
// JSON for a clean comparison). The qwen3-14b model config and hardware config are
// git-tracked under model_configs/ and hardware_config.json, so the run is offline-safe.
func TestNoOpByteIdentity_AdapterBlindRunMatchesBaseline(t *testing.T) {
	if os.Getenv("BLIS_NOOP_SUBPROCESS") == "1" {
		rootCmd.SetArgs([]string{
			"run", "--model", "qwen/qwen3-14b", "--seed", "42",
			"--defaults-filepath", "../defaults.yaml",
		})
		_ = rootCmd.Execute()
		os.Exit(0)
	}

	golden, err := os.ReadFile(baselineNoopGolden)
	if err != nil {
		t.Fatalf("read golden baseline: %v", err)
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestNoOpByteIdentity_AdapterBlindRunMatchesBaseline")
	cmd.Env = append(os.Environ(), "BLIS_NOOP_SUBPROCESS=1")
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = io.Discard // logrus diagnostics go to stderr (INV-6: not part of deterministic output)
	if err := cmd.Run(); err != nil {
		t.Fatalf("subprocess run failed: %v\nstdout:\n%s", err, stdout.String())
	}

	gotPreamble, gotJSON, err := splitMetricsOutput(stdout.String())
	if err != nil {
		t.Fatalf("parse adapter-blind stdout: %v\nstdout:\n%s", err, stdout.String())
	}
	wantPreamble, wantJSON, err := splitMetricsOutput(string(golden))
	if err != nil {
		t.Fatalf("parse golden baseline: %v", err)
	}

	// The preamble (everything before the JSON object) must match exactly — it
	// carries no floating-point content, so any difference is a real regression.
	if gotPreamble != wantPreamble {
		t.Errorf("INV-6 VIOLATION: adapter-blind run preamble differs from baseline.\n"+
			"--- got ---\n%q\n--- want ---\n%q", gotPreamble, wantPreamble)
	}

	// The metrics object must match structurally, with numeric fields compared
	// within tolerance to absorb cross-architecture float64 last-ULP drift.
	if diff := compareMetricsJSON(gotJSON, wantJSON, noopFloatTolerance); diff != "" {
		t.Errorf("INV-6 VIOLATION: adapter-blind run metrics differ from pre-feature baseline "+
			"beyond float tolerance (%g).\n%s\n--- got ---\n%s\n--- want (golden) ---\n%s",
			noopFloatTolerance, diff, gotJSON, wantJSON)
	}
}

// splitMetricsOutput separates the "=== Simulation Metrics ===" preamble from the
// pretty-printed JSON metrics object. It returns the preamble (text before the first
// '{'), the JSON substring (first '{' through the matching final '}'), and an error
// if no JSON object is present.
func splitMetricsOutput(out string) (preamble, jsonBody string, err error) {
	start := strings.IndexByte(out, '{')
	end := strings.LastIndexByte(out, '}')
	if start < 0 || end < start {
		return "", "", fmt.Errorf("no JSON object found in output")
	}
	return out[:start], out[start : end+1], nil
}

// compareMetricsJSON decodes both JSON documents and compares them recursively.
// Numeric leaves must agree within relTol (relative tolerance); every other leaf
// (strings, bools, null) and the overall structure (keys, array lengths) must match
// exactly. It returns "" when the documents match, or a human-readable description of
// the first mismatch otherwise. Numbers are decoded via json.Number so the raw
// representation is available for exact string equality before falling back to a
// tolerant float comparison.
func compareMetricsJSON(got, want string, relTol float64) string {
	var gotVal, wantVal interface{}
	if err := decodeJSONNumbers(got, &gotVal); err != nil {
		return fmt.Sprintf("got is not valid JSON: %v", err)
	}
	if err := decodeJSONNumbers(want, &wantVal); err != nil {
		return fmt.Sprintf("golden is not valid JSON: %v", err)
	}
	return diffJSONValues("", gotVal, wantVal, relTol)
}

func decodeJSONNumbers(s string, v interface{}) error {
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	return dec.Decode(v)
}

// diffJSONValues recursively compares two decoded JSON values. `path` is the JSON
// pointer-ish location used in mismatch messages.
func diffJSONValues(path string, got, want interface{}, relTol float64) string {
	switch wantTyped := want.(type) {
	case map[string]interface{}:
		gotMap, ok := got.(map[string]interface{})
		if !ok {
			return fmt.Sprintf("%s: type mismatch (got %T, want object)", pathOrRoot(path), got)
		}
		if len(gotMap) != len(wantTyped) {
			return fmt.Sprintf("%s: key count mismatch (got %d, want %d)", pathOrRoot(path), len(gotMap), len(wantTyped))
		}
		for k, wv := range wantTyped {
			gv, present := gotMap[k]
			if !present {
				return fmt.Sprintf("%s: missing key %q", pathOrRoot(path), k)
			}
			if d := diffJSONValues(path+"."+k, gv, wv, relTol); d != "" {
				return d
			}
		}
		return ""
	case []interface{}:
		gotSlice, ok := got.([]interface{})
		if !ok {
			return fmt.Sprintf("%s: type mismatch (got %T, want array)", pathOrRoot(path), got)
		}
		if len(gotSlice) != len(wantTyped) {
			return fmt.Sprintf("%s: array length mismatch (got %d, want %d)", pathOrRoot(path), len(gotSlice), len(wantTyped))
		}
		for i := range wantTyped {
			if d := diffJSONValues(fmt.Sprintf("%s[%d]", path, i), gotSlice[i], wantTyped[i], relTol); d != "" {
				return d
			}
		}
		return ""
	case json.Number:
		gotNum, ok := got.(json.Number)
		if !ok {
			return fmt.Sprintf("%s: type mismatch (got %T, want number)", pathOrRoot(path), got)
		}
		// Exact string match is the common case (integers, terminating decimals).
		if string(gotNum) == string(wantTyped) {
			return ""
		}
		gotF, gErr := gotNum.Float64()
		wantF, wErr := wantTyped.Float64()
		if gErr != nil || wErr != nil {
			return fmt.Sprintf("%s: non-parseable number (got %q, want %q)", pathOrRoot(path), gotNum, wantTyped)
		}
		if !floatsWithinTolerance(gotF, wantF, relTol) {
			return fmt.Sprintf("%s: number mismatch beyond tolerance (got %s, want %s)", pathOrRoot(path), gotNum, wantTyped)
		}
		return ""
	default:
		// Strings, bools, null: require exact equality.
		if got != want {
			return fmt.Sprintf("%s: value mismatch (got %v, want %v)", pathOrRoot(path), got, want)
		}
		return ""
	}
}

func pathOrRoot(path string) string {
	if path == "" {
		return "(root)"
	}
	return strings.TrimPrefix(path, ".")
}

// floatsWithinTolerance reports whether a and b agree within relTol relative tolerance,
// with an absolute-tolerance fallback so values near zero compare sensibly.
func floatsWithinTolerance(a, b, relTol float64) bool {
	if a == b {
		return true
	}
	diff := math.Abs(a - b)
	scale := math.Max(math.Abs(a), math.Abs(b))
	if scale == 0 {
		return diff <= relTol
	}
	return diff/scale <= relTol
}

// TestCompareMetricsJSON_ToleratesDriftCatchesRegressions is the companion invariant
// test for the byte-identity guard: it verifies the comparator's central law —
// cross-architecture float last-ULP drift is accepted, while any real behavioral
// change (numeric shift beyond tolerance, changed string, or altered structure) is
// rejected. The drift row uses the exact arm64-golden vs amd64-CI values that
// triggered the original failure.
func TestCompareMetricsJSON_ToleratesDriftCatchesRegressions(t *testing.T) {
	tests := []struct {
		name      string
		got       string
		want      string
		wantMatch bool // true => comparator returns "" (documents match)
	}{
		{
			name:      "identical",
			got:       `{"e2e_p90_ms": 11150.5137, "completed_requests": 100}`,
			want:      `{"e2e_p90_ms": 11150.5137, "completed_requests": 100}`,
			wantMatch: true,
		},
		{
			// The exact values from the failing CI run vs the arm64 golden.
			name:      "cross-arch ULP drift is tolerated",
			got:       `{"e2e_p90_ms": 11150.513700000001, "e2e_p99_ms": 15622.560570000003, "itl_p99_ms": 13.742359999999724}`,
			want:      `{"e2e_p90_ms": 11150.5137, "e2e_p99_ms": 15622.56057, "itl_p99_ms": 13.742359999999962}`,
			wantMatch: true,
		},
		{
			name:      "real numeric regression is caught",
			got:       `{"e2e_p90_ms": 11200.0}`,
			want:      `{"e2e_p90_ms": 11150.5137}`,
			wantMatch: false,
		},
		{
			name:      "integer count change is caught",
			got:       `{"completed_requests": 99}`,
			want:      `{"completed_requests": 100}`,
			wantMatch: false,
		},
		{
			name:      "changed string field is caught",
			got:       `{"instance_id": "instance-0"}`,
			want:      `{"instance_id": "cluster"}`,
			wantMatch: false,
		},
		{
			name:      "renamed key is caught",
			got:       `{"e2e_p90_ms": 11150.5137}`,
			want:      `{"e2e_mean_ms": 11150.5137}`,
			wantMatch: false,
		},
		{
			name:      "extra key is caught",
			got:       `{"a": 1, "b": 2}`,
			want:      `{"a": 1}`,
			wantMatch: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			diff := compareMetricsJSON(tc.got, tc.want, noopFloatTolerance)
			if tc.wantMatch && diff != "" {
				t.Errorf("expected match, got diff: %s", diff)
			}
			if !tc.wantMatch && diff == "" {
				t.Errorf("expected mismatch to be reported, got match")
			}
		})
	}
}

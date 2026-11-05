package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeSet writes one coefficient set under <dir>/coefficients/<name>, creating any
// intermediate directories so name may be nested (e.g. "sub/deep.yaml").
func writeSet(t *testing.T, dir, name, body string) {
	t.Helper()
	path := filepath.Join(dir, "coefficients", name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// A well-formed set validates: exit 0, and the summary names it on stdout.
const validSet = `
kind: CoefficientSet
name: cost-model-primitives-h200
coefficients:
  - gemm_eps_max_bf16:
      value: 0.72
      units: dimensionless
      method: measured
      fitted: true
      scope:
        hardware: [h200]
      sources:
        - {kind: model, cite: "operator table", role: primary}
`

func TestRunValidSetPasses(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "primitives.yaml", validSet)

	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr:\n%s", code, errBuf.String())
	}
	if !strings.Contains(out.String(), "all 1 set(s) validate") {
		t.Errorf("stdout missing the summary line:\n%s", out.String())
	}
}

// A set with two entries sharing one name AND one scope is rejected: a resolver would
// keep whichever came last, so the file would mean something other than it says.
const duplicateSet = `
kind: CoefficientSet
name: dup
coefficients:
  - gemm_eps_max_bf16:
      value: 0.72
      units: dimensionless
      method: measured
      fitted: false
      scope:
        hardware: [h200]
  - gemm_eps_max_bf16:
      value: 0.80
      units: dimensionless
      method: measured
      fitted: false
      scope:
        hardware: [h200]
`

// The command must surface a duplicate as a failure (exit 1) with the finding on stderr.
func TestRunDuplicateScopeFails(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "dup.yaml", duplicateSet)

	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 1 {
		t.Fatalf("exit = %d, want 1; stderr:\n%s", code, errBuf.String())
	}
	if !strings.Contains(errBuf.String(), "duplicate entry name") {
		t.Errorf("stderr did not report the duplicate:\n%s", errBuf.String())
	}
}

// A directory holding both passing and failing sets must process all of them, not stop
// at the first failure: exit 1, the valid set still summarized on stdout, the bad one
// marked FAILED on stdout, and the aggregate tally on stderr. This is the multi-file
// aggregation the per-PR CI gate depends on.
func TestRunMixedPassAndFailAggregates(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "good.yaml", validSet)
	writeSet(t, dir, "dup.yaml", duplicateSet)

	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 1 {
		t.Fatalf("exit = %d, want 1; stderr:\n%s", code, errBuf.String())
	}
	so := out.String()
	if !strings.Contains(so, "good.yaml") || !strings.Contains(so, "coefficient(s)") {
		t.Errorf("stdout missing the passing set's summary row:\n%s", so)
	}
	if !strings.Contains(so, "dup.yaml") || !strings.Contains(so, "FAILED") {
		t.Errorf("stdout missing the failing set's row:\n%s", so)
	}
	if !strings.Contains(errBuf.String(), "1 of 2 set(s) failed") {
		t.Errorf("stderr missing the aggregate tally:\n%s", errBuf.String())
	}
}

// A set that fails to parse — here an unknown key inside an entry, which the strict
// loader rejects — is a load failure, also exit 1.
func TestRunUnparseableSetFails(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "bad.yaml", `
kind: CoefficientSet
name: bad
coefficients:
  - x:
      value: 1
      units: dimensionless
      method: measured
      bogus_key: 1
      scope:
        hardware: [h200]
`)

	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 1 {
		t.Fatalf("exit = %d, want 1; stderr:\n%s", code, errBuf.String())
	}
}

// prefixLines attributes every line of a multi-line diagnostic and normalizes CRLF, so a
// decoder error that spans lines does not leave later lines unprefixed or trailing a
// carriage return.
func TestPrefixLinesNormalizesCRLF(t *testing.T) {
	got := prefixLines("c.yaml", "first\r\nsecond\r\n")
	want := "c.yaml: first\nc.yaml: second\n"
	if got != want {
		t.Errorf("prefixLines = %q, want %q", got, want)
	}
}

// A non-finite coefficient value is rejected: the schema's FiniteField check runs inside
// (*coefficient.Set).Validate, so the command enforces it for free — the point of
// validating through the same schema a consumer loads these sets with.
func TestRunRejectsNonFiniteValue(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "nan.yaml", `
kind: CoefficientSet
name: nan
coefficients:
  - x:
      value: .nan
      units: dimensionless
      method: measured
      fitted: false
      scope:
        hardware: [h200]
`)

	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 1 {
		t.Fatalf("exit = %d, want 1; stderr:\n%s", code, errBuf.String())
	}
	if !strings.Contains(errBuf.String(), "finite number") {
		t.Errorf("stderr did not report the non-finite value:\n%s", errBuf.String())
	}
}

func TestRunUsageOnWrongArgs(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := run(nil, &out, &errBuf); code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(errBuf.String(), "usage:") {
		t.Errorf("stderr missing usage:\n%s", errBuf.String())
	}
}

// A root that exists but holds no coefficient sets is misuse, not success: exit 2 naming
// the path, so a mistyped-but-present root cannot pass as "nothing to check".
func TestRunNoSetsIsMisuse(t *testing.T) {
	dir := t.TempDir()
	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 2 {
		t.Fatalf("exit = %d, want 2; stderr:\n%s", code, errBuf.String())
	}
	if !strings.Contains(errBuf.String(), "no coefficient sets") {
		t.Errorf("stderr did not name the missing path:\n%s", errBuf.String())
	}
}

// A nonexistent root is diagnosed as a bad root, not as an empty registry — so a typo in
// the path is distinguishable from a real registry with nothing to validate.
func TestRunNonexistentRootIsMisuse(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := run([]string{filepath.Join(t.TempDir(), "nope")}, &out, &errBuf); code != 2 {
		t.Fatalf("exit = %d, want 2; stderr:\n%s", code, errBuf.String())
	}
	if !strings.Contains(errBuf.String(), "does not exist or is not a directory") {
		t.Errorf("stderr did not diagnose the bad root:\n%s", errBuf.String())
	}
}

// A .yaml file that is not a well-formed coefficient set is a load failure (exit 1), not a
// crash: malformed content is reported and the run continues.
func TestRunMalformedYamlFails(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "garbage.yaml", "not a coefficient set, just a scalar\n")

	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 1 {
		t.Fatalf("exit = %d, want 1; stderr:\n%s", code, errBuf.String())
	}
	if !strings.Contains(out.String(), "FAILED") {
		t.Errorf("stdout should mark the malformed set FAILED:\n%s", out.String())
	}
}

// A dangling symlink matching the extension is discovered (its target cannot be resolved,
// so it falls through under its own name) and surfaces as a load failure, exit 1 — not a
// silent skip.
func TestRunDanglingSymlinkFails(t *testing.T) {
	dir := t.TempDir()
	coeff := filepath.Join(dir, "coefficients")
	if err := os.MkdirAll(coeff, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(dir, "missing-target.yaml"), filepath.Join(coeff, "dangling.yaml")); err != nil {
		t.Skipf("symlinks unavailable on this filesystem: %v", err)
	}
	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 1 {
		t.Fatalf("exit = %d, want 1; stderr:\n%s", code, errBuf.String())
	}
	if !strings.Contains(out.String(), "dangling.yaml") || !strings.Contains(out.String(), "FAILED") {
		t.Errorf("the dangling set should be listed FAILED, not skipped:\n%s", out.String())
	}
}

// Discovery matches blis-registry's validator/validate.py: a recursive walk under
// coefficients/ taking both .yaml and .yml, case-insensitively. A .yml set, a set in a
// nested subdirectory, and an upper-case extension must all be found and validated —
// otherwise this gate could pass over a set the registry's own gate rejects, the exact
// two-gates-diverge failure the registry CI job (inference-sim/blis-registry#20) exists
// to prevent.
func TestRunDiscoversYmlNestedAndUppercase(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "flat.yaml", validSet)
	writeSet(t, dir, "alt.yml", validSet)
	writeSet(t, dir, filepath.Join("nested", "deep.yaml"), validSet)
	writeSet(t, dir, "UPPER.YAML", validSet)

	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr:\n%s", code, errBuf.String())
	}
	so := out.String()
	for _, want := range []string{"flat.yaml", "alt.yml", filepath.Join("nested", "deep.yaml"), "UPPER.YAML"} {
		if !strings.Contains(so, want) {
			t.Errorf("discovery missed %q:\n%s", want, so)
		}
	}
	if !strings.Contains(so, "all 4 set(s) validate") {
		t.Errorf("stdout summary wrong, want all 4 valid:\n%s", so)
	}
}

const validAcceptance = `
kind: AcceptanceProfileSet
name: inferencex-golden-al
profiles:
  - model: deepseek-v4.1-flash
    speculative_method: dspark
    num_spec_tokens: 5
    thinking: "off"
    category: coding
    acceptance_length: 4.07
    method: measured
    provenance:
      source: golden.yaml
      hardware: b300
      engine_image: vllm/vllm-openai:deepseekv41-flash-0909
      sampling: {temperature: 1.0}
`

func writeAcceptance(t *testing.T, dir, name, body string) {
	t.Helper()
	path := filepath.Join(dir, "acceptance", name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// Acceptance-profile sets under acceptance/ are validated beside the coefficient sets
// and counted in the summary; an invalid one fails the run.
func TestRunValidatesAcceptanceSets(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "primitives.yaml", validSet)
	writeAcceptance(t, dir, "inferencex-golden-al.yaml", validAcceptance)

	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr:\n%s", code, errBuf.String())
	}
	so := out.String()
	if !strings.Contains(so, filepath.Join("acceptance", "inferencex-golden-al.yaml")) ||
		!strings.Contains(so, "1 profile(s)") || !strings.Contains(so, "all 2 set(s) validate") {
		t.Errorf("stdout does not report the acceptance set:\n%s", so)
	}

	writeAcceptance(t, dir, "bad.yaml", strings.Replace(validAcceptance, "4.07", "6.5", 1))
	out.Reset()
	errBuf.Reset()
	if code := run([]string{dir}, &out, &errBuf); code != 1 {
		t.Fatalf("exit = %d, want 1 for an AL above K+1", code)
	}
	if !strings.Contains(errBuf.String(), "outside [1, 6]") || !strings.Contains(errBuf.String(), "1 of 3 set(s) failed") {
		t.Errorf("stderr:\n%s", errBuf.String())
	}
}

// Routing-profile sets under routing/ are validated beside the others; an invalid one fails.
func TestRunValidatesRoutingSets(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "primitives.yaml", validSet)
	body := `kind: RoutingProfileSet
name: r
role: sensitivity
profiles:
  - {model: m, category: text, phase: decode, placement: linear, expert_parallel: 8,
     method: measured, provenance: {source: s},
     points: [{batch_tokens: 64, token_imbalance: 1.6, expert_imbalance: 0.8}]}
`
	path := filepath.Join(dir, "routing", "r.yaml")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 0 || !strings.Contains(out.String(), "all 2 set(s) validate") {
		t.Fatalf("exit = %d; stdout:\n%s\nstderr:\n%s", code, out.String(), errBuf.String())
	}
	if err := os.WriteFile(path, []byte(strings.Replace(body, "1.6", "0.6", 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errBuf.Reset()
	if code := run([]string{dir}, &out, &errBuf); code != 1 {
		t.Fatalf("exit = %d, want 1 for a busiest rank below the mean", code)
	}
}

package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// exercise runs the validator against a root and returns its exit code and the two
// streams, so a test can assert both the scriptable code and the human-readable report.
func exercise(t *testing.T, root string) (int, string, string) {
	t.Helper()
	var out, errb bytes.Buffer
	code := run([]string{root}, &out, &errb)
	return code, out.String(), errb.String()
}

// catalogWith builds a throwaway catalog from a path->contents map and returns its root.
// An empty contents string creates the directory alone, which is how a test sets up a
// model entry that is missing its model.yaml.
func catalogWith(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for rel, body := range files {
		full := filepath.Join(root, rel)
		if body == "" {
			if err := os.MkdirAll(full, 0o700); err != nil {
				t.Fatalf("mkdir %s: %v", rel, err)
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(full), 0o700); err != nil {
			t.Fatalf("mkdir for %s: %v", rel, err)
		}
		if err := os.WriteFile(full, []byte(body), 0o600); err != nil {
			t.Fatalf("write %s: %v", rel, err)
		}
	}
	return root
}

// Shared minimal fixtures for tests that build a model entry and want to isolate one
// failure: a well-formed model.yaml and a non-empty config.json object, so a test that is
// not about those files does not also trip their required-presence checks.
const (
	validModelYAML = `name: tiny-llm
source:
  provider: huggingface
  repo: example/Tiny-LLM
  revision: deadbeef
`
	validConfigJSON = `{"hidden_size": 2048}
`
)

// TestRunValidCatalogValidatesEveryKind is the headline: the committed fixture catalog
// holds one of every artifact kind — a graph, a config.json, a model.yaml, a chip, a
// fabric, two storage tiers, a NESTED workload shape, and a licence registry with one
// measured run — and all of them must validate,
// with a zero exit and nothing on stderr. The workload fixture is deliberately the nested
// form; the live catalog's workloads are still flat (catalog #16), so this path is covered
// by a local fixture rather than the live tree.
func TestRunValidCatalogValidatesEveryKind(t *testing.T) {
	code, out, errb := exercise(t, filepath.Join("testdata", "catalog-ok"))
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstdout:\n%s\nstderr:\n%s", code, out, errb)
	}
	if errb != "" {
		t.Errorf("a clean catalog wrote to stderr:\n%s", errb)
	}
	// Every artifact kind must appear in the report, named, so a kind silently dropped
	// from the walk is caught here rather than by its absence going unnoticed. The two
	// model-dir files (config.json and model.yaml) and the graph all count, so a complete
	// entry contributes three artifacts.
	for _, want := range []string{
		"tiny-llm", "config.json", "h100", "ib-400g", "cpu_dram", "nvme_gen4", "chatbot",
		"measurements/sources.yaml", "EXACT/RESOLVED", "all 10 artifact(s) validate",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("report does not mention %q:\n%s", want, out)
		}
	}
}

// TestRunMissingArgument: the CLI takes exactly one root. No argument (or more than one)
// is a usage error, exit 2, distinct from a validation failure.
func TestRunMissingArgument(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run(nil, &out, &errb); code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(errb.String(), "usage") {
		t.Errorf("stderr should carry usage, got: %s", errb.String())
	}
}

// TestRunRootDoesNotExist: a mistyped path is a usage-level error (exit 2), not a report
// that zero artifacts validated.
func TestRunRootDoesNotExist(t *testing.T) {
	code, _, errb := exercise(t, filepath.Join(t.TempDir(), "nope"))
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(errb, "does not exist") {
		t.Errorf("stderr should explain the missing root, got: %s", errb)
	}
}

// TestRunNotACatalog: an existing directory with none of the catalog namespaces must fail
// loudly (exit 2) rather than report "all 0 artifacts validate" — the hole a mistyped but
// existing root would otherwise open.
func TestRunNotACatalog(t *testing.T) {
	code, _, errb := exercise(t, t.TempDir())
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(errb, "not a catalog") {
		t.Errorf("stderr should say it is not a catalog, got: %s", errb)
	}
}

// TestRunNamespacePresentButNoArtifacts: a directory that satisfies the namespace gate
// with only an empty (or optional) namespace — here an empty devices/ — validates nothing,
// and reporting "all 0 validate" with exit 0 would pass a non-catalog. A run that finds no
// artifact at all is a structural error (exit 2), not a success.
func TestRunNamespacePresentButNoArtifacts(t *testing.T) {
	root := catalogWith(t, map[string]string{
		"devices": "", // an empty devices/ directory: catalog-shaped, holds nothing
	})
	code, out, errb := exercise(t, root)
	if code != 2 {
		t.Fatalf("exit = %d, want 2\nstdout:\n%s\nstderr:\n%s", code, out, errb)
	}
	if !strings.Contains(errb, "no catalog artifacts") {
		t.Errorf("stderr should say nothing was found to validate, got: %s", errb)
	}
	if strings.Contains(out, "validate") {
		t.Errorf("stdout must not claim anything validated:\n%s", out)
	}
}

// TestRunModelNameMismatch: a model.yaml whose name disagrees with its directory is the
// identity failure the Python gate catches too; the Go gate must report it and exit 1.
func TestRunModelNameMismatch(t *testing.T) {
	root := catalogWith(t, map[string]string{
		"models/tiny-llm/config.json": validConfigJSON,
		"models/tiny-llm/model.yaml": `name: not-tiny
source:
  provider: huggingface
  repo: example/Tiny-LLM
  revision: deadbeef
`,
	})
	code, _, errb := exercise(t, root)
	if code != 1 {
		t.Fatalf("exit = %d, want 1\nstderr:\n%s", code, errb)
	}
	if !strings.Contains(errb, "not-tiny") || !strings.Contains(errb, "tiny-llm") {
		t.Errorf("stderr should name both the claimed name and the directory, got: %s", errb)
	}
}

// TestRunMissingModelYaml: a model directory with its config.json but no model.yaml is an
// incomplete entry; the walk must report the absent model.yaml and exit 1, mirroring
// validate_models. (config.json is present here so the failure isolates to model.yaml.)
func TestRunMissingModelYaml(t *testing.T) {
	root := catalogWith(t, map[string]string{
		"models/orphan/config.json": validConfigJSON,
	})
	code, _, errb := exercise(t, root)
	if code != 1 {
		t.Fatalf("exit = %d, want 1\nstderr:\n%s", code, errb)
	}
	if !strings.Contains(errb, "orphan") || !strings.Contains(errb, "model.yaml") {
		t.Errorf("stderr should report the missing model.yaml, got: %s", errb)
	}
}

// TestRunMissingConfigJson: a model directory with its model.yaml but no config.json is an
// incomplete entry too — the verbatim vendor file is required, mirroring validate_models —
// so the walk reports the absence and exits 1.
func TestRunMissingConfigJson(t *testing.T) {
	root := catalogWith(t, map[string]string{
		"models/tiny-llm/model.yaml": validModelYAML,
	})
	code, _, errb := exercise(t, root)
	if code != 1 {
		t.Fatalf("exit = %d, want 1\nstderr:\n%s", code, errb)
	}
	if !strings.Contains(errb, "config.json") || !strings.Contains(errb, "needs config.json") {
		t.Errorf("stderr should report the missing config.json, got: %s", errb)
	}
}

// TestRunMalformedConfigJson covers validate_models' structural config.json failure modes:
// an empty object and a valid-JSON-but-not-an-object both fail as "not a non-empty JSON
// object", and malformed JSON fails as a parse error. All exit 1.
func TestRunMalformedConfigJson(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"empty object", "{}\n", "non-empty JSON object"},
		{"valid JSON but not an object", "[1, 2, 3]\n", "non-empty JSON object"},
		{"malformed JSON", "{not json\n", "invalid JSON"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			root := catalogWith(t, map[string]string{
				"models/tiny-llm/model.yaml":  validModelYAML,
				"models/tiny-llm/config.json": c.body,
			})
			code, _, errb := exercise(t, root)
			if code != 1 {
				t.Fatalf("exit = %d, want 1\nstderr:\n%s", code, errb)
			}
			if !strings.Contains(errb, "config.json") || !strings.Contains(errb, c.want) {
				t.Errorf("stderr should report the config.json fault %q, got: %s", c.want, errb)
			}
		})
	}
}

// TestRunMalformedChip: a chip that loads but fails validation (a non-positive datasheet
// figure) must be reported with its field and exit 1 — the coverage this CLI adds over a
// graph-only gate.
func TestRunMalformedChip(t *testing.T) {
	root := catalogWith(t, map[string]string{
		"hardware/h100.yaml": `Provenance: vendor_spec
TFlopsPeak: 0
BwPeakTBs: 3.35
MemoryGiB: 80.0
IntraNodeBwGBps: 450
`,
	})
	code, _, errb := exercise(t, root)
	if code != 1 {
		t.Fatalf("exit = %d, want 1\nstderr:\n%s", code, errb)
	}
	if !strings.Contains(errb, "TFlopsPeak") {
		t.Errorf("stderr should name the offending field, got: %s", errb)
	}
}

// TestRunHardwareInvariants pins the schema-structural hardware/networks invariants
// restored in #32 — the dimensionless-field ban, required/positive SMCount, and the
// non-finite check on chips and fabrics — each a malformed edit the deleted blis-catalog
// validate_catalog.py rejected and the Go gate once silently accepted. (The fourth gap in
// #32, the SM-count citation, is a catalog-provenance policy deliberately not enforced
// here; see spec/hardware/hardware.go and the README.) Each case applies one mutation to
// an otherwise-clean chip or fabric and asserts the run fails (exit 1) and names the
// offending field.
func TestRunHardwareInvariants(t *testing.T) {
	// cleanChip is a well-formed h100 the cases mutate one line at a time. Kept consistent
	// with cleanChipYAML in spec/hardware/validate_test.go (its unit-level counterpart);
	// the two are in different packages and so cannot share one literal, and both must
	// gain any field this schema makes newly required.
	const cleanChip = `Provenance: vendor_spec
TFlopsPeak: 989.5
TFlopsFP8: 1979.0
BwPeakTBs: 3.35
MemoryGiB: 80.0
IntraNodeBwGBps: 450
SMCount: 132
`
	cases := []struct {
		name string
		file string // "hardware/h100.yaml" or "networks/ib-400g.yaml"
		body string
		want string // a substring the stderr report must contain
	}{
		// A dimensionless factor hidden under an underscore must fail as an unknown
		// field, not be stripped as a comment. (A bare `mfu` already failed; the gap was
		// specifically the underscore-hidden form.)
		{"underscore-hidden dimensionless field", "hardware/h100.yaml",
			cleanChip + "_mfu: 0.85\n", "_mfu"},
		// A chip with no SMCount at all is rejected: the field is required and positive.
		{"missing SMCount", "hardware/h100.yaml", `Provenance: vendor_spec
TFlopsPeak: 989.5
TFlopsFP8: 1979.0
BwPeakTBs: 3.35
MemoryGiB: 80.0
IntraNodeBwGBps: 450
`, "SMCount"},
		// A non-finite datasheet figure on a chip is rejected.
		{"NaN chip bandwidth", "hardware/h100.yaml", `Provenance: vendor_spec
TFlopsPeak: 989.5
TFlopsFP8: 1979.0
BwPeakTBs: .nan
MemoryGiB: 80.0
IntraNodeBwGBps: 450
SMCount: 132
`, "BwPeakTBs"},
		// A non-finite inter-node bandwidth on a fabric is rejected.
		{"NaN fabric bandwidth", "networks/ib-400g.yaml",
			"Provenance: vendor_spec\nInterNodeBwGBps: .nan\n", "InterNodeBwGBps"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			root := catalogWith(t, map[string]string{c.file: c.body})
			code, _, errb := exercise(t, root)
			if code != 1 {
				t.Fatalf("exit = %d, want 1\nstderr:\n%s", code, errb)
			}
			if !strings.Contains(errb, c.want) {
				t.Errorf("stderr should name %q, got: %s", c.want, errb)
			}
		})
	}
}

// TestRunMalformedFabric: a fabric that loads but fails validation (a non-positive
// inter-node bandwidth) must be reported with its field and exit 1 — the networks path is
// new coverage, so a negative test pins that its failures bubble up.
func TestRunMalformedFabric(t *testing.T) {
	root := catalogWith(t, map[string]string{
		"networks/ib-400g.yaml": `Provenance: vendor_spec
InterNodeBwGBps: 0
`,
	})
	code, _, errb := exercise(t, root)
	if code != 1 {
		t.Fatalf("exit = %d, want 1\nstderr:\n%s", code, errb)
	}
	if !strings.Contains(errb, "InterNodeBwGBps") {
		t.Errorf("stderr should name the offending field, got: %s", errb)
	}
}

// TestRunMalformedStorageDevice: a storage tier that loads but fails validation (a
// non-positive bandwidth) must be reported, named by tier, and exit 1 — the devices path
// is new coverage, so a negative test pins that a per-tier failure bubbles up.
func TestRunMalformedStorageDevice(t *testing.T) {
	root := catalogWith(t, map[string]string{
		"devices/storage.yaml": `cpu_dram: {read_bandwidth_mb_s: -1, write_bandwidth_mb_s: 2.0e4, base_latency_us: 1.0}
`,
	})
	code, _, errb := exercise(t, root)
	if code != 1 {
		t.Fatalf("exit = %d, want 1\nstderr:\n%s", code, errb)
	}
	if !strings.Contains(errb, "read_bandwidth_mb_s") || !strings.Contains(errb, "cpu_dram") {
		t.Errorf("stderr should name the tier and the offending field, got: %s", errb)
	}
}

// TestRunFlatWorkloadIsRejected pins the behaviour the task calls out: the live catalog's
// workloads are still FLAT, and a flat file is not a Shape, so the strict loader rejects
// it and the run exits 1. This is EXPECTED until catalog #16 migrates the workloads to the
// nested form; it is pinned so the rejection is a known, tested state rather than a
// surprise.
//
// Cross-repo ordering (this schema-side change does not and cannot enforce it; issue #22's
// scope note leaves wiring to blis-catalog): the nested-workload migration (catalog #16)
// must land BEFORE this binary is wired into blis-catalog's CI (blis-catalog #14), so the
// gate never runs against the flat workloads it is designed to reject. Running the
// validator over today's live catalog therefore exits 1 on exactly these four workload
// files and nothing else — the expected pre-#16 state.
func TestRunFlatWorkloadIsRejected(t *testing.T) {
	root := catalogWith(t, map[string]string{
		"workloads/chatbot.yaml": `prefix_tokens: 0
prompt_tokens: 256
output_tokens: 256
`,
	})
	code, _, errb := exercise(t, root)
	if code != 1 {
		t.Fatalf("exit = %d, want 1\nstderr:\n%s", code, errb)
	}
	if !strings.Contains(errb, "chatbot") {
		t.Errorf("stderr should name the rejected workload file, got: %s", errb)
	}
}

// TestRunUnreadableNamespaceIsReported: an existing but unreadable namespace directory
// must not pass silently. filepath.Glob swallows a directory read error and returns no
// paths, which would let the run report "all N validate" with the namespace unchecked; the
// os.ReadDir-based discovery surfaces the error as a failure and exits 1.
func TestRunUnreadableNamespaceIsReported(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root bypasses directory permissions, so an unreadable dir cannot be exercised")
	}
	root := catalogWith(t, map[string]string{
		"hardware/h100.yaml": `Provenance: vendor_spec
TFlopsPeak: 989.5
BwPeakTBs: 3.35
MemoryGiB: 80.0
IntraNodeBwGBps: 450
`,
		"networks/ib-400g.yaml": `Provenance: vendor_spec
InterNodeBwGBps: 50
`,
	})
	nsdir := filepath.Join(root, "networks")
	if err := os.Chmod(nsdir, 0o000); err != nil {
		t.Fatalf("chmod: %v", err)
	}
	// Restore read/exec before TempDir's RemoveAll runs (LIFO: this fires first).
	t.Cleanup(func() { _ = os.Chmod(nsdir, 0o700) })

	code, out, errb := exercise(t, root)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (an unreadable namespace must not pass)\nstdout:\n%s\nstderr:\n%s", code, out, errb)
	}
	if !strings.Contains(errb, "networks") {
		t.Errorf("stderr should report the unreadable namespace, got: %s", errb)
	}
}

// TestRunInvalidGraph: a graph that parses but fails schema validation (here, a cyclic
// layer DAG) must be reported and exit 1 — the original coverage, still working after the
// walk was broadened.
func TestRunInvalidGraph(t *testing.T) {
	root := catalogWith(t, map[string]string{
		"models/tiny-llm/config.json": validConfigJSON,
		"models/tiny-llm/model.yaml":  validModelYAML,
		"models/tiny-llm/graph.yaml": `kind: ModelGraph
name: tiny-llm
derived_from:
  format: hf_config_json
  path: config.json
  sha256: dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
  deriver_version: 1
global:
  hidden_size: 2048
  vocab_size: 32000
  weight_dtype: bf16
layer_kinds:
  - id: dense
    nodes:
      - {op: Elementwise, role: a}
      - {op: Elementwise, role: b}
    edges: [[0, 1], [1, 0]]
stack:
  pattern: [dense]
  repeat: 1
`,
	})
	code, _, errb := exercise(t, root)
	if code != 1 {
		t.Fatalf("exit = %d, want 1\nstderr:\n%s", code, errb)
	}
	if !strings.Contains(errb, "cyclic") {
		t.Errorf("stderr should report the cycle, got: %s", errb)
	}
}

const replayWorkload = `_comment: the 256k AgentX corpus
replay:
  scenario: inferencex-agentx-mvp
  harness_version: 0.12.0
  corpus:
    loader: semianalysis_cc_traces_weka_062126_256k
    format: weka-trace-v7
    encoding: jsonl
    source: {huggingface: semianalysisai/cc-traces-weka-062126-256k, revision: 8fecd2fc}
    sha256: e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef
  category: coding
  thinking: "on"
`

// A workloads/ file whose top-level key is replay: is validated as a replay workload, and
// one the scenario's validator would refuse fails the run.
func TestRunReplayWorkload(t *testing.T) {
	code, out, errb := exercise(t, catalogWith(t, map[string]string{"workloads/agentx.yaml": replayWorkload}))
	if code != 0 || !strings.Contains(out, "replay inferencex-agentx-mvp") {
		t.Fatalf("exit = %d\nstdout:\n%s\nstderr:\n%s", code, out, errb)
	}
	bad := strings.Replace(replayWorkload, "  category: coding\n", "  cache_bust: every_turn\n", 1)
	code, _, errb = exercise(t, catalogWith(t, map[string]string{"workloads/agentx.yaml": bad}))
	if code != 1 || !strings.Contains(errb, "cache_bust") {
		t.Fatalf("exit = %d, want 1 for a refused cache_bust\nstderr:\n%s", code, errb)
	}
	lanes := strings.Replace(replayWorkload, "  category: coding\n", "  concurrency: 8\n", 1)
	code, _, errb = exercise(t, catalogWith(t, map[string]string{"workloads/agentx.yaml": lanes}))
	if code != 1 || !strings.Contains(errb, "concurrency") {
		t.Fatalf("exit = %d, want 1 for a catalog entry stating concurrency\nstderr:\n%s", code, errb)
	}
}

// A catalog's committed measurements must be redistributable: a run whose licence is
// unverified fails the catalog even though it is well formed.
func TestRunRefusesAnUnredistributableMeasurement(t *testing.T) {
	ok := filepath.Join("testdata", "catalog-ok", "measurements")
	sources, err := os.ReadFile(filepath.Join(ok, "sources.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	runBody, err := os.ReadFile(filepath.Join(ok, "synthetic-fixture", "synthetic-h100-tp4.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	moved := strings.Replace(string(runBody), "license_ref: synthetic-fixture", "license_ref: hud", 1)
	moved = strings.Replace(moved, "verdict: EXACT\nresolution: RESOLVED\n", "verdict: PARTIAL\n", 1)
	root := catalogWith(t, map[string]string{
		"measurements/sources.yaml":                string(sources),
		"measurements/hud/synthetic-h100-tp4.yaml": moved,
	})
	code, _, errb := exercise(t, root)
	if code != 1 || !strings.Contains(errb, "does not permit redistribution") {
		t.Fatalf("exit = %d, want 1 with a redistribution refusal; stderr:\n%s", code, errb)
	}
}

package contracts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// scenarioRef builds a scenario naming references by identity. The values are the ones the
// vendored testdata/ catalog actually carries, so a passing test is proof the resolver
// resolves real names, not just that it runs.
func scenarioRef() *scenario.Scenario {
	return &scenario.Scenario{
		Kind:          "Scenario",
		Name:          "ref",
		Model:         "deepseek-v3",
		EngineVersion: "0.29.0",
		Cluster: scenario.Cluster{
			Hardware:    "h200",
			Fabric:      "ib-400g",
			Nodes:       2,
			GPUsPerNode: 8,
		},
	}
}

// TestResolveAgainstCatalogAllPresent is the positive contract: a scenario whose model,
// hardware and fabric name real catalog entries resolves clean against the vendored
// snapshot. It is the counterpart to the issue's bug — hardware: nvidia-h200 validates
// clean today — proving a correct name resolves where a wrong one is caught below.
func TestResolveAgainstCatalogAllPresent(t *testing.T) {
	rep := ValidateAgainstCatalog(Bundle{Scenario: scenarioRef()}, catalogFixtures, "")
	if !rep.OK() {
		t.Fatalf("a scenario naming real catalog entries failed to resolve:\n%s", rep.Field.Error())
	}
}

// TestResolveAgainstCatalogCatchesMisses is the issue's headline: a plausible-but-wrong name
// (the vendor-prefixed nvidia-h200 the issue cites, a nonexistent model, a stray fabric) is
// now caught, and each error names the path it looked for and enumerates what the catalog
// has, so the message teaches the convention in one turn.
func TestResolveAgainstCatalogCatchesMisses(t *testing.T) {
	s := scenarioRef()
	s.Model = "deepseek-v99"             // no such model dir
	s.Cluster.Hardware = "nvidia-h200"   // the issue's exact mistake: catalog has h200
	s.Cluster.Fabric = "infiniband-400g" // catalog has ib-400g

	rep := ValidateAgainstCatalog(Bundle{Scenario: s}, catalogFixtures, "")
	if rep.OK() {
		t.Fatal("a scenario naming three nonexistent entries resolved clean")
	}
	msg := rep.Field.Error()

	// The hardware miss is the issue's canonical case: name the looked-for path and the
	// real inventory, including the h200 the author meant.
	if !strings.Contains(msg, "hardware/nvidia-h200.yaml") {
		t.Errorf("the hardware error should name the path it looked for; got:\n%s", msg)
	}
	if !strings.Contains(msg, "h200") {
		t.Errorf("the hardware error should list the catalog's real chips so the author sees h200; got:\n%s", msg)
	}
	// Each kind of reference is reported, not just the first.
	for _, want := range []string{"nvidia-h200", "deepseek-v99", "infiniband-400g"} {
		if !strings.Contains(msg, want) {
			t.Errorf("the report should name the offending reference %q; got:\n%s", want, msg)
		}
	}
}

// TestResolveStorageAgainstInventory checks the membership-in-a-mapping-file case: a tier
// the catalog's devices/storage.yaml has resolves, and one it does not is named alongside
// the real inventory, mirroring the deployment offload-tier check this follows.
func TestResolveStorageAgainstInventory(t *testing.T) {
	s := scenarioRef()
	s.Cluster.Storage = []string{"nvme_gen4", "unobtainium"}

	rep := ValidateAgainstCatalog(Bundle{Scenario: s}, catalogFixtures, "")
	if rep.OK() {
		t.Fatal("a nonexistent storage tier resolved clean")
	}
	msg := rep.Field.Error()
	if !strings.Contains(msg, "unobtainium") || !strings.Contains(msg, "nvme_gen4") {
		t.Errorf("the storage error should name the bad tier and list the real inventory; got:\n%s", msg)
	}
	if strings.Count(msg, "unobtainium") == 0 {
		t.Errorf("the real tier nvme_gen4 should not itself be reported as a miss; got:\n%s", msg)
	}
}

// TestResolveCoefficientsAgainstRegistry resolves against a separate registry root, since a
// scenario's coefficient sets live in blis-registry, not the catalog. A tiny registry is
// built on disk because testdata/ vendors the catalog only.
func TestResolveCoefficientsAgainstRegistry(t *testing.T) {
	reg := t.TempDir()
	if err := os.MkdirAll(filepath.Join(reg, "coefficients"), 0o755); err != nil {
		t.Fatal(err)
	}
	// Only the file's presence is checked by the resolver, so an empty file suffices.
	if err := os.WriteFile(filepath.Join(reg, "coefficients", "cost-model-primitives-h200.yaml"),
		[]byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	s := scenarioRef()
	s.Coefficients = []string{"cost-model-primitives-h200", "not-a-set"}

	rep := ValidateAgainstCatalog(Bundle{Scenario: s}, catalogFixtures, reg)
	if rep.OK() {
		t.Fatal("a nonexistent coefficient set resolved clean")
	}
	msg := rep.Field.Error()
	if !strings.Contains(msg, "not-a-set") {
		t.Errorf("the error should name the missing coefficient set; got:\n%s", msg)
	}
	if !strings.Contains(msg, "the registry has") {
		t.Errorf("a coefficient error should say the registry (not the catalog) is where it looked; got:\n%s", msg)
	}
	if strings.Contains(msg, "cost-model-primitives-h200\"") {
		t.Errorf("the set that exists should not be reported as a miss; got:\n%s", msg)
	}
}

// TestResolveMissingRootsAreReported pins that a reference made with no root to resolve it
// against is an error naming the missing root, not a silent pass: a caller that forgot to
// pass a catalog should not read green.
func TestResolveMissingRootsAreReported(t *testing.T) {
	// A scenario that references the catalog, but no catalog root given.
	rep := ValidateAgainstCatalog(Bundle{Scenario: scenarioRef()}, "", "")
	if rep.OK() {
		t.Fatal("references with no catalog root resolved clean")
	}
	if !strings.Contains(rep.Field.Error(), "no catalog root") {
		t.Errorf("the error should say the catalog root was missing; got:\n%s", rep.Field.Error())
	}
}

// TestResolveRejectsNonStemNames pins that a reference is a bare filename stem (#27): a name
// carrying a path separator or a ".." is rejected as malformed rather than silently resolving
// a file OUTSIDE its namespace. Without the stem guard, hardware: "../models/<m>/graph" would
// stat models/<m>/graph.yaml — a real file — and resolve clean, which is worse than a miss.
func TestResolveRejectsNonStemNames(t *testing.T) {
	// Each of these would, under a naive filepath.Join + Stat, point at something that
	// exists in the vendored catalog outside the hardware namespace.
	for _, bad := range []string{"../models/deepseek-v3/graph", "h200/../h200", "../../etc/hosts"} {
		s := scenarioRef()
		s.Cluster.Hardware = bad
		rep := ValidateAgainstCatalog(Bundle{Scenario: s}, catalogFixtures, "")
		if rep.OK() {
			t.Errorf("a non-stem hardware name %q resolved clean; it should be rejected as malformed", bad)
			continue
		}
		if !strings.Contains(rep.Field.Error(), "bare filename stem") {
			t.Errorf("the error for %q should explain a reference must be a stem; got:\n%s",
				bad, rep.Field.Error())
		}
	}

	// A lone "." is the directory-namespace variant: filepath.Join(root, "models", ".")
	// cleans to models/ itself, which os.Stat reports as an existing directory, so a model
	// named "." would resolve clean without the stem guard. ".." is the parent-hop variant.
	// (An empty model is not resolved at all — it is Scenario.Validate's "model required", so
	// it is deliberately out of scope here.)
	for _, bad := range []string{".", ".."} {
		s := scenarioRef()
		s.Model = bad
		rep := ValidateAgainstCatalog(Bundle{Scenario: s}, catalogFixtures, "")
		if rep.OK() {
			t.Errorf("a model named %q resolved clean; it should be rejected as not an identity", bad)
		}
	}
}

// TestResolveWorkloadShape resolves the Scenario's workload Shape arm against
// workloads/<name>.yaml. The Shape arm names a catalog workload by identity, so a miss must
// be caught like any other by-name reference; the Trace arm is a file path, not an identity,
// and is deliberately not resolved.
func TestResolveWorkloadShape(t *testing.T) {
	// Hit: chatbot is a vendored workload.
	s := scenarioRef()
	s.Workload = &workload.Binding{Shape: "chatbot"}
	if rep := ValidateAgainstCatalog(Bundle{Scenario: s}, catalogFixtures, ""); !rep.OK() {
		t.Fatalf("a real workload shape failed to resolve:\n%s", rep.Field.Error())
	}

	// Miss: names the looked-for path and lists the real shapes.
	s.Workload = &workload.Binding{Shape: "chatbott"}
	rep := ValidateAgainstCatalog(Bundle{Scenario: s}, catalogFixtures, "")
	if rep.OK() {
		t.Fatal("a nonexistent workload shape resolved clean")
	}
	msg := rep.Field.Error()
	if !strings.Contains(msg, "workloads/chatbott.yaml") || !strings.Contains(msg, "chatbot") {
		t.Errorf("the error should name the looked-for path and list real shapes; got:\n%s", msg)
	}

	// The Trace arm is a path, not an identity: a shape-less binding resolves clean here
	// (the trace file's existence is not this resolver's concern).
	s.Workload = &workload.Binding{Trace: &workload.TraceRef{Data: "traces/x.csv"}}
	if rep := ValidateAgainstCatalog(Bundle{Scenario: s}, catalogFixtures, ""); !rep.OK() {
		t.Errorf("a trace-arm workload should not be resolved as a catalog name:\n%s", rep.Field.Error())
	}
}

// TestResolveFileRejectsDirectoryNamedYAML pins that a path which stats but is not a regular
// file does not count as resolved. A directory named `h200.yaml/` under hardware/ would
// satisfy an error-only os.Stat check and read as the chip file, which it is not.
func TestResolveFileRejectsDirectoryNamedYAML(t *testing.T) {
	root := t.TempDir()
	// Build a catalog where hardware/h200.yaml is a DIRECTORY, not a file.
	if err := os.MkdirAll(filepath.Join(root, "hardware", "h200.yaml"), 0o755); err != nil {
		t.Fatal(err)
	}
	s := scenarioRef()
	s.Model = ""          // skip model resolution (no models/ dir in this temp root)
	s.Cluster.Fabric = "" // skip fabric
	s.Cluster.Hardware = "h200"
	rep := ValidateAgainstCatalog(Bundle{Scenario: s}, root, "")
	if rep.OK() {
		t.Fatal("a directory named h200.yaml/ resolved as if it were the chip file")
	}
	if !strings.Contains(rep.Field.Error(), "hardware/h200.yaml") {
		t.Errorf("the error should still name the looked-for file; got:\n%s", rep.Field.Error())
	}
}

// TestResolveNilScenarioIsReported pins that a bundle with no scenario is told so rather than
// returning an empty green a caller could mistake for success.
func TestResolveNilScenarioIsReported(t *testing.T) {
	rep := ValidateAgainstCatalog(Bundle{}, catalogFixtures, "")
	if rep.OK() {
		t.Fatal("a bundle with no scenario resolved clean, which reads as success")
	}
	if !strings.Contains(rep.Field.Error(), "no scenario") {
		t.Errorf("the error should say there was no scenario to resolve; got:\n%s", rep.Field.Error())
	}
}

// The hardware reference also resolves the cluster's packaging against the chip's entry
// (SUB-3): a scenario contradicting the entry is refused at the field it states, and the
// same scenario restating the entry, or omitting what the entry fixes, resolves (SIM-9).
func TestResolveAgainstCatalogChecksPackaging(t *testing.T) {
	for _, c := range []struct {
		name     string
		hardware string
		gpn, gpr int
		refused  string
	}{
		{"NVL72 trays omitted", "gb200-nvl72", 0, 0, ""},
		{"NVL72 trays restated", "gb200-nvl72", 4, 72, ""},
		{"NVL72 at 8 per node", "gb200-nvl72", 8, 0, "scenario.cluster.gpus_per_node"},
		{"NVL72 at 12 per rack", "gb200-nvl72", 4, 12, "scenario.cluster.gpus_per_rack"},
		{"H200 at 8 per node", "h200", 8, 0, ""},
		{"H200 with no node size", "h200", 0, 0, "scenario.cluster.gpus_per_node"},
		{"H200 with a rack", "h200", 8, 16, "scenario.cluster.gpus_per_rack"},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := scenarioRef()
			s.Cluster.Hardware, s.Cluster.GPUsPerNode, s.Cluster.GPUsPerRack = c.hardware, c.gpn, c.gpr
			rep := ValidateAgainstCatalog(Bundle{Scenario: s}, catalogFixtures, "")
			if c.refused == "" {
				if !rep.OK() {
					t.Fatalf("refused:\n%s", rep.Field.Error())
				}
				return
			}
			found := false
			for _, p := range rep.Field.Errors() {
				found = found || p.Path == c.refused
			}
			if !found {
				t.Fatalf("not refused at %s:\n%s", c.refused, rep.Field.Error())
			}
		})
	}
}

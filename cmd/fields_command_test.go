package cmd

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
)

// fieldsFixture is a scenario of the engine this build registers.
const fieldsFixture = "llama-3.1-70b-instruct-h200-tp4-4node.yaml"

// reportedByTheEngine builds the scenario's engine directly through the engine factory and asks
// it for its list. It shares the engine selection and the registered roots with the command
// (selectEngine, registerContractEngines, contractRoots) and nothing that handles the list
// itself, so a command that edits the list cannot edit the expectation with it.
func reportedByTheEngine(t *testing.T, scenarioPath string) fields.FieldList {
	t.Helper()
	sel, err := selectEngine(scenarioPath)
	if err != nil {
		t.Fatal(err)
	}
	registerContractEngines()
	eng, err := engine.New(engine.Spec{Kind: sel.Ref.Kind, UpstreamVersion: sel.Ref.Version,
		Roots: contractRoots(scenarioPath)})
	if err != nil {
		t.Fatalf("building the engine: %v", err)
	}
	return eng.Fields()
}

// samePrinted reports whether printed (the bytes `fields` wrote) is exactly the canonical form
// of want, with the first difference named.
func samePrinted(printed []byte, want fields.FieldList) (string, bool) {
	canon, err := want.Marshal()
	if err != nil {
		return err.Error(), false
	}
	if bytes.Equal(printed, canon) {
		return "", true
	}
	got, err := fields.Parse(printed)
	if err != nil {
		return "the output is not a valid field list: " + err.Error(), false
	}
	in := map[string]bool{}
	for _, f := range want.Fields {
		in[string(f.Source)+"/"+f.Name] = true
	}
	for _, f := range got.Fields {
		if !in[string(f.Source)+"/"+f.Name] {
			return "prints " + f.Name + ", which the engine does not report", false
		}
		delete(in, string(f.Source)+"/"+f.Name)
	}
	for k := range in {
		return "omits " + k + ", which the engine reports", false
	}
	return "the entries match by name but differ in content or order", false
}

// `fields` prints what the engine reports and nothing else (X19, SUB-1): the output is the
// substrate's canonical form of Fields(), field for field and byte for byte, and it states the
// engine, its release and the document kind.
func TestFieldsCommand_PrintsExactlyWhatTheEngineReports(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	want := reportedByTheEngine(t, filepath.Join(pdScenarios, fieldsFixture))
	if len(want.Fields) < 100 {
		t.Fatalf("non-vacuity: the engine reports %d fields", len(want.Fields))
	}
	args := []string{"fields", "--scenarios", pdScenarios, "--scenario", fieldsFixture, "--catalog", catalog, "--registry", registry}
	out := capRawCLI(t, args...)
	if why, ok := samePrinted([]byte(out), want); !ok {
		t.Errorf("fields does not print what the engine reports: %s", why)
	}
	got, err := fields.Parse([]byte(out))
	if err != nil {
		t.Fatal(err)
	}
	if got.Kind != fields.Kind || got.Engine.Kind != "vllm" || got.Engine.Version == "" {
		t.Errorf("the output names kind %q, engine %+v; want a FieldList of a named engine and release", got.Kind, got.Engine)
	}
	if again := capRawCLI(t, args...); again != out {
		t.Error("two runs print different bytes: the output is not stable")
	}
}

// The comparison has its known-bad inputs (X18): an extra field, a missing one and an altered
// one are each found, so the test above cannot pass for a command that edits the list.
func TestFieldsCommand_TheComparisonFindsAnAddedAMissingAndAnAlteredField(t *testing.T) {
	want := reportedByTheEngine(t, filepath.Join(pdScenarios, fieldsFixture))
	canon, _ := want.Marshal()
	if _, ok := samePrinted(canon, want); !ok {
		t.Fatal("the engine's own list is not equal to itself")
	}
	clone := func() fields.FieldList {
		c := want
		c.Fields = append([]fields.Field(nil), want.Fields...)
		return c
	}
	missing := clone()
	missing.Fields = missing.Fields[1:]
	added := clone()
	added.Fields = append(added.Fields, fields.Field{Name: "ZZZ_NOT_THE_ENGINES", Source: fields.SourceEnv,
		Status: fields.StatusNotModelled, Evidence: fields.Evidence{Issue: "inference-sim/blis-vllm#53"},
		Estimation: fields.Estimation{Rule: "none"}})
	altered := clone()
	altered.Fields[2].Status = fields.StatusProvenInert
	for name, printed := range map[string]fields.FieldList{"an omitted field": missing, "an added field": added, "an altered field": altered} {
		b, err := printed.Marshal()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if _, ok := samePrinted(b, want); ok {
			t.Errorf("%s was not found by the comparison", name)
		}
	}
}

// The command refuses to guess an engine: with no scenario it prints no list.
func TestFieldsCommand_NeedsAScenario(t *testing.T) {
	cmd := exec.Command(os.Args[0], "-test.run=^TestKernelCLILeg$")
	cmd.Env = append(os.Environ(), kernelCLIArgsEnv+"="+strings.Join([]string{"fields", "--scenarios", pdScenarios}, "\x1f"))
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err == nil || out.Len() != 0 {
		t.Errorf("fields with no --scenario: err %v, output %d bytes; want a refusal and no output", err, out.Len())
	}
}

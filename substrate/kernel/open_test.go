package latencykernel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
	"github.com/inference-sim/blis-substrate/load"
)

// openRules stands in for an engine release's rules: OpenInputs only asks whether rules
// exist for the scenario's release.
type openRules struct{ rules.EngineRules }

const openScenario = `kind: Scenario
name: open-test
model: llama-3.1-8b-instruct
coefficients: [cost-model-primitives, cost-model-host-overheads]
engine_version: "0.32.0"
workload:
  shape: chatbot
cluster:
  hardware: h100
  nodes: 1
  gpus_per_node: 8
---
kind: Deployment
name: tp1
pools:
  - role: colocated
    nodes: 1
    parallel: {tp: 1}
`

func openRepos(t *testing.T, d *load.Data, body string) (Repos, string) {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "s.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return Repos{
		Scenarios: dir, Data: d,
		Rules: func(string) rules.EngineRules { return openRules{} },
		EngineSettings: func(*deployment.Deployment, int) (enginesettings.Settings, error) {
			return enginesettings.Settings{}, nil
		},
	}, "s.yaml"
}

// kernel.Open reads the scenario's documents through the loader (SUB-9): the documents it
// returns are the loader's, and a name the data in use does not have is refused with the
// data's source and the file the entry belongs at.
func TestOpenReadsThroughTheLoader(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	r, name := openRepos(t, d, openScenario)
	in, err := OpenInputs(name, r, 0)
	if err != nil {
		t.Fatal(err)
	}
	g, _ := d.Model("llama-3.1-8b-instruct")
	chip, _ := d.Chip("h100")
	if in.Model != g || in.Chip != chip || len(in.Coefficients) != 2 || len(in.Devices) == 0 {
		t.Errorf("OpenInputs did not take its documents from the loader: %+v", in)
	}
	rep := d.Report()
	for _, f := range rep.Unloaded {
		if f == "coefficients/cost-model-primitives.yaml" || f == "coefficients/cost-model-host-overheads.yaml" {
			t.Errorf("%s was loaded but is reported unloaded", f)
		}
	}

	r, name = openRepos(t, d, strings.Replace(openScenario, "hardware: h100", "hardware: h900", 1))
	if _, err := OpenInputs(name, r, 0); err == nil ||
		!strings.Contains(err.Error(), "embedded release") || !strings.Contains(err.Error(), "hardware/h900.yaml") {
		t.Errorf("a chip the data lacks must be refused naming the source and the path: %v", err)
	}
	r.Data = nil
	if _, err := OpenInputs(name, r, 0); err == nil || !strings.Contains(err.Error(), "Repos.Data") {
		t.Errorf("Open without data must be refused: %v", err)
	}
}

// RecordSelected hands every coefficient the kernel's lookups returned to the loader's
// report, against the set's file.
func TestRecordSelectedReachesTheReport(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	set, _ := d.CoefficientSet("cost-model-fabric")
	k := &Kernel{selected: []resolve.Resolved{{Entry: set.Coefficients[0], Set: "cost-model-fabric"}}}
	k.RecordSelected(d)
	rep := d.Report()
	if len(rep.Selected) != 1 || rep.Selected[0].File != "coefficients/cost-model-fabric.yaml" ||
		rep.Selected[0].Entry != "fabric_bandwidth_efficiency" || rep.Selected[0].Scope.Key() != set.Coefficients[0].Scope.Key() {
		t.Errorf("report: %+v", rep.Selected)
	}
}

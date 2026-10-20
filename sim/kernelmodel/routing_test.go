package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/routing"
)

const epScenario = "deepseek-v4.1-flash-h100-tp8-dspark5.yaml"

const routingSet = `kind: RoutingProfileSet
name: capture
role: ROLE
profiles:
  - {model: deepseek-v4.1-flash, category: text, phase: decode, placement: linear,
     expert_parallel: 8, method: measured, provenance: {source: "hf://capture@rev", hardware: b200, tier: 2},
     points: [{batch_tokens: 1, token_imbalance: 2.0, expert_imbalance: 1.5},
              {batch_tokens: 8192, token_imbalance: 1.6, expert_imbalance: 1.0}]}
`

// registryWith copies the test registry's coefficients and adds one routing set.
func registryWith(t *testing.T, role string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.Symlink(filepath.Join(repos().Registry, "coefficients"), filepath.Join(dir, "coefficients")); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "routing"), 0o755); err != nil {
		t.Fatal(err)
	}
	body := strings.Replace(routingSet, "ROLE", role, 1)
	if err := os.WriteFile(filepath.Join(dir, "routing", "capture.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func openWith(t *testing.T, registry string, o RoutingOverrides) *Model {
	t.Helper()
	SetRoutingOverrides(o)
	t.Cleanup(func() { SetRoutingOverrides(RoutingOverrides{}) })
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	r.Registry = registry
	m, err := OpenPool(epScenario, r, 0)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func decodeStep(m *Model) time.Duration {
	b := kernel.Batch{DecodeThreshold: 8, Reqs: make([]kernel.ReqShape, 64)}
	for i := range b.Reqs {
		b.Reqs[i] = kernel.ReqShape{Scheduled: 1, Computed: 4095, PromptLen: 4096}
	}
	return m.k.StepTime(b).Overlap
}

// The resolution reaches the kernel and the results file: a command-line imbalance slows an
// expert-parallel step, and the record names the layer that supplied it. With nothing
// stated the step is the evenly loaded one and the record says why.
func TestRoutingResolvesPerPoolAndIsRecorded(t *testing.T) {
	reg := registryWith(t, "sensitivity")
	even := openWith(t, reg, RoutingOverrides{})
	if r := even.Provenance().Routing; r == nil || r.SourceLayer != string(routing.LayerDefault) || r.Reason == "" {
		t.Fatalf("the default is not recorded with its reason: %+v", r)
	}
	cli := openWith(t, reg, RoutingOverrides{CLI: &routing.Value{TokenImbalance: 1.8, Source: "--routing-token-imbalance"}})
	if r := cli.Provenance().Routing; r == nil || r.SourceLayer != "cli" || r.TokenImbalance != 1.8 {
		t.Fatalf("the cli layer is not recorded: %+v", r)
	}
	if decodeStep(cli) <= decodeStep(even) {
		t.Errorf("a 1.8x busiest-rank load did not slow the step: %v against %v", decodeStep(cli), decodeStep(even))
	}
}

// A sensitivity set applies only when named; a default-role set applies to every run whose
// key it carries, here through the workload's category.
func TestRoutingRegistryRoles(t *testing.T) {
	text := &routing.Workload{Category: "text"}
	sens := openWith(t, registryWith(t, "sensitivity"), RoutingOverrides{Workload: text})
	if r := sens.Provenance().Routing; r.SourceLayer != string(routing.LayerDefault) {
		t.Errorf("an unnamed sensitivity set applied: %+v", r)
	}
	def := openWith(t, registryWith(t, "default"), RoutingOverrides{Workload: text})
	r := def.Provenance().Routing
	if r.SourceLayer != string(routing.LayerRegistry) || r.Set != "capture" || r.MeasuredOn != "b200" ||
		len(r.Keys) != 1 || r.Keys[0] != "deepseek-v4.1-flash/text/decode/linear/ep8" {
		t.Fatalf("the registry profile is not resolved or not recorded: %+v", r)
	}
	if decodeStep(def) <= decodeStep(sens) {
		t.Errorf("the registry curve did not reach the step")
	}
}

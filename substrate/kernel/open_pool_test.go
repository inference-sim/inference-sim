package latencykernel

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/load"
)

// poolScenario states a cluster of the given chip, optionally with node pools of other chips.
func poolScenario(chip, pools string) string {
	return `kind: Scenario
name: pool-chip
model: llama-3.1-8b-instruct
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric]
engine_version: "0.32.0"
workload:
  shape: chatbot
cluster:
  hardware: ` + chip + `
  nodes: 1
  gpus_per_node: 8
` + pools + `---
kind: Deployment
name: tp1
pools:
  - role: colocated
    nodes: 1
    parallel: {tp: 1, pp: 1, dp: 1}
`
}

const twoChipPools = `  pools:
    - name: old
      hardware: h100
    - name: new
      hardware: h200
`

func priceNodePool(t *testing.T, d *load.Data, body, nodePool string) (kernel.StepEstimate, kernel.MemoryBreakdown, float64) {
	t.Helper()
	r, name := openRepos(t, d, body)
	r.Rules = func(string) rules.EngineRules { return inertRules{} }
	in, err := OpenInputsIn(name, r, 0, nodePool)
	if err != nil {
		t.Fatalf("node pool %q: %v", nodePool, err)
	}
	k, err := New(in)
	if err != nil {
		t.Fatalf("node pool %q: %v", nodePool, err)
	}
	b := kernel.Batch{DecodeThreshold: 1}
	for range 16 {
		b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 1, Computed: 2047, PromptLen: 2048})
	}
	return k.StepTime(b), k.FixedBytes(), in.Chip.MemoryGiB
}

// Each node pool of one scenario is priced and sized as a scenario of its own chip is: the
// same step time, the same fixed bytes, the same device memory the KV budget is cut from.
func TestEachNodePoolIsPricedByItsOwnChip(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ pool, chip, other string }{
		{"old", "h100", "h200"}, {"new", "h200", "h100"},
	} {
		wantStep, wantFixed, wantMem := priceNodePool(t, d, poolScenario(c.chip, ""), "")
		gotStep, gotFixed, gotMem := priceNodePool(t, d, poolScenario("h100", twoChipPools), c.pool)
		if !sameStep(gotStep, wantStep) || gotFixed != wantFixed || gotMem != wantMem {
			t.Errorf("pool %q (%s): step %v fixed %v mem %v; a %s scenario gives step %v fixed %v mem %v",
				c.pool, c.chip, gotStep, gotFixed, gotMem, c.chip, wantStep, wantFixed, wantMem)
		}
		otherStep, _, otherMem := priceNodePool(t, d, poolScenario(c.other, ""), "")
		if sameStep(gotStep, otherStep) && gotMem == otherMem {
			t.Errorf("pool %q prices like %s: the test does not tell the chips apart", c.pool, c.other)
		}
	}
}

// A single-chip scenario prices identically whether or not a node pool is named: the cluster's
// chip is the default, and OpenInputs is unchanged.
func TestNoNodePoolIsTheClusterChip(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	wantStep, wantFixed, wantMem := priceNodePool(t, d, poolScenario("h100", ""), "")
	gotStep, gotFixed, gotMem := priceNodePool(t, d, poolScenario("h100", twoChipPools), "")
	if !sameStep(gotStep, wantStep) || gotFixed != wantFixed || gotMem != wantMem {
		t.Errorf("a cluster with pools, none named, must price as its own chip")
	}
	namedStep, _, _ := priceNodePool(t, d, poolScenario("h100", "  pools:\n    - name: only\n"), "only")
	if !sameStep(namedStep, wantStep) {
		t.Errorf("a pool that states no chip carries the cluster's: step %v, want %v", namedStep, wantStep)
	}
}

func TestUnknownNodePoolIsRefusedByName(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	r, name := openRepos(t, d, poolScenario("h100", twoChipPools))
	_, err = OpenInputsIn(name, r, 0, "missing")
	if err == nil || !strings.Contains(err.Error(), `"missing"`) || !strings.Contains(err.Error(), "old") {
		t.Errorf("an unlisted node pool must be refused naming it and the pools: %v", err)
	}
	r, name = openRepos(t, d, poolScenario("h100", ""))
	if _, err = OpenInputsIn(name, r, 0, "any"); err == nil {
		t.Errorf("a node pool named in a cluster that declares none must be refused")
	}
}

func sameStep(a, b kernel.StepEstimate) bool { return reflect.DeepEqual(a, b) }

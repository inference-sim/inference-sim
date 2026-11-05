package contracts

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
)

const bundleScenario = `kind: Scenario
name: granite-230b-h200-tp8
model: granite-5-230b
coefficients: [cost-model-primitives-h200]
engine_version: "0.32.0"
workload:
  shape: chatbot
cluster:
  hardware: h200
  nodes: 1
  gpus_per_node: 8
`

const bundleDeployment = `kind: Deployment
name: granite-230b-h200-tp8
pools:
  - role: colocated
    nodes: 1
    parallel:
      tp: 8
      pp: 1
      dp: 1
    engine:
      block_size: 16
`

func TestLoadBundle(t *testing.T) {
	path := write(t, "b.yaml", bundleScenario+"---\n"+bundleDeployment)
	sc, dep, err := LoadBundle(path)
	if err != nil {
		t.Fatalf("LoadBundle: %v", err)
	}
	if sc.Name != "granite-230b-h200-tp8" || sc.EngineVersion != "0.32.0" || sc.Cluster.Hardware != "h200" {
		t.Errorf("scenario did not parse: %+v", sc)
	}
	if len(dep.Pools) != 1 || dep.Pools[0].Parallel.TP != 8 {
		t.Fatalf("deployment did not parse: %+v", dep.Pools)
	}
	var e map[string]any
	if err := dep.Pools[0].Engine.Decode(&e); err != nil || e["block_size"] != 16 {
		t.Errorf("engine block = %v (err %v), want block_size 16", e, err)
	}
}

// The bundle loader is the scenario and deployment loaders over one stream: each half must
// decode to exactly what LoadScenario and LoadDeployment decode from that half alone, over
// every field.
func TestLoadBundleMatchesTheSingleDocumentLoaders(t *testing.T) {
	sc, dep, err := LoadBundle(write(t, "b.yaml", bundleScenario+"---\n"+bundleDeployment))
	if err != nil {
		t.Fatalf("LoadBundle: %v", err)
	}
	wantSc, err := LoadScenario(write(t, "s.yaml", bundleScenario))
	if err != nil {
		t.Fatalf("LoadScenario: %v", err)
	}
	wantDep, err := LoadDeployment(write(t, "d.yaml", bundleDeployment))
	if err != nil {
		t.Fatalf("LoadDeployment: %v", err)
	}
	if !reflect.DeepEqual(sc, wantSc) {
		t.Errorf("scenario half differs from LoadScenario:\n got %+v\nwant %+v", sc, wantSc)
	}
	// EngineBlock holds an unexported YAML node, so compare it with its own Equal and the
	// rest of each pool with DeepEqual.
	if len(dep.Pools) != len(wantDep.Pools) {
		t.Fatalf("pools: got %d, want %d", len(dep.Pools), len(wantDep.Pools))
	}
	for i := range dep.Pools {
		if !dep.Pools[i].Engine.Equal(wantDep.Pools[i].Engine) {
			t.Errorf("pool %d engine block differs from LoadDeployment", i)
		}
	}
	gotRest, wantRest := *dep, *wantDep
	gotRest.Pools, wantRest.Pools = nil, nil
	if !reflect.DeepEqual(gotRest, wantRest) {
		t.Errorf("deployment half differs from LoadDeployment:\n got %+v\nwant %+v", gotRest, wantRest)
	}
	for i := range dep.Pools {
		g, w := dep.Pools[i], wantDep.Pools[i]
		g.Engine, w.Engine = deployment.EngineBlock{}, deployment.EngineBlock{}
		if !reflect.DeepEqual(g, w) {
			t.Errorf("pool %d differs from LoadDeployment:\n got %+v\nwant %+v", i, g, w)
		}
	}
}

func TestLoadBundleRefusals(t *testing.T) {
	for _, c := range []struct {
		name, body, want string
	}{
		{"empty file", "", "is empty"},
		{"no deployment", bundleScenario, "holds a scenario but no deployment"},
		{"no pools", bundleScenario + "---\nkind: Deployment\nname: x\npools: []\n", "declares no pools"},
		{"unknown scenario field", "kind: Scenario\nnmae: typo\n---\n" + bundleDeployment, "decoding the scenario"},
		{"unknown deployment field", bundleScenario + "---\nkind: Deployment\npoolz: []\n", "decoding the deployment"},
	} {
		t.Run(c.name, func(t *testing.T) {
			path := write(t, "b.yaml", c.body)
			_, _, err := LoadBundle(path)
			if err == nil || !strings.Contains(err.Error(), c.want) || !strings.Contains(err.Error(), path) {
				t.Fatalf("err = %v, want one containing %q and the path", err, c.want)
			}
		})
	}
	if _, _, err := LoadBundle("/nonexistent/bundle.yaml"); err == nil || !strings.Contains(err.Error(), "opening") {
		t.Fatalf("missing file: err = %v, want an opening error", err)
	}
}

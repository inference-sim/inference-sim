// Original to blis-vllm.

package build

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// A run needs its scenario, scenario directory and registry, and the refusal names every one
// that is missing.
func TestAdopt_NeedsTheScenarioAndItsRoots(t *testing.T) {
	_, err := Adopt(AdoptInputs{})
	if err == nil {
		t.Fatal("no scenario must be refused")
	}
	for _, flag := range []string{"--scenario", "--scenarios", "--registry"} {
		if !strings.Contains(err.Error(), flag) {
			t.Errorf("refusal %q must name %s", err, flag)
		}
	}
}

// A P/D scenario adopts as disaggregated and refuses to open without a P/D topology; with one
// it opens the first pool, sized by the kernel, at the catalog the locator names.
func TestAdopt_DisaggregatedNeedsAPDTopology(t *testing.T) {
	repos := vendoredRepos(t)
	in := AdoptInputs{Scenario: "glm-5-h200-3p1d-ib.yaml", ScenariosDir: repos.Scenarios, Registry: repos.Registry,
		Catalog: func() (string, error) { return repos.Catalog, nil }}
	a, err := Adopt(in)
	if err != nil || !a.Disaggregated || len(a.Shape.Roles) != 2 {
		t.Fatalf("adopt: %+v, %v", a, err)
	}
	if err := OpenFirstPool(in, &a); err == nil || !strings.Contains(err.Error(), "needs a P/D topology") {
		t.Errorf("no topology: %v", err)
	}
	in.PDTopology = true
	opened := false
	in.BeforeOpen = func(kernelmodel.Shape) error { opened = true; return nil }
	if err := OpenFirstPool(in, &a); err != nil || a.Model == nil || a.Settings.KVBlocks <= 0 || a.CatalogRoot != repos.Catalog || !opened {
		t.Fatalf("with a topology: model %v, %d blocks, root %q, hook %v, err %v", a.Model != nil, a.Settings.KVBlocks, a.CatalogRoot, opened, err)
	}
	if a.Deployment.Model == "" || a.Deployment.TP < 1 {
		t.Errorf("deployment %+v, want the first pool's model and parallelism", a.Deployment)
	}
	in.Catalog = func() (string, error) { return "", errors.New("no catalog located") }
	if err := OpenFirstPool(in, &a); err == nil || !strings.Contains(err.Error(), "no catalog located") {
		t.Errorf("no catalog: %v", err)
	}
	in.Catalog = func() (string, error) { return repos.Catalog, nil }
	in.BeforeOpen = func(kernelmodel.Shape) error { return errors.New("bad routing flags") }
	if err := OpenFirstPool(in, &a); err == nil || !strings.Contains(err.Error(), "bad routing flags") {
		t.Errorf("a failing hook: %v", err)
	}
}

// A colocated scenario adopts as colocated and opens with no topology.
func TestAdopt_Colocated(t *testing.T) {
	repos := vendoredRepos(t)
	in := AdoptInputs{Scenario: "glm-5-h200-tp8-mtp3.yaml", ScenariosDir: repos.Scenarios, Registry: repos.Registry,
		Catalog: func() (string, error) { return repos.Catalog, nil }}
	a, err := Adopt(in)
	if err != nil || a.Disaggregated {
		t.Fatalf("adopt: %+v, %v", a, err)
	}
	if err := OpenFirstPool(in, &a); err != nil || a.Model == nil {
		t.Fatalf("open: %v", err)
	}
	if _, err := OpenModel("missing.yaml", repos.Scenarios, repos.Catalog, repos.Registry); err == nil ||
		!strings.Contains(err.Error(), `"missing.yaml"`) {
		t.Errorf("a missing scenario: %v", err)
	}
}

// Dense data and expert parallelism are refused; an MoE model runs both.
func TestCheckParallelism(t *testing.T) {
	for _, c := range []struct {
		dp       int
		ep, moe  bool
		wantFrag string
	}{
		{0, false, true, "it must be >= 1"},
		{2, false, false, "DP placement is modelled for MoE"},
		{1, true, false, "which vLLM refuses"},
		{2, true, true, ""},
		{1, false, false, ""},
	} {
		err := CheckParallelism("s.yaml", "m", c.dp, c.ep, c.moe)
		if c.wantFrag == "" {
			if err != nil {
				t.Errorf("dp %d ep %v moe %v: %v", c.dp, c.ep, c.moe, err)
			}
			continue
		}
		if err == nil || !strings.Contains(err.Error(), c.wantFrag) {
			t.Errorf("dp %d ep %v moe %v: %v, want %q", c.dp, c.ep, c.moe, err, c.wantFrag)
		}
	}
}

// A scenario stating two colocated pools is refused: a run simulates one colocated engine,
// and the second pool would be dropped without a word.
func TestAdopt_RefusesTwoColocatedPools(t *testing.T) {
	repos := vendoredRepos(t)
	raw, err := os.ReadFile(filepath.Join(repos.Scenarios, "glm-5-h200-3p1d-ib.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	two := strings.ReplaceAll(strings.ReplaceAll(string(raw), "role: prefill", "role: colocated"), "role: decode", "role: colocated")
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "two.yaml"), []byte(two), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = Adopt(AdoptInputs{Scenario: "two.yaml", ScenariosDir: dir, Registry: repos.Registry})
	if err == nil || !strings.Contains(err.Error(), "states 2 colocated pools") {
		t.Errorf("two colocated pools: %v", err)
	}
}


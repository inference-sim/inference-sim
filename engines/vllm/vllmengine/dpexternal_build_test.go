// Original to blis-vllm.
// H3 of the DP external load-balancing RFC (inference-sim#2699).

package vllmengine

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"

	"github.com/inference-sim/inference-sim/engines/vllm/core/vllm"
)

// TestExternalDPPoolBuildsOneEndpointPerRank pins the engine-side half of the external DP build
// (inference-sim#2699): a pool whose parallel.data_parallel_load_balancing is external must build a
// group the router sees as one endpoint per rank. The deployment states the mode under parallel,
// which the engine block does not carry, so complete() translates it into the engine's
// data_parallel_external_lb; without that translation the group builds internal (one endpoint) and
// an external deployment runs collapsed while reporting external. extdp-032.yaml is a dp=4 colocated
// pool stating external, so the built group must expose 4 endpoints, not 1.
func TestExternalDPPoolBuildsOneEndpointPerRank(t *testing.T) {
	Register()
	abs, err := filepath.Abs(filepath.Join("testdata", "extdp-032.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	eng, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.34.0+0fda86a35419",
		Roots: map[string]string{RootScenario: abs}})
	if err != nil {
		t.Fatalf("building the external dp=4 pool: %v", err)
	}
	g, ok := eng.(*vllm.DPGroup)
	if !ok {
		t.Fatalf("built %T, want a *DPGroup", eng)
	}
	if got := g.Endpoints(); got != 4 {
		t.Errorf("external dp=4 group exposes %d router endpoint(s), want 4 (one per rank); "+
			"the deployment's parallel.data_parallel_load_balancing: external must reach the engine's "+
			"data_parallel_external_lb, or the group builds internal and runs collapsed while reporting external", got)
	}
}

// TestInternalDPPoolStaysOneEndpoint is the no-op guard: a dp pool that states no balancing (the
// internal default) still builds as one endpoint, so the translation does not disturb internal/unset
// deployments.
func TestInternalDPPoolStaysOneEndpoint(t *testing.T) {
	Register()
	abs, err := filepath.Abs(filepath.Join("testdata", "d116-032.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	eng, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.34.0+0fda86a35419",
		Roots: map[string]string{RootScenario: abs}})
	if err != nil {
		t.Fatalf("building the internal dp=4 pool: %v", err)
	}
	g, ok := eng.(*vllm.DPGroup)
	if !ok {
		t.Fatalf("built %T, want a *DPGroup", eng)
	}
	if got := g.Endpoints(); got != 1 {
		t.Errorf("internal dp=4 group exposes %d endpoints, want 1 (vLLM's internal balancer is one endpoint)", got)
	}
}

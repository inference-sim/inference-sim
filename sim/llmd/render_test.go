package llmd

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/inference-sim/blis-vllm/compat/contract"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// renderStating is an engine that states whether its model server answers the router's render
// calls (engine.RenderServing), as blis-vllm's vLLM engine does from engine.enable_scale_out.
// Embedding the compatibility engine keeps every other optional interface it implements.
type renderStating struct {
	*contract.Engine
	serves bool
}

func (r renderStating) ServesRender() bool { return r.serves }

// renderFailures is the number of failed render calls the router's token producer has counted in
// this process (token_producer_render_failures_total, summed over its labels).
func renderFailures(t *testing.T) float64 {
	t.Helper()
	families, err := metrics.Registry.Gather()
	if err != nil {
		t.Fatal(err)
	}
	var n float64
	for _, f := range families {
		if strings.HasSuffix(f.GetName(), "token_producer_render_failures_total") {
			for _, m := range f.GetMetric() {
				n += m.GetCounter().GetValue()
			}
		}
	}
	return n
}

// noScorerConfig is preciseConfig without any prefix awareness: the same picker and profile
// handler and no scorer.
const noScorerConfig = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: max-score-picker
`

func servingRender(serves bool) cluster.DeploymentConfig {
	dep := testDeployment(4)
	dep.NewEngine = compatEngineFactoryWith(0, pdTicks, func(e engine.Engine) engine.Engine {
		return renderStating{Engine: e.(*contract.Engine), serves: serves}
	})
	return dep
}

func assigned(reqs []*sim.Request) []string {
	out := make([]string, len(reqs))
	for i, r := range reqs {
		out[i] = r.AssignedInstance
	}
	return out
}

// vLLM v0.33.0 answers every render call with 403 unless the server runs with --enable-scale-out
// (vllm/entrypoints/scale_out/factories.py). The router's token producer then counts a render
// failure and goes on without token IDs, so its prefix scorers have nothing to match and the
// pick falls to load alone (inference-sim#2129). With the flag the same configuration routes by
// prefix.
func TestRender_RefusedWithoutScaleOutAndRoutingFallsToLoad(t *testing.T) {
	groups := func() []*sim.Request { return prefixGroups(3, 15, 256, 30000) }

	before := renderFailures(t)
	refused := runPrecise(t, servingRender(false), groups(), preciseConfig(""), Options{})
	failed := renderFailures(t) - before
	if failed <= 0 {
		t.Errorf("a server that refuses render calls: the token producer counted %v failures, want some", failed)
	}
	loadOnly := runPrecise(t, servingRender(false), groups(), noScorerConfig, Options{})
	if got, want := assigned(refused.reqs), assigned(loadOnly.reqs); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("render refused: endpoint choices differ from a load-only configuration\n got  %v\n want %v", got, want)
	}

	before = renderFailures(t)
	served := runPrecise(t, servingRender(true), groups(), preciseConfig(""), Options{})
	if d := renderFailures(t) - before; d != 0 {
		t.Errorf("a server that answers render calls: the token producer counted %v failures, want none", d)
	}
	for g := 0; g < 3; g++ {
		where := map[string]int{}
		for _, r := range served.reqs {
			if r.PrefixGroup == "g"+string(rune('0'+g)) {
				where[r.AssignedInstance]++
			}
		}
		if len(where) != 1 {
			t.Errorf("render served: prefix group %d spread over %v; prefix-aware routing keeps it together", g, where)
		}
	}
}

// An engine that does not say is taken to serve render calls, as the simulated server did before
// engines could state it (the compatibility engines of older releases always serve them).
func TestRender_AnEngineThatStatesNothingServesRenderCalls(t *testing.T) {
	before := renderFailures(t)
	runPrecise(t, testDeployment(4), prefixGroups(2, 5, 256, 30000), preciseConfig(""), Options{})
	if d := renderFailures(t) - before; d != 0 {
		t.Errorf("an engine that states nothing: %v render failures, want none", d)
	}
}

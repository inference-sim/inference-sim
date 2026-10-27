package harness

import (
	"testing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// The scenario's enable_prefix_caching must reach the simulator, and the Deployment is where
// that becomes checkable without running a simulation. Five links sit between the YAML and
// the gate (scenario -> Engine -> Deployment -> BatchConfig option -> BatchContext); this
// pins the first three, where a silent break would leave every unit test passing.
func TestScenarioPrefixCachingReachesTheDeployment(t *testing.T) {
	const scenario = "gpt-oss-120b-h200-fp4-vllm-tp4.yaml"
	committed := kernelmodel.Repos{
		Scenarios: kernelmodel.DefaultScenarios(),
		Catalog:   kernelmodel.DefaultCatalog(),
		Registry:  kernelmodel.DefaultRegistry(),
	}
	m, err := kernelmodel.Open(scenario, committed)
	if err != nil {
		t.Fatalf("Open(%s): %v", scenario, err)
	}
	// The committed files state nothing, so the engine default applies: caching is ON.
	if m.Deployment().PrefixCachingDisabled {
		t.Error("a scenario that states nothing must take vLLM's default of ON, so " +
			"PrefixCachingDisabled should be false")
	}
}

// The benchmark's main-phase requests share no prompt prefix, and neither do the requests
// `blis replay` builds from the point's trace -- so with prefix caching enabled no MAIN request
// is served from another main request's cache, in the benchmark or in the simulation.
//
// The benchmark side, from benchmark_serving.py: request i's prompt is the token ids
// (offsets[i] + i + j) % vocab for j < input_lens[i], offsets drawn at random per request, with
// no shared prefix at --random-prefix-len 0. The simulation side is checked here: the trace
// carries no prefix group, and the token ids replay generates for one-round sessions share no
// leading block.
//
// What this does NOT cover is the warm-up: its 2*CONC copies of request 0 do leave request 0's
// prompt cached for the main phase. That is the gap the coverage report flags on every point
// with prefix caching enabled.
func TestThePointsRequestsShareNoPrefixToCache(t *testing.T) {
	ws, err := LoadBenchmarkWorkloads(DefaultWorkloadsPath(0))
	if err != nil {
		t.Fatal(err)
	}
	pl, err := ws.At(1024, 1024, 16)
	if err != nil {
		t.Fatal(err)
	}
	header, records := pointTrace(pl)
	for _, r := range records {
		if r.PrefixGroup != "" || r.PrefixLength != 0 {
			t.Fatalf("record %d states prefix group %q of %d tokens", r.RequestID, r.PrefixGroup, r.PrefixLength)
		}
	}
	reqs, _, err := workload.LoadTraceV2SessionBlueprints(&workload.TraceV2{Header: *header, Records: records},
		42, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	const blockSize = 16
	for i := 1; i < len(reqs); i++ {
		a, b := reqs[0].InputTokens, reqs[i].InputTokens
		same := len(a) >= blockSize && len(b) >= blockSize
		for j := 0; same && j < blockSize; j++ {
			same = a[j] == b[j]
		}
		if same {
			t.Errorf("requests 0 and %d share their first %d-token block; the simulated main phase "+
				"would hit the prefix cache where the benchmark's does not", i, blockSize)
		}
	}
}

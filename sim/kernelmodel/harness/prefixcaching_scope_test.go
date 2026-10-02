package harness

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// The scenario's enable_prefix_caching must reach the simulator, and the Deployment is where
// that becomes checkable without running a simulation. Five links sit between the YAML and
// the gate (scenario -> Engine -> Deployment -> BatchConfig option -> BatchContext); this
// pins the first three, where a silent break would leave every unit test passing.
func TestScenarioPrefixCachingReachesTheDeployment(t *testing.T) {
	const scenario = "gpt-oss-120b-h200-fp4-vllm-tp4.yaml"
	committed := kernelmodel.Repos{
		Scenarios: "/Users/sri/Documents/Projects/blis-latency-kernel/testdata/aisimulate",
		Catalog:   "/Users/sri/Documents/Projects/blis-catalog",
		Registry:  "/Users/sri/Documents/Projects/blis-registry",
	}
	m, err := kernelmodel.Open(scenario, committed)
	if err != nil {
		t.Skipf("scenario unavailable: %v", err)
	}
	// The committed files state nothing, so the engine default applies: caching is ON.
	if m.Deployment().PrefixCachingDisabled {
		t.Error("a scenario that states nothing must take vLLM's default of ON, so " +
			"PrefixCachingDisabled should be false")
	}
}

// This corpus cannot exercise prefix caching, and that must be stated rather than assumed.
//
// The harness sets PrefixLength to 0 because AISimulate's own spec sets
// cached_prefix_tokens to zero, and the generator then draws independent tokens per request.
// So no request shares a leading block with another, and disabling reuse changes nothing on
// these points -- correctly, not because the gate is broken.
//
// The consequence is worth pinning: a future change that gives the workload a shared prefix
// WOULD make the setting matter, and this test is where that becomes visible. Without it, a
// reader could conclude from an unchanged score that the feature does not work.
func TestThisCorpusHasNoSharedPrefixToCache(t *testing.T) {
	w := AISimulateWorkload(1024, 1024, 16)
	spec := &workload.WorkloadSpec{
		Version: "v1", Seed: 42, Category: "language",
		Clients: []workload.ClientSpec{{
			ID: "closed-loop", Concurrency: 16,
			InputDist: workload.DistSpec{
				Type: "empirical", Params: pdfParams(uniformPDF(w.ISLLow, w.ISLHigh))},
			OutputDist: workload.DistSpec{
				Type: "empirical", Params: pdfParams(uniformPDF(w.OSLLow, w.OSLHigh))},
			PrefixLength: 0,
			Streaming:    true,
		}},
		NumRequests: 64,
	}
	if err := spec.Validate(); err != nil {
		t.Fatalf("spec: %v", err)
	}
	wl, err := workload.GenerateWorkload(spec, 1<<62, 64)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(wl.Requests) < 2 {
		t.Skip("need at least two requests")
	}

	const blockSize = 16
	worst := 0
	for i := 1; i < len(wl.Requests) && i < 16; i++ {
		a, b := wl.Requests[0].InputTokens, wl.Requests[i].InputTokens
		shared := 0
		for off := 0; off+blockSize <= len(a) && off+blockSize <= len(b); off += blockSize {
			same := true
			for j := 0; j < blockSize; j++ {
				if a[off+j] != b[off+j] {
					same = false
					break
				}
			}
			if !same {
				break
			}
			shared++
		}
		if shared > worst {
			worst = shared
		}
	}
	if worst != 0 {
		t.Errorf("requests share up to %d leading block(s); this corpus now DOES exercise "+
			"prefix caching, so the scores depend on enable_prefix_caching and the "+
			"per-deployment setting must be carried", worst)
	}
}

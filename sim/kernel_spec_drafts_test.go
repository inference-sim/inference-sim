package sim_test

import (
	"math"
	"testing"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-schemas/kernel"

	sim "github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	_ "github.com/inference-sim/inference-sim/sim/vendored" // the adapter's vendoring root

	host "github.com/inference-sim/inference-sim/sim"
)

// The kernel adapter's rejected-drafts law (INV-K2, #1932), checked through a whole simulation:
// it moved here from sim/kernelmodel when the adapter moved to blis-vllm (compat/kernelmodel),
// because it drives the host's event loop.

// stepRecorder prices with a Model and records, for every step, the request shapes the
// kernel was handed.
type stepRecorder struct {
	*kernelmodel.Model
	steps [][]kernel.ReqShape
}

func (r *stepRecorder) StepTime(batch []*sim.Request) int64 {
	r.steps = append(r.steps, r.BatchShapesForTest(batch))
	return r.Model.StepTime(batch)
}

// openDSparkDrafting opens the colocated DeepSeek-V4.1-Flash DSpark scenario of this
// repository's testdata/scenarios with its pool drafting k tokens, through the path
// kernelmodel's OpenPool takes.
func openDSparkDrafting(t *testing.T, k int) (*kernelmodel.Model, error) {
	t.Helper()
	r := kernelmodel.DefaultRepos()
	r.Scenarios = "../testdata/scenarios"
	in, err := latencykernel.OpenInputs("deepseek-v4.1-flash-h100-tp8-dspark5.yaml", r, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := kernelmodel.DraftForTest(&in, "dspark", k); err != nil {
		t.Fatal(err)
	}
	return kernelmodel.FromInputsForTest(in)
}

// INV-K2: rejected drafts do not advance Computed (#1932). The kernel reads ReqShape.Computed for
// context -- KV reads, ring sizes, sparse selections -- so after a speculative step it must
// be the rolled-back position: the previous Computed plus the tokens accepted, never plus the
// verify width. Driven through a real simulation of one DSpark request at acceptance 0.502:
//
//   - every decode step is priced at the verify width 1 + K;
//   - between consecutive steps Computed advances by the granted advance, at most 1 + K;
//   - some step accepts fewer than the verify width, so the law discriminates;
//   - the advances sum to the request's completion boundary, as with no speculation.
func TestRejectedDraftsDoNotAdvanceComputed(t *testing.T) {
	const k, acceptance, input, output = 5, 0.502, 300, 120
	m, err := openDSparkDrafting(t, k)
	if err != nil {
		t.Fatal(err)
	}
	rec := &stepRecorder{Model: m}
	cfg := sim.SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                1,
		KVCacheConfig:       sim.NewKVCacheConfig(10000, 64, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(16, 8192, 0),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, "", "", 1, 1, false, 0),
		SpeculativeConfig:   sim.SpeculativeConfig{K: k, Acceptance: acceptance, Method: "dspark"},
	}
	s, err := host.NewSimulator(cfg, sim.MustNewKVStoreFromConfig(cfg.KVCacheConfig), rec)
	if err != nil {
		t.Fatal(err)
	}
	req := &sim.Request{ID: "r", InputTokens: make([]sim.TokenID, input), OutputTokens: make([]sim.TokenID, output),
		State: sim.StateQueued}
	s.InjectArrival(req)
	s.Run()
	if req.State != sim.StateCompleted {
		t.Fatalf("request state %q, want completed", req.State)
	}
	var decodes []kernel.ReqShape
	for _, step := range rec.steps {
		if len(step) == 1 && step[0].Computed >= input {
			decodes = append(decodes, step[0])
		}
	}
	if len(decodes) < 2 {
		t.Fatalf("%d decode steps recorded; the law needs two", len(decodes))
	}
	short := false
	for i, d := range decodes {
		if d.Scheduled != 1+k {
			t.Errorf("decode step %d scheduled %d, want the verify width %d", i, d.Scheduled, 1+k)
		}
		if i == 0 {
			continue
		}
		advance := d.Computed - decodes[i-1].Computed
		if advance < 1 || advance > 1+k {
			t.Errorf("decode step %d: Computed moved %d -> %d, an advance of %d outside [1, %d]",
				i, decodes[i-1].Computed, d.Computed, advance, 1+k)
		}
		short = short || advance < 1+k
	}
	if !short {
		t.Error("every step accepted the whole verify width at acceptance 0.502; Computed is tracking the verify width")
	}
	if want := int64(input + output - 1); req.ProgressIndex != want {
		t.Errorf("final ProgressIndex %d, want the completion boundary %d", req.ProgressIndex, want)
	}
}

package cluster

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/load"

	"github.com/inference-sim/blis-vllm/compat/dplb"
	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/internal/kernelfixtures"
	"github.com/inference-sim/inference-sim/sim"
)

// plainLatency hides a latency model's DPStepPricer, so a lockstep group prices each rank
// at its own batch with no padding.
type plainLatency struct{ cengine.LatencyModel }

// Through the cluster, a lockstep replica priced by the kernel pads its ranks: a
// long-context decode on one rank, stepping beside short prefills on the other, costs the
// padded token count, so it finishes later than the same run priced without padding -- and
// never sooner, since padding only adds work.
//
// Padding applies when the step runs under a CUDA graph on every rank, so the test states the
// graph mode it needs. The fixture states none (it states only what its corpus states, X19),
// and vLLM 0.32.0 then resolves decode steps to FULL graphs, under which this batch is not
// padded: the long request finishes at the same time padded and unpadded. Under PIECEWISE,
// which the earlier fixture stated, padding adds 0.13% to the long request's E2E.
func TestDPLockstep_KernelPaddingReachesTheCluster(t *testing.T) {
	data, err := kernelmodel.OpenData(load.Options{})
	if err != nil {
		t.Fatal(err)
	}
	const fixture = "minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml"
	raw, err := os.ReadFile(filepath.Join(kernelfixtures.AISimulate(), fixture))
	if err != nil {
		t.Fatal(err)
	}
	scenario := strings.Replace(string(raw), "      max_model_len:", "      compilation_config: {cudagraph_mode: PIECEWISE}\n      max_model_len:", 1)
	if scenario == string(raw) {
		t.Fatalf("%s has no max_model_len line to put cudagraph_mode before", fixture)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, fixture), []byte(scenario), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := kernelmodel.Open(fixture, kernelmodel.Repos{Scenarios: dir, Data: data})
	if err != nil {
		t.Fatal(err)
	}
	longE2E := func(lm cengine.LatencyModel) float64 {
		reqs := []*sim.Request{dpReq("long", 0, 8000, 200)}
		for i := 1; i <= 20; i++ {
			reqs = append(reqs, dpReq(fmt.Sprintf("s%d", i), int64(i)*dplb.StatsIntervalUs+1, 200, 1))
		}
		cfg := newTestDeploymentConfig(2)
		cfg.DPLockstepRanks = 2
		editEngine(&cfg, func(c *simConfig) { c.LatencyModel = lm })
		cs := newTestCluster(cfg, NewSliceRequestSource(reqs), nil)
		mustRun(t, cs)
		if reqs[0].AssignedInstance != "instance_0" || reqs[1].AssignedInstance != "instance_1" {
			t.Fatalf("long on %s, first short on %s; want ranks 0 and 1", reqs[0].AssignedInstance, reqs[1].AssignedInstance)
		}
		return cs.AggregatedMetrics().RequestE2Es["long"]
	}
	padded, unpadded := longE2E(m), longE2E(plainLatency{m})
	if padded <= unpadded {
		t.Errorf("the long request's E2E is %v padded and %v unpadded; padding must slow it", padded, unpadded)
	}
}

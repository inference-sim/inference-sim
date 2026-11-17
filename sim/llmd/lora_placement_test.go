package llmd

import (
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"

	"github.com/inference-sim/blis-vllm/compat/contract"
	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// holderEngine is the compatibility engine with the dynamic adapter load a v0.33 engine offers
// (blis-vllm core/vllm Engine.LoadAdapter), publishing the adapters it holds as its residency.
type holderEngine struct {
	*contract.Engine
	resident []string
}

func (e *holderEngine) LoadAdapter(name string) error {
	if !slices.Contains(e.resident, name) {
		e.resident = append(e.resident, name)
	}
	return nil
}
func (e *holderEngine) UnloadAdapter(name string) (bool, error) { return false, nil }
func (e *holderEngine) ResidentAdapterIDs() []string            { return slices.Clone(e.resident) }
func (e *holderEngine) AdapterCapacity() int                    { return 4 }

func holderFactory() func(int, sim.InstanceConfig) (*sim.EngineDriver, error) {
	return func(index int, ic sim.InstanceConfig) (*sim.EngineDriver, error) {
		c := engineConfigIn(ic)
		eng, err := contract.New(fmt.Sprintf("instance_%d", index), c)
		if err != nil {
			return nil, err
		}
		dcfg := sim.EngineDriverConfig{Horizon: c.Horizon, ResumeGapFromLastToken: true, ScrapeEngineMetrics: true,
			ApplyAtStepStart: true, Environment: testutil.FixedPull(c.BlockSizeTokens, nil)}
		if m := c.LatencyModel; m != nil {
			dcfg.PostDecodeFixedOverhead = m.PostDecodeFixedOverhead()
			dcfg.OutputTokenProcessingTime = m.OutputTokenProcessingTime()
			dcfg.Environment = testutil.WithHost(dcfg.Environment, func(n int) kernel.HostDemand {
				return kernel.HostDemand{Service: time.Duration(m.QueueingTime(&cengine.Request{
					InputTokens: make([]cengine.TokenID, n)})) * time.Microsecond}
			})
		}
		return sim.NewEngineDriver(&holderEngine{Engine: eng}, dcfg), nil
	}
}

// The router's lora-affinity scorer reads the adapter sets each engine publishes in
// vllm:lora_requests_info, which vLLM fills from the requests it is running and holding in its
// queue, not from the adapters it has loaded (sim/engine_driver.go, sim/llmd/server.go). An adapter
// placed on an instance and idle is not advertised, so placing adapters ahead of traffic changes
// no routing decision: the router sends every request where it would have without placement. What
// placement changes is the engine's own cost for the first request of a placed adapter (it is
// already loaded), which is why it is reported by the engine and not through the router.
func TestLoRAPlacement_LeavesRoutingToTheRoutersOwnSignal(t *testing.T) {
	adapters := []string{"a0", "a1", "a2", "a3"}
	run := func(placement cluster.LoRAPlacementConfig) []string {
		reqs := testRequests(40, 13, 20000, 0)
		for i, r := range reqs {
			r.Adapter = adapters[i%len(adapters)]
		}
		dep := testDeployment(4)
		dep.NewEngine = holderFactory()
		dep.LoRAPlacement = placement
		cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
		seed := int64(3)
		cs.SetGateway(New(Options{Model: "test-model", Seed: &seed,
			EPPFlags: append(DefaultEPPFlags(), "--config-text="+loraAffinityConfig)}))
		if err := cs.Run(); err != nil {
			t.Fatalf("Run: %v", err)
		}
		out := make([]string, len(reqs))
		for i, r := range reqs {
			out[i] = r.AssignedInstance
		}
		return out
	}
	placed := cluster.LoRAPlacementConfig{Creation: cluster.LoRACreationPrePlacement,
		Assignment: map[int][]string{0: {"a0"}, 1: {"a1"}, 2: {"a2"}, 3: {"a3"}}}
	with, without := run(placed), run(cluster.LoRAPlacementConfig{})
	if !slices.Equal(with, without) {
		t.Errorf("placing adapters changed the router's decisions:\n with    %v\n without %v", with, without)
	}
}

package cluster

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-schemas/engine"

	"github.com/inference-sim/blis-vllm/compat/contract"
	"github.com/inference-sim/blis-vllm/compat/kv"
	"github.com/inference-sim/inference-sim/sim"
)

// countingEngine counts the steps the driver asks of the engine.
type countingEngine struct {
	*contract.Engine
	schedules int
}

func (c *countingEngine) Schedule(now engine.Time) (engine.Batch, error) {
	c.schedules++
	return c.Engine.Schedule(now)
}

// A slow secondary tier through the engine contract: every waiting request defers on a KV fetch
// that takes ~123 ms, and the compatibility engine behind engine.Engine (blis-vllm
// compat/contract) sleeps to the fetch's wake, as its own host loop's idle wake does
// (Simulator.scheduleNextStep), rather than polling an empty step per step time; an arrival
// during the sleep is served at once (scheduleStepBy). Run on the slow-tier workloads both ways
// (arrivals packed before the first fetch, and spaced wider than one), the driven engine
// completes every request at the host loop's instants, with its report.
func TestContractEngine_SlowTierDeferralWakeMatchesTheHostLoop(t *testing.T) {
	for _, c := range []struct {
		name       string
		gap        int64
		supersedes bool
	}{{"packed", 500, false}, {"spaced", 40_000, true}} {
		t.Run(c.name, func(t *testing.T) {
			cfg := offloadE2ECfgReadBW(7, 0.1)
			host := NewInstanceSimulator(InstanceID("instance_0"), cfg)
			eng, err := contract.New("instance_0", cfg)
			if err != nil {
				t.Fatal(err)
			}
			counted := &countingEngine{Engine: eng}
			drv := sim.NewEngineDriver(counted, sim.EngineDriverConfig{Horizon: cfg.Horizon, ResumeGapFromLastToken: true,
				ScrapeEngineMetrics:       true,
				PostDecodeFixedOverhead:   cfg.LatencyModel.PostDecodeFixedOverhead(),
				OutputTokenProcessingTime: cfg.LatencyModel.OutputTokenProcessingTime(),
				QueueingTime:              cfg.LatencyModel.QueueingTime})
			driven := NewEngineInstance(InstanceID("instance_0"), cfg, drv)
			for _, inst := range []*InstanceSimulator{host, driven} {
				for _, r := range cyclingPrefixWorkloadSpaced(7, 12, 4, c.gap) {
					inst.InjectRequest(r)
				}
				inst.Run()
			}
			if oc := host.sim.KVCache.(*kv.OffloadCache); oc.DeferralsStarted() == 0 {
				t.Fatal("the slow tier deferred no request; the comparison is vacuous")
			}
			if n := host.sim.IdleWakesSuperseded(); (n > 0) != c.supersedes {
				t.Fatalf("the host loop superseded %d idle wakes; the leg does not exercise what it names", n)
			}
			want, got := host.Metrics(), driven.Metrics()
			if got.CompletedRequests != 48 || got.CompletedRequests != want.CompletedRequests {
				t.Fatalf("completed %d through the contract, %d on the host loop, of 48", got.CompletedRequests, want.CompletedRequests)
			}
			if !reflect.DeepEqual(got.RequestCompletionTimes, want.RequestCompletionTimes) {
				t.Errorf("completion times differ from the host loop's")
			}
			if !reflect.DeepEqual(got.RequestTTFTs, want.RequestTTFTs) {
				t.Errorf("TTFTs differ from the host loop's")
			}
			if got.SimEndedTime != want.SimEndedTime || got.CPUTierHitTokens != want.CPUTierHitTokens ||
				got.PrefixCacheHitTokens != want.PrefixCacheHitTokens {
				t.Errorf("SimEndedTime %d / %d, CPU-tier hits %d / %d, prefix hits %d / %d (contract / host loop)",
					got.SimEndedTime, want.SimEndedTime, got.CPUTierHitTokens, want.CPUTierHitTokens,
					got.PrefixCacheHitTokens, want.PrefixCacheHitTokens)
			}
			if hs := host.sim.StepCountForTest(); counted.schedules > 2*hs+100 {
				t.Errorf("the driver asked the contract engine for %d steps against the host loop's %d: it polls through the fetch",
					counted.schedules, hs)
			}
		})
	}
}

package cluster

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/contract"
	"github.com/inference-sim/inference-sim/sim"
)

// The blocks an engine reached through the engine contract caches -- its prefix cache's, its CPU
// tier's and a secondary tier's -- are the blocks the compatibility engine's own host loop
// caches on the same workload (InstanceSimulator.CachedBlockHashes, CPUBlockHashes,
// TierBlockHashes, which the precise-prefix router's tests check its index against).
func TestContractEngine_BlockHashesAreTheHostLoops(t *testing.T) {
	cfg := offloadE2ECfgReadBW(7, 0.1)
	host := NewInstanceSimulator(InstanceID("instance_0"), cfg)
	eng, err := contract.New("instance_0", cfg)
	if err != nil {
		t.Fatal(err)
	}
	drv := sim.NewEngineDriver(eng, sim.EngineDriverConfig{Horizon: cfg.Horizon, ResumeGapFromLastToken: true,
		ScrapeEngineMetrics: true, RecordPrefixLookups: true, ApplyAtStepStart: true,
		PostDecodeFixedOverhead:   cfg.LatencyModel.PostDecodeFixedOverhead(),
		OutputTokenProcessingTime: cfg.LatencyModel.OutputTokenProcessingTime(),
		QueueingTime:              cfg.LatencyModel.QueueingTime})
	driven := NewEngineInstance(InstanceID("instance_0"), cfg, drv)
	for _, inst := range []*InstanceSimulator{host, driven} {
		for _, r := range cyclingPrefixWorkloadSpaced(7, 12, 4, 500) {
			inst.InjectRequest(r)
		}
		inst.Run()
	}
	for _, c := range []struct {
		name      string
		got, want []string
	}{
		{"prefix cache", driven.CachedBlockHashes(), host.CachedBlockHashes()},
		{"CPU tier", driven.CPUBlockHashes(), host.CPUBlockHashes()},
		{"shared_storage tier", driven.TierBlockHashes("shared_storage"), host.TierBlockHashes("shared_storage")},
	} {
		if len(c.want) == 0 {
			t.Fatalf("the host loop's %s holds no block; the comparison is vacuous", c.name)
		}
		if !reflect.DeepEqual(c.got, c.want) {
			t.Errorf("%s: %d blocks through the contract, %d on the host loop", c.name, len(c.got), len(c.want))
		}
	}
}

package cmd

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/spf13/pflag"
)

// TestPrefixCachingFlag_RegisteredOnRunAndReplay pins the CLI surface required by
// INV-13: the deployment setting must be re-suppliable on both legs.
func TestPrefixCachingFlag_RegisteredOnRunAndReplay(t *testing.T) {
	for _, c := range []struct {
		name string
		cmd  interface{ Flags() *pflag.FlagSet }
	}{
		{"run", runCmd},
		{"replay", replayCmd},
	} {
		f := c.cmd.Flags().Lookup("no-enable-prefix-caching")
		if f == nil {
			t.Errorf("%s missing --no-enable-prefix-caching", c.name)
			continue
		}
		if f.DefValue != "false" {
			t.Errorf("%s --no-enable-prefix-caching default=%q, want false (INV-6)", c.name, f.DefValue)
		}
	}
}

// TestBatchConfigFromCLI_PrefixCachingDisabled proves the user-facing knob reaches
// the exact BatchConfig constructor shared by run and replay. If either the option body
// or this production wiring is removed, this test fails while the lower-level batch
// formation tests can still pass.
func TestBatchConfigFromCLI_PrefixCachingDisabled(t *testing.T) {
	orig := noEnablePrefixCaching
	defer func() { noEnablePrefixCaching = orig }()

	noEnablePrefixCaching = false
	if got := batchConfigFromCLI().PrefixCachingDisabled; got {
		t.Fatal("default CLI state must keep prefix caching enabled")
	}

	noEnablePrefixCaching = true
	if got := batchConfigFromCLI().PrefixCachingDisabled; !got {
		t.Fatal("--no-enable-prefix-caching must set BatchConfig.PrefixCachingDisabled")
	}
}

func prefixCachingTestDeployment(t *testing.T, batch sim.BatchConfig) cluster.DeploymentConfig {
	t.Helper()
	catalogDir, hwPath := setupTrainedPhysicsTestFixtures(t)
	hfConfig, err := latency.ParseHFConfig(testCatalogConfigPath(catalogDir, "test-model"))
	if err != nil {
		t.Fatalf("ParseHFConfig: %v", err)
	}
	mc, err := latency.GetModelConfigFromHF(hfConfig)
	if err != nil {
		t.Fatalf("GetModelConfigFromHF: %v", err)
	}
	hwCfg, err := latency.GetHWConfig(hwPath, "H100")
	if err != nil {
		t.Fatalf("GetHWConfig: %v", err)
	}
	return cluster.DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             10_000_000,
			Seed:                99,
			KVCacheConfig:       sim.NewKVCacheConfig(1000, 16, 0, 0.9, 100.0, 0),
			BatchConfig:         batch,
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1, 0, 0, 0, 100, 0, 0, 0, 0, 0}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(*mc, hwCfg, "test-model", "H100", 1, 1, false, "", "trained-physics", 4096),
			PolicyConfig:        sim.NewPolicyConfig("fcfs", ""),
		},
		NumInstances:    1,
		AdmissionPolicy: "always-admit",
		RoutingPolicy:   "round-robin",
	}
}

func runPrefixCachingConfig(t *testing.T, cfg cluster.DeploymentConfig, reqs []*sim.Request) (*cluster.ClusterSimulator, map[string]float64, map[string]float64) {
	t.Helper()
	cs := cluster.NewClusterSimulator(cfg, cluster.NewSliceRequestSource(reqs), nil)
	if err := cs.Run(); err != nil {
		t.Fatalf("simulation failed: %v", err)
	}
	m := cs.AggregatedMetrics()
	if len(m.RequestTTFTs) == 0 {
		t.Fatal("simulation completed no requests; test is vacuous")
	}
	return cs, m.RequestTTFTs, m.RequestE2Es
}

func assertFloatMapEqual(t *testing.T, label string, want, got map[string]float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s map size: run=%d replay=%d", label, len(want), len(got))
	}
	for id, v := range want {
		if g, ok := got[id]; !ok {
			t.Errorf("%s: request %s missing from replay", label, id)
		} else if g != v {
			t.Errorf("%s: request %s run=%f replay=%f", label, id, v, g)
		}
	}
}

// TestINV13_RunReplayParity_PrefixCachingDisabled locks the full setting thread:
// real CLI state -> shared BatchConfig wiring -> Simulator -> BatchContext -> prefix
// cache gate, then export/reload and replay with the same deployment setting.
//
// The enabled control must record a positive cache-hit rate on the same shared-prefix
// workload, while the disabled run must record zero. That non-vacuity check proves the
// flag changes actual simulator behavior rather than merely surviving configuration.
func TestINV13_RunReplayParity_PrefixCachingDisabled(t *testing.T) {
	orig := noEnablePrefixCaching
	defer func() { noEnablePrefixCaching = orig }()

	noEnablePrefixCaching = false
	enabledCfg := prefixCachingTestDeployment(t, batchConfigFromCLI())
	enabledCS, _, _ := runPrefixCachingConfig(t, enabledCfg, makeSharedPrefixGroupRequests())
	if got := enabledCS.AggregatedMetrics().CacheHitRate; got <= 0 {
		t.Fatalf("enabled control should observe shared-prefix cache hits, got %v", got)
	}

	noEnablePrefixCaching = true
	disabledCfg := prefixCachingTestDeployment(t, batchConfigFromCLI())
	runCS, runTTFT, runE2E := runPrefixCachingConfig(t, disabledCfg, makeSharedPrefixGroupRequests())
	if got := runCS.AggregatedMetrics().CacheHitRate; got != 0 {
		t.Fatalf("--no-enable-prefix-caching should suppress GPU prefix hits, got %v", got)
	}

	dir := t.TempDir()
	records := workload.RequestsToTraceRecords(makeSharedPrefixGroupRequests())
	header := &workload.TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated"}
	hdrPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")
	if err := workload.ExportTraceV2(header, records, hdrPath, dataPath); err != nil {
		t.Fatalf("ExportTraceV2: %v", err)
	}
	traceData, err := workload.LoadTraceV2(hdrPath, dataPath)
	if err != nil {
		t.Fatalf("LoadTraceV2: %v", err)
	}
	replayReqs, err := workload.LoadTraceV2Requests(traceData, disabledCfg.SimConfig.Seed)
	if err != nil {
		t.Fatalf("LoadTraceV2Requests: %v", err)
	}
	replayCS, replayTTFT, replayE2E := runPrefixCachingConfig(t, disabledCfg, replayReqs)
	if got := replayCS.AggregatedMetrics().CacheHitRate; got != 0 {
		t.Fatalf("replay with prefix caching disabled should also have zero hit rate, got %v", got)
	}

	assertFloatMapEqual(t, "TTFT", runTTFT, replayTTFT)
	assertFloatMapEqual(t, "E2E", runE2E, replayE2E)
}

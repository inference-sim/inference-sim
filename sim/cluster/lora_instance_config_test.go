package cluster

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

// Config-time rank coupling: per-instance max_lora_rank and capacity. LoRA construction
// funcs are registered via the blank import in lora_import_test.go.

const (
	instCfgFootprint = 2.0e6 // footprint_bytes_per_rank in every fixture below
	instCfgGPUMemGiB = 80.0
	instCfgMemUtil   = 0.9
	instCfgBlockSize = 16
)

// instanceConfigFixture builds a valid two-instance deployment with per-instance slots:
// adapters a8, b16, c64 (ranks 8, 16, 64), cluster-wide capacity 3, pre-placement, and
// KV auto-calc enabled on a model CalculateKVBlocks accepts. ranks/caps/placement are
// the per-test variables; pass nil lists for the cluster-wide configuration.
func instanceConfigFixture(t *testing.T, ranks, caps []int, placement map[int][]string) DeploymentConfig {
	t.Helper()
	dc := newTestDeploymentConfig(2)
	hw := testRooflineHWCalib()
	hw.MemoryGiB = instCfgGPUMemGiB
	dc.ModelHardwareConfig = sim.NewModelHardwareConfig(kvAutoCalcTestModel(), hw,
		"test-model", "H100", 1, 1, false, "", "roofline", 0)
	dc.KVCacheConfig = sim.NewKVCacheConfig(10000, instCfgBlockSize, 0, 0, 0, 0)
	capacity := 3
	base, bw, fp := 1000.0, 2.0e6, instCfgFootprint
	dc.LoRAConfig = sim.LoRAConfig{
		AdapterCapacity:       &capacity,
		LoadBaseLatencyUs:     &base,
		LoadBandwidthBytesUs:  &bw,
		FootprintBytesPerRank: &fp,
		Adapters: []sim.AdapterSpec{
			{ID: "a8", Rank: 8}, {ID: "b16", Rank: 16}, {ID: "c64", Rank: 64},
		},
		CreationPolicy: "pre-placement",
	}
	dc.RoutingPolicy = "route-to-holder"
	dc.LoRAAdapterPlacement = placement
	dc.LoRAInstanceMaxRank = ranks
	dc.LoRAInstanceCapacity = caps
	dc.KVAutoCalc = KVAutoCalcConfig{
		Enabled:              true,
		GPUMemoryUtilization: instCfgMemUtil,
		Params:               kvAutoCalcTestParams(),
	}
	return dc
}

func requirePanicContaining(t *testing.T, want string, f func()) {
	t.Helper()
	defer func() {
		t.Helper()
		r := recover()
		if r == nil {
			t.Fatalf("expected a panic containing %q, got none", want)
		}
		if msg := r.(string); !strings.Contains(msg, want) {
			t.Fatalf("panic %q does not contain %q", msg, want)
		}
	}()
	f()
}

var instCfgPlacement = map[int][]string{0: {"a8", "b16"}, 1: {"c64"}}

// Each guard in ValidateLoRAInstanceConfig, one mutation of a valid fixture per case. The
// premise — that the unmutated fixture passes — is asserted first, so a case can only
// fail on its own mutation.
func TestValidateLoRAInstanceConfig_Guards(t *testing.T) {
	valid := func() DeploymentConfig {
		return instanceConfigFixture(t, []int{16, 64}, []int{2, 1}, instCfgPlacement)
	}
	if err := ValidateLoRADeployment(valid()); err != nil {
		t.Fatalf("premise: valid fixture rejected: %v", err)
	}
	cases := []struct {
		name   string
		mutate func(*DeploymentConfig)
		want   string
	}{
		{"ranks without capacities", func(d *DeploymentConfig) { d.LoRAInstanceCapacity = nil }, "must be set together"},
		{"capacities without ranks", func(d *DeploymentConfig) { d.LoRAInstanceMaxRank = nil }, "must be set together"},
		{"LoRA disabled", func(d *DeploymentConfig) {
			d.Adapters = nil
			d.LoRAAdapterPlacement = nil
		}, "LoRA disabled"},
		{"rank list too short", func(d *DeploymentConfig) { d.LoRAInstanceMaxRank = []int{64} }, "lora_instance_max_rank has 1 entries"},
		{"capacity list too long", func(d *DeploymentConfig) { d.LoRAInstanceCapacity = []int{2, 1, 1} }, "lora_instance_capacity has 3 entries"},
		{"node pools", func(d *DeploymentConfig) { d.NodePools = []NodePoolConfig{{Name: "p"}} }, "not supported with node_pools"},
		{"PD disaggregation", func(d *DeploymentConfig) { d.PrefillInstances = 1 }, "not supported with PD disaggregation"},
		{"explicit KV blocks", func(d *DeploymentConfig) { d.KVAutoCalc.Enabled = false }, "requires auto-calculated KV capacity"},
		{"rank outside vLLM's set", func(d *DeploymentConfig) { d.LoRAInstanceMaxRank = []int{12, 64} }, "lora_instance_max_rank[0] = 12"},
		{"zero capacity", func(d *DeploymentConfig) { d.LoRAInstanceCapacity = []int{2, 0} }, "lora_instance_capacity[1] = 0"},
		{"seeded rank above cap", func(d *DeploymentConfig) { d.LoRAInstanceMaxRank = []int{8, 64} }, `adapter "b16" of rank 16, above its max_lora_rank 8`},
		{"seeded count above capacity", func(d *DeploymentConfig) { d.LoRAInstanceCapacity = []int{1, 1} }, "instance 0 assigned 2 adapters, exceeds capacity 1"},
		{"scheduled rank above cap", func(d *DeploymentConfig) {
			d.PlacementSchedule = []sim.PlacementScheduleEntry{{AtUs: 10, Placement: map[int][]string{0: {"c64"}}}}
		}, `adapter "c64" of rank 64, above its max_lora_rank 16`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := valid()
			tc.mutate(&d)
			err := ValidateLoRADeployment(d)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got error %v, want one containing %q", err, tc.want)
			}
		})
	}
}

// The per-instance capacity, not the cluster-wide one, bounds a placement: the same
// placement passes without the lists (cluster-wide capacity 3) and fails with them.
func TestValidateLoRAPlacement_UsesPerInstanceCapacity(t *testing.T) {
	placement := map[int][]string{0: {"a8", "b16"}}
	if err := ValidateLoRADeployment(instanceConfigFixture(t, nil, nil, placement)); err != nil {
		t.Fatalf("premise: placement within the cluster-wide capacity rejected: %v", err)
	}
	err := ValidateLoRADeployment(instanceConfigFixture(t, []int{16, 64}, []int{1, 3}, placement))
	if err == nil || !strings.Contains(err.Error(), "exceeds capacity 1") {
		t.Fatalf("got %v, want the per-instance capacity 1 to reject 2 seeded adapters", err)
	}
}

// Absent lists are valid whatever else is configured, including with LoRA off (R20).
func TestValidateLoRAInstanceConfig_AbsentIsNil(t *testing.T) {
	dc := newTestDeploymentConfig(2)
	if err := ValidateLoRAInstanceConfig(dc, nil); err != nil {
		t.Fatalf("absent lists with LoRA off: got %v, want nil", err)
	}
}

// NewClusterSimulator enforces the same checks, by panic (library layer).
func TestNewClusterSimulator_PanicsOnInvalidInstanceConfig(t *testing.T) {
	dc := instanceConfigFixture(t, []int{8, 64}, []int{2, 1}, instCfgPlacement)
	requirePanicContaining(t, "above its max_lora_rank 8", func() {
		NewClusterSimulator(dc, NewSliceRequestSource(nil), nil)
	})
}

// expectedKVBlocks is the independent computation: CalculateKVBlocks on the fixture's
// model and GPU, net of a reservation of capacity × footprint × rank.
func expectedKVBlocks(t *testing.T, capacity, rank int) (reserved int64, blocks int64) {
	t.Helper()
	reserved = int64(float64(capacity) * instCfgFootprint * float64(rank))
	blocks, err := latency.CalculateKVBlocks(kvAutoCalcTestModel(), sim.HardwareCalib{MemoryGiB: instCfgGPUMemGiB},
		1, 1, instCfgBlockSize, instCfgMemUtil, kvAutoCalcTestParams(),
		latency.WithAdapterReservedBytes(reserved))
	if err != nil {
		t.Fatalf("setup: CalculateKVBlocks: %v", err)
	}
	return reserved, blocks
}

// Each instance's reservation is its own capacity × footprint × max rank, and its KV
// blocks are CalculateKVBlocks net of exactly that reservation — checked against an
// independent computation, on the echo and on the constructed instance.
func TestLoRAInstanceConfig_ReservationAndKVPerInstance(t *testing.T) {
	ranks, caps := []int{16, 64}, []int{2, 1}
	cs := NewClusterSimulator(instanceConfigFixture(t, ranks, caps, instCfgPlacement), NewSliceRequestSource(nil), nil)
	mustRun(t, cs)
	echoes := cs.LoRAInstanceEchoes()
	if len(echoes) != 2 {
		t.Fatalf("got %d echoes, want 2", len(echoes))
	}
	for i, e := range echoes {
		wantReserved, wantBlocks := expectedKVBlocks(t, caps[i], ranks[i])
		if e.MaxLoRARank != ranks[i] || e.AdapterCapacity != caps[i] {
			t.Errorf("instance %d echo = (rank %d, capacity %d), want (%d, %d)", i, e.MaxLoRARank, e.AdapterCapacity, ranks[i], caps[i])
		}
		if e.AdapterReservedBytes != wantReserved {
			t.Errorf("instance %d reservation = %d, want %d", i, e.AdapterReservedBytes, wantReserved)
		}
		if e.TotalKVBlocks != wantBlocks {
			t.Errorf("instance %d echoed KV blocks = %d, want %d", i, e.TotalKVBlocks, wantBlocks)
		}
		if got := cs.Instances()[i].TotalKVBlocks(); got != wantBlocks {
			t.Errorf("instance %d constructed with %d KV blocks, want %d", i, got, wantBlocks)
		}
	}
	if echoes[0].TotalKVBlocks == echoes[1].TotalKVBlocks {
		t.Errorf("premise: the two reservations should give different KV budgets, both got %d", echoes[0].TotalKVBlocks)
	}
}

// With every instance's list entry equal to the cluster-wide value (the catalog maximum
// rank and the global capacity), a run is indistinguishable from one without the lists,
// provided the global KV count carries the same reservation — which is what the CLI's
// global auto-calc computes. Compared on the full aggregated output, under adapter traffic.
func TestLoRAInstanceConfig_UniformListsMatchClusterWide(t *testing.T) {
	adapters := []string{"a8", "b16", "c64"}
	placement := map[int][]string{0: {"a8", "b16"}, 1: {"c64", "a8"}}
	run := func(dc DeploymentConfig) string {
		cs := NewClusterSimulator(dc, NewSliceRequestSource(zipfianAdapterRequests(200, adapters)), nil)
		mustRun(t, cs)
		out, err := json.Marshal(cs.AggregatedMetrics().BuildOutput("cluster"))
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		return string(out)
	}
	_, globalBlocks := expectedKVBlocks(t, 3, 64)
	clusterWide := instanceConfigFixture(t, nil, nil, placement)
	clusterWide.KVCacheConfig = sim.NewKVCacheConfig(globalBlocks, instCfgBlockSize, 0, 0, 0, 0)
	uniform := instanceConfigFixture(t, []int{64, 64}, []int{3, 3}, placement)

	want, got := run(clusterWide), run(uniform)
	if !strings.Contains(want, `"a8"`) {
		t.Fatalf("premise: no adapter traffic reached the output: %s", want)
	}
	if got != want {
		t.Errorf("uniform per-instance lists changed the run:\n got %s\nwant %s", got, want)
	}
}

// A run-time load of an adapter above the instance's cap panics. Here c64 is placed
// nowhere, so route-to-holder falls back to unconstrained routing and an instance capped
// at rank 8 or 16 is asked to cold-load it — a load vLLM would refuse.
func TestLoRAInstanceConfig_RunTimeLoadAboveCapPanics(t *testing.T) {
	dc := instanceConfigFixture(t, []int{8, 16}, []int{2, 2}, map[int][]string{0: {"a8"}, 1: {"b16"}})
	reqs := newTestRequests(5)
	for _, r := range reqs {
		r.Adapter = "c64"
	}
	cs := NewClusterSimulator(dc, NewSliceRequestSource(reqs), nil)
	requirePanicContaining(t, `adapter "c64" has rank 64, above this instance's max_lora_rank`, func() {
		_ = cs.Run()
	})
}

// The seed site is guarded too, independently of the cluster's validation: an instance
// built directly with a cap below a seeded adapter's rank panics at ApplyInitialCreation.
func TestLoRAInstanceConfig_SeedAboveCapPanics(t *testing.T) {
	dc := instanceConfigFixture(t, nil, nil, nil)
	simCfg := dc.SimConfig
	rank := 8
	simCfg.InstanceMaxRank = &rank
	inst := NewInstanceSimulator("instance_0", simCfg)
	requirePanicContaining(t, `adapter "c64" has rank 64, above this instance's max_lora_rank 8`, func() {
		inst.ApplyInitialCreation([]string{"c64"})
	})
}

// A missing GPU memory figure is fatal on this path rather than a fallback to the
// inherited global KV count, which would silently drop the per-instance reservation.
func TestLoRAInstanceConfig_NoGPUMemoryPanics(t *testing.T) {
	dc := instanceConfigFixture(t, []int{16, 64}, []int{2, 1}, instCfgPlacement)
	dc.HWConfig.MemoryGiB = 0
	requirePanicContaining(t, "hardware MemoryGiB is 0", func() {
		NewClusterSimulator(dc, NewSliceRequestSource(nil), nil)
	})
}

// The echo is absent without the lists, refuses to be read before Run, and carries each
// instance's own post-run preemption count. The second run is KV-tight on purpose — 6.5
// GiB leaves about 2200 and 1200 blocks, and 300 requests arrive at once — so the counts
// are non-zero and differ between instances, and a missing or cluster-total count fails.
func TestLoRAInstanceEchoes(t *testing.T) {
	cs := NewClusterSimulator(instanceConfigFixture(t, nil, nil, instCfgPlacement), NewSliceRequestSource(nil), nil)
	mustRun(t, cs)
	if e := cs.LoRAInstanceEchoes(); e != nil {
		t.Errorf("echo without per-instance lists = %v, want nil", e)
	}

	dc := instanceConfigFixture(t, []int{16, 64}, []int{2, 1}, instCfgPlacement)
	dc.HWConfig.MemoryGiB = 6.5
	reqs := zipfianAdapterRequests(300, []string{"a8", "b16", "c64"})
	for _, r := range reqs {
		r.ArrivalTime = 0
	}
	cs = NewClusterSimulator(dc, NewSliceRequestSource(reqs), nil)
	requirePanicContaining(t, "called before Run()", func() { cs.LoRAInstanceEchoes() })
	mustRun(t, cs)
	a, b := cs.Instances()[0].Metrics().PreemptionCount, cs.Instances()[1].Metrics().PreemptionCount
	if a+b == 0 || a == b {
		t.Fatalf("premise: want non-zero, unequal per-instance preemptions, got %d and %d", a, b)
	}
	for i, e := range cs.LoRAInstanceEchoes() {
		if want := cs.Instances()[i].Metrics().PreemptionCount; e.PreemptionCount != want {
			t.Errorf("instance %d echoed %d preemptions, instance recorded %d", i, e.PreemptionCount, want)
		}
		if e.InstanceID != string(cs.Instances()[i].ID()) {
			t.Errorf("echo %d names %q, want %q", i, e.InstanceID, cs.Instances()[i].ID())
		}
	}
}

// The global CLI caps max-model-len once, from the global KV count, which is net of the
// cluster-wide reservation. Each instance must instead be capped from the UNCAPPED value
// by its own KV budget, as vLLM's _maybe_limit_model_len does per server. The fixture's
// MaxModelLen of 1000 stands for a stale global cap; LoRAInstanceMaxModelLen carries the
// uncapped value, here far above either instance's KV, so each ends at its own blocks × 16.
func TestLoRAInstanceConfig_MaxModelLenRecappedPerInstance(t *testing.T) {
	ranks, caps := []int{16, 64}, []int{2, 1}
	dc := instanceConfigFixture(t, ranks, caps, instCfgPlacement)
	dc.MaxModelLen = 1000
	dc.LoRAInstanceMaxModelLen = 1 << 40
	cs := NewClusterSimulator(dc, NewSliceRequestSource(nil), nil)
	mustRun(t, cs)
	echoes := cs.LoRAInstanceEchoes()
	for i, e := range echoes {
		_, blocks := expectedKVBlocks(t, caps[i], ranks[i])
		if want := blocks * instCfgBlockSize; e.MaxModelLen != want {
			t.Errorf("instance %d max_model_len = %d, want its own KV-feasible %d", i, e.MaxModelLen, want)
		}
	}
	if echoes[0].MaxModelLen == echoes[1].MaxModelLen {
		t.Errorf("premise: the two KV budgets should give different caps, both %d", echoes[0].MaxModelLen)
	}

	// Without an uncapped value (0), the configured MaxModelLen is kept when it fits.
	dc = instanceConfigFixture(t, ranks, caps, instCfgPlacement)
	dc.MaxModelLen = 1000
	cs = NewClusterSimulator(dc, NewSliceRequestSource(nil), nil)
	mustRun(t, cs)
	for i, e := range cs.LoRAInstanceEchoes() {
		if e.MaxModelLen != 1000 {
			t.Errorf("instance %d max_model_len = %d, want the configured 1000", i, e.MaxModelLen)
		}
	}
}

// The rank guard runs where a load STARTS, before a victim is evicted, so a load whose
// completion would fall past the horizon cannot leave an eviction recorded for a load
// vLLM would refuse. Here the only request arrives 1000 µs before the horizon and the
// c64 load takes about 1064 µs, so completion never runs.
func TestLoRAInstanceConfig_AboveCapLoadNearHorizonPanics(t *testing.T) {
	dc := instanceConfigFixture(t, []int{8, 16}, []int{2, 2}, map[int][]string{0: {"a8"}, 1: {"b16"}})
	reqs := newTestRequests(1)
	reqs[0].Adapter = "c64"
	dc.Horizon = reqs[0].ArrivalTime + 1000
	cs := NewClusterSimulator(dc, NewSliceRequestSource(reqs), nil)
	requirePanicContaining(t, `adapter "c64" has rank 64, above this instance's max_lora_rank`, func() {
		_ = cs.Run()
	})
}

// A prefetch above the cap panics when it is requested, before it can evict anything.
func TestLoRAInstanceConfig_PrefetchAboveCapPanicsAtStart(t *testing.T) {
	cs := NewClusterSimulator(instanceConfigFixture(t, []int{8, 16}, []int{2, 2}, map[int][]string{0: {"a8"}, 1: {"b16"}}),
		NewSliceRequestSource(nil), nil)
	requirePanicContaining(t, `adapter "c64" has rank 64, above this instance's max_lora_rank 8`, func() {
		cs.Instances()[0].StartPrefetch(0, "c64")
	})
}

// ValidateLoRADeployment sizes every instance's KV, so a GPU with no memory figure or a
// reservation larger than the GPU is a validation error the CLI can report, not a
// construction panic.
func TestValidateLoRAInstanceConfig_SizesKV(t *testing.T) {
	dc := instanceConfigFixture(t, []int{16, 64}, []int{2, 1}, instCfgPlacement)
	dc.HWConfig.MemoryGiB = 0
	if err := ValidateLoRADeployment(dc); err == nil || !strings.Contains(err.Error(), "hardware MemoryGiB is 0") {
		t.Errorf("no GPU memory: got %v", err)
	}
	dc = instanceConfigFixture(t, []int{512, 64}, []int{100, 1}, instCfgPlacement)
	if err := ValidateLoRADeployment(dc); err == nil || !strings.Contains(err.Error(), "instance 0: per-instance KV sizing with max_lora_rank=512") {
		t.Errorf("oversized reservation: got %v", err)
	}
}

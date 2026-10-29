package cmd

import (
	"strings"
	"testing"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"
)

// The build site's per-pool engine settings (poolOverrides, resolvePoolConfig): a P/D pool's own
// settings are applied over the run's engine configuration, field by field, before the cluster
// builds that pool's instances (rolesOf). Moved from sim/cluster/resolve_test.go with the
// resolution itself (phase 2c), which the cluster no longer holds.

// poolTestGlobal is a run's engine configuration: TP 4 on H100, 5000 blocks of 16 tokens,
// max_model_len maxLen.
func poolTestGlobal(maxLen int64) engineConfig {
	return engineConfig{
		Horizon:             1000000,
		Seed:                42,
		KVCacheConfig:       cengine.NewKVCacheConfig(5000, 16, 0, 0, 0, 0),
		BatchConfig:         cengine.NewBatchConfig(256, 2048, 0),
		ModelHardwareConfig: cengine.NewModelHardwareConfig(cengine.ModelConfig{}, "test-model", "H100", 4, 1, false, maxLen),
	}
}

func intPtr(v int) *int     { return &v }
func i64Ptr(v int64) *int64 { return &v }
func boolPtr(v bool) *bool  { return &v }

// BC-P2-1: no overrides leave the run's configuration unchanged.
func TestResolvePoolConfig_NoOverrides_ReturnsGlobalUnchanged(t *testing.T) {
	global := poolTestGlobal(8192)
	got := resolvePoolConfig(global, poolOverrides{})
	if got.TP != global.TP || got.GPU != global.GPU || got.MaxModelLen != global.MaxModelLen ||
		got.TotalKVBlocks != global.TotalKVBlocks || got.Horizon != global.Horizon || got.Seed != global.Seed {
		t.Errorf("resolved %+v, want the global configuration unchanged", got)
	}
}

// BC-P2-2: each override applies, independently, and the fields no override names stay the run's.
func TestResolvePoolConfig_AllOverrides_Applied(t *testing.T) {
	global := poolTestGlobal(8192)
	o := poolOverrides{TP: intPtr(2), GPU: "A100", MaxModelLen: i64Ptr(4096), TotalKVBlocks: i64Ptr(3000),
		MaxNumSeqs: i64Ptr(64), MaxNumBatchedTokens: i64Ptr(1024), PrefixCachingDisabled: boolPtr(true),
		LongPrefillTokenThreshold: i64Ptr(512), Scheduler: "priority-fcfs", PreemptionPolicy: "priority"}
	got := resolvePoolConfig(global, o)
	for _, c := range []struct {
		name      string
		got, want any
	}{
		{"TP", got.TP, 2}, {"GPU", got.GPU, "A100"}, {"MaxModelLen", got.MaxModelLen, int64(4096)},
		{"TotalKVBlocks", got.TotalKVBlocks, int64(3000)}, {"MaxNumSeqs", got.MaxNumSeqs, int64(64)},
		{"MaxNumBatchedTokens", got.MaxNumBatchedTokens, int64(1024)},
		{"PrefixCachingDisabled", got.PrefixCachingDisabled, true},
		{"LongPrefillTokenThreshold", got.LongPrefillTokenThreshold, int64(512)},
		{"Scheduler", got.Scheduler, "priority-fcfs"}, {"PreemptionPolicy", got.PreemptionPolicy, "priority"},
		{"Horizon (not overridden)", got.Horizon, global.Horizon},
		{"BlockSizeTokens (not overridden)", got.BlockSizeTokens, global.BlockSizeTokens},
	} {
		if c.got != c.want {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}

// Only the fields an override names change.
func TestResolvePoolConfig_PartialOverrides_OnlySpecifiedFieldsChange(t *testing.T) {
	got := resolvePoolConfig(poolTestGlobal(8192), poolOverrides{TP: intPtr(8)})
	if got.TP != 8 || got.GPU != "H100" || got.MaxModelLen != 8192 || got.TotalKVBlocks != 5000 {
		t.Errorf("TP %d, GPU %q, MaxModelLen %d, TotalKVBlocks %d; want 8, H100, 8192, 5000",
			got.TP, got.GPU, got.MaxModelLen, got.TotalKVBlocks)
	}
}

// Resolving a pool's configuration leaves the run's own untouched: the result is a copy.
func TestResolvePoolConfig_DoesNotMutateGlobal(t *testing.T) {
	global := poolTestGlobal(0)
	_ = resolvePoolConfig(global, poolOverrides{TP: intPtr(8), TotalKVBlocks: i64Ptr(10)})
	if global.TP != 4 || global.TotalKVBlocks != 5000 {
		t.Errorf("the global configuration changed: TP %d, TotalKVBlocks %d", global.TP, global.TotalKVBlocks)
	}
}

// Resolve(Resolve(g, o), o) == Resolve(g, o) (R7's companion law to the value tests above).
func TestResolvePoolConfig_Idempotent(t *testing.T) {
	o := poolOverrides{TP: intPtr(8), GPU: "A100", MaxModelLen: i64Ptr(4096), TotalKVBlocks: i64Ptr(3000)}
	once := resolvePoolConfig(poolTestGlobal(8192), o)
	twice := resolvePoolConfig(once, o)
	if once.TP != twice.TP || once.GPU != twice.GPU || once.MaxModelLen != twice.MaxModelLen ||
		once.TotalKVBlocks != twice.TotalKVBlocks || once.Horizon != twice.Horizon ||
		once.BlockSizeTokens != twice.BlockSizeTokens {
		t.Errorf("not idempotent: once %+v, twice %+v", once, twice)
	}
}

// INV-P2-1: a pool whose KV pool is smaller than the run's takes its own capped max_model_len,
// which fits its pool.
func TestResolvePoolConfig_MaxModelLen_CappedToPoolKVCapacity(t *testing.T) {
	global := poolTestGlobal(131072)
	global.KVCacheConfig = cengine.NewKVCacheConfig(10000, 16, 0, 0, 0, 0)
	const poolBlocks, poolMax = int64(2048), int64(2048 * 16)
	got := resolvePoolConfig(global, poolOverrides{TotalKVBlocks: i64Ptr(poolBlocks), MaxModelLen: i64Ptr(poolMax)})
	if got.TotalKVBlocks != poolBlocks || got.MaxModelLen != poolMax {
		t.Fatalf("TotalKVBlocks %d, MaxModelLen %d; want %d, %d", got.TotalKVBlocks, got.MaxModelLen, poolBlocks, poolMax)
	}
	if need := (got.MaxModelLen + got.BlockSizeTokens - 1) / got.BlockSizeTokens; need > got.TotalKVBlocks {
		t.Errorf("max_model_len %d needs %d blocks, more than the pool's %d", got.MaxModelLen, need, got.TotalKVBlocks)
	}
}

// BC-3: the run's SLO priority overrides reach every pool's engine.
func TestResolvePoolConfig_SLOPriorityOverrides_Propagate(t *testing.T) {
	global := poolTestGlobal(0)
	global.SLOPriorityOverrides = map[string]int{"background": 10, "batch": 0}
	for _, o := range []poolOverrides{{}, {TP: intPtr(8)}, {TotalKVBlocks: i64Ptr(100)}} {
		got := resolvePoolConfig(global, o).SLOPriorityOverrides
		if got["background"] != 10 || got["batch"] != 0 {
			t.Errorf("overrides %+v: SLOPriorityOverrides = %v", o, got)
		}
	}
}

// The prefill and decode pools' settings do not bleed into each other (rolesOf).
func TestRolesOf_CrossPoolIsolation(t *testing.T) {
	r := rolesOf(poolTestGlobal(0), poolOverrides{TP: intPtr(8)}, poolOverrides{TP: intPtr(2)})
	if r.Prefill == nil || r.Decode == nil {
		t.Fatal("both pools stated settings, so both roles have a configuration")
	}
	if r.Prefill.TP != 8 || r.Decode.TP != 2 {
		t.Errorf("prefill TP %d, decode TP %d; want 8 and 2", r.Prefill.TP, r.Decode.TP)
	}
	if got := engineConfigIn(*r.Prefill).TP; got != 8 {
		t.Errorf("the prefill role's engine is built at TP %d, want 8", got)
	}
	if empty := rolesOf(poolTestGlobal(0), poolOverrides{}, poolOverrides{}); !empty.IsEmpty() {
		t.Error("pools that state nothing of their own take the run's configuration (no role configuration)")
	}
}

// Validate refuses each invalid setting a pool can state (R3), naming the pool, and accepts the
// valid ones and the empty settings.
func TestPoolOverrides_Validate(t *testing.T) {
	for _, c := range []struct {
		name string
		o    poolOverrides
		want string
	}{
		{"TP 0", poolOverrides{TP: intPtr(0)}, "TP must be > 0"},
		{"TP negative", poolOverrides{TP: intPtr(-1)}, "TP must be > 0"},
		{"MaxModelLen 0", poolOverrides{MaxModelLen: i64Ptr(0)}, "MaxModelLen must be > 0"},
		{"MaxModelLen negative", poolOverrides{MaxModelLen: i64Ptr(-1)}, "MaxModelLen must be > 0"},
		{"TotalKVBlocks 0", poolOverrides{TotalKVBlocks: i64Ptr(0)}, "TotalKVBlocks must be > 0"},
		{"TotalKVBlocks negative", poolOverrides{TotalKVBlocks: i64Ptr(-1)}, "TotalKVBlocks must be > 0"},
		{"MaxNumSeqs 0", poolOverrides{MaxNumSeqs: i64Ptr(0)}, "MaxNumSeqs must be > 0"},
		{"MaxNumBatchedTokens 0", poolOverrides{MaxNumBatchedTokens: i64Ptr(0)}, "MaxNumBatchedTokens must be > 0"},
		{"KVTransferTicksPerBlock negative", poolOverrides{KVTransferTicksPerBlock: i64Ptr(-1)}, "KVTransferTicksPerBlock must be >= 0"},
		{"LongPrefillTokenThreshold negative", poolOverrides{LongPrefillTokenThreshold: i64Ptr(-1)}, "LongPrefillTokenThreshold must be >= 0"},
		{"an unknown DP balancer", poolOverrides{DPLoadBalancing: "random"}, "is not internal, hybrid or external"},
	} {
		err := c.o.Validate("test pool")
		if err == nil || !strings.Contains(err.Error(), c.want) || !strings.Contains(err.Error(), "test pool") {
			t.Errorf("%s: Validate() = %v, want an error naming the pool and %q", c.name, err, c.want)
		}
	}
	for _, o := range []poolOverrides{{}, {TP: intPtr(4)}, {MaxModelLen: i64Ptr(8192)}, {TotalKVBlocks: i64Ptr(5000)},
		{TP: intPtr(4), GPU: "H100", MaxModelLen: i64Ptr(8192), TotalKVBlocks: i64Ptr(5000), DPLoadBalancing: "hybrid"}} {
		if err := o.Validate("test pool"); err != nil {
			t.Errorf("Validate(%+v) = %v, want nil", o, err)
		}
	}
}

// IsEmpty is true only when no setting is stated (each field's branch).
func TestPoolOverrides_IsEmpty(t *testing.T) {
	if !(poolOverrides{}).IsEmpty() {
		t.Error("empty settings: IsEmpty() = false")
	}
	for name, o := range map[string]poolOverrides{
		"TP": {TP: intPtr(4)}, "GPU": {GPU: "A100"}, "MaxModelLen": {MaxModelLen: i64Ptr(8192)},
		"TotalKVBlocks": {TotalKVBlocks: i64Ptr(5000)}, "MaxNumSeqs": {MaxNumSeqs: i64Ptr(1)},
		"PrefixCachingDisabled": {PrefixCachingDisabled: boolPtr(false)}, "Scheduler": {Scheduler: "fcfs"},
		"DPLoadBalancing": {DPLoadBalancing: "hybrid"},
	} {
		if o.IsEmpty() {
			t.Errorf("%s set: IsEmpty() = true", name)
		}
	}
}

// Original to blis-vllm.

package build

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// A P/D pool's settings are its own kernel's and its own scenario pool's: the sizing the kernel
// resolves (KV blocks, window, caps, prefix caching, the sequence state), the pool's
// parallelism and hardware, its own long-prefill threshold and DP balancing mode, and its own
// scheduling policy. Two pools that state different knobs get different settings.
func TestPoolSettingsOf_AreThePoolsOwn(t *testing.T) {
	repos := pdVariant(t,
		"      long_prefill_token_threshold: 512\n      scheduling_policy: priority\n",
		"      enable_prefix_caching: false\n")
	hit := engine.KVPrefixHit{}
	for _, c := range []struct {
		role deployment.Role
		want func(t *testing.T, ps PoolSettings)
	}{
		{deployment.RolePrefill, func(t *testing.T, ps PoolSettings) {
			if ps.LongPrefillTokenThreshold != 512 || ps.Policy != (EnginePolicy{Scheduler: "priority-fcfs", Preemption: "priority"}) ||
				ps.PrefixCachingDisabled {
				t.Errorf("prefill: %+v; want its own threshold 512, priority policy, prefix caching on", ps)
			}
		}},
		{deployment.RoleDecode, func(t *testing.T, ps PoolSettings) {
			if ps.LongPrefillTokenThreshold != 0 || ps.Policy != (EnginePolicy{Scheduler: "fcfs", Preemption: "fcfs"}) ||
				!ps.PrefixCachingDisabled {
				t.Errorf("decode: %+v; want vLLM's defaults (no threshold, fcfs) and its own prefix caching off", ps)
			}
		}},
	} {
		m := openRole(t, repos, c.role)
		ps, err := PoolSettingsOf("variant.yaml", m, 0, hit, EngineKnobFlags{})
		if err != nil {
			t.Fatalf("%s: %v", c.role, err)
		}
		s, err := m.Settings()
		if err != nil {
			t.Fatal(err)
		}
		d := m.Deployment()
		if ps.TP != d.TP || ps.GPU != d.Hardware || ps.TotalKVBlocks != s.KVBlocks || ps.MaxModelLen != int64(s.MaxModelLen) ||
			ps.MaxNumSeqs != int64(s.MaxNumSeqs) || ps.MaxNumBatchedTokens != int64(s.MaxNumBatchedTokens) ||
			ps.PrefixCachingDisabled != s.PrefixCachingDisabled || ps.DPLoadBalancing != string(m.DPLoadBalancing().Mode()) {
			t.Errorf("%s: %+v does not carry the kernel's sizing and the pool's deployment (%+v, %+v)", c.role, ps, s, d)
		}
		if (ps.SequenceState.Hold == nil) != (s.SequenceState.Hold == nil) {
			t.Errorf("%s: the sequence state was not carried", c.role)
		}
		c.want(t, ps)
	}
}

// A pool whose model holds bounded per-sequence state (DeepSeek-V4.1's window rings and
// compressor state) carries it, as the pool's kernel sizes it.
func TestPoolSettingsOf_CarriesTheSequenceState(t *testing.T) {
	m, err := kernelmodel.OpenRole("deepseek-v4.1-flash-h100-1p1d-dspark5.yaml", vendoredRepos(t), deployment.RoleDecode)
	if err != nil {
		t.Fatal(err)
	}
	s, err := m.Settings()
	if err != nil {
		t.Fatal(err)
	}
	if s.SequenceState.Hold == nil {
		t.Fatal("non-vacuity: the fixture's decode pool holds no sequence state")
	}
	ps, err := PoolSettingsOf("deepseek-v4.1-flash-h100-1p1d-dspark5.yaml", m, 0, engine.KVPrefixHit{}, EngineKnobFlags{})
	if err != nil {
		t.Fatal(err)
	}
	if ps.SequenceState.Hold == nil || ps.SequenceState.Gate == nil {
		t.Errorf("the pool's sequence state was dropped: %+v", ps.SequenceState)
	}
	if len(ps.Notes) == 0 {
		t.Error("a pool with a sequence state reports its fixed charge as a startup note")
	}
}

// The LoRA reservation comes out of the pool's KV budget, and the run's balancing flag, when
// given, sets the mode for every pool.
func TestPoolSettingsOf_ReservationAndFlag(t *testing.T) {
	repos := pdVariant(t, "", "")
	m := openRole(t, repos, deployment.RoleDecode)
	full, err := PoolSettingsOf("variant.yaml", m, 0, engine.KVPrefixHit{}, EngineKnobFlags{})
	if err != nil {
		t.Fatal(err)
	}
	s, err := m.Settings()
	if err != nil {
		t.Fatal(err)
	}
	reserve := 64 * s.KVBytesPerBlock
	less, err := PoolSettingsOf("variant.yaml", m, reserve, engine.KVPrefixHit{}, EngineKnobFlags{DPLoadBalancing: "external", DPLoadBalancingFromFlag: true})
	if err != nil {
		t.Fatal(err)
	}
	if less.TotalKVBlocks >= full.TotalKVBlocks {
		t.Errorf("a %d-byte reservation left %d blocks of %d; want fewer", reserve, less.TotalKVBlocks, full.TotalKVBlocks)
	}
	if less.DPLoadBalancing != "external" || full.DPLoadBalancing != "internal" {
		t.Errorf("modes %q (flag) and %q (none); want external and the pool's internal", less.DPLoadBalancing, full.DPLoadBalancing)
	}
	if _, err := PoolSettingsOf("variant.yaml", m, 1<<62, engine.KVPrefixHit{}, EngineKnobFlags{}); err == nil ||
		!strings.Contains(err.Error(), "decode pool") {
		t.Errorf("a reservation larger than the device: %v, want a refusal naming the decode pool", err)
	}
}

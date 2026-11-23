// Original to blis-vllm.

package tiering

import (
	"strings"
	"testing"
)

// The backpressure detector on virtual time (EMABackpressureDetector,
// vllm/v1/kv_offload/tiering/backpressure.py at vLLM v0.33.0). Every time here is the
// simulator's clock, which the test sets.

type vclock struct{ now float64 }

func (c *vclock) at() float64 { return c.now }

func detector(t *testing.T, cfg map[string]any, tier string) (*EMADetector, *vclock) {
	t.Helper()
	c := &vclock{}
	cfg["backpressure_cls"] = "EMABackpressureDetector"
	d, err := NewBackpressure(cfg, tier, "", c.at)
	if err != nil {
		t.Fatal(err)
	}
	return d, c
}

// store completes a store of one MiB that took took seconds of virtual time.
func store(d *EMADetector, c *vclock, took float64) {
	c.now += took
	d.Update(c.now-took, mib)
}

// Slow stores put the tier under pressure once the warmup is over, and DropStorePolicy then
// drops every store (backpressure.py:55-59, 341-366).
func TestSlowStoresPutTheTierUnderPressureAndDropStores(t *testing.T) {
	d, c := detector(t, map[string]any{"policy_cls": "DropStorePolicy", "warmup_completions": 2}, "fs")
	store(d, c, 0.001)
	store(d, c, 0.001)
	if !d.ShouldStore(4) {
		t.Fatal("after a fast warmup: dropped, want stored")
	}
	// fs's local water marks: high 0.010 s/MiB. One store at 0.2 s/MiB lifts the EMA past it.
	store(d, c, 0.2)
	if !d.IsUnderPressure() || d.ShouldStore(4) {
		t.Fatal("after a slow store: not under pressure or stored, want pressure and a drop")
	}
	if s, b := d.PopStoresDropped(); s != 1 || b != 4 {
		t.Errorf("dropped %d stores of %d blocks, want 1 of 4", s, b)
	}
}

// With no completions, the EMA halves every decay_half_life_s of virtual time until it is below
// the low water mark, and pressure clears (:325-339, 386-396). Then every second store in the
// cooldown is dropped (:410-423).
func TestPressureDecaysOnVirtualTimeAndTheCooldownDropsEverySecondStore(t *testing.T) {
	d, c := detector(t, map[string]any{"policy_cls": "DropStorePolicy", "warmup_completions": 0}, "fs")
	store(d, c, 0.2) // EMA 0.03 s/MiB, high 0.010, low 0.002
	if !d.IsUnderPressure() {
		t.Fatal("not under pressure after a slow store")
	}
	// 0.03 halves to below 0.002 after four half-lives (8 s); after 6 s it is still 0.00375.
	c.now += 6
	if !d.IsUnderPressure() {
		t.Fatal("pressure cleared after 6 s, want it held until the EMA is below the low water mark")
	}
	c.now += 2
	if d.IsUnderPressure() {
		t.Fatal("pressure held after 8 s of idle decay, want it cleared")
	}
	got := []bool{d.ShouldStore(1), d.ShouldStore(1), d.ShouldStore(1), d.ShouldStore(1)}
	if want := []bool{true, false, true, false}; !equalBools(got, want) {
		t.Errorf("stores in the cooldown %v, want %v", got, want)
	}
	c.now += 1.5 // past cooldown_s (1 s)
	if got := []bool{d.ShouldStore(1), d.ShouldStore(1)}; !equalBools(got, []bool{true, true}) {
		t.Errorf("stores after the cooldown %v, want both stored", got)
	}
}

// Three failed stores in a row put a storage tier under pressure; a tier with no
// failure_threshold (p2p) ignores failures (:259-273, 368-384).
func TestConsecutiveFailuresPutAStorageTierUnderPressure(t *testing.T) {
	for _, tc := range []struct {
		tier string
		want bool
	}{{"fs", true}, {"obj", true}, {"p2p", false}} {
		d, _ := detector(t, map[string]any{}, tc.tier)
		d.OnStoreFailed()
		d.OnStoreFailed()
		if d.IsUnderPressure() {
			t.Errorf("%s: under pressure after two failures, want three", tc.tier)
		}
		d.OnStoreFailed()
		if got := d.IsUnderPressure(); got != tc.want {
			t.Errorf("%s: under pressure %v after three failures, want %v", tc.tier, got, tc.want)
		}
	}
}

// The factory refuses what vLLM would raise on (factory.py:73-90; backpressure.py:275-323).
func TestTheFactoryRefusesWhatVLLMRaisesOn(t *testing.T) {
	for _, tc := range []struct {
		cfg  map[string]any
		want string
	}{
		{map[string]any{}, "backpressure_cls"},
		{map[string]any{"backpressure_cls": "BackpressureDetector"}, "not a detector"},
		{map[string]any{"backpressure_cls": "EMABackpressureDetector", "policy_cls": "DropAccountingPolicy"}, "policy_cls"},
		{map[string]any{"backpressure_cls": "EMABackpressureDetector", "alpha": 0.0}, "alpha"},
		{map[string]any{"backpressure_cls": "EMABackpressureDetector", "low_water_s": 1.0}, "low_water_s"},
		{map[string]any{"backpressure_cls": "EMABackpressureDetector", "failure_threshold": 0}, "failure_threshold"},
		{map[string]any{"backpressure_cls": "EMABackpressureDetector", "ramp_factor": 3.0}, "ramp_factor"},
	} {
		if _, err := NewBackpressure(tc.cfg, "fs", "", nil); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%v: %v, want an error naming %q", tc.cfg, err, tc.want)
		}
	}
	// A REMOTE fs tier takes the network water marks.
	d, err := NewBackpressure(map[string]any{"backpressure_cls": "EMABackpressureDetector"}, "fs", "remote", nil)
	if err != nil || d.high != 0.050 || d.low != 0.010 {
		t.Errorf("remote fs: %v, high %v low %v, want 0.050 and 0.010", err, d.high, d.low)
	}
}

func equalBools(a, b []bool) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Original to blis-sglang.

package sched

import (
	"reflect"
	"testing"
)

// --prefill-decode-interval N makes the scheduler, after it launches an extend batch, run N rounds
// without admitting new prefill (scheduler.py:1400-1426,3972,4015@297e54fd). The deferral applies
// only while the running batch holds at least --prefill-decode-interval-min-bs requests; a smaller
// running batch admits prefill every round. The rules are the source's, so the tests state them
// as laws of the mode sequence rather than as recorded values.

// modeSequence feeds the named request at each step ("" for none), then idles, and returns the mode each step
// launched ("-" when it launched nothing). The requests are long enough to keep decoding
// throughout.
func modeSequence(t *testing.T, interval, minBS int, arrivals []string, steps int) []Mode {
	t.Helper()
	cfg := baseConfig(t)
	cfg.EnableOverlap = false
	cfg.MaxTotalNumTokens, cfg.MaxReqLen, cfg.MaxReqInputLen = 8192, 4000, 3900
	cfg.PrefillDecodeInterval = interval
	cfg.PrefillDecodeIntervalMinBs = minBS
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	var modes []Mode
	for i := 0; i < steps; i++ {
		var arr []Arrival
		if i < len(arrivals) && arrivals[i] != "" {
			arr = []Arrival{{RID: arrivals[i], Prompt: []int64{1, 2, 3, int64(10 + i)}, MaxNewTokens: 100, IgnoreEOS: true}}
		}
		rec := mustStep(t, s, arr...)
		if len(rec.Forwards) == 0 {
			modes = append(modes, "-")
			continue
		}
		modes = append(modes, rec.Forwards[0].Mode)
	}
	return modes
}

func TestIntervalZeroAdmitsPrefillEveryRound(t *testing.T) {
	got := modeSequence(t, 0, 0, []string{"a", "b", "c"}, 4)
	want := []Mode{Extend, Extend, Extend, Decode}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("modes %v, want %v", got, want)
	}
}

// After an extend, a waiting request waits exactly N decode rounds, and then is prefilled.
func TestIntervalDefersPrefillForNDecodeRounds(t *testing.T) {
	for _, n := range []int{1, 3} {
		arrivals := []string{"a", "b"}
		got := modeSequence(t, n, 0, arrivals, 3+n)
		want := []Mode{Extend}
		for i := 0; i < n; i++ {
			want = append(want, Decode)
		}
		want = append(want, Extend)
		if !reflect.DeepEqual(got[:len(want)], want) {
			t.Fatalf("interval %d: modes %v, want a prefix %v", n, got, want)
		}
	}
}

// The wait is counted from the latest extend: the deferred request's own prefill re-arms it.
func TestIntervalIsRearmedByEachExtend(t *testing.T) {
	got := modeSequence(t, 2, 0, []string{"a", "b", "", "", "c"}, 8)
	// a: Extend; b waits 2 decode rounds, Extend; c, arriving after that, waits 2 more, Extend.
	want := []Mode{Extend, Decode, Decode, Extend, Decode, Decode, Extend, Decode}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("modes %v, want %v", got, want)
	}
}

// A running batch smaller than the minimum admits prefill every round, and the count is dropped.
func TestIntervalMinBatchSizeAdmitsPrefillToSmallRunningBatches(t *testing.T) {
	// With one request running, below the minimum of two, b is prefilled at once; with two running,
	// the interval holds c back for its three rounds.
	got := modeSequence(t, 3, 2, []string{"a", "b", "c"}, 6)
	want := []Mode{Extend, Extend, Decode, Decode, Decode, Extend}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("modes %v, want %v", got, want)
	}
}

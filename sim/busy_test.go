package sim

import (
	"math"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

func batchOf(reqs ...engine.Scheduled) engine.Batch { return engine.Batch{Reqs: reqs} }

// A prompt chunk is prefill, a token past the prompt is decode, and a step that holds both
// charges its time to each by its tokens (inference-sim#2081).
func TestBusySplitsAStepByItsTokens(t *testing.T) {
	var l BusyLedger
	l.Record(batchOf(
		engine.Scheduled{ID: "a", Tokens: 300, Computed: 0, PromptLen: 1000},   // 300 prefill
		engine.Scheduled{ID: "b", Tokens: 100, Computed: 950, PromptLen: 1000}, // 50 prefill, 50 decode
		engine.Scheduled{ID: "c", Tokens: 50, Computed: 2000, PromptLen: 1000}, // 50 decode
	), 0, 1000)
	got := l.Over(0, 1000)
	if got.BusyUs != 1000 {
		t.Fatalf("busy %v us, want 1000", got.BusyUs)
	}
	if math.Abs(got.PrefillUs-1000*350.0/450) > 1e-9 || math.Abs(got.DecodeUs-1000*100.0/450) > 1e-9 {
		t.Errorf("prefill %v, decode %v us; want 350/450 and 100/450 of 1000 (350 of 450 tokens are prefill)", got.PrefillUs, got.DecodeUs)
	}
}

// Idle time is not busy, a step is counted only inside the window, and steps that overlap
// (pipeline stages) count their overlap once.
func TestBusyClipsToTheWindowAndCountsOverlapOnce(t *testing.T) {
	var l BusyLedger
	decode := batchOf(engine.Scheduled{ID: "a", Tokens: 1, Computed: 10, PromptLen: 10})
	l.Record(decode, 0, 100)
	l.Record(decode, 50, 150) // started while the first still ran
	l.Record(decode, 300, 400)
	if got := l.Over(0, 1000).BusyUs; got != 250 {
		t.Errorf("busy %v us over the run, want 250 (0-150 once and 300-400)", got)
	}
	if got := l.Over(100, 350).BusyUs; got != 100 {
		t.Errorf("busy %v us over 100-350, want 100 (100-150 and 300-350)", got)
	}
	if got := l.Over(150, 300).BusyUs; got != 0 {
		t.Errorf("busy %v us over the idle gap, want 0", got)
	}
}

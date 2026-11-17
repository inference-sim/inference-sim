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
	l.Record(nil, batchOf(
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
	l.Record(nil, decode, 0, 100)
	l.Record(nil, decode, 50, 150) // started while the first still ran
	l.Record(nil, decode, 300, 400)
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

// flopEngine states a fixed FLOP count per token and a peak, as an engine.Flopped does.
type flopEngine struct {
	engine.Engine
	perToken, peak float64
}

func (f flopEngine) StepFLOPs(b engine.Batch) float64 { return f.perToken * float64(b.Tokens()) }
func (f flopEngine) PeakFLOPsPerGPU() float64         { return f.peak }

// An engine that states FLOPs has each step's counted, in proportion to the part of the step
// inside the window; one that does not states none and no peak.
func TestBusyCountsFLOPsInProportionToTheWindow(t *testing.T) {
	var l BusyLedger
	b := batchOf(engine.Scheduled{ID: "a", Tokens: 100, Computed: 0, PromptLen: 100})
	l.Record(flopEngine{perToken: 1e9, peak: 5e14}, b, 0, 1000)
	got := l.Over(0, 500)
	if got.FLOPs != 5e10 || got.PeakFLOPsPerGPU != 5e14 {
		t.Errorf("half the step: %v FLOPs, peak %v; want 5e10 and 5e14", got.FLOPs, got.PeakFLOPsPerGPU)
	}
	var none BusyLedger
	none.Record(nil, b, 0, 1000)
	if g := none.Over(0, 1000); g.FLOPs != 0 || g.PeakFLOPsPerGPU != 0 {
		t.Errorf("an engine with no FLOPs gave %v FLOPs, peak %v", g.FLOPs, g.PeakFLOPsPerGPU)
	}
}

// sim/busy.go: how long an instance's engine spent running steps, and on what (inference-sim
// #2081). Throughput and requests in flight say nothing of whether the GPUs were working: an
// agentic session is in flight while it waits on a tool. The engine driver already knows when
// every step starts and ends, so it records them here.
package sim

import "github.com/inference-sim/blis-substrate/contracts/engine"

// BusyStep is one engine step that ran a batch: when it started and ended on the simulated
// clock, and how many of the tokens it scheduled were prefill and decode.
type BusyStep struct {
	StartUs, EndUs int64
	// PrefillTokens are scheduled tokens of requests still computing their prompt; DecodeTokens
	// are the rest, speculative drafts included.
	PrefillTokens, DecodeTokens int
	// FLOPs are the step's model FLOPs, zero when the engine states none (engine.Flopped).
	FLOPs float64
}

// BusyLedger is an instance's steps, in the order the engine started them.
type BusyLedger struct {
	steps []BusyStep
	// end is the latest step end recorded. Pipeline stages let a step start before the one
	// ahead of it ends; the overlap counts once.
	end int64
	// peak is one GPU's datasheet dense peak in FLOP/s, as the engine stated it with the steps'
	// FLOPs; zero when it states none.
	peak float64
}

// Busy is a ledger's time over an interval, in microseconds. PrefillUs and DecodeUs add up to
// BusyUs up to rounding.
type Busy struct {
	BusyUs, PrefillUs, DecodeUs float64
	// FLOPs are the model FLOPs of the steps over the interval, each step's in proportion to
	// the part of it inside; PeakFLOPsPerGPU is one GPU's peak, both zero when the engine states
	// none.
	FLOPs, PeakFLOPsPerGPU float64
}

// BusyOf splits a batch's scheduled tokens into prefill and decode: a request's tokens are
// prefill while it has prompt left to compute (Computed below PromptLen), up to that prompt's
// end.
func BusyOf(b engine.Batch) (prefill, decode int) {
	for _, s := range b.Reqs {
		p := max(0, min(s.Tokens, s.PromptLen-s.Computed))
		prefill += p
		decode += s.Tokens - p
	}
	return prefill, decode
}

// Record adds the step of batch b that ran from start to end. An engine that states model
// FLOPs (engine.Flopped) is asked for the step's.
func (l *BusyLedger) Record(eng engine.Engine, b engine.Batch, start, end int64) {
	start = max(start, l.end)
	if end <= start {
		return
	}
	pf, dc := BusyOf(b)
	var flops float64
	if f, ok := eng.(engine.Flopped); ok {
		if l.peak = f.PeakFLOPsPerGPU(); l.peak > 0 {
			flops = f.StepFLOPs(b)
		}
	}
	l.steps = append(l.steps, BusyStep{StartUs: start, EndUs: end, PrefillTokens: pf, DecodeTokens: dc, FLOPs: flops})
	l.end = end
}

// Over is the busy time inside [startUs, endUs): each step is clipped to the interval, and its
// time is split between prefill and decode by the step's scheduled tokens.
func (l *BusyLedger) Over(startUs, endUs int64) Busy {
	out := Busy{PeakFLOPsPerGPU: l.peak}
	for _, s := range l.steps {
		a, z := max(s.StartUs, startUs), min(s.EndUs, endUs)
		if z <= a {
			continue
		}
		d := float64(z - a)
		out.BusyUs += d
		out.FLOPs += s.FLOPs * d / float64(s.EndUs-s.StartUs)
		if n := s.PrefillTokens + s.DecodeTokens; n > 0 {
			out.PrefillUs += d * float64(s.PrefillTokens) / float64(n)
			out.DecodeUs += d * float64(s.DecodeTokens) / float64(n)
		}
	}
	return out
}

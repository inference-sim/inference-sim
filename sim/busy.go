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
}

// BusyLedger is an instance's steps, in the order the engine started them.
type BusyLedger struct {
	steps []BusyStep
	// end is the latest step end recorded. Pipeline stages let a step start before the one
	// ahead of it ends; the overlap counts once.
	end int64
}

// Busy is a ledger's time over an interval, in microseconds. PrefillUs and DecodeUs add up to
// BusyUs up to rounding.
type Busy struct {
	BusyUs, PrefillUs, DecodeUs float64
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

// Record adds the step of batch b that ran from start to end.
func (l *BusyLedger) Record(b engine.Batch, start, end int64) {
	start = max(start, l.end)
	if end <= start {
		return
	}
	pf, dc := BusyOf(b)
	l.steps = append(l.steps, BusyStep{StartUs: start, EndUs: end, PrefillTokens: pf, DecodeTokens: dc})
	l.end = end
}

// Over is the busy time inside [startUs, endUs): each step is clipped to the interval, and its
// time is split between prefill and decode by the step's scheduled tokens.
func (l *BusyLedger) Over(startUs, endUs int64) Busy {
	var out Busy
	for _, s := range l.steps {
		a, z := max(s.StartUs, startUs), min(s.EndUs, endUs)
		if z <= a {
			continue
		}
		d := float64(z - a)
		out.BusyUs += d
		if n := s.PrefillTokens + s.DecodeTokens; n > 0 {
			out.PrefillUs += d * float64(s.PrefillTokens) / float64(n)
			out.DecodeUs += d * float64(s.DecodeTokens) / float64(n)
		}
	}
	return out
}

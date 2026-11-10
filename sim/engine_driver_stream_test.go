package sim

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
)

// streamRun drives three overlapping requests (output 25, 40 and 7 tokens) through the fake FIFO
// engine with the given stream interval and returns the requests and the driver.
func streamRun(t *testing.T, interval int) ([]*Request, *EngineDriver) {
	t.Helper()
	eng, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 256, StepBaseUs: 100, StepPerTokenUs: 1})
	if err != nil {
		t.Fatal(err)
	}
	d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40, OutputTokenProcessingTime: 5, StreamInterval: interval})
	reqs := []*Request{
		{ID: "a", InputTokens: tokens(10), OutputTokens: tokens(25), ArrivalTime: 0},
		{ID: "b", InputTokens: tokens(12), OutputTokens: tokens(40), ArrivalTime: 50},
		{ID: "c", InputTokens: tokens(8), OutputTokens: tokens(7), ArrivalTime: 400},
	}
	for _, r := range reqs {
		d.InjectArrival(r)
	}
	d.Run()
	if d.Metrics.CompletedRequests != len(reqs) {
		t.Fatalf("interval %d: completed %d of %d", interval, d.Metrics.CompletedRequests, len(reqs))
	}
	return reqs, d
}

// vLLM's rule (output_processor.py:324-343@5df63a72): the first output at once, then a group each
// time stream_interval tokens have accumulated, and the rest at the finish. A request alone with
// 25 tokens at interval 10 is sent 1, then 10 (tokens 2-11), 10 (12-21) and 4 (22-25): after the
// first token, three client ITL entries, each the steps since the last send plus the group's
// output processing. The engine's own per-step entries are the 24 single steps of the default.
func TestEngineDriver_StreamIntervalGroupsDelivery(t *testing.T) {
	eng, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 64, StepBaseUs: 100, StepPerTokenUs: 1})
	if err != nil {
		t.Fatal(err)
	}
	d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40, OutputTokenProcessingTime: 5, StreamInterval: 10})
	req := &Request{ID: "r", InputTokens: tokens(10), OutputTokens: tokens(25), ArrivalTime: 0}
	d.InjectArrival(req)
	d.Run()
	want := []int64{10*101 + 10*5, 10*101 + 10*5, 4*101 + 4*5}
	if !reflect.DeepEqual(req.ClientITL, want) {
		t.Errorf("ClientITL = %v, want %v", req.ClientITL, want)
	}
	if len(req.ITL) != 24 {
		t.Errorf("per-step ITL entries = %d, want 24: the engine's steps do not change", len(req.ITL))
	}
	if req.FirstTokenTime != 115 {
		t.Errorf("TTFT = %d, want 115: the first output is sent at once", req.FirstTokenTime)
	}
	if got, w := d.Metrics.RequestE2Es["r"], float64(115+24*101+24*5); got != w {
		t.Errorf("E2E = %v, want %v: grouping moves no token's production", got, w)
	}
}

// At the stated InferenceX values the scheduler, the first tokens, every request's end, the
// per-step intervals and the request's mean ITL are those of the default, and the client's ITL
// entries are fewer and longer. The same run at 0 and 1 is identical and records no client
// entries (X13): this fails if the interval is not applied, or if it leaks into a figure.
func TestEngineDriver_StreamIntervalMovesClientITLAndLeavesTheRestAlone(t *testing.T) {
	base, bd := streamRun(t, 1)
	zero, zd := streamRun(t, 0)
	for i := range base {
		if !reflect.DeepEqual(base[i].ITL, zero[i].ITL) || base[i].ClientITL != nil || zero[i].ClientITL != nil {
			t.Fatalf("interval 0 and 1 differ, or record client entries, for %s", base[i].ID)
		}
	}
	if bd.stepIdx != zd.stepIdx {
		t.Fatalf("steps at 0 and 1: %d and %d", zd.stepIdx, bd.stepIdx)
	}
	for _, s := range []int{10, 20, 32} {
		got, gd := streamRun(t, s)
		if gd.stepIdx != bd.stepIdx {
			t.Errorf("interval %d: %d engine steps, %d at the default: scheduling changed", s, gd.stepIdx, bd.stepIdx)
		}
		if !reflect.DeepEqual(gd.Metrics.AllITLs, bd.Metrics.AllITLs) || !reflect.DeepEqual(gd.Metrics.StepOwners, bd.Metrics.StepOwners) {
			t.Errorf("interval %d: the per-step interval series moved", s)
		}
		var clients, steps int
		for i := range base {
			b, g := base[i], got[i]
			if g.FirstTokenTime != b.FirstTokenTime {
				t.Errorf("interval %d, %s: TTFT %d, want %d", s, g.ID, g.FirstTokenTime, b.FirstTokenTime)
			}
			if g.DepartureTime != b.DepartureTime {
				t.Errorf("interval %d, %s: departure %d, want %d", s, g.ID, g.DepartureTime, b.DepartureTime)
			}
			if !reflect.DeepEqual(g.ITL, b.ITL) {
				t.Errorf("interval %d, %s: the per-step ITL entries moved", s, g.ID)
			}
			if gd.Metrics.RequestE2Es[g.ID] != bd.Metrics.RequestE2Es[b.ID] || gd.Metrics.RequestITLs[g.ID] != bd.Metrics.RequestITLs[b.ID] {
				t.Errorf("interval %d, %s: E2E or the request's mean ITL (its TPOT) moved", s, g.ID)
			}
			var sb, sg int64
			for _, v := range b.ITL {
				sb += v
			}
			for _, v := range g.ClientITL {
				sg += v
			}
			if sb != sg {
				t.Errorf("interval %d, %s: client entries sum to %d, want the request's %d", s, g.ID, sg, sb)
			}
			clients += len(g.ClientITL)
			steps += len(b.ITL)
		}
		if clients >= steps {
			t.Errorf("interval %d: %d client entries for %d engine steps; grouping must give fewer", s, clients, steps)
		}
		var longest, longestStep int64
		for i := range base {
			for _, v := range got[i].ClientITL {
				longest = max(longest, v)
			}
			for _, v := range base[i].ITL {
				longestStep = max(longestStep, v)
			}
		}
		if longest <= longestStep {
			t.Errorf("interval %d: longest client gap %d, longest step %d; it must grow", s, longest, longestStep)
		}
	}
}

// A request that finishes before one interval has accumulated is sent its tokens at the finish,
// as one group: the first output at once and the rest together.
func TestEngineDriver_StreamIntervalLongerThanTheOutputSendsTheRestAtTheFinish(t *testing.T) {
	got, _ := streamRun(t, 32)
	c := got[2] // seven output tokens
	if len(c.ClientITL) != 1 {
		t.Fatalf("ClientITL = %v, want one entry: the 6 tokens after the first, sent at the finish", c.ClientITL)
	}
}

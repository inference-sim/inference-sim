// Original to blis-sglang.

package sched

import (
	"fmt"
	"strings"
	"testing"
)

// Speculative decoding at top-k 1 (S30, inference-sim/inference-sim#2355). The scenario states how many
// of a pass's drafts the verify accepts; the scheduler decides what that does to the request's
// output, its committed KV and the KV the next pass reserves.

const (
	specSteps = 3
	specDraft = specSteps + 1 // num_draft_tokens at top-k 1
	specEOS   = int64(151643)
)

func specConfig(t *testing.T, pageSize int) Config {
	t.Helper()
	cfg := baseConfig(t)
	cfg.ChunkedPrefillSize = nil
	cfg.MaxTotalNumTokens = 4096
	cfg.PageSize = pageSize
	cfg.Allocator = "TokenToKVPoolAllocator"
	if pageSize > 1 {
		cfg.Allocator = "PagedTokenToKVPoolAllocator"
	}
	cfg.SpecNumSteps, cfg.SpecNumDraftTokens = specSteps, specDraft
	return cfg
}

func specScript(n int) []int64 {
	out := make([]int64, n)
	for i := range out {
		out[i] = int64(100 + i)
	}
	return out
}

func specPrompt(n int) []int64 {
	out := make([]int64, n)
	for i := range out {
		out[i] = int64(1 + i)
	}
	return out
}

func mustLaunch(t *testing.T, s *Scheduler) *Launched {
	t.Helper()
	l, err := s.Launch()
	if err != nil || l == nil {
		t.Fatalf("no pass launched: %v", err)
	}
	return l
}

func mustComplete(t *testing.T, s *Scheduler, l *Launched, accepted map[string]int) Completion {
	t.Helper()
	c, err := s.Complete(l, accepted)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// started runs the prefill of one request and returns the scheduler and the request: it holds its
// first output token, the bonus the first verify pass feeds.
func started(t *testing.T, cfg Config, promptLen, maxNew int, sc []int64) (*Scheduler, *Req) {
	t.Helper()
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	s.handleGenerateRequest(Arrival{RID: "a", Prompt: specPrompt(promptLen), MaxNewTokens: maxNew, Script: sc})
	c := mustComplete(t, s, mustLaunch(t, s), nil)
	if c.Emitted["a"] != 1 {
		t.Fatalf("the prefill emitted %v, want one token", c.Emitted)
	}
	return s, s.last.Reqs[0]
}

// A verify pass emits the accepted drafts and the one token the target samples after them, and
// commits that many positions (eagle_worker_v2.py _resolve_spec_v2_tokens: accept_lens = correct
// drafts + 1; batch_result_processor.py). Absent from the outcome means none accepted.
func TestVerifyEmitsAcceptedDraftsPlusOne(t *testing.T) {
	for _, ps := range []int{1, 16} {
		for accepted := 0; accepted <= specDraft-1; accepted++ {
			t.Run(fmt.Sprintf("page%d/accept%d", ps, accepted), func(t *testing.T) {
				s, r := started(t, specConfig(t, ps), 50, 1000, specScript(200))
				committed := r.C.KV.KVCommittedLen
				c := mustComplete(t, s, mustLaunch(t, s), map[string]int{"a": accepted})
				if got := c.Emitted["a"]; got != accepted+1 {
					t.Fatalf("emitted %d tokens, want %d", got, accepted+1)
				}
				if got := r.C.KV.KVCommittedLen - committed; got != accepted+1 {
					t.Fatalf("committed advanced %d, want %d", got, accepted+1)
				}
				if got := len(r.C.OutputIDs); got != 1+accepted+1 {
					t.Fatalf("output holds %d tokens, want %d", got, 1+accepted+1)
				}
			})
		}
	}
	t.Run("no outcome accepts none", func(t *testing.T) {
		s, _ := started(t, specConfig(t, 1), 50, 1000, specScript(200))
		if c := mustComplete(t, s, mustLaunch(t, s), nil); c.Emitted["a"] != 1 {
			t.Fatalf("emitted %d tokens without an acceptance, want 1", c.Emitted["a"])
		}
	})
	t.Run("an outcome past the drafts is clamped to them", func(t *testing.T) {
		s, _ := started(t, specConfig(t, 1), 50, 1000, specScript(200))
		if c := mustComplete(t, s, mustLaunch(t, s), map[string]int{"a": 99}); c.Emitted["a"] != specDraft {
			t.Fatalf("emitted %d tokens, want the %d the pass can verify", c.Emitted["a"], specDraft)
		}
	})
}

// The next pass reserves twice the draft width past the committed KV, rounded up to a page, and
// never gives back what an earlier pass reserved (allocation_sizing.py get_alloc_reserve_per_decode,
// page_aligned_decode_alloc_lens; mem_cache/allocation.py alloc_for_spec_decode).
func TestDecodeReservesTwiceTheDraftWidthAheadOfCommitted(t *testing.T) {
	const reserve = 2 * specDraft
	for _, ps := range []int{1, 16} {
		t.Run(fmt.Sprintf("page%d", ps), func(t *testing.T) {
			s, r := started(t, specConfig(t, ps), 50, 1000, specScript(200))
			prev := r.C.KV.KVAllocatedLen
			for i, acc := range []int{0, 3, 1, 2, 0, 3} {
				committed := r.C.KV.KVCommittedLen
				l := mustLaunch(t, s)
				if r.C.KV.KVCommittedLen != committed {
					t.Fatalf("pass %d: preparing the pass moved committed %d -> %d", i, committed, r.C.KV.KVCommittedLen)
				}
				want := max(prev, ceilTo(committed+reserve, ps))
				if got := r.C.KV.KVAllocatedLen; got != want {
					t.Fatalf("pass %d: allocated %d, want max(%d, ceil(%d+%d)) = %d", i, got, prev, committed, reserve, want)
				}
				prev = r.C.KV.KVAllocatedLen
				mustComplete(t, s, l, map[string]int{"a": acc})
			}
		})
	}
}

// A reserve is never given back: a request that holds more than the committed context plus the
// reserve (an earlier pass reserved it) keeps it, and a pass asks the allocator for nothing.
func TestDecodeNeverShrinksWhatARequestHolds(t *testing.T) {
	s, r := started(t, specConfig(t, 1), 50, 1000, specScript(200))
	r.C.KV.KVAllocatedLen += 40
	held := r.C.KV.KVAllocatedLen
	_, usedBefore := s.Pool()
	mustLaunch(t, s)
	if r.C.KV.KVAllocatedLen != held {
		t.Fatalf("allocated %d -> %d: the reserve was given back", held, r.C.KV.KVAllocatedLen)
	}
	if _, used := s.Pool(); used != usedBefore {
		t.Fatalf("a pass that needs nothing took %d slots", used-usedBefore)
	}
}

// The pass the engine prices computes the committed context and verifies the whole draft width:
// the forward record names the position the verify starts at.
func TestVerifyPassStartsAtTheCommittedContext(t *testing.T) {
	s, r := started(t, specConfig(t, 1), 50, 1000, specScript(200))
	for i, acc := range []int{2, 0, 3} {
		committed := r.C.KV.KVCommittedLen
		l := mustLaunch(t, s)
		if got := l.Forward.SeqLens[0] - 1; got != committed {
			t.Fatalf("pass %d: computed context %d, want the committed %d", i, got, committed)
		}
		mustComplete(t, s, l, map[string]int{"a": acc})
	}
}

// A request ends inside the accepted window: the tokens past its end are never output, the request
// finishes for the reason SGLang gives, and the KV past the end is freed (INV-1).
func TestFinishInsideTheAcceptedWindow(t *testing.T) {
	cases := []struct {
		name       string
		scriptLen  int
		maxNew     int
		wantReason string
		wantOutput int
	}{
		{"stop token on the second accepted token", 4, 1000, "matched_token", 4},
		{"stop token on the bonus", 2, 1000, "matched_token", 2},
		{"length on the third accepted token", 200, 4, "length", 4},
		{"length exactly at the window's end", 200, 5, "length", 5},
	}
	for _, ps := range []int{1, 16} {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("page%d/%s", ps, tc.name), func(t *testing.T) {
				sc := specScript(tc.scriptLen)
				if tc.wantReason == "matched_token" {
					sc[len(sc)-1] = specEOS
				}
				cfg := specConfig(t, ps)
				s, _ := started(t, cfg, 50, tc.maxNew, sc)
				c := mustComplete(t, s, mustLaunch(t, s), map[string]int{"a": specDraft - 1})
				if len(c.Finished) != 1 || c.Finished[0].Reason != tc.wantReason || c.Finished[0].Output != tc.wantOutput {
					t.Fatalf("finished %+v, want %s with %d tokens", c.Finished, tc.wantReason, tc.wantOutput)
				}
				if got := 1 + c.Emitted["a"]; got != tc.wantOutput {
					t.Fatalf("emitted %d tokens after the first, output %d", c.Emitted["a"], tc.wantOutput)
				}
				if total, used := s.Pool(); used != 0 {
					t.Fatalf("%d of %d KV slots still held by a finished request", used, total)
				}
			})
		}
	}
}

// INV-1 under speculation: every request ends with exactly its script, whatever the acceptance, and
// the pool is whole when the last one is done.
func TestSpecRunConservesRequestsAndKV(t *testing.T) {
	for _, overlap := range []bool{false, true} {
		for _, ps := range []int{1, 16} {
			t.Run(fmt.Sprintf("overlap=%v/page%d", overlap, ps), func(t *testing.T) {
				cfg := specConfig(t, ps)
				cfg.EnableOverlap = overlap
				s, err := build(cfg, 13)
				if err != nil {
					t.Fatal(err)
				}
				lens := map[string]int{"r0": 7, "r1": 31, "r2": 12, "r3": 64}
				for i, id := range []string{"r0", "r1", "r2", "r3"} {
					sc := specScript(lens[id])
					sc[len(sc)-1] = specEOS
					s.handleGenerateRequest(Arrival{RID: id, Prompt: specPrompt(20 + 9*i), MaxNewTokens: 1000, Script: sc})
				}
				done := map[string]int{}
				var pending []*Launched
				for step := 0; step < 500 && len(done) < len(lens); step++ {
					if l, err := s.Launch(); err != nil {
						t.Fatal(err)
					} else if l != nil {
						pending = append(pending, l)
					}
					limit := 1
					if overlap {
						limit = 2
					}
					for len(pending) >= limit || (len(pending) > 0 && step > 400) {
						l := pending[0]
						pending = pending[1:]
						acc := map[string]int{}
						for i, id := range l.Forward.RIDs {
							acc[id] = (step + i) % specDraft
						}
						c := mustComplete(t, s, l, acc)
						for _, f := range c.Finished {
							done[f.ID] = f.Output
						}
					}
				}
				for id, n := range lens {
					if done[id] != n {
						t.Fatalf("%s ended with %d tokens, want %d (finished: %v)", id, done[id], n, done)
					}
				}
				if total, used := s.Pool(); used != 0 {
					t.Fatalf("%d of %d KV slots still held after every request finished", used, total)
				}
			})
		}
	}
}

// A reserve of two draft widths per request is what the allocator is asked for, so a pool that
// holds the decode step of a request without speculation can fail it with speculation: the
// scheduler retracts to make room, as it does for any decode step that does not fit.
func TestSpecReserveCanRetractWhereDecodeWouldFit(t *testing.T) {
	retractions := func(draft int) int {
		cfg := specConfig(t, 1)
		cfg.MaxTotalNumTokens = 2 * (40 + 6)
		if draft == 0 {
			cfg.SpecNumSteps, cfg.SpecNumDraftTokens = 0, 0
		}
		s, err := build(cfg, 13)
		if err != nil {
			t.Fatal(err)
		}
		for i, id := range []string{"x", "y"} {
			// Distinct prompts, so the radix tree shares none of the KV.
			p := specPrompt(40)
			p[0] = int64(9000 + i)
			s.handleGenerateRequest(Arrival{RID: id, Prompt: p, MaxNewTokens: 6, Script: specScript(6)})
		}
		n := 0
		for i := 0; i < 6; i++ {
			l, err := s.Launch()
			if err != nil {
				t.Fatal(err)
			}
			_, pre := s.TakeReports()
			n += len(pre)
			if l != nil {
				mustComplete(t, s, l, nil)
			}
		}
		return n
	}
	if got := retractions(0); got != 0 {
		t.Fatalf("without speculation the pool should hold both requests, retracted %d", got)
	}
	if got := retractions(specDraft); got == 0 {
		t.Fatal("with speculation the reserve should not fit both requests, and nothing was retracted")
	}
}

// What the port does not model is refused by name (SGL-6), citing #68.
func TestUnmodelledSpeculationIsRefusedByName(t *testing.T) {
	cases := []struct {
		name string
		edit func(*Config)
	}{
		{"mixed chunked prefill", func(c *Config) { c.ChunkedPrefillSize = new(int); *c.ChunkedPrefillSize = 64; c.IsMixedChunk = true }},
		{"tree verification (more draft tokens than a chain)", func(c *Config) { c.SpecNumDraftTokens = specDraft + 2 }},
		{"a draft chain that is not steps+1 wide", func(c *Config) { c.SpecNumDraftTokens = 1 }},
		{"draft tokens without steps", func(c *Config) { c.SpecNumSteps = 0 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := specConfig(t, 1)
			tc.edit(&cfg)
			_, err := build(cfg, 13)
			if err == nil || !strings.Contains(err.Error(), "inference-sim/inference-sim#2355") {
				t.Fatalf("got %v, want a refusal citing inference-sim/inference-sim#2355", err)
			}
		})
	}
}

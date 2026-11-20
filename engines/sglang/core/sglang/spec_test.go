// Original to blis-sglang.

package sglang

import (
	"fmt"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"

	"github.com/inference-sim/inference-sim/engines/sglang/core/sched"
)

// Speculative decoding at top-k 1 as the contract sees it (S30, inference-sim/blis-sglang#68).

func specEngine(t *testing.T, steps int, mutate func(*sched.Config)) *Engine {
	t.Helper()
	tr, _ := load(t, "retract1")
	cfg := tr.Config
	cfg.ChunkedPrefillSize = nil
	cfg.MaxTotalNumTokens = 8192
	cfg.SpecNumSteps, cfg.SpecNumDraftTokens = steps, steps+1
	if mutate != nil {
		mutate(&cfg)
	}
	return newEngine(t, cfg)
}

func prefilled(t *testing.T, e *Engine, promptLen int, output []int32) {
	t.Helper()
	if err := e.Add(engine.Request{ID: "a", Prompt: outputOf(promptLen, 7), Output: output, Tags: map[string]string{"ignore_eos": "true"}}, 0); err != nil {
		t.Fatal(err)
	}
	b, err := e.Schedule(0)
	if err != nil || b.Empty() {
		t.Fatalf("prefill: %+v %v", b, err)
	}
	if _, err := e.Complete(b.ID, engine.Outcome{}, 1); err != nil {
		t.Fatal(err)
	}
}

// A decode pass under speculation is a verify pass: it schedules the pending token and the drafts
// after it over the committed context, and a prefill chunk carries no drafts.
func TestAVerifyPassSchedulesTheDraftWidthOverTheCommittedContext(t *testing.T) {
	for _, steps := range []int{1, 3, 5} {
		t.Run(fmt.Sprintf("steps%d", steps), func(t *testing.T) {
			e := specEngine(t, steps, nil)
			if err := e.Add(engine.Request{ID: "a", Prompt: outputOf(50, 7), Output: outputOf(30, 9)}, 0); err != nil {
				t.Fatal(err)
			}
			b, _ := e.Schedule(0)
			if r := b.Reqs[0]; r.Tokens != 50 || r.Drafts != 0 {
				t.Fatalf("prefill chunk %+v carries drafts or is not the prompt", r)
			}
			if _, err := e.Complete(b.ID, engine.Outcome{}, 1); err != nil {
				t.Fatal(err)
			}
			committed := 50 // the prompt; the first output token is the bonus the verify feeds
			for i, acc := range []int{0, steps, 1} {
				b, err := e.Schedule(engine.Time(2 + 2*i))
				if err != nil || b.Empty() {
					t.Fatalf("pass %d: %+v %v", i, b, err)
				}
				r := b.Reqs[0]
				if r.Tokens != steps+1 || r.Drafts != steps || r.Computed != committed {
					t.Fatalf("pass %d: %+v, want %d tokens, %d drafts over %d computed", i, r, steps+1, steps, committed)
				}
				res, err := e.Complete(b.ID, engine.Outcome{Accepted: map[string]int{"a": acc}}, engine.Time(3+2*i))
				if err != nil {
					t.Fatal(err)
				}
				if res.Emitted["a"] != acc+1 {
					t.Fatalf("pass %d emitted %d, want %d", i, res.Emitted["a"], acc+1)
				}
				committed += acc + 1
			}
		})
	}
}

// Without speculation a decode pass is one token, no drafts: the contract is unchanged.
func TestWithoutSpeculationADecodeIsOneToken(t *testing.T) {
	e := specEngine(t, 0, func(c *sched.Config) { c.SpecNumSteps, c.SpecNumDraftTokens = 0, 0 })
	prefilled(t, e, 50, outputOf(10, 9))
	b, _ := e.Schedule(2)
	if r := b.Reqs[0]; r.Tokens != 1 || r.Drafts != 0 {
		t.Fatalf("decode pass %+v, want one token and no drafts", r)
	}
}

// The EAGLE family counts the draft slots of the output in the input length the front of the server
// checks (compute_num_reserved_tokens, managers/utils.py:484-500@297e54fd): an input that fits a
// context without speculation can fail it with.
func TestDraftSlotsCountInTheInputLengthCheck(t *testing.T) {
	const context, steps = 100, 3
	cases := []struct {
		name   string
		steps  int
		prompt int
		reject bool
	}{
		{"fits with the reserved slots", steps, context - steps - 2, false},
		{"the reserved slots fill the context", steps, context - steps - 1, true},
		{"the same input without speculation", 0, context - steps - 1, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := specEngine(t, tc.steps, func(c *sched.Config) {
				c.ContextLen, c.MaxContextLen = context, context+4+tc.steps+1
				if tc.steps == 0 {
					c.SpecNumSteps, c.SpecNumDraftTokens = 0, 0
				}
			})
			err := e.Add(engine.Request{ID: "a", Prompt: outputOf(tc.prompt, 7), Output: outputOf(2, 9)}, 0)
			if tc.reject != (err != nil && strings.Contains(err.Error(), "context length")) {
				t.Fatalf("prompt %d: got %v, rejected expected=%v", tc.prompt, err, tc.reject)
			}
		})
	}
}

// The request pool's rows are longer by the draft tokens (get_req_to_token_extra_context_len,
// 4 + num_draft_tokens), so a request can hold its reserve at the end of the context.
func TestSchedConfigWidensTheRequestRowsByTheDraftTokens(t *testing.T) {
	rt := Runtime{MaxTotalNumTokens: 4096, ReqPoolSize: 9, MaxRunningRequests: 9, ContextLen: 8192, EOSTokenIDs: []int64{1}, VocabSize: 10}
	for _, tc := range []struct{ page, draft, extra int }{
		{1, 0, 4}, {1, 2, 4 + 2}, {1, 4, 4 + 4},
		// at a page size above one the double-buffered reserve plus the page's rounding can be wider
		{16, 0, 4}, {16, 4, max(4+4, 2*4+15)}, {128, 8, max(4+8, 2*8+127)},
	} {
		a := SchedArgs{PageSize: tc.page, SchedulePolicy: "fcfs", MaxPrefillTokens: 16384, ScheduleConservativeness: 1.0,
			DisaggregationMode: "null", SpeculativeNumSteps: max(tc.draft-1, 0), SpeculativeNumDraftTokens: tc.draft}
		c, err := SchedConfig(a, rt, DefaultEnv())
		if err != nil {
			t.Fatal(err)
		}
		if want := rt.ContextLen + tc.extra; c.MaxContextLen != want {
			t.Errorf("page %d draft %d: row length %d, want %d", tc.page, tc.draft, c.MaxContextLen, want)
		}
		if c.SpecNumSteps != a.SpeculativeNumSteps || c.SpecNumDraftTokens != tc.draft {
			t.Errorf("draft %d: the scheduler is configured with steps %d, draft %d", tc.draft, c.SpecNumSteps, c.SpecNumDraftTokens)
		}
	}
}

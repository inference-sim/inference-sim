// Original to blis-sglang.

package sglang

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"

	blissglang "github.com/inference-sim/inference-sim/engines/sglang"
	"github.com/inference-sim/inference-sim/engines/sglang/core/sched"
	"github.com/inference-sim/inference-sim/engines/sglang/internal/testgz"
)

const (
	goldenDir   = "../sched/testdata/golden-sched"
	scenarioDir = "../../oracle/scenarios-sched"
)

type flat struct{}

func (flat) StepTime(b engine.Batch) engine.Duration { return 1000 + engine.Duration(b.Tokens()) }

type tracedStep struct {
	Forwards []sched.Forward `json:"forwards"`
}

type goldenTrace struct {
	Config sched.Config `json:"config"`
	Steps  []tracedStep `json:"steps"`
}

type scenarioFile struct {
	SampledToken int64 `json:"sampled_token"`
	Requests     []struct {
		RID          string  `json:"rid"`
		ArrivalStep  int     `json:"arrival_step"`
		Prompt       []int32 `json:"prompt"`
		MaxNewTokens int     `json:"max_new_tokens"`
		IgnoreEOS    *bool   `json:"ignore_eos"`
		Priority     *int    `json:"priority"`
		RoutingKey   string  `json:"routing_key"`
	} `json:"requests"`
}

func load(t *testing.T, name string) (goldenTrace, scenarioFile) {
	t.Helper()
	var tr goldenTrace
	var sc scenarioFile
	for path, v := range map[string]any{filepath.Join(goldenDir, name+".trace.json"): &tr, filepath.Join(scenarioDir, name+".json"): &sc} {
		b, err := testgz.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(b, v); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
	}
	return tr, sc
}

func emptyList() fields.FieldList {
	l, err := blissglang.Fields()
	if err != nil {
		panic(err)
	}
	return l
}

func newEngine(t *testing.T, cfg sched.Config) *Engine {
	t.Helper()
	e, err := New(cfg, flat{}, emptyList())
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func outputOf(n int, tok int32) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = tok
	}
	return out
}

// The engine, driven the way a simulator drives it, runs the passes SGLang's own Scheduler ran:
// the same requests in the same order with the same token counts, step by step, in the normal loop
// (each pass completed before the next is scheduled) and in the overlap loop (the next pass is
// scheduled before the last is completed). The goldens are core/sched's (recorded by CI).
func TestEnginePassesEqualSGLangs(t *testing.T) {
	files, _ := testgz.Glob(filepath.Join(goldenDir, "*.trace.json"))
	ran := 0
	for _, f := range files {
		name := strings.TrimSuffix(filepath.Base(f), ".trace.json")
		tr, sc := load(t, name)
		skip := tr.Config.Policy == "random" || tr.Config.EnablePriorityScheduling || tr.Config.MaxQueuedRequests != nil ||
			tr.Config.PrefillMaxRequests != nil || tr.Config.DisableConsecutivePrefillOverlap || sc.SampledToken != 13
		for _, r := range sc.Requests {
			skip = skip || r.RoutingKey != "" || r.Priority != nil
		}
		if skip {
			continue
		}
		ran++
		t.Run(name, func(t *testing.T) {
			e := newEngine(t, tr.Config)
			var inflight []engine.BatchID
			for i, step := range tr.Steps {
				for _, r := range sc.Requests {
					if r.ArrivalStep != i {
						continue
					}
					req := engine.Request{ID: r.RID, Prompt: r.Prompt, Output: outputOf(r.MaxNewTokens, 13), MaxOutputTokens: r.MaxNewTokens}
					if r.IgnoreEOS == nil || *r.IgnoreEOS {
						req.Tags = map[string]string{"ignore_eos": "true"}
					}
					err := e.Add(req, engine.Time(i))
					if err != nil {
						t.Fatalf("step %d: %v", i, err)
					}
				}
				b, err := e.Schedule(engine.Time(i))
				if err != nil {
					t.Fatalf("step %d: %v", i, err)
				}
				got := map[string]int{}
				var order []string
				for _, s := range b.Reqs {
					got[s.ID], order = s.Tokens, append(order, s.ID)
				}
				want := map[string]int{}
				var wantOrder []string
				for _, f := range step.Forwards {
					for j, id := range f.RIDs {
						n := 1
						if f.Mode != sched.Decode {
							n = f.ExtendSeqLens[j]
						}
						want[id], wantOrder = n, append(wantOrder, id)
					}
				}
				if !reflect.DeepEqual(order, wantOrder) || !reflect.DeepEqual(got, want) {
					t.Fatalf("step %d: SGLang ran %v %v, the engine %v %v", i, wantOrder, want, order, got)
				}
				if !b.Empty() {
					inflight = append(inflight, b.ID)
				}
				// The overlap loop completes the last pass after launching this one; the normal loop
				// completes this one.
				keep := 0
				if tr.Config.EnableOverlap && !b.Empty() {
					keep = 1
				}
				for len(inflight) > keep {
					if _, err := e.Complete(inflight[0], engine.Outcome{}, engine.Time(i)); err != nil {
						t.Fatalf("step %d: %v", i, err)
					}
					inflight = inflight[1:]
				}
			}
		})
	}
	if ran < 10 {
		t.Fatalf("only %d scenarios ran", ran)
	}
}

// The substrate's laws for an engine: protocol errors, ordering, conservation, determinism.
func TestEngineKeepsTheContractsLaws(t *testing.T) {
	for _, overlap := range []bool{false, true} {
		t.Run(fmt.Sprintf("overlap=%v", overlap), func(t *testing.T) {
			tr, _ := load(t, "retract1")
			cfg := tr.Config
			cfg.EnableOverlap = overlap
			var arr []enginetest.Arrival
			for i := 0; i < 12; i++ {
				p := make([]int32, 20+3*i)
				for j := range p {
					p[j] = int32(1000 + (i%3)*100 + j)
				}
				arr = append(arr, enginetest.Arrival{At: engine.Time(i * 500), Req: engine.Request{ID: fmt.Sprintf("r%d", i), Prompt: p, Output: outputOf(6+i, 9)}})
			}
			enginetest.TestEngine(t, func() (engine.Engine, error) { return New(cfg, flat{}, emptyList()) }, arr, enginetest.Options{})
		})
	}
}

func TestWhatThePortDoesNotModelIsRefusedByName(t *testing.T) {
	tr, _ := load(t, "retract1")
	e := newEngine(t, tr.Config)
	base := engine.Request{ID: "x", Prompt: []int32{1, 2, 3}, Output: []int32{4, 5}}
	cases := map[string]func(*engine.Request){
		"LoRA":        func(r *engine.Request) { r.Adapter, r.AdapterPath = "a", "/p" },
		"disagg":      func(r *engine.Request) { r.RemoteDecode = true },
		"no output":   func(r *engine.Request) { r.Output = nil },
		"bad adapter": func(r *engine.Request) { r.AdapterPath = "/p" },
	}
	var names []string
	for n := range cases {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		r := base
		cases[n](&r)
		if err := e.Add(r, 0); err == nil {
			t.Errorf("%s: the engine took the request", n)
		}
	}
	if e.Snapshot().Waiting != 0 {
		t.Fatalf("a refused request left %d waiting", e.Snapshot().Waiting)
	}
	cfg := tr.Config
	cfg.DisableConsecutivePrefillOverlap = true
	if _, err := New(cfg, flat{}, emptyList()); err == nil || !strings.Contains(err.Error(), "blis-sglang#4") {
		t.Fatalf("got %v, want the consecutive-prefill option refused naming blis-sglang#4", err)
	}
	if _, err := New(tr.Config, nil, emptyList()); err == nil {
		t.Fatal("an engine without a pricer was built")
	}
}

// A request ends by its assigned output ("stop") or by the client's budget ("length"), and the
// pool is whole afterwards.
func TestFinishReasonsAndAWholePool(t *testing.T) {
	tr, _ := load(t, "retract1")
	e := newEngine(t, tr.Config)
	before := e.Snapshot().KVBlocksUsed
	if err := e.Add(engine.Request{ID: "stop", Prompt: []int32{1, 2, 3, 4}, Output: []int32{9, 9, 9}}, 0); err != nil {
		t.Fatal(err)
	}
	if err := e.Add(engine.Request{ID: "cap", Prompt: []int32{5, 6, 7, 8}, Output: []int32{9, 9, 9, 9, 9}, MaxOutputTokens: 2}, 0); err != nil {
		t.Fatal(err)
	}
	reasons := map[string]engine.FinishReason{}
	for i := 0; i < 20 && len(reasons) < 2; i++ {
		b, err := e.Schedule(engine.Time(i))
		if err != nil {
			t.Fatal(err)
		}
		if b.Empty() {
			continue
		}
		res, err := e.Complete(b.ID, engine.Outcome{}, engine.Time(i))
		if err != nil {
			t.Fatal(err)
		}
		for _, f := range res.Finished {
			reasons[f.ID] = f.Reason
		}
	}
	if reasons["stop"] != engine.FinishStop || reasons["cap"] != engine.FinishLength {
		t.Fatalf("finish reasons %v, want stop for the assigned output and length for the budget", reasons)
	}
	if got := e.Snapshot().KVBlocksUsed; got > before {
		t.Fatalf("%d blocks in use after both finished, had %d (the radix cache keeps finished prefixes as evictable, not used)", got, before)
	}
}

// The server's front rejects an input at or past the context length, and an input with a budget
// past it, as an error to the client; --allow-auto-truncate cuts them instead (tokenizer_manager.py:
// 1331-1372@297e54fd).
func TestTheContextLengthRejectsOrTruncatesAsTheServerDoes(t *testing.T) {
	tr, _ := load(t, "retract1")
	long := make([]int32, 60)
	for i := range long {
		long[i] = int32(100 + i)
	}
	run := func(auto bool, r engine.Request) (*Engine, error) {
		cfg := tr.Config
		cfg.ContextLen, cfg.AllowAutoTruncate = 50, auto
		e := newEngine(t, cfg)
		return e, e.Add(r, 0)
	}
	cases := []struct {
		name string
		req  engine.Request
	}{
		{"input past the context", engine.Request{ID: "a", Prompt: long, Output: []int32{9, 9}}},
		{"input and budget past the context", engine.Request{ID: "b", Prompt: long[:40], Output: outputOf(20, 9), MaxOutputTokens: 20}},
	}
	for _, c := range cases {
		if _, err := run(false, c.req); err == nil || !strings.Contains(err.Error(), "context length") {
			t.Errorf("%s: got %v, want a context-length error", c.name, err)
		}
	}
	// Truncated: the input is cut to the context, and the budget to what is left of it.
	e, err := run(true, engine.Request{ID: "b", Prompt: long[:40], Output: outputOf(20, 9), MaxOutputTokens: 20})
	if err != nil {
		t.Fatal(err)
	}
	emitted := 0
	for i := 0; i < 40; i++ {
		b, err := e.Schedule(engine.Time(i))
		if err != nil {
			t.Fatal(err)
		}
		if b.Empty() {
			continue
		}
		res, err := e.Complete(b.ID, engine.Outcome{}, engine.Time(i))
		if err != nil {
			t.Fatal(err)
		}
		emitted += res.Emitted["b"]
	}
	if emitted != 10 {
		t.Fatalf("the truncated request produced %d tokens, want 10 (context 50 less its 40 input tokens)", emitted)
	}
}

// A request the simulator ends (a client abort, or a deadline) while a pass that holds it is in
// flight leaves the engine at that instant: completing the pass returns it no token, no first
// token and no finish, and the pool it held is free again. The orderings around the in-flight one
// (ended before it is scheduled, ended after its pass completed) behave as they always did.
func TestFinishWhileAPassIsInFlightRetiresTheRequestWithoutOutput(t *testing.T) {
	tr, _ := load(t, "retract1")
	for _, reason := range []engine.FinishReason{engine.FinishAbort, engine.FinishTimeout} {
		for _, overlap := range []bool{false, true} {
			for _, afterFirst := range []bool{false, true} {
				name := fmt.Sprintf("%v/overlap=%v/afterFirstPass=%v", reason, overlap, afterFirst)
				t.Run(name, func(t *testing.T) {
					cfg := tr.Config
					cfg.EnableOverlap = overlap
					e := newEngine(t, cfg)
					idle := e.Snapshot().KVBlocksUsed
					for _, id := range []string{"x", "y"} {
						if err := e.Add(engine.Request{ID: id, Prompt: []int32{1, 2, 3, 4}, Output: outputOf(6, 9)}, 0); err != nil {
							t.Fatal(err)
						}
					}
					now := engine.Time(1)
					if afterFirst {
						b, err := e.Schedule(now)
						if err != nil {
							t.Fatal(err)
						}
						if _, err := e.Complete(b.ID, engine.Outcome{}, now+1); err != nil {
							t.Fatal(err)
						}
						now += 2
					}
					b, err := e.Schedule(now)
					if err != nil || b.Empty() {
						t.Fatalf("no pass in flight: %v %+v", err, b)
					}
					if err := e.Finish("x", reason, now+1); err != nil {
						t.Fatal(err)
					}
					res, err := e.Complete(b.ID, engine.Outcome{}, now+2)
					if err != nil {
						t.Fatal(err)
					}
					if n, ok := res.Emitted["x"]; ok {
						t.Fatalf("an ended request emitted %d tokens", n)
					}
					for _, id := range res.FirstTokens {
						if id == "x" {
							t.Fatal("an ended request got a first token")
						}
					}
					for _, f := range res.Finished {
						if f.ID == "x" {
							t.Fatalf("an ended request reported finished again: %+v", f)
						}
					}
					// The other request goes on to its end; the pool returns to idle.
					yDone := false
					for i := 0; i < 40 && !yDone; i++ {
						now += 3
						nb, err := e.Schedule(now)
						if err != nil {
							t.Fatal(err)
						}
						if nb.Empty() {
							continue
						}
						r, err := e.Complete(nb.ID, engine.Outcome{}, now+1)
						if err != nil {
							t.Fatal(err)
						}
						if _, ok := r.Emitted["x"]; ok {
							t.Fatal("an ended request emitted in a later pass")
						}
						for _, f := range r.Finished {
							yDone = yDone || f.ID == "y"
						}
					}
					if !yDone {
						t.Fatal("the other request never finished")
					}
					if got := e.Snapshot().KVBlocksUsed; got > idle {
						t.Fatalf("%d blocks in use at the end, %d at the start", got, idle)
					}
				})
			}
		}
	}
}

// The same holds for a decode pass in flight: the request already ran its prompt and sits in the
// running batch when it is ended.
func TestFinishWhileADecodePassIsInFlightRetiresTheRequest(t *testing.T) {
	tr, _ := load(t, "retract1")
	e := newEngine(t, tr.Config)
	if err := e.Add(engine.Request{ID: "x", Prompt: []int32{1, 2, 3, 4}, Output: outputOf(6, 9)}, 0); err != nil {
		t.Fatal(err)
	}
	for i, now := 0, engine.Time(1); i < 2; i, now = i+1, now+2 {
		b, err := e.Schedule(now)
		if err != nil || b.Empty() {
			t.Fatalf("pass %d: %v %+v", i, err, b)
		}
		if i == 0 {
			if _, err := e.Complete(b.ID, engine.Outcome{}, now+1); err != nil {
				t.Fatal(err)
			}
			continue
		}
		if err := e.Finish("x", engine.FinishAbort, now+1); err != nil {
			t.Fatal(err)
		}
		res, err := e.Complete(b.ID, engine.Outcome{}, now+2)
		if err != nil || len(res.Emitted) != 0 || len(res.Finished) != 0 {
			t.Fatalf("the ended request showed output: %v %+v", err, res)
		}
	}
	if got := e.Snapshot().KVBlocksUsed; got != 0 {
		t.Fatalf("%d blocks in use after the only request ended", got)
	}
}

// Ended before its first pass is scheduled, or after its last pass completed, a request leaves
// quietly: it is never scheduled, or already reported finished.
func TestFinishOutsideAPassIsQuiet(t *testing.T) {
	tr, _ := load(t, "retract1")
	e := newEngine(t, tr.Config)
	if err := e.Add(engine.Request{ID: "x", Prompt: []int32{1, 2, 3, 4}, Output: outputOf(2, 9)}, 0); err != nil {
		t.Fatal(err)
	}
	if err := e.Finish("x", engine.FinishAbort, 1); err != nil {
		t.Fatal(err)
	}
	if b, err := e.Schedule(2); err != nil || !b.Empty() {
		t.Fatalf("an ended waiting request was scheduled: %v %+v", err, b)
	}
	if err := e.Add(engine.Request{ID: "z", Prompt: []int32{1, 2, 3, 4}, Output: outputOf(1, 9)}, 3); err != nil {
		t.Fatal(err)
	}
	b, err := e.Schedule(4)
	if err != nil {
		t.Fatal(err)
	}
	res, err := e.Complete(b.ID, engine.Outcome{}, 5)
	if err != nil || len(res.Finished) != 1 || res.Finished[0].ID != "z" {
		t.Fatalf("z did not finish in its pass: %v %+v", err, res)
	}
	if err := e.Finish("z", engine.FinishAbort, 6); err != nil {
		t.Fatal(err)
	}
	if s := e.Snapshot(); s.Running+s.Waiting+s.InFlight != 0 {
		t.Fatalf("engine not idle: %+v", s)
	}
}

// A request ended while its prompt is cut into chunks (the server's chunked prefill) gets no next
// chunk, whether the end lands while a chunk is in flight or between two chunks.
func TestFinishDuringChunkedPrefillStopsTheChunks(t *testing.T) {
	tr, _ := load(t, "retract1")
	prompt := make([]int32, 100)
	for i := range prompt {
		prompt[i] = int32(10 + i)
	}
	for _, completeFirst := range []bool{false, true} {
		t.Run(fmt.Sprintf("completeFirstChunk=%v", completeFirst), func(t *testing.T) {
			e := newEngine(t, tr.Config)
			if err := e.Add(engine.Request{ID: "x", Prompt: prompt, Output: outputOf(2, 9)}, 0); err != nil {
				t.Fatal(err)
			}
			b, err := e.Schedule(1)
			if err != nil || len(b.Reqs) != 1 || b.Reqs[0].Tokens >= len(prompt) {
				t.Fatalf("first pass is not a chunk: %v %+v", err, b)
			}
			if completeFirst {
				if _, err := e.Complete(b.ID, engine.Outcome{}, 2); err != nil {
					t.Fatal(err)
				}
				if b, err = e.Schedule(3); err != nil || b.Empty() {
					t.Fatalf("no second chunk: %v %+v", err, b)
				}
			}
			if err := e.Finish("x", engine.FinishTimeout, 4); err != nil {
				t.Fatal(err)
			}
			res, err := e.Complete(b.ID, engine.Outcome{}, 5)
			if err != nil || len(res.Emitted) != 0 || len(res.FirstTokens) != 0 || len(res.Finished) != 0 {
				t.Fatalf("the ended request showed output: %v %+v", err, res)
			}
			for i := 0; i < 3; i++ {
				nb, err := e.Schedule(engine.Time(6 + i))
				if err != nil || !nb.Empty() {
					t.Fatalf("a chunk of an ended request was scheduled: %v %+v", err, nb)
				}
			}
			if got := e.Snapshot().KVBlocksUsed; got != 0 {
				t.Fatalf("%d blocks in use after the only request ended", got)
			}
		})
	}
}

// Two requests share one prefill pass; the simulator ends one of them while the pass is in
// flight. The other is untouched: it emits its first token and goes on to finish.
func TestFinishOfOneRequestOfAPassLeavesTheOtherAlone(t *testing.T) {
	tr, _ := load(t, "retract1")
	for _, reason := range []engine.FinishReason{engine.FinishAbort, engine.FinishTimeout} {
		t.Run(fmt.Sprint(reason), func(t *testing.T) {
			e := newEngine(t, tr.Config)
			for _, rq := range []struct {
				id string
				p  []int32
			}{{"x", []int32{1, 2, 3, 4}}, {"y", []int32{7, 8, 9, 10}}} {
				// A budget keeps the scheduler's estimate of what each may grow to small enough for both.
				if err := e.Add(engine.Request{ID: rq.id, Prompt: rq.p, Output: outputOf(3, 9), MaxOutputTokens: 3}, 0); err != nil {
					t.Fatal(err)
				}
			}
			b, err := e.Schedule(1)
			if err != nil || len(b.Reqs) != 2 {
				t.Fatalf("the two requests are not in one pass: %v %+v", err, b)
			}
			if err := e.Finish("x", reason, 2); err != nil {
				t.Fatal(err)
			}
			res, err := e.Complete(b.ID, engine.Outcome{}, 3)
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := res.Emitted["x"]; ok || res.Emitted["y"] != 1 || !reflect.DeepEqual(res.FirstTokens, []string{"y"}) {
				t.Fatalf("emitted %v, first tokens %v: want only y's first token", res.Emitted, res.FirstTokens)
			}
			var fin []string
			for i := 0; i < 10 && len(fin) == 0; i++ {
				nb, err := e.Schedule(engine.Time(4 + 2*i))
				if err != nil {
					t.Fatal(err)
				}
				if nb.Empty() {
					continue
				}
				r, err := e.Complete(nb.ID, engine.Outcome{}, engine.Time(5+2*i))
				if err != nil {
					t.Fatal(err)
				}
				for _, f := range r.Finished {
					fin = append(fin, f.ID)
					if f.OutputTokens != 3 {
						t.Fatalf("y finished with %d tokens, want 3", f.OutputTokens)
					}
				}
			}
			if !reflect.DeepEqual(fin, []string{"y"}) {
				t.Fatalf("finished %v, want only y", fin)
			}
		})
	}
}

// A request ended mid-pass is freed at once and publishes nothing to the prefix cache: a request
// with the same prompt that follows reads no prefix. (SGLang marks the request to finish and frees
// it with a checkpoint after one more pass, scheduler.py:5775-5792@297e54fd; the engine contract
// ends the request at Finish, so the engine frees it then, without one.)
func TestAnEndedRequestLeavesNoPrefixBehind(t *testing.T) {
	tr, _ := load(t, "retract1")
	e := newEngine(t, tr.Config)
	p := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	if err := e.Add(engine.Request{ID: "x", Prompt: p, Output: outputOf(3, 9)}, 0); err != nil {
		t.Fatal(err)
	}
	b, err := e.Schedule(1)
	if err != nil || b.Empty() {
		t.Fatal(err)
	}
	if err := e.Finish("x", engine.FinishAbort, 2); err != nil {
		t.Fatal(err)
	}
	if _, err := e.Complete(b.ID, engine.Outcome{}, 3); err != nil {
		t.Fatal(err)
	}
	if err := e.Add(engine.Request{ID: "z", Prompt: p, Output: outputOf(3, 9)}, 4); err != nil {
		t.Fatal(err)
	}
	b, err = e.Schedule(5)
	if err != nil || len(b.Reqs) != 1 || b.Reqs[0].ID != "z" {
		t.Fatalf("z is not scheduled: %v %+v", err, b)
	}
	if got := b.Reqs[0].Computed; got != 0 {
		t.Fatalf("z found %d tokens of an ended request's prompt cached, want 0", got)
	}
}

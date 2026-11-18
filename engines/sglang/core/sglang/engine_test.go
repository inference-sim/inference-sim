// Original to blis-sglang.

package sglang

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/spec/fields"

	blissglang "github.com/inference-sim/blis-sglang"
	"github.com/inference-sim/blis-sglang/core/sched"
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
		b, err := os.ReadFile(path)
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
	files, _ := filepath.Glob(filepath.Join(goldenDir, "*.trace.json"))
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
	if _, err := New(cfg, flat{}, emptyList()); err == nil || !strings.Contains(err.Error(), "#59") {
		t.Fatalf("got %v, want the consecutive-prefill option refused naming #59", err)
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

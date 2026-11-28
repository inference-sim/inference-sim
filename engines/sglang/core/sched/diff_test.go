// Original to blis-sglang.

package sched

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
	"github.com/inference-sim/inference-sim/engines/sglang/internal/testgz"
)

// The scheduler replay (T2, FID-4): the scenario's requests arrive at the steps it names, and
// every step of the port equals the step of SGLang's own Scheduler that oracle/run_sched.py
// recorded: the forward pass launched (mode, requests in order, sequence, extend and prefix
// lengths, rows, free tokens), the requests that finished, and the state the loop left (waiting
// queue, running batch, chunked request, free and evictable tokens, free rows, new_token_ratio).

// GoldenDir holds the traces oracle/run_sched.py recorded, one per scenario of
// oracle/scenarios-sched.
const GoldenDir = "testdata/golden-sched"

// ScenarioDir holds the scenarios the traces were recorded from.
const ScenarioDir = "../../oracle/scenarios-sched"

type traceStep struct {
	Step     int       `json:"step"`
	Arrivals []string  `json:"arrivals"`
	Forwards []Forward `json:"forwards"`
	Outputs  []struct {
		Finished []string `json:"finished"`
	} `json:"outputs"`
	After *State `json:"after"`
}

type trace struct {
	Config Config      `json:"config"`
	Steps  []traceStep `json:"steps"`
}

type scenario struct {
	Page         int   `json:"page_size"`
	SampledToken int64 `json:"sampled_token"`
	Requests     []struct {
		RID          string  `json:"rid"`
		ArrivalStep  int     `json:"arrival_step"`
		Prompt       []int64 `json:"prompt"`
		MaxNewTokens int     `json:"max_new_tokens"`
		IgnoreEOS    *bool   `json:"ignore_eos"`
		Priority     *int    `json:"priority"`
		RoutingKey   string  `json:"routing_key"`
	} `json:"requests"`
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := testgz.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

func loadGoldens(t *testing.T) map[string]string {
	t.Helper()
	files, err := testgz.Glob(filepath.Join(GoldenDir, "*.trace.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Fatalf("no trace under %s: CI records them (oracle/run_sched.py)", GoldenDir)
	}
	out := map[string]string{}
	for _, f := range files {
		out[strings.TrimSuffix(filepath.Base(f), ".trace.json")] = f
	}
	return out
}

// build makes the scheduler a trace's configuration describes, over the cache framework's own
// allocator, request pool and radix cache.
func build(cfg Config, sampled int64) (*Scheduler, error) {
	var alloc cache.Allocator
	if cfg.PageSize == 1 {
		alloc = cache.NewTokenAllocator(cfg.MaxTotalNumTokens, nil, cfg.NeedSort)
	} else {
		alloc = cache.NewPagedAllocator(cfg.MaxTotalNumTokens, cfg.PageSize, nil, cfg.NeedSort)
	}
	pool := cache.NewRequestPool(cfg.ReqPoolSize, cfg.MaxContextLen)
	tree, err := cache.NewRadixCache(cache.RadixConfig{PageSize: cfg.PageSize, Disable: cfg.Disable, EvictionPolicy: cfg.radixEvictionPolicy()}, alloc, pool)
	if err != nil {
		return nil, err
	}
	return New(cfg, tree, alloc, pool, sampled)
}

// sglangOrder is the order SGLang's random policy gave the waiting queue in a step. The port draws
// its own shuffle from a named stream, which cannot equal Python's; to compare what follows from
// the order, the trace's is read back. SGLang admits a prefix of the shuffled queue and leaves the
// rest in place, so the shuffled queue is the requests the step's forward passes took from it, in
// batch order, then the requests waiting after the step; requests not in the queue (those already
// running or chunking) drop out.
func sglangOrder(step traceStep) func([]string) []string {
	return func(queue []string) []string {
		in := map[string]bool{}
		for _, r := range queue {
			in[r] = true
		}
		var order []string
		take := func(r string) {
			if in[r] {
				order = append(order, r)
				delete(in, r)
			}
		}
		for _, f := range step.Forwards {
			for _, r := range f.RIDs {
				take(r)
			}
		}
		if step.After != nil {
			for _, r := range step.After.Waiting {
				take(r)
			}
		}
		return order
	}
}

// replay runs a scenario on the port and returns each step's record. A non-nil tr supplies
// SGLang's shuffle for the random policy.
func replay(sc scenario, cfg Config, steps int, tr *trace) ([]StepRecord, error) {
	s, err := build(cfg, sc.SampledToken)
	if err != nil {
		return nil, err
	}
	return runScenario(s, sc, steps, tr)
}

// runScenario steps a scheduler through a scenario's arrivals.
func runScenario(s *Scheduler, sc scenario, steps int, tr *trace) ([]StepRecord, error) {
	var out []StepRecord
	for i := 0; i < steps; i++ {
		if tr != nil {
			s.Shuffle = sglangOrder(tr.Steps[i])
		}
		var arr []Arrival
		for _, r := range sc.Requests {
			if r.ArrivalStep == i {
				ignore := true
				if r.IgnoreEOS != nil {
					ignore = *r.IgnoreEOS
				}
				arr = append(arr, Arrival{RID: r.RID, Prompt: r.Prompt, MaxNewTokens: r.MaxNewTokens, IgnoreEOS: ignore,
					Priority: r.Priority, RoutingKey: r.RoutingKey})
			}
		}
		rec, err := s.Step(arr)
		if err != nil {
			return out, fmt.Errorf("step %d: %w", i, err)
		}
		out = append(out, rec)
	}
	return out, nil
}

// asJSON prints a record the way the trace holds it, so that what a pointer holds shows.
func asJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func finishedOf(st traceStep) []string {
	out := []string{}
	for _, o := range st.Outputs {
		out = append(out, o.Finished...)
	}
	return out
}

// compare returns the first way a step of the port differs from the trace's, or "".
func compare(want traceStep, got StepRecord) string {
	if !reflect.DeepEqual(want.Forwards == nil, got.Forwards == nil) && len(want.Forwards) != len(got.Forwards) {
		return fmt.Sprintf("forwards: SGLang launched %d, the port %d", len(want.Forwards), len(got.Forwards))
	}
	for i := range want.Forwards {
		w, g := want.Forwards[i], got.Forwards[i]
		w.Decoding = normalize(w.Decoding)
		g.Decoding = normalize(g.Decoding)
		if !reflect.DeepEqual(w, g) {
			return fmt.Sprintf("forward %d: SGLang %s, port %s", i, asJSON(w), asJSON(g))
		}
	}
	if wf, gf := finishedOf(want), append([]string{}, got.Finished...); !reflect.DeepEqual(wf, gf) {
		return fmt.Sprintf("finished: SGLang %v, port %v", wf, gf)
	}
	if want.After != nil && !reflect.DeepEqual(*want.After, got.After) {
		return fmt.Sprintf("state after the step:\nSGLang %s\nport   %s", asJSON(*want.After), asJSON(got.After))
	}
	return ""
}

func normalize(x []string) []string {
	if len(x) == 0 {
		return nil
	}
	return x
}

func TestStepsEqualSGLangs(t *testing.T) {
	for name, path := range loadGoldens(t) {
		t.Run(name, func(t *testing.T) {
			var tr trace
			readJSON(t, path, &tr)
			var sc scenario
			readJSON(t, filepath.Join(ScenarioDir, name+".json"), &sc)
			got, err := replay(sc, tr.Config, len(tr.Steps), &tr)
			if err != nil {
				t.Fatal(err)
			}
			forwards := 0
			for i, want := range tr.Steps {
				forwards += len(want.Forwards)
				if d := compare(want, got[i]); d != "" {
					t.Fatalf("step %d: %s", i, d)
				}
			}
			if forwards == 0 {
				t.Fatal("the trace launched no forward pass")
			}
		})
	}
}

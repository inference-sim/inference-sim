package sched

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// eosToken is tests/v1/core/utils.py:43@5df63a72 (EOS_TOKEN_ID), which the oracle's GPU mock
// samples as each request's last assigned token.
const eosToken = 50256

// scenario is the oracle's scenario file (oracle/README.md).
type scenario struct {
	Name   string `json:"name"`
	Config struct {
		MaxNumSeqs                int    `json:"max_num_seqs"`
		MaxNumBatchedTokens       int    `json:"max_num_batched_tokens"`
		BlockSize                 int    `json:"block_size"`
		NumBlocks                 int    `json:"num_blocks"`
		MaxModelLen               int    `json:"max_model_len"`
		LongPrefillTokenThreshold int    `json:"long_prefill_token_threshold"`
		EnablePrefixCaching       bool   `json:"enable_prefix_caching"`
		SchedulingPolicy          string `json:"scheduling_policy"`
	} `json:"config"`
	Requests []struct {
		ID          string  `json:"id"`
		ArrivalStep int     `json:"arrival_step"`
		Prompt      []int64 `json:"prompt"`
		Output      []int64 `json:"output"`
		MaxTokens   int     `json:"max_tokens"`
		Priority    int     `json:"priority"`
	} `json:"requests"`
	MaxSteps int `json:"max_steps"`
}

// step is one step of a trace: the fields the oracle records, compared field by field.
type step struct {
	Step       int                `json:"step"`
	Scheduled  map[string]int     `json:"scheduled"`
	Preempted  []string           `json:"preempted"`
	Finished   []string           `json:"finished"`
	FreeBlocks int                `json:"free_blocks"`
	Running    []string           `json:"running"`
	NumRunning int                `json:"num_running"`
	NumWaiting int                `json:"num_waiting"`
	KVUsage    float64            `json:"kv_usage"`
	Blocks     map[string][][]any `json:"blocks"`
}

type trace struct {
	Steps    []step `json:"steps"`
	Finished map[string]struct {
		Reason       string `json:"reason"`
		OutputTokens int    `json:"output_tokens"`
		Step         int    `json:"step"`
	} `json:"finished"`
}

// run drives the scenario through the Go scheduler exactly as oracle/run_scheduler.py drives
// vLLM's: arrivals by step, schedule(), the same GPU mock, update_from_output().
func run(t *testing.T, sc scenario) trace {
	t.Helper()
	c := sc.Config
	s, err := New(Config{
		MaxNumSeqs: c.MaxNumSeqs, MaxNumBatchedTokens: c.MaxNumBatchedTokens, BlockSize: c.BlockSize,
		NumBlocks: c.NumBlocks, MaxModelLen: c.MaxModelLen, LongPrefillTokenThreshold: c.LongPrefillTokenThreshold,
		EnableChunkedPrefill: true, Policy: Policy(c.SchedulingPolicy), EnablePrefixCaching: c.EnablePrefixCaching,
		ReserveFullSequence: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	reqs := sc.Requests
	sort.SliceStable(reqs, func(i, j int) bool {
		if reqs[i].ArrivalStep != reqs[j].ArrivalStep {
			return reqs[i].ArrivalStep < reqs[j].ArrivalStep
		}
		return reqs[i].ID < reqs[j].ID
	})
	outputs := map[string][]int64{}
	sampled := map[string]int{}
	next := 0
	maxSteps := sc.MaxSteps
	if maxSteps == 0 {
		maxSteps = 10000
	}
	tr := trace{}
	tr.Finished = map[string]struct {
		Reason       string `json:"reason"`
		OutputTokens int    `json:"output_tokens"`
		Step         int    `json:"step"`
	}{}
	for i := 0; i < maxSteps; i++ {
		for next < len(reqs) && reqs[next].ArrivalStep <= i {
			r := reqs[next]
			mt := r.MaxTokens
			if mt == 0 {
				mt = 1 << 20
			}
			if err := s.Add(&Request{ID: r.ID, Priority: r.Priority, ArrivalTime: float64(r.ArrivalStep),
				Prompt: r.Prompt, MaxTokens: mt, EOS: eosToken}); err != nil {
				t.Fatal(err)
			}
			outputs[r.ID] = r.Output
			next++
		}
		if !s.HasUnfinished() && next == len(reqs) {
			break
		}
		out := s.Schedule()
		mock := map[string]int64{}
		for id := range out.Scheduled {
			r := s.Request(id)
			if r.IsPrefillChunk {
				continue
			}
			k, o := sampled[id], outputs[id]
			if k == len(o)-1 {
				mock[id] = eosToken
			} else {
				mock[id] = o[k]
			}
			sampled[id] = k + 1
		}
		var done []string
		for _, f := range s.Update(out, mock) {
			done = append(done, f.ID)
			reason := map[Status]string{FinishedStopped: "stop", FinishedLength: "length"}[f.Status]
			tr.Finished[f.ID] = struct {
				Reason       string `json:"reason"`
				OutputTokens int    `json:"output_tokens"`
				Step         int    `json:"step"`
			}{reason, sampled[f.ID], i}
		}
		sort.Strings(done)
		nr, nw := s.Counts()
		st := step{
			Step: i, Scheduled: out.Scheduled, Preempted: sortedStrings(out.Preempted), Finished: done,
			FreeBlocks: s.KV.Pool.NumFree(), Running: sortedStrings(s.Running()), NumRunning: nr, NumWaiting: nw,
			KVUsage: s.KV.Pool.Usage(), Blocks: map[string][][]any{},
		}
		for _, id := range st.Running {
			for _, b := range s.KV.Blocks(id) {
				var h any
				if b.Hash != nil {
					h = hex.EncodeToString(b.Hash)
				}
				st.Blocks[id] = append(st.Blocks[id], []any{h, float64(b.RefCnt)})
			}
		}
		tr.Steps = append(tr.Steps, st)
	}
	return tr
}

func sortedStrings(in []string) []string {
	out := append([]string{}, in...)
	sort.Strings(out)
	if out == nil {
		out = []string{}
	}
	return out
}

// earliestDivergence compares two traces step by step, field by field, and names the first
// difference (#2021 B8: report the earliest divergence, not only the final mismatch).
func earliestDivergence(want, got trace) string {
	for i := 0; i < max(len(want.Steps), len(got.Steps)); i++ {
		if i >= len(want.Steps) {
			return fmt.Sprintf("step %d: Go ran a step the oracle did not (scheduled %v)", i, got.Steps[i].Scheduled)
		}
		if i >= len(got.Steps) {
			return fmt.Sprintf("step %d: the oracle ran a step Go did not (scheduled %v)", i, want.Steps[i].Scheduled)
		}
		w, g := want.Steps[i], got.Steps[i]
		for _, f := range []struct {
			name string
			w, g any
		}{
			{"scheduled", nonNil(w.Scheduled), nonNil(g.Scheduled)},
			{"preempted", sortedStrings(w.Preempted), sortedStrings(g.Preempted)},
			{"finished", sortedStrings(w.Finished), sortedStrings(g.Finished)},
			{"running", sortedStrings(w.Running), sortedStrings(g.Running)},
			{"num_running", w.NumRunning, g.NumRunning},
			{"num_waiting", w.NumWaiting, g.NumWaiting},
			{"free_blocks", w.FreeBlocks, g.FreeBlocks},
			{"kv_usage", fmt.Sprintf("%.12f", w.KVUsage), fmt.Sprintf("%.12f", g.KVUsage)},
			{"blocks", nonNilBlocks(w.Blocks), nonNilBlocks(g.Blocks)},
		} {
			if !reflect.DeepEqual(f.w, f.g) {
				return fmt.Sprintf("step %d, field %s: oracle %v, Go %v", i, f.name, f.w, f.g)
			}
		}
	}
	if !reflect.DeepEqual(want.Finished, got.Finished) {
		return fmt.Sprintf("finished: oracle %v, Go %v", want.Finished, got.Finished)
	}
	return ""
}

func nonNil(m map[string]int) map[string]int {
	if m == nil {
		return map[string]int{}
	}
	return m
}

func nonNilBlocks(m map[string][][]any) map[string][][]any {
	if m == nil {
		return map[string][][]any{}
	}
	return m
}

// TestDifferentialAgainstVLLM runs every scenario under oracle/scenarios through the Go
// scheduler and compares it, step by step, with the trace vLLM v0.32.0's own Scheduler
// produced for it (testdata/golden/<name>.json, written by oracle/run_scheduler.py). The
// target is zero mismatches; a failure names the earliest step and field that differ.
func TestDifferentialAgainstVLLM(t *testing.T) {
	scenarios, err := filepath.Glob("../../oracle/scenarios/*.json")
	if err != nil || len(scenarios) == 0 {
		t.Fatalf("no oracle scenarios found (%v)", err)
	}
	for _, path := range scenarios {
		name := strings.TrimSuffix(filepath.Base(path), ".json")
		t.Run(name, func(t *testing.T) {
			var sc scenario
			readJSON(t, path, &sc)
			golden := filepath.Join("testdata", "golden", name+".json")
			if _, err := os.Stat(golden); err != nil {
				t.Fatalf("no golden trace %s: generate it with oracle/run_scheduler.py (see docs/testing.md)", golden)
			}
			var want trace
			readJSON(t, golden, &want)
			if d := earliestDivergence(want, run(t, sc)); d != "" {
				t.Fatalf("earliest divergence from vLLM v0.32.0: %s", d)
			}
		})
	}
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

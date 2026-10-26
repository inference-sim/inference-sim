package engine

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/core/sched"
)

// eosToken is tests/v1/core/utils.py:43@5df63a72, which the GPU mock samples last.
const eosToken = 50256

type scenario struct {
	Config struct {
		MaxNumSeqs                int    `json:"max_num_seqs"`
		MaxNumBatchedTokens       int    `json:"max_num_batched_tokens"`
		BlockSize                 int    `json:"block_size"`
		NumBlocks                 int    `json:"num_blocks"`
		MaxModelLen               int    `json:"max_model_len"`
		LongPrefillTokenThreshold int    `json:"long_prefill_token_threshold"`
		EnablePrefixCaching       bool   `json:"enable_prefix_caching"`
		SchedulingPolicy          string `json:"scheduling_policy"`
		PipelineParallelSize      int    `json:"pipeline_parallel_size"`
		AsyncScheduling           bool   `json:"async_scheduling"`
		UseV2ModelRunner          bool   `json:"use_v2_model_runner"`
		Speculative               *struct {
			Method    string `json:"method"`
			NumTokens int    `json:"num_speculative_tokens"`
			Draft     string `json:"draft"`
		} `json:"speculative"`
	} `json:"config"`
	Requests []struct {
		ID          string  `json:"id"`
		ArrivalStep int     `json:"arrival_step"`
		Prompt      []int64 `json:"prompt"`
		Output      []int64 `json:"output"`
		MaxTokens   int     `json:"max_tokens"`
		Priority    int     `json:"priority"`
		// Accepted is the drafts accepted at each verify step, an input (#2021 B3).
		Accepted []int `json:"accepted"`
	} `json:"requests"`
	MaxSteps int `json:"max_steps"`
}

type batchRec struct {
	Batch     int            `json:"batch"`
	Scheduled map[string]int `json:"scheduled"`
	Preempted []string       `json:"preempted"`
	Samples   []string       `json:"samples"`
	// SpecScheduled is how many drafts each request verifies in the batch.
	SpecScheduled map[string]int `json:"spec_scheduled,omitempty"`
}

type stepRec struct {
	Step           int                `json:"step"`
	Idle           bool               `json:"idle"`
	ScheduledBatch *int               `json:"scheduled_batch"`
	QueueLen       int                `json:"queue_len"`
	Processed      bool               `json:"processed"`
	Finished       []string           `json:"finished"`
	FreeBlocks     int                `json:"free_blocks"`
	NumRunning     int                `json:"num_running"`
	NumWaiting     int                `json:"num_waiting"`
	Running        []string           `json:"running"`
	Computed       map[string]int     `json:"computed"`
	Placeholders   map[string]int     `json:"placeholders"`
	Blocks         map[string][][]any `json:"blocks"`
}

type trace struct {
	Provenance struct {
		Tier           int    `json:"tier"`
		VLLMVersion    string `json:"vllm_version"`
		VLLMCommit     string `json:"vllm_commit"`
		ScenarioSHA256 string `json:"scenario_sha256"`
		MockVersion    string `json:"mock_version"`
		Engine         struct {
			MaxConcurrentBatches int `json:"max_concurrent_batches"`
		} `json:"engine"`
	} `json:"provenance"`
	Batches  []batchRec `json:"batches"`
	Steps    []stepRec  `json:"steps"`
	Finished map[string]struct {
		Reason       string `json:"reason"`
		OutputTokens int    `json:"output_tokens"`
		Step         int    `json:"step"`
	} `json:"finished"`
}

// mockGPU is the shared GPU mock (oracle/run_enginecore.py): each request in a batch's
// Samples samples its next assigned output token when the batch's output is asked for, the
// last one as EOS.
type mockGPU struct {
	outputs map[string][]int64
	sampled map[string]int
	// accepted and verified are the drafts accepted at each of a request's verify steps, and
	// how many verify steps it has had.
	accepted map[string][]int
	verified map[string]int
	// async is whether the engine schedules asynchronously: its drafts are placeholders the
	// worker fills, so each is a draft the plan may accept.
	async bool
}

func (g *mockGPU) Output(b Batch) map[string][]int64 {
	out := map[string][]int64{}
	for _, id := range b.Samples {
		n := 1
		if drafts := b.Out.SpecDecode[id]; len(drafts) > 0 {
			// The plan's accepted drafts, at most the real ones: a placeholder (-1) the
			// scheduler set for a first decode is always rejected, while an async placeholder
			// is the worker's real draft (oracle/run_enginecore.py).
			real := len(drafts)
			if !g.async {
				real = 0
				for _, d := range drafts {
					if d >= 0 {
						real++
					}
				}
			}
			plan, i := g.accepted[id], g.verified[id]
			g.verified[id] = i + 1
			if i < len(plan) {
				n += min(plan[i], real)
			}
		}
		o := g.outputs[id]
		for j := 0; j < n; j++ {
			k := g.sampled[id]
			if k >= len(o)-1 {
				out[id] = append(out[id], eosToken)
			} else {
				out[id] = append(out[id], o[k])
			}
			g.sampled[id] = k + 1
			if k >= len(o)-1 {
				break
			}
		}
	}
	return out
}

// run drives a scenario through Core exactly as oracle/run_enginecore.py drives vLLM's
// EngineCore: arrivals by step, then one engine step.
func run(t *testing.T, sc scenario) trace {
	t.Helper()
	c := sc.Config
	pp := max(c.PipelineParallelSize, 1)
	maxLen := c.MaxModelLen
	if maxLen == 0 {
		maxLen = c.MaxNumBatchedTokens
	}
	var spec sched.Spec
	if sp := c.Speculative; sp != nil {
		spec = sched.ResolveSpec(sched.SpecOptions{Method: sp.Method, NumTokens: sp.NumTokens,
			NextnPredictLayers: draftNextnLayers(t, sp.Draft)})
	}
	queue := MaxConcurrentBatches(pp, c.AsyncScheduling, c.UseV2ModelRunner)
	s, err := sched.New(sched.Config{
		MaxNumSeqs: c.MaxNumSeqs, MaxNumBatchedTokens: c.MaxNumBatchedTokens, BlockSize: c.BlockSize,
		NumBlocks: c.NumBlocks, MaxModelLen: maxLen, LongPrefillTokenThreshold: c.LongPrefillTokenThreshold,
		EnableChunkedPrefill: true, Policy: sched.Policy(c.SchedulingPolicy), EnablePrefixCaching: c.EnablePrefixCaching,
		ReserveFullSequence: true, Async: c.AsyncScheduling, DecodeStagger: pp, UseV2ModelRunner: c.UseV2ModelRunner,
		Spec: spec, MaxConcurrentBatches: queue,
	})
	if err != nil {
		t.Fatal(err)
	}
	gpu := &mockGPU{outputs: map[string][]int64{}, sampled: map[string]int{}, accepted: map[string][]int{},
		verified: map[string]int{}, async: c.AsyncScheduling}
	core := &Core{Sched: s, QueueSize: queue, GPU: gpu}
	reqs := append(sc.Requests[:0:0], sc.Requests...)
	sort.SliceStable(reqs, func(i, j int) bool {
		if reqs[i].ArrivalStep != reqs[j].ArrivalStep {
			return reqs[i].ArrivalStep < reqs[j].ArrivalStep
		}
		return reqs[i].ID < reqs[j].ID
	})
	maxSteps := sc.MaxSteps
	if maxSteps == 0 {
		maxSteps = 10000
	}
	var tr trace
	tr.Finished = map[string]struct {
		Reason       string `json:"reason"`
		OutputTokens int    `json:"output_tokens"`
		Step         int    `json:"step"`
	}{}
	next := 0
	for i := 0; i < maxSteps; i++ {
		for next < len(reqs) && reqs[next].ArrivalStep <= i {
			r := reqs[next]
			mt := r.MaxTokens
			if mt == 0 {
				mt = 1 << 20
			}
			if err := s.Add(&sched.Request{ID: r.ID, Priority: r.Priority, ArrivalTime: float64(r.ArrivalStep),
				Prompt: r.Prompt, MaxTokens: mt, EOS: eosToken}); err != nil {
				t.Fatal(err)
			}
			gpu.outputs[r.ID] = r.Output
			gpu.accepted[r.ID] = r.Accepted
			next++
		}
		if !core.HasWork() && next == len(reqs) {
			break
		}
		if !core.HasWork() {
			tr.Steps = append(tr.Steps, stepRec{Step: i, Idle: true})
			continue
		}
		res := core.Step()
		if spec.NumTokens > 0 && !c.AsyncScheduling && res.Executed && res.Scheduled != nil {
			// EngineCore.post_step (core.py:689-696): without async scheduling, the drafter's
			// proposals reach the scheduler after a step that ran, num_speculative_tokens for
			// every request it ran, as the oracle's drafter mock proposes them.
			drafts := map[string][]int64{}
			for id := range res.Scheduled.Out.Scheduled {
				drafts[id] = make([]int64, spec.NumTokens)
			}
			s.UpdateDrafts(drafts)
		}
		rec := stepRec{Step: i, QueueLen: core.InFlight(), Processed: res.Processed != nil, Finished: []string{}}
		if res.Scheduled != nil {
			b := res.Scheduled
			id := b.ID
			rec.ScheduledBatch = &id
			pre := append([]string{}, b.Out.Preempted...)
			sort.Strings(pre)
			samples := append([]string{}, b.Samples...)
			sort.Strings(samples)
			br := batchRec{Batch: b.ID, Scheduled: b.Out.Scheduled, Preempted: nonNil(pre), Samples: nonNil(samples)}
			for rid, d := range b.Out.SpecDecode {
				if len(d) > 0 {
					if br.SpecScheduled == nil {
						br.SpecScheduled = map[string]int{}
					}
					br.SpecScheduled[rid] = len(d)
				}
			}
			tr.Batches = append(tr.Batches, br)
		}
		for _, f := range res.Finished {
			rec.Finished = append(rec.Finished, f.ID)
			reason := map[sched.Status]string{sched.FinishedStopped: "stop", sched.FinishedLength: "length"}[f.Status]
			tr.Finished[f.ID] = struct {
				Reason       string `json:"reason"`
				OutputTokens int    `json:"output_tokens"`
				Step         int    `json:"step"`
			}{reason, gpu.sampled[f.ID], i}
		}
		sort.Strings(rec.Finished)
		nr, nw := s.Counts()
		rec.NumRunning, rec.NumWaiting, rec.FreeBlocks = nr, nw, s.KV.Pool.NumFree()
		rec.Running = []string{}
		rec.Computed, rec.Placeholders, rec.Blocks = map[string]int{}, map[string]int{}, map[string][][]any{}
		running := s.Running()
		sort.Strings(running)
		for _, id := range running {
			r := s.Request(id)
			rec.Running = append(rec.Running, id)
			rec.Computed[id] = r.NumComputed
			rec.Placeholders[id] = r.Placeholders
			for _, b := range s.KV.Blocks(id) {
				var h any
				if b.Hash != nil {
					h = hex.EncodeToString(b.Hash)
				}
				rec.Blocks[id] = append(rec.Blocks[id], []any{h, float64(b.RefCnt)})
			}
		}
		tr.Steps = append(tr.Steps, rec)
	}
	return tr
}

func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

// earliestDivergence names the first batch, then the first step and field, that differ.
func earliestDivergence(want, got trace) string {
	for i := 0; i < max(len(want.Batches), len(got.Batches)); i++ {
		if i >= len(want.Batches) {
			return fmt.Sprintf("batch %d: blis-vllm scheduled a batch vLLM did not (%v)", i, got.Batches[i].Scheduled)
		}
		if i >= len(got.Batches) {
			return fmt.Sprintf("batch %d: vLLM scheduled a batch blis-vllm did not (%v)", i, want.Batches[i].Scheduled)
		}
		w, g := want.Batches[i], got.Batches[i]
		if !reflect.DeepEqual(nonNilMap(w.Scheduled), nonNilMap(g.Scheduled)) {
			return fmt.Sprintf("batch %d, scheduled: vLLM %v, blis-vllm %v", i, w.Scheduled, g.Scheduled)
		}
		if !reflect.DeepEqual(nonNil(w.Preempted), nonNil(g.Preempted)) {
			return fmt.Sprintf("batch %d, preempted: vLLM %v, blis-vllm %v", i, w.Preempted, g.Preempted)
		}
		if !reflect.DeepEqual(nonNil(w.Samples), nonNil(g.Samples)) {
			return fmt.Sprintf("batch %d, samples: vLLM %v, blis-vllm %v", i, w.Samples, g.Samples)
		}
		if !reflect.DeepEqual(nonNilMap(w.SpecScheduled), nonNilMap(g.SpecScheduled)) {
			return fmt.Sprintf("batch %d, spec_scheduled: vLLM %v, blis-vllm %v", i, w.SpecScheduled, g.SpecScheduled)
		}
	}
	for i := 0; i < max(len(want.Steps), len(got.Steps)); i++ {
		if i >= len(want.Steps) || i >= len(got.Steps) {
			return fmt.Sprintf("step %d: the runs have %d and %d steps", i, len(want.Steps), len(got.Steps))
		}
		w, g := want.Steps[i], got.Steps[i]
		for _, f := range []struct {
			name string
			w, g any
		}{
			{"idle", w.Idle, g.Idle},
			{"scheduled_batch", ptr(w.ScheduledBatch), ptr(g.ScheduledBatch)},
			{"queue_len", w.QueueLen, g.QueueLen},
			{"processed", w.Processed, g.Processed},
			{"finished", nonNil(w.Finished), nonNil(g.Finished)},
			{"free_blocks", w.FreeBlocks, g.FreeBlocks},
			{"num_running", w.NumRunning, g.NumRunning},
			{"num_waiting", w.NumWaiting, g.NumWaiting},
			{"running", nonNil(w.Running), nonNil(g.Running)},
			{"computed", nonNilMap(w.Computed), nonNilMap(g.Computed)},
			{"placeholders", nonNilMap(w.Placeholders), nonNilMap(g.Placeholders)},
			{"blocks", nonNilBlocks(w.Blocks), nonNilBlocks(g.Blocks)},
		} {
			if !reflect.DeepEqual(f.w, f.g) {
				return fmt.Sprintf("step %d, field %s: vLLM %v, blis-vllm %v", i, f.name, f.w, f.g)
			}
		}
	}
	if !reflect.DeepEqual(want.Finished, got.Finished) {
		return fmt.Sprintf("finished: vLLM %v, blis-vllm %v", want.Finished, got.Finished)
	}
	return ""
}

func ptr(p *int) any {
	if p == nil {
		return nil
	}
	return *p
}

func nonNilMap(m map[string]int) map[string]int {
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

// checkProvenance refuses a trace not produced from this exact scenario by the pinned
// vLLM release under the current GPU mock.
func checkProvenance(t *testing.T, scenarioPath string, tr trace) {
	t.Helper()
	raw, err := os.ReadFile(scenarioPath)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	var mocks struct {
		Version string `json:"version"`
	}
	readJSON(t, "../../oracle/mocks.json", &mocks)
	p := tr.Provenance
	for _, c := range []struct{ field, got, want string }{
		{"scenario_sha256", p.ScenarioSHA256, hex.EncodeToString(sum[:])},
		{"vllm_version", strings.SplitN(p.VLLMVersion, "+", 2)[0], "0.32.0"},
		{"vllm_commit", p.VLLMCommit, "5df63a7203cdb9328d824a27b682e77c7b9928be"},
		{"mock_version", p.MockVersion, mocks.Version},
	} {
		if c.got != c.want {
			t.Fatalf("trace for %s records %s %q, want %q: regenerate it with oracle/run_enginecore.py",
				filepath.Base(scenarioPath), c.field, c.got, c.want)
		}
	}
	if p.Tier != 3 {
		t.Fatalf("trace for %s records tier %d, want 3", filepath.Base(scenarioPath), p.Tier)
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

// TestEngineCoreAgainstVLLM runs every tier-3 scenario (oracle/scenarios-t3) through Core
// and compares it, batch by batch and step by step, with the trace vLLM v0.32.0's own
// EngineCore.step_with_batch_queue produced (testdata/golden/<name>.json). The target is
// zero mismatches.
func TestEngineCoreAgainstVLLM(t *testing.T) {
	scenarios, err := filepath.Glob("../../oracle/scenarios-t3/*.json")
	if err != nil || len(scenarios) == 0 {
		t.Fatalf("no tier-3 scenarios (%v)", err)
	}
	for _, path := range scenarios {
		name := strings.TrimSuffix(filepath.Base(path), ".json")
		t.Run(name, func(t *testing.T) {
			var sc scenario
			readJSON(t, path, &sc)
			var want trace
			readJSON(t, filepath.Join("testdata", "golden", name+".json"), &want)
			checkProvenance(t, path, want)
			got := run(t, sc)
			if q := MaxConcurrentBatches(max(sc.Config.PipelineParallelSize, 1), sc.Config.AsyncScheduling, sc.Config.UseV2ModelRunner); q != want.Provenance.Engine.MaxConcurrentBatches {
				t.Fatalf("max_concurrent_batches: vLLM %d, blis-vllm %d", want.Provenance.Engine.MaxConcurrentBatches, q)
			}
			if d := earliestDivergence(want, got); d != "" {
				t.Fatalf("earliest divergence from vLLM v0.32.0: %s", d)
			}
		})
	}
}

// TestEngineCoreFuzzCorpusAgainstVLLM is the seeded fuzz test of tier 3: BLIS_VLLM_T3_FUZZ_DIR
// names a directory of oracle/gen_fuzz_t3.py scenarios with vLLM's trace for each
// (<name>.trace.json). Every scenario must match at zero mismatches.
func TestEngineCoreFuzzCorpusAgainstVLLM(t *testing.T) {
	dir := os.Getenv("BLIS_VLLM_T3_FUZZ_DIR")
	if dir == "" {
		t.Skip("BLIS_VLLM_T3_FUZZ_DIR is unset: the tier-3 fuzz corpus was NOT checked (CI's oracle job sets it)")
	}
	traces, err := filepath.Glob(filepath.Join(dir, "*.trace.json"))
	if err != nil || len(traces) == 0 {
		t.Fatalf("no traces in %s (%v)", dir, err)
	}
	mismatches := 0
	for _, tp := range traces {
		name := strings.TrimSuffix(filepath.Base(tp), ".trace.json")
		var sc scenario
		readJSON(t, filepath.Join(dir, name+".json"), &sc)
		var want trace
		readJSON(t, tp, &want)
		checkProvenance(t, filepath.Join(dir, name+".json"), want)
		if d := earliestDivergence(want, run(t, sc)); d != "" {
			mismatches++
			if mismatches <= 20 {
				t.Errorf("%s: earliest divergence from vLLM v0.32.0: %s", name, d)
			}
		}
	}
	t.Logf("tier-3 fuzz corpus: %d scenarios, %d mismatches", len(traces), mismatches)
	if mismatches > 0 {
		t.Fail()
	}
}

// TestScenarioFromEnv checks one scenario against one trace (BLIS_VLLM_T3_SCENARIO,
// BLIS_VLLM_T3_TRACE), for shrinking a failing fuzz seed.
func TestScenarioFromEnv(t *testing.T) {
	sp, tp := os.Getenv("BLIS_VLLM_T3_SCENARIO"), os.Getenv("BLIS_VLLM_T3_TRACE")
	if sp == "" || tp == "" {
		t.Skip("BLIS_VLLM_T3_SCENARIO and BLIS_VLLM_T3_TRACE are unset")
	}
	var sc scenario
	readJSON(t, sp, &sc)
	var want trace
	readJSON(t, tp, &want)
	if d := earliestDivergence(want, run(t, sc)); d != "" {
		t.Fatalf("earliest divergence from vLLM v0.32.0: %s", d)
	}
}

// draftNextnLayers is a draft checkpoint's num_nextn_predict_layers (oracle/drafts/<name>), 0
// for none.
func draftNextnLayers(t *testing.T, draft string) int {
	t.Helper()
	if draft == "" {
		return 0
	}
	var c struct {
		Layers int `json:"num_nextn_predict_layers"`
	}
	readJSON(t, filepath.Join("..", "..", "oracle", "drafts", draft, "config.json"), &c)
	return c.Layers
}

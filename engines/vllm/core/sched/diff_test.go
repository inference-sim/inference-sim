package sched

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/core/kv"
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
		// The knobs vLLM's test helper fixes or omits, passed to its config classes.
		Watermark             float64 `json:"watermark"`
		MaxNumActiveSeqs      int     `json:"max_num_active_seqs"`
		MaxNumScheduledTokens int     `json:"max_num_scheduled_tokens"`
		LongPrefillAdaptive   bool    `json:"long_prefill_token_threshold_adaptive"`
		MaxLoRAs              int     `json:"max_loras"`
		KVEvents              bool    `json:"kv_events"`
		KVGroups              []struct {
			Kind          string `json:"kind"`
			SlidingWindow int    `json:"sliding_window"`
		} `json:"kv_groups"`
		// KVLayers is a model's attention layers as runs of one spec each, in model order,
		// which vLLM's get_kv_cache_groups groups (kv.GroupsOfLayers).
		KVLayers                    []kvLayerRun `json:"kv_layers"`
		DisableHybridKVCacheManager bool         `json:"disable_hybrid_kv_cache_manager"`
		Speculative                 *struct {
			Method           string `json:"method"`
			NumTokens        int    `json:"num_speculative_tokens"`
			Draft            string `json:"draft"`
			ParallelDrafting bool   `json:"parallel_drafting"`
		} `json:"speculative"`
	} `json:"config"`
	Requests []struct {
		ID          string  `json:"id"`
		ArrivalStep int     `json:"arrival_step"`
		Prompt      []int64 `json:"prompt"`
		Output      []int64 `json:"output"`
		MaxTokens   int     `json:"max_tokens"`
		Priority    int     `json:"priority"`
		// Accepted is the GPU mock's draft acceptance: drafts accepted at each verify step.
		Accepted []int  `json:"accepted"`
		LoRA     string `json:"lora"`
	} `json:"requests"`
	MaxSteps int `json:"max_steps"`
}

// step is one step of a trace: the fields the oracle records, compared field by field.
type step struct {
	Step       int              `json:"step"`
	Scheduled  map[string]int   `json:"scheduled"`
	Preempted  []string         `json:"preempted"`
	Finished   []string         `json:"finished"`
	FreeBlocks int              `json:"free_blocks"`
	Running    []string         `json:"running"`
	NumRunning int              `json:"num_running"`
	NumWaiting int              `json:"num_waiting"`
	KVUsage    float64          `json:"kv_usage"`
	Blocks     map[string][]any `json:"blocks"`
	// SpecScheduled is the drafts each request verified this step, when any did.
	SpecScheduled map[string]int `json:"spec_scheduled,omitempty"`
	// KVEvents are the step's KV cache events, by content, when the scenario turns them on:
	// ["stored", hashes, parent, tokens, block_size, lora_name] or ["removed", hashes].
	KVEvents []any `json:"kv_events,omitempty"`
}

type trace struct {
	Provenance struct {
		VLLMVersion    string `json:"vllm_version"`
		VLLMCommit     string `json:"vllm_commit"`
		ScenarioSHA256 string `json:"scenario_sha256"`
		MockVersion    string `json:"mock_version"`
	} `json:"provenance"`
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
	var spec Spec
	if c.Speculative != nil {
		spec = ResolveSpec(SpecOptions{Method: c.Speculative.Method, NumTokens: c.Speculative.NumTokens,
			ParallelDrafting: c.Speculative.ParallelDrafting, NextnPredictLayers: draftNextnLayers(t, c.Speculative.Draft)})
	}
	s, err := New(Config{
		MaxNumSeqs: c.MaxNumSeqs, MaxNumBatchedTokens: c.MaxNumBatchedTokens, BlockSize: c.BlockSize,
		NumBlocks: c.NumBlocks, MaxModelLen: c.MaxModelLen, LongPrefillTokenThreshold: c.LongPrefillTokenThreshold,
		EnableChunkedPrefill: true, Policy: Policy(c.SchedulingPolicy), EnablePrefixCaching: c.EnablePrefixCaching,
		ReserveFullSequence: true, Watermark: c.Watermark, MaxNumActiveSeqs: c.MaxNumActiveSeqs,
		MaxNumScheduledTokens: c.MaxNumScheduledTokens, LongPrefillAdaptive: c.LongPrefillAdaptive,
		Spec: spec, MaxLoRAs: c.MaxLoRAs, KVEvents: c.KVEvents, KVGroups: kvGroups(t, c.KVGroups, c.KVLayers, c.DisableHybridKVCacheManager),
	})
	if err != nil {
		t.Fatal(err)
	}
	// LoRA ids numbered from 1 in name order, as the oracle numbers them.
	var adapters []string
	for _, r := range sc.Requests {
		if r.LoRA != "" && indexOfString(adapters, r.LoRA) < 0 {
			adapters = append(adapters, r.LoRA)
		}
	}
	sort.Strings(adapters)
	reqs := sc.Requests
	sort.SliceStable(reqs, func(i, j int) bool {
		if reqs[i].ArrivalStep != reqs[j].ArrivalStep {
			return reqs[i].ArrivalStep < reqs[j].ArrivalStep
		}
		return reqs[i].ID < reqs[j].ID
	})
	outputs := map[string][]int64{}
	accepted := map[string][]int{}
	verifies := map[string]int{}
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
			req := &Request{ID: r.ID, Priority: r.Priority, ArrivalTime: float64(r.ArrivalStep),
				Prompt: r.Prompt, MaxTokens: mt, EOS: eosToken}
			if r.LoRA != "" {
				// vLLM's LoRA block key (kv_cache_utils.py:578-595@5df63a72), with the oracle's
				// adapter path.
				req.LoRA = indexOfString(adapters, r.LoRA) + 1
				req.LoRAName = r.LoRA
				req.Keys.Extra = []kv.ExtraKey{{"lora", r.LoRA, "/adapters/" + r.LoRA}}
			}
			if err := s.Add(req); err != nil {
				t.Fatal(err)
			}
			outputs[r.ID] = r.Output
			accepted[r.ID] = r.Accepted
			next++
		}
		if !s.HasUnfinished() && next == len(reqs) {
			break
		}
		out := s.Schedule()
		// The GPU mock (oracle/mocks.json): a step that completes a request's known tokens
		// samples one token, plus, when it verifies drafts, the scenario's accepted count for
		// that verify step (at most the real drafts; a -1 placeholder is always rejected).
		mock := map[string][]int64{}
		specScheduled := map[string]int{}
		for id := range out.Scheduled {
			r := s.Request(id)
			if r.IsPrefillChunk {
				continue
			}
			n := 1
			if drafts := out.SpecDecode[id]; len(drafts) > 0 {
				specScheduled[id] = len(drafts)
				real := 0
				for _, d := range drafts {
					if d >= 0 {
						real++
					}
				}
				plan, v := accepted[id], verifies[id]
				verifies[id] = v + 1
				a := 0
				if v < len(plan) {
					a = plan[v]
				}
				n = 1 + min(a, real)
			}
			o := outputs[id]
			for j := 0; j < n && sampled[id] < len(o); j++ {
				k := sampled[id]
				if k == len(o)-1 {
					mock[id] = append(mock[id], eosToken)
				} else {
					mock[id] = append(mock[id], o[k])
				}
				sampled[id] = k + 1
			}
		}
		var done []string
		fins := s.Update(out, mock)
		if s.cfg.Spec.NumTokens > 0 && len(out.Scheduled) > 0 {
			// The drafter proposes K drafts for every request the step ran, as the oracle's
			// mock does after update_from_output (EngineCore.post_step).
			drafts := map[string][]int64{}
			for id := range out.Scheduled {
				drafts[id] = make([]int64, s.cfg.Spec.NumTokens)
			}
			s.UpdateDrafts(drafts)
		}
		for _, f := range fins {
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
			KVUsage: s.KV.Pool.Usage(), Blocks: map[string][]any{},
		}
		if len(specScheduled) > 0 {
			st.SpecScheduled = specScheduled
		}
		if c.KVEvents {
			st.KVEvents = eventsJSON(s.TakeKVEvents())
		}
		for _, id := range st.Running {
			if s.KV.NumGroups() == 1 {
				st.Blocks[id] = blockRows(s.KV.Blocks(id), false)
				continue
			}
			// A hybrid model's blocks, one list per group, a null slot written as the oracle
			// writes it.
			rows := []any{}
			for g := 0; g < s.KV.NumGroups(); g++ {
				rows = append(rows, blockRows(s.KV.GroupBlocks(g, id), true))
			}
			st.Blocks[id] = rows
		}
		tr.Steps = append(tr.Steps, st)
	}
	return tr
}

// eventsJSON writes KV events as the oracle does: a block hash is the integer vLLM publishes
// by default, the low 64 bits of the hash read big-endian (VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES,
// vllm/envs.py:2033-2035 and kv_cache_utils.py:96-99@5df63a72), in decimal.
func eventsJSON(evs []kv.Event) []any {
	out := []any{}
	h := func(b kv.BlockHash) any {
		if b == nil {
			return nil
		}
		return strconv.FormatUint(binary.BigEndian.Uint64(b[len(b)-8:]), 10)
	}
	hs := func(bs []kv.BlockHash) []any {
		o := []any{}
		for _, b := range bs {
			o = append(o, h(b))
		}
		return o
	}
	for _, e := range evs {
		if e.Removed {
			out = append(out, []any{"removed", hs(e.Hashes)})
			continue
		}
		toks := []any{}
		for _, t := range e.Tokens {
			toks = append(toks, float64(t))
		}
		var lora any
		if e.LoRA != "" {
			lora = e.LoRA
		}
		out = append(out, []any{"stored", hs(e.Hashes), h(e.Parent), toks, float64(e.BlockSize), lora})
	}
	return out
}

func nonNilAny(v []any) []any {
	if v == nil {
		return []any{}
	}
	return v
}

func indexOfString(xs []string, x string) int {
	for i, v := range xs {
		if v == x {
			return i
		}
	}
	return -1
}

// draftNextnLayers reads an MTP drafter's num_nextn_predict_layers from its configuration-only
// checkpoint under oracle/drafts, the file vLLM's SpeculativeConfig reads in the oracle.
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
			{"spec_scheduled", nonNil(w.SpecScheduled), nonNil(g.SpecScheduled)},
			{"kv_events", nonNilAny(w.KVEvents), nonNilAny(g.KVEvents)},
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

func nonNilBlocks(m map[string][]any) map[string][]any {
	if m == nil {
		return map[string][]any{}
	}
	return m
}

// blockRows writes a block table as the oracle does: [hash or null, ref count] per block,
// and, in a hybrid model, a null block as ["null", 0].
func blockRows(bs []*kv.Block, hybrid bool) []any {
	rows := []any{}
	for _, b := range bs {
		if hybrid && b.Null {
			rows = append(rows, []any{"null", float64(0)})
			continue
		}
		var h any
		if b.Hash != nil {
			h = hex.EncodeToString(b.Hash)
		}
		rows = append(rows, []any{h, float64(b.RefCnt)})
	}
	return rows
}

// kvLayerRun is a run of attention layers of one spec in a scenario's "kv_layers".
//
// DeepSeek V4's runs also state their own block size, tokens per state, state bytes, element
// size, alignment and bounded replay, as oracle/run_scheduler.py layer_specs builds them.
type kvLayerRun struct {
	Count          int    `json:"count"`
	Spec           string `json:"spec"`
	SlidingWindow  int    `json:"sliding_window"`
	NumKVHeads     int    `json:"num_kv_heads"`
	HeadSize       int    `json:"head_size"`
	BlockSize      int    `json:"block_size"`
	TokensPerState int    `json:"tokens_per_state"`
	StateBytes     int    `json:"state_bytes"`
	ElemBytes      int    `json:"elem_bytes"`
	Alignment      int    `json:"alignment"`
	BoundedReplay  bool   `json:"bounded_replay"`
}

// kvGroups is the scenario's KV cache groups for the scheduler: its "kv_layers" through
// GroupsOfLayers, or its "kv_groups" through ResolveGroups.
func kvGroups(t *testing.T, gs []struct {
	Kind          string `json:"kind"`
	SlidingWindow int    `json:"sliding_window"`
}, runs []kvLayerRun, disableHybrid bool) []kv.Group {
	t.Helper()
	if len(runs) > 0 {
		var layers []kv.Layer
		for _, r := range runs {
			spec, err := kv.ParseLayerSpec(r.Spec)
			if err != nil {
				t.Fatal(err)
			}
			heads := r.NumKVHeads
			if spec == kv.LayerMLA || spec == kv.LayerSlidingWindowMLA || spec == kv.LayerCircularBuffer {
				heads = 1 // DeepSeek V4's caches are one latent head (layer_specs)
			}
			for i := 0; i < r.Count; i++ {
				layers = append(layers, kv.Layer{Spec: spec, Window: r.SlidingWindow, KVHeads: heads, HeadSize: r.HeadSize,
					BlockSize: r.BlockSize, TokensPerState: r.TokensPerState, StateBytes: r.StateBytes,
					ElemBytes: r.ElemBytes, Alignment: r.Alignment, BoundedReplay: r.BoundedReplay})
			}
		}
		groups, err := kv.GroupsOfLayers(layers, disableHybrid)
		if err != nil {
			t.Fatal(err)
		}
		return groups
	}
	var layers []kv.Group
	for _, g := range gs {
		switch g.Kind {
		case "full":
			layers = append(layers, kv.Group{Kind: kv.FullAttention})
		case "sliding_window":
			layers = append(layers, kv.Group{Kind: kv.SlidingWindow, Window: g.SlidingWindow})
		default:
			t.Fatalf("unknown KV cache group kind %q", g.Kind)
		}
	}
	return kv.ResolveGroups(layers, disableHybrid)
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
			checkProvenance(t, path, want)
			if d := earliestDivergence(want, run(t, sc)); d != "" {
				t.Fatalf("earliest divergence from vLLM v0.32.0: %s", d)
			}
		})
	}
}

// checkProvenance refuses a golden trace that was not produced from this exact scenario file
// by the pinned vLLM release under the current GPU mock: a trace is only evidence for the
// scenario it was run on.
func checkProvenance(t *testing.T, scenarioPath string, tr trace) {
	t.Helper()
	raw, err := os.ReadFile(scenarioPath)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	p := tr.Provenance
	for _, c := range []struct{ field, got, want string }{
		{"scenario_sha256", p.ScenarioSHA256, hex.EncodeToString(sum[:])},
		// A build may carry a local version label ("0.32.0+cpu"); the release is what counts.
		{"vllm_version", strings.SplitN(p.VLLMVersion, "+", 2)[0], "0.32.0"},
		{"vllm_commit", p.VLLMCommit, "5df63a7203cdb9328d824a27b682e77c7b9928be"},
		{"mock_version", p.MockVersion, mockVersion(t)},
	} {
		if c.got != c.want {
			t.Fatalf("golden trace for %s records %s %q, want %q: regenerate it with oracle/run_scheduler.py (docs/testing.md)",
				filepath.Base(scenarioPath), c.field, c.got, c.want)
		}
	}
}

// mockVersion is oracle/mocks.json's version, the GPU mock both sides follow.
func mockVersion(t *testing.T) string {
	t.Helper()
	var m struct {
		Version string `json:"version"`
	}
	readJSON(t, "../../oracle/mocks.json", &m)
	return m.Version
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

// TestFuzzCorpusAgainstVLLM is the seeded fuzz test (#2021 B8). BLIS_VLLM_FUZZ_DIR names a
// directory of scenarios from oracle/gen_fuzz.py with the trace vLLM v0.32.0's Scheduler
// produced for each (<name>.trace.json, oracle/run_scheduler.py in directory mode); CI
// generates both from fixed seeds. Every scenario must match at zero mismatches; a failure
// names the seed and the earliest divergence, and the seed is then shrunk and kept under
// oracle/scenarios as a permanent test.
func TestFuzzCorpusAgainstVLLM(t *testing.T) {
	dir := os.Getenv("BLIS_VLLM_FUZZ_DIR")
	if dir == "" {
		t.Skip("BLIS_VLLM_FUZZ_DIR is unset: the fuzz corpus was NOT checked (CI's oracle job sets it)")
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
			t.Errorf("%s: earliest divergence from vLLM v0.32.0: %s", name, d)
		}
	}
	t.Logf("fuzz corpus: %d scenarios, %d mismatches", len(traces), mismatches)
}

// TestScenarioFromEnv checks one scenario against one oracle trace, named by
// BLIS_VLLM_SCENARIO and BLIS_VLLM_TRACE. It is the Go half of oracle/shrink.py, which
// shrinks a failing fuzz seed to the smallest scenario that still diverges.
func TestScenarioFromEnv(t *testing.T) {
	sp, tp := os.Getenv("BLIS_VLLM_SCENARIO"), os.Getenv("BLIS_VLLM_TRACE")
	if sp == "" || tp == "" {
		t.Skip("BLIS_VLLM_SCENARIO and BLIS_VLLM_TRACE are unset")
	}
	var sc scenario
	readJSON(t, sp, &sc)
	var want trace
	readJSON(t, tp, &want)
	checkProvenance(t, sp, want)
	if d := earliestDivergence(want, run(t, sc)); d != "" {
		t.Fatalf("earliest divergence from vLLM v0.32.0: %s", d)
	}
}

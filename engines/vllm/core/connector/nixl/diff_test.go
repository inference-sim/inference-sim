package nixl

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

// scenario is a connector scenario (oracle/README.md, "Connector scenarios").
type scenario struct {
	Config struct {
		MaxNumSeqs                int     `json:"max_num_seqs"`
		MaxNumBatchedTokens       int     `json:"max_num_batched_tokens"`
		BlockSize                 int     `json:"block_size"`
		NumBlocks                 int     `json:"num_blocks"`
		MaxModelLen               int     `json:"max_model_len"`
		LongPrefillTokenThreshold int     `json:"long_prefill_token_threshold"`
		Watermark                 float64 `json:"watermark"`
		EnablePrefixCaching       *bool   `json:"enable_prefix_caching"`
		EnableChunkedPrefill      *bool   `json:"enable_chunked_prefill"`
		SchedulingPolicy          string  `json:"scheduling_policy"`
		Connector                 struct {
			Name                string         `json:"name"`
			KVRole              string         `json:"kv_role"`
			KVLoadFailurePolicy string         `json:"kv_load_failure_policy"`
			ExtraConfig         map[string]any `json:"extra_config"`
		} `json:"connector"`
	} `json:"config"`
	Requests []struct {
		ID            string  `json:"id"`
		ArrivalStep   int     `json:"arrival_step"`
		Prompt        []int64 `json:"prompt"`
		Output        []int64 `json:"output"`
		MaxTokens     int     `json:"max_tokens"`
		Priority      int     `json:"priority"`
		RemotePrefill bool    `json:"remote_prefill"`
		RemoteDecode  bool    `json:"remote_decode"`
		RecvDelay     int     `json:"recv_delay"`
		RecvFailed    bool    `json:"recv_failed"`
		InvalidBlocks []int   `json:"invalid_blocks"`
		SendDelay     int     `json:"send_delay"`
		// AbortStep, when set, is the step the client aborts the request at.
		AbortStep *int `json:"abort_step"`
	} `json:"requests"`
	MaxSteps int `json:"max_steps"`
}

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
	// The connector-visible state.
	WaitingForRemoteKV []string `json:"waiting_for_remote_kv"`
	RecvLanded         []string `json:"recv_landed"`
	DelayFreed         []string `json:"delay_freed"`
	KVHoldingWaiting   []string `json:"kv_holding_waiting"`
	KVFetchStages      []int    `json:"kv_fetch_stages"`
	NumRequests        int      `json:"num_requests"`
	Aborted            []string `json:"aborted"`
	Transport          struct {
		Recving       []string `json:"recving"`
		Sending       []string `json:"sending"`
		Failed        []string `json:"failed"`
		InvalidBlocks int      `json:"invalid_blocks"`
	} `json:"transport"`
}

type finishRecord struct {
	Reason       string `json:"reason"`
	OutputTokens int    `json:"output_tokens"`
	Step         int    `json:"step"`
}

type trace struct {
	Provenance struct {
		Tier           int    `json:"tier"`
		VLLMVersion    string `json:"vllm_version"`
		VLLMCommit     string `json:"vllm_commit"`
		ScenarioSHA256 string `json:"scenario_sha256"`
		MockVersion    string `json:"mock_version"`
	} `json:"provenance"`
	Steps    []step                  `json:"steps"`
	Finished map[string]finishRecord `json:"finished"`
}

// run drives a connector scenario through core/sched with this package's NIXL scheduler,
// exactly as oracle/run_connector.py drives vLLM's: the same GPU mock and the same transport
// mock, step by step.
func run(t *testing.T, sc scenario) trace {
	t.Helper()
	c := sc.Config
	if c.Connector.Name != "NixlConnector" {
		t.Fatalf("connector %q: this test drives NixlConnector", c.Connector.Name)
	}
	caching, chunked := true, true
	if c.EnablePrefixCaching != nil {
		caching = *c.EnablePrefixCaching
	}
	if c.EnableChunkedPrefill != nil {
		chunked = *c.EnableChunkedPrefill
	}
	s, err := sched.New(sched.Config{
		MaxNumSeqs: c.MaxNumSeqs, MaxNumBatchedTokens: c.MaxNumBatchedTokens, BlockSize: c.BlockSize,
		NumBlocks: c.NumBlocks, MaxModelLen: c.MaxModelLen, LongPrefillTokenThreshold: c.LongPrefillTokenThreshold,
		EnableChunkedPrefill: chunked, Policy: sched.Policy(c.SchedulingPolicy), EnablePrefixCaching: caching,
		ReserveFullSequence: true, Watermark: c.Watermark,
	})
	if err != nil {
		t.Fatal(err)
	}
	ncfg := Default()
	if v, ok := c.Connector.ExtraConfig["kv_recompute_threshold"].(float64); ok {
		ncfg.RecomputeThreshold = int(v)
	}
	if v, ok := c.Connector.ExtraConfig["bidirectional_kv_xfer"].(bool); ok {
		ncfg.Bidirectional = v
	}
	conn := New(ncfg)
	if err := s.SetConnector(conn, sched.ConnectorConfig{RecomputeLoadFailures: c.Connector.KVLoadFailurePolicy == "recompute"}); err != nil {
		t.Fatal(err)
	}

	reqs := sc.Requests
	sort.SliceStable(reqs, func(i, j int) bool {
		if reqs[i].ArrivalStep != reqs[j].ArrivalStep {
			return reqs[i].ArrivalStep < reqs[j].ArrivalStep
		}
		return reqs[i].ID < reqs[j].ID
	})
	byID := map[string]int{}
	for i, r := range reqs {
		byID[r.ID] = i
	}
	sampled := map[string]int{}
	finishedAt := map[string]int{}
	loadStarted := map[string]int{}
	tr := trace{Finished: map[string]finishRecord{}}
	maxSteps := sc.MaxSteps
	if maxSteps == 0 {
		maxSteps = 10000
	}
	next := 0
	for i := 0; i < maxSteps; i++ {
		for next < len(reqs) && reqs[next].ArrivalStep <= i {
			r := reqs[next]
			mt := r.MaxTokens
			if mt == 0 {
				mt = 1 << 20
			}
			req := &sched.Request{ID: r.ID, Priority: r.Priority, ArrivalTime: float64(r.ArrivalStep),
				Prompt: r.Prompt, MaxTokens: mt, EOS: eosToken}
			switch {
			case r.RemotePrefill:
				req.Transfer = &sched.TransferParams{DoRemotePrefill: true, HasRemote: true,
					RemoteBlocks: (len(r.Prompt) + c.BlockSize - 1) / c.BlockSize}
			case r.RemoteDecode:
				req.Transfer = &sched.TransferParams{DoRemoteDecode: true}
			}
			if err := s.Add(req); err != nil {
				t.Fatal(err)
			}
			next++
		}
		if !s.HasRequests() && next == len(reqs) {
			break
		}
		out := s.Schedule()
		conn.BuildMeta()
		for _, id := range s.WaitingForRemoteKV() {
			if _, ok := loadStarted[id]; !ok {
				loadStarted[id] = i
			}
		}
		// The GPU mock.
		mock := map[string][]int64{}
		for id := range out.Scheduled {
			r := s.Request(id)
			if r.IsPrefillChunk {
				continue
			}
			o := reqs[byID[id]].Output
			k := sampled[id]
			if k == len(o)-1 {
				mock[id] = []int64{eosToken}
			} else {
				mock[id] = []int64{o[k]}
			}
			sampled[id] = k + 1
		}
		// The transport mock: what the worker reports this step.
		var o sched.TransferOutcome
		var st step
		invalid := 0
		for _, r := range reqs {
			req := s.Request(r.ID)
			if req == nil {
				continue
			}
			// A pull lands at its step even if the request was aborted since (run_connector.py).
			if r.RemotePrefill && (req.Status == sched.WaitingForRemoteKVs || req.Status >= sched.FinishedStopped) {
				if started, ok := loadStarted[r.ID]; ok && i == started+r.RecvDelay {
					o.FinishedRecving = append(o.FinishedRecving, r.ID)
					if r.RecvFailed {
						o.FailedRecving = append(o.FailedRecving, r.ID)
					}
					var pulled []int
					for _, b := range s.KV.Blocks(r.ID) {
						if b.Hash == nil && !b.Null {
							pulled = append(pulled, b.ID)
						}
					}
					for _, idx := range r.InvalidBlocks {
						if idx < len(pulled) {
							o.InvalidBlocks = append(o.InvalidBlocks, pulled[idx])
							invalid++
						}
					}
				}
			}
			if r.RemoteDecode && req.Status >= sched.FinishedStopped {
				if at, ok := finishedAt[r.ID]; ok && i == at+max(r.SendDelay, 1) {
					o.FinishedSending = append(o.FinishedSending, r.ID)
				}
			}
		}
		st.Transport.Recving = sortedStrings(o.FinishedRecving)
		st.Transport.Sending = sortedStrings(o.FinishedSending)
		st.Transport.Failed = sortedStrings(o.FailedRecving)
		st.Transport.InvalidBlocks = uniqueInts(o.InvalidBlocks)
		var aborts []string
		for _, r := range reqs {
			if r.AbortStep != nil && *r.AbortStep == i {
				aborts = append(aborts, r.ID)
			}
		}
		sort.Strings(aborts)
		if len(aborts) > 0 {
			// EngineCore applies aborts before update_from_output (vllm/v1/engine/core.py:678-683).
			s.Abort(aborts)
		}
		st.Aborted = aborts
		if st.Aborted == nil {
			st.Aborted = []string{}
		}
		var done []string
		for _, f := range s.UpdateWith(out, mock, o) {
			done = append(done, f.ID)
			if _, ok := finishedAt[f.ID]; !ok {
				finishedAt[f.ID] = i
			}
			reason := map[sched.Status]string{sched.FinishedStopped: "stop", sched.FinishedLength: "length",
				sched.FinishedError: "error"}[f.Status]
			tr.Finished[f.ID] = finishRecord{reason, sampled[f.ID], i}
		}
		nr, nw := s.Counts()
		st.Step, st.Scheduled, st.Preempted, st.Finished = i, out.Scheduled, sortedStrings(out.Preempted), sortedStrings(done)
		st.FreeBlocks, st.Running, st.NumRunning, st.NumWaiting = s.KV.Pool.NumFree(), sortedStrings(s.Running()), nr, nw
		st.KVUsage, st.Blocks = s.KV.Pool.Usage(), map[string][][]any{}
		for _, id := range st.Running {
			for _, b := range s.KV.Blocks(id) {
				var h any
				if b.Hash != nil {
					h = hex.EncodeToString(b.Hash)
				}
				st.Blocks[id] = append(st.Blocks[id], []any{h, float64(b.RefCnt)})
			}
		}
		st.WaitingForRemoteKV = sortedStrings(s.WaitingForRemoteKV())
		st.RecvLanded = sortedStrings(s.RecvLanded())
		st.DelayFreed = sortedStrings(s.DelayFreed())
		st.KVHoldingWaiting = sortedStrings(s.Holding())
		fc := s.FetchCounts()
		st.KVFetchStages = fc[:]
		st.NumRequests = s.NumRequests()
		tr.Steps = append(tr.Steps, st)
	}
	return tr
}

func uniqueInts(in []int) int {
	seen := map[int]bool{}
	for _, v := range in {
		seen[v] = true
	}
	return len(seen)
}

func sortedStrings(in []string) []string {
	out := append([]string{}, in...)
	sort.Strings(out)
	return out
}

// earliestDivergence names the first step and field that differ (#2021 B8).
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
			{"transport", w.Transport, g.Transport},
			{"aborted", sortedStrings(w.Aborted), sortedStrings(g.Aborted)},
			{"finished", sortedStrings(w.Finished), sortedStrings(g.Finished)},
			{"running", sortedStrings(w.Running), sortedStrings(g.Running)},
			{"num_running", w.NumRunning, g.NumRunning},
			{"num_waiting", w.NumWaiting, g.NumWaiting},
			{"waiting_for_remote_kv", sortedStrings(w.WaitingForRemoteKV), sortedStrings(g.WaitingForRemoteKV)},
			{"recv_landed", sortedStrings(w.RecvLanded), sortedStrings(g.RecvLanded)},
			{"delay_freed", sortedStrings(w.DelayFreed), sortedStrings(g.DelayFreed)},
			{"kv_holding_waiting", sortedStrings(w.KVHoldingWaiting), sortedStrings(g.KVHoldingWaiting)},
			{"kv_fetch_stages", w.KVFetchStages, g.KVFetchStages},
			{"num_requests", w.NumRequests, g.NumRequests},
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

// checkProvenance refuses a trace not produced from this exact scenario by the pinned vLLM
// release under the current mocks, at tier 4.
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
	readJSON(t, "../../../oracle/mocks.json", &mocks)
	p := tr.Provenance
	for _, c := range []struct{ field, got, want string }{
		{"scenario_sha256", p.ScenarioSHA256, hex.EncodeToString(sum[:])},
		{"vllm_version", strings.SplitN(p.VLLMVersion, "+", 2)[0], "0.32.0"},
		{"vllm_commit", p.VLLMCommit, "5df63a7203cdb9328d824a27b682e77c7b9928be"},
		{"mock_version", p.MockVersion, mocks.Version},
	} {
		if c.got != c.want {
			t.Fatalf("trace for %s records %s %q, want %q: regenerate it with oracle/run_connector.py",
				filepath.Base(scenarioPath), c.field, c.got, c.want)
		}
	}
	if p.Tier != 4 {
		t.Fatalf("trace for %s records tier %d, want 4", filepath.Base(scenarioPath), p.Tier)
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

// TestNIXLAgainstVLLM runs every NIXL scenario under oracle/scenarios/connector through
// core/sched with this package's scheduler side, and compares each step with the trace vLLM
// v0.32.0's own Scheduler and NixlPullConnectorScheduler produced (testdata/golden). The
// target is zero mismatches.
func TestNIXLAgainstVLLM(t *testing.T) {
	scenarios, err := filepath.Glob("../../../oracle/scenarios/connector/nixl-*.json")
	if err != nil || len(scenarios) == 0 {
		t.Fatalf("no NIXL scenarios (%v)", err)
	}
	for _, path := range scenarios {
		name := strings.TrimSuffix(filepath.Base(path), ".json")
		t.Run(name, func(t *testing.T) {
			var sc scenario
			readJSON(t, path, &sc)
			var want trace
			readJSON(t, filepath.Join("testdata", "golden", name+".json"), &want)
			checkProvenance(t, path, want)
			if d := earliestDivergence(want, run(t, sc)); d != "" {
				t.Fatalf("earliest divergence from vLLM v0.32.0: %s", d)
			}
		})
	}
}

// TestNIXLFuzzCorpusAgainstVLLM is the seeded fuzz test for the NIXL connector:
// BLIS_VLLM_CONNECTOR_FUZZ_DIR names a directory of oracle/gen_connector_fuzz.py scenarios
// with vLLM's trace for each (<name>.trace.json).
func TestNIXLFuzzCorpusAgainstVLLM(t *testing.T) {
	dir := os.Getenv("BLIS_VLLM_CONNECTOR_FUZZ_DIR")
	if dir == "" {
		t.Skip("BLIS_VLLM_CONNECTOR_FUZZ_DIR is unset: the connector fuzz corpus was NOT checked (CI's oracle job sets it)")
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
	t.Logf("connector fuzz corpus: %d scenarios, %d mismatches", len(traces), mismatches)
}

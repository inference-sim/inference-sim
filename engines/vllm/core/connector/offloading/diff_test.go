package offloading

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

	"github.com/inference-sim/blis-vllm/core/kv"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// eosToken is tests/v1/core/utils.py:43@5df63a72, which the GPU mock samples last.
const eosToken = 50256

// pageSize is the page size the CPU chunk size is rounded up to: the scenarios state a KV page
// that is a multiple of 16 KiB (oracle/run_offloading.py kv_cache_config), so the chunk size and
// num_chunks are the same on a 4 KiB-page and a 16 KiB-page host.
const pageSize = 16384

// scenario is an offloading scenario (oracle/run_offloading.py).
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
		KVPage                    struct {
			NumKVHeads int `json:"num_kv_heads"`
			HeadSize   int `json:"head_size"`
		} `json:"kv_page"`
		KVGroups []struct {
			Kind          string `json:"kind"`
			SlidingWindow int    `json:"sliding_window"`
		} `json:"kv_groups"`
		Transfer struct {
			LoadDelay  *int `json:"load_delay"`
			StoreDelay *int `json:"store_delay"`
		} `json:"transfer"`
		Connector struct {
			Name        string         `json:"name"`
			ExtraConfig map[string]any `json:"extra_config"`
		} `json:"connector"`
	} `json:"config"`
	Requests []struct {
		ID          string  `json:"id"`
		ArrivalStep int     `json:"arrival_step"`
		Prompt      []int64 `json:"prompt"`
		Output      []int64 `json:"output"`
		MaxTokens   int     `json:"max_tokens"`
		Priority    int     `json:"priority"`
		AbortStep   *int    `json:"abort_step"`
		LoadDelay   *int    `json:"load_delay"`
		StoreDelay  *int    `json:"store_delay"`
	} `json:"requests"`
	MaxSteps int `json:"max_steps"`
}

type cpuState struct {
	Chunks    [][]any `json:"chunks"`
	Free      int     `json:"free"`
	Evictable int     `json:"evictable"`
}

type offState struct {
	Loads       [][]any  `json:"loads"`
	Stores      [][]any  `json:"stores"`
	Flush       []int    `json:"flush"`
	Completed   []int    `json:"completed"`
	Jobs        []int    `json:"jobs"`
	PendingPush bool     `json:"pending_push"`
	CPU         cpuState `json:"cpu"`
}

type step struct {
	Step               int            `json:"step"`
	Idle               bool           `json:"idle,omitempty"`
	Scheduled          map[string]int `json:"scheduled"`
	Preempted          []string       `json:"preempted"`
	Finished           []string       `json:"finished"`
	Aborted            []string       `json:"aborted"`
	FreeBlocks         int            `json:"free_blocks"`
	Running            []string       `json:"running"`
	NumRunning         int            `json:"num_running"`
	NumWaiting         int            `json:"num_waiting"`
	KVUsage            float64        `json:"kv_usage"`
	Blocks             map[string]any `json:"blocks"`
	WaitingForRemoteKV []string       `json:"waiting_for_remote_kv"`
	RecvLanded         []string       `json:"recv_landed"`
	DelayFreed         []string       `json:"delay_freed"`
	KVHoldingWaiting   []string       `json:"kv_holding_waiting"`
	KVFetchStages      []int          `json:"kv_fetch_stages"`
	NumRequests        int            `json:"num_requests"`
	Offload            *offState      `json:"offload"`
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
		CPUChunks      int    `json:"cpu_chunks"`
	} `json:"provenance"`
	Steps    []step                  `json:"steps"`
	Finished map[string]finishRecord `json:"finished"`
}

func intOr(v *int, def int) int {
	if v != nil {
		return *v
	}
	return def
}

// configure is the scenario's scheduler and connector, as oracle/run_offloading.py builds vLLM's.
func configure(t *testing.T, sc scenario) (*sched.Scheduler, *Scheduler, int) {
	t.Helper()
	c := sc.Config
	caching, chunked := true, true
	if c.EnablePrefixCaching != nil {
		caching = *c.EnablePrefixCaching
	}
	if c.EnableChunkedPrefill != nil {
		chunked = *c.EnableChunkedPrefill
	}
	var layers []kv.Group
	var groups []Group
	for _, g := range c.KVGroups {
		switch g.Kind {
		case "full":
			layers = append(layers, kv.Group{Kind: kv.FullAttention})
			groups = append(groups, Group{BlockSize: c.BlockSize})
		case "sliding_window":
			layers = append(layers, kv.Group{Kind: kv.SlidingWindow, Window: g.SlidingWindow})
			groups = append(groups, Group{BlockSize: c.BlockSize, Window: g.SlidingWindow})
		default:
			t.Fatalf("unknown KV cache group kind %q", g.Kind)
		}
	}
	if len(groups) == 0 {
		groups = []Group{{BlockSize: c.BlockSize}}
	}
	s, err := sched.New(sched.Config{
		MaxNumSeqs: c.MaxNumSeqs, MaxNumBatchedTokens: c.MaxNumBatchedTokens, BlockSize: c.BlockSize,
		NumBlocks: c.NumBlocks, MaxModelLen: c.MaxModelLen, LongPrefillTokenThreshold: c.LongPrefillTokenThreshold,
		EnableChunkedPrefill: chunked, Policy: sched.Policy(c.SchedulingPolicy), EnablePrefixCaching: caching,
		ReserveFullSequence: true, Watermark: c.Watermark, KVGroups: kv.ResolveGroups(layers, false),
	})
	if err != nil {
		t.Fatal(err)
	}
	extra := c.Connector.ExtraConfig
	bpc := 1
	if v, ok := extra["blocks_per_chunk"].(float64); ok {
		bpc = int(v)
	}
	promptOnly := true
	if v, ok := extra["offload_prompt_only"].(bool); ok {
		promptOnly = v
	}
	cpuBytes, _ := extra["cpu_bytes_to_use"].(float64)
	heads, head := max(c.KVPage.NumKVHeads, 1), max(c.KVPage.HeadSize, 1)
	// A layer's page: K and V, float32, block_size tokens (AttentionSpec.page_size_bytes); one
	// layer per group, so the worker's bytes per block are the groups' pages
	// (_group_kv_bytes_per_block, offloading/config.py:46-58; config.py:221-224).
	perBlock := int64(2*c.BlockSize*heads*head*4) * int64(len(groups))
	chunks := NumChunks(int64(cpuBytes), perBlock, 1, bpc, pageSize)
	conn, err := New(Config{Groups: groups, BlocksPerChunk: bpc, TokensPerHash: s.KV.HashBlockSize(), NumChunks: chunks,
		OffloadPromptOnly: promptOnly, PrefixCaching: caching})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetConnector(conn, sched.ConnectorConfig{Consumer: true, Producer: true}); err != nil {
		t.Fatal(err)
	}
	return s, conn, chunks
}

// run drives a scenario through core/sched with this package's connector, exactly as
// oracle/run_offloading.py drives vLLM's: the same GPU mock and the same worker mock.
func run(t *testing.T, sc scenario) trace {
	t.Helper()
	c := sc.Config
	s, conn, chunks := configure(t, sc)
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
	delay := func(id string, store bool) int {
		r := reqs[byID[id]]
		if store {
			return intOr(r.StoreDelay, intOr(c.Transfer.StoreDelay, 1))
		}
		return intOr(r.LoadDelay, intOr(c.Transfer.LoadDelay, 1))
	}
	var unsubmitted []int
	completesAt := map[int]int{}
	jobReq := map[int]string{}
	isLoad := map[int]bool{}
	sampled := map[string]int{}
	tr := trace{Finished: map[string]finishRecord{}}
	tr.Provenance.CPUChunks = chunks
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
			if err := s.Add(&sched.Request{ID: r.ID, Priority: r.Priority, ArrivalTime: float64(r.ArrivalStep),
				Prompt: r.Prompt, MaxTokens: mt, EOS: eosToken}); err != nil {
				t.Fatal(err)
			}
			next++
		}
		if !s.HasRequests() {
			if next == len(reqs) {
				break
			}
			tr.Steps = append(tr.Steps, step{Step: i, Idle: true})
			continue
		}
		out := s.Schedule()
		meta := out.KVConnectorMetadata.(Meta)
		off := &offState{Flush: append([]int{}, meta.Flush...)}
		for _, j := range meta.Loads {
			off.Loads = append(off.Loads, jobRow(j))
			jobReq[j.ID], isLoad[j.ID] = j.Req, true
		}
		for _, j := range meta.Stores {
			off.Stores = append(off.Stores, jobRow(j))
			jobReq[j.ID] = j.Req
		}
		// The worker mock (oracle/run_offloading.py).
		flush := map[int]bool{}
		for _, j := range meta.Flush {
			flush[j] = true
		}
		var now []int
		for _, j := range meta.Stores {
			if flush[j.ID] {
				unsubmitted = append(unsubmitted, j.ID)
			} else {
				now = append(now, j.ID)
			}
		}
		for _, j := range unsubmitted {
			completesAt[j] = i + delay(jobReq[j], true)
		}
		unsubmitted = nil
		for j := range flush {
			if _, ok := completesAt[j]; ok {
				completesAt[j] = i
			}
		}
		for _, j := range meta.Loads {
			completesAt[j.ID] = i + delay(j.Req, false)
		}
		unsubmitted = append(unsubmitted, now...)
		// The GPU mock.
		mock := map[string][]int64{}
		for id := range out.Scheduled {
			r := s.Request(id)
			if r.IsPrefillChunk {
				continue
			}
			o := reqs[byID[id]].Output
			if k := sampled[id]; k < len(o) {
				if k == len(o)-1 {
					mock[id] = []int64{eosToken}
				} else {
					mock[id] = []int64{o[k]}
				}
				sampled[id] = k + 1
			}
		}
		var done []int
		for j, at := range completesAt {
			if at <= i {
				done = append(done, j)
			}
		}
		sort.Ints(done)
		recv := map[string]bool{}
		for _, j := range done {
			delete(completesAt, j)
			if isLoad[j] {
				recv[jobReq[j]] = true
			}
		}
		var o sched.TransferOutcome
		for id := range recv {
			o.FinishedRecving = append(o.FinishedRecving, id)
		}
		sort.Strings(o.FinishedRecving)
		if len(done) > 0 {
			o.WorkerMeta = done
		}
		off.Completed = append([]int{}, done...)
		var aborts []string
		for _, r := range reqs {
			if r.AbortStep != nil && *r.AbortStep == i {
				aborts = append(aborts, r.ID)
			}
		}
		sort.Strings(aborts)
		if len(aborts) > 0 {
			s.Abort(aborts)
		}
		st := step{Step: i, Scheduled: out.Scheduled, Preempted: sortedStrings(out.Preempted), Aborted: append([]string{}, aborts...)}
		var fin []string
		for _, f := range s.UpdateWith(out, mock, o) {
			fin = append(fin, f.ID)
			reason := map[sched.Status]string{sched.FinishedStopped: "stop", sched.FinishedLength: "length",
				sched.FinishedError: "error"}[f.Status]
			tr.Finished[f.ID] = finishRecord{reason, sampled[f.ID], i}
		}
		st.Finished = sortedStrings(fin)
		nr, nw := s.Counts()
		st.FreeBlocks, st.Running, st.NumRunning, st.NumWaiting = s.KV.Pool.NumFree(), sortedStrings(s.Running()), nr, nw
		st.KVUsage, st.Blocks = s.KV.Pool.Usage(), map[string]any{}
		for _, id := range st.Running {
			if s.KV.NumGroups() == 1 {
				st.Blocks[id] = blockRows(s.KV.Blocks(id), false)
				continue
			}
			rows := []any{}
			for g := 0; g < s.KV.NumGroups(); g++ {
				rows = append(rows, blockRows(s.KV.GroupBlocks(g, id), true))
			}
			st.Blocks[id] = rows
		}
		st.WaitingForRemoteKV = sortedStrings(s.WaitingForRemoteKV())
		st.RecvLanded = sortedStrings(s.RecvLanded())
		st.DelayFreed = sortedStrings(s.DelayFreed())
		st.KVHoldingWaiting = sortedStrings(s.Holding())
		fc := s.FetchCounts()
		st.KVFetchStages = fc[:]
		st.NumRequests = s.NumRequests()
		off.Jobs = conn.Jobs()
		off.PendingPush = conn.HasPendingPushWork()
		cs, free, ev := conn.CPU()
		off.CPU = cpuState{Chunks: [][]any{}, Free: free, Evictable: ev}
		for _, ch := range cs {
			off.CPU.Chunks = append(off.CPU.Chunks, []any{fmt.Sprintf("%x/%d", ch.Key.HashPrefix(), ch.Key.Group()),
				float64(ch.ID), float64(ch.RefCnt)})
		}
		st.Offload = off
		tr.Steps = append(tr.Steps, st)
	}
	return tr
}

func jobRow(j Job) []any {
	return []any{float64(j.ID), j.Req, floats(j.CPU), []any{floats(j.GPU), floats(j.GroupSizes), floats(j.BlockIndices)}}
}

func floats(in []int) []any {
	out := make([]any, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}

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

func sortedStrings(in []string) []string {
	out := append([]string{}, in...)
	sort.Strings(out)
	return out
}

// normalize makes a step comparable: nil and empty lists alike.
func normalize(v any) any {
	raw, _ := json.Marshal(v)
	var out any
	_ = json.Unmarshal(raw, &out)
	return scrub(out)
}

func scrub(v any) any {
	switch x := v.(type) {
	case nil:
		return []any{}
	case []any:
		for i := range x {
			x[i] = scrub(x[i])
		}
		return x
	case map[string]any:
		for k := range x {
			x[k] = scrub(x[k])
		}
		return x
	}
	return v
}

// earliestDivergence names the first step and field that differ.
func earliestDivergence(want, got trace) string {
	if want.Provenance.CPUChunks != got.Provenance.CPUChunks+0 {
		return fmt.Sprintf("CPU tier: oracle %d chunks, Go %d", want.Provenance.CPUChunks, got.Provenance.CPUChunks)
	}
	for i := 0; i < max(len(want.Steps), len(got.Steps)); i++ {
		if i >= len(want.Steps) {
			return fmt.Sprintf("step %d: Go ran a step the oracle did not (scheduled %v)", i, got.Steps[i].Scheduled)
		}
		if i >= len(got.Steps) {
			return fmt.Sprintf("step %d: the oracle ran a step Go did not (scheduled %v)", i, want.Steps[i].Scheduled)
		}
		if want.Steps[i].Idle || got.Steps[i].Idle {
			if want.Steps[i].Idle != got.Steps[i].Idle {
				return fmt.Sprintf("step %d: idle oracle %v, Go %v", i, want.Steps[i].Idle, got.Steps[i].Idle)
			}
			continue
		}
		w, g := normalize(want.Steps[i]).(map[string]any), normalize(got.Steps[i]).(map[string]any)
		keys := make([]string, 0, len(w))
		for k := range w {
			if k != "idle" {
				keys = append(keys, k)
			}
		}
		sort.Strings(keys)
		// Offloading decisions first: they explain the rest.
		sort.SliceStable(keys, func(a, b int) bool { return keys[a] == "offload" && keys[b] != "offload" })
		for _, k := range keys {
			if k == "offload" {
				wo, go_ := w[k].(map[string]any), g[k].(map[string]any)
				for _, f := range []string{"loads", "stores", "flush", "completed", "jobs", "pending_push", "cpu"} {
					if !reflect.DeepEqual(wo[f], go_[f]) {
						return fmt.Sprintf("step %d, offload.%s: oracle %v, Go %v", i, f, wo[f], go_[f])
					}
				}
				continue
			}
			if k == "kv_usage" {
				if fmt.Sprintf("%.12f", w[k]) != fmt.Sprintf("%.12f", g[k]) {
					return fmt.Sprintf("step %d, kv_usage: oracle %v, Go %v", i, w[k], g[k])
				}
				continue
			}
			if !reflect.DeepEqual(w[k], g[k]) {
				return fmt.Sprintf("step %d, field %s: oracle %v, Go %v", i, k, w[k], g[k])
			}
		}
	}
	if !reflect.DeepEqual(want.Finished, got.Finished) {
		return fmt.Sprintf("finished: oracle %v, Go %v", want.Finished, got.Finished)
	}
	return ""
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
			t.Fatalf("trace for %s records %s %q, want %q: regenerate it with oracle/run_offloading.py",
				filepath.Base(scenarioPath), c.field, c.got, c.want)
		}
	}
	if p.Tier != 4 {
		t.Fatalf("trace for %s records tier %d, want 4", filepath.Base(scenarioPath), p.Tier)
	}
}

// TestOffloadingAgainstVLLM runs every OffloadingConnector scenario under
// oracle/scenarios/offloading through core/sched with this package's scheduler side, and compares
// each step with the trace vLLM v0.32.0's own Scheduler and OffloadingConnectorScheduler produced
// (testdata/golden). The target is zero mismatches.
func TestOffloadingAgainstVLLM(t *testing.T) {
	scenarios, err := filepath.Glob("../../../oracle/scenarios/offloading/offloading-*.json")
	if err != nil || len(scenarios) == 0 {
		t.Fatalf("no offloading scenarios (%v)", err)
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

// TestOffloadingFuzzCorpusAgainstVLLM is the seeded fuzz test: BLIS_VLLM_OFFLOAD_FUZZ_DIR names a
// directory of oracle/gen_offloading_fuzz.py scenarios with vLLM's trace for each
// (<name>.trace.json).
func TestOffloadingFuzzCorpusAgainstVLLM(t *testing.T) {
	dir := os.Getenv("BLIS_VLLM_OFFLOAD_FUZZ_DIR")
	if dir == "" {
		t.Skip("BLIS_VLLM_OFFLOAD_FUZZ_DIR is unset: the offloading fuzz corpus was NOT checked (CI's oracle job sets it)")
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
	t.Logf("offloading fuzz corpus: %d scenarios, %d mismatches", len(traces), mismatches)
	if mismatches > 0 {
		t.Fatalf("%d of %d fuzz scenarios differ from vLLM", mismatches, len(traces))
	}
}

// Check refuses each configuration the package does not model, and accepts the CPU tier through
// either spec.
func TestCheckRefusesWhatIsNotModelled(t *testing.T) {
	zero := 0
	for _, o := range []Options{{}, {SpecName: "CPUOffloadingSpec"}, {SpecName: "TieringOffloadingSpec", EvictionPolicy: "lru", WorldSize: 1}} {
		if err := Check(o); err != nil {
			t.Errorf("Check(%+v) = %v, want nil", o, err)
		}
	}
	for _, c := range []struct {
		o    Options
		want string
	}{
		{Options{SpecName: "Other"}, "spec_name"},
		{Options{SpecName: "TieringOffloadingSpec", SecondaryTiers: 1}, "issues/13"},
		{Options{EvictionPolicy: "arc"}, "issues/1)"},
		{Options{StoreThreshold: 2}, "store_threshold"},
		{Options{WorldSize: 2}, "workers"},
		{Options{RetentionInterval: &zero}, "issues/7"},
	} {
		if err := Check(c.o); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("Check(%+v) = %v, want an error naming %q", c.o, err, c.want)
		}
	}
	if _, err := New(Config{Groups: []Group{{BlockSize: 4}, {BlockSize: 8, Window: 8}}, TokensPerHash: 4}); err == nil {
		t.Error("New accepted groups of two block sizes")
	}
}

// The scheduler refuses the connector beside the batch queue and a drafter, which are not
// diff-tested with it yet.
func TestSetConnectorRefusesBatchQueueAndDrafter(t *testing.T) {
	for _, cfg := range []sched.Config{
		{MaxNumSeqs: 2, MaxNumBatchedTokens: 32, BlockSize: 4, NumBlocks: 20, MaxModelLen: 64, MaxConcurrentBatches: 2},
		{MaxNumSeqs: 2, MaxNumBatchedTokens: 32, BlockSize: 4, NumBlocks: 20, MaxModelLen: 64, Async: true},
		{MaxNumSeqs: 2, MaxNumBatchedTokens: 32, BlockSize: 4, NumBlocks: 20, MaxModelLen: 64,
			Spec: sched.ResolveSpec(sched.SpecOptions{Method: "dspark", NumTokens: 7})},
	} {
		s, err := sched.New(cfg)
		if err != nil {
			t.Fatal(err)
		}
		conn, _ := New(Config{Groups: []Group{{BlockSize: 4}}, TokensPerHash: 4, NumChunks: 4})
		if err := s.SetConnector(conn, sched.ConnectorConfig{Consumer: true, Producer: true}); err == nil {
			t.Errorf("SetConnector accepted the OffloadingConnector beside %+v", cfg)
		}
	}
}

package multi

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

	"github.com/inference-sim/blis-vllm/core/connector/nixl"
	"github.com/inference-sim/blis-vllm/core/connector/offloading"
	"github.com/inference-sim/blis-vllm/core/engine"
	"github.com/inference-sim/blis-vllm/core/kv"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// eosToken is tests/v1/core/utils.py:43@5df63a72, which the GPU mock samples last.
const eosToken = 50256

// pageSize is the page size the CPU chunk size is rounded up to (oracle/run_offloading.py).
const pageSize = 16384

// childCfg is one entry of kv_connector_extra_config.connectors. vLLM passes each entry to
// KVTransferConfig(**entry), so its own field names (kv_connector, kv_connector_extra_config) are
// read as well as the scenario files' (name, extra_config).
type childCfg struct {
	Name                   string         `json:"name"`
	KVConnector            string         `json:"kv_connector"`
	KVRole                 string         `json:"kv_role"`
	ExtraConfig            map[string]any `json:"extra_config"`
	KVConnectorExtraConfig map[string]any `json:"kv_connector_extra_config"`
}

func (c childCfg) name() string {
	if c.Name != "" {
		return c.Name
	}
	return c.KVConnector
}

func (c childCfg) extra() map[string]any {
	if c.ExtraConfig != nil {
		return c.ExtraConfig
	}
	return c.KVConnectorExtraConfig
}

// scenarioReq is a request of a MultiConnector scenario: the NIXL scenarios' transfer fields and
// the offloading scenarios' delays.
type scenarioReq struct {
	ID            string  `json:"id"`
	ArrivalStep   int     `json:"arrival_step"`
	Prompt        []int64 `json:"prompt"`
	Output        []int64 `json:"output"`
	MaxTokens     int     `json:"max_tokens"`
	Priority      int     `json:"priority"`
	AbortStep     *int    `json:"abort_step"`
	Accepted      []int   `json:"accepted"`
	RemotePrefill bool    `json:"remote_prefill"`
	RemoteDecode  bool    `json:"remote_decode"`
	RecvDelay     int     `json:"recv_delay"`
	RecvFailed    bool    `json:"recv_failed"`
	InvalidBlocks []int   `json:"invalid_blocks"`
	SendDelay     int     `json:"send_delay"`
	LoadDelay     *int    `json:"load_delay"`
	StoreDelay    *int    `json:"store_delay"`
}

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
		TensorParallelSize        int     `json:"tensor_parallel_size"`
		AsyncScheduling           bool    `json:"async_scheduling"`
		PipelineParallelSize      int     `json:"pipeline_parallel_size"`
		UseV2ModelRunner          bool    `json:"use_v2_model_runner"`
		Speculative               *struct {
			Method    string `json:"method"`
			NumTokens int    `json:"num_speculative_tokens"`
			Draft     string `json:"draft"`
		} `json:"speculative"`
		KVPage struct {
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
			Name                string `json:"name"`
			KVRole              string `json:"kv_role"`
			KVLoadFailurePolicy string `json:"kv_load_failure_policy"`
			ExtraConfig         struct {
				Connectors []childCfg `json:"connectors"`
			} `json:"extra_config"`
		} `json:"connector"`
	} `json:"config"`
	Requests []scenarioReq `json:"requests"`
	MaxSteps int           `json:"max_steps"`
}

type cpuState struct {
	Chunks    [][]any `json:"chunks"`
	Free      int     `json:"free"`
	Evictable int     `json:"evictable"`
}

// offState is the offloading child's step record (oracle/run_offloading.py).
type offState struct {
	Loads       [][]any  `json:"loads"`
	Stores      [][]any  `json:"stores"`
	Flush       []int    `json:"flush"`
	Completed   []int    `json:"completed"`
	Jobs        []int    `json:"jobs"`
	PendingPush bool     `json:"pending_push"`
	CPU         cpuState `json:"cpu"`
}

// step is the offloading step record, the transport the NIXL child sees, and which child was
// chosen to load each request still known to the wrapper.
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
	Transport          struct {
		Recving       []string `json:"recving"`
		Sending       []string `json:"sending"`
		Failed        []string `json:"failed"`
		InvalidBlocks int      `json:"invalid_blocks"`
	} `json:"transport"`
	Chosen        map[string]int `json:"chosen"`
	SpecScheduled map[string]int `json:"spec_scheduled,omitempty"`
	Processed     bool           `json:"processed,omitempty"`
	QueueLen      int            `json:"queue_len,omitempty"`
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

// draftNextnLayers is a draft checkpoint's num_nextn_predict_layers (oracle/drafts), 0 for none.
func draftNextnLayers(t *testing.T, draft string) int {
	t.Helper()
	if draft == "" {
		return 0
	}
	var c struct {
		Layers int `json:"num_nextn_predict_layers"`
	}
	readJSON(t, filepath.Join("..", "..", "..", "oracle", "drafts", draft, "config.json"), &c)
	return c.Layers
}

func intOr(v *int, def int) int {
	if v != nil {
		return *v
	}
	return def
}

// setup is a scenario's scheduler, wrapper and children. off and nx are nil when the scenario
// lists no such child; offIdx is the offloading child's place in the list.
type setup struct {
	s      *sched.Scheduler
	m      *Connector
	off    *offloading.Scheduler
	nx     *nixl.Scheduler
	offIdx int
	nxIdx  int
	chunks int
	queue  int
	spec   sched.Spec
}

func isProducer(role string) bool { return role == "kv_producer" || role == "kv_both" }

// configure builds the scenario's scheduler and wrapper as oracle/run_multiconnector.py builds
// vLLM's: children in the listed order, the wrapper's own role on the scheduler.
func configure(t *testing.T, sc scenario) setup {
	t.Helper()
	c := sc.Config
	caching, chunked := true, true
	if c.EnablePrefixCaching != nil {
		caching = *c.EnablePrefixCaching
	}
	if c.EnableChunkedPrefill != nil {
		chunked = *c.EnableChunkedPrefill
	}
	var spec sched.Spec
	if sp := c.Speculative; sp != nil {
		spec = sched.ResolveSpec(sched.SpecOptions{Method: sp.Method, NumTokens: sp.NumTokens,
			NextnPredictLayers: draftNextnLayers(t, sp.Draft)})
	}
	var layers []kv.Group
	var groups []offloading.Group
	extra := max(0, spec.PrefillLookahead-1)
	for _, g := range c.KVGroups {
		switch g.Kind {
		case "full":
			layers = append(layers, kv.Group{Kind: kv.FullAttention})
			groups = append(groups, offloading.Group{BlockSize: c.BlockSize})
		case "sliding_window":
			layers = append(layers, kv.Group{Kind: kv.SlidingWindow, Window: g.SlidingWindow})
			groups = append(groups, offloading.Group{BlockSize: c.BlockSize, Window: g.SlidingWindow, ExtraRetained: extra})
		default:
			t.Fatalf("unknown KV cache group kind %q", g.Kind)
		}
	}
	if len(groups) == 0 {
		groups = []offloading.Group{{BlockSize: c.BlockSize}}
	}
	pp := max(c.PipelineParallelSize, 1)
	v2 := c.UseV2ModelRunner || (c.Speculative != nil && c.Speculative.Method == "dspark")
	queue := 1
	if c.AsyncScheduling || pp > 1 {
		queue = engine.MaxConcurrentBatches(pp, c.AsyncScheduling, v2)
	}
	s, err := sched.New(sched.Config{
		MaxNumSeqs: c.MaxNumSeqs, MaxNumBatchedTokens: c.MaxNumBatchedTokens, BlockSize: c.BlockSize,
		NumBlocks: c.NumBlocks, MaxModelLen: c.MaxModelLen, LongPrefillTokenThreshold: c.LongPrefillTokenThreshold,
		EnableChunkedPrefill: chunked, Policy: sched.Policy(c.SchedulingPolicy), EnablePrefixCaching: caching,
		ReserveFullSequence: true, Watermark: c.Watermark, Spec: spec, KVGroups: kv.ResolveGroups(layers, false),
		Async: c.AsyncScheduling, DecodeStagger: pp, UseV2ModelRunner: v2, MaxConcurrentBatches: queue,
	})
	if err != nil {
		t.Fatal(err)
	}
	if c.Connector.Name != "MultiConnector" {
		t.Fatalf("connector %q: this test drives MultiConnector", c.Connector.Name)
	}
	top := c.Connector.KVRole
	if top == "" {
		top = "kv_both"
	}
	out := setup{s: s, queue: queue, spec: spec, offIdx: -1, nxIdx: -1}
	var children []Child
	for _, cc := range c.Connector.ExtraConfig.Connectors {
		role := cc.KVRole
		if role == "" {
			role = top
		}
		ex := cc.extra()
		switch cc.name() {
		case "NixlConnector", "NixlPullConnector":
			if err := nixl.CheckWorker(pp, s.KV.NumGroups() > 1, role == "kv_consumer"); err != nil {
				t.Fatal(err)
			}
			ncfg := nixl.Default()
			ncfg.PrefillBackoff = nixl.Backoff(spec, false)
			if v, ok := ex["kv_recompute_threshold"].(float64); ok {
				ncfg.RecomputeThreshold = int(v)
			}
			if v, ok := ex["bidirectional_kv_xfer"].(bool); ok {
				ncfg.Bidirectional = v
			}
			out.nx = nixl.New(ncfg)
			out.nxIdx = len(children)
			children = append(children, Child{Name: cc.name(), Conn: out.nx, Producer: isProducer(role)})
		case "OffloadingConnector":
			bpc := 1
			if v, ok := ex["blocks_per_chunk"].(float64); ok {
				bpc = int(v)
			}
			promptOnly := true
			if v, ok := ex["offload_prompt_only"].(bool); ok {
				promptOnly = v
			}
			cpuBytes, _ := ex["cpu_bytes_to_use"].(float64)
			heads, head := max(c.KVPage.NumKVHeads, 1), max(c.KVPage.HeadSize, 1)
			perBlock := int64(2*c.BlockSize*heads*head*4) * int64(len(groups))
			workers := max(c.TensorParallelSize, 1) * pp
			out.chunks = offloading.NumChunks(int64(cpuBytes), perBlock, workers, bpc, pageSize)
			eagle := make([]bool, s.KV.NumGroups())
			if spec.EagleBlockDrop {
				for g := range eagle {
					eagle[g] = s.KV.GroupSpec(g).Eagle
				}
			}
			out.off, err = offloading.New(offloading.Config{Groups: groups, BlocksPerChunk: bpc,
				TokensPerHash: s.KV.HashBlockSize(), NumChunks: out.chunks, OffloadPromptOnly: promptOnly,
				PrefixCaching: caching, Workers: workers, EagleGroups: eagle})
			if err != nil {
				t.Fatal(err)
			}
			out.offIdx = len(children)
			children = append(children, Child{Name: cc.name(), Conn: out.off, Producer: isProducer(role)})
		default:
			t.Fatalf("child connector %q is not modelled", cc.name())
		}
	}
	out.m, err = New(children)
	if err != nil {
		t.Fatal(err)
	}
	err = s.SetConnector(out.m, sched.ConnectorConfig{RecomputeLoadFailures: c.Connector.KVLoadFailurePolicy == "recompute",
		Consumer: top == "kv_consumer" || top == "kv_both", Producer: isProducer(top)})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// worker is the offloading worker mock of oracle/run_offloading.py (Worker).
type worker struct {
	delay       func(id string, store bool) int
	unsubmitted []int
	completesAt map[int]int
	jobReq      map[int]string
	isLoad      map[int]bool
	workers     int
}

func (w *worker) execute(m offloading.Meta, step int) (sched.TransferOutcome, []int) {
	for _, j := range m.Loads {
		w.jobReq[j.ID], w.isLoad[j.ID] = j.Req, true
	}
	for _, j := range m.Stores {
		w.jobReq[j.ID] = j.Req
	}
	flush := map[int]bool{}
	for _, j := range m.Flush {
		flush[j] = true
	}
	var now []int
	for _, j := range m.Stores {
		if flush[j.ID] {
			w.unsubmitted = append(w.unsubmitted, j.ID)
		} else {
			now = append(now, j.ID)
		}
	}
	for _, j := range w.unsubmitted {
		w.completesAt[j] = step + w.delay(w.jobReq[j], true)
	}
	w.unsubmitted = nil
	for j := range flush {
		if _, ok := w.completesAt[j]; ok {
			w.completesAt[j] = step
		}
	}
	for _, j := range m.Loads {
		w.completesAt[j.ID] = step + w.delay(j.Req, false)
	}
	w.unsubmitted = append(w.unsubmitted, now...)
	var done []int
	for j, at := range w.completesAt {
		if at <= step {
			done = append(done, j)
		}
	}
	sort.Ints(done)
	recv := map[string]bool{}
	for _, j := range done {
		delete(w.completesAt, j)
		if w.isLoad[j] {
			recv[w.jobReq[j]] = true
		}
	}
	var o sched.TransferOutcome
	for id := range recv {
		o.FinishedRecving = append(o.FinishedRecving, id)
	}
	sort.Strings(o.FinishedRecving)
	if len(done) > 0 {
		cs := make([]offloading.Completion, len(done))
		for k, j := range done {
			cs[k] = offloading.Completion{Job: j, Workers: w.workers}
		}
		o.WorkerMeta = cs
	}
	return o, done
}

// sampler is the GPU mock: a sampling request's next assigned token, the last as EOS, plus the
// drafts the scenario's plan accepts at each verify step.
type sampler struct {
	reqs     []scenarioReq
	byID     map[string]int
	sampled  map[string]int
	verified map[string]int
	async    bool
}

func (g *sampler) sample(id string, drafts []int64) []int64 {
	r := g.reqs[g.byID[id]]
	n := 1
	if len(drafts) > 0 {
		real := len(drafts)
		if !g.async {
			real = 0
			for _, d := range drafts {
				if d >= 0 {
					real++
				}
			}
		}
		v := g.verified[id]
		g.verified[id] = v + 1
		if v < len(r.Accepted) {
			n += min(r.Accepted[v], real)
		}
	}
	var out []int64
	for j := 0; j < n; j++ {
		k := g.sampled[id]
		if k >= len(r.Output)-1 {
			out = append(out, eosToken)
		} else {
			out = append(out, r.Output[k])
		}
		g.sampled[id] = k + 1
		if k >= len(r.Output)-1 {
			break
		}
	}
	return out
}

// harness is the transport and GPU mocks of one run: the offloading worker for the offloading
// child, and the NIXL transport mock (oracle/run_connector.py transport_report) for the NIXL
// child. A transfer is reported once, in the first output applied at or after its step.
type harness struct {
	su          setup
	sc          scenario
	reqs        []scenarioReq
	byID        map[string]int
	w           *worker
	g           *sampler
	loadStarted map[string]int
	// nixlOwned are the remote prefills the wrapper assigned to the NIXL child when they began to
	// wait: the only pulls its transport lands. One the offloading child won waits for its CPU load,
	// and NIXL, given no external tokens, only notifies the prefiller (oracle/run_multiconnector.py).
	nixlOwned  map[string]bool
	finishedAt map[string]int
	reported   map[string]bool
	tr         trace
}

func newHarness(t *testing.T, sc scenario) *harness {
	t.Helper()
	su := configure(t, sc)
	c := sc.Config
	reqs := sc.Requests
	sort.SliceStable(reqs, func(i, j int) bool {
		if reqs[i].ArrivalStep != reqs[j].ArrivalStep {
			return reqs[i].ArrivalStep < reqs[j].ArrivalStep
		}
		return reqs[i].ID < reqs[j].ID
	})
	h := &harness{su: su, sc: sc, reqs: reqs, byID: map[string]int{}, loadStarted: map[string]int{}, nixlOwned: map[string]bool{},
		finishedAt: map[string]int{}, reported: map[string]bool{}, tr: trace{Finished: map[string]finishRecord{}}}
	for i, r := range reqs {
		h.byID[r.ID] = i
	}
	workers := 1
	if su.off != nil {
		workers = su.off.Workers()
	}
	h.w = &worker{completesAt: map[int]int{}, jobReq: map[int]string{}, isLoad: map[int]bool{}, workers: workers,
		delay: func(id string, store bool) int {
			r := reqs[h.byID[id]]
			if store {
				return intOr(r.StoreDelay, intOr(c.Transfer.StoreDelay, 1))
			}
			return intOr(r.LoadDelay, intOr(c.Transfer.LoadDelay, 1))
		}}
	h.g = &sampler{reqs: reqs, byID: h.byID, sampled: map[string]int{}, verified: map[string]int{}, async: c.AsyncScheduling}
	h.tr.Provenance.CPUChunks = su.chunks
	return h
}

// arrivals adds the requests that arrive at step i.
func (h *harness) arrivals(t *testing.T, next, i int) int {
	t.Helper()
	for next < len(h.reqs) && h.reqs[next].ArrivalStep <= i {
		r := h.reqs[next]
		mt := r.MaxTokens
		if mt == 0 {
			mt = 1 << 20
		}
		req := &sched.Request{ID: r.ID, Priority: r.Priority, ArrivalTime: float64(r.ArrivalStep),
			Prompt: r.Prompt, MaxTokens: mt, EOS: eosToken}
		bs := h.sc.Config.BlockSize
		switch {
		case r.RemotePrefill:
			req.Transfer = &sched.TransferParams{DoRemotePrefill: true, HasRemote: true, RemoteBlocks: (len(r.Prompt) + bs - 1) / bs}
		case r.RemoteDecode:
			req.Transfer = &sched.TransferParams{DoRemoteDecode: true}
		}
		if err := h.su.s.Add(req); err != nil {
			t.Fatal(err)
		}
		next++
	}
	return next
}

func (h *harness) aborts(i int) []string {
	var out []string
	for _, r := range h.reqs {
		if r.AbortStep != nil && *r.AbortStep == i {
			out = append(out, r.ID)
		}
	}
	sort.Strings(out)
	return out
}

// noteLoads records the step a request first waits for remote KV, when its pull starts.
func (h *harness) noteLoads(i int) {
	for _, id := range h.su.s.WaitingForRemoteKV() {
		if _, ok := h.loadStarted[id]; !ok {
			h.loadStarted[id] = i
			if c, ok := h.su.m.Chosen(id); ok && c == h.su.nxIdx {
				h.nixlOwned[id] = true
			}
		}
	}
}

// nixlReport is the NIXL transport mock's report at step i.
func (h *harness) nixlReport(i int) sched.TransferOutcome {
	var o sched.TransferOutcome
	s := h.su.s
	for _, r := range h.reqs {
		req := s.Request(r.ID)
		if req == nil {
			continue
		}
		if r.RemotePrefill && h.nixlOwned[r.ID] && (req.Status == sched.WaitingForRemoteKVs || req.Status >= sched.FinishedStopped) {
			if started, ok := h.loadStarted[r.ID]; ok && !h.reported["recv/"+r.ID] && i >= started+r.RecvDelay {
				h.reported["recv/"+r.ID] = true
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
					}
				}
			}
		}
		if r.RemoteDecode && req.Status >= sched.FinishedStopped {
			if at, ok := h.finishedAt[r.ID]; ok && !h.reported["send/"+r.ID] && i >= at+max(r.SendDelay, 1) {
				h.reported["send/"+r.ID] = true
				o.FinishedSending = append(o.FinishedSending, r.ID)
			}
		}
	}
	o.FinishedRecving = sortedStrings(o.FinishedRecving)
	o.FinishedSending = sortedStrings(o.FinishedSending)
	return o
}

// merge is the worker output the wrapper's scheduler sees: the NIXL report, the offloading
// worker's finished loads, and the offloading worker's completions in the offloading child's
// slot of a WorkerMeta.
func (h *harness) merge(st *step, nx, off sched.TransferOutcome) sched.TransferOutcome {
	o := nx
	st.Transport.Recving, st.Transport.Sending = nx.FinishedRecving, nx.FinishedSending
	st.Transport.Failed = sortedStrings(nx.FailedRecving)
	inv := map[int]bool{}
	for _, b := range nx.InvalidBlocks {
		inv[b] = true
	}
	st.Transport.InvalidBlocks = len(inv)
	o.FinishedRecving = sortedStrings(append(append([]string{}, nx.FinishedRecving...), off.FinishedRecving...))
	if off.WorkerMeta != nil {
		wm := make(WorkerMeta, h.su.m.Len())
		wm[h.su.offIdx] = off.WorkerMeta
		o.WorkerMeta = wm
	}
	return o
}

// execute runs the offloading worker on the offloading child's slice of a step's metadata.
func (h *harness) execute(meta any, i int) (sched.TransferOutcome, *offState) {
	if h.su.off == nil {
		return sched.TransferOutcome{}, &offState{}
	}
	m, _ := meta.(Meta)
	om, _ := m.At(h.su.offIdx).(offloading.Meta)
	off := offRows(om)
	o, done := h.w.execute(om, i)
	off.Completed = append([]int{}, done...)
	return o, off
}

func (h *harness) finish(f sched.Finished, i int) {
	if _, ok := h.finishedAt[f.ID]; !ok {
		h.finishedAt[f.ID] = i
	}
	reason := map[sched.Status]string{sched.FinishedStopped: "stop", sched.FinishedLength: "length",
		sched.FinishedError: "error"}[f.Status]
	h.tr.Finished[f.ID] = finishRecord{reason, h.g.sampled[f.ID], i}
}

func (h *harness) drafts(ids map[string]int) {
	if h.su.spec.NumTokens == 0 {
		return
	}
	d := map[string][]int64{}
	for id := range ids {
		if h.su.s.Request(id) != nil {
			d[id] = make([]int64, h.su.spec.NumTokens)
		}
	}
	h.su.s.UpdateDrafts(d)
}

// run drives a scenario through core/sched with the wrapper and both children, exactly as
// oracle/run_multiconnector.py drives vLLM's (the batch queue through core/engine when more than
// one batch is in flight).
func run(t *testing.T, sc scenario) trace {
	t.Helper()
	h := newHarness(t, sc)
	if h.su.queue > 1 {
		return h.runQueued(t)
	}
	s := h.su.s
	maxSteps := sc.MaxSteps
	if maxSteps == 0 {
		maxSteps = 10000
	}
	next := 0
	for i := 0; i < maxSteps; i++ {
		next = h.arrivals(t, next, i)
		if !s.HasRequests() {
			if next == len(h.reqs) {
				break
			}
			h.tr.Steps = append(h.tr.Steps, step{Step: i, Idle: true})
			continue
		}
		out := s.Schedule()
		h.noteLoads(i)
		offOut, off := h.execute(out.KVConnectorMetadata, i)
		mock := map[string][]int64{}
		spec := map[string]int{}
		for _, id := range out.Order {
			if s.Request(id).IsPrefillChunk {
				continue
			}
			if d := out.SpecDecode[id]; len(d) > 0 {
				spec[id] = len(d)
			}
			mock[id] = h.g.sample(id, out.SpecDecode[id])
		}
		st := step{Step: i, Scheduled: out.Scheduled, Preempted: sortedStrings(out.Preempted)}
		o := h.merge(&st, h.nixlReport(i), offOut)
		aborts := h.aborts(i)
		if len(aborts) > 0 {
			s.Abort(aborts)
		}
		st.Aborted = append([]string{}, aborts...)
		if len(spec) > 0 {
			st.SpecScheduled = spec
		}
		var fin []string
		for _, f := range s.UpdateWith(out, mock, o) {
			fin = append(fin, f.ID)
			h.finish(f, i)
		}
		h.drafts(out.Scheduled)
		st.Finished = sortedStrings(fin)
		h.record(&st, off)
		h.tr.Steps = append(h.tr.Steps, st)
	}
	return h.tr
}

// queuedGPU is the GPU and worker mocks for core/engine's batch queue: the offloading worker
// runs on a batch's metadata when the batch executes, the NIXL report and the worker's output
// are taken when the batch's output is applied.
type queuedGPU struct {
	h      *harness
	i      *int
	offOut map[int]sched.TransferOutcome
	cur    *step
}

func (q *queuedGPU) Output(b engine.Batch) map[string][]int64 {
	out := map[string][]int64{}
	for _, id := range b.Samples {
		out[id] = q.h.g.sample(id, b.Out.SpecDecode[id])
	}
	return out
}

func (q *queuedGPU) Transfers(b engine.Batch) sched.TransferOutcome {
	off := q.offOut[b.ID]
	delete(q.offOut, b.ID)
	q.cur.Processed = true
	return q.h.merge(q.cur, q.h.nixlReport(*q.i), off)
}

func (h *harness) runQueued(t *testing.T) trace {
	t.Helper()
	s := h.su.s
	var i int
	var st step
	q := &queuedGPU{h: h, i: &i, offOut: map[int]sched.TransferOutcome{}, cur: &st}
	core := &engine.Core{Sched: s, QueueSize: h.su.queue, GPU: q}
	maxSteps := h.sc.MaxSteps
	if maxSteps == 0 {
		maxSteps = 10000
	}
	next := 0
	for i = 0; i < maxSteps; i++ {
		next = h.arrivals(t, next, i)
		if !core.HasWork() && next == len(h.reqs) {
			break
		}
		st = step{Step: i, Scheduled: map[string]int{}, Preempted: []string{}}
		st.Transport.Recving, st.Transport.Sending, st.Transport.Failed = []string{}, []string{}, []string{}
		off := &offState{}
		aborts := h.aborts(i)
		if len(aborts) > 0 {
			core.Abort(aborts)
		}
		st.Aborted = append([]string{}, aborts...)
		var res engine.StepResult
		if core.HasWork() {
			var block bool
			res, block = core.StepBegin()
			h.noteLoads(i)
			if b := res.Scheduled; b != nil {
				st.Scheduled = b.Out.Scheduled
				st.Preempted = sortedStrings(b.Out.Preempted)
				for id, d := range b.Out.SpecDecode {
					if len(d) > 0 {
						if st.SpecScheduled == nil {
							st.SpecScheduled = map[string]int{}
						}
						st.SpecScheduled[id] = len(d)
					}
				}
				var o sched.TransferOutcome
				o, off = h.execute(b.Out.KVConnectorMetadata, i)
				q.offOut[b.ID] = o
			}
			if block {
				res = core.StepEnd(res)
			}
			if !h.sc.Config.AsyncScheduling && res.Executed && res.Scheduled != nil {
				h.drafts(res.Scheduled.Out.Scheduled)
			}
		} else {
			st.Idle = true
		}
		var fin []string
		for _, f := range res.Finished {
			fin = append(fin, f.ID)
			h.finish(f, i)
		}
		st.Finished = sortedStrings(fin)
		st.QueueLen = core.InFlight()
		h.record(&st, off)
		h.tr.Steps = append(h.tr.Steps, st)
	}
	return h.tr
}

func offRows(m offloading.Meta) *offState {
	off := &offState{Flush: append([]int{}, m.Flush...)}
	for _, j := range m.Loads {
		off.Loads = append(off.Loads, jobRow(j))
	}
	for _, j := range m.Stores {
		off.Stores = append(off.Stores, jobRow(j))
	}
	return off
}

func jobRow(j offloading.Job) []any {
	return []any{float64(j.ID), j.Req, floats(j.CPU), []any{floats(j.GPU), floats(j.GroupSizes), floats(j.BlockIndices)}}
}

func floats(in []int) []any {
	out := make([]any, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}

// record is the step's state after its output.
func (h *harness) record(st *step, off *offState) {
	s := h.su.s
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
	st.Chosen = map[string]int{}
	for _, r := range h.reqs {
		if i, ok := h.su.m.Chosen(r.ID); ok {
			st.Chosen[r.ID] = i
		}
	}
	if h.su.off != nil {
		off.Jobs = h.su.off.Jobs()
		off.PendingPush = h.su.m.HasPendingPushWork()
		cs, free, ev := h.su.off.CPU()
		off.CPU = cpuState{Chunks: [][]any{}, Free: free, Evictable: ev}
		for _, ch := range cs {
			off.CPU.Chunks = append(off.CPU.Chunks, []any{fmt.Sprintf("%x/%d", ch.Key.HashPrefix(), ch.Key.Group()),
				float64(ch.ID), float64(ch.RefCnt)})
		}
		st.Offload = off
	}
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

// earliestDivergence names the first step and field that differ. The wrapper's choice and the
// children's decisions come first: they explain the rest.
func earliestDivergence(want, got trace) string {
	if want.Provenance.CPUChunks != got.Provenance.CPUChunks {
		return fmt.Sprintf("CPU tier: oracle %d chunks, Go %d", want.Provenance.CPUChunks, got.Provenance.CPUChunks)
	}
	rank := map[string]int{"chosen": 0, "offload": 1, "transport": 2}
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
		sort.SliceStable(keys, func(a, b int) bool {
			ra, oka := rank[keys[a]]
			rb, okb := rank[keys[b]]
			switch {
			case oka && okb:
				return ra < rb
			default:
				return oka && !okb
			}
		})
		for _, k := range keys {
			if k == "offload" {
				wo, wok := w[k].(map[string]any)
				go_, gok := g[k].(map[string]any)
				if wok != gok {
					return fmt.Sprintf("step %d, offload: oracle %v, Go %v", i, w[k], g[k])
				}
				if !wok {
					continue // a scenario without an offloading child records none
				}
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
			t.Fatalf("trace for %s records %s %q, want %q: regenerate it with oracle/run_multiconnector.py",
				filepath.Base(scenarioPath), c.field, c.got, c.want)
		}
	}
	if p.Tier != 4 {
		t.Fatalf("trace for %s records tier %d, want 4", filepath.Base(scenarioPath), p.Tier)
	}
}

// TestMultiConnectorAgainstVLLM runs every MultiConnector scenario under
// oracle/scenarios/multiconnector through core/sched with this package's wrapper over the NIXL
// and offloading schedulers, and compares each step with the trace vLLM v0.32.0's own
// MultiConnector, NixlConnectorScheduler and OffloadingConnectorScheduler produced on its real
// Scheduler (testdata/golden). The target is zero mismatches.
func TestMultiConnectorAgainstVLLM(t *testing.T) {
	scenarios, err := filepath.Glob("../../../oracle/scenarios/multiconnector/multi-*.json")
	if err != nil || len(scenarios) == 0 {
		t.Fatalf("no MultiConnector scenarios (%v)", err)
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

// TestMultiConnectorFuzzCorpusAgainstVLLM is the seeded fuzz test: BLIS_VLLM_MULTI_FUZZ_DIR names
// a directory of oracle MultiConnector fuzz scenarios with vLLM's trace for each
// (<name>.trace.json).
func TestMultiConnectorFuzzCorpusAgainstVLLM(t *testing.T) {
	dir := os.Getenv("BLIS_VLLM_MULTI_FUZZ_DIR")
	if dir == "" {
		t.Skip("BLIS_VLLM_MULTI_FUZZ_DIR is unset: the MultiConnector fuzz corpus was NOT checked (CI's oracle job sets it)")
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
	t.Logf("MultiConnector fuzz corpus: %d scenarios, %d mismatches", len(traces), mismatches)
	if mismatches > 0 {
		t.Fatalf("%d of %d fuzz scenarios differ from vLLM", mismatches, len(traces))
	}
}

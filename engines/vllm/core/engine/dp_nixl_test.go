package engine

import (
	"encoding/hex"
	"fmt"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/core/connector/nixl"
	"github.com/inference-sim/blis-vllm/core/kv"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// dpnScenario is a data-parallel connector scenario (oracle/README.md, "Data-parallel connector
// scenarios"): the DP scenario of oracle/run_dp.py with a NIXL connector on every rank.
type dpnScenario struct {
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
		DataParallelSize          int     `json:"data_parallel_size"`
		PrefillScheduleInterval   int     `json:"prefill_schedule_interval"`
		DPSyncInterval            int     `json:"dp_sync_interval"`
		StatsInterval             int     `json:"stats_interval"`
		StatsLag                  int     `json:"stats_lag"`
		AsyncScheduling           bool    `json:"async_scheduling"`
		Connector                 struct {
			Name                string         `json:"name"`
			KVRole              string         `json:"kv_role"`
			KVRoles             []string       `json:"kv_roles"`
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
		Rank          *int    `json:"rank"`
		RemotePrefill bool    `json:"remote_prefill"`
		RemoteDecode  bool    `json:"remote_decode"`
		RecvDelay     int     `json:"recv_delay"`
		RecvFailed    bool    `json:"recv_failed"`
		InvalidBlocks []int   `json:"invalid_blocks"`
		SendDelay     int     `json:"send_delay"`
		AbortStep     *int    `json:"abort_step"`
	} `json:"requests"`
	MaxSteps int `json:"max_steps"`
}

type dpnTransport struct {
	Recving       []string `json:"recving"`
	Sending       []string `json:"sending"`
	Failed        []string `json:"failed"`
	InvalidBlocks int      `json:"invalid_blocks"`
}

// dpnRank is one rank's record of one lockstep step: the DP fields oracle/run_dp.py records and,
// beside them, what oracle/run_connector.py records for a scheduler.
type dpnRank struct {
	Rank      int            `json:"rank"`
	Dummy     bool           `json:"dummy"`
	Throttled bool           `json:"throttled"`
	Scheduled map[string]int `json:"scheduled"`
	Preempted []string       `json:"preempted"`
	Finished  []string       `json:"finished"`
	Aborted   []string       `json:"aborted"`
	Transport dpnTransport   `json:"transport"`
	// Processed and QueueLen are a batch-queue rank's (async scheduling only).
	Processed *bool `json:"processed,omitempty"`
	QueueLen  *int  `json:"queue_len,omitempty"`

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
}

type dpnStep struct {
	Step        int       `json:"step"`
	Ranks       []dpnRank `json:"ranks"`
	Waves       []int     `json:"waves"`
	Running     []bool    `json:"running"`
	StepCounter []int     `json:"step_counter"`
	Counts      [][]int   `json:"counts"`
}

type dpnFinish struct {
	Reason       string `json:"reason"`
	Rank         int    `json:"rank"`
	OutputTokens int    `json:"output_tokens"`
	Step         int    `json:"step"`
}

type dpnTrace struct {
	Provenance struct {
		Tier           int    `json:"tier"`
		VLLMVersion    string `json:"vllm_version"`
		VLLMCommit     string `json:"vllm_commit"`
		ScenarioSHA256 string `json:"scenario_sha256"`
		MockVersion    string `json:"mock_version"`
	} `json:"provenance"`
	Routed   map[string]int       `json:"routed"`
	Steps    []dpnStep            `json:"steps"`
	Finished map[string]dpnFinish `json:"finished"`
}

// dpnGPU is one rank's GPU and transport mock (oracle/run_connector.py: the sampling of the tier-2
// mock for a synchronous rank, QueuedGPU for a batch-queue rank): a request that reaches the end
// of its known tokens samples its next assigned token, the last one as EOS, and the transfers the
// worker finished ride on the output that is applied. The pull and hold bookkeeping is shared by
// the ranks, keyed by request ID, which names one request on one rank.
type dpnGPU struct {
	s           *sched.Scheduler
	reqs        map[string]int
	sc          *dpnScenario
	step        *int
	async       bool
	sampled     map[string]int
	loadStarted map[string]int
	finishedAt  map[string]int
	reported    map[string]bool
	// rec is the record of the rank's step in progress, where the transport mock notes its report.
	rec *dpnRank
}

func (g *dpnGPU) Output(b Batch) map[string][]int64 {
	out := map[string][]int64{}
	for _, id := range b.Samples {
		o := g.sc.Requests[g.reqs[id]].Output
		k := g.sampled[id]
		switch {
		case g.async:
			// A step scheduled past the last token samples EOS again, which the engine ignores.
			if k >= len(o)-1 {
				out[id] = []int64{eosToken}
			} else {
				out[id] = []int64{o[k]}
			}
			g.sampled[id] = k + 1
		case k < len(o):
			if k == len(o)-1 {
				out[id] = []int64{eosToken}
			} else {
				out[id] = []int64{o[k]}
			}
			g.sampled[id] = k + 1
		}
	}
	return out
}

// Transfers is run_connector.py transport_report: each transfer reported once, in the first output
// applied at or after its step.
func (g *dpnGPU) Transfers(Batch) sched.TransferOutcome {
	var o sched.TransferOutcome
	i := *g.step
	for _, r := range g.sc.Requests {
		req := g.s.Request(r.ID)
		if req == nil {
			continue
		}
		// A pull lands at its step even if the request was aborted since.
		if r.RemotePrefill && (req.Status == sched.WaitingForRemoteKVs || req.Status >= sched.FinishedStopped) {
			if started, ok := g.loadStarted[r.ID]; ok && !g.reported["recv/"+r.ID] && i >= started+r.RecvDelay {
				g.reported["recv/"+r.ID] = true
				o.FinishedRecving = append(o.FinishedRecving, r.ID)
				if r.RecvFailed {
					o.FailedRecving = append(o.FailedRecving, r.ID)
				}
				var pulled []int
				for _, b := range g.s.KV.Blocks(r.ID) {
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
			if at, ok := g.finishedAt[r.ID]; ok && !g.reported["send/"+r.ID] && i >= at+max(r.SendDelay, 1) {
				g.reported["send/"+r.ID] = true
				o.FinishedSending = append(o.FinishedSending, r.ID)
			}
		}
	}
	// The worker reports sets; the transport mock fixes their order by request ID on both sides.
	o.FinishedRecving = dpnSorted(o.FinishedRecving)
	o.FinishedSending = dpnSorted(o.FinishedSending)
	g.rec.Transport = dpnTransport{Recving: o.FinishedRecving, Sending: o.FinishedSending,
		Failed: dpnSorted(o.FailedRecving), InvalidBlocks: dpnUnique(o.InvalidBlocks)}
	if g.async {
		processed := true
		g.rec.Processed = &processed
	}
	return o
}

func dpnSorted(in []string) []string {
	out := append([]string{}, in...)
	sort.Strings(out)
	return out
}

func dpnUnique(in []int) int {
	seen := map[int]bool{}
	for _, v := range in {
		seen[v] = true
	}
	return len(seen)
}

// runDPNixl drives a DP connector scenario exactly as oracle/run_dp.py drives vLLM's DP engine-core
// loop over ranks that each hold a NixlConnector scheduler: core.DP over one sched.Scheduler per
// rank with this repository's NIXL scheduler installed, and a mock transport per rank.
func runDPNixl(t *testing.T, sc dpnScenario) dpnTrace {
	t.Helper()
	c := sc.Config
	n := c.DataParallelSize
	roles := c.Connector.KVRoles
	if len(roles) == 0 {
		for i := 0; i < n; i++ {
			roles = append(roles, c.Connector.KVRole)
		}
	}
	if c.Connector.Name != "NixlConnector" || len(roles) != n {
		t.Fatalf("connector %q with %d roles for %d ranks", c.Connector.Name, len(roles), n)
	}
	caching, chunked := true, true
	if c.EnablePrefixCaching != nil {
		caching = *c.EnablePrefixCaching
	}
	if c.EnableChunkedPrefill != nil {
		chunked = *c.EnableChunkedPrefill
	}
	queue := MaxConcurrentBatches(1, c.AsyncScheduling, false)
	reqs := append(sc.Requests[:0:0], sc.Requests...)
	sort.SliceStable(reqs, func(i, j int) bool {
		if reqs[i].ArrivalStep != reqs[j].ArrivalStep {
			return reqs[i].ArrivalStep < reqs[j].ArrivalStep
		}
		return reqs[i].ID < reqs[j].ID
	})
	sc.Requests = reqs
	byID := map[string]int{}
	for i, r := range reqs {
		byID[r.ID] = i
	}
	var step int
	loadStarted, finishedAt := map[string]int{}, map[string]int{}
	d := &DP{LB: NewLoadBalancer(n)}
	var gpus []*dpnGPU
	var conns []*nixl.Scheduler
	for i := 0; i < n; i++ {
		s, err := sched.New(sched.Config{
			MaxNumSeqs: c.MaxNumSeqs, MaxNumBatchedTokens: c.MaxNumBatchedTokens, BlockSize: c.BlockSize,
			NumBlocks: c.NumBlocks, MaxModelLen: c.MaxModelLen, LongPrefillTokenThreshold: c.LongPrefillTokenThreshold,
			EnableChunkedPrefill: chunked, Policy: sched.Policy(c.SchedulingPolicy), EnablePrefixCaching: caching,
			ReserveFullSequence: true, Watermark: c.Watermark, KVGroups: kv.ResolveGroups(nil, false),
			Async: c.AsyncScheduling, DecodeStagger: 1, MaxConcurrentBatches: queue,
		})
		if err != nil {
			t.Fatal(err)
		}
		role := roles[i]
		ncfg := nixl.Default()
		if v, ok := c.Connector.ExtraConfig["kv_recompute_threshold"].(float64); ok {
			ncfg.RecomputeThreshold = int(v)
		}
		if v, ok := c.Connector.ExtraConfig["bidirectional_kv_xfer"].(bool); ok {
			ncfg.Bidirectional = v
		}
		conn := nixl.New(ncfg)
		if err := s.SetConnector(conn, sched.ConnectorConfig{RecomputeLoadFailures: c.Connector.KVLoadFailurePolicy == "recompute",
			Consumer: role == "kv_consumer" || role == "kv_both", Producer: role == "kv_producer" || role == "kv_both"}); err != nil {
			t.Fatal(err)
		}
		gpu := &dpnGPU{s: s, reqs: byID, sc: &sc, step: &step, async: c.AsyncScheduling, sampled: map[string]int{},
			loadStarted: loadStarted, finishedAt: finishedAt, reported: map[string]bool{}}
		gpus = append(gpus, gpu)
		conns = append(conns, conn)
		d.Ranks = append(d.Ranks, &Core{Sched: s, QueueSize: queue, GPU: gpu})
		d.Waves = append(d.Waves, &Wave{PrefillScheduleInterval: max(c.PrefillScheduleInterval, 1), SyncInterval: max(c.DPSyncInterval, 1)})
	}
	maxSteps := sc.MaxSteps
	if maxSteps == 0 {
		maxSteps = 10000
	}
	statsEvery := max(c.StatsInterval, 1)
	tr := dpnTrace{Routed: map[string]int{}, Finished: map[string]dpnFinish{}}
	var history [][][3]float64
	record := func() {
		counts := make([][3]float64, n)
		for i, r := range d.Ranks {
			nr, nw := r.Sched.Counts()
			counts[i] = [3]float64{float64(nw), float64(nr), r.Sched.KV.Pool.Usage()}
		}
		history = append(history, counts)
	}
	record()
	next := 0
	for step = 0; step < maxSteps; step++ {
		if step%statsEvery == 0 {
			if idx := len(history) - 1 - c.StatsLag; idx >= 0 {
				d.LB.Publish(history[idx])
			} else {
				d.LB.Publish(make([][3]float64, n))
			}
		}
		for next < len(reqs) && reqs[next].ArrivalStep <= step {
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
			var rank int
			if r.Rank != nil {
				rank = d.AddTo(r.ID, *r.Rank)
			} else {
				rank = d.Add(r.ID)
			}
			if err := d.Ranks[rank].Sched.Add(req); err != nil {
				t.Fatal(err)
			}
			tr.Routed[r.ID] = rank
			next++
		}
		idle := next == len(reqs)
		for i, r := range d.Ranks {
			if r.HasWork() || d.Waves[i].Running {
				idle = false
			}
		}
		if idle {
			break
		}
		rec := dpnStep{Step: step}
		recs := make([]*dpnRank, n)
		for i, r := range d.Ranks {
			rr := &dpnRank{Rank: i, Scheduled: map[string]int{}, Preempted: []string{}, Finished: []string{},
				Aborted: []string{}, Transport: dpnTransport{Recving: []string{}, Sending: []string{}, Failed: []string{}}}
			if c.AsyncScheduling {
				processed := false
				rr.Processed = &processed
			}
			recs[i], gpus[i].rec = rr, rr
			// The client's aborts of the requests routed here at this step. A rank queues them on
			// its EngineCore (a batch-queue rank, whenever); a synchronous rank applies them in
			// the step it runs (oracle/run_connector.py).
			for _, q := range reqs {
				if q.AbortStep != nil && *q.AbortStep == step {
					if rank, ok := tr.Routed[q.ID]; ok && rank == i {
						rr.Aborted = append(rr.Aborted, q.ID)
					}
				}
			}
			sort.Strings(rr.Aborted)
			for _, id := range rr.Aborted {
				d.LB.Finished(id)
			}
			if len(rr.Aborted) > 0 && (c.AsyncScheduling || r.Sched.HasRequests()) {
				r.Abort(rr.Aborted)
			}
		}
		out := d.StepBegin()
		for i, r := range d.Ranks {
			if b := out[i].Step.Scheduled; b != nil {
				// A pull starts when the schedule that parks the request ran.
				for _, id := range r.Sched.WaitingForRemoteKV() {
					if _, ok := loadStarted[id]; !ok {
						loadStarted[id] = step
					}
				}
				conns[i].BuildMeta()
				recs[i].Scheduled = b.Out.Scheduled
				recs[i].Preempted = dpnSorted(b.Out.Preempted)
			}
			if out[i].Block {
				out[i].Step = r.StepEnd(out[i].Step)
			}
		}
		d.EndStep(out)
		for i, r := range d.Ranks {
			rr := recs[i]
			rr.Dummy, rr.Throttled = out[i].Dummy, out[i].Throttled
			if rr.Scheduled == nil {
				rr.Scheduled = map[string]int{}
			}
			if c.AsyncScheduling {
				ql := r.InFlight()
				rr.QueueLen = &ql
			}
			for _, f := range out[i].Step.Finished {
				rr.Finished = append(rr.Finished, f.ID)
				if _, ok := finishedAt[f.ID]; !ok {
					finishedAt[f.ID] = step
				}
				reason := map[sched.Status]string{sched.FinishedStopped: "stop", sched.FinishedLength: "length",
					sched.FinishedError: "error"}[f.Status]
				tr.Finished[f.ID] = dpnFinish{Reason: reason, Rank: i, OutputTokens: gpus[i].sampled[f.ID], Step: step}
				d.LB.Finished(f.ID)
			}
			sort.Strings(rr.Finished)
			dpnState(r.Sched, rr)
			rec.Ranks = append(rec.Ranks, *rr)
		}
		for i, w := range d.Waves {
			nr, nw := d.Ranks[i].Sched.Counts()
			rec.Waves = append(rec.Waves, w.Current)
			rec.Running = append(rec.Running, w.Running)
			rec.StepCounter = append(rec.StepCounter, w.StepCounter)
			rec.Counts = append(rec.Counts, []int{nr, nw})
		}
		record()
		tr.Steps = append(tr.Steps, rec)
	}
	return tr
}

// dpnState fills a rank's KV and connector-visible state as the oracle records it after each step
// (run_scheduler.py kv_state, run_connector.py connector_state_of).
func dpnState(s *sched.Scheduler, rr *dpnRank) {
	nr, nw := s.Counts()
	rr.FreeBlocks, rr.Running, rr.NumRunning, rr.NumWaiting = s.KV.Pool.NumFree(), dpnSorted(s.Running()), nr, nw
	rr.KVUsage, rr.Blocks = s.KV.Pool.Usage(), map[string]any{}
	for _, id := range rr.Running {
		rows := []any{}
		for _, b := range s.KV.Blocks(id) {
			var h any
			if b.Hash != nil {
				h = hex.EncodeToString(b.Hash)
			}
			rows = append(rows, []any{h, float64(b.RefCnt)})
		}
		rr.Blocks[id] = rows
	}
	rr.WaitingForRemoteKV = dpnSorted(s.WaitingForRemoteKV())
	rr.RecvLanded = dpnSorted(s.RecvLanded())
	rr.DelayFreed = dpnSorted(s.DelayFreed())
	rr.KVHoldingWaiting = dpnSorted(s.Holding())
	fc := s.FetchCounts()
	rr.KVFetchStages = fc[:]
	rr.NumRequests = s.NumRequests()
}

// dpnDivergence names the first routing, step, rank and field that differ.
func dpnDivergence(want, got dpnTrace) string {
	if !reflect.DeepEqual(want.Routed, got.Routed) {
		for id, w := range want.Routed {
			if got.Routed[id] != w {
				return fmt.Sprintf("routing of %s: vLLM rank %d, blis-vllm rank %d", id, w, got.Routed[id])
			}
		}
		return fmt.Sprintf("routing: vLLM %v, blis-vllm %v", want.Routed, got.Routed)
	}
	for i := 0; i < max(len(want.Steps), len(got.Steps)); i++ {
		if i >= len(want.Steps) || i >= len(got.Steps) {
			return fmt.Sprintf("step %d: the runs have %d and %d steps", i, len(want.Steps), len(got.Steps))
		}
		w, g := want.Steps[i], got.Steps[i]
		for r := range w.Ranks {
			wr, gr := w.Ranks[r], g.Ranks[r]
			for _, f := range []struct {
				name string
				w, g any
			}{
				{"scheduled", nonNilMap(wr.Scheduled), nonNilMap(gr.Scheduled)},
				{"dummy", wr.Dummy, gr.Dummy},
				{"throttled", wr.Throttled, gr.Throttled},
				{"processed", wr.Processed, gr.Processed},
				{"queue_len", wr.QueueLen, gr.QueueLen},
				{"preempted", dpnSorted(wr.Preempted), dpnSorted(gr.Preempted)},
				{"transport", wr.Transport, gr.Transport},
				{"aborted", dpnSorted(wr.Aborted), dpnSorted(gr.Aborted)},
				{"finished", dpnSorted(wr.Finished), dpnSorted(gr.Finished)},
				{"running", dpnSorted(wr.Running), dpnSorted(gr.Running)},
				{"num_running", wr.NumRunning, gr.NumRunning},
				{"num_waiting", wr.NumWaiting, gr.NumWaiting},
				{"waiting_for_remote_kv", dpnSorted(wr.WaitingForRemoteKV), dpnSorted(gr.WaitingForRemoteKV)},
				{"recv_landed", dpnSorted(wr.RecvLanded), dpnSorted(gr.RecvLanded)},
				{"delay_freed", dpnSorted(wr.DelayFreed), dpnSorted(gr.DelayFreed)},
				{"kv_holding_waiting", dpnSorted(wr.KVHoldingWaiting), dpnSorted(gr.KVHoldingWaiting)},
				{"kv_fetch_stages", wr.KVFetchStages, gr.KVFetchStages},
				{"num_requests", wr.NumRequests, gr.NumRequests},
				{"free_blocks", wr.FreeBlocks, gr.FreeBlocks},
				{"kv_usage", fmt.Sprintf("%.12f", wr.KVUsage), fmt.Sprintf("%.12f", gr.KVUsage)},
				{"blocks", nonNilAny(wr.Blocks), nonNilAny(gr.Blocks)},
			} {
				if !reflect.DeepEqual(f.w, f.g) {
					return fmt.Sprintf("step %d, rank %d, %s: vLLM %v, blis-vllm %v", i, r, f.name, f.w, f.g)
				}
			}
		}
		for _, f := range []struct {
			name string
			w, g any
		}{
			{"waves", w.Waves, g.Waves}, {"running", w.Running, g.Running},
			{"step_counter", w.StepCounter, g.StepCounter}, {"counts", w.Counts, g.Counts},
		} {
			if !reflect.DeepEqual(f.w, f.g) {
				return fmt.Sprintf("step %d, %s: vLLM %v, blis-vllm %v", i, f.name, f.w, f.g)
			}
		}
	}
	if !reflect.DeepEqual(want.Finished, got.Finished) {
		return fmt.Sprintf("finished: vLLM %v, blis-vllm %v", want.Finished, got.Finished)
	}
	return ""
}

func nonNilAny(m map[string]any) map[string]any {
	if m == nil {
		return map[string]any{}
	}
	return m
}

// TestDPNixlAgainstVLLM runs every data-parallel connector scenario (oracle/scenarios-dp-nixl)
// through core.DP over one sched.Scheduler per rank with the NIXL scheduler installed, and compares
// it, step by step and rank by rank, with the trace vLLM v0.32.0 produced with its real Scheduler
// and NixlConnector scheduler side on every rank under its DP engine-core loop and load balancer
// (oracle/run_dp.py, testdata/golden-dp-nixl/<name>.json): the routing, each rank's schedule,
// transfers, KV and connector state, dummy batches, and the waves.
func TestDPNixlAgainstVLLM(t *testing.T) {
	scenarios, err := filepath.Glob("../../oracle/scenarios-dp-nixl/*.json")
	if err != nil || len(scenarios) == 0 {
		t.Fatalf("no DP connector scenarios (%v)", err)
	}
	for _, path := range scenarios {
		name := strings.TrimSuffix(filepath.Base(path), ".json")
		t.Run(name, func(t *testing.T) {
			var sc dpnScenario
			readJSON(t, path, &sc)
			var want dpnTrace
			readJSON(t, filepath.Join("testdata", "golden-dp-nixl", name+".json"), &want)
			var p dpTrace
			p.Provenance.Tier, p.Provenance.VLLMVersion, p.Provenance.VLLMCommit = want.Provenance.Tier, want.Provenance.VLLMVersion, want.Provenance.VLLMCommit
			p.Provenance.ScenarioSHA256, p.Provenance.MockVersion = want.Provenance.ScenarioSHA256, want.Provenance.MockVersion
			checkDPProvenance(t, path, p)
			if d := dpnDivergence(want, runDPNixl(t, sc)); d != "" {
				t.Fatalf("earliest divergence from vLLM v0.32.0: %s", d)
			}
		})
	}
}

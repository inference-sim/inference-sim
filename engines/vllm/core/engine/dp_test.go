package engine

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/core/sched"
)

type dpScenario struct {
	Config struct {
		MaxNumSeqs                int    `json:"max_num_seqs"`
		MaxNumBatchedTokens       int    `json:"max_num_batched_tokens"`
		BlockSize                 int    `json:"block_size"`
		NumBlocks                 int    `json:"num_blocks"`
		MaxModelLen               int    `json:"max_model_len"`
		LongPrefillTokenThreshold int    `json:"long_prefill_token_threshold"`
		EnablePrefixCaching       bool   `json:"enable_prefix_caching"`
		SchedulingPolicy          string `json:"scheduling_policy"`
		DataParallelSize          int    `json:"data_parallel_size"`
		PrefillScheduleInterval   int    `json:"prefill_schedule_interval"`
		DPSyncInterval            int    `json:"dp_sync_interval"`
		StatsInterval             int    `json:"stats_interval"`
		StatsLag                  int    `json:"stats_lag"`
		AsyncScheduling           bool   `json:"async_scheduling"`
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

type dpRankRec struct {
	Rank      int            `json:"rank"`
	Dummy     bool           `json:"dummy"`
	Scheduled map[string]int `json:"scheduled"`
	Finished  []string       `json:"finished"`
	Throttled bool           `json:"throttled"`
	// Processed and QueueLen are the async trace's (oracle/run_dp_async.py): whether the rank's
	// step applied a batch's output, and its batch queue's length after the step.
	Processed *bool `json:"processed,omitempty"`
	QueueLen  *int  `json:"queue_len,omitempty"`
}

type dpStepRec struct {
	Step        int         `json:"step"`
	Arrivals    [][]any     `json:"arrivals"`
	Ranks       []dpRankRec `json:"ranks"`
	Waves       []int       `json:"waves"`
	Running     []bool      `json:"running"`
	StepCounter []int       `json:"step_counter"`
	Counts      [][]int     `json:"counts"`
}

type dpTrace struct {
	Provenance struct {
		Tier           int    `json:"tier"`
		VLLMVersion    string `json:"vllm_version"`
		VLLMCommit     string `json:"vllm_commit"`
		ScenarioSHA256 string `json:"scenario_sha256"`
		MockVersion    string `json:"mock_version"`
	} `json:"provenance"`
	Routed map[string]int `json:"routed"`
	Steps  []dpStepRec    `json:"steps"`
}

// runDP drives a DP scenario exactly as oracle/run_dp.py drives vLLM's DP engine-core loop.
func runDP(t *testing.T, sc dpScenario) dpTrace {
	t.Helper()
	c := sc.Config
	n := c.DataParallelSize
	maxLen := c.MaxModelLen
	if maxLen == 0 {
		maxLen = c.MaxNumBatchedTokens
	}
	gpu := &mockGPU{outputs: map[string][]int64{}, sampled: map[string]int{}, async: c.AsyncScheduling}
	d := &DP{LB: NewLoadBalancer(n)}
	// With async scheduling every rank runs the AsyncScheduler and step_with_batch_queue with a
	// queue of max_concurrent_batches (oracle/run_dp_async.py); otherwise step() (run_dp.py).
	queue := MaxConcurrentBatches(1, c.AsyncScheduling, false)
	for i := 0; i < n; i++ {
		s, err := sched.New(sched.Config{
			MaxNumSeqs: c.MaxNumSeqs, MaxNumBatchedTokens: c.MaxNumBatchedTokens, BlockSize: c.BlockSize,
			NumBlocks: c.NumBlocks, MaxModelLen: maxLen, LongPrefillTokenThreshold: c.LongPrefillTokenThreshold,
			EnableChunkedPrefill: true, Policy: sched.Policy(c.SchedulingPolicy), EnablePrefixCaching: c.EnablePrefixCaching,
			ReserveFullSequence: true, Async: c.AsyncScheduling, DecodeStagger: 1, MaxConcurrentBatches: queue,
		})
		if err != nil {
			t.Fatal(err)
		}
		d.Ranks = append(d.Ranks, &Core{Sched: s, QueueSize: queue, GPU: gpu})
		d.Waves = append(d.Waves, &Wave{PrefillScheduleInterval: max(c.PrefillScheduleInterval, 1), SyncInterval: max(c.DPSyncInterval, 1)})
	}
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
	statsEvery := max(c.StatsInterval, 1)
	tr := dpTrace{Routed: map[string]int{}}
	// The coordinator publishes the counts it received stats_lag steps ago (the oracle's
	// transport delay).
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
	for step := 0; step < maxSteps; step++ {
		if step%statsEvery == 0 {
			if idx := len(history) - 1 - c.StatsLag; idx >= 0 {
				d.LB.Publish(history[idx])
			} else {
				d.LB.Publish(make([][3]float64, n))
			}
		}
		rec := dpStepRec{Step: step, Arrivals: [][]any{}}
		for next < len(reqs) && reqs[next].ArrivalStep <= step {
			r := reqs[next]
			mt := r.MaxTokens
			if mt == 0 {
				mt = 1 << 20
			}
			rank := d.Add(r.ID)
			if err := d.Ranks[rank].Sched.Add(&sched.Request{ID: r.ID, Priority: r.Priority, ArrivalTime: float64(r.ArrivalStep),
				Prompt: r.Prompt, MaxTokens: mt, EOS: eosToken}); err != nil {
				t.Fatal(err)
			}
			gpu.outputs[r.ID] = r.Output
			tr.Routed[r.ID] = rank
			rec.Arrivals = append(rec.Arrivals, []any{r.ID, float64(rank)})
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
		for i, rs := range d.Step() {
			rr := dpRankRec{Rank: i, Dummy: rs.Dummy, Throttled: rs.Throttled, Scheduled: map[string]int{}, Finished: []string{}}
			if rs.Step.Scheduled != nil {
				rr.Scheduled = rs.Step.Scheduled.Out.Scheduled
			}
			if c.AsyncScheduling {
				processed, ql := rs.Step.Processed != nil, d.Ranks[i].InFlight()
				rr.Processed, rr.QueueLen = &processed, &ql
			}
			for _, f := range rs.Step.Finished {
				rr.Finished = append(rr.Finished, f.ID)
				d.LB.Finished(f.ID)
			}
			sort.Strings(rr.Finished)
			rec.Ranks = append(rec.Ranks, rr)
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

func dpDivergence(want, got dpTrace) string {
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
				{"finished", nonNil(wr.Finished), nonNil(gr.Finished)},
				{"processed", wr.Processed, gr.Processed},
				{"queue_len", wr.QueueLen, gr.QueueLen},
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
	return ""
}

func checkDPProvenance(t *testing.T, scenarioPath string, tr dpTrace) {
	t.Helper()
	var asT trace
	asT.Provenance.Tier = tr.Provenance.Tier
	asT.Provenance.VLLMVersion = tr.Provenance.VLLMVersion
	asT.Provenance.VLLMCommit = tr.Provenance.VLLMCommit
	asT.Provenance.ScenarioSHA256 = tr.Provenance.ScenarioSHA256
	asT.Provenance.MockVersion = tr.Provenance.MockVersion
	checkProvenance(t, scenarioPath, asT)
}

// TestDPAgainstVLLM runs every DP scenario (oracle/scenarios-dp) through DP and compares it,
// step by step and rank by rank, with the trace vLLM v0.32.0's own DP engine-core loop and
// load balancer produced (testdata/golden-dp/<name>.json).
func TestDPAgainstVLLM(t *testing.T) {
	scenarios, err := filepath.Glob("../../oracle/scenarios-dp/*.json")
	if err != nil || len(scenarios) == 0 {
		t.Fatalf("no DP scenarios (%v)", err)
	}
	for _, path := range scenarios {
		name := strings.TrimSuffix(filepath.Base(path), ".json")
		t.Run(name, func(t *testing.T) {
			var sc dpScenario
			readJSON(t, path, &sc)
			var want dpTrace
			readJSON(t, filepath.Join("testdata", "golden-dp", name+".json"), &want)
			checkDPProvenance(t, path, want)
			if d := dpDivergence(want, runDP(t, sc)); d != "" {
				t.Fatalf("earliest divergence from vLLM v0.32.0: %s", d)
			}
		})
	}
}

// TestDPFuzzCorpusAgainstVLLM: BLIS_VLLM_DP_FUZZ_DIR names oracle/gen_fuzz_dp.py scenarios
// with vLLM's trace for each; every one must match at zero mismatches.
func TestDPFuzzCorpusAgainstVLLM(t *testing.T) {
	dir := os.Getenv("BLIS_VLLM_DP_FUZZ_DIR")
	if dir == "" {
		t.Skip("BLIS_VLLM_DP_FUZZ_DIR is unset: the DP fuzz corpus was NOT checked (CI's oracle job sets it)")
	}
	traces, err := filepath.Glob(filepath.Join(dir, "*.trace.json"))
	if err != nil || len(traces) == 0 {
		t.Fatalf("no traces in %s (%v)", dir, err)
	}
	mismatches := 0
	for _, tp := range traces {
		name := strings.TrimSuffix(filepath.Base(tp), ".trace.json")
		var sc dpScenario
		readJSON(t, filepath.Join(dir, name+".json"), &sc)
		var want dpTrace
		readJSON(t, tp, &want)
		checkDPProvenance(t, filepath.Join(dir, name+".json"), want)
		if d := dpDivergence(want, runDP(t, sc)); d != "" {
			mismatches++
			if mismatches <= 20 {
				t.Errorf("%s: earliest divergence from vLLM v0.32.0: %s", name, d)
			}
		}
	}
	t.Logf("DP fuzz corpus: %d scenarios, %d mismatches", len(traces), mismatches)
	if mismatches > 0 {
		t.Fail()
	}
}

// TestDPAsyncAgainstVLLM runs every async DP scenario (oracle/scenarios-dp-async) through DP,
// each rank running the AsyncScheduler and its batch queue, and compares it, step by step and
// rank by rank, with the trace vLLM v0.32.0's DP engine-core loop produced with each rank's own
// step_with_batch_queue (oracle/run_dp_async.py, testdata/golden-dp-async/<name>.json): the
// routing, every rank's schedule, finishes, dummy batches and throttle, which steps applied an
// output, each rank's queue length, and the waves.
func TestDPAsyncAgainstVLLM(t *testing.T) {
	scenarios, err := filepath.Glob("../../oracle/scenarios-dp-async/*.json")
	if err != nil || len(scenarios) == 0 {
		t.Fatalf("no async DP scenarios (%v)", err)
	}
	for _, path := range scenarios {
		name := strings.TrimSuffix(filepath.Base(path), ".json")
		t.Run(name, func(t *testing.T) {
			var sc dpScenario
			readJSON(t, path, &sc)
			if !sc.Config.AsyncScheduling {
				t.Fatalf("%s does not turn async scheduling on", name)
			}
			var want dpTrace
			readJSON(t, filepath.Join("testdata", "golden-dp-async", name+".json"), &want)
			checkDPProvenance(t, path, want)
			if d := dpDivergence(want, runDP(t, sc)); d != "" {
				t.Fatalf("earliest divergence from vLLM v0.32.0: %s", d)
			}
		})
	}
}

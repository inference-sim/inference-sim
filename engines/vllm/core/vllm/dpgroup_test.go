package vllm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"
	"gopkg.in/yaml.v3"
)

// dpGoldenScenario and dpGoldenTrace are oracle/run_dp.py's scenario and vLLM v0.32.0's trace of
// it (core/engine/testdata/golden-dp), as far as the contract path compares them.
type dpGoldenScenario struct {
	Config struct {
		MaxNumSeqs              int    `json:"max_num_seqs"`
		MaxNumBatchedTokens     int    `json:"max_num_batched_tokens"`
		BlockSize               int    `json:"block_size"`
		NumBlocks               int    `json:"num_blocks"`
		MaxModelLen             int    `json:"max_model_len"`
		EnablePrefixCaching     bool   `json:"enable_prefix_caching"`
		SchedulingPolicy        string `json:"scheduling_policy"`
		DataParallelSize        int    `json:"data_parallel_size"`
		PrefillScheduleInterval int    `json:"prefill_schedule_interval"`
		DPSyncInterval          int    `json:"dp_sync_interval"`
		StatsInterval           int    `json:"stats_interval"`
		StatsLag                int    `json:"stats_lag"`
		AsyncScheduling         bool   `json:"async_scheduling"`
	} `json:"config"`
	Requests []struct {
		ID          string  `json:"id"`
		ArrivalStep int     `json:"arrival_step"`
		Prompt      []int32 `json:"prompt"`
		Output      []int32 `json:"output"`
		MaxTokens   int     `json:"max_tokens"`
		Priority    int     `json:"priority"`
	} `json:"requests"`
}

type dpGoldenTrace struct {
	Routed map[string]int `json:"routed"`
	Steps  []struct {
		Step  int `json:"step"`
		Ranks []struct {
			Rank      int            `json:"rank"`
			Dummy     bool           `json:"dummy"`
			Scheduled map[string]int `json:"scheduled"`
			Finished  []string       `json:"finished"`
			Throttled bool           `json:"throttled"`
			QueueLen  *int           `json:"queue_len"`
		} `json:"ranks"`
	} `json:"steps"`
}

// TestDPGroupAgainstVLLM drives every DP golden scenario through the DP group by blis-schemas'
// engine.Engine contract (Add, Schedule, StepTime, Complete), one lockstep iteration a step, and
// compares the routing and every rank's scheduled tokens and finishes with vLLM v0.32.0's own DP
// engine-core loop and internal load balancer (oracle/run_dp.py, the same goldens tier 3 checks
// core/engine.DP against). The coordinator's publication runs every stats_interval steps, which
// is a time interval through the contract: one step is one microsecond here.
func TestDPGroupAgainstVLLM(t *testing.T) {
	scenarios, _ := filepath.Glob(filepath.Join("..", "..", "oracle", "scenarios-dp", "*.json"))
	async, _ := filepath.Glob(filepath.Join("..", "..", "oracle", "scenarios-dp-async", "*.json"))
	if len(scenarios) == 0 || len(async) == 0 {
		t.Fatal("no DP scenarios found")
	}
	ran := 0
	for _, path := range append(scenarios, async...) {
		name := strings.TrimSuffix(filepath.Base(path), ".json")
		var sc dpGoldenScenario
		readFile(t, path, &sc)
		if sc.Config.StatsLag > 0 {
			// The coordinator's transport lag is the oracle's model of a delayed snapshot; the
			// contract path publishes the live counts at the interval's first routing.
			continue
		}
		var want dpGoldenTrace
		golden := "golden-dp"
		if sc.Config.AsyncScheduling {
			golden = "golden-dp-async"
		}
		readFile(t, filepath.Join("..", "engine", "testdata", golden, name+".json"), &want)
		t.Run(name, func(t *testing.T) {
			if d := dpGroupDivergence(t, sc, want); d != "" {
				t.Errorf("earliest divergence from vLLM v0.32.0: %s", d)
			}
		})
		ran++
	}
	if ran < 8 {
		t.Fatalf("compared %d DP goldens, want at least 8 (4 with a batch queue per rank)", ran)
	}
}

func readFile(t *testing.T, path string, v any) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatal(err)
	}
}

func dpGroupDivergence(t *testing.T, sc dpGoldenScenario, want dpGoldenTrace) string {
	t.Helper()
	c := sc.Config
	maxLen := c.MaxModelLen
	if maxLen == 0 {
		maxLen = c.MaxNumBatchedTokens
	}
	policy := c.SchedulingPolicy
	if policy == "" {
		policy = "fcfs"
	}
	src := fmt.Sprintf("block_size: %d\nmax_num_seqs: %d\nmax_num_batched_tokens: %d\nmax_model_len: %d\n"+
		"num_gpu_blocks: %d\nuse_v2_model_runner: false\nasync_scheduling: %t\npipeline_parallel_size: 1\n"+
		"scheduling_policy: %s\nenable_prefix_caching: %t\ndata_parallel_size: %d\nprefill_schedule_interval: %d\n"+
		"dp_sync_interval: %d\ndp_stats_interval_us: %d\n",
		c.BlockSize, c.MaxNumSeqs, c.MaxNumBatchedTokens, maxLen, c.NumBlocks, c.AsyncScheduling, policy, c.EnablePrefixCaching,
		c.DataParallelSize, max(c.PrefillScheduleInterval, 1), max(c.DPSyncInterval, 1), max(c.StatsInterval, 1))
	// The oracle builds each rank's Scheduler with vLLM's create_scheduler, which makes no
	// startup check of the KV cache against max_model_len; a KV-pressure scenario states fewer
	// blocks than one max_model_len sequence. So the settings are resolved at a window the
	// blocks hold, and the scheduler then runs at the scenario's own max_model_len.
	fits := min(maxLen, (c.NumBlocks-1)*c.BlockSize)
	var node yaml.Node
	if err := yaml.Unmarshal([]byte(strings.Replace(src, fmt.Sprintf("max_model_len: %d\n", maxLen),
		fmt.Sprintf("max_model_len: %d\n", fits), 1)), &node); err != nil {
		t.Fatal(err)
	}
	cfg, err := decodeConfig(*node.Content[0])
	if err != nil {
		t.Fatal(err)
	}
	st, err := resolve(cfg)
	if err != nil {
		t.Fatal(err)
	}
	st.sched.MaxModelLen = maxLen
	g, err := newDPGroup(st, FixedPricer(100))
	if err != nil {
		t.Fatal(err)
	}
	reqs := append(sc.Requests[:0:0], sc.Requests...)
	sort.SliceStable(reqs, func(i, j int) bool {
		if reqs[i].ArrivalStep != reqs[j].ArrivalStep {
			return reqs[i].ArrivalStep < reqs[j].ArrivalStep
		}
		return reqs[i].ID < reqs[j].ID
	})
	routed := map[string]int{}
	next := 0
	// The contract's batches in flight, oldest first.
	var inflight []engine.Batch
	for step := 0; step < len(want.Steps)+1; step++ {
		now := engine.Time(step)
		for next < len(reqs) && reqs[next].ArrivalStep <= step {
			r := reqs[next]
			mt := r.MaxTokens
			if mt == 0 {
				mt = 1 << 20 // the oracle's unset max_tokens
			}
			if err := g.Add(engine.Request{ID: r.ID, Prompt: r.Prompt, Output: r.Output, MaxOutputTokens: mt,
				Priority: r.Priority, ArrivalTime: engine.Time(r.ArrivalStep) * 1e6}, now); err != nil {
				t.Fatal(err)
			}
			routed[r.ID] = g.rankOf[r.ID]
			next++
		}
		if step == len(want.Steps) {
			break
		}
		w := want.Steps[step]
		// One iteration of the DP loop: Schedule takes every rank's step; then, while a rank's
		// step waits on a batch in flight, that batch completes in the same iteration, as vLLM's
		// loop blocks on it.
		before := g.iterations
		b, err := g.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		if g.iterations == before {
			// The group has nothing to step; vLLM's loop records the idle iteration while
			// requests are still to arrive, with no rank scheduling or running a dummy batch.
			for r, wr := range w.Ranks {
				if wr.Dummy || len(wr.Scheduled) > 0 {
					return fmt.Sprintf("step %d, rank %d: the group did not step, vLLM's rank did (dummy %t, %v)",
						step, r, wr.Dummy, wr.Scheduled)
				}
			}
			continue
		}
		// One step is one microsecond here: an iteration of dummy batches never holds the next.
		g.busyUntil = 0
		steps := g.last
		if g.open != nil {
			steps = g.open.steps
		}
		scheduled := make([]map[string]int, len(g.ranks))
		for r := range scheduled {
			scheduled[r] = map[string]int{}
			if st := steps[r].Step.Scheduled; st != nil {
				for id, n := range st.Out.Scheduled {
					scheduled[r][id] = n
				}
			}
		}
		if !b.Empty() {
			inflight = append(inflight, b)
		}
		fin := map[string]bool{}
		for g.open != nil {
			if len(inflight) == 0 {
				return fmt.Sprintf("step %d: a rank waits on a batch, but none is in flight", step)
			}
			res, err := g.Complete(inflight[0].ID, engine.Outcome{}, now)
			if err != nil {
				t.Fatal(err)
			}
			inflight = inflight[1:]
			for _, f := range res.Finished {
				fin[f.ID] = true
			}
		}
		for r, wr := range w.Ranks {
			got := scheduled[r]
			if (len(got) > 0 || len(wr.Scheduled) > 0) && !reflect.DeepEqual(got, wr.Scheduled) {
				return fmt.Sprintf("step %d, rank %d, scheduled: vLLM %v, blis-vllm %v", step, r, wr.Scheduled, got)
			}
			for _, id := range wr.Finished {
				if !fin[id] {
					return fmt.Sprintf("step %d, rank %d: vLLM finished %s, blis-vllm did not", step, r, id)
				}
			}
			if rs := g.pendingOrLast(r); rs != nil && (rs.Throttled != wr.Throttled || rs.Dummy != wr.Dummy) {
				return fmt.Sprintf("step %d, rank %d: vLLM throttled %t dummy %t, blis-vllm throttled %t dummy %t",
					step, r, wr.Throttled, wr.Dummy, rs.Throttled, rs.Dummy)
			}
			if wr.QueueLen != nil && g.ranks[r].e.core.InFlight() != *wr.QueueLen {
				return fmt.Sprintf("step %d, rank %d: vLLM's batch queue holds %d, blis-vllm's %d",
					step, r, *wr.QueueLen, g.ranks[r].e.core.InFlight())
			}
		}
	}
	if !reflect.DeepEqual(routed, want.Routed) {
		return fmt.Sprintf("routing: vLLM %v, blis-vllm %v", want.Routed, routed)
	}
	return ""
}

// The DP group keeps every law of the engine contract (blis-schemas enginetest).
func TestDPGroupConformance(t *testing.T) {
	src := "block_size: 4\nmax_num_seqs: 4\nmax_num_batched_tokens: 32\nmax_model_len: 256\nnum_gpu_blocks: 80\n" +
		"use_v2_model_runner: false\nasync_scheduling: false\npipeline_parallel_size: 1\ndata_parallel_size: 2\n"
	tr := enginetest.TestEngine(t, build(t, src), arrivals(req("a", 12, 4), req("b", 20, 6), req("c", 8, 3), req("d", 30, 2)),
		enginetest.Options{})
	for id, f := range tr.Finished {
		if f.Reason != engine.FinishStop {
			t.Errorf("%s finished %+v, want stop", id, f)
		}
	}
	if len(tr.Finished) != 4 {
		t.Errorf("finished %d of 4 requests", len(tr.Finished))
	}
}

// tokenPricer prices a batch at its token count, so the group's step shows which rank's batch it
// priced.
type tokenPricer struct{}

func (tokenPricer) StepTime(b engine.Batch) engine.Duration { return engine.Duration(b.Tokens()) }

// A lockstep iteration lasts as long as its slowest rank's forward pass: the group's all-reduce
// makes every rank wait for it (run_busy_loop, vllm/v1/engine/core.py:2328-2405@5df63a72). It is
// the maximum over ranks, never their sum or one rank's alone.
func TestDPGroupStepIsTheSlowestRank(t *testing.T) {
	src := "block_size: 4\nmax_num_seqs: 4\nmax_num_batched_tokens: 64\nmax_model_len: 256\nnum_gpu_blocks: 80\n" +
		"use_v2_model_runner: false\nasync_scheduling: false\npipeline_parallel_size: 1\ndata_parallel_size: 2\n"
	var node yaml.Node
	if err := yaml.Unmarshal([]byte(src), &node); err != nil {
		t.Fatal(err)
	}
	e, err := Factory{Pricer: func(engine.Spec) (Pricer, error) { return tokenPricer{}, nil }}.New(
		engine.Spec{Kind: Kind, UpstreamVersion: Version, Config: *node.Content[0]})
	if err != nil {
		t.Fatal(err)
	}
	g := e.(*DPGroup)
	// The balancer sends the first request to rank 0 and the second to rank 1 (its rotating
	// start), so the two ranks run prompts of 10 and 30 tokens in the same iteration.
	if err := g.Add(req("a", 10, 2), 0); err != nil {
		t.Fatal(err)
	}
	if err := g.Add(req("b", 30, 2), 0); err != nil {
		t.Fatal(err)
	}
	if g.rankOf["a"] == g.rankOf["b"] {
		t.Fatalf("both requests went to rank %d", g.rankOf["a"])
	}
	b, err := g.Schedule(0)
	if err != nil {
		t.Fatal(err)
	}
	if got := g.StepTime(b); got != 30 {
		t.Errorf("the iteration's step time is %d, want 30: the slower rank's 30-token prefill", got)
	}
}

// With a batch queue per rank (async scheduling) the DP group keeps every law of the engine
// contract (blis-schemas enginetest), two lockstep iterations' batches in flight.
func TestDPGroupBatchQueueConformance(t *testing.T) {
	src := "block_size: 4\nmax_num_seqs: 4\nmax_num_batched_tokens: 32\nmax_model_len: 256\nnum_gpu_blocks: 80\n" +
		"use_v2_model_runner: false\nasync_scheduling: true\npipeline_parallel_size: 1\ndata_parallel_size: 2\n"
	tr := enginetest.TestEngine(t, build(t, src), arrivals(req("a", 12, 4), req("b", 20, 6), req("c", 8, 3), req("d", 30, 2)),
		enginetest.Options{})
	if len(tr.Finished) != 4 {
		t.Errorf("finished %d of 4 requests", len(tr.Finished))
	}
	for id, f := range tr.Finished {
		if f.Reason != engine.FinishStop {
			t.Errorf("%s finished %+v, want stop", id, f)
		}
	}
}

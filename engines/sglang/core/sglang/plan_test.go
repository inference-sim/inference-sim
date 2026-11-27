// Original to blis-sglang.

package sglang

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"

	"github.com/inference-sim/inference-sim/engines/sglang/internal/testgz"
)

// scenarioArgs are the ServerArgs an oracle scheduler scenario states (oracle/run_sched.py
// sched_server_args): the scenario's own keys, and its server_args extras, over SGLang's defaults.
func scenarioArgs(t *testing.T, sc map[string]any) SchedArgs {
	t.Helper()
	a := SchedArgs{
		PageSize:                    int(sc["page_size"].(float64)),
		SchedulePolicy:              sc["schedule_policy"].(string),
		MaxPrefillTokens:            16384,
		ScheduleConservativeness:    1.0,
		PriorityPreemptionThreshold: 10,
		DisableOverlapSchedule:      !sc["overlap"].(bool),
		DisableRadixCache:           sc["disable_radix_cache"].(bool),
		RandomSeed:                  int(sc["seed"].(float64)),
		DisaggregationMode:          "null",
	}
	if v, ok := sc["chunked_prefill_size"].(float64); ok {
		n := int(v)
		a.ChunkedPrefillSize = &n
	}
	extra, _ := sc["server_args"].(map[string]any)
	for k, v := range extra {
		switch k {
		case "enable_mixed_chunk":
			a.EnableMixedChunk = v.(bool)
		case "enable_priority_scheduling":
			a.EnablePriorityScheduling = v.(bool)
		case "disable_priority_preemption":
			a.DisablePriorityPreemption = v.(bool)
		case "priority_scheduling_preemption_threshold":
			a.PriorityPreemptionThreshold = int(v.(float64))
		case "schedule_low_priority_values_first":
			a.ScheduleLowPriorityValuesFirst = v.(bool)
		case "abort_on_priority_when_disabled":
			a.AbortOnPriorityWhenDisabled = v.(bool)
		case "max_queued_requests":
			n := int(v.(float64))
			a.MaxQueuedRequests = &n
		case "prefill_max_requests":
			n := int(v.(float64))
			a.PrefillMaxRequests = &n
		case "retraction_policy":
			a.RetractionPolicy = v.(string)
		case "allow_auto_truncate":
			a.AllowAutoTruncate = v.(bool)
		case "max_prefill_tokens":
			a.MaxPrefillTokens = int(v.(float64))
		default:
			t.Fatalf("scenario sets server_arg %q, which the derivation test does not map", k)
		}
	}
	return a
}

func readJSON(t *testing.T, path string) map[string]any {
	t.Helper()
	b, err := testgz.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	return m
}

// TestSchedConfigEqualsWhatSGLangsSchedulerResolved derives the scheduler's settings from each
// oracle scenario's ServerArgs and requires every setting SGLang's real Scheduler recorded in the
// scenario's golden trace (oracle/run_sched.py config_of) to be the one derived. The pool sizes
// and the model's facts are the runtime's, taken from the same recording.
func TestSchedConfigEqualsWhatSGLangsSchedulerResolved(t *testing.T) {
	traces, err := testgz.Glob("../sched/testdata/golden-sched/*.trace.json")
	if err != nil || len(traces) == 0 {
		t.Fatalf("no golden scheduler traces: %v", err)
	}
	for _, tr := range traces {
		name := strings.TrimSuffix(filepath.Base(tr), ".trace.json")
		t.Run(name, func(t *testing.T) {
			sc := readJSON(t, filepath.Join("../../oracle/scenarios-sched", name+".json"))
			rec := readJSON(t, tr)
			want := rec["config"].(map[string]any)

			var eos []int64
			for _, v := range want["eos_token_ids"].([]any) {
				eos = append(eos, int64(v.(float64)))
			}
			rt := Runtime{
				MaxTotalNumTokens:  int(want["max_total_num_tokens"].(float64)),
				ReqPoolSize:        int(want["req_pool_size"].(float64)),
				MaxRunningRequests: int(want["max_running_requests"].(float64)),
				ContextLen:         int(want["context_len"].(float64)),
				EOSTokenIDs:        eos,
				VocabSize:          int(want["vocab_size"].(float64)),
			}
			env := DefaultEnv()
			if e, ok := rec["sglang_env"].(map[string]any); ok && e["SGLANG_DISABLE_OVERLAP_AFTER_PREFILL"] != nil {
				env.DisableOverlapAfterPrefill = true
			}
			got, err := SchedConfig(scenarioArgs(t, sc), rt, env)
			if err != nil {
				t.Fatal(err)
			}
			b, _ := json.Marshal(got)
			var gotm map[string]any
			_ = json.Unmarshal(b, &gotm)
			for k, w := range want {
				if g := gotm[k]; !reflect.DeepEqual(g, w) {
					t.Errorf("%s: derived %v, SGLang's Scheduler resolved %v", k, g, w)
				}
			}
		})
	}
}

// The oracle's scenarios do not reach the cases below (a conservativeness above 1, pipeline
// parallelism, a disabled chunk size with mixed chunking, a disaggregated pool), so these are
// computed from SGLang's source by hand, each with the line it follows. Recording them with
// SGLang's Scheduler is the follow-up in the pull request.
func TestSchedConfigFollowsSGLangsArithmeticWhereNoScenarioReachesIt(t *testing.T) {
	base := func() (SchedArgs, Runtime) {
		return SchedArgs{PageSize: 1, SchedulePolicy: "fcfs", MaxPrefillTokens: 16384, ScheduleConservativeness: 1.0, DisaggregationMode: "null"},
			Runtime{MaxTotalNumTokens: 4096, ReqPoolSize: 9, MaxRunningRequests: 9, ContextLen: 8192, EOSTokenIDs: []int64{1}, VocabSize: 10}
	}
	t.Run("a conservativeness above one caps the initial new-token ratio at one (new_token_ratio_tracker.py:21-31)", func(t *testing.T) {
		a, rt := base()
		a.ScheduleConservativeness = 2.0
		c, err := SchedConfig(a, rt, DefaultEnv())
		if err != nil {
			t.Fatal(err)
		}
		if c.NewTokenRatio.Init != 1.0 || c.NewTokenRatio.Min != 0.14 {
			t.Errorf("init %v min %v, want 1 and 0.14", c.NewTokenRatio.Init, c.NewTokenRatio.Min)
		}
	})
	t.Run("a factor above one caps the minimum ratio at one", func(t *testing.T) {
		a, rt := base()
		env := DefaultEnv()
		env.MinNewTokenRatioFactor = 3
		c, _ := SchedConfig(a, rt, env)
		if c.NewTokenRatio.Min != 1.0 {
			t.Errorf("min %v, want 1", c.NewTokenRatio.Min)
		}
	})
	t.Run("the prefill-decode interval and its minimum batch size are carried to the scheduler (scheduler.py:1353-1357)", func(t *testing.T) {
		a, rt := base()
		four := 4
		a.PrefillDecodeInterval, a.PrefillDecodeIntervalMinBs = &four, 3
		c, err := SchedConfig(a, rt, DefaultEnv())
		if err != nil {
			t.Fatal(err)
		}
		if c.PrefillDecodeInterval != 4 || c.PrefillDecodeIntervalMinBs != 3 {
			t.Errorf("interval %d min_bs %d, want 4 and 3", c.PrefillDecodeInterval, c.PrefillDecodeIntervalMinBs)
		}
	})
	t.Run("the micro-batch is the running requests over the pipeline size, at least one (scheduler.py:1237-1244)", func(t *testing.T) {
		a, rt := base()
		a.PPSize = 2
		c, _ := SchedConfig(a, rt, DefaultEnv())
		if c.PPMaxMicroBatchSize != 4 {
			t.Errorf("micro-batch %d, want 9/2 = 4", c.PPMaxMicroBatchSize)
		}
		rt.MaxRunningRequests = 1
		if c, _ = SchedConfig(a, rt, DefaultEnv()); c.PPMaxMicroBatchSize != 1 {
			t.Errorf("micro-batch %d, want 1", c.PPMaxMicroBatchSize)
		}
		five := 5
		a.PPMaxMicroBatchSize = &five
		if c, _ = SchedConfig(a, rt, DefaultEnv()); c.PPMaxMicroBatchSize != 5 {
			t.Errorf("micro-batch %d, want the stated 5", c.PPMaxMicroBatchSize)
		}
	})
	t.Run("mixed chunking needs chunked prefill: a chunk size of zero or less turns both off (scheduler.py:1372-1378)", func(t *testing.T) {
		a, rt := base()
		a.EnableMixedChunk = true
		minus := -1
		a.ChunkedPrefillSize = &minus
		c, _ := SchedConfig(a, rt, DefaultEnv())
		if c.ChunkedPrefillSize != nil || c.IsMixedChunk {
			t.Errorf("chunk %v mixed %v, want neither", c.ChunkedPrefillSize, c.IsMixedChunk)
		}
	})
	t.Run("a prefill or decode pool sorts its free pages (kv_cache_configurator.py:932)", func(t *testing.T) {
		for mode, want := range map[string]bool{"null": false, "prefill": true, "decode": true} {
			a, rt := base()
			a.DisaggregationMode = mode
			if c, _ := SchedConfig(a, rt, DefaultEnv()); c.NeedSort != want {
				t.Errorf("mode %s: NeedSort %v, want %v", mode, c.NeedSort, want)
			}
		}
	})
	t.Run("SGLang refuses shortest-prefill-first without a radix cache", func(t *testing.T) {
		a, rt := base()
		a.SchedulePolicy, a.DisableRadixCache = "shortest-prefill-first", true
		if _, err := SchedConfig(a, rt, DefaultEnv()); err == nil {
			t.Error("want a refusal")
		}
	})
}

// memKernel is a cost model with fixed resident bytes; PoolPlan reads nothing else of it.
type memKernel struct {
	kernel.Kernel
	weights, perToken int64
}

func (m memKernel) FixedBytes() kernel.MemoryBreakdown {
	return kernel.MemoryBreakdown{Weights: m.weights}
}
func (m memKernel) SequenceVariableBytes(tokens int) int64 { return m.perToken * int64(tokens) }

func TestPoolPlanSizesThePoolFromTheMemoryTheWeightsLeave(t *testing.T) {
	const gib = int64(1 << 30)
	k := memKernel{weights: 40 * gib, perToken: 1 << 20} // 40 GiB of weights, 1 MiB per token
	in := PoolInput{PreModelLoadGiB: 80, MemFractionStatic: 0.9, PageSize: 1, ContextLen: 8192, AttnDPSize: 1}
	// SGLang keeps 80*(1-0.9) = 8 GiB as slack: 80-40-8 = 32 GiB of KV = 32768 tokens.
	rt, err := PoolPlan(k, in)
	if err != nil {
		t.Fatal(err)
	}
	if rt.MaxTotalNumTokens != 32768 {
		t.Errorf("tokens %d, want 32768", rt.MaxTotalNumTokens)
	}
	// resolve_max_num_reqs: int(32768/8192*512) = 2048, floor 2048, at most tokens/2.
	if rt.ReqPoolSize != 2048 || rt.MaxRunningRequests != 2048 {
		t.Errorf("request pool %d, running %d, want 2048", rt.ReqPoolSize, rt.MaxRunningRequests)
	}
	in.PageSize = 64
	in.MaxTotalTokens = new(int)
	*in.MaxTotalTokens = 1000
	if rt, _ = PoolPlan(k, in); rt.MaxTotalNumTokens != 960 {
		t.Errorf("--max-total-tokens 1000 at page 64: %d tokens, want 960", rt.MaxTotalNumTokens)
	}
	in.MaxTotalTokens = nil
	mrr := 16
	in.MaxRunningRequests = &mrr
	if rt, _ = PoolPlan(k, in); rt.ReqPoolSize != 16 {
		t.Errorf("--max-running-requests 16: pool %d", rt.ReqPoolSize)
	}
	if _, err = PoolPlan(memKernel{weights: 75 * gib, perToken: 1 << 20}, in); err == nil {
		t.Error("weights that leave nothing for the KV cache must be refused")
	}
	in.AttnDPSize = 8
	if _, err = PoolPlan(k, in); err == nil {
		t.Error("DP attention must be refused by name")
	}
}

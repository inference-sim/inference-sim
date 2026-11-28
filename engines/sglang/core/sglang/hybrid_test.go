// Original to blis-sglang.

package sglang

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"

	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
	"github.com/inference-sim/inference-sim/engines/sglang/core/sched"
	"github.com/inference-sim/inference-sim/engines/sglang/internal/testgz"
)

// The hybrid model's pools as the engine builds them (inference-sim/inference-sim#2366): Build makes
// the request pool that holds the state slots and the radix cache that keeps states in them from the
// scheduler configuration alone, and the configuration is the one SGLang's Scheduler resolved.

const hybridGoldens = "../sched/testdata/golden-sched-mamba/*.trace.json"

type hybridTrace struct {
	Config sched.Config `json:"config"`
	Steps  []struct {
		Forwards []sched.Forward `json:"forwards"`
		After    *sched.State    `json:"after"`
	} `json:"steps"`
}

// hybridArgs are the ServerArgs of an oracle hybrid scenario.
func hybridArgs(t *testing.T, sc map[string]any, level, interval int) SchedArgs {
	t.Helper()
	a := scenarioArgs(t, withoutKeys(sc, "max_mamba_cache_size", "mamba_radix_cache_strategy", "mamba_track_interval",
		"hybrid_cache_checkpoints_level", "enable_mixed_chunk"))
	extra, _ := sc["server_args"].(map[string]any)
	a.MambaStrategy = extra["mamba_radix_cache_strategy"].(string)
	a.HybridCacheCheckpointsLevel = level
	a.MambaTrackInterval = interval
	a.EnableMixedChunk, _ = extra["enable_mixed_chunk"].(bool)
	return a
}

func withoutKeys(sc map[string]any, keys ...string) map[string]any {
	out := map[string]any{}
	for k, v := range sc {
		out[k] = v
	}
	extra := map[string]any{}
	for k, v := range sc["server_args"].(map[string]any) {
		extra[k] = v
	}
	for _, k := range keys {
		delete(extra, k)
	}
	out["server_args"] = extra
	return out
}

func TestSchedConfigEqualsWhatSGLangsSchedulerResolvedForAHybridModel(t *testing.T) {
	traces, err := testgz.Glob(hybridGoldens)
	if err != nil || len(traces) == 0 {
		t.Fatalf("no hybrid scheduler traces: %v", err)
	}
	for _, tr := range traces {
		name := strings.TrimSuffix(filepath.Base(tr), ".trace.json")
		t.Run(name, func(t *testing.T) {
			sc := readJSON(t, filepath.Join("../../oracle/scenarios-sched-mamba", name+".json"))
			var rec hybridTrace
			raw, _ := testgz.ReadFile(tr)
			if err := json.Unmarshal(raw, &rec); err != nil {
				t.Fatal(err)
			}
			w := rec.Config.Mamba
			extra := sc["server_args"].(map[string]any)
			interval := 256
			if v, ok := extra["mamba_track_interval"].(float64); ok {
				interval = int(v)
			}
			level := 2
			if v, ok := extra["hybrid_cache_checkpoints_level"].(float64); ok {
				level = int(v)
			}
			rt := Runtime{MaxTotalNumTokens: rec.Config.MaxTotalNumTokens, ReqPoolSize: rec.Config.ReqPoolSize,
				MaxRunningRequests: rec.Config.MaxRunningRequests, ContextLen: rec.Config.ContextLen,
				EOSTokenIDs: rec.Config.EOSTokenIDs, VocabSize: rec.Config.VocabSize, MambaSize: w.MambaSize}
			got, err := SchedConfig(hybridArgs(t, sc, level, interval), rt, DefaultEnv())
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got.Mamba, w) {
				t.Errorf("derived %+v, SGLang's Scheduler resolved %+v", got.Mamba, w)
			}
		})
	}
}

// Build, not the test's own wiring, makes the scheduler a hybrid trace records: every forward pass and
// every state the loop leaves equal SGLang's, with the tracking arrays and state slots in them.
func TestBuildMakesTheHybridSchedulerSGLangRecorded(t *testing.T) {
	traces, err := testgz.Glob(hybridGoldens)
	if err != nil || len(traces) == 0 {
		t.Fatalf("no hybrid scheduler traces: %v", err)
	}
	for _, tr := range traces {
		name := strings.TrimSuffix(filepath.Base(tr), ".trace.json")
		t.Run(name, func(t *testing.T) {
			var rec hybridTrace
			raw, _ := testgz.ReadFile(tr)
			if err := json.Unmarshal(raw, &rec); err != nil {
				t.Fatal(err)
			}
			var sc struct {
				Sampled  int64 `json:"sampled_token"`
				Requests []struct {
					RID         string  `json:"rid"`
					ArrivalStep int     `json:"arrival_step"`
					Prompt      []int64 `json:"prompt"`
					MaxNew      int     `json:"max_new_tokens"`
					IgnoreEOS   *bool   `json:"ignore_eos"`
				} `json:"requests"`
			}
			raw, _ = testgz.ReadFile(filepath.Join("../../oracle/scenarios-sched-mamba", name+".json"))
			if err := json.Unmarshal(raw, &sc); err != nil {
				t.Fatal(err)
			}
			s, err := Build(rec.Config, sc.Sampled)
			if err != nil {
				t.Fatal(err)
			}
			for i, want := range rec.Steps {
				var arr []sched.Arrival
				for _, r := range sc.Requests {
					if r.ArrivalStep == i {
						ignore := r.IgnoreEOS == nil || *r.IgnoreEOS
						arr = append(arr, sched.Arrival{RID: r.RID, Prompt: r.Prompt, MaxNewTokens: r.MaxNew, IgnoreEOS: ignore})
					}
				}
				got, err := s.Step(arr)
				if err != nil {
					t.Fatalf("step %d: %v", i, err)
				}
				if len(want.Forwards) != len(got.Forwards) {
					t.Fatalf("step %d: SGLang launched %d passes, the port %d", i, len(want.Forwards), len(got.Forwards))
				}
				for k := range want.Forwards {
					if w, g := asJSON(want.Forwards[k]), asJSON(got.Forwards[k]); w != g {
						t.Fatalf("step %d pass %d:\nSGLang %s\nport   %s", i, k, w, g)
					}
				}
				if want.After == nil {
					continue
				}
				if w, g := asJSON(*want.After), asJSON(got.After); w != g {
					t.Fatalf("step %d state:\nSGLang %s\nport   %s", i, w, g)
				}
			}
		})
	}
}

func asJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// hybridKernel is a cost model with a recurrent state per request: the full-attention layers' KV per
// token and the bytes of the states a request holds.
type hybridKernel struct {
	kernel.Kernel
	weights, perToken, fixedPerSeq int64
}

func (m hybridKernel) FixedBytes() kernel.MemoryBreakdown {
	return kernel.MemoryBreakdown{Weights: m.weights}
}
func (m hybridKernel) SequenceVariableBytes(tokens int) int64 { return m.perToken * int64(tokens) }
func (m hybridKernel) SequenceFixedBytes() int64              { return m.fixedPerSeq }

// A hybrid model's state pool takes its share of the cache budget, the full-attention KV pool keeps
// the rest, and a running request is charged the state slots its strategy holds, so the request pool
// is what the state slots over that charge allow (_handle_max_mamba_cache, resolve_max_num_reqs).
func TestHybridPoolPlanSharesTheBudgetAndCapsTheRequests(t *testing.T) {
	const gib = int64(1 << 30)
	// 32 GiB of cache budget (80 GiB device, 40 GiB of weights, 8 GiB slack), 1 MiB per KV token, and 64
	// MiB of state per slot; extra_buffer with overlap holds the working slot and 2 tracking slots.
	k := hybridKernel{weights: 40 * gib, perToken: 1 << 20, fixedPerSeq: 3 * 64 << 20}
	in := PoolInput{PreModelLoadGiB: 80, MemFractionStatic: 0.9, PageSize: 1, ContextLen: 8192, AttnDPSize: 1}
	plain, err := PoolPlan(k, in)
	if err != nil {
		t.Fatal(err)
	}
	in.Mamba = &MambaPoolInput{Strategy: cache.MambaExtraBuffer, FullMemoryRatio: 0.9}
	rt, err := PoolPlan(k, in)
	if err != nil {
		t.Fatal(err)
	}
	if rt.MambaSize <= 0 || rt.MaxTotalNumTokens >= plain.MaxTotalNumTokens {
		t.Fatalf("mamba slots %d, tokens %d against %d without a state pool: the state pool takes no share", rt.MambaSize, rt.MaxTotalNumTokens, plain.MaxTotalNumTokens)
	}
	// The budget is split 0.9 : 1.9 between state and KV: the state pool holds (size+1) slots of 64 MiB.
	state := float64(rt.MambaSize+1) * 64 / 1024
	if kv := float64(rt.MaxTotalNumTokens) / 1024; state+kv > 32.0001 || state+kv < 32-1.0/16 {
		t.Errorf("state %.3f GiB and KV %.3f GiB do not use the 32 GiB budget", state, kv)
	}
	// 3 + 2 tracking slots are charged to each running request.
	if want := min(plain.ReqPoolSize, rt.MambaSize/5); rt.ReqPoolSize != want || rt.MaxRunningRequests != want {
		t.Errorf("request pool %d, running %d; want %d (state slots %d over 5)", rt.ReqPoolSize, rt.MaxRunningRequests, want, rt.MambaSize)
	}
	// A stated state pool is taken as it is, and bounds the requests by what a request is charged.
	size := 40
	in.Mamba.MaxMambaCacheSize = &size
	if rt, err = PoolPlan(k, in); err != nil || rt.MambaSize != 40 || rt.ReqPoolSize != 8 {
		t.Errorf("--max-mamba-cache-size 40: %+v, %v; want 40 slots and 8 requests", rt, err)
	}
	// Without the overlap schedule the working slot has one tracking slot, so a request is charged 4.
	in.Mamba.DisableOverlap = true
	if rt, err = PoolPlan(k, in); err != nil || rt.ReqPoolSize != 10 {
		t.Errorf("no overlap: %+v, %v; want 10 requests", rt, err)
	}
	// A state pool the memory cannot hold is refused.
	in.Mamba.MaxMambaCacheSize = nil
	if _, err = PoolPlan(hybridKernel{weights: 75 * gib, perToken: 1 << 20, fixedPerSeq: 3 * 64 << 20}, in); err == nil {
		t.Error("weights that leave nothing for the state pool must be refused")
	}
}

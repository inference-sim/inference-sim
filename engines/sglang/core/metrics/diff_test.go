// Original to blis-sglang.

package metrics

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
	"github.com/inference-sim/inference-sim/engines/sglang/core/sched"
	"github.com/inference-sim/inference-sim/engines/sglang/internal/testgz"
)

// The metrics replay (T6, FID-4): the scenario's requests arrive on the scheduling port, and after
// every step the series of the classes the manifest calls exact equal the registry of SGLang's own
// scheduler, which oracle/run_metrics.py recorded with --enable-metrics.

const (
	goldenDir   = "testdata/golden-metrics"
	scenarioDir = "../../oracle/scenarios-metrics"
	classesPath = "../../docs/metrics-classes.json"
)

type traceStep struct {
	Metrics  [][3]json.RawMessage `json:"metrics"`
	KVEvents []string             `json:"kv_events"`
	After    json.RawMessage      `json:"after"`
}

type trace struct {
	Config sched.Config `json:"config"`
	Steps  []traceStep  `json:"steps"`
}

type scenario struct {
	SampledToken int64 `json:"sampled_token"`
	Requests     []struct {
		RID          string  `json:"rid"`
		ArrivalStep  int     `json:"arrival_step"`
		Prompt       []int64 `json:"prompt"`
		MaxNewTokens int     `json:"max_new_tokens"`
		IgnoreEOS    *bool   `json:"ignore_eos"`
		Priority     *int    `json:"priority"`
	} `json:"requests"`
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := testgz.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

func exactNames(t *testing.T) map[string]bool {
	var c struct {
		Exact []string `json:"exact_series"`
	}
	readJSON(t, classesPath, &c)
	m := map[string]bool{}
	for _, n := range c.Exact {
		m[n] = true
	}
	return m
}

func key(name string, labels map[string]string) string { return name + "{" + labelKey(labels) + "}" }

func build(cfg sched.Config, sampled int64) (*sched.Scheduler, error) {
	var alloc cache.Allocator
	if cfg.PageSize == 1 {
		alloc = cache.NewTokenAllocator(cfg.MaxTotalNumTokens, nil, cfg.NeedSort)
	} else {
		alloc = cache.NewPagedAllocator(cfg.MaxTotalNumTokens, cfg.PageSize, nil, cfg.NeedSort)
	}
	pool := cache.NewRequestPool(cfg.ReqPoolSize, cfg.MaxContextLen)
	policy := cfg.RadixEvictionPolicy
	if policy == "" {
		policy = "lru"
	}
	tree, err := cache.NewRadixCache(cache.RadixConfig{PageSize: cfg.PageSize, Disable: cfg.Disable, EvictionPolicy: policy, KVCacheEvents: cfg.EnableKVEvents, DynamoFormat: cfg.KVEventsDynamo}, alloc, pool)
	if err != nil {
		return nil, err
	}
	return sched.New(cfg, tree, alloc, pool, sampled)
}

func TestSeriesEqualSGLangs(t *testing.T) {
	exact := exactNames(t)
	files, _ := testgz.Glob(filepath.Join(goldenDir, "*.trace.json"))
	if len(files) == 0 {
		t.Fatalf("no trace under %s: CI records them (oracle/run_metrics.py)", goldenDir)
	}
	for _, f := range files {
		name := filepath.Base(f[:len(f)-len(".trace.json")])
		t.Run(name, func(t *testing.T) {
			var tr trace
			readJSON(t, f, &tr)
			var sc scenario
			readJSON(t, filepath.Join(scenarioDir, name+".json"), &sc)
			s, err := build(tr.Config, sc.SampledToken)
			if err != nil {
				t.Fatal(err)
			}
			col := New(tr.Config)
			compared, batches := 0, 0
			for i, step := range tr.Steps {
				var arr []sched.Arrival
				for _, r := range sc.Requests {
					if r.ArrivalStep == i {
						ignore := true
						if r.IgnoreEOS != nil {
							ignore = *r.IgnoreEOS
						}
						arr = append(arr, sched.Arrival{RID: r.RID, Prompt: r.Prompt, MaxNewTokens: r.MaxNewTokens, IgnoreEOS: ignore, Priority: r.Priority})
					}
				}
				rec, err := s.Step(arr)
				if err != nil {
					t.Fatalf("step %d: %v", i, err)
				}
				col.Observe(rec.Reports)
				var gotEvents []string
				for _, r := range rec.Reports {
					if len(r.Events) > 0 {
						gotEvents = append(gotEvents, hex.EncodeToString(EncodeKVEventBatch(0, r.Events, 0)))
					}
				}
				if step.After == nil {
					// The scheduler stopped inside this step: the runner read nothing after it.
					continue
				}
				if !reflect.DeepEqual(step.KVEvents, gotEvents) {
					t.Fatalf("step %d: KV event batches\n SGLang %v\n port   %v", i, step.KVEvents, gotEvents)
				}
				batches += len(gotEvents)
				if step.Metrics == nil {
					continue
				}
				want := map[string]float64{}
				for _, m := range step.Metrics {
					var n string
					var l map[string]string
					var v float64
					if err := errors.Join(json.Unmarshal(m[0], &n), json.Unmarshal(m[1], &l)); err != nil {
						t.Fatal(err)
					}
					if string(m[2]) == "NaN" || string(m[2]) == "null" {
						v = math.NaN()
					} else {
						if err := json.Unmarshal(m[2], &v); err != nil {
							t.Fatal(err)
						}
					}
					if exact[n] {
						want[key(n, l)] = v
					}
				}
				got := map[string]float64{}
				for _, sr := range col.Series() {
					if exact[sr.Name] {
						got[key(sr.Name, sr.Labels)] = sr.Value
					}
				}
				if d := diff(want, got); d != "" {
					t.Fatalf("step %d: %s", i, d)
				}
				compared += len(want)
			}
			if compared == 0 && batches == 0 {
				t.Fatal("the trace holds neither exact series nor KV event batches")
			}
		})
	}
}

func diff(want, got map[string]float64) string {
	var keys []string
	for k := range want {
		keys = append(keys, k)
	}
	for k := range got {
		if _, ok := want[k]; !ok {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	for _, k := range keys {
		w, wok := want[k]
		g, gok := got[k]
		switch {
		case !wok:
			return fmt.Sprintf("the port has %s = %v, SGLang has no such series", k, g)
		case !gok:
			return fmt.Sprintf("SGLang has %s = %v, the port has no such series", k, w)
		case math.Abs(w-g) > 1e-9*math.Max(1, math.Abs(w)):
			return fmt.Sprintf("%s: SGLang %v, port %v", k, w, g)
		}
	}
	return ""
}

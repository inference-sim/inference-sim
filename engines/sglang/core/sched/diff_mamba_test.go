// Original to blis-sglang.

package sched

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
	"github.com/inference-sim/inference-sim/engines/sglang/internal/testgz"
)

// The scheduler replay over a hybrid (linear-attention) model (inference-sim/inference-sim#2366):
// the same comparison as TestStepsEqualSGLangs, for the traces oracle/run_sched_mamba.py recorded of
// SGLang's own Scheduler over the configuration of a reduced Qwen3.5 hybrid model. Besides the forward
// passes, finished requests and queues, every step must equal SGLang's recurrent-state bookkeeping:
// the batch's tracking arrays (which requests leave a state snapshot, at which length, into which
// slot, and the copies of matched states), the free state slots, the tree's state ledgers, and each
// request's state slot, tracking slots and tracked lengths.

// MambaGoldenDir holds the traces, one per scenario of oracle/scenarios-sched-mamba.
const MambaGoldenDir = "testdata/golden-sched-mamba"

// MambaScenarioDir holds the scenarios they were recorded from.
const MambaScenarioDir = "../../oracle/scenarios-sched-mamba"

// buildHybrid makes the scheduler a hybrid trace's configuration describes: the request pool that
// holds the state slots, the token allocator, and the radix cache with the state component.
func buildHybrid(cfg Config, sampled int64) (*Scheduler, error) {
	m := cfg.Mamba
	pool := cache.NewHybridRequestPool(cfg.ReqPoolSize, m.MambaSize, cfg.MaxContextLen, m.ExtraBuffer, m.Lazy, cfg.EnableOverlap)
	var alloc cache.Allocator
	if cfg.PageSize == 1 {
		alloc = cache.NewTokenAllocator(cfg.MaxTotalNumTokens, nil, cfg.NeedSort)
	} else {
		alloc = cache.NewPagedAllocator(cfg.MaxTotalNumTokens, cfg.PageSize, nil, cfg.NeedSort)
	}
	tree, err := cache.NewRadixCache(cache.RadixConfig{PageSize: cfg.PageSize, Disable: cfg.Disable, EvictionPolicy: cfg.radixEvictionPolicy(),
		Mamba: &cache.MambaTreeConfig{CheckpointInterval: m.TreeSlotInterval, PathStateCap: m.PathStateCap, CheckpointsLevel: m.CheckpointsLevel, Pool: pool}},
		alloc, pool.RequestPool)
	if err != nil {
		return nil, err
	}
	return New(cfg, tree, alloc, pool, sampled)
}

func replayHybrid(sc scenario, cfg Config, steps int, tr *trace) ([]StepRecord, error) {
	s, err := buildHybrid(cfg, sc.SampledToken)
	if err != nil {
		return nil, err
	}
	return runScenario(s, sc, steps, tr)
}

func TestHybridStepsEqualSGLangs(t *testing.T) {
	files, err := testgz.Glob(filepath.Join(MambaGoldenDir, "*.trace.json"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no hybrid trace under %s: CI records them (oracle/run_sched_mamba.py): %v", MambaGoldenDir, err)
	}
	for _, path := range files {
		name := strings.TrimSuffix(filepath.Base(path), ".trace.json")
		t.Run(name, func(t *testing.T) {
			var tr trace
			readJSON(t, path, &tr)
			var sc scenario
			readJSON(t, filepath.Join(MambaScenarioDir, name+".json"), &sc)
			if tr.Config.Mamba == nil {
				t.Fatal("the trace names no mamba settings")
			}
			got, err := replayHybrid(sc, tr.Config, len(tr.Steps), &tr)
			if err != nil {
				t.Fatal(err)
			}
			forwards := 0
			for i, want := range tr.Steps {
				forwards += len(want.Forwards)
				if d := compare(want, got[i]); d != "" {
					t.Fatalf("step %d: %s", i, d)
				}
			}
			if forwards == 0 {
				t.Fatal("the trace launched no forward pass")
			}
		})
	}
}

// The reporter's gauges of the state pool are not modelled: a hybrid scheduler with metrics on is
// refused by name, never run with those gauges missing.
func TestHybridSchedulerWithMetricsIsRefusedByName(t *testing.T) {
	files, err := testgz.Glob(filepath.Join(MambaGoldenDir, "hybrid-basic.trace.json"))
	if err != nil || len(files) != 1 {
		t.Fatalf("no hybrid-basic trace: %v", err)
	}
	var tr trace
	readJSON(t, files[0], &tr)
	tr.Config.EnableMetrics = true
	_, err = buildHybrid(tr.Config, 13)
	if err == nil || !strings.Contains(err.Error(), "#2451") || !strings.Contains(err.Error(), "state pool") {
		t.Fatalf("got %v, want a refusal naming the state pool's metrics and #2451", err)
	}
}

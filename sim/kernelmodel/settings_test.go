package kernelmodel

import (
	"os"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// Laws Settings must satisfy on every committed scenario, so a simulator sized from it is
// sized from the kernel and from nothing else:
//
//   - each admission value is the pool's own (Engine), unchanged;
//   - the aggregate budget is the per-rank budget times the data-parallel width exactly when
//     the kernel's budget says it scaled, and equal to it otherwise;
//   - the per-block cost times the per-rank blocks fits in what the kernel said is allocatable.
func TestSettingsAreTheKernelsOwnAnswers(t *testing.T) {
	entries, err := os.ReadDir(DefaultScenarios())
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".yaml") {
			continue
		}
		m, err := Open(e.Name(), repos())
		if err != nil {
			t.Fatalf("Open(%s): %v", e.Name(), err)
		}
		budget, budgetErr := m.KVBudget()
		s, err := m.Settings()
		if budgetErr != nil {
			if err == nil {
				t.Errorf("%s: the kernel cannot derive a budget (%v) but Settings succeeded", e.Name(), budgetErr)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s: Settings: %v", e.Name(), err)
		}
		eng, _ := m.Engine()
		if s.BlockSize != eng.BlockSize || s.MaxNumSeqs != eng.MaxNumSeqs ||
			s.MaxNumBatchedTokens != eng.MaxNumBatchedTokens {
			t.Errorf("%s: settings %+v do not restate the pool's engine %+v", e.Name(), s, eng)
		}
		if n, ok := eng.MaxModelLen.Stated(); ok && s.MaxModelLen != n {
			t.Errorf("%s: max_model_len %d, the pool states %d", e.Name(), s.MaxModelLen, n)
		}
		if s.DataParallel != m.DataParallelWidth() {
			t.Errorf("%s: dp %d, kernel resolved %d", e.Name(), s.DataParallel, m.DataParallelWidth())
		}
		want := s.KVBlocks
		if budget.DPScaled {
			want *= int64(s.DataParallel)
		}
		if s.AggregateKVBlocks != want || s.AggregateKVBlocks != budget.TotalBlocks {
			t.Errorf("%s: aggregate %d, per-rank %d, dp %d, scaled %v, kernel total %d",
				e.Name(), s.AggregateKVBlocks, s.KVBlocks, s.DataParallel, budget.DPScaled, budget.TotalBlocks)
		}
		if s.KVBlocks*s.KVBytesPerBlock > budget.AllocatableBytes {
			t.Errorf("%s: %d blocks of %d bytes exceed the %d allocatable bytes",
				e.Name(), s.KVBlocks, s.KVBytesPerBlock, budget.AllocatableBytes)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("no scenario produced settings; the laws were checked against nothing")
	}
}

// max_model_len resolves as vLLM v0.31.0 starts the engine: a count is kept even when one rank
// cannot hold it (the caller refuses that), unstated takes the graph's window, and auto takes
// the graph's window shrunk to what one rank's KV cache holds.
func TestMaxModelLenResolvesAsTheEngineStarts(t *testing.T) {
	m := &Model{id: identity{defaultWindow: 1 << 20}}
	for _, tt := range []struct {
		name       string
		stated     deployment.MaxModelLen
		rankTokens int64
		want       int
	}{
		{"a stated count", 32768, 1000, 32768},
		{"unstated", 0, 1000, 1 << 20},
		{"auto, the cache holds less than the window", deployment.MaxModelLenAuto, 400_000, 400_000},
		{"auto, the cache holds the whole window", deployment.MaxModelLenAuto, 4 << 20, 1 << 20},
	} {
		if got, _ := m.resolvedMaxModelLen(tt.stated, tt.rankTokens); got != tt.want {
			t.Errorf("%s: max_model_len %d, want %d", tt.name, got, tt.want)
		}
	}
	if got, _ := (&Model{}).resolvedMaxModelLen(deployment.MaxModelLenAuto, 1000); got != 0 {
		t.Errorf("auto on a graph stating no window resolved to %d; it must stay unknown (0)", got)
	}
}

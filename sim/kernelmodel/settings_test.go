package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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
		window, _ := eng.MaxModelLen.Stated()
		if s.BlockSize != eng.BlockSize || s.MaxNumSeqs != eng.MaxNumSeqs ||
			s.MaxNumBatchedTokens != eng.MaxNumBatchedTokens || s.MaxModelLen != window {
			t.Errorf("%s: settings %+v do not restate the pool's engine %+v", e.Name(), s, eng)
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

// A pool that states max_model_len auto is refused by Settings, naming auto, rather than
// sized: the engine fits auto to the KV memory at startup and the kernel does not report the
// window that resolves to, so passing -1 on, or taking the kernel's upper bound, would bound
// requests by a window vLLM may not run.
func TestSettingsRefuseAnAutoWindow(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("../../testdata/scenarios", "glm-5-h200-tp8-mtp3.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	for _, spelling := range []string{"auto", "-1"} {
		body := strings.Replace(string(raw), "max_model_len: 32768", "max_model_len: "+spelling, 1)
		if body == string(raw) {
			t.Fatal("fixture states no max_model_len: 32768 to replace")
		}
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		r := repos()
		r.Scenarios = dir
		m, err := Open("variant.yaml", r)
		if err != nil {
			t.Fatalf("%s: Open: %v", spelling, err)
		}
		if e, _ := m.Engine(); !e.MaxModelLen.Auto() {
			t.Fatalf("%s: engine max_model_len %d, want auto", spelling, e.MaxModelLen)
		}
		s, err := m.Settings()
		if err == nil {
			t.Fatalf("%s: Settings sized an auto window as %d, want a refusal", spelling, s.MaxModelLen)
		}
		if !strings.Contains(err.Error(), "max_model_len auto") {
			t.Errorf("%s: refusal %q does not name max_model_len auto", spelling, err)
		}
	}
}

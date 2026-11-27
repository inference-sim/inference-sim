// Original to blis-vllm.

package offloading

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/engines/vllm/internal/testgz"
)

// The Mamba oracle is vLLM v0.34.0's own Scheduler and OffloadingConnectorScheduler over KV cache
// groups that include Mamba align state (oracle/run_offloading_mamba.py); the other offloading
// goldens are v0.33.0's (oracle/run_offloading.py). Each trace names the checkout it ran against.
const (
	mambaOracleVersion = "0.34.0"
	mambaOracleCommit  = "62d4bc3c7e33b1b6c906e233e6af2d3506c482ca"
)

// TestOffloadingBesideMambaAgainstVLLM replays every scenario under oracle/scenarios/offloading-mamba
// and compares each step (jobs, GPU blocks per group, CPU chunks in LRU order) with the trace the
// pinned vLLM produced.
func TestOffloadingBesideMambaAgainstVLLM(t *testing.T) {
	scenarios, err := filepath.Glob("../../../oracle/scenarios/offloading-mamba/offloading-mamba-*.json")
	if err != nil || len(scenarios) == 0 {
		t.Fatalf("no Mamba offloading scenarios (%v)", err)
	}
	for _, path := range scenarios {
		name := strings.TrimSuffix(filepath.Base(path), ".json")
		t.Run(name, func(t *testing.T) {
			var sc scenario
			readJSON(t, path, &sc)
			var want trace
			readJSON(t, filepath.Join("testdata", "golden-mamba", name+".json"), &want)
			if v := strings.SplitN(want.Provenance.VLLMVersion, "+", 2)[0]; v != mambaOracleVersion || want.Provenance.VLLMCommit != mambaOracleCommit {
				t.Fatalf("trace records vLLM %s at %s, want %s at %s", v, want.Provenance.VLLMCommit, mambaOracleVersion, mambaOracleCommit)
			}
			if d := earliestDivergence(want, run(t, sc)); d != "" {
				t.Fatalf("earliest divergence from vLLM v%s: %s", mambaOracleVersion, d)
			}
		})
	}
}

// TestOffloadingBesideMambaFuzzCorpusAgainstVLLM is the seeded fuzz test:
// BLIS_VLLM_OFFLOAD_MAMBA_FUZZ_DIR names a directory of oracle/gen_offloading_mamba_fuzz.py
// scenarios with vLLM's trace for each (<name>.trace.json).
func TestOffloadingBesideMambaFuzzCorpusAgainstVLLM(t *testing.T) {
	dir := os.Getenv("BLIS_VLLM_OFFLOAD_MAMBA_FUZZ_DIR")
	if dir == "" {
		t.Skip("BLIS_VLLM_OFFLOAD_MAMBA_FUZZ_DIR is unset: the Mamba offloading fuzz corpus was NOT checked")
	}
	traces, err := testgz.Glob(filepath.Join(dir, "*.trace.json"))
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
		if d := earliestDivergence(want, run(t, sc)); d != "" {
			mismatches++
			if mismatches <= 20 {
				t.Errorf("%s: earliest divergence from vLLM v%s: %s", name, mambaOracleVersion, d)
			}
		}
	}
	t.Logf("Mamba offloading fuzz corpus: %d scenarios, %d mismatches", len(traces), mismatches)
	if mismatches > 0 {
		t.Fatalf("%d of %d fuzz scenarios differ from vLLM", mismatches, len(traces))
	}
}

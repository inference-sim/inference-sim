// Original to blis-vllm.

package kernelmodel

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// A stated cudagraph_mode that the kernel prices as another one is a substitution in the pool's provenance,
// which the grade reads: the DeepSeek-V4.1 DEP8 scenario states FULL_AND_PIECEWISE and a DeepEP
// high-throughput all-to-all at dp 8, a layout that cannot be captured, so the pool is priced with no graphs.
// Where no override applies, or the mode is unstated, nothing is listed.
func TestAStatedCUDAGraphModePricedAsAnotherIsASubstitution(t *testing.T) {
	const scenario = "deepseek-v4.1-flash-h100-2n-dep8-dspark5.yaml"
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "scenarios", scenario))
	if err != nil {
		t.Fatal(err)
	}
	// The scenario states the mode either at the engine block's top level or inside compilation_config
	// (the block takes vLLM's names from the last pull request of the stack), and the all-to-all by its
	// name; the test removes the mode wherever it is and swaps the all-to-all.
	const a2a = "deepep_high_throughput"
	modeLine := regexp.MustCompile(`(?m)^[ \t]*(compilation_config: \{)?cudagraph_mode: FULL_AND_PIECEWISE\}?\n`)
	if !modeLine.Match(raw) || !strings.Contains(string(raw), a2a) {
		t.Fatalf("%s no longer states the mode and the all-to-all this test edits", scenario)
	}
	open := func(t *testing.T, body string) *Model {
		t.Helper()
		r := repos()
		r.Scenarios = t.TempDir()
		if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		m, err := Open("v.yaml", r)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	cudagraph := func(m *Model) (out []string) {
		for _, s := range m.Provenance().Substitutions {
			if s.Setting == "compilation_config.cudagraph_mode" {
				out = append(out, s.Stated+" -> "+s.Priced)
			}
		}
		return
	}
	if got := cudagraph(open(t, string(raw))); len(got) != 1 || got[0] != "FULL_AND_PIECEWISE -> NONE" {
		t.Errorf("a stated mode the layout cannot capture: substitutions %q, want one from FULL_AND_PIECEWISE to NONE", got)
	}
	if got := cudagraph(open(t, strings.Replace(string(raw), a2a, "allgather_reducescatter", 1))); len(got) != 0 {
		t.Errorf("a layout that can be captured lists %q", got)
	}
	if got := cudagraph(open(t, modeLine.ReplaceAllString(string(raw), ""))); len(got) != 0 {
		t.Errorf("an unstated mode that the layout turns off lists %q; it is the engine's own default resolving", got)
	}
}

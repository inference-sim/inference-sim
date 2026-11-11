// Original to blis-vllm.

package kernelmodel

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// openWithEngineLines opens a fixture with its own cudagraph_mode removed and lines put before its
// engine block's block_size key.
func openWithEngineLines(t *testing.T, path string, lines ...string) *Model {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", path))
	if err != nil {
		t.Fatal(err)
	}
	body := regexp.MustCompile(`(?m)^\s*cudagraph_mode:.*\n`).ReplaceAllString(string(raw), "")
	re := regexp.MustCompile(`(?m)^(\s*)block_size:`)
	if !re.MatchString(body) {
		t.Fatalf("%s states no block_size line to put the keys before", path)
	}
	body = re.ReplaceAllString(body, "${1}"+strings.Join(lines, "\n${1}")+"\n${1}block_size:")
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

// A deployment that states cudagraph_mode or the capture sizes is priced under the kernel's
// vLLM 0.31.0 policy, which agrees with 0.32.0's on the default, on the two forced-off cases and
// on the sizes, and differs where 0.32.0 lowers a mode against the attention backend. The
// difference is disclosed as an assumption that names the field, the stated value and
// blis-vllm#78, for the point's substitution list; unstated and NONE disclose nothing.
func TestStatedCUDAGraphSettingsDiscloseTheKernelsPolicy(t *testing.T) {
	const fixture = "kerneltests/testdata/aisimulate/llama-3.1-70b-instruct-h200-fp8-vllm-tp2.yaml"
	mine := func(m *Model) (out []string) {
		for _, a := range m.EngineAssumptions() {
			if strings.HasPrefix(a, "compilation_config.cudagraph_") {
				out = append(out, a)
			}
		}
		return
	}
	if got := mine(openWithEngineLines(t, fixture)); len(got) != 0 {
		t.Errorf("a deployment that states neither discloses %v", got)
	}
	if got := mine(openWithEngineLines(t, fixture, "cudagraph_mode: NONE")); len(got) != 0 {
		t.Errorf("cudagraph_mode NONE discloses %v; v0.32.0 keeps NONE", got)
	}
	got := mine(openWithEngineLines(t, fixture, "cudagraph_mode: FULL", "cudagraph_capture_sizes: [1, 2, 4, 8]"))
	if len(got) != 2 || !strings.HasPrefix(got[0], "compilation_config.cudagraph_mode = FULL: ") ||
		!strings.HasPrefix(got[1], "compilation_config.cudagraph_capture_sizes = [1 2 4 8]: ") {
		t.Fatalf("a deployment that states both discloses %q, want one assumption for each, naming the value", got)
	}
	for _, a := range got {
		if !strings.Contains(a, "blis-vllm#78") || !strings.Contains(a, "0.31.0") || !strings.Contains(a, "0.32.0") {
			t.Errorf("the assumption %q does not name the policies and blis-vllm#78", a)
		}
	}
}

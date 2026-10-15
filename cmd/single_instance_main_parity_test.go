package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestSingleInstance_MatchesMainGolden: a single-instance run behind the embedded llm-d router
// prints, and writes as per-request metrics, exactly what the main branch's native-routing
// binary did for the same flags (cmd/testdata/golden, produced by `blis run` built from main at
// 0d092d4e). The router can only pick the one instance, and with the default configuration it
// admits every request on arrival, so nothing a user reads may change.
//
// Regenerate (only when main's own single-instance output changes, e.g. a kernel release):
//
//	blis run --scenario <s> --scenarios <kernel testdata/aisimulate> --catalog testdata/catalog \
//	  --registry testdata/registry --defaults-filepath defaults.yaml --num-instances 1 --rate 10 \
//	  --num-requests 200 --seed 42 --metrics-path cmd/testdata/golden/main-single-<s>.metrics.json \
//	  > cmd/testdata/golden/main-single-<s>.stdout
func TestSingleInstance_MatchesMainGolden(t *testing.T) {
	for _, scenario := range []string{
		"llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml",
		"gpt-oss-120b-h200-fp4-vllm-tp4.yaml",
	} {
		t.Run(scenario, func(t *testing.T) {
			base := filepath.Join("testdata", "golden", "main-single-"+strings.TrimSuffix(scenario, ".yaml"))
			wantStdout, err := os.ReadFile(base + ".stdout")
			if err != nil {
				t.Fatal(err)
			}
			wantMetrics, err := os.ReadFile(base + ".metrics.json")
			if err != nil {
				t.Fatal(err)
			}
			metrics := filepath.Join(t.TempDir(), "metrics.json")
			stdout, stderr, err := runKernelCLI(t, "run", "--scenario", scenario, "--num-instances", "1",
				"--rate", "10", "--num-requests", "200", "--seed", "42", "--metrics-path", metrics)
			if err != nil {
				t.Fatalf("run: %v\n%s", err, lastLines(stderr, 10))
			}
			if stdout != string(wantStdout) {
				t.Errorf("stdout differs from main's\n--- main\n%s\n--- this branch\n%s", wantStdout, stdout)
			}
			got, err := os.ReadFile(metrics)
			if err != nil {
				t.Fatal(err)
			}
			// The provenance block records the catalog path as given; the golden was written with
			// the repository-relative one.
			_, catalog, _ := kernelRepos(t)
			got = []byte(strings.ReplaceAll(string(got), `"path": "`+catalog+`"`, `"path": "testdata/catalog"`))
			if string(got) != string(wantMetrics) {
				t.Errorf("per-request metrics differ from main's (%s)", base+".metrics.json")
			}
		})
	}
}

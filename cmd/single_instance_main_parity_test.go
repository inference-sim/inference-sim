package cmd

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// TestSingleInstance_MatchesMainGolden: a single-instance run behind the embedded llm-d router
// prints, and writes as per-request metrics, what the main branch's native-routing binary did for
// the same flags: the same text, with every number equal to the last few ULPs (sameOutput; the
// goldens were recorded on amd64, and an arm64 build rounds some percentiles in the last digit) (cmd/testdata/golden, produced by `blis run` built from main at
// 63ffc020 with this branch's blis-latency-kernel v0.2.0 / blis-schemas / testdata bump applied
// to it as a local throwaway commit (main itself still pins v0.1.0), so the vendored catalog is a
// clean tree and records "dirty": false as a CI checkout does). The router can only pick the one instance, and with the default configuration it
// admits every request on arrival, so nothing a user reads may change.
//
// Regenerate with `scripts/regen-main-single-goldens.sh 63ffc020` (only when main's own
// single-instance output changes, e.g. a kernel release): it applies the bump to that commit itself
// and reproduces these files byte for byte except the masked catalog "revision". By hand:
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
			if why := sameOutput(string(wantStdout), stdout); why != "" {
				t.Errorf("stdout differs from main's: %s\n--- main\n%s\n--- this branch\n%s", why, wantStdout, stdout)
			}
			got, err := os.ReadFile(metrics)
			if err != nil {
				t.Fatal(err)
			}
			// The catalog provenance block records the catalog's path as given and its clone's git
			// revision; the vendored catalog lives in this repository, so its revision is this
			// repository's commit. Neither is a simulation result.
			_, catalog, _ := kernelRepos(t)
			got = []byte(strings.ReplaceAll(string(got), `"path": "`+catalog+`"`, `"path": "testdata/catalog"`))
			got, wantMetrics = revisionRE.ReplaceAll(got, []byte(`"revision": "-"`)), revisionRE.ReplaceAll(wantMetrics, []byte(`"revision": "-"`))
			if why := sameOutput(string(wantMetrics), string(got)); why != "" {
				t.Errorf("per-request metrics differ from main's (%s): %s", base+".metrics.json", why)
			}
		})
	}
}

// revisionRE matches a git revision in a provenance block.
var revisionRE = regexp.MustCompile(`"revision": "[0-9a-f]{40}"`)

// numberRE matches a JSON or plain-text number.
var numberRE = regexp.MustCompile(`-?\d+(\.\d+)?([eE][-+]?\d+)?`)

// sameOutput reports "" when got is want with every number equal within a relative 1e-9 (floating
// point formatted on another architecture), and otherwise the first difference.
func sameOutput(want, got string) string {
	wantText, gotText := numberRE.ReplaceAllString(want, "#"), numberRE.ReplaceAllString(got, "#")
	if wantText != gotText {
		return "the text around the numbers differs"
	}
	wn, gn := numberRE.FindAllString(want, -1), numberRE.FindAllString(got, -1)
	for i := range wn {
		a, errA := strconv.ParseFloat(wn[i], 64)
		b, errB := strconv.ParseFloat(gn[i], 64)
		if errA != nil || errB != nil {
			if wn[i] != gn[i] {
				return fmt.Sprintf("number %d: %s vs %s", i, wn[i], gn[i])
			}
			continue
		}
		if math.Abs(a-b) > 1e-9*math.Max(math.Abs(a), math.Abs(b)) {
			return fmt.Sprintf("number %d: %s vs %s", i, wn[i], gn[i])
		}
	}
	return ""
}

func TestSameOutput(t *testing.T) {
	if why := sameOutput(`{"p99": 24682.288969999983, "n": 3}`, `{"p99": 24682.288969999998, "n": 3}`); why != "" {
		t.Errorf("a last-ULP difference must compare equal: %s", why)
	}
	for _, got := range []string{`{"p99": 24682.29, "n": 3}`, `{"p99": 24682.288969999983, "n": 4}`, `{"p98": 24682.288969999983, "n": 3}`} {
		if sameOutput(`{"p99": 24682.288969999983, "n": 3}`, got) == "" {
			t.Errorf("%s must differ", got)
		}
	}
}

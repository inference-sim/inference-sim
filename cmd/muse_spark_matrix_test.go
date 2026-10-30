package cmd

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// Muse-Spark-Open-816B-A42B (NVFP4) runs end to end on every layout its vLLM recipe and internal
// deployment measurements serve it with, on the three clusters it is served on: H100 over RoCE
// (two 8-GPU nodes), H200 over InfiniBand, and GB200 in one NVL72 rack (four-GPU trays, the
// rack's NVLink switch fabric). Each scenario in testdata/scenarios/muse-spark is the recipe's
// strategy (TP, TP + EP, DP + EP, TP16, TP16 + EP, PP2 x TP8) with and without its DSpark drafter,
// or an arm of the measured campaign (B: PP2 x TP8 at 256k; C: B with DSpark; B-1M: B at the
// 1,048,576-token window; W: TP16 x EP16; D: C with an fp8 KV cache). Each must complete every
// request, and the three clusters must price the same layout apart: the RoCE H100, InfiniBand
// H200 and NVLink GB200 pools differ by interconnect, and a nominal-bandwidth model priced them
// alike.
//
// What vLLM refuses, the scenario set states as such: H200 DP8 x EP8 with DSpark at 262,144
// tokens leaves a rank fewer blocks than one full-context request's full-attention group, and
// vLLM refuses it at start-up; the 131,072-token arm runs.
func TestMuseSparkRunsOnEveryLayoutAndCluster(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the binary and runs 36 scenarios")
	}
	blis := builtBlis(t)
	_, catalog, registry := kernelRepos(t)
	dir, err := filepath.Abs(filepath.Join("..", "testdata", "scenarios", "muse-spark"))
	if err != nil {
		t.Fatal(err)
	}
	files, err := filepath.Glob(filepath.Join(dir, "*.yaml"))
	if err != nil || len(files) < 30 {
		t.Fatalf("found %d Muse scenarios (%v); the matrix is 36", len(files), err)
	}
	sort.Strings(files)
	refused := map[string]string{
		"muse-h200-ib-dep8-dspark.yaml": "cannot hold one max_model_len 262144 sequence",
	}
	run := func(name string) (map[string]any, string, error) {
		c := exec.Command(blis, "run", "--scenarios", dir, "--scenario", name, "--catalog", catalog,
			"--registry", registry, "--rate", "2", "--num-requests", "8",
			"--prompt-tokens", "4096", "--prompt-tokens-min", "4096", "--prompt-tokens-max", "4096",
			"--prompt-tokens-stdev", "0", "--output-tokens", "64", "--output-tokens-min", "64",
			"--output-tokens-max", "64", "--output-tokens-stdev", "0")
		var out, errBuf bytes.Buffer
		c.Stdout, c.Stderr = &out, &errBuf
		err := c.Run()
		i := strings.Index(out.String(), "{")
		var m map[string]any
		if i >= 0 {
			_ = json.NewDecoder(strings.NewReader(out.String()[i:])).Decode(&m)
		}
		return m, errBuf.String(), err
	}
	tps := map[string]float64{}
	for _, f := range files {
		name := filepath.Base(f)
		t.Run(name, func(t *testing.T) {
			m, stderr, err := run(name)
			if want, ok := refused[name]; ok {
				if err == nil || !strings.Contains(stderr, want) {
					t.Errorf("want vLLM's refusal %q, got %v\n%s", want, err, lastLines(stderr, 3))
				}
				return
			}
			if err != nil || m == nil {
				t.Fatalf("did not complete: %v\n%s", err, lastLines(stderr, 3))
			}
			if got := m["completed_requests"]; got != float64(8) {
				t.Errorf("completed %v of 8 requests", got)
			}
			tps[name], _ = m["tokens_per_sec"].(float64)
		})
	}
	for _, layout := range []string{"pp2tp8", "tep16"} {
		roce, ib := tps["muse-h100-roce-"+layout+".yaml"], tps["muse-h200-ib-"+layout+".yaml"]
		nvl := tps["muse-gb200-nvl-"+layout+".yaml"]
		if roce == ib || ib == nvl || roce == nvl {
			t.Errorf("%s prices alike across clusters: roce %v, ib %v, nvlink %v", layout, roce, ib, nvl)
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "muse-h200-ib-dep8-dspark-128k.yaml")); err != nil {
		t.Error("the runnable 131,072-token H200 DP8 DSpark arm is missing")
	}
}

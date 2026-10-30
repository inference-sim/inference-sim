package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
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
// The -routed files serve a measured layout as two replicas behind the llm-d router
// configuration of internal deployment measurements (precise prefix-cache routing from the
// engines' KV events, prefix-cache affinity on the in-flight load, a token-load scorer, flow
// control), and run with --num-instances 2.
//
// The -1p1d file is the measured P/D arm: TP16 + EP16 on each side over RoCE H100, NIXL pulling
// the prompt's KV to the decode side, run with one prefill and one decode instance. Every
// request must be disaggregated, and none dropped.
//
// What vLLM refuses, the scenario set states as such: H200 DP8 x EP8 with DSpark at 262,144
// tokens leaves a rank fewer blocks than one full-context request's full-attention group, and
// vLLM refuses it at start-up; the 131,072-token arm runs.
func TestMuseSparkRunsOnEveryLayoutAndCluster(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the binary and runs 40 scenarios")
	}
	blis := builtBlis(t)
	_, catalog, registry := kernelRepos(t)
	dir, err := filepath.Abs(filepath.Join("..", "testdata", "scenarios", "muse-spark"))
	if err != nil {
		t.Fatal(err)
	}
	files, err := filepath.Glob(filepath.Join(dir, "*.yaml"))
	if err != nil || len(files) < 30 {
		t.Fatalf("found %d Muse scenarios (%v); the matrix is 40", len(files), err)
	}
	sort.Strings(files)
	refused := map[string]string{
		"muse-h200-ib-dep8-dspark.yaml": "cannot hold one max_model_len 262144 sequence",
	}
	run := func(name string) (map[string]any, string, error) {
		args := []string{"run", "--scenarios", dir, "--scenario", name, "--catalog", catalog,
			"--registry", registry, "--rate", "2", "--num-requests", "8",
			"--prompt-tokens", "4096", "--prompt-tokens-min", "4096", "--prompt-tokens-max", "4096",
			"--prompt-tokens-stdev", "0", "--output-tokens", "64", "--output-tokens-min", "64",
			"--output-tokens-max", "64", "--output-tokens-stdev", "0"}
		switch {
		case strings.HasSuffix(name, "-routed.yaml"):
			args = append(args, "--num-instances", "2")
		case strings.HasSuffix(name, "-1p1d.yaml"):
			args = append(args, "--num-instances", "2", "--prefill-instances", "1", "--decode-instances", "1")
		}
		c := exec.Command(blis, args...)
		var out, errBuf bytes.Buffer
		c.Stdout, c.Stderr = &out, &errBuf
		err := c.Run()
		// The cluster's block: the only one on one instance, the one so named beside the
		// instances' on several.
		var m map[string]any
		for rest := out.String(); ; {
			i := strings.Index(rest, "{")
			if i < 0 {
				break
			}
			dec := json.NewDecoder(strings.NewReader(rest[i:]))
			var b map[string]any
			if dec.Decode(&b) != nil {
				break
			}
			if m == nil || b["instance_id"] == "cluster" {
				m = b
			}
			rest = rest[i+int(dec.InputOffset()):]
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
			if got := m["dropped_unservable"]; got != nil && got != float64(0) {
				t.Errorf("dropped %v requests as unservable", got)
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

// The report's router configuration routes Muse by its KV cache: on a workload of eight agents,
// each with its own 16k-token context, two H200 replicas behind it hit their prefix caches more
// often than behind a load-only router, which places each agent's turns without regard to where
// its context is cached. It holds only if the 0.32.0 engines' KV events reach the router's
// precise-prefix producer and name the chains the router was shown.
func TestMuseSparkPrecisePrefixRoutingFollowsTheCache(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the binary and runs two clusters")
	}
	blis := builtBlis(t)
	_, catalog, registry := kernelRepos(t)
	dir, err := filepath.Abs(filepath.Join("..", "testdata", "scenarios", "muse-spark"))
	if err != nil {
		t.Fatal(err)
	}
	tmp := t.TempDir()
	workload := filepath.Join(tmp, "agents.yaml")
	if err := os.WriteFile(workload, []byte(museAgents), 0o644); err != nil {
		t.Fatal(err)
	}
	routed, err := os.ReadFile(filepath.Join(dir, "muse-h200-ib-tep8-dspark-routed.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	loadOnly, err := filepath.Abs(filepath.Join("..", "examples", "llm-d", "load-only.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const key = "endpoint_picker_config: router/precise-affinity-flow-control.yaml"
	other := strings.Replace(string(routed), key, "endpoint_picker_config: "+loadOnly, 1)
	if other == string(routed) {
		t.Fatal("the routed scenario does not name the report's router configuration")
	}
	if err := os.WriteFile(filepath.Join(tmp, "load-only.yaml"), []byte(other), 0o644); err != nil {
		t.Fatal(err)
	}
	hitRate := func(scenarios, name string) float64 {
		c := exec.Command(blis, "run", "--scenarios", scenarios, "--scenario", name, "--catalog", catalog,
			"--registry", registry, "--num-instances", "2", "--num-requests", "64", "--workload-spec", workload)
		var out, errBuf bytes.Buffer
		c.Stdout, c.Stderr = &out, &errBuf
		if err := c.Run(); err != nil {
			t.Fatalf("%s: %v\n%s", name, err, lastLines(errBuf.String(), 6))
		}
		_, after, ok := strings.Cut(out.String(), "Cache Hit Rate: ")
		if !ok {
			t.Fatalf("%s: no cache hit rate in the summary", name)
		}
		var v float64
		if _, err := fmt.Sscanf(after, "%f", &v); err != nil {
			t.Fatalf("%s: cache hit rate: %v", name, err)
		}
		return v
	}
	precise := hitRate(dir, "muse-h200-ib-tep8-dspark-routed.yaml")
	load := hitRate(tmp, "load-only.yaml")
	if precise <= load {
		t.Errorf("precise prefix routing hit %.4f, load-only %.4f: routing ignores the engines' caches", precise, load)
	}
}

// museAgents is eight agents at 3 requests/s in all, each turn a 512-token message after the
// agent's own 16,384-token context.
const museAgents = `version: "2"
seed: 7
category: language
aggregate_rate: 3.0
cohorts:
  - id: agents
    population: 8
    rate_fraction: 1.0
    arrival: {process: poisson}
    prefix_group: agent
    prefix_sharing: per_member
    prefix_length: 16384
    input_distribution: {type: constant, params: {value: 512}}
    output_distribution: {type: constant, params: {value: 64}}
`

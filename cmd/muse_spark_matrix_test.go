package cmd

import (
	"bytes"
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
	// vLLM refuses Muse-Spark on Hopper with data parallelism above one (its MoE layers take the
	// Marlin path, which requires DP1 and PCP1), and the engine refuses the same, naming vLLM's
	// message (blis-vllm#94, SIM-9). The dspark scenario was refused for its window before; the
	// start-up refusal now comes first.
	const hopperDP = "requires data parallelism 1 and prefill-context parallelism 1"
	refused := map[string]string{
		"muse-h200-ib-dep8-dspark.yaml":      hopperDP,
		"muse-h200-ib-dep8-dspark-128k.yaml": hopperDP,
		"muse-h200-ib-dep8.yaml":             hopperDP,
	}
	run := func(name string) (map[string]float64, string, error) {
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
		if err != nil {
			return nil, errBuf.String(), err
		}
		// The cluster's figures, from the one Result document stdout holds (#2058): the INV-1
		// ledger and the engine's whole-run rates (clusterFigures).
		m := clusterFigures(t, out.String())
		v, _ := resultValue(resultDoc(t, out.String()), "engine.cluster.whole_run.tokens_per_sec")
		m["tokens_per_sec"], _ = asFloat(v)
		return m, errBuf.String(), nil
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
			if got := m["completed_requests"]; got != 8 {
				t.Errorf("completed %v of 8 requests", got)
			}
			if got := m["dropped_unservable"]; got != 0 {
				t.Errorf("dropped %v requests as unservable", got)
			}
			tps[name] = m["tokens_per_sec"]
		})
	}
	for _, layout := range []string{"pp2tp8", "tep16"} {
		roce, ib := tps["muse-h100-roce-"+layout+".yaml"], tps["muse-h200-ib-"+layout+".yaml"]
		nvl := tps["muse-gb200-nvl-"+layout+".yaml"]
		if roce == ib || ib == nvl || roce == nvl {
			t.Errorf("%s prices alike across clusters: roce %v, ib %v, nvlink %v", layout, roce, ib, nvl)
		}
	}
	// Arm D is arm C with fp8 KV and the OffloadingConnector's 1 TiB CPU tier: its engine moves KV
	// to and from that tier, priced by the kernel, so it does not price as arm C does.
	if c, d := tps["muse-h100-roce-armC.yaml"], tps["muse-h100-roce-armD.yaml"]; c == 0 || d == 0 || c == d {
		t.Errorf("arm D (CPU tier, fp8 KV) prices %v against arm C's %v", d, c)
	}
	if _, err := os.Stat(filepath.Join(dir, "muse-h200-ib-dep8-dspark-128k.yaml")); err != nil {
		t.Error("the runnable 131,072-token H200 DP8 DSpark arm is missing")
	}
}

// The report's router configuration routes Muse by its KV cache: each of eight agents has its own
// long context, and two H200 replicas behind a router that follows the engines' KV events keep each
// agent on the replica that holds its context, where a router that ignores the caches (the
// load-only configuration) cannot. It holds only if the 0.33.0 engines' KV
// events reach the router's precise-prefix producer and name the chains the router was shown.
//
// Why a cache-ignoring router must miss here, whatever the step times. The KV pool is set to
// the smallest the scenario serves (gpu_memory_utilization 0.657: one maximum-length request of
// 262,144 tokens must fit, so the pool is not much larger; it was 0.56 until blis-vllm#247 put
// the vision tower's bytes in the resident weights and 0.567 until blis-substrate#379 put Muse Spark's
// per-layer RoPE tables there too, each of which leaves the smaller pool too small for four
// contexts, where precise routing and load-only routing both hit 0.37), and the eight contexts are 40,960 tokens
// each: 331,776 tokens in all, more than one replica holds, while the four a replica serves under
// affinity are 166,000 and fit. A router that places a turn without regard to the cache sends
// turns of all eight agents to one replica, whose pool then evicts the least recently used
// context before its owner's next turn; a router that follows the cache keeps four agents on each
// replica and every later turn hits. The earlier form of this test (16,384-token contexts at the
// pool's full size, every context cached on both replicas after one miss) passed on a margin of a
// single block hit in 66 (0.7576 against 0.7424) that moved with the step times.
func TestMuseSparkPrecisePrefixRoutingFollowsTheCache(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the binary and runs two clusters")
	}
	blis := builtBlis(t)
	_, catalog, registry := kernelRepos(t)
	src, err := filepath.Abs(filepath.Join("..", "testdata", "scenarios", "muse-spark"))
	if err != nil {
		t.Fatal(err)
	}
	tmp := t.TempDir()
	dir := filepath.Join(tmp, "scenarios")
	if err := os.CopyFS(dir, os.DirFS(src)); err != nil {
		t.Fatal(err)
	}
	workload := filepath.Join(tmp, "agents.yaml")
	if err := os.WriteFile(workload, []byte(museAgents), 0o644); err != nil {
		t.Fatal(err)
	}
	const (
		routedName = "muse-h200-ib-tep8-dspark-routed.yaml"
		picker     = "endpoint_picker_config: router/precise-affinity-flow-control.yaml"
		fullPool   = "gpu_memory_utilization: 0.9"
		smallPool  = "gpu_memory_utilization: 0.657"
	)
	routed, err := os.ReadFile(filepath.Join(src, routedName))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(routed), picker) || !strings.Contains(string(routed), fullPool) {
		t.Fatal("the routed scenario does not name the report's router configuration and a 0.9 pool")
	}
	small := strings.Replace(string(routed), fullPool, smallPool, 1)
	if err := os.WriteFile(filepath.Join(dir, routedName), []byte(small), 0o644); err != nil {
		t.Fatal(err)
	}
	// The same deployment behind the load-only router, which ignores the engines' caches.
	loadOnly, err := filepath.Abs(filepath.Join("..", "examples", "llm-d", "load-only.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "load-only.yaml"), []byte(strings.Replace(small, picker, "endpoint_picker_config: "+loadOnly, 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	hitRate := func(name string) float64 {
		c := exec.Command(blis, "run", "--scenarios", dir, "--scenario", name, "--catalog", catalog,
			"--registry", registry, "--num-instances", "2", "--num-requests", "64", "--workload-spec", workload)
		var out, errBuf bytes.Buffer
		c.Stdout, c.Stderr = &out, &errBuf
		if err := c.Run(); err != nil {
			t.Fatalf("%s: %v\n%s", name, err, lastLines(errBuf.String(), 6))
		}
		// The whole-run prefix-cache hit rate the text summary printed, from the Result (#2058).
		return clusterFigures(t, out.String())["cache_hit_rate_whole_run"]
	}
	precise := hitRate(routedName)
	if load := hitRate("load-only.yaml"); precise < load+0.05 {
		t.Errorf("precise prefix routing hit %.4f, load-only %.4f: routing ignores the engines' caches "+
			"(it must beat a cache-ignoring router by at least 0.05 here)", precise, load)
	}
	if precise < 0.4 {
		t.Errorf("precise prefix routing hit %.4f; with four contexts a replica, later turns hit and the rate is above 0.4", precise)
	}
}

// museAgents is eight agents at 3 requests/s in all, each turn a 512-token message after the
// agent's own 40,960-token context.
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
    prefix_length: 40960
    input_distribution: {type: constant, params: {value: 512}}
    output_distribution: {type: constant, params: {value: 64}}
`

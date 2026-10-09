package cmd

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/spf13/cobra"
)

// Behavioral contract for `--latency-model blis-latency-kernel` on `blis run` and
// `blis replay`.
//
// The kernel already satisfied sim.LatencyModel and was reachable only from the
// standalone scoring binaries (cmd/metricscore and friends), which construct it
// directly. `blis run` accepted roofline and trained-physics alone, so the CLI with
// every serving knob on it -- P/D roles, --dp, speculative tokens, workload specs --
// could not be driven against the kernel at all. These tests pin the behaviour that
// closes that gap, at the CLI boundary rather than on the wiring.
//
// Each leg re-execs this test binary as a real `blis` invocation so the cobra tree
// executes and a logrus.Fatalf surfaces as a non-zero exit, following
// cmd/catalog_cli_test.go.
//
//	C1 — a kernel run completes and reports metrics.
//	C3 — --dp > 1 and --enable-expert-parallel are ACCEPTED on the kernel backend.
//	C5 — a missing scenario is refused naming what could not be resolved, never
//	     silently served by another backend.

const kernelLegEnv = "BLIS_KERNEL_LEG"

// kernelRepos is the three artifact roots a kernel scenario resolves against: the pinned
// kernel module's scenario fixtures and the vendored catalog and registry. A missing root
// fails, naming the command that provides it -- never a skip, which would read as a pass.
func kernelRepos(t *testing.T) (scenarios, catalog, registry string) {
	t.Helper()
	r := kernelmodel.DefaultRepos()
	if err := kernelmodel.RequireRepos(r); err != nil {
		t.Fatal(err)
	}
	return r.Scenarios, r.Catalog, r.Registry
}

func runKernelLeg(t *testing.T, leg string, extra ...string) (stdout, stderr string, err error) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_KernelBackend$")
	env := append(os.Environ(), kernelLegEnv+"="+leg)
	if len(extra) > 0 {
		env = append(env, "BLIS_KERNEL_EXTRA="+strings.Join(extra, "\x1f"))
	}
	cmd.Env = env
	var out, errBuf bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errBuf
	err = cmd.Run()
	return out.String(), errBuf.String(), err
}

// TestRunCmd_KernelBackend drives the real CLI on the kernel backend.
func TestRunCmd_KernelBackend(t *testing.T) {
	if leg := os.Getenv(kernelLegEnv); leg != "" {
		args := []string{
			"run", "--latency-model", "blis-latency-kernel",
			"--scenario", os.Getenv("BLIS_KERNEL_SCENARIO"),
			"--scenarios", os.Getenv("BLIS_KERNEL_SCENARIOS"),
			"--catalog", os.Getenv("BLIS_KERNEL_CATALOG"),
			"--registry", os.Getenv("BLIS_KERNEL_REGISTRY"),
			"--seed", "42", "--num-requests", "8", "--concurrency", "2",
			"--defaults-filepath", "../defaults.yaml",
		}
		if e := os.Getenv("BLIS_KERNEL_EXTRA"); e != "" {
			args = append(args, strings.Split(e, "\x1f")...)
		}
		rootCmd.SetArgs(args)
		if execErr := rootCmd.Execute(); execErr != nil {
			os.Exit(1)
		}
		os.Exit(0)
	}

	scenarios, catalog, registry := kernelRepos(t)
	t.Setenv("BLIS_KERNEL_SCENARIOS", scenarios)
	t.Setenv("BLIS_KERNEL_CATALOG", catalog)
	t.Setenv("BLIS_KERNEL_REGISTRY", registry)

	tests := []struct {
		name      string
		scenario  string
		extra     []string
		wantFatal bool
		wantMsg   string
	}{
		{
			// C1: the run completes on the kernel and prints metrics.
			name:     "a kernel run reports metrics",
			scenario: "gpt-oss-120b-h200-fp4-vllm-tp4.yaml",
		},
		{
			// The deployment has one source. Re-supplying a fact the scenario states is
			// refused naming both, rather than reconciled by precedence logic.
			name:      "a duplicated deployment flag is refused",
			scenario:  "gpt-oss-120b-h200-fp4-vllm-tp4.yaml",
			extra:     []string{"--tp", "2"},
			wantFatal: true,
			wantMsg:   "--tp is not accepted",
		},
		{
			// The engine knobs that size admission and the KV pool are the kernel's answers
			// for the pool. A flag restating one would run a scheduler sized differently from
			// the engine the kernel priced, so it is refused naming where the answer comes from.
			name:      "a restated engine knob is refused",
			scenario:  "gpt-oss-120b-h200-fp4-vllm-tp4.yaml",
			extra:     []string{"--total-kv-blocks", "100"},
			wantFatal: true,
			wantMsg:   "--total-kv-blocks is not accepted",
		},
		{
			// C3: the kernel models DP/EP step time, so a scenario that states expert
			// parallelism runs -- the gate that names trained-physics as the only
			// option must admit this backend.
			name:     "a scenario with expert parallelism runs",
			scenario: "minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml",
		},
		{
			// C5: an unresolvable scenario is refused naming it, not served by a
			// fallback backend.
			name:      "a missing scenario is refused",
			scenario:  "no-such-scenario.yaml",
			wantFatal: true,
			wantMsg:   "no-such-scenario",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("BLIS_KERNEL_SCENARIO", tt.scenario)
			stdout, stderr, err := runKernelLeg(t, "go", tt.extra...)

			if tt.wantFatal {
				if err == nil {
					t.Fatalf("expected refusal for scenario %q\nstdout:\n%s", tt.scenario, stdout)
				}
				if tt.wantMsg != "" && !strings.Contains(stderr, tt.wantMsg) {
					t.Errorf("refusal should name %q, got stderr:\n%s", tt.wantMsg, stderr)
				}
				return
			}
			if err != nil {
				t.Fatalf("expected the run to succeed, got %v\nstderr:\n%s", err, stderr)
			}
			// Non-vacuity: a silently-empty run would otherwise satisfy "no error".
			if !strings.Contains(stdout, "completed_requests") {
				t.Errorf("expected simulation metrics on stdout, got:\n%s", stdout)
			}
		})
	}
}

// A kernel run's results file must attribute the result to the catalog it read, as every
// other backend's does (#1900): the catalog supplies the model graph, the chip and the
// fabric, so a result without a catalog revision cannot be reproduced. Before the fix the
// kernel path resolved the catalog without recording it, and the block was simply absent.
func TestRunCmd_KernelBackend_RecordsCatalogProvenance(t *testing.T) {
	scenarios, catalog, registry := kernelRepos(t)
	t.Setenv("BLIS_KERNEL_SCENARIOS", scenarios)
	t.Setenv("BLIS_KERNEL_CATALOG", catalog)
	t.Setenv("BLIS_KERNEL_REGISTRY", registry)
	t.Setenv("BLIS_KERNEL_SCENARIO", "gpt-oss-120b-h200-fp4-vllm-tp4.yaml")
	metricsFile := filepath.Join(t.TempDir(), "metrics.json")

	if _, stderr, err := runKernelLeg(t, "go", "--metrics-path", metricsFile); err != nil {
		t.Fatalf("kernel run failed: %v\nstderr:\n%s", err, stderr)
	}
	got := readMetricsFile(t, metricsFile).Catalog
	if got == nil {
		t.Fatal("the kernel run's results file carries no catalog provenance block")
	}
	if got.Path != catalog {
		t.Errorf("catalog provenance path is %q, want the catalog the run read, %q", got.Path, catalog)
	}
	if got.Revision == "" {
		t.Error("catalog provenance carries no revision field")
	}
}

// The run is sized from the kernel and from nothing else: every engine knob the simulated
// scheduler reads is the kernel's Settings answer for the pool, per data-parallel rank.
// Checked on a dense deployment and on a dp=2 MoE one, where the per-rank budget and the
// aggregate differ -- the case a mix-up between them would get wrong.
func TestAdoptKernelDeployment_SizesTheRunFromTheKernel(t *testing.T) {
	scenarios, catalog, registry := kernelRepos(t)
	saved := []any{latencyModelBackend, kernelScenario, kernelScenarioDir, kernelRegistry,
		catalogPath, totalKVBlocks, blockSizeTokens, maxNumSeqs, maxNumBatchedTokens, maxModelLen,
		noEnablePrefixCaching, numSpeculativeTokens, speculativeMethod, model, gpu,
		tensorParallelism, dataParallelism, enableExpertParallel, kernelOpened,
		kernelDeploymentExperts, kernelDeploymentTopK}
	defer func() {
		latencyModelBackend, kernelScenario, kernelScenarioDir, kernelRegistry =
			saved[0].(string), saved[1].(string), saved[2].(string), saved[3].(string)
		catalogPath, totalKVBlocks, blockSizeTokens = saved[4].(string), saved[5].(int64), saved[6].(int64)
		maxNumSeqs, maxNumBatchedTokens, maxModelLen = saved[7].(int64), saved[8].(int64), saved[9].(int64)
		noEnablePrefixCaching, numSpeculativeTokens = saved[10].(bool), saved[11].(int)
		speculativeMethod, model, gpu = saved[12].(string), saved[13].(string), saved[14].(string)
		tensorParallelism, dataParallelism, enableExpertParallel = saved[15].(int), saved[16].(int), saved[17].(bool)
		kernelOpened, _ = saved[18].(*kernelmodel.Model)
		kernelDeploymentExperts, kernelDeploymentTopK = saved[19].(int), saved[20].(int)
	}()

	for _, scenario := range []string{
		"llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml",
		"minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml",
	} {
		t.Run(scenario, func(t *testing.T) {
			cmd := &cobra.Command{}
			registerSimConfigFlags(cmd)
			if err := cmd.ParseFlags([]string{
				"--latency-model", "blis-latency-kernel", "--scenario", scenario,
				"--scenarios", scenarios, "--registry", registry, "--catalog", catalog,
			}); err != nil {
				t.Fatal(err)
			}
			adoptKernelDeployment(cmd)

			st, err := kernelOpened.Settings()
			if err != nil {
				t.Fatal(err)
			}
			for _, c := range []struct {
				name      string
				got, want int64
			}{
				{"KV blocks per replica", totalKVBlocks, st.KVBlocks},
				{"block size", blockSizeTokens, int64(st.BlockSize)},
				{"max_num_seqs", maxNumSeqs, int64(st.MaxNumSeqs)},
				{"max_num_batched_tokens", maxNumBatchedTokens, int64(st.MaxNumBatchedTokens)},
				{"max_model_len", maxModelLen, int64(st.MaxModelLen)},
				{"speculative tokens", int64(numSpeculativeTokens), int64(st.SpeculativeTokens)},
				{"dp", int64(dataParallelism), int64(st.DataParallel)},
			} {
				if c.got != c.want {
					t.Errorf("%s: the run uses %d, the kernel answers %d", c.name, c.got, c.want)
				}
			}
			if noEnablePrefixCaching != st.PrefixCachingDisabled {
				t.Errorf("prefix caching disabled: run %t, kernel %t", noEnablePrefixCaching, st.PrefixCachingDisabled)
			}
			if st.DataParallel > 1 && st.AggregateKVBlocks <= st.KVBlocks {
				t.Errorf("dp %d: aggregate %d is not above per-rank %d, so this case does not "+
					"distinguish them", st.DataParallel, st.AggregateKVBlocks, st.KVBlocks)
			}
		})
	}
}

// kernelCLIArgsEnv carries a full blis command line, \x1f-separated, to the re-exec'd leg.
const kernelCLIArgsEnv = "BLIS_KERNEL_CLI_ARGS"

// TestKernelCLILeg is the re-exec target for runKernelCLI: it executes the command line it is
// handed as a real `blis` invocation, so a logrus.Fatalf surfaces as a non-zero exit.
func TestKernelCLILeg(t *testing.T) {
	raw := os.Getenv(kernelCLIArgsEnv)
	if raw == "" {
		t.Skip("re-exec target only; driven by runKernelCLI")
	}
	rootCmd.SetArgs(strings.Split(raw, "\x1f"))
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
	os.Exit(0)
}

// runKernelCLI runs `blis <args...>` on the kernel backend against the pinned scenario
// fixtures and the vendored catalog and registry.
func runKernelCLI(t *testing.T, args ...string) (stdout, stderr string, err error) {
	t.Helper()
	scenarios, catalog, registry := kernelRepos(t)
	full := append(args, "--latency-model", "blis-latency-kernel",
		"--catalog", catalog, "--registry", registry, "--defaults-filepath", "../defaults.yaml")
	if !slices.Contains(args, "--scenarios") {
		full = append(full, "--scenarios", scenarios)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestKernelCLILeg$")
	cmd.Env = append(os.Environ(), kernelCLIArgsEnv+"="+strings.Join(full, "\x1f"))
	var out, errBuf bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errBuf
	err = cmd.Run()
	return out.String(), errBuf.String(), err
}

// A run's exported trace, replayed on the same scenario and horizon, reproduces the run
// (INV-13 is stated for identical flags including --horizon, since replay otherwise derives
// its horizon from the trace): the
// replay prices with the same kernel and sizes itself from the same Settings, so its stdout
// is byte-identical. Replay could not run the kernel at all before this; it is deprecated in
// favour of `blis run` (#1901), and this pins the parity that removal will be checked against.
func TestReplayCmd_KernelBackend_ReproducesTheRun(t *testing.T) {
	const scenario = "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml"
	prefix := filepath.Join(t.TempDir(), "trace")
	common := []string{"--scenario", scenario, "--seed", "7", "--horizon", "600000000"}
	runOut, runErr, err := runKernelCLI(t, append([]string{"run", "--num-requests", "40", "--rate", "15",
		"--trace-output", prefix}, common...)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, runErr)
	}
	repOut, repErr, err := runKernelCLI(t, append([]string{"replay", "--trace-header", prefix + ".yaml",
		"--trace-data", prefix + ".csv"}, common...)...)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, repErr)
	}
	if !strings.Contains(runOut, `"completed_requests": 40`) {
		t.Fatalf("the run did not complete its 40 requests; the comparison would be vacuous:\n%s", runOut)
	}
	if repOut != runOut {
		t.Errorf("replay stdout differs from the run it replays\n--- run\n%s\n--- replay\n%s", runOut, repOut)
	}
}

// An agentic trace -- the path the Weka, Exgentic and OpenTelemetry corpora take today --
// replays on the kernel as closed-loop sessions: every round of every session completes, and
// a second replay at the same seed is byte-identical (INV-6).
func TestReplayCmd_KernelBackend_ReplaysAnAgenticTrace(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "weka.jsonl")
	var sessions []string
	for s := 0; s < 6; s++ {
		sessions = append(sessions, fmt.Sprintf(`{"id":"sess-%d","models":["m"],"requests":[`+
			`{"type":"n","t":0.0,"in":%d,"out":40,"api_time":1.0},`+
			`{"type":"n","t":4.0,"in":%d,"out":30,"api_time":1.0},`+
			`{"type":"n","t":9.0,"in":%d,"out":20,"api_time":1.0}]}`, s, 900+s*50, 1400+s*50, 2000+s*50))
	}
	if err := os.WriteFile(in, []byte(strings.Join(sessions, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	prefix := filepath.Join(dir, "agentic")
	if err := runConvertWeka(in, prefix, workload.WekaConvertOptions{ContextGrowth: "accumulate", MinRounds: 1}); err != nil {
		t.Fatal(err)
	}
	args := []string{"replay", "--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv",
		"--scenario", "gpt-oss-120b-h200-fp4-vllm-tp4.yaml", "--session-mode", "closed-loop", "--seed", "3"}
	first, stderr, err := runKernelCLI(t, args...)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, stderr)
	}
	if !strings.Contains(first, `"completed_requests": 18`) {
		t.Errorf("6 sessions x 3 rounds should complete 18 requests:\n%s", first)
	}
	second, _, err := runKernelCLI(t, args...)
	if err != nil {
		t.Fatal(err)
	}
	if second != first {
		t.Error("two replays at one seed differ")
	}
}

// pdScenarios is this repository's disaggregated scenario fixtures: every vendored kernel
// scenario is colocated.
const pdScenarios = "../testdata/scenarios"

// A disaggregated scenario runs as its pools: three prefill and one decode instance complete
// every request, and the topology is checked against what the pools hold and what a scenario
// states -- a P/D topology over a colocated scenario, more instances than a pool's nodes hold,
// and a per-role flag restating the scenario are each refused.
func TestRunCmd_KernelBackend_Disaggregated(t *testing.T) {
	pd := []string{"--scenarios", pdScenarios, "--scenario", "glm-5-h200-3p1d-ib.yaml",
		"--pd-decider", "always", "--num-requests", "30", "--rate", "6", "--seed", "1"}
	for _, tt := range []struct {
		name    string
		args    []string
		wantErr string
	}{
		{name: "3P1D completes every request",
			args: append([]string{"run", "--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1"}, pd...)},
		{name: "more prefill instances than the pool holds",
			args:    append([]string{"run", "--num-instances", "5", "--prefill-instances", "4", "--decode-instances", "1"}, pd...),
			wantErr: "prefill pool holds 3 rank(s)"},
		{name: "a per-role flag restating the scenario",
			args:    append([]string{"run", "--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1", "--prefill-tp", "4"}, pd...),
			wantErr: "--prefill-tp is not accepted"},
		{name: "a P/D topology over a colocated scenario",
			args: []string{"run", "--scenario", "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml", "--num-instances", "2",
				"--prefill-instances", "1", "--decode-instances", "1", "--num-requests", "4", "--rate", "2"},
			wantErr: "states no prefill pool"},
		{name: "more colocated instances than the pool holds",
			args: []string{"run", "--scenario", "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml", "--num-instances", "3",
				"--num-requests", "4", "--rate", "2"},
			wantErr: "colocated pool holds 2 rank(s)"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			stdout, stderr, err := runKernelCLI(t, tt.args...)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(stderr, tt.wantErr) {
					t.Fatalf("want a refusal naming %q, got err=%v\nstderr:\n%s", tt.wantErr, err, stderr)
				}
				return
			}
			if err != nil {
				t.Fatalf("%v\n%s", err, stderr)
			}
			if !strings.Contains(stdout, `"completed_requests": 30`) {
				t.Errorf("the cluster did not complete its 30 requests:\n%s", stdout)
			}
		})
	}
}

// The handoff is priced by the prefill pool's kernel between the two instances' placements in
// their pools, and moving more KV never costs less.
func TestKernelPools_TheHandoffIsTheKernelsPrice(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	saved := []any{kernelScenario, kernelScenarioDir, kernelRegistry, kernelOpened,
		prefillInstances, decodeInstances, prefillDecodeInstances, encodeInstances}
	defer func() {
		kernelScenario, kernelScenarioDir, kernelRegistry = saved[0].(string), saved[1].(string), saved[2].(string)
		kernelOpened, _ = saved[3].(*kernelmodel.Model)
		prefillInstances, decodeInstances = saved[4].(int), saved[5].(int)
		prefillDecodeInstances, encodeInstances = saved[6].(int), saved[7].(int)
	}()
	kernelScenario, kernelScenarioDir, kernelRegistry = "glm-5-h200-3p1d-ib.yaml", pdScenarios, registry
	repos := kernelmodel.Repos{Scenarios: pdScenarios, Catalog: catalog, Registry: registry}
	m, err := kernelmodel.Open(kernelScenario, repos)
	if err != nil {
		t.Fatal(err)
	}
	kernelOpened = m
	prefillInstances, decodeInstances, prefillDecodeInstances, encodeInstances = 3, 1, 0, 0

	p := openKernelPools(catalog)
	price := p.transferTime()
	var prev int64
	for _, tokens := range []int64{64, 1024, 8192, 65536} {
		for from := 0; from < 3; from++ {
			got := price(tokens, cluster.InstanceID(fmt.Sprintf("instance_%d", from)), "instance_3")
			want := p.prefill.PDTransferTicks(tokens, p.prefill.PlacementOf(from), p.decode.PlacementOf(0))
			if got != want {
				t.Errorf("%d tokens from prefill rank %d: priced %d, the prefill kernel says %d", tokens, from, got, want)
			}
			if from == 0 {
				if got < prev {
					t.Errorf("%d tokens priced %d, below a smaller transfer's %d", tokens, got, prev)
				}
				prev = got
			}
		}
	}
}

// writeKernelOffloadConfig writes a --kv-offload-config whose one secondary tier is the named
// catalog device, with a CPU tier small enough that the secondary tier is exercised.
func writeKernelOffloadConfig(t *testing.T, tierExtra string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "offload.yaml")
	body := "kv_offload:\n" +
		"  cpu_bytes_to_use: 4294967296\n" +
		"  secondary_tiers:\n" +
		"    - type: fs\n" +
		"      root_dir: /mnt/kv\n" +
		"      direct_io: true\n" +
		"      device_class: nvme_gen4\n" + tierExtra
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// KV offload on the kernel: a tier named by its catalog device is priced by the kernel, a run
// with it completes and its trace replays byte-identically, and a tier stating its own physics
// -- or a legacy-tier flag restating the kernel's price -- is refused.
func TestRunCmd_KernelBackend_KVOffload(t *testing.T) {
	const scenario = "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml"
	common := []string{"--scenario", scenario, "--seed", "4", "--horizon", "600000000"}
	base := append([]string{"--num-requests", "30", "--rate", "10"}, common...)

	prefix := filepath.Join(t.TempDir(), "trace")
	runOut, stderr, err := runKernelCLI(t, append([]string{"run", "--kv-offload-config",
		writeKernelOffloadConfig(t, ""), "--trace-output", prefix}, base...)...)
	if err != nil {
		t.Fatalf("offload run: %v\n%s", err, stderr)
	}
	if !strings.Contains(runOut, `"completed_requests": 30`) {
		t.Fatalf("the offload run did not complete its 30 requests:\n%s", runOut)
	}
	repOut, stderr, err := runKernelCLI(t, append([]string{"replay", "--trace-header", prefix + ".yaml",
		"--trace-data", prefix + ".csv"}, common...)...)
	if err != nil {
		t.Fatalf("offload replay: %v\n%s", err, stderr)
	}
	if repOut != runOut {
		t.Errorf("the offload run's replay differs from the run\n--- run\n%s\n--- replay\n%s", runOut, repOut)
	}

	legacy, stderr, err := runKernelCLI(t, append([]string{"run", "--kv-cpu-blocks", "2000"}, base...)...)
	if err != nil || !strings.Contains(legacy, `"completed_requests": 30`) {
		t.Errorf("legacy CPU tier run: %v\n%s\n%s", err, legacy, stderr)
	}

	for _, tt := range []struct {
		name, want string
		args       []string
	}{
		{"a tier stating its own bandwidth", "states read_bandwidth",
			[]string{"run", "--kv-offload-config", writeKernelOffloadConfig(t, "      read_bandwidth: 7000.0\n")}},
		{"a legacy-tier bandwidth flag", "--kv-transfer-bandwidth is not accepted",
			[]string{"run", "--kv-cpu-blocks", "2000", "--kv-transfer-bandwidth", "5"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, stderr, err := runKernelCLI(t, append(tt.args, base...)...)
			if err == nil || !strings.Contains(stderr, tt.want) {
				t.Errorf("want a refusal naming %q, got err=%v\n%s", tt.want, err, stderr)
			}
		})
	}
}

// The offload prices are the kernel's: the attached tier pricer and the legacy per-block
// charge equal TierTime in whole ticks rounded up, and moving more bytes never costs less.
func TestApplyKernelOffloadPricing_IsTheKernelsTierTime(t *testing.T) {
	scenarios, catalog, registry := kernelRepos(t)
	saved := []any{kernelOpened, kvCPUBlocks, blockSizeTokens}
	defer func() {
		kernelOpened, _ = saved[0].(*kernelmodel.Model)
		kvCPUBlocks, blockSizeTokens = saved[1].(int64), saved[2].(int64)
	}()
	m, err := kernelmodel.Open("gpt-oss-120b-h200-fp4-vllm-tp4.yaml",
		kernelmodel.Repos{Scenarios: scenarios, Catalog: catalog, Registry: registry})
	if err != nil {
		t.Fatal(err)
	}
	kernelOpened, kvCPUBlocks, blockSizeTokens = m, 100, 16
	cfg := sim.KVOffloadConfig{Enabled: true, Tiers: []sim.KVOffloadTier{{DeviceClass: "nvme_gen4"}}}
	legacy := applyKernelOffloadPricing(&cfg)

	ticks := func(tier string, dir kernel.Direction, bytes int64, q int) int64 {
		return max(1, (m.Kernel().TierTime(tier, dir, bytes, q).Nanoseconds()+999)/1000)
	}
	if want := ticks("cpu_dram", kernel.DirectionFromTier, m.Kernel().SequenceVariableBytes(16), 1); legacy != want {
		t.Errorf("legacy per-block charge %d, the kernel prices one block's reload at %d", legacy, want)
	}
	price := cfg.Tiers[0].ServiceTime
	if price == nil {
		t.Fatal("no pricer attached to the tier")
	}
	var prev int64
	for _, bytes := range []int64{1 << 10, 1 << 20, 1 << 26, 1 << 30} {
		for _, q := range []int{1, 4} {
			for _, write := range []bool{false, true} {
				dir := kernel.DirectionFromTier
				if write {
					dir = kernel.DirectionToTier
				}
				if got, want := price(write, bytes, q), ticks("nvme_gen4", dir, bytes, q); got != want {
					t.Errorf("%d bytes q=%d write=%t: priced %d, kernel %d", bytes, q, write, got, want)
				}
			}
		}
		if got := price(false, bytes, 1); got < prev {
			t.Errorf("%d bytes priced %d, below a smaller transfer's %d", bytes, got, prev)
		} else {
			prev = got
		}
	}
}

package cmd

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/spf13/cobra"
	"pgregory.net/rapid"
)

// Behavioral contract for the blis-latency-kernel backend -- the only latency backend --
// on `blis run` and `blis replay`.
//
// The kernel prices every step of `blis run` and `blis replay`, with every serving knob
// on the CLI -- P/D roles, dp placement, speculative tokens, workload specs -- driven
// against it. These tests pin that behaviour at the CLI boundary rather than on the wiring.
//
// Each leg re-execs this test binary as a real `blis` invocation so the cobra tree
// executes and a logrus.Fatalf surfaces as a non-zero exit, following
// cmd/catalog_cli_test.go.
//
//	C1 — a kernel run completes and reports metrics.
//	C3 — a scenario stating dp > 1 and expert parallelism runs.
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
			"run",
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
			wantMsg:   "unknown flag: --tp",
		},
		{
			// The engine knobs that size admission and the KV pool are the kernel's answers
			// for the pool. A flag restating one would run a scheduler sized differently from
			// the engine the kernel priced, so it is refused naming where the answer comes from.
			name:      "a restated engine knob is refused",
			scenario:  "gpt-oss-120b-h200-fp4-vllm-tp4.yaml",
			extra:     []string{"--total-kv-blocks", "100"},
			wantFatal: true,
			wantMsg:   "unknown flag: --total-kv-blocks",
		},
		{
			// C3: the kernel models DP/EP step time, so a scenario that states expert
			// parallelism runs.
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
	saved := []any{kernelScenario, kernelScenarioDir, kernelRegistry,
		catalogPath, totalKVBlocks, blockSizeTokens, maxNumSeqs, maxNumBatchedTokens, maxModelLen,
		noEnablePrefixCaching, numSpeculativeTokens, speculativeMethod, model, gpu,
		tensorParallelism, dataParallelism, enableExpertParallel, kernelOpened,
		kernelDeploymentExperts, kernelDeploymentTopK}
	defer func() {
		kernelScenario, kernelScenarioDir, kernelRegistry = saved[0].(string), saved[1].(string), saved[2].(string)
		catalogPath, totalKVBlocks, blockSizeTokens = saved[3].(string), saved[4].(int64), saved[5].(int64)
		maxNumSeqs, maxNumBatchedTokens, maxModelLen = saved[6].(int64), saved[7].(int64), saved[8].(int64)
		noEnablePrefixCaching, numSpeculativeTokens = saved[9].(bool), saved[10].(int)
		speculativeMethod, model, gpu = saved[11].(string), saved[12].(string), saved[13].(string)
		tensorParallelism, dataParallelism, enableExpertParallel = saved[14].(int), saved[15].(int), saved[16].(bool)
		kernelOpened, _ = saved[17].(*kernelmodel.Model)
		kernelDeploymentExperts, kernelDeploymentTopK = saved[18].(int), saved[19].(int)
	}()

	for _, scenario := range []string{
		"llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml",
		"minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml",
	} {
		t.Run(scenario, func(t *testing.T) {
			cmd := &cobra.Command{}
			registerSimConfigFlags(cmd)
			if err := cmd.ParseFlags([]string{
				"--scenario", scenario,
				"--scenarios", scenarios, "--registry", registry, "--catalog", catalog,
			}); err != nil {
				t.Fatal(err)
			}
			adoptKernelDeployment(cmd)

			// The oracle is the scenario opened independently, not the model adoption kept:
			// were adoption to open the wrong pool, the two would disagree.
			independent, err := kernelmodel.Open(scenario,
				kernelmodel.Repos{Scenarios: scenarios, Catalog: catalog, Registry: registry})
			if err != nil {
				t.Fatal(err)
			}
			st, err := independent.Settings()
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
	full := append(args,
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
// is byte-identical. Replay could not run the kernel at all before this; this pins run/replay
// parity (INV-13).
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
			wantErr: "unknown flag: --prefill-tp"},
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

// An instance's rank in its pool is a bijection onto each pool's ranks: under the cluster's own
// numbering, the prefill instances are prefill ranks 0..P-1 and the decode instances decode
// ranks 0..D-1, each once.
func TestRankInPool_IsABijectionOntoEachPoolsRanks(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		p := rapid.IntRange(1, 64).Draw(rt, "prefill")
		d := rapid.IntRange(1, 64).Draw(rt, "decode")
		seen := map[cluster.PoolRole]map[int]bool{cluster.PoolRolePrefill: {}, cluster.PoolRoleDecode: {}}
		for id, role := range cluster.BuildPoolMembershipFromIndices(p+d, p, d, 0, 0) {
			r := rankInPool(cluster.InstanceID(id), p)
			if seen[role][r] {
				rt.Fatalf("%s: rank %d of the %v pool assigned twice", id, r, role)
			}
			seen[role][r] = true
		}
		for role, want := range map[cluster.PoolRole]int{cluster.PoolRolePrefill: p, cluster.PoolRoleDecode: d} {
			for r := 0; r < want; r++ {
				if !seen[role][r] {
					rt.Fatalf("no instance is rank %d of the %v pool (%d ranks)", r, role, want)
				}
			}
		}
	})
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
		{"a legacy-tier bandwidth flag", "unknown flag: --kv-transfer-bandwidth",
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
	// Laws over the whole domain, both directions: more bytes or a deeper queue never makes a
	// transfer cheaper.
	rapid.Check(t, func(rt *rapid.T) {
		write := rapid.Bool().Draw(rt, "write")
		a := rapid.Int64Range(0, 1<<32).Draw(rt, "bytes")
		b := rapid.Int64Range(a, 1<<32).Draw(rt, "more")
		q := rapid.IntRange(1, 64).Draw(rt, "q")
		r := rapid.IntRange(q, 64).Draw(rt, "deeper")
		if price(write, b, q) < price(write, a, q) {
			rt.Fatalf("write=%t: %d bytes cheaper than %d at depth %d", write, b, a, q)
		}
		if price(write, a, r) < price(write, a, q) {
			rt.Fatalf("write=%t: %d bytes cheaper at depth %d than %d", write, a, r, q)
		}
	})
}

// A LoRA adapter reservation shrinks each P/D pool's KV budget, as it does a colocated one's:
// both pools' overrides hold strictly fewer blocks with the reservation than without.
func TestPoolOverrides_TheLoRAReservationShrinksBothPools(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	saved := []any{kernelScenario, kernelScenarioDir, kernelRegistry, kernelOpened,
		prefillInstances, decodeInstances, loraReservedBytesForKV}
	defer func() {
		kernelScenario, kernelScenarioDir, kernelRegistry = saved[0].(string), saved[1].(string), saved[2].(string)
		kernelOpened, _ = saved[3].(*kernelmodel.Model)
		prefillInstances, decodeInstances, loraReservedBytesForKV = saved[4].(int), saved[5].(int), saved[6].(int64)
	}()
	kernelScenario, kernelScenarioDir, kernelRegistry = "glm-5-h200-3p1d-ib.yaml", pdScenarios, registry
	m, err := kernelmodel.Open(kernelScenario, kernelmodel.Repos{Scenarios: pdScenarios, Catalog: catalog, Registry: registry})
	if err != nil {
		t.Fatal(err)
	}
	kernelOpened, prefillInstances, decodeInstances = m, 3, 1
	pools := openKernelPools(catalog)

	loraReservedBytesForKV = 0
	p0, d0 := pools.overrides()
	loraReservedBytesForKV = 4 << 30
	p1, d1 := pools.overrides()
	if *p1.TotalKVBlocks >= *p0.TotalKVBlocks || *d1.TotalKVBlocks >= *d0.TotalKVBlocks {
		t.Errorf("a 4 GiB reservation left prefill %d -> %d and decode %d -> %d blocks; both must shrink",
			*p0.TotalKVBlocks, *p1.TotalKVBlocks, *d0.TotalKVBlocks, *d1.TotalKVBlocks)
	}
}

// A LoRA adapter reservation re-sizes a kernel run's KV pool to the kernel's budget with the
// reservation set aside -- strictly smaller than without it -- and no reservation leaves the
// pool the kernel's plain answer.
func TestApplyKernelLoRAReservation_ShrinksThePoolByTheKernelsAnswer(t *testing.T) {
	scenarios, catalog, registry := kernelRepos(t)
	saved := []any{kernelOpened, totalKVBlocks, loraReservedBytesForKV, kernelScenario}
	defer func() {
		kernelOpened, _ = saved[0].(*kernelmodel.Model)
		totalKVBlocks, loraReservedBytesForKV, kernelScenario = saved[1].(int64), saved[2].(int64), saved[3].(string)
	}()
	kernelScenario = "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml"
	m, err := kernelmodel.Open(kernelScenario, kernelmodel.Repos{Scenarios: scenarios, Catalog: catalog, Registry: registry})
	if err != nil {
		t.Fatal(err)
	}
	plain, _ := m.Settings()
	kernelOpened = m

	totalKVBlocks, loraReservedBytesForKV = plain.KVBlocks, 0
	applyKernelLoRAReservation()
	if totalKVBlocks != plain.KVBlocks {
		t.Errorf("no reservation moved the pool from %d to %d blocks", plain.KVBlocks, totalKVBlocks)
	}
	loraReservedBytesForKV = 8 << 30
	applyKernelLoRAReservation()
	want, _ := m.SettingsReserving(8 << 30)
	if totalKVBlocks != want.KVBlocks || totalKVBlocks >= plain.KVBlocks {
		t.Errorf("an 8 GiB reservation sized the pool at %d blocks; the kernel says %d (plain %d)",
			totalKVBlocks, want.KVBlocks, plain.KVBlocks)
	}
}

// e2eMean reads the cluster's mean end-to-end latency from a run's stdout.
func e2eMean(t *testing.T, stdout string) float64 {
	t.Helper()
	i := strings.Index(stdout, `"instance_id": "cluster"`)
	if i < 0 {
		i = 0
	}
	const key = `"e2e_mean_ms": `
	j := strings.Index(stdout[i:], key)
	if j < 0 {
		t.Fatalf("no e2e_mean_ms in:\n%s", stdout)
	}
	rest := stdout[i+j+len(key):]
	v, err := strconv.ParseFloat(strings.TrimRight(rest[:strings.IndexAny(rest, ",\n")], " "), 64)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// Speculation comes from the scenario: a scenario that drafts tokens requires the acceptance
// rate (a property of the workload, not the deployment) and is refused naming the scenario
// without it; with it, accepting more of each draft never makes requests slower -- every step
// pays the same verify width and advances further.
func TestRunCmd_KernelBackend_SpeculationFromTheScenario(t *testing.T) {
	base := []string{"run", "--scenarios", pdScenarios, "--scenario", "glm-5-h200-tp8-mtp3.yaml",
		"--num-requests", "24", "--rate", "6", "--seed", "2"}
	if _, stderr, err := runKernelCLI(t, base...); err == nil || !strings.Contains(stderr, "drafts 3 tokens") {
		t.Fatalf("a drafting scenario without an acceptance rate was not refused naming it: %v\n%s", err, stderr)
	}
	var prev float64
	for i, acc := range []string{"0.0", "0.4", "0.8", "1.0"} {
		out, stderr, err := runKernelCLI(t, append(base, "--speculative-acceptance-rate", acc)...)
		if err != nil {
			t.Fatalf("acceptance %s: %v\n%s", acc, err, stderr)
		}
		e2e := e2eMean(t, out)
		if i > 0 && e2e > prev {
			t.Errorf("raising acceptance to %s lengthened mean E2E from %.3f to %.3f ms", acc, prev, e2e)
		}
		prev = e2e
	}
}

// writeScenarioVariant writes the committed 3P1D fixture, edited by each old->new pair, into a
// fresh scenario directory, so each refusal below is driven by one stated difference.
func writeScenarioVariant(t *testing.T, edits ...string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(pdScenarios, "glm-5-h200-3p1d-ib.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	for i := 0; i+1 < len(edits); i += 2 {
		if !strings.Contains(body, edits[i]) {
			t.Fatalf("fixture has no %q to edit", edits[i])
		}
		body = strings.Replace(body, edits[i], edits[i+1], 1)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// What a disaggregated kernel run refuses, each named, rather than simulating an engine the
// scenario did not describe; and what it accepts at the boundary.
func TestRunCmd_KernelBackend_DisaggregatedRefusals(t *testing.T) {
	pd := func(dir string, extra ...string) []string {
		return append([]string{"run", "--scenarios", dir, "--scenario", "variant.yaml", "--pd-decider", "always",
			"--num-requests", "6", "--rate", "3"}, extra...)
	}
	topology := []string{"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1"}
	plain := writeScenarioVariant(t)
	for _, tt := range []struct {
		name, want string
		args       []string
	}{
		{"a disaggregated scenario run without a P/D topology", "needs a P/D topology",
			pd(plain)},
		{"a pool stating no max_model_len", "states no engine.max_model_len",
			pd(writeScenarioVariant(t, "      max_model_len: 32768\n      cudagraph_mode: PIECEWISE\n      gpu_memory_utilization: 0.9\n  - role: decode",
				"      cudagraph_mode: PIECEWISE\n      gpu_memory_utilization: 0.9\n  - role: decode"), topology...)},
		{"pools drafting differently", "one speculative configuration applies to the run",
			pd(writeScenarioVariant(t, "      gpu_memory_utilization: 0.9\n\npd_transfer",
				"      gpu_memory_utilization: 0.9\n      speculative:\n        method: mtp\n        num_spec_tokens: 2\n\npd_transfer"), topology...)},
		{"pools with no fabric between them", "cluster.fabric",
			pd(writeScenarioVariant(t, "  fabric: ib-400g\n", ""), topology...)},
		{"a per-role MoE comm flag", "unknown flag: --decode-moe-comm-backend",
			pd(plain, append(topology, "--decode-moe-comm-backend", "naive")...)},
		{"more decode instances than the pool holds", "decode pool holds 1 rank(s)",
			pd(plain, "--num-instances", "5", "--prefill-instances", "3", "--decode-instances", "2")},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, stderr, err := runKernelCLI(t, tt.args...)
			if err == nil || !strings.Contains(stderr, tt.want) {
				t.Errorf("want a refusal naming %q, got err=%v\n%s", tt.want, err, lastLines(stderr, 3))
			}
		})
	}
	// The pools hold exactly 3 prefill and 1 decode rank: that topology runs.
	if out, stderr, err := runKernelCLI(t, pd(plain, topology...)...); err != nil ||
		!strings.Contains(out, `"completed_requests": 6`) {
		t.Errorf("a topology exactly filling the pools was not served: %v\n%s", err, lastLines(stderr, 3))
	}
}

func lastLines(s string, n int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	return strings.Join(lines[max(0, len(lines)-n):], "\n")
}

// A disaggregated run's exported trace, replayed on the same scenario, reproduces the run
// (INV-13): replay opens the same pools, prices the same handoffs, and places the same ranks.
func TestReplayCmd_KernelBackend_ReproducesADisaggregatedRun(t *testing.T) {
	prefix := filepath.Join(t.TempDir(), "trace")
	common := []string{"--scenarios", pdScenarios, "--scenario", "glm-5-h200-3p1d-ib.yaml", "--pd-decider", "always",
		"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1", "--seed", "9",
		"--horizon", "600000000"}
	runOut, stderr, err := runKernelCLI(t, append([]string{"run", "--num-requests", "24", "--rate", "6",
		"--trace-output", prefix}, common...)...)
	if err != nil || !strings.Contains(runOut, `"completed_requests": 24`) {
		t.Fatalf("run: %v\n%s\n%s", err, runOut, lastLines(stderr, 3))
	}
	repOut, stderr, err := runKernelCLI(t, append([]string{"replay", "--trace-header", prefix + ".yaml",
		"--trace-data", prefix + ".csv"}, common...)...)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 3))
	}
	if repOut != runOut {
		t.Errorf("the disaggregated run's replay differs from the run\n--- run\n%s\n--- replay\n%s", runOut, repOut)
	}
}

// A disaggregated scenario whose pools both draft tokens: the decode sub-request carries the
// run's speculative configuration. Its exported trace replays byte-identically (INV-13), the
// output-token total is the workload's at every acceptance rate (speculation changes how many
// steps a request takes, never how many tokens it emits), and accepting more of each draft
// never lengthens mean E2E.
func TestRunCmd_KernelBackend_DisaggregatedSpeculation(t *testing.T) {
	const draft = "      gpu_memory_utilization: 0.9\n      speculative:\n        method: mtp\n        num_spec_tokens: 3\n"
	dir := writeScenarioVariant(t,
		"      gpu_memory_utilization: 0.9\n  - role: decode", draft+"  - role: decode",
		"      gpu_memory_utilization: 0.9\n\npd_transfer", draft+"\npd_transfer")
	common := []string{"--scenarios", dir, "--scenario", "variant.yaml", "--pd-decider", "always",
		"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1", "--seed", "4",
		"--horizon", "600000000"}
	var prevE2E, firstE2E float64
	var tokens int
	for i, acc := range []string{"0.0", "0.5", "1.0"} {
		prefix := filepath.Join(t.TempDir(), "trace")
		args := append([]string{"run", "--num-requests", "16", "--rate", "4", "--trace-output", prefix,
			"--speculative-acceptance-rate", acc}, common...)
		runOut, stderr, err := runKernelCLI(t, args...)
		if err != nil || !strings.Contains(runOut, `"completed_requests": 16`) {
			t.Fatalf("acceptance %s: %v\n%s", acc, err, lastLines(stderr, 3))
		}
		repOut, stderr, err := runKernelCLI(t, append([]string{"replay", "--trace-header", prefix + ".yaml",
			"--trace-data", prefix + ".csv", "--speculative-acceptance-rate", acc}, common...)...)
		if err != nil {
			t.Fatalf("acceptance %s replay: %v\n%s", acc, err, lastLines(stderr, 3))
		}
		if repOut != runOut {
			t.Errorf("acceptance %s: the replay differs from the run\n--- run\n%s\n--- replay\n%s", acc, runOut, repOut)
		}
		got := clusterMetricInt(t, runOut, "total_output_tokens")
		e2e := e2eMean(t, runOut)
		if i > 0 {
			if got != tokens {
				t.Errorf("acceptance %s emitted %d output tokens, acceptance 0.0 emitted %d", acc, got, tokens)
			}
			if e2e > prevE2E {
				t.Errorf("raising acceptance to %s lengthened mean E2E from %.3f to %.3f ms", acc, prevE2E, e2e)
			}
		}
		if i == 0 {
			firstE2E = e2e
		}
		tokens, prevE2E = got, e2e
	}
	// Non-vacuity: a decode pool that ignored the draft would price every acceptance alike.
	if prevE2E >= firstE2E {
		t.Errorf("accepting every draft left mean E2E at %.3f ms against %.3f with none; the "+
			"decode pool is not speculating", prevE2E, firstE2E)
	}
}

// A disaggregated run offloads KV like a colocated one, each pool sizing and pricing the same
// tiers by its own kernel: with the tiered offload and with the legacy CPU tier the run
// completes, and its exported trace replays byte-identically (INV-13).
func TestRunCmd_KernelBackend_DisaggregatedKVOffload(t *testing.T) {
	common := []string{"--scenarios", pdScenarios, "--scenario", "glm-5-h200-3p1d-ib.yaml", "--pd-decider", "always",
		"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1", "--seed", "5",
		"--horizon", "600000000"}
	for _, tt := range []struct {
		name    string
		offload []string
	}{
		{"tiered offload", []string{"--kv-offload-config", writeKernelOffloadConfig(t, "")}},
		{"legacy CPU tier", []string{"--kv-cpu-blocks", "2000"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			prefix := filepath.Join(t.TempDir(), "trace")
			args := append(append([]string{"run", "--num-requests", "16", "--rate", "4", "--trace-output", prefix},
				tt.offload...), common...)
			runOut, stderr, err := runKernelCLI(t, args...)
			if err != nil || !strings.Contains(runOut, `"completed_requests": 16`) {
				t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
			}
			replay := append([]string{"replay", "--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv"}, common...)
			if tt.offload[0] == "--kv-cpu-blocks" {
				replay = append(replay, tt.offload...) // the legacy tier is a flag, not a header field
			}
			repOut, stderr, err := runKernelCLI(t, replay...)
			if err != nil {
				t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 3))
			}
			if repOut != runOut {
				t.Errorf("the replay differs from the run\n--- run\n%s\n--- replay\n%s", runOut, repOut)
			}
		})
	}
}

// Each pool of a disaggregated run sizes and prices the run's one offload description by its
// own kernel. The fixture's decode pool caches in bf16 and its prefill pool in fp8, so their
// blocks hold different bytes, and a pool priced by the other's kernel cannot pass:
//
//   - per-block bytes, every tier's service time and the legacy reload charge are each the
//     pool's own kernel's answer;
//   - the run's description is not mutated, and the pools share no tier slice.
func TestKernelPools_ApplyOffload_EachPoolByItsOwnKernel(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	dir := writeScenarioVariant(t, "      cache_dtype: fp8\n      block_size: 64\n      max_num_batched_tokens: 8192\n      max_num_seqs: 256\n      max_model_len: 32768\n      cudagraph_mode: PIECEWISE\n      gpu_memory_utilization: 0.9\n\npd_transfer",
		"      cache_dtype: auto\n      block_size: 64\n      max_num_batched_tokens: 8192\n      max_num_seqs: 256\n      max_model_len: 32768\n      cudagraph_mode: PIECEWISE\n      gpu_memory_utilization: 0.9\n\npd_transfer")
	repos := kernelmodel.Repos{Scenarios: dir, Catalog: catalog, Registry: registry}
	pools := &kernelPools{}
	for _, c := range []struct {
		role deployment.Role
		dst  **kernelmodel.Model
	}{{deployment.RolePrefill, &pools.prefill}, {deployment.RoleDecode, &pools.decode}} {
		m, err := kernelmodel.OpenRole("variant.yaml", repos, c.role)
		if err != nil {
			t.Fatal(err)
		}
		*c.dst = m
	}
	saved := []int64{kvCPUBlocks, blockSizeTokens}
	defer func() { kvCPUBlocks, blockSizeTokens = saved[0], saved[1] }()
	kvCPUBlocks, blockSizeTokens = 100, 64

	run := sim.KVOffloadConfig{Enabled: true, CPUBytesToUse: 1 << 32, PerBlockBytes: 1,
		Tiers: []sim.KVOffloadTier{{DeviceClass: "nvme_gen4"}}}
	var pre, dec cluster.PoolOverrides
	pools.applyOffload(run, &pre, &dec)

	if run.Tiers[0].ServiceTime != nil || run.PerBlockBytes != 1 {
		t.Error("applyOffload mutated the run's offload description")
	}
	ticks := func(m *kernelmodel.Model, tier string, dir kernel.Direction, bytes int64) int64 {
		return max(1, (m.Kernel().TierTime(tier, dir, bytes, 1).Nanoseconds()+999)/1000)
	}
	bytesOf := map[string]int64{}
	for _, c := range []struct {
		name string
		m    *kernelmodel.Model
		o    cluster.PoolOverrides
	}{{"prefill", pools.prefill, pre}, {"decode", pools.decode, dec}} {
		if c.o.KVOffload == nil || c.o.KVTransferTicksPerBlock == nil {
			t.Fatalf("%s pool: no offload override", c.name)
		}
		want := c.m.Kernel().SequenceVariableBytes(64)
		bytesOf[c.name] = want
		if got := c.o.KVOffload.PerBlockBytes; got != want {
			t.Errorf("%s pool: %d bytes per block, its kernel says %d", c.name, got, want)
		}
		if got, w := *c.o.KVTransferTicksPerBlock, ticks(c.m, "cpu_dram", kernel.DirectionFromTier, want); got != w {
			t.Errorf("%s pool: legacy reload %d ticks, its kernel prices %d", c.name, got, w)
		}
		for _, b := range []int64{want, 1 << 20, 1 << 28} {
			if got, w := c.o.KVOffload.Tiers[0].ServiceTime(false, b, 1), ticks(c.m, "nvme_gen4", kernel.DirectionFromTier, b); got != w {
				t.Errorf("%s pool: %d-byte read priced %d, its kernel %d", c.name, b, got, w)
			}
		}
		if err := c.o.Validate(c.name); err != nil {
			t.Errorf("%s pool overrides invalid: %v", c.name, err)
		}
	}
	if bytesOf["prefill"] == bytesOf["decode"] {
		t.Fatalf("fixture lost its point: both pools hold %d bytes per block", bytesOf["prefill"])
	}
	if &pre.KVOffload.Tiers[0] == &dec.KVOffload.Tiers[0] {
		t.Error("the pools share one tier slice, so one pool's pricer overwrites the other's")
	}
}

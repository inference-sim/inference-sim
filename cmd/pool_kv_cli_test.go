package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// A P/D deployment whose prefill pool runs NIXL beside a CPU tier (MultiConnector) and whose decode
// pool runs NIXL alone (#307, #1910 point 3): each pool states its own connector and offload, and the
// deployment-level blocks are the default. Before the change the scenario could not be stated (the
// decoder refused a pool's offload and pd_transfer), and a deployment-level offload gave the decode
// pool a tier it does not have.

const poolKVScenario = "deepseek-v4.1-flash-h200-1p1d-dp8-prefill-cpu1t.yaml"

// tierLinks are the Result's tier-copy links, by node, with their peak concurrent transfers.
func tierLinks(t *testing.T, stdout string) map[string]float64 {
	t.Helper()
	out := map[string]float64{}
	var walk func(any)
	walk = func(v any) {
		switch x := v.(type) {
		case map[string]any:
			for k, e := range x {
				if strings.HasPrefix(k, "link_peak_concurrent_transfers_tier-cpu_dram") {
					out[k], _ = e.(float64)
				}
				walk(e)
			}
		case []any:
			for _, e := range x {
				walk(e)
			}
		}
	}
	walk(resultDoc(t, stdout))
	return out
}

func runPoolKV(t *testing.T, dir, file string) (string, string, error) {
	t.Helper()
	return runKernelCLI(t, "run", "--scenarios", dir, "--scenario", file,
		"--num-instances", "2", "--prefill-instances", "1", "--decode-instances", "1",
		"--rate", "4", "--num-requests", "8", "--seed", "1")
}

// variantDir copies the scenario and the router file it names into a temp dir with edit applied.
func variantDir(t *testing.T, edit func(string) string) string {
	t.Helper()
	root := filepath.Join("..", "testdata", "scenarios")
	body, err := os.ReadFile(filepath.Join(root, "poolkv", poolKVScenario))
	if err != nil {
		t.Fatal(err)
	}
	router, err := os.ReadFile(filepath.Join(root, "router", "pd-always.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "poolkv"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "router"), 0o755); err != nil {
		t.Fatal(err)
	}
	for p, b := range map[string]string{filepath.Join("poolkv", poolKVScenario): edit(string(body)),
		filepath.Join("router", "pd-always.yaml"): string(router)} {
		if err := os.WriteFile(filepath.Join(dir, p), []byte(b), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return filepath.Join(dir, "poolkv")
}

func TestRunCmd_PrefillPoolOffloadsAndDecodePoolDoesNot(t *testing.T) {
	stdout, stderr, err := runPoolKV(t, filepath.Join("..", "testdata", "scenarios", "poolkv"), poolKVScenario)
	if err != nil {
		t.Fatalf("a CPU tier on the prefill pool only did not run: %v\n%s", err, lastLines(stderr, 4))
	}
	links := tierLinks(t, stdout)
	if _, ok := links["link_peak_concurrent_transfers_tier-cpu_dram-to-node-0"]; !ok {
		t.Errorf("tier links %v: the prefill pool (node 0) moved no KV to its CPU tier", links)
	}
	for k := range links {
		if strings.HasSuffix(k, "node-1") {
			t.Errorf("tier link %s: the decode pool (node 1) has no CPU tier, and used one", k)
		}
	}
}

// The deployment-level offload block stays the default: with it and no pool block, both pools run the
// tier, as before the change.
func TestRunCmd_DeploymentLevelOffloadStaysTheDefaultOfEveryPool(t *testing.T) {
	dir := variantDir(t, poolKVDeploymentOffload)
	stdout, stderr, err := runPoolKV(t, dir, poolKVScenario)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 4))
	}
	links := tierLinks(t, stdout)
	for _, k := range []string{"link_peak_concurrent_transfers_tier-cpu_dram-to-node-0", "link_peak_concurrent_transfers_tier-cpu_dram-to-node-1"} {
		if _, ok := links[k]; !ok {
			t.Errorf("tier links %v: want %s under the deployment-level default", links, k)
		}
	}
}

func poolKVDeploymentOffload(s string) string {
	s = strings.Replace(s, "    offload:\n      connector: OffloadingConnector\n      spec: TieringOffloadingSpec\n      eviction_policy: lru\n      tiers:\n        - {tier: cpu_dram, bytes: 1099511627776}\n", "", 1)
	s = strings.Replace(s, "    pd_transfer: {connector: NixlConnector}\n", "", 1)
	return strings.Replace(s, "pd_transfer:\n  connector: MultiConnector\n",
		"pd_transfer:\n  connector: MultiConnector\n\noffload:\n  connector: OffloadingConnector\n  spec: TieringOffloadingSpec\n  eviction_policy: lru\n  tiers:\n    - {tier: cpu_dram, bytes: 1099511627776}\n", 1)
}

// A pool that inherits a MultiConnector but has no tier is refused with the pool and the way out, and a
// pool tier beside a plain NIXL connector is refused naming the pool.
func TestRunCmd_AnInvalidPerPoolCombinationIsStillRefused(t *testing.T) {
	inherits := variantDir(t, func(s string) string {
		return strings.Replace(s, "    pd_transfer: {connector: NixlConnector}\n", "", 1)
	})
	_, stderr, err := runPoolKV(t, inherits, poolKVScenario)
	if err == nil {
		t.Fatal("a decode pool running a MultiConnector over NIXL alone ran")
	}
	for _, want := range []string{"decode pool (pools[1])", "MultiConnector", "NixlConnector"} {
		if !strings.Contains(stderr, want) {
			t.Errorf("the refusal does not name %q:\n%s", want, lastLines(stderr, 3))
		}
	}
	plain := variantDir(t, func(s string) string {
		return strings.Replace(s, "pd_transfer:\n  connector: MultiConnector\n", "pd_transfer:\n  connector: NixlConnector\n", 1)
	})
	_, stderr, err = runPoolKV(t, plain, poolKVScenario)
	if err == nil || !strings.Contains(stderr, "prefill") || !strings.Contains(stderr, "MultiConnector") {
		t.Errorf("a prefill tier beside plain NIXL: err %v\n%s", err, lastLines(stderr, 3))
	}
}

// A trace header holds one offload block, so a P/D run whose pools differ writes no trace rather than
// one a replay could not rebuild (INV-13).
func TestRunCmd_PoolsWithDifferentOffloadWriteNoTrace(t *testing.T) {
	prefix := filepath.Join(t.TempDir(), "trace")
	_, stderr, err := runKernelCLI(t, "run", "--scenarios", filepath.Join("..", "testdata", "scenarios", "poolkv"),
		"--scenario", poolKVScenario, "--num-instances", "2", "--prefill-instances", "1", "--decode-instances", "1",
		"--rate", "4", "--num-requests", "8", "--seed", "1", "--trace-output", prefix)
	if err == nil || !strings.Contains(stderr, "cannot record") {
		t.Errorf("a trace of pools that offload differently: err %v\n%s", err, lastLines(stderr, 3))
	}
}

func TestReplayCmd_PoolsWithDifferentOffloadWriteNoTrace(t *testing.T) {
	dir := t.TempDir()
	header, data := filepath.Join(dir, "input.yaml"), filepath.Join(dir, "input.csv")
	h := workload.TraceHeader{
		Version: workload.TraceVersion, TimeUnit: "microseconds", Mode: "converted",
	}
	records := make([]workload.TraceRecord, 8)
	for i := range records {
		records[i] = workload.TraceRecord{
			RequestID: i, ArrivalTimeUs: int64(i) * 250_000,
			InputTokens: 256, OutputTokens: 16, Status: "ok",
		}
	}
	if err := workload.ExportTraceV2(&h, records, header, data); err != nil {
		t.Fatal(err)
	}
	scenarios := filepath.Join("..", "testdata", "scenarios", "poolkv")
	replay := func(scenarios, header, data string, extra ...string) (string, string, error) {
		t.Helper()
		args := []string{"replay", "--trace-header", header, "--trace-data", data, "--scenarios", scenarios,
			"--scenario", poolKVScenario, "--num-instances", "2", "--prefill-instances", "1",
			"--decode-instances", "1", "--seed", "1"}
		return runKernelCLI(t, append(args, extra...)...)
	}
	stdout, stderr, err := replay(scenarios, header, data)
	if err != nil {
		t.Fatalf("the converted workload should run with the scenario's per-pool offload: %v\n%s", err, lastLines(stderr, 4))
	}
	links := tierLinks(t, stdout)
	if _, ok := links["link_peak_concurrent_transfers_tier-cpu_dram-to-node-0"]; !ok {
		t.Fatalf("the replay did not use the prefill CPU tier: %v", links)
	}

	prefix := filepath.Join(dir, "output")
	stdout, stderr, err = replay(scenarios, header, data, "--trace-output", prefix)
	if err == nil || !strings.Contains(stderr, "--trace-output: the prefill and decode pools run different KV offload") {
		t.Errorf("re-export cannot represent the deployment: err %v\n%s", err, lastLines(stderr, 3))
	}
	if strings.TrimSpace(stdout) != "" {
		t.Error("an unrecordable re-export should fail before producing a Result")
	}
	for _, suffix := range []string{".yaml", ".csv"} {
		if _, err := os.Stat(prefix + suffix); !os.IsNotExist(err) {
			t.Errorf("refused re-export left %s: %v", suffix, err)
		}
	}

	// The same converted input can be exported and replayed when both pools agree.
	symmetric := variantDir(t, poolKVDeploymentOffload)
	prefix = filepath.Join(dir, "symmetric")
	_, stderr, err = replay(symmetric, header, data, "--trace-output", prefix)
	if err != nil {
		t.Fatalf("the common offload can be exported: %v\n%s", err, lastLines(stderr, 4))
	}
	stdout, stderr, err = replay(symmetric, prefix+".yaml", prefix+".csv")
	if err != nil {
		t.Fatalf("the common offload can be replayed: %v\n%s", err, lastLines(stderr, 4))
	}
	links = tierLinks(t, stdout)
	for _, link := range []string{"link_peak_concurrent_transfers_tier-cpu_dram-to-node-0",
		"link_peak_concurrent_transfers_tier-cpu_dram-to-node-1"} {
		if _, ok := links[link]; !ok {
			t.Errorf("replay of the symmetric trace lost %q: %v", link, links)
		}
	}
}

// A tier on the decode pool beside the decode pool's own plain NIXL connector is refused where the
// pools are priced, naming the decode pool and its connector (the prefill pool's MultiConnector must
// not stand in for it).
func TestRunCmd_ADecodePoolTierBesideItsOwnNixlIsRefusedNamingThePool(t *testing.T) {
	dir := variantDir(t, func(s string) string {
		return strings.Replace(s, "    pd_transfer: {connector: NixlConnector}\n",
			"    pd_transfer: {connector: NixlConnector}\n    offload:\n      connector: OffloadingConnector\n      spec: TieringOffloadingSpec\n      tiers:\n        - {tier: cpu_dram, bytes: 1099511627776}\n", 1)
	})
	_, stderr, err := runPoolKV(t, dir, poolKVScenario)
	if err == nil {
		t.Fatal("a decode pool tier beside plain NIXL ran")
	}
	if !strings.Contains(stderr, "offloads KV in the decode pool") || !strings.Contains(stderr, "pd_transfer.connector \\\"NixlConnector") {
		t.Errorf("the refusal does not name the decode pool and its connector:\n%s", lastLines(stderr, 3))
	}
}

// External DP on a P/D deployment is driven when both pools balance externally at the same
// data-parallel width (inference-sim#2699). A deployment that asks for it on one pool only (here the
// prefill pool external, the decode pool left internal) is refused at build time, naming the
// constraint -- not run silently as one collapsed instance while the Result reports external. The
// refusal returns before the simulation loop, so this guards the cmd per-pool resolution
// (poolExternalDP) without the cost of the full external run. The both-pools-external P/D run is
// exercised end-to-end by the attached DP8 proof run, and the decode-completion path it relies on is
// pinned at the driver level (sim.TestExternalDP_RemotePrefilledDecodeCompletes).
func TestRunCmd_PDExternalDPMixedWidthRefused(t *testing.T) {
	dir := variantDir(t, func(s string) string {
		// The first of the two identical pool parallel blocks is the prefill pool's; mark it external
		// and leave the decode pool internal, a mixed deployment the resolver must refuse.
		return strings.Replace(s, "dp: 8, enable_expert_parallel: true}",
			"dp: 8, enable_expert_parallel: true, data_parallel_load_balancing: external}", 1)
	})
	_, stderr, err := runPoolKV(t, dir, poolKVScenario)
	if err == nil {
		t.Fatal("a P/D deployment external on one pool only ran; it must be refused, not run collapsed-internal")
	}
	for _, frag := range []string{"both pools external", "same data-parallel width"} {
		if !strings.Contains(stderr, frag) {
			t.Errorf("the refusal must name the P/D-external width constraint (%q); stderr:\n%s", frag, lastLines(stderr, 4))
		}
	}
}

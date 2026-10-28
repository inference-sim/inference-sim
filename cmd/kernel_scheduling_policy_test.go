package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// mixedPriorityWorkload is a burst of n requests in two SLO classes sharing one engine:
// critical (most urgent) and batch (least), equally often, each input+output tokens long.
func mixedPriorityWorkload(t *testing.T, n, input, output int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "mixed.yaml")
	client := func(id, class string) string {
		return "  - id: " + id + "\n    slo_class: " + class + "\n    rate_fraction: 0.5\n" +
			"    arrival: {process: poisson}\n" +
			"    input_distribution: {type: constant, params: {value: " + strconv.Itoa(input) + "}}\n" +
			"    output_distribution: {type: constant, params: {value: " + strconv.Itoa(output) + "}}\n"
	}
	body := "version: \"2\"\nseed: 11\naggregate_rate: 200\nnum_requests: " + strconv.Itoa(n) + "\nclients:\n" +
		client("crit", "critical") + client("bulk", "batch")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// meanE2EByClass runs `blis run` with a --metrics-path file and returns each SLO class's mean
// end-to-end latency (ms) over its completed requests.
func meanE2EByClass(t *testing.T, args ...string) map[string]float64 {
	t.Helper()
	path := filepath.Join(t.TempDir(), "metrics.json")
	_, stderr, err := runKernelCLI(t, append(args, "--metrics-path", path)...)
	if err != nil {
		t.Fatalf("%v: %v\n%s", args, err, lastLines(stderr, 3))
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var file struct {
		Requests []struct {
			SLOClass string  `json:"slo_class"`
			E2E      float64 `json:"e2e_ms"`
		} `json:"requests"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatal(err)
	}
	sum, n := map[string]float64{}, map[string]int{}
	for _, r := range file.Requests {
		if r.E2E > 0 {
			sum[r.SLOClass] += r.E2E
			n[r.SLOClass]++
		}
	}
	out := map[string]float64{}
	for c, s := range sum {
		out[c] = s / float64(n[c])
	}
	if n["critical"] == 0 || n["batch"] == 0 {
		t.Fatalf("both classes must complete requests, got %v", n)
	}
	return out
}

// A scenario's engine.scheduling_policy is vLLM's, and in vLLM it decides two things: the
// waiting queue's order and the preemption victim. priority is BLIS's priority-fcfs scheduler
// and priority preemption; fcfs is fcfs for both. An explicit --preemption-policy wins, and the
// run's results file records it.
func TestAdoptKernelDeployment_SchedulingPolicySetsQueueOrderAndPreemption(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	setupKernelTestFixtures(t)
	savedScheduler, savedPreemption, savedModel, savedGPU := scheduler, preemptionPolicy, model, gpu
	saved := []any{totalKVBlocks, blockSizeTokens, maxNumSeqs, maxNumBatchedTokens, maxModelLen, tensorParallelism}
	t.Cleanup(func() {
		scheduler, preemptionPolicy, model, gpu = savedScheduler, savedPreemption, savedModel, savedGPU
		totalKVBlocks, blockSizeTokens, maxNumSeqs = saved[0].(int64), saved[1].(int64), saved[2].(int64)
		maxNumBatchedTokens, maxModelLen, tensorParallelism = saved[3].(int64), saved[4].(int64), saved[5].(int)
	})
	for _, tt := range []struct {
		policy, wantScheduler, wantPreemption string
		extra                                 []string
	}{
		{"priority", "priority-fcfs", "priority", nil},
		{"fcfs", "fcfs", "fcfs", nil},
		{"priority", "priority-fcfs", "fcfs", []string{"--preemption-policy", "fcfs"}},
	} {
		dir := writeColocatedVariant(t, "      gpu_memory_utilization: 0.92\n",
			"      gpu_memory_utilization: 0.92\n      scheduling_policy: "+tt.policy+"\n")
		cmd := &cobra.Command{}
		registerSimConfigFlags(cmd)
		if err := cmd.ParseFlags(append([]string{"--scenario", "variant.yaml", "--scenarios", dir,
			"--registry", registry, "--catalog", catalog}, tt.extra...)); err != nil {
			t.Fatal(err)
		}
		adoptKernelDeployment(cmd)
		if scheduler != tt.wantScheduler || preemptionPolicy != tt.wantPreemption {
			t.Errorf("scheduling_policy %s %v: scheduler %q preemption %q, want %q and %q", tt.policy, tt.extra,
				scheduler, preemptionPolicy, tt.wantScheduler, tt.wantPreemption)
		}
		if len(tt.extra) > 0 && (len(scenarioOverrides) != 1 || scenarioOverrides[0].Setting != "preemption_policy" ||
			scenarioOverrides[0].Scenario != "priority" || scenarioOverrides[0].Run != "fcfs") {
			t.Errorf("an explicit --preemption-policy overriding the scenario must be recorded, got %+v", scenarioOverrides)
		}
	}
}

// Under KV pressure, a priority scenario preempts the least urgent request: holding the queue
// order fixed (priority-fcfs) and changing only the victim choice -- the scenario's priority
// preemption against an explicit --preemption-policy fcfs -- critical requests do no worse with
// priority preemption, and batch requests pay for it. Non-vacuous: the run preempts.
func TestRunCmd_PrioritySchedulingPolicyPreemptsTheLeastUrgent(t *testing.T) {
	dir := writeColocatedVariant(t, "      gpu_memory_utilization: 0.92\n",
		"      gpu_memory_utilization: 0.22\n      scheduling_policy: priority\n")
	args := []string{"run", "--scenarios", dir, "--scenario", "variant.yaml", "--seed", "11",
		"--workload-spec", mixedPriorityWorkload(t, 120, 6000, 600)}
	out, stderr, err := runKernelCLI(t, args...)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 3))
	}
	if !strings.Contains(out, `"preemption_count": `) || strings.Contains(out, `"preemption_count": 0,`) {
		t.Fatalf("the fixture must preempt for the victim choice to matter:\n%s", lastLines(out, 40))
	}
	withPriority := meanE2EByClass(t, args...)
	withFCFS := meanE2EByClass(t, append(args, "--preemption-policy", "fcfs")...)
	if withPriority["critical"] > withFCFS["critical"] {
		t.Errorf("priority preemption made critical requests slower: %.1f ms against %.1f with fcfs victims",
			withPriority["critical"], withFCFS["critical"])
	}
	if withPriority["critical"] == withFCFS["critical"] && withPriority["batch"] == withFCFS["batch"] {
		t.Error("the victim choice changed nothing; the test does not exercise preemption")
	}
}

// Each P/D pool runs its own engine's scheduling policy. With the decode pool the bottleneck
// (it admits 2 sequences) and only it stating priority, critical requests finish ahead of
// batch ones; stating priority on the prefill pool instead -- the pools swapped -- does not
// separate them. A run that applied one pool's policy to both, or swapped them, fails.
func TestRunCmd_EachPDPoolRunsItsOwnSchedulingPolicy(t *testing.T) {
	run := func(dir string) map[string]float64 {
		return meanE2EByClass(t, pdPolicyRunArgs(t, dir)...)
	}
	variant := func(prefillPolicy, decodePolicy string) string {
		return pdPolicyVariant(t, prefillPolicy, decodePolicy)
	}
	decodePriority := run(variant("fcfs", "priority"))
	prefillPriority := run(variant("priority", "fcfs"))
	if decodePriority["critical"] >= decodePriority["batch"] {
		t.Errorf("decode pool states priority: critical %.1f ms is not ahead of batch %.1f ms",
			decodePriority["critical"], decodePriority["batch"])
	}
	gap := func(m map[string]float64) float64 { return m["batch"] - m["critical"] }
	if gap(decodePriority) <= gap(prefillPriority) {
		t.Errorf("the decode pool's priority separates classes by %.1f ms, the prefill pool's by %.1f ms; "+
			"the bottleneck pool's policy must decide it", gap(decodePriority), gap(prefillPriority))
	}
}

// Each P/D pool's overrides carry its own engine's queue order and preemption victim, so the
// pools can differ; a run-level --scheduler or --preemption-policy sets both pools instead (the
// overrides leave them to the global) and is recorded once per pool whose engine stated otherwise.
func TestKernelPools_Overrides_CarryEachPoolsOwnSchedulingPolicy(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	dir := writeScenarioVariant(t,
		"      gpu_memory_utilization: 0.9\n  - role: decode",
		"      gpu_memory_utilization: 0.9\n      scheduling_policy: fcfs\n  - role: decode",
		"      gpu_memory_utilization: 0.9\n\npd_transfer",
		"      gpu_memory_utilization: 0.9\n      scheduling_policy: priority\n\npd_transfer")
	saved := []any{kernelScenario, kernelScenarioDir, kernelRegistry, kernelOpened, prefillInstances,
		decodeInstances, numInstances, scheduler, preemptionPolicy, schedulerSource, preemptionSource}
	defer func() {
		kernelScenario, kernelScenarioDir, kernelRegistry = saved[0].(string), saved[1].(string), saved[2].(string)
		kernelOpened, _ = saved[3].(*kernelmodel.Model)
		prefillInstances, decodeInstances, numInstances = saved[4].(int), saved[5].(int), saved[6].(int)
		scheduler, preemptionPolicy = saved[7].(string), saved[8].(string)
		schedulerSource, preemptionSource = saved[9].(string), saved[10].(string)
		scenarioOverrides = nil
	}()
	m, err := kernelmodel.Open("variant.yaml", kernelmodel.Repos{Scenarios: dir, Catalog: catalog, Registry: registry})
	if err != nil {
		t.Fatal(err)
	}
	kernelScenario, kernelScenarioDir, kernelRegistry, kernelOpened = "variant.yaml", dir, registry, m
	prefillInstances, decodeInstances, numInstances = 3, 1, 4

	schedulerSource, preemptionSource, scenarioOverrides = "", "", nil
	pre, dec := openKernelPools(catalog).overrides()
	if pre.Scheduler != "fcfs" || pre.PreemptionPolicy != "fcfs" {
		t.Errorf("prefill pool states fcfs: overrides scheduler %q preemption %q", pre.Scheduler, pre.PreemptionPolicy)
	}
	if dec.Scheduler != "priority-fcfs" || dec.PreemptionPolicy != "priority" {
		t.Errorf("decode pool states priority: overrides scheduler %q preemption %q", dec.Scheduler, dec.PreemptionPolicy)
	}
	if len(scenarioOverrides) != 0 {
		t.Errorf("a run taking each pool's own policy overrides nothing, recorded %+v", scenarioOverrides)
	}

	scheduler, preemptionPolicy = "fcfs", "fcfs"
	schedulerSource, preemptionSource, scenarioOverrides = "--scheduler", "--preemption-policy", nil
	pre, dec = openKernelPools(catalog).overrides()
	if pre.Scheduler != "" || pre.PreemptionPolicy != "" || dec.Scheduler != "" || dec.PreemptionPolicy != "" {
		t.Errorf("a run-level policy must reach both pools through the global, got prefill %+v decode %+v",
			[]string{pre.Scheduler, pre.PreemptionPolicy}, []string{dec.Scheduler, dec.PreemptionPolicy})
	}
	got := map[string]string{}
	for _, o := range scenarioOverrides {
		got[o.Setting] = o.Scenario + "->" + o.Run
	}
	want := map[string]string{
		"scheduler (decode pool)":         "priority-fcfs->fcfs",
		"preemption_policy (decode pool)": "priority->fcfs",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("overrides recorded %v, want only the decode pool's departures %v", got, want)
	}
}

// A P/D run's results file records a run-level --scheduler once per pool it overrides, named
// by pool, and nothing at run level -- which, taken from the first pool, would misstate the
// other's engine. The prefill pool states priority and the decode pool fcfs, so --scheduler
// fcfs departs from the prefill pool only.
func TestRunCmd_PDSchedulerOverrideIsRecordedPerPool(t *testing.T) {
	dir := writeScenarioVariant(t,
		"      gpu_memory_utilization: 0.9\n  - role: decode",
		"      gpu_memory_utilization: 0.9\n      scheduling_policy: priority\n  - role: decode",
		"      gpu_memory_utilization: 0.9\n\npd_transfer",
		"      gpu_memory_utilization: 0.9\n      scheduling_policy: fcfs\n\npd_transfer")
	path := filepath.Join(t.TempDir(), "metrics.json")
	_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml",
		"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1",
		"--num-requests", "4", "--rate", "2", "--scheduler", "fcfs", "--metrics-path", path)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 3))
	}
	var file struct {
		Overrides []struct{ Setting, Scenario, Run, Source string } `json:"overrides"`
	}
	if err := json.Unmarshal([]byte(readFile(t, path)), &file); err != nil {
		t.Fatal(err)
	}
	if len(file.Overrides) != 1 || file.Overrides[0].Setting != "scheduler (prefill pool)" ||
		file.Overrides[0].Scenario != "priority-fcfs" || file.Overrides[0].Run != "fcfs" ||
		file.Overrides[0].Source != "--scheduler" {
		t.Errorf("want one record, the prefill pool's priority-fcfs -> fcfs from --scheduler; got %+v", file.Overrides)
	}
}

// pdPolicyLine is the scenario line stating policy, or nothing for "" (the field omitted).
func pdPolicyLine(policy string) string {
	if policy == "" {
		return ""
	}
	return "      scheduling_policy: " + policy + "\n"
}

// pdPolicyVariant writes the 3P1D fixture with each pool stating the given policy ("" omits
// the field) and a decode pool that admits 2 sequences, so the decode pool is the bottleneck.
func pdPolicyVariant(t *testing.T, prefillPolicy, decodePolicy string) string {
	t.Helper()
	return writeScenarioVariant(t,
		"      gpu_memory_utilization: 0.9\n  - role: decode",
		"      gpu_memory_utilization: 0.9\n"+pdPolicyLine(prefillPolicy)+"  - role: decode",
		"      max_num_seqs: 256\n      max_model_len: 32768\n      cudagraph_mode: PIECEWISE\n      gpu_memory_utilization: 0.9\n\npd_transfer",
		"      max_num_seqs: 2\n      max_model_len: 32768\n      cudagraph_mode: PIECEWISE\n      gpu_memory_utilization: 0.9\n"+
			pdPolicyLine(decodePolicy)+"\npd_transfer")
}

// pdPolicyRunArgs is a 3P1D mixed-priority run of the scenario in dir.
func pdPolicyRunArgs(t *testing.T, dir string) []string {
	return []string{"run", "--scenarios", dir, "--scenario", "variant.yaml",
		"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1", "--seed", "11",
		"--workload-spec", mixedPriorityWorkload(t, 80, 512, 128)}
}

// A pool that omits engine.scheduling_policy runs vLLM's default, fcfs -- never the other
// pool's stated policy. Metamorphic: omitting the field is byte-identical to stating fcfs, in
// either pool, whatever the other pool states. The asymmetric case (prefill priority, decode
// omitted) is the one a run-level default would get wrong: it would hand the decode pool the
// prefill pool's priority. Non-vacuity: decode priority differs from decode fcfs here.
func TestRunCmd_AnOmittedPoolPolicyIsVLLMsDefaultNotTheOtherPools(t *testing.T) {
	stdout := func(prefill, decode string) string {
		out, stderr, err := runKernelCLI(t, pdPolicyRunArgs(t, pdPolicyVariant(t, prefill, decode))...)
		if err != nil {
			t.Fatalf("prefill %q decode %q: %v\n%s", prefill, decode, err, lastLines(stderr, 3))
		}
		return out
	}
	for _, tt := range []struct{ name, omittedP, omittedD, statedP, statedD string }{
		{"decode omitted behind a priority prefill pool", "priority", "", "priority", "fcfs"},
		{"prefill omitted in front of a priority decode pool", "", "priority", "fcfs", "priority"},
		{"both omitted", "", "", "fcfs", "fcfs"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if stdout(tt.omittedP, tt.omittedD) != stdout(tt.statedP, tt.statedD) {
				t.Errorf("omitting the field differs from stating vLLM's default fcfs")
			}
		})
	}
	if stdout("priority", "priority") == stdout("priority", "fcfs") {
		t.Fatal("the decode pool's policy changes nothing here; the case does not distinguish an inherited policy")
	}
}

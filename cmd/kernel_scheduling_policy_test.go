package cmd

import (
	"reflect"
	"testing"

	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

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
		dir := writeColocatedVariant(t, "      max_model_len: 32768\n",
			"      max_model_len: 32768\n      scheduling_policy: "+tt.policy+"\n")
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
	saved := []any{kernelScenario, kernelScenarioDir, registryPath, kernelOpened, prefillInstances,
		decodeInstances, numInstances, scheduler, preemptionPolicy, schedulerSource, preemptionSource}
	defer func() {
		kernelScenario, kernelScenarioDir, registryPath = saved[0].(string), saved[1].(string), saved[2].(string)
		kernelOpened, _ = saved[3].(*kernelmodel.Model)
		prefillInstances, decodeInstances, numInstances = saved[4].(int), saved[5].(int), saved[6].(int)
		scheduler, preemptionPolicy = saved[7].(string), saved[8].(string)
		schedulerSource, preemptionSource = saved[9].(string), saved[10].(string)
		scenarioOverrides = nil
	}()
	m, err := kernelmodel.Open("variant.yaml", reposOf(dir, catalog, registry))
	if err != nil {
		t.Fatal(err)
	}
	kernelScenario, kernelScenarioDir, registryPath, kernelOpened = "variant.yaml", dir, registry, m
	prefillInstances, decodeInstances, numInstances = 3, 1, 4

	schedulerSource, preemptionSource, scenarioOverrides = "", "", nil
	pre, dec := openKernelPools().overrides()
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
	pre, dec = openKernelPools().overrides()
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

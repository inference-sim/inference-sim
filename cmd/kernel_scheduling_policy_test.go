package cmd

import (
	"testing"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/engines/vllm/core/kernelmodel"
)

// A scenario's engine.scheduling_policy is vLLM's, and in vLLM it decides two things: the
// waiting queue's order and the preemption victim. priority is BLIS's priority-fcfs scheduler
// and priority preemption; fcfs is fcfs for both. The scenario is the only source (#2037), so
// the run records no override.
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
	}{
		{"priority", "priority-fcfs", "priority"},
		{"fcfs", "fcfs", "fcfs"},
	} {
		dir := writeColocatedVariant(t, "      max_model_len: 32768\n",
			"      max_model_len: 32768\n      scheduling_policy: "+tt.policy+"\n")
		cmd := &cobra.Command{}
		registerSimConfigFlags(cmd)
		if err := cmd.ParseFlags([]string{"--scenario", "variant.yaml", "--scenarios", dir,
			"--registry", registry, "--catalog", catalog}); err != nil {
			t.Fatal(err)
		}
		adoptKernelDeployment(cmd)
		if scheduler != tt.wantScheduler || preemptionPolicy != tt.wantPreemption {
			t.Errorf("scheduling_policy %s: scheduler %q preemption %q, want %q and %q", tt.policy,
				scheduler, preemptionPolicy, tt.wantScheduler, tt.wantPreemption)
		}
		if len(scenarioOverrides) != 0 {
			t.Errorf("scheduling_policy %s: a run taking the scenario's policy overrides nothing, recorded %+v",
				tt.policy, scenarioOverrides)
		}
	}
}

// Each P/D pool's overrides carry its own engine's queue order and preemption victim, so the
// pools can differ. No run-level setting replaces them (#2037).
func TestKernelPools_Overrides_CarryEachPoolsOwnSchedulingPolicy(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	dir := writeScenarioVariant(t,
		"      compilation_config: {cudagraph_mode: PIECEWISE}\n  - role: decode",
		"      compilation_config: {cudagraph_mode: PIECEWISE}\n      scheduling_policy: fcfs\n  - role: decode",
		"      compilation_config: {cudagraph_mode: PIECEWISE}\n\npd_transfer",
		"      compilation_config: {cudagraph_mode: PIECEWISE}\n      scheduling_policy: priority\n\npd_transfer")
	saved := []any{kernelScenario, kernelScenarioDir, registryPath, kernelOpened, prefillInstances,
		decodeInstances, numInstances, scheduler, preemptionPolicy}
	defer func() {
		kernelScenario, kernelScenarioDir, registryPath = saved[0].(string), saved[1].(string), saved[2].(string)
		kernelOpened, _ = saved[3].(*kernelmodel.Model)
		prefillInstances, decodeInstances, numInstances = saved[4].(int), saved[5].(int), saved[6].(int)
		scheduler, preemptionPolicy = saved[7].(string), saved[8].(string)
		scenarioOverrides = nil
	}()
	m, err := kernelmodel.Open("variant.yaml", reposOf(dir, catalog, registry))
	if err != nil {
		t.Fatal(err)
	}
	kernelScenario, kernelScenarioDir, registryPath, kernelOpened = "variant.yaml", dir, registry, m
	prefillInstances, decodeInstances, numInstances = 3, 1, 4

	scenarioOverrides = nil
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
}

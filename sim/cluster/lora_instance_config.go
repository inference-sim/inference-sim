package cluster

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// VLLMAllowedMaxLoRARanks is the set vLLM accepts for --max-lora-rank
// (vllm/config/lora.py, MaxLoRARanks). A per-instance cap outside it is a deployment
// vLLM would refuse to start, so the simulator refuses it too.
var VLLMAllowedMaxLoRARanks = []int{1, 8, 16, 32, 64, 128, 256, 320, 512}

// ValidateLoRAInstanceConfig checks DeploymentConfig.LoRAInstanceMaxRank and
// LoRAInstanceCapacity (config-time rank coupling). Both absent is always valid and
// inert (R20). Otherwise, in this order, the first violation is returned:
//
//  1. both lists are set, or neither;
//  2. the LoRA subsystem is active (registry non-nil);
//  3. each list has exactly NumInstances entries;
//  4. no node pools and no PD pools: those paths size KV per pool or per role, and a
//     per-instance reservation there is not implemented;
//  5. KV capacity is auto-calculated (KVAutoCalc.Enabled): with an explicit
//     --total-kv-blocks the per-instance reservation would be silently ignored;
//  6. every rank is in VLLMAllowedMaxLoRARanks and every capacity is >= 1.
//
// The per-adapter checks (a seeded or scheduled adapter's rank within its instance's cap,
// and the per-instance count within its capacity) run in validatePlacementMap, which
// ValidateLoRAPlacement and ValidateLoRAPlacementSchedule share.
func ValidateLoRAInstanceConfig(dc DeploymentConfig, registry sim.AdapterRegistry) error {
	ranks, caps := dc.LoRAInstanceMaxRank, dc.LoRAInstanceCapacity
	if len(ranks) == 0 && len(caps) == 0 {
		return nil
	}
	if len(ranks) == 0 || len(caps) == 0 {
		return fmt.Errorf("lora_instance_max_rank and lora_instance_capacity must be set together "+
			"(got %d and %d entries)", len(ranks), len(caps))
	}
	if registry == nil {
		return fmt.Errorf("lora_instance_max_rank set but LoRA disabled: the adapter subsystem is " +
			"inactive (no adapters/capacity configured)")
	}
	if len(ranks) != dc.NumInstances {
		return fmt.Errorf("lora_instance_max_rank has %d entries, want one per instance (%d)",
			len(ranks), dc.NumInstances)
	}
	if len(caps) != dc.NumInstances {
		return fmt.Errorf("lora_instance_capacity has %d entries, want one per instance (%d)",
			len(caps), dc.NumInstances)
	}
	if len(dc.NodePools) > 0 {
		return fmt.Errorf("lora_instance_max_rank is not supported with node_pools: pool placement " +
			"sizes KV per GPU type, and a per-instance LoRA reservation there is not implemented")
	}
	if dc.PrefillInstances > 0 || dc.DecodeInstances > 0 || dc.SharedInstances > 0 || dc.EncodeInstances > 0 {
		return fmt.Errorf("lora_instance_max_rank is not supported with PD disaggregation: pool " +
			"roles override KV capacity, and a per-instance LoRA reservation there is not implemented")
	}
	if !dc.KVAutoCalc.Enabled {
		return fmt.Errorf("lora_instance_max_rank requires auto-calculated KV capacity: with an " +
			"explicit --total-kv-blocks (or no extractable KV params) each instance's LoRA " +
			"reservation could not be subtracted from its KV budget")
	}
	for i, r := range ranks {
		if !isVLLMMaxLoRARank(r) {
			return fmt.Errorf("lora_instance_max_rank[%d] = %d is not a rank vLLM accepts %v",
				i, r, VLLMAllowedMaxLoRARanks)
		}
	}
	for i, c := range caps {
		if c < 1 {
			return fmt.Errorf("lora_instance_capacity[%d] = %d, must be >= 1", i, c)
		}
	}
	return nil
}

func isVLLMMaxLoRARank(r int) bool {
	for _, a := range VLLMAllowedMaxLoRARanks {
		if r == a {
			return true
		}
	}
	return false
}

// ValidateLoRADeployment runs every LoRA construction-time check in its fixed order:
// the per-instance configuration first (so a placement is judged against the caps it
// will run under), then the t=0 placement, then the schedule. It builds the read-only
// registry once. NewClusterSimulator panics on its error (library layer); cmd calls it
// first and exits with logrus.Fatalf, so a user sees a message rather than a stack.
func ValidateLoRADeployment(dc DeploymentConfig) error {
	registry, err := sim.BuildAdapterRegistry(dc.ToSimConfig())
	if err != nil {
		return err
	}
	if err := ValidateLoRAInstanceConfig(dc, registry); err != nil {
		return err
	}
	if err := ValidateLoRAPlacement(dc, registry); err != nil {
		return err
	}
	return ValidateLoRAPlacementSchedule(dc, registry)
}

// instanceCapacity returns instance idx's slot count: its per-instance capacity when
// configured, else the cluster-wide AdapterCapacity. Caller guarantees AdapterCapacity
// is non-nil (the registry, and so any validation, exists only when it is set).
func (d DeploymentConfig) instanceCapacity(idx int) int {
	if len(d.LoRAInstanceCapacity) > 0 {
		return d.LoRAInstanceCapacity[idx]
	}
	return *d.AdapterCapacity
}

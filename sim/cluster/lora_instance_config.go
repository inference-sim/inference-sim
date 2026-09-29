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
//  6. every rank is in VLLMAllowedMaxLoRARanks and every capacity is >= 1;
//  7. every instance's KV can be sized net of its reservation (sizeLoRAInstance).
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
	// Size every instance now, on a copy, so a GPU with no memory figure or a reservation
	// the GPU cannot hold is an error the CLI reports, not a construction panic. PD pools
	// are refused above, so every instance runs the deployment's own SimConfig.
	for i := range ranks {
		simCfg := dc.SimConfig
		if _, _, err := sizeLoRAInstance(&simCfg, dc, i); err != nil {
			return err
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

// sizeLoRAInstance specializes simCfg to instance idx's slot configuration and computes
// that instance's reservation and KV blocks, without installing the blocks. It is the one
// computation behind both ValidateLoRAInstanceConfig (on a copy, so the CLI can report a
// failure) and applyLoRAInstanceConfig (on the instance's own config). A fresh pointer
// pair is installed, so no instance shares a mutable field with another or with the
// DeploymentConfig (R8). Caller guarantees the lists are set and of full length.
func sizeLoRAInstance(simCfg *sim.SimConfig, d DeploymentConfig, idx int) (reserved, blocks int64, err error) {
	rank, capacity := d.LoRAInstanceMaxRank[idx], d.LoRAInstanceCapacity[idx]
	simCfg.InstanceMaxRank = &rank
	simCfg.AdapterCapacity = &capacity

	ac, err := sim.BuildAdapterCost(*simCfg)
	if err != nil || ac == nil {
		return 0, 0, fmt.Errorf("instance %d: per-instance LoRA cost model: err=%v, active=%v", idx, err, ac != nil)
	}
	kv := d.KVAutoCalc
	kv.AdapterReservedBytes = int64(ac.AdapterReservedBytes()) // NewCostModel caps it at 1e18
	gpuMemoryGiB := simCfg.HWConfig.MemoryGiB
	if gpuMemoryGiB <= 0 {
		return 0, 0, fmt.Errorf("instance %d: per-instance LoRA configuration needs the GPU's "+
			"memory to size KV, but hardware MemoryGiB is %v", idx, gpuMemoryGiB)
	}
	blocks, err = perInstanceKVBlocks(simCfg, gpuMemoryGiB, kv)
	if err != nil {
		return 0, 0, fmt.Errorf("instance %d: per-instance KV sizing with max_lora_rank=%d, "+
			"capacity=%d (reservation %d bytes) failed: %v", idx, rank, capacity, kv.AdapterReservedBytes, err)
	}
	return kv.AdapterReservedBytes, blocks, nil
}

// applyLoRAInstanceConfig specializes instance idx's SimConfig to its own slot
// configuration and installs the KV blocks sizeLoRAInstance computes from its own
// reservation. No-op when the per-instance lists are absent (INV-6).
//
// MaxModelLen is re-derived per instance: when the CLI supplied the pre-cap value
// (LoRAInstanceMaxModelLen), the instance starts from it and setInstanceKVBlocks caps it
// by this instance's own KV budget, rather than keeping a cap the CLI computed from the
// cluster-wide reservation.
//
// A failure panics, never falling back to the inherited global capacity as
// applyPerInstanceKVCapacity does for node pools: that would run the instance with a
// reservation it never subtracted. ValidateLoRAInstanceConfig runs the same sizing first,
// so the panic is reachable only by a caller that skipped validation.
//
// It returns the instance's echo (PreemptionCount still zero; LoRAInstanceEchoes fills it
// after the run), or nil when inert.
func applyLoRAInstanceConfig(simCfg *sim.SimConfig, d DeploymentConfig, idx int, id InstanceID) *sim.LoRAInstanceEcho {
	if len(d.LoRAInstanceMaxRank) == 0 {
		return nil
	}
	reserved, blocks, err := sizeLoRAInstance(simCfg, d, idx)
	if err != nil {
		panic(fmt.Sprintf("ClusterSimulator: %s: %v", id, err))
	}
	if d.LoRAInstanceMaxModelLen > 0 {
		simCfg.MaxModelLen = d.LoRAInstanceMaxModelLen
	}
	setInstanceKVBlocks(simCfg, blocks, simCfg.HWConfig.MemoryGiB, simCfg.GPU)
	return &sim.LoRAInstanceEcho{
		InstanceID:           string(id),
		MaxLoRARank:          *simCfg.InstanceMaxRank,
		AdapterCapacity:      *simCfg.AdapterCapacity,
		AdapterReservedBytes: reserved,
		TotalKVBlocks:        simCfg.TotalKVBlocks,
		MaxModelLen:          simCfg.MaxModelLen,
	}
}

// LoRAInstanceEchoes returns each instance's per-instance LoRA configuration with its
// preemption count, in construction order, or nil when the per-instance lists were not
// set. Panics before Run(), like PerInstanceMetrics, since the counts are post-run.
func (c *ClusterSimulator) LoRAInstanceEchoes() []sim.LoRAInstanceEcho {
	if !c.hasRun {
		panic("ClusterSimulator.LoRAInstanceEchoes() called before Run()")
	}
	var out []sim.LoRAInstanceEcho
	for _, inst := range c.instances {
		if inst.loraEcho == nil {
			continue
		}
		e := *inst.loraEcho
		e.PreemptionCount = inst.Metrics().PreemptionCount
		out = append(out, e)
	}
	return out
}

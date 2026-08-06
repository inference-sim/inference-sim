package cluster

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/sirupsen/logrus"
)

// KVAutoCalcConfig carries the inputs that latency.CalculateKVBlocks needs beyond
// what an instance's sim.SimConfig already holds, so a node-pool-placed instance can
// recompute its KV-block capacity from its ACTUAL placed GPU memory rather than
// inheriting the single global capacity computed once from the --hardware GPU (#1522).
//
// The instance's own SimConfig already supplies ModelConfig, TP, DP, BlockSizeTokens,
// and MaxModelLen; this struct supplies the remaining CalculateKVBlocks arguments
// (memory utilization, model architecture params, LoRA reservation) which are not
// part of SimConfig. Mirrors how HWConfigByGPU (issue #893) was threaded onto
// DeploymentConfig to carry per-GPU execution calibration into the placement sites.
//
// Zero value is inert: Enabled=false ⇒ no recalculation ⇒ every instance keeps its
// inherited global TotalKVBlocks ⇒ output byte-identical to a pre-#1522 build (INV-6).
// The CLI sets Enabled only when --total-kv-blocks was NOT explicitly supplied
// (auto-calc active) AND an analytical backend with extractable KV params is in use.
type KVAutoCalcConfig struct {
	// Enabled gates the per-instance recalculation. False ⇒ no-op (global capacity
	// preserved). True only when capacity is auto-calculated (no explicit
	// --total-kv-blocks); an explicit global override keeps a uniform capacity.
	Enabled bool
	// GPUMemoryUtilization is the fraction of GPU HBM available for KV cache, in
	// (0, 1.0] (the --gpu-memory-utilization flag).
	GPUMemoryUtilization float64
	// Params holds the HF-derived model-architecture parameters (MoE indicators,
	// activation type, embedding tying, expert FFN dims) required by CalculateKVBlocks.
	Params latency.KVCapacityParams
	// AdapterReservedBytes is the static LoRA adapter HBM reservation subtracted from
	// KV capacity (0 ⇒ none). Matches the global/per-pool CLI auto-calc reservation.
	AdapterReservedBytes int64
}

// applyPerInstanceKVCapacity recomputes simCfg.TotalKVBlocks (and, if needed, caps
// simCfg.MaxModelLen) for an instance placed on a GPU with gpuMemoryGiB of HBM.
//
// It runs at each node-pool placement site AFTER the GPU-authoritative HWConfig
// override (issue #893), giving the placed GPU authority over KV capacity just as it
// already has authority over execution calibration — so a mixed H100+L40S pool no
// longer forces every instance onto the global GPU's capacity (INV-P2-1).
//
// Behavior:
//   - cfg.Enabled == false               → no-op (global capacity preserved).
//   - gpuMemoryGiB <= 0                   → warn, keep global capacity (R1: observable
//     fallback, never silent; validation should prevent this, defensive here).
//   - CalculateKVBlocks returns an error  → warn, keep global capacity (e.g. a GPU too
//     small to hold the model; never panics on user/config input).
//   - success                            → set TotalKVBlocks to the per-GPU value; if
//     MaxModelLen is set and the new (smaller) capacity cannot hold it, cap MaxModelLen
//     to newBlocks*blockSize (mirrors the CLI per-pool auto-cap) so the instance still
//     constructs instead of failing NewSimulator's KV-too-small check.
//
// CalculateKVBlocks reads only hc.MemoryGiB from the HardwareCalib, so a calib carrying
// just the placed GPU memory is sufficient. gpuType is used only for log context.
func applyPerInstanceKVCapacity(simCfg *sim.SimConfig, gpuMemoryGiB float64, cfg KVAutoCalcConfig, gpuType string) {
	if !cfg.Enabled {
		return
	}
	if gpuMemoryGiB <= 0 {
		logrus.Warnf("[cluster] per-instance KV auto-calc for GPU %q skipped: gpu_memory_gib=%v not positive; "+
			"using inherited total-kv-blocks=%d", gpuType, gpuMemoryGiB, simCfg.TotalKVBlocks)
		return
	}

	hc := sim.HardwareCalib{MemoryGiB: gpuMemoryGiB}
	blocks, err := latency.CalculateKVBlocks(
		simCfg.ModelConfig, hc, simCfg.TP, simCfg.EffectiveDP(),
		simCfg.BlockSizeTokens, cfg.GPUMemoryUtilization, cfg.Params,
		latency.WithAdapterReservedBytes(cfg.AdapterReservedBytes),
	)
	if err != nil {
		logrus.Warnf("[cluster] per-instance KV auto-calc for GPU %q failed: %v; "+
			"using inherited total-kv-blocks=%d", gpuType, err, simCfg.TotalKVBlocks)
		return
	}

	simCfg.TotalKVBlocks = blocks

	// Cap MaxModelLen to the per-GPU KV-feasible maximum when the recomputed (possibly
	// smaller) capacity cannot hold the configured MaxModelLen. Mirrors the CLI per-pool
	// auto-cap (cmd/root.go) so a small-memory GPU constructs successfully rather than
	// tripping NewSimulator's "KV cache too small for MaxModelLen" error.
	if simCfg.MaxModelLen > 0 {
		kvFeasibleMax := blocks * simCfg.BlockSizeTokens
		if kvFeasibleMax < simCfg.MaxModelLen {
			logrus.Infof("[cluster] per-instance KV auto-calc for GPU %q: auto-capped max-model-len=%d "+
				"(pool KV capacity smaller than global)", gpuType, kvFeasibleMax)
			simCfg.MaxModelLen = kvFeasibleMax
		}
	}

	logrus.Infof("[cluster] per-instance KV auto-calc for GPU %q: total-kv-blocks=%d "+
		"(GPU=%.0f GiB, TP=%d, DP=%d)", gpuType, blocks, gpuMemoryGiB, simCfg.TP, simCfg.EffectiveDP())
}

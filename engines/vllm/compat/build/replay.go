// Moved from inference-sim/cmd/kv_offload.go at 972b4abebc9d84130692e349b75da66c51781e16.
// Moved from inference-sim/cmd/frontend.go at 972b4abebc9d84130692e349b75da66c51781e16.
// Moved from inference-sim/cmd/root.go at 972b4abebc9d84130692e349b75da66c51781e16.

package build

import (
	"fmt"
	"reflect"

	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-vllm/compat/engine"
)

// ReconcileReplayKVOffload is replay's header-authoritative reconciliation of the KV-offload
// config (#1587, BC-G6). header is the config the trace header records, translated into the
// engine's terms by the caller (the trace format is the simulator's); flagChanged is whether the
// scenario's offload block or --kv-offload-config stated one, and flagCfg what they resolve to.
//   - A recorded config this engine cannot reconstruct or validate is refused (INV-13, never a
//     silent degrade to single-tier).
//   - A stated config is accepted only if it resolves identical to the header (INV-13
//     "identical flags"); a genuine mismatch is refused.
//   - A stated config that would ADD offload to a trace captured without it is refused, unless
//     allowFlagAdd: an observe trace (mode "real") or an agentic replay records no BLIS
//     deployment, so its offload config comes from the scenario or the flag (#1583, #1937).
//
// It returns the inert zero value when neither the trace nor the caller states a config.
func ReconcileReplayKVOffload(header engine.KVOffloadConfig, flagChanged bool, flagCfg engine.KVOffloadConfig, allowFlagAdd bool) (engine.KVOffloadConfig, error) {
	if header.IsEnabled() {
		if err := header.Validate(); err != nil {
			return engine.KVOffloadConfig{}, fmt.Errorf("blis replay cannot reproduce the trace's kv_offload config: %w (INV-13: never silent degradation)", err)
		}
		// Compare user-facing fields only: PerBlockBytes is DERIVED from the model KV
		// size (not a user knob) and the replay-path flagCfg cannot compute it (the kernel
		// prices it, and reconcile runs before the latency config resolves),
		// so it is zero on the flag side. The header value is authoritative by design
		// (INV-13 cross-host, never re-resolved), so exclude it from the equality gate;
		// otherwise identical flags would spuriously conflict.
		flagCmp, headerCmp := flagCfg, header
		flagCmp.PerBlockBytes, headerCmp.PerBlockBytes = 0, 0
		if flagChanged && !reflect.DeepEqual(flagCmp, headerCmp) {
			return engine.KVOffloadConfig{}, fmt.Errorf("the scenario's offload block or --kv-offload-config conflicts with the kv_offload config recorded in the trace header; on replay the header is authoritative — replay with the scenario and flags the run used")
		}
		return header, nil
	}
	if flagChanged && flagCfg.IsEnabled() {
		if allowFlagAdd {
			// Observe trace (#1583): the operator supplies the offload config the
			// server ran with, to model the observed deployment for the sim side of
			// the hit-rate comparison. PerBlockBytes is 0 here (model not yet
			// resolved); replay derives it after the latency config resolves.
			return flagCfg, nil
		}
		return engine.KVOffloadConfig{}, fmt.Errorf("the trace was captured without a kv_offload config; --kv-offload-config or the scenario's offload block cannot add one on replay of a sim-generated trace (INV-13). Only observe traces (mode \"real\") and agentic replays, which record no BLIS deployment, take their offload config from the scenario or the flag on replay")
	}
	return engine.KVOffloadConfig{}, nil
}

// RequireOffloadForTieredObservation enforces BC-10 (#1583): a trace whose observed KV hit-rate
// was captured from the multi-tier offload family is comparable on replay only if the replay
// reproduces a tiered offload config. A replay with none would produce a GPU-only hit-rate that
// `blis calibrate` then compares against a tiered real number -- silent degradation -- so it is
// refused (INV-13). source is the observation's source label, tiered whether it names the tiered
// family; the GPU-prefix-cache fallback source has no such requirement.
func RequireOffloadForTieredObservation(tiered bool, source string, offloadEnabled bool) error {
	if tiered && !offloadEnabled {
		return fmt.Errorf("the trace's observed_kv_metrics were captured from the tiered KV-offload counters (source=%q) but this replay has no offload config; replay would produce a GPU-only hit-rate that calibrate cannot compare against the tiered observation — pass --kv-offload-config <file> matching the observed deployment (INV-13, never silent degradation)", source)
	}
	return nil
}

// RequireOneOffloadModel refuses the multi-tier offload chain together with the legacy single
// CPU tier (--kv-cpu-blocks): they are distinct offload models (#1590, H1). onReplay names the
// trace's config as the chain's source, as replay's message does.
func RequireOneOffloadModel(offloadEnabled bool, kvCPUBlocks int64, onReplay bool) error {
	if !offloadEnabled || kvCPUBlocks <= 0 {
		return nil
	}
	if onReplay {
		return fmt.Errorf("--kv-cpu-blocks conflicts with the trace's kv_offload config (distinct KV-offload models); set only one")
	}
	return fmt.Errorf("--kv-offload-config and --kv-cpu-blocks are mutually exclusive (distinct KV-offload models); set only one")
}

// FrontendConfigOf is the engine's API-server render stage from the command line's
// --renderer-num-workers and --api-server-count (vLLM renderer_num_workers and
// --api-server-count), refused when either is negative.
func FrontendConfigOf(rendererNumWorkers, apiServerCount int) (engine.FrontendConfig, error) {
	c := engine.FrontendConfig{RendererNumWorkers: rendererNumWorkers, APIServerCount: apiServerCount}
	if err := c.Validate(); err != nil {
		return engine.FrontendConfig{}, fmt.Errorf("--renderer-num-workers / --api-server-count: %v", err)
	}
	return c, nil
}

// DPLockstepRanks is how many consecutive engine replicas of one logical instance step in
// lockstep as one vLLM DP group: dp for an MoE model at dp > 1, which runs as DPEngineCoreProc
// ranks that step in lockstep (vllm/v1/engine/core.py:1386-1395 at v0.31.0) with or without
// expert parallelism (use_all2all is true at dp > 1, vllm/config/parallel.py:738-743), behind
// one API endpoint (#1909); else 0, an independent engine per replica.
func DPLockstepRanks(isMoE bool, dp int) int {
	if !isMoE || dp <= 1 {
		return 0
	}
	return dp
}

// RankNode is the node a replica rank is placed on (the kernel's placement), for
// CheckHybridDPNodeLocal.
type RankNode func(rank int) int

// CheckHybridDPNodeLocal refuses the hybrid DP load balancer for a colocated replica whose
// ranks span nodes. vLLM's hybrid balancer runs one front-end per node over that node's ranks
// (DPLBAsyncMPClient over engine_ranks_managed, vllm/v1/engine/core_client.py:1500-1520;
// api_server_count defaults to data_parallel_size_local, vllm/entrypoints/cli/serve.py:
// 112-120), so a replica across nodes is several endpoints, which BLIS does not model. On one
// node hybrid is the internal balancer. dp is the replica's ranks, instances the run's engine
// count; nodeOf places a rank. A non-hybrid mode, no replica (dp <= 1) or no placement (nil
// nodeOf) is not checked.
func CheckHybridDPNodeLocal(mode string, dp, instances int, nodeOf RankNode) error {
	if mode != string(deployment.DPLBHybrid) || dp <= 1 || nodeOf == nil {
		return nil
	}
	for first := 0; first+dp <= instances; first += dp {
		if a, b := nodeOf(first), nodeOf(first+dp-1); a != b {
			return fmt.Errorf("--dp-load-balancing hybrid: the replica of ranks %d-%d spans nodes %d-%d, and a "+
				"hybrid front-end balances only its own node's ranks (one endpoint per node), which BLIS does "+
				"not model; use internal or external", first, first+dp-1, a, b)
		}
	}
	return nil
}

// OffloadRecord is a resolved KV-offload config as a trace header records it (BC-G6): the
// simulator's trace format carries these fields, and KVOffloadOfRecord reads them back into the
// engine's config. Field for field the trace header's kv_offload block.
type OffloadRecord struct {
	CPUBytesToUse, PerBlockBytes, BlockSize, BlocksPerChunk, TokensPerHash   int64
	EvictionPolicy                                                           string
	OffloadPromptOnly, SelfDescribingKVEvents, SimpleCPUOffload, LazyOffload bool
	Tiers                                                                    []OffloadTierRecord
}

// OffloadTierRecord is one secondary tier of an OffloadRecord.
type OffloadTierRecord struct {
	Type, RootDir                               string
	NReadThreads, NWriteThreads                 int64
	Locality                                    string
	EnableKVEvents, DirectIO                    bool
	DeviceClass                                 string
	ReadBandwidth, WriteBandwidth, BaseLatency  float64
	SaturationQueueDepth                        int64
	SingleTransferFraction, LatencyJitterStddev float64
	CapacityBytes                               int64
}

// KVOffloadOfRecord reconstructs the engine's resolved KV-offload config from a trace's record of
// it (BC-G6), the inert zero value when the trace records none (nil). It is the lossless inverse
// of the record a run writes from the same config. It neither validates nor reconciles:
// ReconcileReplayKVOffload does.
func KVOffloadOfRecord(r *OffloadRecord) engine.KVOffloadConfig {
	if r == nil {
		return engine.KVOffloadConfig{}
	}
	c := engine.KVOffloadConfig{
		Enabled:                true,
		CPUBytesToUse:          r.CPUBytesToUse,
		PerBlockBytes:          r.PerBlockBytes,
		BlockSize:              r.BlockSize,
		BlocksPerChunk:         r.BlocksPerChunk,
		TokensPerHash:          r.TokensPerHash,
		EvictionPolicy:         r.EvictionPolicy,
		OffloadPromptOnly:      r.OffloadPromptOnly,
		SelfDescribingKVEvents: r.SelfDescribingKVEvents,
		Simple:                 r.SimpleCPUOffload,
		LazyOffload:            r.LazyOffload,
	}
	for _, t := range r.Tiers {
		c.Tiers = append(c.Tiers, engine.KVOffloadTier{
			Type:                   t.Type,
			RootDir:                t.RootDir,
			NReadThreads:           t.NReadThreads,
			NWriteThreads:          t.NWriteThreads,
			Locality:               t.Locality,
			EnableKVEvents:         t.EnableKVEvents,
			DirectIO:               t.DirectIO,
			DeviceClass:            t.DeviceClass,
			ReadBandwidth:          t.ReadBandwidth,
			WriteBandwidth:         t.WriteBandwidth,
			BaseLatency:            t.BaseLatency,
			SaturationQueueDepth:   t.SaturationQueueDepth,
			SingleTransferFraction: t.SingleTransferFraction,
			LatencyJitterStddev:    t.LatencyJitterStddev,
			CapacityBytes:          t.CapacityBytes,
		})
	}
	return c
}

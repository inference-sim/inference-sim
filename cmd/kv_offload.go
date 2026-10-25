package cmd

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// kvOffloadConfigPath backs the single new --kv-offload-config flag (H5, #1587).
// Absent ("") ⇒ the offload subsystem is inert and output is byte-identical to a
// build without the feature (BC-G5). Registered on run and replay via the shared
// registerSimConfigFlags (INV-13 parity).
var kvOffloadConfigPath string

// simToHeaderOffload converts a resolved sim.KVOffloadConfig into its trace-header
// serialization (H5, #1587). Returns nil when inert, so a disabled run omits the
// kv_offload header key entirely (BC-G5). Lossless inverse of headerToSimOffload.
func simToHeaderOffload(c sim.KVOffloadConfig) *workload.TraceKVOffloadConfig {
	if !c.IsEnabled() {
		return nil
	}
	h := &workload.TraceKVOffloadConfig{
		CPUBytesToUse:          c.CPUBytesToUse,
		PerBlockBytes:          c.PerBlockBytes,
		BlockSize:              c.BlockSize,
		BlocksPerChunk:         c.BlocksPerChunk,
		TokensPerHash:          c.TokensPerHash,
		EvictionPolicy:         c.EvictionPolicy,
		OffloadPromptOnly:      c.OffloadPromptOnly,
		SelfDescribingKVEvents: c.SelfDescribingKVEvents,
		SimpleCPUOffload:       c.Simple,
		LazyOffload:            c.LazyOffload,
	}
	for _, t := range c.Tiers {
		h.Tiers = append(h.Tiers, workload.TraceKVOffloadTier{
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
	return h
}

// replayRecordsDeployment reports whether a replay's trace records the deployment BLIS ran:
// a BLIS-generated trace does, and INV-13 keeps its header authoritative, so neither the
// scenario's offload block nor --kv-offload-config may add tiers it did not run. An observe
// trace (mode "real") records a server's requests, not BLIS's deployment, and an agentic
// replay (#1937) has no trace at all -- its header is synthesized -- so for both the scenario
// and an agreeing flag supply the tiers.
func replayRecordsDeployment(h workload.TraceHeader, agentic bool) bool {
	return !agentic && h.Mode != "real"
}

// headerToSimOffload reads a trace header's offload block back into the engine's offload
// config (BC-G6): the trace format is the simulator's, so its field mapping lives beside
// simToHeaderOffload, of which it is the lossless inverse. Returns the inert zero value when
// the header carries no offload block. It neither validates nor reconciles the result: the
// engine does that (reconcileReplayKVOffload at the build site).
func headerToSimOffload(h *workload.TraceKVOffloadConfig) sim.KVOffloadConfig {
	if h == nil {
		return kvOffloadOfRecord(nil)
	}
	r := &offloadRecord{
		CPUBytesToUse:          h.CPUBytesToUse,
		PerBlockBytes:          h.PerBlockBytes,
		BlockSize:              h.BlockSize,
		BlocksPerChunk:         h.BlocksPerChunk,
		TokensPerHash:          h.TokensPerHash,
		EvictionPolicy:         h.EvictionPolicy,
		OffloadPromptOnly:      h.OffloadPromptOnly,
		SelfDescribingKVEvents: h.SelfDescribingKVEvents,
		SimpleCPUOffload:       h.SimpleCPUOffload,
		LazyOffload:            h.LazyOffload,
	}
	for _, t := range h.Tiers {
		r.Tiers = append(r.Tiers, offloadTierRecord{
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
	return kvOffloadOfRecord(r)
}

package cmd

import (
	"fmt"
	"reflect"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
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

// reconcileReplayKVOffload resolves the KV-offload config for blis replay. Unlike
// --lora-config (flags-only on replay), the offload config round-trips through the
// trace header, so on replay the HEADER is authoritative (BC-G6):
//   - a recorded config that this binary cannot reconstruct/validate ⇒ logrus.Fatalf
//     (INV-13, never a silent degrade to single-tier);
//   - a --kv-offload-config flag is accepted only if it resolves identical to the
//     header (INV-13 "identical flags"); a genuine mismatch ⇒ Fatalf;
//   - a flag that would ADD offload to a trace captured without it ⇒ Fatalf.
//
// Returns the inert zero value when the trace carries no offload config.
// allowFlagAdd is true for OBSERVE traces (Mode "real"): they carry real-server
// timing, not a sim-generated offload config, so supplying --kv-offload-config to
// model the observed deployment for the sim side of calibration (#1583) is allowed.
// It is false for sim-generated (run/replay) traces, where INV-13 run/replay parity
// forbids a flag adding an offload config the source run did not have.
func reconcileReplayKVOffload(cmd *cobra.Command, headerBlock *workload.TraceKVOffloadConfig, allowFlagAdd bool) sim.KVOffloadConfig {
	// The scenario's offload block is a source on replay as on run: it and the flag resolve
	// together (resolveKVOffloadConfig), and the result must match the header.
	flagChanged := cmd.Flags().Changed("kv-offload-config") || scenarioOffload != nil
	var flagCfg sim.KVOffloadConfig
	if flagChanged {
		// resolveKVOffloadConfig itself logrus.Fatalf's on a malformed flag file.
		flagCfg = resolveKVOffloadConfig(cmd)
	}
	cfg, err := resolveReplayKVOffload(headerBlock, flagChanged, flagCfg, allowFlagAdd)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return cfg
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

// resolveReplayKVOffload is the PURE, error-returning core of replay's
// header-authoritative reconciliation (Deviation #9 — every reject branch is
// unit-testable). See reconcileReplayKVOffload for the policy summary.
func resolveReplayKVOffload(headerBlock *workload.TraceKVOffloadConfig, flagChanged bool, flagCfg sim.KVOffloadConfig, allowFlagAdd bool) (sim.KVOffloadConfig, error) {
	headerOffload := headerToSimOffload(headerBlock)
	if headerOffload.IsEnabled() {
		if err := headerOffload.Validate(); err != nil {
			return sim.KVOffloadConfig{}, fmt.Errorf("blis replay cannot reproduce the trace's kv_offload config: %w (INV-13: never silent degradation)", err)
		}
		// Compare user-facing fields only: PerBlockBytes is DERIVED from the model KV
		// size (not a user knob) and the replay-path flagCfg cannot compute it (the kernel
		// prices it, and reconcile runs before the latency config resolves),
		// so it is zero on the flag side. The header value is authoritative by design
		// (INV-13 cross-host, never re-resolved), so exclude it from the equality gate;
		// otherwise identical flags would spuriously conflict.
		flagCmp, headerCmp := flagCfg, headerOffload
		flagCmp.PerBlockBytes, headerCmp.PerBlockBytes = 0, 0
		if flagChanged && !reflect.DeepEqual(flagCmp, headerCmp) {
			return sim.KVOffloadConfig{}, fmt.Errorf("the scenario's offload block or --kv-offload-config conflicts with the kv_offload config recorded in the trace header; on replay the header is authoritative — replay with the scenario and flags the run used")
		}
		return headerOffload, nil
	}
	if flagChanged && flagCfg.IsEnabled() {
		if allowFlagAdd {
			// Observe trace (#1583): the operator supplies the offload config the
			// server ran with, to model the observed deployment for the sim side of
			// the hit-rate comparison. PerBlockBytes is 0 here (model not yet
			// resolved); replay derives it after the latency config resolves.
			return flagCfg, nil
		}
		return sim.KVOffloadConfig{}, fmt.Errorf("the trace was captured without a kv_offload config; --kv-offload-config or the scenario's offload block cannot add one on replay of a sim-generated trace (INV-13). Only observe traces (mode \"real\") and agentic replays, which record no BLIS deployment, take their offload config from the scenario or the flag on replay")
	}
	return sim.KVOffloadConfig{}, nil
}

// validateObservedKVReplayable enforces BC-10 (#1583): a trace whose observed KV
// hit-rate was captured from the multi-tier offload family (source "tiered") is only
// comparable on replay if this replay reproduces a tiered offload config. If the
// reconciled offload config is NOT enabled, the simulator would produce a GPU-only
// hit-rate that `blis calibrate` would then compare against a tiered real number —
// silent degradation. Fail loudly instead (INV-13). The GPU-prefix-cache fallback
// source has no such requirement (it is a GPU-only observation to begin with).
// Pure and error-returning (Deviation #9) so the guard is unit-testable; the CLI
// caller turns the error into logrus.Fatalf.
func validateObservedKVReplayable(obs *workload.TraceObservedKVMetrics, offloadEnabled bool) error {
	if obs == nil {
		return nil
	}
	if obs.Source == workload.ObservedKVSourceTiered && !offloadEnabled {
		return fmt.Errorf("the trace's observed_kv_metrics were captured from the tiered KV-offload counters (source=%q) but this replay has no offload config; replay would produce a GPU-only hit-rate that calibrate cannot compare against the tiered observation — pass --kv-offload-config <file> matching the observed deployment (INV-13, never silent degradation)", obs.Source)
	}
	return nil
}

// headerToSimOffload reconstructs a resolved sim.KVOffloadConfig from a trace header
// (BC-G6). Returns the inert zero value when the header carries no offload block.
// Lossless inverse of simToHeaderOffload. The caller (replay) runs Validate() on the
// result and logrus.Fatalf's if the recorded config cannot be reproduced.
func headerToSimOffload(h *workload.TraceKVOffloadConfig) sim.KVOffloadConfig {
	if h == nil {
		return sim.KVOffloadConfig{}
	}
	c := sim.KVOffloadConfig{
		Enabled:                true,
		CPUBytesToUse:          h.CPUBytesToUse,
		PerBlockBytes:          h.PerBlockBytes,
		BlockSize:              h.BlockSize,
		BlocksPerChunk:         h.BlocksPerChunk,
		TokensPerHash:          h.TokensPerHash,
		EvictionPolicy:         h.EvictionPolicy,
		OffloadPromptOnly:      h.OffloadPromptOnly,
		SelfDescribingKVEvents: h.SelfDescribingKVEvents,
		Simple:                 h.SimpleCPUOffload,
		LazyOffload:            h.LazyOffload,
	}
	for _, t := range h.Tiers {
		c.Tiers = append(c.Tiers, sim.KVOffloadTier{
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

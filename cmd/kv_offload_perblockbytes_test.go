package cmd

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// #1590 Task 1 / BC-C13: PerBlockBytes is a model-DERIVED field. The replay-path
// flagCfg cannot compute it (no sim/latency import; reconcile runs before the
// latency config resolves), so it is 0 on the flag side while the header carries
// the run-computed value. Reconcile must exclude it from the flag-vs-header
// equality gate (header authoritative) — otherwise identical flags spuriously
// conflict and INV-13 replay breaks.
func TestResolveReplayKVOffload_PerBlockBytesExcluded(t *testing.T) {
	header := &workload.TraceKVOffloadConfig{
		CPUBytesToUse: 1024, PerBlockBytes: 4096, BlockSize: 16, BlocksPerChunk: 1, TokensPerHash: 16,
		EvictionPolicy: "lru", OffloadPromptOnly: true,
		Tiers: []workload.TraceKVOffloadTier{{
			Type: "fs", RootDir: "/mnt", NReadThreads: 16, NWriteThreads: 16,
			DirectIO: true, ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
		}},
	}

	// Identical flags EXCEPT the derived PerBlockBytes (0 on the flag side).
	flag := headerToSimOffload(header)
	flag.PerBlockBytes = 0
	got, err := resolveReplayKVOffload(header, true, flag, false)
	if err != nil {
		t.Fatalf("identical flags with a differing DERIVED PerBlockBytes must NOT conflict (INV-13), got %v", err)
	}
	if got.PerBlockBytes != 4096 {
		t.Fatalf("reconcile must return the header's authoritative PerBlockBytes, got %d", got.PerBlockBytes)
	}

	// A genuine divergence in a user-facing field still conflicts.
	bad := headerToSimOffload(header)
	bad.PerBlockBytes = 0
	bad.CPUBytesToUse = 2048
	if _, err := resolveReplayKVOffload(header, true, bad, false); err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("a real user-facing divergence must still conflict, got %v", err)
	}
}

// #1590 Task 1 / M2: PerBlockBytes round-trips through the trace header.
func TestKVOffloadHeaderConversion_PerBlockBytesRoundTrip(t *testing.T) {
	cfg, err := resolveKVOffload(validBlock(), testDevices(), 16)
	if err != nil {
		t.Fatal(err)
	}
	cfg.PerBlockBytes = 8192 // as cmd/root.go sets it after resolving the model
	back := headerToSimOffload(simToHeaderOffload(cfg))
	if back.PerBlockBytes != 8192 {
		t.Fatalf("PerBlockBytes must round-trip through the trace header, got %d", back.PerBlockBytes)
	}
	if !reflect.DeepEqual(cfg, back) {
		t.Fatalf("sim->header->sim not identity with PerBlockBytes set:\n got  %+v\n want %+v", back, cfg)
	}
}

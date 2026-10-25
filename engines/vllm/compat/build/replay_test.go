// Moved from inference-sim/cmd/kv_offload_test.go at 972b4abebc9d84130692e349b75da66c51781e16.

package build

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/engine"
)

// validRecorded is a resolved offload config as a trace header records it: an explicit
// bandwidth/latency triple, so it validates without a device table.
func validRecorded() engine.KVOffloadConfig {
	return engine.KVOffloadConfig{
		Enabled: true, CPUBytesToUse: 1024, BlockSize: 16, BlocksPerChunk: 1, TokensPerHash: 16,
		EvictionPolicy: "lru", OffloadPromptOnly: true,
		Tiers: []engine.KVOffloadTier{{
			Type: "fs", RootDir: "/mnt", NReadThreads: 16, NWriteThreads: 16,
			DirectIO: true, ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
		}},
	}
}

// On replay the trace header is authoritative (BC-G6): it is used verbatim, a stated config may
// only restate it (PerBlockBytes, derived later, excluded), an unreproducible one is refused, and
// a stated config cannot add offload to a sim-generated trace without one, though an observe or
// agentic replay takes it.
func TestReconcileReplayKVOffload(t *testing.T) {
	recorded := validRecorded()
	if got, err := ReconcileReplayKVOffload(recorded, false, engine.KVOffloadConfig{}, false); err != nil || !reflect.DeepEqual(got, recorded) {
		t.Errorf("header, no flag: %+v %v; want the header verbatim", got, err)
	}
	restated := validRecorded()
	restated.PerBlockBytes = 0
	withBytes := validRecorded()
	withBytes.PerBlockBytes = 4096
	if got, err := ReconcileReplayKVOffload(withBytes, true, restated, false); err != nil || got.PerBlockBytes != 4096 {
		t.Errorf("an agreeing flag (PerBlockBytes aside): %+v %v; want the header's", got, err)
	}
	other := validRecorded()
	other.CPUBytesToUse = 2048
	if _, err := ReconcileReplayKVOffload(recorded, true, other, false); err == nil || !strings.Contains(err.Error(), "conflicts with the kv_offload config recorded in the trace header") {
		t.Errorf("a disagreeing flag: %v", err)
	}
	bad := validRecorded()
	bad.BlockSize = 0
	if _, err := ReconcileReplayKVOffload(bad, false, engine.KVOffloadConfig{}, false); err == nil || !strings.Contains(err.Error(), "cannot reproduce the trace's kv_offload config") {
		t.Errorf("an unreproducible header: %v", err)
	}
	if got, err := ReconcileReplayKVOffload(engine.KVOffloadConfig{}, false, engine.KVOffloadConfig{}, false); err != nil || got.IsEnabled() {
		t.Errorf("no header, no flag: %+v %v; want inert", got, err)
	}
	if _, err := ReconcileReplayKVOffload(engine.KVOffloadConfig{}, true, recorded, false); err == nil || !strings.Contains(err.Error(), "cannot add one on replay of a sim-generated trace") {
		t.Errorf("a flag adding offload to a sim-generated trace: %v", err)
	}
	if got, err := ReconcileReplayKVOffload(engine.KVOffloadConfig{}, true, recorded, true); err != nil || !reflect.DeepEqual(got, recorded) {
		t.Errorf("a flag on an observe or agentic replay: %+v %v; want the flag's", got, err)
	}
	if got, err := ReconcileReplayKVOffload(engine.KVOffloadConfig{}, true, engine.KVOffloadConfig{}, false); err != nil || got.IsEnabled() {
		t.Errorf("a stated but inert config: %+v %v; want inert, no refusal", got, err)
	}
}

// A tiered observed hit-rate needs a replay that reproduces a tiered offload config (BC-10); a
// GPU-prefix-cache observation, or none, needs nothing.
func TestRequireOffloadForTieredObservation(t *testing.T) {
	if err := RequireOffloadForTieredObservation(true, "tiered", false); err == nil || !strings.Contains(err.Error(), `source="tiered"`) {
		t.Errorf("tiered, no offload: %v", err)
	}
	for _, c := range []struct {
		tiered, enabled bool
	}{{true, true}, {false, false}, {false, true}} {
		if err := RequireOffloadForTieredObservation(c.tiered, "x", c.enabled); err != nil {
			t.Errorf("tiered %v, offload %v: %v, want none", c.tiered, c.enabled, err)
		}
	}
}

// The offload chain and the legacy CPU tier are distinct models: both together are refused, each
// command naming the chain's source.
func TestRequireOneOffloadModel(t *testing.T) {
	if err := RequireOneOffloadModel(true, 8, true); err == nil || !strings.Contains(err.Error(), "conflicts with the trace's kv_offload config") {
		t.Errorf("replay, both: %v", err)
	}
	if err := RequireOneOffloadModel(true, 8, false); err == nil || !strings.Contains(err.Error(), "--kv-offload-config and --kv-cpu-blocks are mutually exclusive") {
		t.Errorf("run, both: %v", err)
	}
	for _, c := range []struct {
		enabled bool
		blocks  int64
	}{{false, 8}, {true, 0}, {false, 0}} {
		if err := RequireOneOffloadModel(c.enabled, c.blocks, true); err != nil {
			t.Errorf("offload %v, cpu blocks %d: %v, want none", c.enabled, c.blocks, err)
		}
	}
}

// The render stage takes the two counts as given; a negative one is refused naming both flags.
func TestFrontendConfigOf(t *testing.T) {
	if c, err := FrontendConfigOf(2, 3); err != nil || c.RendererNumWorkers != 2 || c.APIServerCount != 3 {
		t.Errorf("FrontendConfigOf(2, 3) = %+v, %v", c, err)
	}
	for _, c := range [][2]int{{-1, 0}, {0, -1}} {
		if _, err := FrontendConfigOf(c[0], c[1]); err == nil || !strings.Contains(err.Error(), "--renderer-num-workers / --api-server-count") {
			t.Errorf("FrontendConfigOf(%d, %d): %v, want a refusal naming the flags", c[0], c[1], err)
		}
	}
}

// An MoE model at dp > 1 is one lockstep DP group of dp ranks; anything else steps independently.
func TestDPLockstepRanks(t *testing.T) {
	for _, c := range []struct {
		moe  bool
		dp   int
		want int
	}{{true, 4, 4}, {true, 2, 2}, {true, 1, 0}, {false, 4, 0}, {false, 1, 0}} {
		if got := DPLockstepRanks(c.moe, c.dp); got != c.want {
			t.Errorf("DPLockstepRanks(moe %v, dp %d) = %d, want %d", c.moe, c.dp, got, c.want)
		}
	}
}

// Hybrid balancing needs each replica's ranks on one node; other modes, no replica, or no
// placement are not checked.
func TestCheckHybridDPNodeLocal(t *testing.T) {
	perNode := func(ranksPerNode int) RankNode { return func(r int) int { return r / ranksPerNode } }
	if err := CheckHybridDPNodeLocal("hybrid", 4, 8, perNode(4)); err != nil {
		t.Errorf("replicas within nodes: %v", err)
	}
	err := CheckHybridDPNodeLocal("hybrid", 4, 8, perNode(2))
	if err == nil || !strings.Contains(err.Error(), "the replica of ranks 0-3 spans nodes 0-1") {
		t.Errorf("a replica across nodes: %v", err)
	}
	// Only the last rank of the second replica crosses a node, so the check must look at each
	// replica's last rank, and at every replica including the last one.
	lastCrosses := func(r int) int {
		if r == 7 {
			return 2
		}
		return r / 4
	}
	err = CheckHybridDPNodeLocal("hybrid", 4, 8, lastCrosses)
	if err == nil || !strings.Contains(err.Error(), "the replica of ranks 4-7 spans nodes 1-2") {
		t.Errorf("the last replica's last rank across a node: %v", err)
	}
	for _, c := range []struct {
		mode string
		dp   int
		node RankNode
	}{{"internal", 4, perNode(2)}, {"external", 4, perNode(2)}, {"hybrid", 1, perNode(1)}, {"hybrid", 4, nil}} {
		if err := CheckHybridDPNodeLocal(c.mode, c.dp, 8, c.node); err != nil {
			t.Errorf("mode %q dp %d: %v, want no check", c.mode, c.dp, err)
		}
	}
}

// A record reads back into the engine's config field for field, enabled, with each tier; no
// record is the inert config.
func TestKVOffloadOfRecord(t *testing.T) {
	if c := KVOffloadOfRecord(nil); c.IsEnabled() || !reflect.DeepEqual(c, engine.KVOffloadConfig{}) {
		t.Errorf("no record: %+v, want the inert zero value", c)
	}
	r := &OffloadRecord{
		CPUBytesToUse: 1024, PerBlockBytes: 64, BlockSize: 16, BlocksPerChunk: 2, TokensPerHash: 16,
		EvictionPolicy: "lru", OffloadPromptOnly: true, SelfDescribingKVEvents: true,
		SimpleCPUOffload: true, LazyOffload: false,
		Tiers: []OffloadTierRecord{{
			Type: "fs", RootDir: "/mnt", NReadThreads: 3, NWriteThreads: 4, Locality: "node",
			EnableKVEvents: true, DirectIO: true, DeviceClass: "nvme", ReadBandwidth: 7000,
			WriteBandwidth: 5000, BaseLatency: 80, SaturationQueueDepth: 32,
			SingleTransferFraction: 0.5, LatencyJitterStddev: 0.1, CapacityBytes: 1 << 30,
		}},
	}
	want := engine.KVOffloadConfig{
		Enabled: true, CPUBytesToUse: 1024, PerBlockBytes: 64, BlockSize: 16, BlocksPerChunk: 2,
		TokensPerHash: 16, EvictionPolicy: "lru", OffloadPromptOnly: true, SelfDescribingKVEvents: true,
		Simple: true, LazyOffload: false,
		Tiers: []engine.KVOffloadTier{{
			Type: "fs", RootDir: "/mnt", NReadThreads: 3, NWriteThreads: 4, Locality: "node",
			EnableKVEvents: true, DirectIO: true, DeviceClass: "nvme", ReadBandwidth: 7000,
			WriteBandwidth: 5000, BaseLatency: 80, SaturationQueueDepth: 32,
			SingleTransferFraction: 0.5, LatencyJitterStddev: 0.1, CapacityBytes: 1 << 30,
		}},
	}
	if got := KVOffloadOfRecord(r); !reflect.DeepEqual(got, want) {
		t.Errorf("KVOffloadOfRecord:\n got %+v\nwant %+v", got, want)
	}
}

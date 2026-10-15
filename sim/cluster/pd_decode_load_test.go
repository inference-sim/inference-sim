package cluster

import (
	"reflect"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
)

// The decode side's load path (INV-PD-8): the decode engine takes its own prefix-cache hit
// first and NIXL pulls only the rest (sched/scheduler.py:525-537 and 949,
// nixl/pull_scheduler.py:149-177 at v0.31.0); under a MultiConnector the decode engine's CPU
// tier never serves a remotely prefilled prompt, because NIXL, listed first, claims it
// (multi_connector.py:429-448).

// extend is r's prompt followed by extra fresh tokens: a later turn of the same session.
func extend(id string, r *sim.Request, arrival int64, extra, outputLen int) *sim.Request {
	next := holdTestRequest(id, 99, arrival, len(r.InputTokens)+extra, outputLen)
	copy(next.InputTokens, r.InputTokens)
	return next
}

// A later turn on the same decode instance pulls only the blocks its prefix cache lacks, and
// its handoff is priced for those; with decode-side prefix caching off it pulls them all.
func TestPDDecodeLoad_PullsOnlyWhatTheDecodeCacheLacks(t *testing.T) {
	turn1 := holdTestRequest("t1", 1, 0, 160, 4)    // 10 full blocks
	turn2 := extend("t2", turn1, 10_000_000, 96, 4) // 16 blocks, the first 10 shared
	for _, tc := range []struct {
		name       string
		noCache    bool
		wantPulled int64
	}{{"decode prefix cache on", false, 6}, {"decode prefix cache off", true, 16}} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := newTestDisaggDeploymentConfig(2, 1, 1)
			cfg.DecodeOverrides.PrefixCachingDisabled = &tc.noCache
			cfg.TraceLevel = "decisions"
			cs := runHold(t, cfg, cloneRequests([]*sim.Request{turn1, turn2})...)
			ps := parentsByID(cs)
			p := ps["t2"]
			for _, r := range cs.Trace().KVTransfers {
				if r.ParentRequestID == "t2" && (r.NumKVBlocks != 16 || r.PulledKVBlocks != tc.wantPulled) {
					t.Errorf("trace record: %d prompt blocks, %d pulled; want 16 and %d", r.NumKVBlocks, r.PulledKVBlocks, tc.wantPulled)
				}
			}
			if p.NumKVBlocks != 16 || p.PulledKVBlocks != tc.wantPulled {
				t.Fatalf("turn 2: %d prompt blocks, %d pulled; want 16 and %d", p.NumKVBlocks, p.PulledKVBlocks, tc.wantPulled)
			}
			if d, want := p.TransferCompleteTime-p.TransferStartTime, testPDTransferTime(tc.wantPulled*16, "", ""); d != want {
				t.Errorf("turn 2's handoff took %d ticks, the price of %d pulled blocks is %d", d, tc.wantPulled, want)
			}
		})
	}
}

// The local hit stops short of the prompt's last token, as vLLM's does
// (kv_cache_manager.py:290-299): a block-aligned prompt the decode cache holds entirely still
// pulls its last block.
func TestPDDecodeLoad_LocalHitStopsBeforeTheLastToken(t *testing.T) {
	turn1 := holdTestRequest("t1", 1, 0, 160, 4)
	again := extend("t2", turn1, 10_000_000, 0, 4)
	p := parentsByID(runHold(t, newTestDisaggDeploymentConfig(2, 1, 1), turn1, again))["t2"]
	if p.PulledKVBlocks != 1 {
		t.Fatalf("a fully cached 10-block prompt pulled %d blocks, want its last 1", p.PulledKVBlocks)
	}
}

// Metamorphic: with every request disaggregated, giving the decode pool a CPU tier changes
// nothing. Its tier stores what the decode engine computes, but a remotely prefilled prompt
// is NIXL's to load, so the tier never shortens a pull. The decode GPU here is small enough
// that turn 1's blocks are evicted from it before turn 2, so a tier that DID serve the pull
// would shrink turn 2's handoff.
func TestPDDecodeLoad_DecodeCPUTierNeverServesAHandoff(t *testing.T) {
	turn1 := holdTestRequest("t1", 1, 0, 160, 4)
	reqs := []*sim.Request{turn1}
	for i := 0; i < 4; i++ { // fillers that evict turn 1 from the decode GPU
		reqs = append(reqs, holdTestRequest(string(rune('f'+i)), 10+i, int64(i+1)*5_000_000, 160, 4))
	}
	reqs = append(reqs, extend("t2", turn1, 30_000_000, 96, 4))

	run := func(decodeTier bool) *ClusterSimulator {
		cfg := newTestDisaggDeploymentConfig(2, 1, 1)
		blocks := int64(24)
		cfg.DecodeOverrides.TotalKVBlocks = &blocks
		if decodeTier {
			cfg.DecodeOverrides.KVOffload = &sim.KVOffloadConfig{
				Enabled: true, CPUBytesToUse: 1000 * 4096, PerBlockBytes: 4096, BlockSize: 16,
				BlocksPerChunk: 1, TokensPerHash: 16, EvictionPolicy: "lru", OffloadPromptOnly: true,
			}
		}
		return runHold(t, cfg, cloneRequests(reqs)...)
	}
	with, without := run(true), run(false)
	if p := parentsByID(without)["t2"]; p.PulledKVBlocks != 16 {
		t.Fatalf("fixture: turn 1 must be gone from the decode GPU by turn 2 (pulled %d, want 16)", p.PulledKVBlocks)
	}
	mw, mo := with.AggregatedMetrics(), without.AggregatedMetrics()
	if !reflect.DeepEqual(mw.RequestE2Es, mo.RequestE2Es) || !reflect.DeepEqual(mw.RequestTTFTs, mo.RequestTTFTs) {
		t.Fatalf("a decode CPU tier changed a fully disaggregated run:\nwith %v\nwithout %v", mw.RequestE2Es, mo.RequestE2Es)
	}
	if got := with.PDMetrics().CPUTierHitRate["decode"]; got != 0 {
		t.Errorf("decode pool CPU-tier hit rate %v, want 0: no handoff reads it", got)
	}
}

// Blocks a pull is still filling are no prefix hit: vLLM allocates them with
// delay_cache_blocks and caches them only once the pull lands (sched/scheduler.py:1216,
// 3052-3085). Two handoffs of the same prompt from two prefill instances to one decode
// instance, overlapping, both pull every block; once the first has landed, a third pulls only
// the last block.
func TestPDDecodeLoad_InFlightBlocksAreNotAPrefixHit(t *testing.T) {
	cfg := newTestDisaggDeploymentConfig(3, 2, 1)
	cfg.PDTransferTime = func(int64, InstanceID, InstanceID) int64 { return 1_000_000 }
	a := holdTestRequest("a", 1, 0, 160, 4)
	b := extend("b", a, 0, 0, 4)
	c := extend("c", a, 10_000_000, 0, 4)
	ps := parentsByID(runHold(t, cfg, a, b, c))
	if ps["a"].PrefillInstanceID == ps["b"].PrefillInstanceID ||
		ps["a"].TransferStartTime != ps["b"].TransferStartTime {
		t.Fatalf("fixture: a and b must hand off together from different prefill instances")
	}
	for _, id := range []string{"a", "b"} {
		if got := ps[id].PulledKVBlocks; got != 10 {
			t.Errorf("%s pulled %d blocks while the other's pull of the same prompt was in flight, want all 10", id, got)
		}
	}
	if got := ps["c"].PulledKVBlocks; got != 1 {
		t.Errorf("after the pulls landed, c pulled %d blocks, want only the last 1", got)
	}
}

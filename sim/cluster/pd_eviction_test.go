package cluster

import (
	cengine "github.com/inference-sim/blis-vllm/compat/engine"
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestEvict_DisaggregatedAtEachStage: the gateway evicts a disaggregated request in its prefill,
// in its KV transfer, or in its decode. Each eviction takes once and ends the request as evicted,
// releases what that stage held (no KV remains allocated, no transfer is left to promote), and
// keeps INV-1 and INV-PD-3 (every transfer started also completes).
func TestEvict_DisaggregatedAtEachStage(t *testing.T) {
	for _, tc := range []struct {
		stage string
		after int64 // eviction delay after dispatch (µs)
		check func(t *testing.T, p *ParentRequest)
	}{
		// One prefill at a time per engine, all arriving at once: two wait in their prefill
		// engines' queues and are evicted there; the two computing are evicted in transfer.
		{"prefill", 300, func(t *testing.T, p *ParentRequest) {
			if p.PrefillCompleteTime == 0 && p.TransferStartTime != 0 {
				t.Errorf("transfer started at %d without a prefill", p.TransferStartTime)
			}
		}},
		{"transfer", 2_000_000, func(t *testing.T, p *ParentRequest) {
			if p.TransferStartTime == 0 || p.TransferCompleteTime != 0 || p.DecodeEnqueueTime != 0 {
				t.Errorf("transfer started %d, completed %d, decode enqueued %d; want evicted mid-transfer",
					p.TransferStartTime, p.TransferCompleteTime, p.DecodeEnqueueTime)
			}
		}},
		{"decode", 6_000_000, func(t *testing.T, p *ParentRequest) {
			// Decoding: the decode engine had emitted tokens for it (its inter-token record) when
			// the eviction took it. The host's request record keeps what the engine emitted, not
			// the engine's own progress, which stays behind the contract.
			if p.DecodeEnqueueTime == 0 || p.DecodeSubReq == nil || len(p.DecodeSubReq.ITL) == 0 {
				t.Errorf("decode enqueued at %d; want evicted while decoding", p.DecodeEnqueueTime)
			}
		}},
	} {
		t.Run(tc.stage, func(t *testing.T) {
			cfg := newTestDisaggDeploymentConfig(4, 2, 2)
			cfg.PDTransferTime = func(int64, InstanceID, InstanceID) int64 { return 5_000_000 } // a long transfer
			reqs := newTestRequests(4)
			if tc.stage == "prefill" {
				editEngine(&cfg, func(c *simConfig) { c.BatchConfig = cengine.NewBatchConfig(1, 2048, 0) })
				for _, r := range reqs {
					r.ArrivalTime = 0
				}
			}
			for _, r := range reqs {
				r.OutputTokens = sim.GenerateRandomTokenIDs(sim.NewPartitionedRNG(sim.NewSimulationKey(1)).ForSubsystem(sim.SubsystemWorkload), 20000)
				r.MaxOutputLen = len(r.OutputTokens)
			}
			gw := &fakeGateway{disaggregate: true, evictAfterUs: tc.after}
			cs := newTestCluster(cfg, NewSliceRequestSource(reqs), nil)
			cs.SetGateway(gw)
			mustRun(t, cs)
			for _, r := range reqs {
				if !gw.evictOK[r.ID] || gw.done[r.ID] != DoneEvicted {
					t.Fatalf("%s: eviction took %v, outcome %v; want evicted", r.ID, gw.evictOK[r.ID], gw.done[r.ID])
				}
			}
			inPrefill := 0
			for _, p := range cs.ParentRequests() {
				tc.check(t, p)
				if p.PrefillCompleteTime == 0 {
					inPrefill++
				}
			}
			if tc.stage == "prefill" && inPrefill != 2 {
				t.Errorf("%d requests evicted in their prefill, want the 2 queued", inPrefill)
			}
			if cs.GatewayEvicted() != len(reqs) {
				t.Errorf("gateway evicted %d, want %d", cs.GatewayEvicted(), len(reqs))
			}
			for _, inst := range cs.Instances() {
				if u := inst.UsedKVBlocks(); u != 0 {
					t.Errorf("%s: %d KV blocks still allocated", inst.ID(), u)
				}
				// The engine published its state after the evictions (an empty step), as vLLM.
				if m := inst.EngineMetricsAt(math.MaxInt64); m.Running != 0 || m.Waiting != 0 || m.KVUsage != 0 {
					t.Errorf("%s: last published running %d, waiting %d, KV usage %g after every request was evicted",
						inst.ID(), m.Running, m.Waiting, m.KVUsage)
				}
				// A transfer evicted before it completed leaves no cached prefix: its KV never arrived.
				if tc.stage == "transfer" && cs.PoolMembership()[string(inst.ID())] == PoolRoleDecode {
					if h := inst.CachedBlockHashes(); len(h) != 0 {
						t.Errorf("%s: %d cached block hashes for KV that never arrived", inst.ID(), len(h))
					}
				}
			}
			if cs.transfersInitiated != cs.transfersCompleted {
				t.Errorf("INV-PD-3: %d transfers initiated, %d completed", cs.transfersInitiated, cs.transfersCompleted)
			}
			if m := cs.AggregatedMetrics(); m.CompletedRequests != 0 || m.StillRunning != 0 {
				t.Errorf("completed %d, still running %d; want every request evicted", m.CompletedRequests, m.StillRunning)
			}
			assertClusterINV1Conservation(t, cs, len(reqs), noRejections, tc.stage)
		})
	}
}

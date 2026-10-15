package cluster

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// DPStatsIntervalUs is how often vLLM's DP coordinator publishes the engines' request
// counts to the front-end's load balancer: min_stats_update_interval_ms = 100
// (vllm/v1/engine/coordinator.py:154-162, :267-269 at v0.31.0).
const DPStatsIntervalUs int64 = 100_000

// dpReplica is one data-parallel MoE replica (#1909): dp engine ranks that step in
// lockstep and sit behind one API endpoint, the pod the router sees.
//
// vLLM v0.31.0 runs an MoE model at dp > 1 as DPEngineCoreProc ranks
// (vllm/v1/engine/core.py:1386-1395), whatever the expert-parallel setting: the MoE layer
// dispatches across the data-parallel ranks either way (use_all2all is true at dp > 1,
// vllm/config/parallel.py:738-743). A dense model runs independent engines instead, which
// BLIS refuses at dp > 1.
type dpReplica struct {
	// id is the endpoint the router sees: rank 0's instance ID.
	id    string
	ranks []*InstanceSimulator
	group *sim.LockstepGroup

	// The front-end's internal DP load balancer: the vllm-dp scorer over the coordinator's
	// last published per-rank counts (see pickRank).
	balancer sim.RoutingPolicy
	// counts is the coordinator's latest [waiting, running] per rank, which the front-end
	// bumps locally on each assignment until the next publish overwrites it
	// (vllm/v1/engine/core_client.py:1626-1628).
	counts []dpRankCounts
	// epoch is the DPStatsIntervalUs period counts was published in; -1 before the first.
	epoch int64
	// start is the balancer's rotating scan start (eng_start_index, :1629-1634).
	start int
}

type dpRankCounts struct {
	waiting, running int
}

// buildDPReplicas groups each run of config.DPLockstepRanks consecutive instances into a
// lockstep replica. A no-op when DPLockstepRanks <= 1.
func (cs *ClusterSimulator) buildDPReplicas(config DeploymentConfig, instanceMap map[InstanceID]*InstanceSimulator) {
	dp := config.DPLockstepRanks
	if dp <= 1 {
		return
	}
	if config.NumInstances%dp != 0 {
		panic(fmt.Sprintf("ClusterSimulator: NumInstances %d is not a multiple of the data-parallel width %d",
			config.NumInstances, dp))
	}
	for _, n := range []int{config.PrefillInstances, config.DecodeInstances, config.SharedInstances, config.EncodeInstances} {
		if n%dp != 0 {
			panic(fmt.Sprintf("ClusterSimulator: pool instance count %d is not a multiple of the data-parallel width %d", n, dp))
		}
	}
	if config.ModelAutoscalerIntervalUs > 0 {
		panic("ClusterSimulator: data-parallel lockstep replicas cannot be autoscaled")
	}
	cs.replicaOf = make(map[string]*dpReplica, config.NumInstances)
	for r := 0; r < config.NumInstances/dp; r++ {
		rep := &dpReplica{epoch: -1, counts: make([]dpRankCounts, dp),
			balancer: sim.NewRoutingPolicy("weighted", []sim.ScorerConfig{{Name: "vllm-dp", Weight: 1}},
				config.BlockSizeTokens, nil)}
		sims := make([]*sim.Simulator, 0, dp)
		for k := 0; k < dp; k++ {
			id := InstanceID(fmt.Sprintf("instance_%d", r*dp+k))
			inst, ok := instanceMap[id]
			if !ok {
				panic(fmt.Sprintf("ClusterSimulator: data-parallel rank %s of replica %d was not placed at startup; "+
					"a lockstep replica needs every rank", id, r))
			}
			rep.ranks = append(rep.ranks, inst)
			sims = append(sims, inst.sim)
		}
		rep.id = string(rep.ranks[0].ID())
		g, err := sim.NewLockstepGroup(sims)
		if err != nil {
			panic(fmt.Sprintf("ClusterSimulator: replica %s: %v", rep.id, err))
		}
		rep.group = g
		cs.dpReplicas = append(cs.dpReplicas, rep)
		for _, inst := range rep.ranks {
			cs.replicaOf[string(inst.ID())] = rep
		}
	}
}

// eventMembers is the instances an event on inst can advance: the whole replica for a
// lockstep rank, else inst alone.
func (cs *ClusterSimulator) eventMembers(inst *InstanceSimulator) []*InstanceSimulator {
	if rep := cs.replicaOf[string(inst.ID())]; rep != nil {
		return rep.ranks
	}
	cs.memberScratch[0] = inst
	return cs.memberScratch[:]
}

// endpointReplica is the replica whose endpoint stands for inst in routing, or nil when inst
// is its own endpoint. Replicas are endpoints only on the colocated routing path: a P/D
// pool's ranks stay individually routable (a documented limitation).
func (cs *ClusterSimulator) endpointReplica(inst *InstanceSimulator) *dpReplica {
	if cs.replicaOf == nil || cs.poolsConfigured() {
		return nil
	}
	return cs.replicaOf[string(inst.ID())]
}

// routable is whether the replica takes traffic: every rank must.
func (rep *dpReplica) routable() bool {
	for _, inst := range rep.ranks {
		if !inst.IsRoutable() {
			return false
		}
	}
	return true
}

// endpointSnapshot is the router's view of a replica: one pod, its signals aggregated over
// the ranks as llm-d's endpoint picker aggregates the pod's per-engine series.
//
// vLLM v0.31.0 labels each gauge with the engine index, one series per DP rank
// (vllm/v1/metrics/loggers.py:480-486, :509-584). llm-d-router reads a pod's waiting and
// running requests as the SUM over those series and its KV cache usage as the MAX
// (pkg/epp/framework/plugins/datalayer/extractor/metrics/extractor.go:146-164, spec.go:147,
// llm-d-router#3085; v0.10.0 read the first series alone). In-flight requests and tokens
// are tracked per endpoint (inflightload/producer.go), so they sum over the ranks too, as
// do the capacity and occupancy counts. Latency averages are the ranks' mean; resident
// adapters their union.
func (cs *ClusterSimulator) endpointSnapshot(rep *dpReplica) sim.RoutingSnapshot {
	agg := sim.NewRoutingSnapshot(rep.id)
	var ttft, itl, avgIn, avgOut, hit float64
	n := float64(len(rep.ranks))
	for _, inst := range rep.ranks {
		s := cs.snapshotProvider.Snapshot(inst.ID(), cs.clock)
		agg.QueueDepth += s.QueueDepth
		agg.BatchSize += s.BatchSize
		agg.FreeKVBlocks += s.FreeKVBlocks
		agg.PreemptionCount += s.PreemptionCount
		agg.TotalKvCapacityTokens += s.TotalKvCapacityTokens
		agg.KvTokensInUse += s.KvTokensInUse
		agg.DispatchRate += s.DispatchRate
		agg.InFlightRequests += cs.inFlightRequests[string(inst.ID())]
		agg.MaxBatchSize += float64(inst.MaxBatchSize())
		agg.KVUtilization = max(agg.KVUtilization, s.KVUtilization)
		ttft += s.TTFT
		itl += s.ITL
		avgIn += s.AvgInTokens
		avgOut += s.AvgOutTokens
		hit += s.CacheHitRate
		for a := range s.ResidentAdapters {
			if agg.ResidentAdapters == nil {
				agg.ResidentAdapters = make(map[string]bool)
			}
			agg.ResidentAdapters[a] = true
		}
	}
	agg.TTFT, agg.ITL = ttft/n, itl/n
	agg.AvgInTokens, agg.AvgOutTokens = avgIn/n, avgOut/n
	agg.CacheHitRate = hit / n
	lead := rep.ranks[0]
	agg.Model = lead.Model
	agg.GPUType = lead.GPU()
	agg.TPDegree = lead.TPDegree
	agg.CostPerHour = 0
	for _, inst := range rep.ranks {
		agg.CostPerHour += inst.CostPerHour
	}
	return agg
}

// A replica endpoint's prefix-cache view is rank 0's cache alone, under rank 0's
// instance ID, which is the endpoint's: the cacheQueryFn entry needs no aggregation.
//
// vLLM publishes each rank's KV events on its own port, base + rank
// (vllm/distributed/kv_events.py:355-356, :530-563 at v0.31.0). llm-d-router subscribes an
// endpoint at the pool's port + its rank index (pkg/kvevents/pool.go:134;
// preciseprefixcache/extractor.go:74-80), and a pod behind vLLM's internal balancer is one
// endpoint of rank index 0, so only rank 0's events reach the index. Ranks 1..dp-1 cache
// prefixes the router never sees.

// pickRank assigns a request that reached the replica's endpoint to a rank: vLLM's
// internal DP load balancer (DPLBAsyncMPClient.get_core_engine_for_request,
// vllm/v1/engine/core_client.py:1589-1640 at v0.31.0), priced by BLIS's vllm-dp scorer.
//
// The balancer reads the coordinator's last published per-rank counts, which the
// coordinator sends at most every DPStatsIntervalUs; BLIS samples the ranks at the first
// assignment of each period. Between publishes the front-end bumps the chosen rank's
// waiting count, and it rotates where its scan starts so ties do not always favour one
// rank. The scorer keeps the first maximum, so presenting the ranks from the rotating start
// is that scan.
//
// The vllm-dp scorer prices waiting x 4 + running. v0.31.0's balancer instead scores
// max(client_count x in_flight, waiting + running), plus waiting x 6 x max(0, kv - 0.5)
// when anything waits; the scorer, not this assignment path, owns that formula.
func (cs *ClusterSimulator) pickRank(rep *dpReplica, req *sim.Request, now int64) *InstanceSimulator {
	if epoch := now / DPStatsIntervalUs; epoch != rep.epoch {
		rep.epoch = epoch
		for k, inst := range rep.ranks {
			rep.counts[k] = dpRankCounts{waiting: inst.QueueDepth(), running: inst.BatchSize()}
		}
	}
	n := len(rep.ranks)
	snaps := make([]sim.RoutingSnapshot, 0, n)
	for i := 0; i < n; i++ {
		k := (rep.start + i) % n
		snap := sim.NewRoutingSnapshot(string(rep.ranks[k].ID()))
		snap.QueueDepth, snap.BatchSize = rep.counts[k].waiting, rep.counts[k].running
		snap.InFlightRequests = cs.inFlightRequests[snap.ID]
		snaps = append(snaps, snap)
	}
	decision := rep.balancer.Route(req, &sim.RouterState{Snapshots: snaps, Clock: now})
	rep.start = (rep.start + 1) % n
	for k, inst := range rep.ranks {
		if string(inst.ID()) == decision.TargetInstance {
			rep.counts[k].waiting++
			return inst
		}
	}
	panic(fmt.Sprintf("pickRank: balancer chose %q, not a rank of replica %s", decision.TargetInstance, rep.id))
}

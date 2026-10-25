package cluster

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// DPStatsIntervalUs is how often a replica's load balancer sees the ranks' published counts
// (sim.DPStatsIntervalUs, the engine's).
const DPStatsIntervalUs = sim.DPStatsIntervalUs

// The DP load-balancing modes (DeploymentConfig.DPLoadBalancing) and their rules are the
// engine's (sim.DPLoadBalancing*, sim.ValidDPLoadBalancing, sim.DPEachRankIsAnEndpoint, from
// blis-vllm compat/dplb): they are vLLM's serve flags. The cluster builds the endpoints and routes.
const (
	DPLoadBalancingInternal = sim.DPLoadBalancingInternal
	DPLoadBalancingHybrid   = sim.DPLoadBalancingHybrid
	DPLoadBalancingExternal = sim.DPLoadBalancingExternal
)

// IsValidDPLoadBalancing reports whether mode names a DP load-balancing mode ("" is internal):
// the engine's rule (sim.ValidDPLoadBalancing).
func IsValidDPLoadBalancing(mode string) bool { return sim.ValidDPLoadBalancing(mode) }

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

	// lb is the engine's load balancer that picks the rank a request reaching the replica's
	// endpoint runs on, and lbRanks the ranks as it reads them.
	lb      *sim.DPBalancer
	lbRanks []sim.DPRank
}

// dpRank is one rank as the replica's load balancer reads it: its live queue, batch and KV
// usage when the counts are published, and the front-end's exact in-flight count on it (the
// cluster's per-rank counter, raised at dispatch and lowered at completion, drop or timeout).
type dpRank struct {
	cs   *ClusterSimulator
	inst *InstanceSimulator
}

func (r dpRank) Counts() sim.DPRankCounts {
	return sim.DPRankCounts{Waiting: r.inst.QueueDepth(), Running: r.inst.BatchSize(), KVUsage: r.inst.KVUtilization()}
}

func (r dpRank) InFlight() int { return r.cs.inFlightRequests[string(r.inst.ID())] }

// buildDPReplicas groups each run of config.DPLockstepRanks consecutive instances into a
// lockstep replica. A no-op when DPLockstepRanks <= 1. A deployment it cannot group -- an
// instance count that is not a whole number of replicas, the autoscaler, or a rank startup
// placement deferred -- is reported as an error, which Run returns before simulating
// anything, so the CLI refuses it naming the cause.
func (cs *ClusterSimulator) buildDPReplicas(config DeploymentConfig, instanceMap map[InstanceID]*InstanceSimulator) error {
	if !IsValidDPLoadBalancing(config.DPLoadBalancing) {
		return fmt.Errorf("unknown DP load-balancing mode %q; want %s, %s or %s", config.DPLoadBalancing,
			DPLoadBalancingInternal, DPLoadBalancingHybrid, DPLoadBalancingExternal)
	}
	dp := config.DPLockstepRanks
	if dp <= 1 {
		return nil
	}
	for _, c := range []struct {
		what string
		n    int
	}{{"instance", config.NumInstances}, {"prefill", config.PrefillInstances},
		{"decode", config.DecodeInstances}, {"shared", config.SharedInstances}, {"encode", config.EncodeInstances}} {
		if c.n%dp != 0 {
			return fmt.Errorf("%d %s instance(s) is not a whole number of data-parallel replicas of %d ranks", c.n, c.what, dp)
		}
	}
	if config.ModelAutoscalerIntervalUs > 0 {
		return fmt.Errorf("data-parallel replicas of %d ranks cannot be autoscaled", dp)
	}
	replicaOf := make(map[string]*dpReplica, config.NumInstances)
	var replicas []*dpReplica
	for r := 0; r < config.NumInstances/dp; r++ {
		rep := &dpReplica{lb: sim.NewDPBalancer(dp)}
		sims := make([]*sim.Simulator, 0, dp)
		for k := 0; k < dp; k++ {
			id := InstanceID(fmt.Sprintf("instance_%d", r*dp+k))
			inst, ok := instanceMap[id]
			if !ok {
				return fmt.Errorf("data-parallel rank %s of replica %d could not be placed at startup, and a "+
					"replica steps only with every rank; give the node pools room for %d instance(s) of TP %d",
					id, r, config.NumInstances, max(config.TP, 1))
			}
			rep.ranks = append(rep.ranks, inst)
			rep.lbRanks = append(rep.lbRanks, dpRank{cs: cs, inst: inst})
			sims = append(sims, inst.engineCore())
		}
		rep.id = string(rep.ranks[0].ID())
		g, err := sim.NewLockstepGroup(sims)
		if err != nil {
			return fmt.Errorf("replica %s: %w", rep.id, err)
		}
		rep.group = g
		replicas = append(replicas, rep)
		for _, inst := range rep.ranks {
			replicaOf[string(inst.ID())] = rep
		}
	}
	cs.dpReplicas, cs.replicaOf = replicas, replicaOf
	// A replica endpoint renders on its own API servers before the balancer picks a rank
	// (INV-R1): how many is the engine's rule (sim.ReplicaAPIServers, vLLM's default when
	// unstated). An external-balancer or P/D-pool rank keeps the render stage it was built
	// with: its own endpoint's servers.
	for _, rep := range replicas {
		if cs.endpointReplica(rep.ranks[0]) == nil {
			continue
		}
		apiServers := sim.ReplicaAPIServers(config.Frontend.APIServerCount, dp)
		sims := make([]*sim.Simulator, len(rep.ranks))
		for k, inst := range rep.ranks {
			sims[k] = inst.engineCore()
		}
		if err := sim.ShareRendererOf(sims, apiServers); err != nil {
			return fmt.Errorf("replica %s: %w", rep.id, err)
		}
	}
	return nil
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
// is its own endpoint: under external DP load balancing every rank is, and a P/D pool's
// ranks stay individually routable (a documented limitation).
func (cs *ClusterSimulator) endpointReplica(inst *InstanceSimulator) *dpReplica {
	if cs.replicaOf == nil || cs.poolsConfigured() || sim.DPEachRankIsAnEndpoint(cs.config.DPLoadBalancing) {
		return nil
	}
	return cs.replicaOf[string(inst.ID())]
}

// EndpointRanks implements GatewayHost: the engines behind the router endpoint id. An instance
// that is its own endpoint is one engine. A data-parallel replica behind vLLM's internal (or
// hybrid) balancer is one endpoint, rank 0's pod, serving every rank's engine: vLLM labels each
// engine's gauges with its index (vllm/v1/metrics/loggers.py:480-486, :509-584 at v0.31.0),
// and llm-d-router reads a pod's waiting and running requests as the sum over those series and
// its KV cache usage as the max (pkg/epp/framework/plugins/datalayer/extractor/metrics/
// extractor.go:146-164, spec.go:147). Ranks 1..dp-1 of such a replica are no endpoint: nil.
func (cs *ClusterSimulator) EndpointRanks(id InstanceID) []*InstanceSimulator {
	inst := cs.instanceByID(id)
	if inst == nil {
		return nil
	}
	rep := cs.endpointReplica(inst)
	if rep == nil {
		return []*InstanceSimulator{inst}
	}
	if inst != rep.ranks[0] {
		return nil
	}
	return rep.ranks
}

// rankFor is the engine that serves a request the router sent to endpoint inst: inst itself, or,
// for a replica endpoint, the rank vLLM's internal DP balancer picks (pickRank).
func (cs *ClusterSimulator) rankFor(inst *InstanceSimulator) *InstanceSimulator {
	if rep := cs.endpointReplica(inst); rep != nil && inst == rep.ranks[0] {
		return cs.pickRank(rep, cs.clock)
	}
	return inst
}

// A replica endpoint's prefix-cache view is rank 0's cache alone, under rank 0's
// instance ID, which is the endpoint's: only rank 0 publishes KV events to the router.
//
// vLLM publishes each rank's KV events on its own port, base + rank
// (vllm/distributed/kv_events.py:355-356, :530-563 at v0.31.0). llm-d-router subscribes an
// endpoint at the pool's port + its rank index (pkg/kvevents/pool.go:134;
// preciseprefixcache/extractor.go:74-80), and a pod behind vLLM's internal balancer is one
// endpoint of rank index 0, so only rank 0's events reach the index. Ranks 1..dp-1 cache
// prefixes the router never sees.

// pickRank assigns a request that reached the replica's endpoint at now to a rank: the engine's
// load balancer (sim.DPBalancer) decides.
func (cs *ClusterSimulator) pickRank(rep *dpReplica, now int64) *InstanceSimulator {
	return rep.ranks[rep.lb.Pick(rep.lbRanks, now)]
}

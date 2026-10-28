package cluster

import (
	"github.com/inference-sim/inference-sim/sim"
)

// PoolRoleConfigs are a disaggregated deployment's per-role instance configurations: the
// engine configuration each P/D pool's instances are built with (its own parallelism, KV pool,
// token budget, scheduling policy and latency model) beside the host settings the cluster reads
// (GPU, TP, MaxNumSeqs). The build site resolves them from each pool's own engine; nil uses the
// deployment's global configuration for that role (BC-P2-1).
type PoolRoleConfigs struct {
	Prefill *sim.InstanceConfig
	Decode  *sim.InstanceConfig
}

// IsEmpty reports whether no role has a configuration of its own.
func (p PoolRoleConfigs) IsEmpty() bool { return p.Prefill == nil && p.Decode == nil }

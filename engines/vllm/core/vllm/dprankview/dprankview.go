// SPDX-License-Identifier: Apache-2.0
// Original to blis-vllm.
// H1 of the DP external load-balancing RFC (inference-sim#2699).

// Package dprankview exposes one vLLM data-parallel group as N router-visible endpoints, one per
// rank, for external data-parallel load balancing. Each view is an engine.Engine the host can
// register as its own InstanceSimulator: Add routes to the view's endpoint, the per-rank reads
// (Metrics, Snapshot, KV cache capacity) report only that rank, and the rank-0 view carries the
// group-step surface (StepDue/LockstepSteps/LastActive) the sim leader drives. The ranks are one
// synchronised group with one lockstep step per wave, so exactly one view — the leader, rank 0 —
// steps the whole group; the others return an empty batch and never complete one. The sim leader/
// member driver (JoinGroup) splits the leader's step across the per-view drivers so each keeps its
// own report.
//
// Keeping the view behind the engine contract in its own package lets the host build it at the one
// permitted site (cmd/engine_build.go) without sim/cluster importing the vLLM engine (CLAUDE.md's
// import rule).
package dprankview

import (
	"github.com/inference-sim/inference-sim/engines/vllm/core/kv"
	vllm "github.com/inference-sim/inference-sim/engines/vllm/core/vllm"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

// view is one router endpoint of a shared DPGroup: the rank it serves, and the group it steps
// (rank 0 only) and reads (its own rank).
type view struct {
	g        *vllm.DPGroup
	endpoint int // the router endpoint this view answers for
	rank     int // the single rank it serves (under external balancing, endpoint == rank)
}

// NewEndpointViews exposes g as one engine.Engine per router endpoint, in endpoint order. Under
// external balancing there is one endpoint per rank, so this returns one view per rank; the rank-0
// view leads the group's lockstep step. The views share g: stepping the leader advances every
// rank, and each view reads only its own rank.
func NewEndpointViews(g *vllm.DPGroup) []engine.Engine {
	n := g.Endpoints()
	out := make([]engine.Engine, n)
	for e := 0; e < n; e++ {
		first, _ := g.EndpointRanks(e)
		out[e] = &view{g: g, endpoint: e, rank: first}
	}
	return out
}

// leads reports whether this view steps the group: rank 0's view is the leader.
func (v *view) leads() bool { return v.rank == 0 }

// Add routes the request to this view's endpoint; the group places it on the endpoint's rank.
func (v *view) Add(r engine.Request, now engine.Time) error {
	return v.g.AddToEndpoint(v.endpoint, r, now)
}

// Schedule steps the whole group on the leader's view; a member view schedules nothing of its own
// (an empty batch, never in flight), since the leader's step already runs every rank.
func (v *view) Schedule(now engine.Time) (engine.Batch, error) {
	if v.leads() {
		return v.g.Schedule(now)
	}
	return engine.Batch{}, nil
}

// StepTime prices the group's lockstep step (the slowest rank's forward pass).
func (v *view) StepTime(b engine.Batch) engine.Duration { return v.g.StepTime(b) }

// Complete applies the group's in-flight step on the leader's view. A member view schedules no
// batch, so it has none to complete: any id is unknown to it.
func (v *view) Complete(id engine.BatchID, o engine.Outcome, end engine.Time) (engine.Result, error) {
	if v.leads() {
		return v.g.Complete(id, o, end)
	}
	return engine.Result{}, engine.ErrUnknownBatch
}

// TransferDone and Finish act by id, so they reach the right rank whichever view the host calls.
func (v *view) TransferDone(id engine.TransferID, ok bool, now engine.Time) error {
	return v.g.TransferDone(id, ok, now)
}
func (v *view) Finish(id string, why engine.FinishReason, now engine.Time) error {
	return v.g.Finish(id, why, now)
}

// NextWake, InFlightLimit, Fields and Plan are the group's: the views share one group's timing,
// batch-slot budget, field list and pool plan.
func (v *view) NextWake(now engine.Time) (engine.Time, bool) { return v.g.NextWake(now) }
func (v *view) InFlightLimit() int                           { return v.g.InFlightLimit() }
func (v *view) Fields() fields.FieldList                     { return v.g.Fields() }
func (v *view) Plan() plan.PoolPlan                          { return v.g.Plan() }

// Metrics, Snapshot and KVCacheCapacity are this view's rank alone — what the router scrapes per
// endpoint.
func (v *view) Metrics(now engine.Time) engine.Metrics { return v.g.RankMetrics(v.rank, now) }
func (v *view) Snapshot() engine.State                 { return v.g.RankSnapshot(v.rank) }
func (v *view) KVCacheCapacity() kv.Capacity           { return v.g.RankKVCacheCapacity(v.rank) }

// Assumptions, EnableKVEvents, KVEventsRefusal and OffloadCachedTokens forward to the group. Each
// view enabling events enables every rank's (idempotent); the leader then routes each rank's events
// to its endpoint (LastKVEventsByRank). OffloadCachedTokens acts by request id, so it reaches the
// request's rank whichever view the host asks.
func (v *view) Assumptions() []string             { return v.g.Assumptions() }
func (v *view) EnableKVEvents()                   { v.g.EnableKVEvents() }
func (v *view) KVEventsRefusal() error            { return v.g.KVEventsRefusal() }
func (v *view) OffloadCachedTokens(id string) int { return v.g.OffloadCachedTokens(id) }

// StepDue, LockstepSteps, LastActive and LastKVEventsByRank are the group-step surface the sim
// leader drives (the groupEngine it steps). They are the group's; the leader view is the one the
// leader driver holds. LastKVEventsByRank lets the leader deliver each rank's KV events to that
// rank's endpoint under external balancing (a Removed or Cleared event names no request, so it
// cannot be placed from the batch split and must carry its producing rank from here).
func (v *view) StepDue(now engine.Time) bool           { return v.g.StepDue(now) }
func (v *view) LockstepSteps() (steps, dummy int)      { return v.g.LockstepSteps() }
func (v *view) LastActive() []bool                     { return v.g.LastActive() }
func (v *view) LastKVEventsByRank() [][]engine.KVEvent { return v.g.LastKVEventsByRank() }

var (
	_ engine.Engine  = (*view)(nil)
	_ engine.Planned = (*view)(nil)
)

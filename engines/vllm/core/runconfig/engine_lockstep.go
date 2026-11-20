// Moved from inference-sim/sim/engine/engine_lockstep.go at ca532abca209af41425c007974a5f3ef336f5981.

// sim/engine_lockstep.go: the engine's part of a data-parallel lockstep step (#1909): a rank's
// schedule and finish, and how a group's step is priced. The group itself, which arms the
// step event and keeps the ranks' clocks, is the host's (dp_lockstep.go).
package runconfig

// DPSyncInterval is how many steps a data-parallel wave runs between the ranks'
// unfinished-request all-reduces: vLLM v0.31.0's dp_sync_interval default
// (vllm/config/parallel.py:150), which blis-schemas has no field to state.
const DPSyncInterval = 16

// DPStepPricer prices one lockstep step of a data-parallel group (#1909).
//
// batches[i] is rank i's scheduled requests; an empty one is a rank with nothing ready,
// which runs vLLM's dummy batch. The result is each rank's own step duration in ticks,
// each >= 1. A pricer applies whatever padding the engine applies across the ranks.
type DPStepPricer interface {
	StepTimesAcrossDP(batches [][]*Request) []int64
}

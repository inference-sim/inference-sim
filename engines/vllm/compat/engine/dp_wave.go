// Moved from inference-sim/sim/dp_lockstep.go at 63b211633e587a48a650922a2a2c2c2927199533.

// A data-parallel wave's continuation rule: whether a lockstep group runs another step. It
// moved here from inference-sim's host (LockstepGroup.step, #2021 B2 verification gap 6),
// which keeps the clocks, the arming of the step event and the event order, and asks this
// type whether the wave goes on.
package engine

// DPWave is vLLM's DPEngineCoreProc wave state for one data-parallel group: the step counter
// its unfinished-request all-reduce runs on (_has_global_unfinished_reqs,
// vllm/v1/engine/core.py:2410-2414@5df63a72). The zero value is a wave that has not started.
type DPWave struct {
	steps int
}

// Step records one lockstep step and reports whether the wave runs another. The ranks
// all-reduce whether any has unfinished requests on the wave's first step and on every
// DPSyncInterval-th step after; between syncs the wave runs on, dummy batches included, so a
// wave runs up to DPSyncInterval-1 dummy steps past its last request (core.py:2370-2394,
// 2410-2414). anyRankHasWork is the all-reduce's input: whether any rank still has a request
// running or waiting. A wave that ends resets the counter for the next (core.py:2393).
func (w *DPWave) Step(anyRankHasWork bool) (runsOn bool) {
	w.steps++
	runsOn = (w.steps != 1 && w.steps%DPSyncInterval != 0) || anyRankHasWork
	if !runsOn {
		w.steps = 0
	}
	return runsOn
}

// Steps is the number of steps the current wave has run.
func (w *DPWave) Steps() int { return w.steps }

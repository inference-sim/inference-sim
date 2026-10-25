// Original to blis-vllm.

package engine

import "testing"

// A wave with no work left ends only on a sync step: the first, then every DPSyncInterval-th,
// as vLLM's _has_global_unfinished_reqs all-reduces (core.py:2410-2414@5df63a72). With work it
// runs on, and an ended wave counts from one again.
func TestDPWaveEndsOnlyOnASyncStep(t *testing.T) {
	var w DPWave
	if w.Step(false) {
		t.Fatal("a wave whose first step finds no work must end at once (step 1 syncs)")
	}
	if w.Steps() != 0 {
		t.Errorf("an ended wave restarts its count, got %d", w.Steps())
	}
	// Work on the first step, then none: the wave runs dummy steps until the next sync.
	if !w.Step(true) {
		t.Fatal("a step with work must run on")
	}
	for step := 2; step < DPSyncInterval; step++ {
		if !w.Step(false) {
			t.Fatalf("step %d is not a sync step, so the wave must run on", step)
		}
	}
	if w.Step(false) {
		t.Fatalf("step %d syncs with no work left, so the wave must end", DPSyncInterval)
	}
	// Work on a sync step keeps the wave running past it.
	for step := 1; step < DPSyncInterval; step++ {
		w.Step(true)
	}
	if !w.Step(true) || w.Steps() != DPSyncInterval {
		t.Errorf("a sync step with work runs on; steps %d, want %d", w.Steps(), DPSyncInterval)
	}
	for step := DPSyncInterval + 1; step < 2*DPSyncInterval; step++ {
		if !w.Step(false) {
			t.Fatalf("step %d is between syncs, so the wave must run on", step)
		}
	}
	if w.Step(false) {
		t.Fatalf("step %d syncs with no work left, so the wave must end", 2*DPSyncInterval)
	}
}

package workload

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// makeBP builds a minimal 1-round blueprint + its round-0 request for tests.
func makeBP(id string, seed int64) (SessionBlueprint, *sim.Request) {
	bp := SessionBlueprint{
		SessionID:     id,
		MaxRounds:     1,
		Horizon:       1 << 62, // effectively unbounded — models the self-draining default (no --horizon cap)
		InputSampler:  &SequenceSampler{values: []int{}},
		OutputSampler: &SequenceSampler{values: []int{}},
	}
	req := &sim.Request{ID: "r_" + id, SessionID: id, RoundIndex: 0, State: sim.StateQueued, ArrivalTime: 0}
	return bp, req
}

func TestBuildSessionPool_DuplicatesToTarget(t *testing.T) {
	bp0, r0 := makeBP("s0", 1)
	bp1, r1 := makeBP("s1", 2)
	// Corpus of 2, want 5 total, pool of 2.
	d, initial, err := BuildSessionPool([]SessionBlueprint{bp0, bp1}, []*sim.Request{r0, r1}, 2, 5, 99)
	if err != nil {
		t.Fatalf("BuildSessionPool: %v", err)
	}
	if len(initial) != 2 {
		t.Fatalf("initial injected = %d, want 2 (pool size)", len(initial))
	}
	// 5 total sessions must be registered, with unique SessionIDs (clones renamed).
	if got := d.TotalSessions(); got != 5 {
		t.Errorf("total sessions = %d, want 5", got)
	}
	if !d.hasUniqueSessionIDs() {
		t.Errorf("duplicated sessions must have unique IDs")
	}
}

func TestSessionPool_RefillAndConservation(t *testing.T) {
	// 4 single-round sessions, pool of 2. Each completion should admit the next
	// until the corpus is exhausted; active count never exceeds 2; exactly 4 start.
	var bps []SessionBlueprint
	var r0s []*sim.Request
	for i := 0; i < 4; i++ {
		bp, r := makeBP(fmt.Sprintf("s%d", i), int64(i))
		bps = append(bps, bp)
		r0s = append(r0s, r)
	}
	d, initial, err := BuildSessionPool(bps, r0s, 2, 4, 7)
	if err != nil {
		t.Fatalf("BuildSessionPool: %v", err)
	}
	if len(initial) != 2 || d.activeCount != 2 {
		t.Fatalf("initial=%d active=%d, want 2/2", len(initial), d.activeCount)
	}

	admitted := 0
	// Complete the 2 initial sessions; each single-round session terminates on
	// its round-0 completion, so each should admit one queued session.
	for _, r := range initial {
		r.State = sim.StateCompleted
		r.ProgressIndex = int64(r.InputLen())
		next := d.OnComplete(r, 1000)
		admitted += len(next)
		if d.activeCount > 2 {
			t.Fatalf("active count %d exceeded pool size 2", d.activeCount)
		}
		for _, n := range next {
			n.State = sim.StateCompleted
			n.ProgressIndex = int64(n.InputLen())
		}
	}
	if admitted != 2 {
		t.Fatalf("admitted %d sessions on first wave, want 2", admitted)
	}
	// Complete the 2 admitted sessions; corpus now exhausted → no more admissions.
	// (We re-drive by completing whatever was admitted.)
	// Fetch them from the queue tail we just admitted:
	for i := 2; i < 4; i++ {
		r := d.queued[i]
		r.State = sim.StateCompleted
		r.ProgressIndex = int64(r.InputLen())
		next := d.OnComplete(r, 2000)
		if len(next) != 0 {
			t.Fatalf("corpus exhausted but admitted %d more", len(next))
		}
	}
	// Conservation: every session started and terminated exactly once.
	if d.totalStarted != 4 || d.totalTerm != 4 {
		t.Errorf("started=%d terminated=%d, want 4/4", d.totalStarted, d.totalTerm)
	}
}

// TestBuildSessionPool_ClonesHaveIndependentSamplers is a regression test for
// the shared-cursor bug in cloneBlueprintForDup: a shallow `bp := src` struct
// copy leaves InputSampler/OutputSampler/ThinkTimeSampler pointing at the SAME
// underlying sampler object as the source for stateful sampler types like
// *SequenceSampler (which carries a mutable per-call cursor). Without cloning
// the sampler itself, a source session and its duplicate advance one shared
// cursor and corrupt each other's per-round token-length sequence.
//
// BuildSessionPool's SessionManager keeps blueprints in an unexported map, so
// there's no way to read back the clone's sampler through the public driver
// API. cloneBlueprintForDup is unexported but same-package, so this test calls
// it directly — the cleanest way to observe sampler independence: sample the
// source's InputSampler twice, then the clone's once, and assert the clone
// still returns the FIRST recorded value (10), not the third (30), which is
// what a shared cursor would produce.
func TestBuildSessionPool_ClonesHaveIndependentSamplers(t *testing.T) {
	src := SessionBlueprint{
		SessionID:     "s0",
		MaxRounds:     3,
		Horizon:       1 << 62,
		InputSampler:  &SequenceSampler{values: []int{10, 20, 30}},
		OutputSampler: &SequenceSampler{values: []int{1, 2, 3}},
	}
	srcR0 := &sim.Request{ID: "r_s0", SessionID: "s0", RoundIndex: 0, State: sim.StateQueued}

	rng := rand.New(rand.NewSource(1))
	clone, _ := cloneBlueprintForDup(src, srcR0, 1, rng)

	// Pointer identity must differ: the clone must not alias the source's sampler.
	if clone.InputSampler == src.InputSampler {
		t.Fatalf("clone.InputSampler shares the same object as src.InputSampler")
	}

	// Advance the source's cursor two steps: 10, then 20.
	if got := src.InputSampler.Sample(nil); got != 10 {
		t.Fatalf("source InputSampler sample #1 = %d, want 10", got)
	}
	if got := src.InputSampler.Sample(nil); got != 20 {
		t.Fatalf("source InputSampler sample #2 = %d, want 20", got)
	}

	// The clone's cursor must be independent: it should still yield the FIRST
	// value (10). A shared cursor (the bug) would instead yield 30, since the
	// source has already advanced past index 0 and 1.
	if got := clone.InputSampler.Sample(nil); got != 10 {
		t.Fatalf("clone InputSampler sample #1 = %d, want 10 (independent cursor) — got the source's advanced value, indicating a shared cursor", got)
	}
}

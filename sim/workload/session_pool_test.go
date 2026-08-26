package workload

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestShuffleSessions_ReproducibleAndLockstep verifies #1480's shuffle contract:
// the permutation is reproducible from a fixed seed, keeps (blueprint, round-0)
// pairs aligned, drops no session, and generally differs across seeds.
func TestShuffleSessions_ReproducibleAndLockstep(t *testing.T) {
	build := func(n int) ([]SessionBlueprint, []*sim.Request) {
		bps := make([]SessionBlueprint, n)
		r0 := make([]*sim.Request, n)
		for i := 0; i < n; i++ {
			bp, req := makeBP(fmt.Sprintf("s%02d", i), int64(i))
			bps[i], r0[i] = bp, req
		}
		return bps, r0
	}
	order := func(bps []SessionBlueprint) string {
		s := ""
		for _, b := range bps {
			s += b.SessionID + ","
		}
		return s
	}

	// Reproducible: the same seed yields the same permutation.
	bpsA, r0A := build(20)
	ShuffleSessions(bpsA, r0A, rand.New(rand.NewSource(7)))
	bpsA2, r0A2 := build(20)
	ShuffleSessions(bpsA2, r0A2, rand.New(rand.NewSource(7)))
	_ = r0A2
	if order(bpsA) != order(bpsA2) {
		t.Errorf("same seed → different order:\n %q\n %q", order(bpsA), order(bpsA2))
	}
	// Lockstep: after the shuffle, blueprint[i] and r0[i] still name the same session.
	for i := range bpsA {
		if bpsA[i].SessionID != r0A[i].SessionID {
			t.Errorf("lockstep broken at %d: bp=%s r0=%s", i, bpsA[i].SessionID, r0A[i].SessionID)
		}
	}
	// Set preserved: every session still present (nothing dropped).
	seen := map[string]bool{}
	for _, b := range bpsA {
		seen[b.SessionID] = true
	}
	if len(seen) != 20 {
		t.Errorf("shuffle changed the session set: %d unique, want 20", len(seen))
	}
	// A different seed generally yields a different order (collision negligible at n=20).
	bpsB, r0B := build(20)
	ShuffleSessions(bpsB, r0B, rand.New(rand.NewSource(999)))
	if order(bpsA) == order(bpsB) {
		t.Errorf("different seeds → identical order (astronomically unlikely; likely a bug)")
	}
}

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

func TestSessionPool_RefillRebasesDeadline(t *testing.T) {
	// Two single-round sessions, pool of 1, each with an ABSOLUTE deadline 5000µs
	// after its original arrival (t=0). Terminating session 0 far in the future must
	// refill session 1 with its deadline REBASED ahead of the admission tick —
	// otherwise the stale past deadline drops it instantly on enqueue, mass-cancelling
	// every wave-2+ session for any deadline-bearing (run/observe) corpus.
	bp0, r0 := makeBP("s0", 1)
	bp1, r1 := makeBP("s1", 2)
	r0.Deadline = 5000
	r1.Deadline = 5000
	d, initial, err := BuildSessionPool([]SessionBlueprint{bp0, bp1}, []*sim.Request{r0, r1}, 1, 2, 7)
	if err != nil {
		t.Fatalf("BuildSessionPool: %v", err)
	}
	if len(initial) != 1 || initial[0].SessionID != "s0" {
		t.Fatalf("initial = %v, want [s0]", initial)
	}
	const tick = 100000
	initial[0].State = sim.StateCompleted
	initial[0].ProgressIndex = int64(initial[0].InputLen())
	next := d.OnComplete(initial[0], tick)
	if len(next) != 1 || next[0].SessionID != "s1" {
		t.Fatalf("expected refill of s1, got %v", next)
	}
	refill := next[0]
	if refill.ArrivalTime != tick {
		t.Errorf("refill arrival = %d, want %d", refill.ArrivalTime, tick)
	}
	if refill.Deadline != tick+5000 {
		t.Errorf("refill deadline = %d, want %d (rebased by the arrival offset); a stale past deadline would mass-cancel wave 2+", refill.Deadline, tick+5000)
	}
}

func TestSessionPool_RefillAndConservation(t *testing.T) {
	// 4 single-round sessions, pool of 2. Each completion admits the next until the
	// corpus is exhausted; the pool never admits more than one replacement per
	// termination (so the concurrently-active count never exceeds the pool size);
	// exactly 4 sessions start and terminate. Asserted purely through observable
	// outputs — the injected slice, the per-completion follow-ups admitted, and
	// Unstarted()/TotalSessions() — so the test survives a rewrite of the driver's
	// internal counters (refactor-survival; principles.md BDD/TDD item 5).
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
	if len(initial) != 2 {
		t.Fatalf("initial injected = %d, want 2 (pool size)", len(initial))
	}
	if d.TotalSessions() != 4 {
		t.Fatalf("total sessions = %d, want 4", d.TotalSessions())
	}

	// started/terminated are counted only through observable outputs (returned
	// requests and OnComplete calls), never by reading a driver field.
	started := len(initial)
	terminated := 0

	// Complete the 2 initial sessions; each single-round session terminates on its
	// round-0 completion. Each termination must admit AT MOST ONE replacement —
	// len(next) <= 1 is the observable form of "active never exceeds the pool".
	var refills []*sim.Request
	for _, r := range initial {
		r.State = sim.StateCompleted
		r.ProgressIndex = int64(r.InputLen())
		next := d.OnComplete(r, 1000)
		terminated++
		if len(next) > 1 {
			t.Fatalf("a single termination admitted %d sessions; the pool bound requires <= 1", len(next))
		}
		started += len(next)
		refills = append(refills, next...)
	}
	if len(refills) != 2 {
		t.Fatalf("first wave admitted %d refills, want 2 (corpus not yet exhausted)", len(refills))
	}

	// Complete the 2 refills; the corpus is now exhausted → no further admissions.
	for _, r := range refills {
		r.State = sim.StateCompleted
		r.ProgressIndex = int64(r.InputLen())
		next := d.OnComplete(r, 2000)
		terminated++
		if len(next) != 0 {
			t.Fatalf("corpus exhausted but a termination admitted %d more", len(next))
		}
	}

	// Conservation, all via the public surface: every session started and
	// terminated exactly once, and none was left unstarted.
	if started != 4 || terminated != 4 {
		t.Errorf("started=%d terminated=%d, want 4/4", started, terminated)
	}
	if d.Unstarted() != 0 {
		t.Errorf("Unstarted() = %d, want 0 (every pooled session was admitted)", d.Unstarted())
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

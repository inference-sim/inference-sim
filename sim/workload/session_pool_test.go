package workload

import (
	"fmt"
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

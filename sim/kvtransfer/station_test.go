package kvtransfer

import (
	"math/rand"
	"runtime"
	"testing"
)

// oneTier is a convenience config for a single tier with symmetric, easy-to-read
// service physics: bandwidth of 1 byte/tick so serviceTicks == base + bytes.
func oneTier(nRead, nWrite int, readBase, writeBase int64) Config {
	return Config{Tiers: []TierConfig{{
		NRead:             nRead,
		NWrite:            nWrite,
		ReadBaseTicks:     readBase,
		WriteBaseTicks:    writeBase,
		ReadBytesPerTick:  1,
		WriteBytesPerTick: 1,
	}}}
}

func mustNew(t *testing.T, cfg Config) *TransferStation {
	t.Helper()
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: unexpected error: %v", err)
	}
	return s
}

func TestNew_Validation(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantErr bool
	}{
		{"empty tiers", Config{}, true},
		{"nil tiers", Config{Tiers: nil}, true},
		{"valid single tier", oneTier(16, 16, 10, 20), false},
		{"valid read-only pool", oneTier(4, 0, 10, 20), false},
		{"valid write-only pool", oneTier(0, 4, 10, 20), false},
		{"zero servers", oneTier(0, 0, 10, 20), true},
		{"negative NRead", Config{Tiers: []TierConfig{{NRead: -1, NWrite: 1, ReadBytesPerTick: 1, WriteBytesPerTick: 1}}}, true},
		{"negative NWrite", Config{Tiers: []TierConfig{{NRead: 1, NWrite: -1, ReadBytesPerTick: 1, WriteBytesPerTick: 1}}}, true},
		{"zero read bandwidth", Config{Tiers: []TierConfig{{NRead: 1, NWrite: 1, ReadBytesPerTick: 0, WriteBytesPerTick: 1}}}, true},
		{"negative write bandwidth", Config{Tiers: []TierConfig{{NRead: 1, NWrite: 1, ReadBytesPerTick: 1, WriteBytesPerTick: -5}}}, true},
		{"negative base", Config{Tiers: []TierConfig{{NRead: 1, NWrite: 1, ReadBaseTicks: -1, ReadBytesPerTick: 1, WriteBytesPerTick: 1}}}, true},
		{"negative max queue depth", Config{Tiers: []TierConfig{{NRead: 1, NWrite: 1, ReadBytesPerTick: 1, WriteBytesPerTick: 1, MaxQueueDepth: -1}}}, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New(tc.cfg)
			if (err != nil) != tc.wantErr {
				t.Fatalf("New err=%v, wantErr=%v", err, tc.wantErr)
			}
		})
	}
}

func TestTierCount(t *testing.T) {
	s := mustNew(t, Config{Tiers: []TierConfig{
		{NRead: 1, NWrite: 1, ReadBytesPerTick: 1, WriteBytesPerTick: 1},
		{NRead: 2, NWrite: 2, ReadBytesPerTick: 1, WriteBytesPerTick: 1},
	}})
	if got := s.TierCount(); got != 2 {
		t.Fatalf("TierCount=%d, want 2", got)
	}
}

// BC-S3: service_time = base + floor(bytes/bandwidth), per (tier, direction),
// with read and write independently representable.
func TestServiceTicks_Formula(t *testing.T) {
	cfg := Config{Tiers: []TierConfig{{
		NRead: 1, NWrite: 1,
		ReadBaseTicks: 100, WriteBaseTicks: 300, // read != write base
		ReadBytesPerTick: 10, WriteBytesPerTick: 4, // read != write bandwidth
	}}}
	s := mustNew(t, cfg)

	tests := []struct {
		dir       Direction
		bytes     int64
		wantTicks int64
	}{
		{Read, 0, 100},    // base only
		{Read, 1000, 200}, // 100 + 1000/10
		{Read, 1005, 200}, // floor(1005/10)=100 → 200
		{Write, 0, 300},   // base only
		{Write, 40, 310},  // 300 + 40/4
		{Write, 43, 310},  // floor(43/4)=10 → 310
	}
	for _, tc := range tests {
		if got := s.ServiceTicks(0, tc.dir, tc.bytes); got != tc.wantTicks {
			t.Errorf("ServiceTicks(%s, %d)=%d, want %d", tc.dir, tc.bytes, got, tc.wantTicks)
		}
	}

	// Read and write must differ for the same byte count (asymmetry is representable).
	if s.ServiceTicks(0, Read, 1000) == s.ServiceTicks(0, Write, 1000) {
		t.Errorf("read and write service times must be independently representable")
	}
}

// BC-S3 linearity: holding base fixed, the variable part scales linearly with
// bytes. Doubling the byte count doubles the variable part.
func TestServiceTicks_Linear(t *testing.T) {
	s := mustNew(t, oneTier(1, 1, 50, 50)) // bandwidth 1 byte/tick, base 50
	base := s.ServiceTicks(0, Read, 0)
	for _, b := range []int64{100, 250, 1000, 7777} {
		single := s.ServiceTicks(0, Read, b) - base
		double := s.ServiceTicks(0, Read, 2*b) - base
		if double != 2*single {
			t.Errorf("bytes=%d: variable part not linear: f(2b)-base=%d, 2*(f(b)-base)=%d", b, double, 2*single)
		}
	}
}

// The variable part of the service time is clamped so that an adversarially
// large byte count cannot overflow int64 when added to the start tick. The clamp
// only ever engages far beyond any physical KV transfer.
func TestServiceTicks_OverflowGuard(t *testing.T) {
	s := mustNew(t, Config{Tiers: []TierConfig{{
		NRead: 1, NWrite: 1,
		ReadBaseTicks: 7, ReadBytesPerTick: 0.5, // < 1 amplifies the variable part
		WriteBytesPerTick: 1,
	}}})
	// A huge byte count would overflow if unclamped; the guard caps the variable
	// part at maxServiceTicks, keeping the result positive and bounded.
	got := s.ServiceTicks(0, Read, int64(1)<<62)
	if got != 7+maxServiceTicks {
		t.Fatalf("ServiceTicks with huge bytes = %d, want %d (base + clamp)", got, 7+maxServiceTicks)
	}
	if got <= 0 {
		t.Fatalf("overflow guard failed: got non-positive service time %d", got)
	}
}

// A single job on a single server completes exactly serviceTicks after its
// submit tick.
func TestSubmitPoll_SingleJobCompletionTime(t *testing.T) {
	s := mustNew(t, oneTier(1, 1, 100, 100)) // bandwidth 1 → serviceTicks = 100 + bytes
	id, ok := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 500, SubmitTick: 1000})
	if !ok {
		t.Fatal("Submit rejected an accepted job")
	}
	wantComplete := int64(1000 + 100 + 500) // submit + base + bytes/bw

	// Not complete before its time.
	if got := s.Poll(wantComplete - 1); len(got) != 0 {
		t.Fatalf("Poll before completion returned %v, want none", got)
	}
	// Complete at its time.
	got := s.Poll(wantComplete)
	if len(got) != 1 || got[0] != id {
		t.Fatalf("Poll at completion returned %v, want [%d]", got, id)
	}
	// Already drained.
	if got := s.Poll(wantComplete + 1000); len(got) != 0 {
		t.Fatalf("Poll after drain returned %v, want none", got)
	}
}

// A Poll with a tick earlier than one already observed is a safe no-op: the
// clock never moves backward (INV-3) and no completions are returned. The
// documented precondition is a non-decreasing now; this verifies the station
// degrades gracefully rather than misbehaving if a caller violates it.
func TestPoll_BackwardIsSafeNoOp(t *testing.T) {
	s := mustNew(t, oneTier(1, 0, 10, 10)) // serviceTicks = 10 + bytes
	id, _ := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 0, SubmitTick: 100})
	// Advance well past completion (job completes at tick 110).
	if got := s.Poll(200); len(got) != 1 || got[0] != id {
		t.Fatalf("expected job %d to complete by tick 200, got %v", id, got)
	}
	// A backward Poll returns nothing and does not disturb the clock.
	if got := s.Poll(50); len(got) != 0 {
		t.Fatalf("backward Poll returned %v, want none", got)
	}
	// A subsequent forward Poll still returns nothing (the job already drained),
	// confirming the backward Poll left the clock intact.
	if got := s.Poll(300); len(got) != 0 {
		t.Fatalf("forward Poll after backward no-op returned %v, want none", got)
	}
}

// BC-S1: in-service jobs per tier never exceed NRead+NWrite, under arrival storms.
// Also checks conservation: every accepted job completes exactly once.
func TestBCS1_NeverExceedServers(t *testing.T) {
	const (
		nRead   = 3
		nWrite  = 2
		nServer = nRead + nWrite
	)
	rng := rand.New(rand.NewSource(42))
	s := mustNew(t, Config{Tiers: []TierConfig{{
		NRead: nRead, NWrite: nWrite,
		ReadBaseTicks: 5, WriteBaseTicks: 7,
		ReadBytesPerTick: 2, WriteBytesPerTick: 3,
	}}})

	accepted := map[JobID]bool{}
	completed := map[JobID]int{}
	tick := int64(0)
	for i := 0; i < 5000; i++ {
		tick += int64(rng.Intn(3)) // non-decreasing, sometimes same tick (storm)
		dir := Read
		if rng.Intn(2) == 0 {
			dir = Write
		}
		id, ok := s.Submit(TransferJob{Tier: 0, Direction: dir, Bytes: int64(rng.Intn(50)), SubmitTick: tick})
		if ok {
			accepted[id] = true
		}
		// BC-S1 check after every submit.
		if inService := s.ActiveJobs(0, Read) + s.ActiveJobs(0, Write); inService > nServer {
			t.Fatalf("in-service %d exceeds %d servers at tick %d", inService, nServer, tick)
		}
		for _, cid := range s.Poll(tick) {
			completed[cid]++
		}
		if inService := s.ActiveJobs(0, Read) + s.ActiveJobs(0, Write); inService > nServer {
			t.Fatalf("in-service %d exceeds %d servers after poll at tick %d", inService, nServer, tick)
		}
	}
	// Drain everything.
	for _, cid := range s.Poll(tick + 1_000_000) {
		completed[cid]++
	}
	// Conservation: every accepted job completed exactly once; nothing else did.
	if len(completed) != len(accepted) {
		t.Fatalf("completed %d jobs, accepted %d", len(completed), len(accepted))
	}
	for id := range accepted {
		if completed[id] != 1 {
			t.Fatalf("job %d completed %d times, want 1", id, completed[id])
		}
	}
	// Fully drained: no jobs in service.
	if s.ActiveJobs(0, Read)+s.ActiveJobs(0, Write) != 0 {
		t.Fatalf("jobs still in service after full drain")
	}
}

// BC-S4: the completion sequence is a total deterministic function of the input,
// identical across repeated fresh runs and across GOMAXPROCS settings — never a
// function of wall clock or goroutine scheduling.
func TestBCS4_DeterministicAcrossRunsAndGOMAXPROCS(t *testing.T) {
	run := func() []JobID {
		s := mustNew(t, Config{Tiers: []TierConfig{
			{NRead: 2, NWrite: 2, ReadBaseTicks: 5, WriteBaseTicks: 9, ReadBytesPerTick: 2, WriteBytesPerTick: 1},
			{NRead: 1, NWrite: 3, ReadBaseTicks: 3, WriteBaseTicks: 4, ReadBytesPerTick: 5, WriteBytesPerTick: 2},
		}})
		rng := rand.New(rand.NewSource(7))
		var seq []JobID
		tick := int64(0)
		for i := 0; i < 2000; i++ {
			tick += int64(rng.Intn(4))
			s.Submit(TransferJob{
				Tier:       rng.Intn(2),
				Direction:  Direction(rng.Intn(2)),
				Bytes:      int64(rng.Intn(100)),
				SubmitTick: tick,
			})
			seq = append(seq, s.Poll(tick)...)
		}
		seq = append(seq, s.Poll(tick+1_000_000)...)
		return seq
	}

	baseline := run()
	// Repeated fresh runs are identical.
	for i := 0; i < 3; i++ {
		if got := run(); !equalIDs(got, baseline) {
			t.Fatalf("run %d completion sequence differs from baseline", i)
		}
	}
	// Identical across GOMAXPROCS values.
	orig := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(orig)
	for _, p := range []int{1, 2, 4} {
		runtime.GOMAXPROCS(p)
		if got := run(); !equalIDs(got, baseline) {
			t.Fatalf("GOMAXPROCS=%d completion sequence differs from baseline", p)
		}
	}
	if len(baseline) == 0 {
		t.Fatal("expected a non-empty completion sequence")
	}
}

// Completions returned by a single Poll are ordered by completion time (BC-S4's
// primary key), so a caller sees them in the order they actually finished.
func TestBCS4_CompletionOrderByTime(t *testing.T) {
	// bandwidth 1 → serviceTicks = base + bytes. Same base; larger job finishes later.
	s := mustNew(t, oneTier(2, 0, 10, 10))
	// Two reads submitted at tick 0; the smaller finishes first.
	small, _ := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 5, SubmitTick: 0})  // completes at 15
	large, _ := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 90, SubmitTick: 0}) // completes at 100
	got := s.Poll(1000)
	want := []JobID{small, large}
	if !equalIDs(got, want) {
		t.Fatalf("completion order %v, want %v (by completion time)", got, want)
	}
}

// MaxQueueDepth=0 (the vLLM default) never rejects, even under a large backlog.
func TestSubmit_UnboundedByDefault(t *testing.T) {
	s := mustNew(t, oneTier(1, 0, 1000, 1000)) // 1 slow server; everything else queues
	for i := 0; i < 10_000; i++ {
		if _, ok := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 100, SubmitTick: 0}); !ok {
			t.Fatalf("unbounded station rejected job %d", i)
		}
	}
}

// A positive MaxQueueDepth bounds the waiting queue and yields the Rejected path
// (Submit → (0,false)); draining below capacity re-opens admission.
func TestSubmit_RejectedWhenQueueFull(t *testing.T) {
	// 1 read server, bandwidth 1, base 100. MaxQueueDepth=2 waiting reads.
	s := mustNew(t, Config{Tiers: []TierConfig{{
		NRead: 1, NWrite: 0, ReadBaseTicks: 100, ReadBytesPerTick: 1, WriteBytesPerTick: 1,
		MaxQueueDepth: 2,
	}}})

	// Job A starts service immediately (server free) → not counted against the queue.
	a, ok := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 0, SubmitTick: 0})
	if !ok {
		t.Fatal("job A rejected")
	}
	// Two more fill the queue (server busy).
	if _, ok := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 0, SubmitTick: 0}); !ok {
		t.Fatal("job B rejected while queue had room")
	}
	if _, ok := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 0, SubmitTick: 0}); !ok {
		t.Fatal("job C rejected while queue had room")
	}
	// Queue now full → reject.
	if id, ok := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 0, SubmitTick: 0}); ok || id != 0 {
		t.Fatalf("expected rejection (0,false), got (%d,%v)", id, ok)
	}

	// Advance so job A completes (at tick 100), freeing the server and pulling one
	// queued job into service → queue has room again.
	if got := s.Poll(100); len(got) == 0 || got[0] != a {
		t.Fatalf("expected job A to complete at tick 100, got %v", got)
	}
	if _, ok := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 0, SubmitTick: 100}); !ok {
		t.Fatal("expected admission after draining below capacity")
	}
}

func TestActiveJobs_CountsByJobDirection(t *testing.T) {
	// 1 read server, 1 write server; slow so jobs stay in service.
	s := mustNew(t, oneTier(1, 1, 1000, 1000))
	s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 0, SubmitTick: 0})
	s.Submit(TransferJob{Tier: 0, Direction: Write, Bytes: 0, SubmitTick: 0})
	if r, w := s.ActiveJobs(0, Read), s.ActiveJobs(0, Write); r != 1 || w != 1 {
		t.Fatalf("ActiveJobs read=%d write=%d, want 1,1", r, w)
	}
}

// Panics guard genuine programmer errors (out-of-range tier, bad direction).
func TestPanics_OnProgrammerError(t *testing.T) {
	s := mustNew(t, oneTier(1, 1, 10, 10))
	assertPanics(t, "out-of-range tier Submit", func() {
		s.Submit(TransferJob{Tier: 5, Direction: Read, Bytes: 0, SubmitTick: 0})
	})
	assertPanics(t, "invalid direction Submit", func() {
		s.Submit(TransferJob{Tier: 0, Direction: Direction(9), Bytes: 0, SubmitTick: 0})
	})
	assertPanics(t, "out-of-range tier ActiveJobs", func() { s.ActiveJobs(5, Read) })
	assertPanics(t, "out-of-range tier ServiceTicks", func() { s.ServiceTicks(-1, Read, 0) })
}

func equalIDs(a, b []JobID) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func assertPanics(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic, got none", name)
		}
	}()
	fn()
}

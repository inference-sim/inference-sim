package kvtransfer

import (
	"sort"
	"testing"
)

// transferStationLike is the minimal surface the BC-S2 metamorphic harness needs.
// Both the real *TransferStation and the throwaway fairShareStation satisfy it,
// so the same metamorphic manipulation runs against both. It is test-only — the
// production package intentionally exposes a concrete type, not an interface.
type transferStationLike interface {
	Submit(TransferJob) (JobID, bool)
	Poll(now int64) []JobID
}

var (
	_ transferStationLike = (*TransferStation)(nil)
	_ transferStationLike = (*fairShareStation)(nil)
)

// --- BC-S2 differential test: drain order vs thread_pool.py's worker loop ------

// vLLM's DualQueueThreadPool._worker (tiering/fs/thread_pool.py:165-197) pops from
// its primary queue if non-empty, else the secondary queue. Load-priority threads
// have primary=load(read); store-priority threads have primary=store(write). These
// tests encode that discipline as the expected drain behavior and assert the
// station reproduces it.

// Priority: with reads AND writes waiting, a read-priority server takes a read and
// a write-priority server takes a write (no fallback needed).
func TestBCS2_Differential_Priority(t *testing.T) {
	// 1 read-priority + 1 write-priority server; bandwidth huge so every job takes
	// exactly its base (10 ticks), making completion order == service order.
	s := mustNew(t, Config{Tiers: []TierConfig{{
		NRead: 1, NWrite: 1,
		ReadBaseTicks: 10, WriteBaseTicks: 10,
		ReadBytesPerTick: 1e9, WriteBytesPerTick: 1e9,
	}}})

	r1, _ := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 1, SubmitTick: 0})
	w1, _ := s.Submit(TransferJob{Tier: 0, Direction: Write, Bytes: 1, SubmitTick: 0})
	r2, _ := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 1, SubmitTick: 0})
	w2, _ := s.Submit(TransferJob{Tier: 0, Direction: Write, Bytes: 1, SubmitTick: 0})

	// After submission: the read-priority server took a read and the write-priority
	// server took a write — priority, not fallback.
	if r, w := s.ActiveJobs(0, Read), s.ActiveJobs(0, Write); r != 1 || w != 1 {
		t.Fatalf("expected 1 read + 1 write in service (priority), got read=%d write=%d", r, w)
	}

	got := s.Poll(1000)
	// Round 1 (tick 10): r1 (id) and w1; round 2 (tick 20): r2 and w2. Within a
	// tick the deterministic tie-break orders Read before Write (BC-S4).
	want := []JobID{r1, w1, r2, w2}
	if !equalIDs(got, want) {
		t.Fatalf("drain order %v, want %v", got, want)
	}
}

// Fallback: with only writes waiting, a read-priority server drains the write
// queue (its primary read queue is empty) — thread_pool.py:174-176. Both servers
// end up serving writes, and a server never idles while a queue is non-empty.
func TestBCS2_Differential_Fallback(t *testing.T) {
	s := mustNew(t, Config{Tiers: []TierConfig{{
		NRead: 1, NWrite: 1,
		ReadBaseTicks: 10, WriteBaseTicks: 10,
		ReadBytesPerTick: 1e9, WriteBytesPerTick: 1e9,
	}}})

	w1, _ := s.Submit(TransferJob{Tier: 0, Direction: Write, Bytes: 1, SubmitTick: 0})
	w2, _ := s.Submit(TransferJob{Tier: 0, Direction: Write, Bytes: 1, SubmitTick: 0})
	w3, _ := s.Submit(TransferJob{Tier: 0, Direction: Write, Bytes: 1, SubmitTick: 0})

	// The read-priority server fell back to a write, so BOTH servers are serving
	// writes: 2 writes in service (the read-priority server would only idle if
	// both queues were empty — BC-S2).
	if w := s.ActiveJobs(0, Write); w != 2 {
		t.Fatalf("expected 2 writes in service via fallback, got %d", w)
	}
	if r := s.ActiveJobs(0, Read); r != 0 {
		t.Fatalf("expected 0 reads in service, got %d", r)
	}

	got := s.Poll(1000)
	// w1 (read-server) and w2 (write-server) at tick 10; w3 at tick 20.
	want := []JobID{w1, w2, w3}
	if !equalIDs(got, want) {
		t.Fatalf("fallback drain order %v, want %v", got, want)
	}
}

// A read-priority server never idles while reads wait: with 1 read-priority + 2
// write-priority servers and 3 reads submitted, all 3 reads enter service at once
// (write-priority servers fall back to reads).
func TestBCS2_NoServerIdlesWhileWorkWaits(t *testing.T) {
	s := mustNew(t, oneTier(1, 2, 1000, 1000)) // slow so jobs stay in service
	for i := 0; i < 3; i++ {
		s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 0, SubmitTick: 0})
	}
	if r := s.ActiveJobs(0, Read); r != 3 {
		t.Fatalf("expected all 3 reads in service (write servers fall back), got %d", r)
	}
}

// --- BC-S5 differential: one job = one unit of service covering many blocks -----

// A single job carrying a large multi-block payload is served as ONE unit that
// occupies exactly one server for base+bytes/bandwidth — never split into
// per-block sub-tasks (enqueue_store(job_id, 1, [task]), fs/manager.py:229).
func TestBCS5_JobIsTheUnitOfService(t *testing.T) {
	s := mustNew(t, oneTier(1, 0, 0, 0)) // 1 read server, bandwidth 1 → serviceTicks = bytes
	// One job of 1000 bytes = many blocks batched into a single task.
	big, _ := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 1000, SubmitTick: 0})
	// A second job cannot start until the first frees the single server, proving
	// the big job occupies exactly one server for its whole duration.
	small, _ := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 1, SubmitTick: 0})

	if s.ActiveJobs(0, Read) != 1 {
		t.Fatalf("expected exactly 1 job in service (the big one), got %d", s.ActiveJobs(0, Read))
	}
	// Big completes at tick 1000; only then does the small one start (completes at 1001).
	if got := s.Poll(999); len(got) != 0 {
		t.Fatalf("big multi-block job completed early: %v", got)
	}
	if got := s.Poll(1000); !equalIDs(got, []JobID{big}) {
		t.Fatalf("expected big job to complete at 1000, got %v", got)
	}
	if got := s.Poll(1001); !equalIDs(got, []JobID{small}) {
		t.Fatalf("expected small job to complete at 1001, got %v", got)
	}
}

// --- BC-S2 metamorphic test + fair-share discrimination ------------------------

// readLatencySum runs a fixed scripted workload (a write burst followed by long
// reads) against a station and returns the sum of read completion latencies. It
// polls tick-by-tick over a bounded horizon so each read's exact completion tick
// is captured, and fails if the horizon is too short to drain all reads.
func readLatencySum(t *testing.T, st transferStationLike) int64 {
	t.Helper()
	const (
		nWrites   = 8
		writeSize = 10 // short writes
		nReads    = 16
		readSize  = 100 // long reads
		horizon   = 200_000
	)
	readIDs := map[JobID]bool{}
	// Write burst first, then the reads arrive (all at tick 0).
	for i := 0; i < nWrites; i++ {
		st.Submit(TransferJob{Tier: 0, Direction: Write, Bytes: writeSize, SubmitTick: 0})
	}
	for i := 0; i < nReads; i++ {
		id, ok := st.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: readSize, SubmitTick: 0})
		if ok {
			readIDs[id] = true
		}
	}

	var sum int64
	remaining := len(readIDs)
	for tick := int64(0); tick <= horizon && remaining > 0; tick++ {
		for _, id := range st.Poll(tick) {
			if readIDs[id] {
				sum += tick // submit tick is 0, so latency == completion tick
				remaining--
			}
		}
	}
	if remaining > 0 {
		t.Fatalf("workload did not drain within horizon %d (%d reads left)", horizon, remaining)
	}
	return sum
}

// TestBCS2_Metamorphic_ReadLatencyNonIncreasing is the discriminating test the
// issue requires: raising NWrite (holding everything else) must NOT raise read
// latency for the station, and a fair-share model must FAIL that relation with
// the opposite sign. If the fair-share leg ever satisfied non-increasing, the
// test would have no power.
func TestBCS2_Metamorphic_ReadLatencyNonIncreasing(t *testing.T) {
	const (
		nReadServers = 1
		lowWrite     = 1
		highWrite    = 8
	)

	// --- The station (correct model): read latency is NON-INCREASING in NWrite,
	// because extra write servers add fallback capacity for reads.
	stationLat := func(nWrite int) int64 {
		s := mustNew(t, Config{Tiers: []TierConfig{{
			NRead: nReadServers, NWrite: nWrite,
			ReadBaseTicks: 0, WriteBaseTicks: 0,
			ReadBytesPerTick: 1, WriteBytesPerTick: 1,
		}}})
		return readLatencySum(t, s)
	}
	lowStation := stationLat(lowWrite)
	highStation := stationLat(highWrite)
	if highStation > lowStation {
		t.Fatalf("BC-S2 violated: read latency rose when NWrite grew (%d→%d): %d → %d",
			lowWrite, highWrite, lowStation, highStation)
	}
	// The effect must be real, not merely non-worse: more write servers genuinely
	// help reads via fallback in this workload.
	if highStation >= lowStation {
		t.Fatalf("expected read latency to strictly improve with more write servers, got %d → %d",
			lowStation, highStation)
	}

	// --- The fair-share caricature (WRONG model): read latency INCREASES with
	// NWrite. This proves the metamorphic relation discriminates rather than
	// merely confirms.
	fairLat := func(nWrite int) int64 {
		fs := newFairShareStation(int64(nReadServers+nWrite), 1.0, 0)
		return readLatencySum(t, fs)
	}
	lowFair := fairLat(lowWrite)
	highFair := fairLat(highWrite)
	if highFair <= lowFair {
		t.Fatalf("fair-share model failed to exhibit the opposite sign (%d → %d); "+
			"the metamorphic test would then lack discriminating power", lowFair, highFair)
	}
}

// --- Throwaway fair-share model (test-only; deliberately the WRONG shape) -------

// fairShareStation models the tier as a single shared-bandwidth pipe whose total
// bandwidth is divided across all provisioned concurrent slots (cap =
// NRead+NWrite) — the shape of sim/cluster's PDTransferContention. Adding threads
// splits the bandwidth more ways, so every transfer slows down: the opposite sign
// to the real station under BC-S2. It exists ONLY to give the metamorphic test
// something it must reject.
type fairShareStation struct {
	cap     int64
	totalBW float64
	base    int64
	nextID  JobID
	done    map[JobID]int64 // id → completion tick
	polled  map[JobID]bool
}

func newFairShareStation(cap int64, totalBW float64, base int64) *fairShareStation {
	return &fairShareStation{
		cap: cap, totalBW: totalBW, base: base,
		done:   map[JobID]int64{},
		polled: map[JobID]bool{},
	}
}

func (f *fairShareStation) Submit(j TransferJob) (JobID, bool) {
	f.nextID++
	// Effective per-transfer bandwidth = totalBW / cap (bandwidth split across all
	// provisioned slots). More threads (larger cap) ⇒ slower transfers.
	eff := f.totalBW / float64(f.cap)
	f.done[f.nextID] = j.SubmitTick + f.base + int64(float64(j.Bytes)/eff)
	return f.nextID, true
}

func (f *fairShareStation) Poll(now int64) []JobID {
	var res []JobID
	for id, at := range f.done {
		if at <= now && !f.polled[id] {
			res = append(res, id)
		}
	}
	sort.Slice(res, func(i, j int) bool {
		if f.done[res[i]] != f.done[res[j]] {
			return f.done[res[i]] < f.done[res[j]]
		}
		return res[i] < res[j]
	})
	for _, id := range res {
		f.polled[id] = true
	}
	return res
}

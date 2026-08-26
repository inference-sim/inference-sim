package kvtransfer

import (
	"container/heap"
	"fmt"
	"math"
)

// Direction distinguishes the two transfer classes a tier serves.
//
// The vocabulary is vLLM's own (tiering/manager.py:109-115): read/write name
// CPU↔secondary transfers. A Read moves a block from the secondary tier up into
// CPU (the "load queue" of the FS thread pool); a Write moves a block from CPU
// down to the secondary tier (the "store queue").
type Direction int

const (
	// Read is a secondary→CPU transfer (vLLM: load queue, load-priority threads).
	Read Direction = iota
	// Write is a CPU→secondary transfer (vLLM: store queue, store-priority threads).
	Write
)

// String renders a Direction for diagnostics and test failure messages.
func (d Direction) String() string {
	switch d {
	case Read:
		return "read"
	case Write:
		return "write"
	default:
		return fmt.Sprintf("Direction(%d)", int(d))
	}
}

// JobID is a stable, monotonically increasing identity for an accepted job.
// IDs start at 1; the zero value is never a valid job and is returned alongside
// a false "accepted" flag when Submit rejects a job (BC: Submit → JobId|Rejected).
type JobID int64

// TransferJob is one unit of service (BC-S5): a single pool task that may cover
// many blocks. Bytes is the total payload of all blocks in the job; the station
// serves the job atomically and never splits it.
type TransferJob struct {
	Tier       int       // secondary-tier index in [0, TierCount)
	Direction  Direction // Read or Write
	Bytes      int64     // total payload bytes (≥ 0)
	SubmitTick int64     // simulation tick at which the job is submitted (≥ 0, non-decreasing)

	// JitterFactor optionally scales this job's (depth-adjusted) service time by a
	// caller-supplied multiplicative factor (#1581, BC-D5): completeAt = start +
	// round(serviceTicks · JitterFactor). It is drawn by the caller from a seeded
	// RNG (the station itself draws no randomness — BC-S4), so the station stays a
	// pure function of its inputs. A value ≤ 0 (the zero-value default) is the
	// "no jitter" sentinel: the exact integer service time is used unchanged
	// (byte-identical, BC-D2/INV-6).
	JitterFactor float64
}

// TierConfig describes one tier's server pool and its per-direction service
// physics (BC-S1, BC-S3).
type TierConfig struct {
	// NRead is the number of read-priority servers (≥ 0). Read-priority servers
	// drain the read queue first, then fall back to the write queue (BC-S2).
	NRead int
	// NWrite is the number of write-priority servers (≥ 0). NRead+NWrite must be
	// ≥ 1 so submitted jobs can eventually be served.
	NWrite int

	// ReadBaseTicks / WriteBaseTicks are the fixed service latencies in ticks
	// (≥ 0), per direction (BC-S3).
	ReadBaseTicks  int64
	WriteBaseTicks int64

	// ReadBytesPerTick / WriteBytesPerTick are the per-direction bandwidths in
	// bytes per tick (> 0). Read and write are independent (BC-S3). With a
	// queue-depth ramp enabled (see below) these are the SATURATED (peak, q≥Qsat)
	// bandwidths; without a ramp they are the constant bandwidth used at all
	// depths.
	ReadBytesPerTick  float64
	WriteBytesPerTick float64

	// SaturationQueueDepth (Qsat) and SingleTransferFraction (f₁) describe the
	// device's queue-depth bandwidth curve (#1581, BC-D1). Effective per-transfer
	// bandwidth ramps linearly from f₁·bw at in-service depth q=1 up to bw (peak)
	// at q=Qsat, and is flat (bw) for q>Qsat. The ramp is DISABLED — bandwidth is
	// the constant bw at every depth, byte-identical to the pre-#1581 linear model
	// (BC-D2, INV-6) — whenever Qsat≤1, f₁≥1, or f₁≤0 (the zero-value default). It
	// applies equally to both directions, indexed by that direction's in-service
	// count. q is fixed at service-start and never recomputed mid-flight (DL-4),
	// which keeps completeAt stable for the completion heap (BC-S4). When Qsat≥2
	// the ramp is active and f₁ must be in (0,1].
	SaturationQueueDepth   int
	SingleTransferFraction float64

	// MaxQueueDepth bounds each of the tier's two queues. 0 means unbounded,
	// matching vLLM's deque()-backed queues; a positive value enables the
	// Submit → Rejected path when the target queue is full.
	MaxQueueDepth int
}

// Config is the full station configuration: one TierConfig per secondary tier,
// in lookup order.
type Config struct {
	Tiers []TierConfig
}

// maxServiceTicks caps the variable (bytes/bandwidth) part of a service time to
// keep completeAt = startTick + serviceTicks well within int64 (≈36k years at
// microsecond ticks). It only ever engages for adversarially large Bytes and is
// a robustness guard, not a physical parameter.
const maxServiceTicks int64 = 1 << 60

// serverClass records which priority group a server belongs to, so a completing
// job returns its server to the correct free pool.
type serverClass int

const (
	readServer serverClass = iota
	writeServer
)

// job is the station's internal representation of a submitted transfer. Queues
// hold *job; the same pointer is pushed onto the completion heap once in service.
type job struct {
	id         JobID
	tier       int
	direction  Direction
	bytes      int64
	submitTick int64
	completeAt int64       // valid once the job is in service
	server     serverClass // which server group is serving it (valid once in service)
	jitter     float64     // caller-supplied service-time multiplier; ≤0 == none (BC-D5)
}

// tierState is the mutable per-tier state.
type tierState struct {
	cfg    TierConfig
	readQ  []*job // FIFO of pending reads (submit order == id order)
	writeQ []*job // FIFO of pending writes

	freeRead  int // idle read-priority servers
	freeWrite int // idle write-priority servers

	activeRead  int // in-service read jobs (BC-S1 accounting; O(1))
	activeWrite int // in-service write jobs
}

// TransferStation is a bounded, multi-tier, priority queueing station. It is a
// pure deterministic data structure: it owns no clock and draws no randomness.
// Callers drive it with Submit (at a job's SubmitTick) and Poll (at the current
// tick). Both SubmitTick and Poll's now must be non-decreasing across calls.
//
// Not safe for concurrent use.
type TransferStation struct {
	tiers    []*tierState
	inflight completionHeap // in-service jobs across all tiers, ordered by BC-S4 tuple
	lastTick int64          // internal clock; never decreases
	nextID   JobID          // last assigned id; ids start at 1

	// pendingCompleted accumulates ids completed since the last Poll, in
	// completion order. Filled by advance(), drained by Poll().
	pendingCompleted []JobID
}

// New validates cfg and constructs a TransferStation. It returns an error rather
// than panicking so the caller (a config resolver) can report a bad tier setup
// at startup.
func New(cfg Config) (*TransferStation, error) {
	if len(cfg.Tiers) == 0 {
		return nil, fmt.Errorf("kvtransfer: Config.Tiers must contain at least one tier")
	}
	tiers := make([]*tierState, len(cfg.Tiers))
	for i, tc := range cfg.Tiers {
		if err := validateTier(i, tc); err != nil {
			return nil, err
		}
		tiers[i] = &tierState{
			cfg:       tc,
			freeRead:  tc.NRead,
			freeWrite: tc.NWrite,
		}
	}
	return &TransferStation{tiers: tiers}, nil
}

func validateTier(i int, tc TierConfig) error {
	if tc.NRead < 0 {
		return fmt.Errorf("kvtransfer: tier %d NRead must be ≥ 0, got %d", i, tc.NRead)
	}
	if tc.NWrite < 0 {
		return fmt.Errorf("kvtransfer: tier %d NWrite must be ≥ 0, got %d", i, tc.NWrite)
	}
	if tc.NRead+tc.NWrite < 1 {
		return fmt.Errorf("kvtransfer: tier %d must have NRead+NWrite ≥ 1, got %d", i, tc.NRead+tc.NWrite)
	}
	if tc.ReadBaseTicks < 0 || tc.WriteBaseTicks < 0 {
		return fmt.Errorf("kvtransfer: tier %d base ticks must be ≥ 0 (read=%d write=%d)", i, tc.ReadBaseTicks, tc.WriteBaseTicks)
	}
	if !(tc.ReadBytesPerTick > 0) {
		return fmt.Errorf("kvtransfer: tier %d ReadBytesPerTick must be > 0, got %g", i, tc.ReadBytesPerTick)
	}
	if !(tc.WriteBytesPerTick > 0) {
		return fmt.Errorf("kvtransfer: tier %d WriteBytesPerTick must be > 0, got %g", i, tc.WriteBytesPerTick)
	}
	if tc.MaxQueueDepth < 0 {
		return fmt.Errorf("kvtransfer: tier %d MaxQueueDepth must be ≥ 0, got %d", i, tc.MaxQueueDepth)
	}
	if tc.SaturationQueueDepth < 0 {
		return fmt.Errorf("kvtransfer: tier %d SaturationQueueDepth must be ≥ 0, got %d", i, tc.SaturationQueueDepth)
	}
	// The single-transfer fraction only constrains behavior when the ramp is
	// active (Qsat ≥ 2); a valid fraction is finite and in (0, 1] (1.0 is a
	// no-op ramp). When Qsat ≤ 1 the fraction is ignored and unconstrained.
	if tc.SaturationQueueDepth >= 2 {
		f := tc.SingleTransferFraction
		if math.IsNaN(f) || math.IsInf(f, 0) || !(f > 0) || f > 1 {
			return fmt.Errorf("kvtransfer: tier %d SingleTransferFraction must be in (0,1] when SaturationQueueDepth ≥ 2, got %g", i, f)
		}
	}
	return nil
}

// TierCount returns the number of configured secondary tiers.
func (s *TransferStation) TierCount() int { return len(s.tiers) }

// ServiceTicks returns the UNCONTENDED (in-service depth q=1) service time in
// ticks a job of the given size would take on the given tier and direction:
// base + floor(bytes/bandwidth) (BC-S3). With a queue-depth ramp configured this
// is the single-transfer cost (f₁·bw); without one it is the constant-bandwidth
// cost. It is a pure query — it does not mutate the station — and panics only on
// an out-of-range tier or invalid direction (programmer error).
func (s *TransferStation) ServiceTicks(tier int, dir Direction, bytes int64) int64 {
	s.checkTier(tier)
	return s.tiers[tier].cfg.serviceTicks(dir, bytes, 1)
}

// ServiceTicksAtDepth returns the service time in ticks for a job of the given
// size at the given in-service concurrency q (same direction, self-included),
// applying the queue-depth bandwidth ramp (#1581, BC-D1). q<1 is treated as 1.
// Pure query; panics only on an out-of-range tier or invalid direction.
func (s *TransferStation) ServiceTicksAtDepth(tier int, dir Direction, bytes int64, q int) int64 {
	s.checkTier(tier)
	return s.tiers[tier].cfg.serviceTicks(dir, bytes, q)
}

// serviceTicks implements BC-S3 + the #1581 queue-depth ramp (BC-D1) for one
// tier: base + floor(bytes/effBW(q)), per direction. effBW(q) ramps linearly
// from f₁·bw at q=1 to bw at q=Qsat and is flat (bw) beyond; when the ramp is
// disabled (Qsat≤1, f₁≥1, or f₁≤0) effBW is the constant bw, making the result
// byte-identical to the pre-#1581 linear formula (BC-D2). The variable part is
// clamped to maxServiceTicks to prevent int64 overflow on adversarially large
// Bytes.
func (c TierConfig) serviceTicks(dir Direction, bytes int64, q int) int64 {
	var base int64
	var bw float64
	switch dir {
	case Read:
		base, bw = c.ReadBaseTicks, c.ReadBytesPerTick
	case Write:
		base, bw = c.WriteBaseTicks, c.WriteBytesPerTick
	default:
		panic(fmt.Sprintf("kvtransfer: invalid direction %d", int(dir)))
	}
	if bytes < 0 {
		panic(fmt.Sprintf("kvtransfer: negative bytes %d", bytes))
	}
	// bw > 0 is guaranteed by New's validation.
	effBW := effectiveBandwidth(bw, q, c.SaturationQueueDepth, c.SingleTransferFraction)
	variableF := float64(bytes) / effBW
	var variable int64
	if variableF >= float64(maxServiceTicks) {
		variable = maxServiceTicks
	} else {
		variable = int64(variableF) // truncation toward zero; deterministic
	}
	return base + variable
}

// effectiveBandwidth applies the queue-depth ramp (BC-D1). It returns the
// constant peak bw when the ramp is disabled (qsat≤1, f1≥1, or f1≤0), the
// single-transfer bandwidth f1·bw at q=1, a linear interpolation for
// 1<q<qsat, and the peak bw for q≥qsat. q is clamped to ≥1.
func effectiveBandwidth(bw float64, q, qsat int, f1 float64) float64 {
	if qsat < 2 || f1 <= 0 || f1 >= 1 {
		return bw // ramp disabled — constant bandwidth (BC-D2)
	}
	if q < 1 {
		q = 1
	}
	if q >= qsat {
		return bw // saturated / flat
	}
	ramp := f1 + (1-f1)*float64(q-1)/float64(qsat-1)
	return bw * ramp
}

// Submit enqueues a transfer job. It returns the assigned JobID and true on
// acceptance, or (0, false) when the target queue is full (MaxQueueDepth > 0 and
// already at capacity). It panics on an out-of-range tier, an invalid direction,
// or negative bytes — all programmer errors.
//
// SubmitTick must be ≥ the SubmitTick/now of every prior Submit/Poll call. The
// station advances its internal clock to SubmitTick (processing any completions
// that are due strictly before, and at, that tick) before enqueuing, so a
// completion that frees a server is visible to this job.
func (s *TransferStation) Submit(j TransferJob) (JobID, bool) {
	s.checkTier(j.Tier)
	if j.Direction != Read && j.Direction != Write {
		panic(fmt.Sprintf("kvtransfer: invalid direction %d", int(j.Direction)))
	}
	if j.Bytes < 0 {
		panic(fmt.Sprintf("kvtransfer: negative bytes %d", j.Bytes))
	}

	// Bring the clock up to the submit tick so completions that are due by now
	// have freed their servers before this job competes for one.
	s.advance(j.SubmitTick)

	ts := s.tiers[j.Tier]
	if ts.cfg.MaxQueueDepth > 0 {
		depth := len(ts.readQ)
		if j.Direction == Write {
			depth = len(ts.writeQ)
		}
		if depth >= ts.cfg.MaxQueueDepth {
			return 0, false // Rejected
		}
	}

	s.nextID++
	nj := &job{
		id:         s.nextID,
		tier:       j.Tier,
		direction:  j.Direction,
		bytes:      j.Bytes,
		submitTick: j.SubmitTick,
		jitter:     j.JitterFactor,
	}
	if j.Direction == Read {
		ts.readQ = append(ts.readQ, nj)
	} else {
		ts.writeQ = append(ts.writeQ, nj)
	}

	// The new job may start immediately if a server is free. Service starts no
	// earlier than the current clock: for the expected non-decreasing SubmitTick
	// this is SubmitTick itself; the max() guards against a caller submitting
	// slightly out of order so no job is ever placed into service in the past
	// (which would leave a completeAt below lastTick).
	startAt := j.SubmitTick
	if startAt < s.lastTick {
		startAt = s.lastTick
	}
	s.assign(j.Tier, startAt)
	return nj.id, true
}

// Poll advances the station's clock to now and returns the ids of jobs that
// completed since the previous Poll, in deterministic completion order (BC-S4).
// now must be ≥ every prior SubmitTick/now. The returned slice may be empty.
func (s *TransferStation) Poll(now int64) []JobID {
	s.advance(now)
	res := s.pendingCompleted
	s.pendingCompleted = nil
	return res
}

// ActiveJobs returns the number of jobs of the given direction currently in
// service (assigned to a server, not yet completed) on the given tier. It counts
// by the job's own direction regardless of which server group is serving it, so
// a read served by a write-priority server (fallback) still counts as an active
// read. Panics on an out-of-range tier or invalid direction.
func (s *TransferStation) ActiveJobs(tier int, dir Direction) int {
	s.checkTier(tier)
	ts := s.tiers[tier]
	switch dir {
	case Read:
		return ts.activeRead
	case Write:
		return ts.activeWrite
	default:
		panic(fmt.Sprintf("kvtransfer: invalid direction %d", int(dir)))
	}
}

// advance processes all completions due at or before target, freeing servers and
// re-assigning work as it goes, then sets the clock to target. It never moves the
// clock backward: a target ≤ lastTick is a no-op on the clock (completions due at
// lastTick were already processed when the clock reached it).
func (s *TransferStation) advance(target int64) {
	if target <= s.lastTick {
		return
	}
	for s.inflight.Len() > 0 {
		next := s.inflight[0]
		if next.completeAt > target {
			break
		}
		j := heap.Pop(&s.inflight).(*job)
		ts := s.tiers[j.tier]
		if j.server == readServer {
			ts.freeRead++
		} else {
			ts.freeWrite++
		}
		if j.direction == Read {
			ts.activeRead--
		} else {
			ts.activeWrite--
		}
		s.pendingCompleted = append(s.pendingCompleted, j.id)
		// Work-conserving: the freed server immediately picks up waiting work.
		// The new job starts at this completion time (j.completeAt ≤ target), so
		// if it is also due by target the loop will complete it in a later pass.
		s.assign(j.tier, j.completeAt)
	}
	s.lastTick = target
}

// assign starts as many waiting jobs on free servers as the priority-with-
// fallback discipline allows, using startTick as the service start time.
//
// One pass is provably complete: the read-priority loop runs until either no
// read-priority server is free or both queues are empty; the write-priority loop
// then does the same. Afterward, if any server is still free, both queues must be
// empty — which is exactly BC-S2 ("a server idles only when both queues empty").
func (s *TransferStation) assign(tier int, startTick int64) {
	ts := s.tiers[tier]

	// Read-priority servers: read queue first, then fall back to the write queue.
	for ts.freeRead > 0 && (len(ts.readQ) > 0 || len(ts.writeQ) > 0) {
		var nj *job
		if len(ts.readQ) > 0 {
			nj = popFront(&ts.readQ)
		} else {
			nj = popFront(&ts.writeQ)
		}
		s.start(ts, nj, readServer, startTick)
		ts.freeRead--
	}

	// Write-priority servers: write queue first, then fall back to the read queue.
	for ts.freeWrite > 0 && (len(ts.writeQ) > 0 || len(ts.readQ) > 0) {
		var nj *job
		if len(ts.writeQ) > 0 {
			nj = popFront(&ts.writeQ)
		} else {
			nj = popFront(&ts.readQ)
		}
		s.start(ts, nj, writeServer, startTick)
		ts.freeWrite--
	}
}

// start puts a job into service on the given server group beginning at startTick,
// computing its completion time and pushing it onto the completion heap. The
// queue-depth ramp uses this job's direction's in-service count INCLUDING itself
// (q = active+1, taken before the increment below), so a lone transfer sees q=1.
// An optional caller-supplied jitter factor (BC-D5) scales the service time; the
// ≤0 sentinel keeps the exact integer path (byte-identical, BC-D2).
func (s *TransferStation) start(ts *tierState, nj *job, sc serverClass, startTick int64) {
	nj.server = sc
	var q int
	if nj.direction == Read {
		q = ts.activeRead + 1
	} else {
		q = ts.activeWrite + 1
	}
	svc := ts.cfg.serviceTicks(nj.direction, nj.bytes, q)
	if nj.jitter > 0 {
		jittered := math.Round(float64(svc) * nj.jitter)
		// Clamp to the same overflow guard serviceTicks uses, so an adversarial
		// (large bytes) × (right-tail factor) product cannot overflow completeAt.
		if jittered >= float64(maxServiceTicks) {
			svc = maxServiceTicks
		} else {
			svc = int64(jittered)
		}
	}
	nj.completeAt = startTick + svc
	if nj.direction == Read {
		ts.activeRead++
	} else {
		ts.activeWrite++
	}
	heap.Push(&s.inflight, nj)
}

func (s *TransferStation) checkTier(tier int) {
	if tier < 0 || tier >= len(s.tiers) {
		panic(fmt.Sprintf("kvtransfer: tier %d out of range [0,%d)", tier, len(s.tiers)))
	}
}

// popFront removes and returns the head of a FIFO queue. It nils the vacated cell
// so the completed job can be garbage-collected; Go's append periodically
// compacts the backing array, keeping amortized cost O(1) and memory bounded by
// the peak queue depth rather than the total job count (objective P).
func popFront(q *[]*job) *job {
	s := *q
	j := s[0]
	s[0] = nil
	*q = s[1:]
	return j
}

// completionHeap is a min-heap of in-service jobs ordered by the BC-S4 tuple
// (completeAt, submitTick, tier, direction, id). Popping it yields jobs in the
// exact order the station reports completions — a total order with no ties, since
// id is unique.
type completionHeap []*job

func (h completionHeap) Len() int { return len(h) }

func (h completionHeap) Less(i, j int) bool {
	a, b := h[i], h[j]
	if a.completeAt != b.completeAt {
		return a.completeAt < b.completeAt
	}
	if a.submitTick != b.submitTick {
		return a.submitTick < b.submitTick
	}
	if a.tier != b.tier {
		return a.tier < b.tier
	}
	if a.direction != b.direction {
		return a.direction < b.direction
	}
	return a.id < b.id
}

func (h completionHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *completionHeap) Push(x any) {
	*h = append(*h, x.(*job))
}

func (h *completionHeap) Pop() any {
	old := *h
	n := len(old)
	j := old[n-1]
	old[n-1] = nil
	*h = old[:n-1]
	return j
}

package workload

import (
	"fmt"
	"math/rand"

	"github.com/inference-sim/inference-sim/sim"
)

// SessionPoolDriver keeps a fixed number of closed-loop sessions active. It
// wraps a SessionManager: intra-session follow-ups are produced by the manager
// unchanged; when a session reaches a terminal state and queued sessions remain,
// the driver admits the next queued session's round-0 request. This models a
// fixed pool of N concurrent "users" drawing from a corpus of captured sessions.
//
// Determinism (INV-6): session order is the expansion order of BuildSessionPool
// (corpus order, then round-robin clones); admission is strictly in that order.
type SessionPoolDriver struct {
	mgr          *SessionManager
	queued       []*sim.Request // all total round-0 requests (originals + clones), in admission order
	nextQueued   int            // index into queued of the next session to admit
	activeCount  int
	totalStarted int // sessions injected so far (initial pool + admitted)
	totalTerm    int // sessions that have reached a terminal state
	totalCount   int // total sessions in the pool (== len of all round-0 requests)
}

// cloneSampler returns a sampler independent of s for stateful sampler types.
// SequenceSampler carries a mutable per-call cursor (index) and replays a fixed
// recorded sequence, so each cloned session needs its own cursor over the same
// values — otherwise a source session and its duplicate corrupt each other's
// per-round sequence via the shared cursor. Stateless samplers (which derive all
// randomness from the *rand.Rand passed to Sample) are safe to share and are
// returned unchanged.
func cloneSampler(s LengthSampler) LengthSampler {
	if seq, ok := s.(*SequenceSampler); ok {
		// Fresh cursor (index=0); the values slice is immutable after construction, so sharing it is safe.
		return &SequenceSampler{values: seq.values}
	}
	return s
}

// cloneBlueprintForDup produces a duplicated blueprint + round-0 request with a
// new unique SessionID and a cache-busting prefix token so the clone does not
// share KV cache with its source. dupIdx is the global clone index (>=1).
func cloneBlueprintForDup(src SessionBlueprint, srcR0 *sim.Request, dupIdx int, rng *rand.Rand) (SessionBlueprint, *sim.Request) {
	newID := fmt.Sprintf("%s_dup%d", src.SessionID, dupIdx)
	bp := src // shallow copy; stateful samplers are independently re-cloned below via cloneSampler
	bp.SessionID = newID
	// Fresh per-session RNG so token IDs differ deterministically from the source.
	bp.RNG = rand.New(rand.NewSource(rng.Int63()))
	// Give the clone independent sampler state. Interface fields copied by the
	// shallow `bp := src` above still point at the SAME underlying sampler
	// object as src for stateful types (e.g. *SequenceSampler's mutable index
	// cursor) — without this, the source and its clone would corrupt each
	// other's per-round sequence by advancing a shared cursor.
	bp.InputSampler = cloneSampler(src.InputSampler)
	bp.OutputSampler = cloneSampler(src.OutputSampler)
	bp.ThinkTimeSampler = cloneSampler(src.ThinkTimeSampler)

	// Cache-busting token prepended to the clone's round-0 input so the block
	// hash chain diverges immediately from the source session (design §6).
	buster := sim.GenerateRandomTokenIDs(bp.RNG, 1)
	newInput := make([]sim.TokenID, 0, len(srcR0.InputTokens)+1)
	newInput = append(newInput, buster...)
	newInput = append(newInput, srcR0.InputTokens...)

	r0 := *srcR0 // shallow copy
	r0.ID = "r_" + newID
	r0.SessionID = newID
	r0.InputTokens = newInput
	return bp, &r0
}

// BuildSessionPool expands the corpus (blueprints + their round-0 requests) to
// total via round-robin duplication, registers all sessions with a
// SessionManager, and returns the driver plus the first concurrent round-0
// requests to inject at simulation start.
//
// concurrent: max concurrently-active sessions (>= 1).
// total: total sessions to replay (>= concurrent). If <= len(corpus), the
// corpus is truncated to total; if greater, clones fill the remainder.
// seed: master seed for clone RNGs (INV-6).
func BuildSessionPool(blueprints []SessionBlueprint, r0Requests []*sim.Request, concurrent, total int, seed int64) (*SessionPoolDriver, []*sim.Request, error) {
	if concurrent < 1 {
		return nil, nil, fmt.Errorf("concurrent must be >= 1, got %d", concurrent)
	}
	if len(blueprints) == 0 {
		return nil, nil, fmt.Errorf("no session blueprints to pool")
	}
	if len(blueprints) != len(r0Requests) {
		return nil, nil, fmt.Errorf("blueprints (%d) and round-0 requests (%d) count mismatch", len(blueprints), len(r0Requests))
	}
	if total < 1 {
		total = len(blueprints)
	}
	if total < concurrent {
		return nil, nil, fmt.Errorf("total (%d) must be >= concurrent (%d)", total, concurrent)
	}

	rng := rand.New(rand.NewSource(seed))
	allBPs := make([]SessionBlueprint, 0, total)
	allR0 := make([]*sim.Request, 0, total)
	dupIdx := 0
	for i := 0; i < total; i++ {
		srcIdx := i % len(blueprints)
		if i < len(blueprints) {
			allBPs = append(allBPs, blueprints[srcIdx])
			allR0 = append(allR0, r0Requests[srcIdx])
		} else {
			dupIdx++
			bp, r0 := cloneBlueprintForDup(blueprints[srcIdx], r0Requests[srcIdx], dupIdx, rng)
			allBPs = append(allBPs, bp)
			allR0 = append(allR0, r0)
		}
	}

	mgr := NewSessionManager(allBPs)
	d := &SessionPoolDriver{
		mgr:        mgr,
		queued:     allR0,
		totalCount: total,
	}
	// Inject the first concurrent sessions immediately.
	initial := make([]*sim.Request, 0, concurrent)
	for i := 0; i < concurrent; i++ {
		initial = append(initial, d.queued[d.nextQueued])
		d.nextQueued++
		d.activeCount++
		d.totalStarted++
	}
	return d, initial, nil
}

// isSessionRequest reports whether a completion drove its session to a terminal
// state. The wrapped SessionManager returns nil follow-ups on every terminal
// path (timeout, dropped, final round, budget, horizon). A round that continues
// returns exactly one follow-up. So: nil follow-ups AND this was a session
// request => the session just terminated.
func isSessionRequest(req *sim.Request) bool { return req.SessionID != "" }

// OnComplete wraps SessionManager.OnComplete. When the inner manager terminates
// a session (returns no follow-up for a session request) and queued sessions
// remain, the driver admits the next session's round-0 request to refill the pool.
func (d *SessionPoolDriver) OnComplete(req *sim.Request, tick int64) []*sim.Request {
	followUps := d.mgr.OnComplete(req, tick)
	if !isSessionRequest(req) {
		return followUps
	}
	if len(followUps) > 0 {
		// Session continues (intra-session follow-up). Pool membership unchanged.
		return followUps
	}
	// Session terminated. Free its slot and admit the next queued session, if any.
	d.activeCount--
	d.totalTerm++
	if d.nextQueued < len(d.queued) {
		next := d.queued[d.nextQueued]
		// Admit at the completion tick (a fresh user starts as this one ends).
		next.ArrivalTime = tick
		d.nextQueued++
		d.activeCount++
		d.totalStarted++
		return []*sim.Request{next}
	}
	return nil
}

// TotalSessions returns the total number of sessions in the pool (after duplication).
func (d *SessionPoolDriver) TotalSessions() int { return d.totalCount }

// Unstarted returns the number of sessions never admitted — nonzero only when a
// hard --horizon cap ends the run with sessions still queued. In self-draining
// mode (no --horizon) this is always 0. The caller logs a warning and reports
// this alongside totalStarted so accounting is closed: totalStarted + Unstarted()
// == TotalSessions() (INV-11 / INV-1). See Task 4 Step 5.
func (d *SessionPoolDriver) Unstarted() int { return d.totalCount - d.totalStarted }

// hasUniqueSessionIDs is a test helper: true iff all pooled round-0 requests
// carry distinct SessionIDs.
func (d *SessionPoolDriver) hasUniqueSessionIDs() bool {
	seen := make(map[string]struct{}, len(d.queued))
	for _, r := range d.queued {
		if _, dup := seen[r.SessionID]; dup {
			return false
		}
		seen[r.SessionID] = struct{}{}
	}
	return true
}

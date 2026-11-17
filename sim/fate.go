package sim

// A request's fate is how it ended (design §8.7, DEV-9): completed; dropped, with a reason;
// timed out; in flight at the horizon; queued at the horizon; or not started. It is written
// once, by FateLedger.End, and a second write is an error. The ledger also keeps its own count
// of the requests the workload created, at the one place every request enters the cluster
// (FateLedger.Create), so a counting error in the fates cannot be hidden by the same error
// elsewhere: the run fails unless the two agree (FateLedger.Check).
//
// The ledger holds the request state machine of SIM-8: a request is created, arrives at the
// gateway, moves through the states of a colocated or a P/D path (validLifecycleMoves), and ends
// in exactly one fate. Every held state has an edge to every fate but not started
// (validLifecycleEdges), so an eviction, a timeout or the horizon can end a request wherever it
// is, and the horizon's fate is read from the state (HorizonFate). A P/D request's stages are
// states of the parent, which ends once.

import (
	"errors"
	"fmt"
	"sort"
)

// ErrNoProgress is the error a run returns when it cannot make progress (SIM-7): work is
// waiting and nothing can move it.
var ErrNoProgress = errors.New("no progress (SIM-7)")

// Fate is how a request ended.
type Fate string

const (
	// FateCompleted is a response the client received in full, including a length-capped one.
	FateCompleted Fate = "completed"
	// FateDropped is a request the system refused or abandoned; FateRecord.Reason says why.
	FateDropped Fate = "dropped"
	// FateTimedOut is a request whose client deadline passed before the response ended.
	FateTimedOut Fate = "timed_out"
	// FateInFlightAtHorizon is a request an engine was running, or whose KV was moving
	// between a prefill and a decode engine, when the horizon ended the run.
	FateInFlightAtHorizon Fate = "in_flight_at_horizon"
	// FateQueuedAtHorizon is a request that had arrived and was waiting (in the gateway, in
	// the cluster's event pipeline or in an engine's queue) when the horizon ended the run.
	FateQueuedAtHorizon Fate = "queued_at_horizon"
	// FateNotStarted is a request the workload created whose arrival fell after the horizon.
	FateNotStarted Fate = "not_started"
)

// fateOrder lists every fate in the order reports print them.
var fateOrder = []Fate{FateCompleted, FateDropped, FateTimedOut, FateInFlightAtHorizon,
	FateQueuedAtHorizon, FateNotStarted}

// Fates returns every fate, in report order.
func Fates() []Fate { return append([]Fate(nil), fateOrder...) }

// IsValidFate reports whether f is one of the fates of §8.7.
func IsValidFate(f Fate) bool {
	for _, g := range fateOrder {
		if f == g {
			return true
		}
	}
	return false
}

// The reasons a request is dropped, named as blis-substrate contracts/spec/result's
// DropReason names them. Each comment names the INV-1 bucket the request lands in.
const (
	DropAdmission       = "admission_rejected"     // an admission check refused it (rejected)
	DropRouting         = "routing_rejected"       // no endpoint could serve it (routing_rejections)
	DropGatewayCapacity = "gateway_queue_rejected" // flow control's queues were full (gateway_queue_rejected)
	DropGatewayExpired  = "gateway_expired"        // it waited in flow control past its TTL (gateway_expired)
	DropGatewayEvicted  = "gateway_evicted"        // the gateway evicted it in flight (gateway_evicted)
	DropUnservable      = "unservable"             // a model server could not serve it (dropped_unservable)
	DropDecodeKV        = "decode_kv_unavailable"  // a P/D decode side could not take its KV (dropped_unservable)
)

// PlannedRound is a round a session planned and never issued: the session ended before it, by a
// timeout, a drop, the horizon or the follow-up budget. It is registered so it has exactly one
// fate, not started (SIM-8), and a row. Its arrival never happened; the row states the arrival
// its session's plan gives it: the clock the session's last issued round ended at, plus the think
// times the plan draws for the rounds from that one to this one (ThinkUs).
type PlannedRound struct {
	// ThinkUs is the sum of the planned think times from the session's last issued round to this
	// round, each drawn from its own round's stream as the round would have drawn it (DEV-6).
	ThinkUs int64
	// ArrivalUs, when HasArrival, is the arrival the planner states itself; otherwise the row's
	// arrival is its session's last issued round's end plus ThinkUs.
	ArrivalUs    int64
	HasArrival   bool
	PromptTokens int
	Phase        Phase
	ID           string
	SessionID    string
	RoundIndex   int
	Model        string
	Adapter      string
	SLOClass     string
	TenantID     string
}

// FateRecord is one request's fate: what it was, why (for a drop), and when it was written.
type FateRecord struct {
	Fate   Fate
	Reason string
	At     int64
}

// LifecycleState is a state of the request state machine (SIM-8, design §8.7).
type LifecycleState string

const (
	// LifecycleCreated: the workload produced the request; its arrival has not happened.
	LifecycleCreated LifecycleState = "created"
	// LifecycleArrived: the request reached the gateway. From here the cluster holds it: in the
	// gateway, in the event pipeline or in an engine's queue, waiting.
	LifecycleArrived LifecycleState = "arrived"
	// LifecycleRunning: a colocated request an engine is running.
	LifecycleRunning LifecycleState = "running"
	// LifecyclePrefillQueued: a P/D request dispatched to its prefill engine, waiting there.
	LifecyclePrefillQueued LifecycleState = "prefill_queued"
	// LifecyclePrefilling: a P/D request whose prefill engine is running its prefill.
	LifecyclePrefilling LifecycleState = "prefilling"
	// LifecycleWaitingDecodeKV: prefill is done; the decode engine has not started the pull of
	// its KV (the handoff waits among the decode engine's waiting handoffs).
	LifecycleWaitingDecodeKV LifecycleState = "waiting_decode_kv"
	// LifecycleTransferring: the decode engine's pull of the request's KV is moving it.
	LifecycleTransferring LifecycleState = "transferring"
	// LifecycleDecodeQueued: the KV landed; the request waits on the decode engine for its first step.
	LifecycleDecodeQueued LifecycleState = "decode_queued"
	// LifecycleDecoding: the decode engine is running the request.
	LifecycleDecoding LifecycleState = "decoding"
	// LifecyclePreempted: an engine preempted the request; it waits in the engine's queue to
	// resume, on whichever stage it was running.
	LifecyclePreempted LifecycleState = "preempted"
	// LifecycleEnded: the request has its fate.
	LifecycleEnded LifecycleState = "ended"
)

// lifecycleOrder lists the states in the order of the machine's path.
var lifecycleOrder = []LifecycleState{LifecycleCreated, LifecycleArrived, LifecycleRunning,
	LifecyclePrefillQueued, LifecyclePrefilling, LifecycleWaitingDecodeKV, LifecycleTransferring,
	LifecycleDecodeQueued, LifecycleDecoding, LifecyclePreempted, LifecycleEnded}

// LifecycleStates returns every state of the machine.
func LifecycleStates() []LifecycleState { return append([]LifecycleState(nil), lifecycleOrder...) }

// heldStates are the states a request is in while the cluster holds it: every state between
// arrival and the fate.
var heldStates = []LifecycleState{LifecycleArrived, LifecycleRunning, LifecyclePrefillQueued,
	LifecyclePrefilling, LifecycleWaitingDecodeKV, LifecycleTransferring, LifecycleDecodeQueued,
	LifecycleDecoding, LifecyclePreempted}

// IsHeld reports whether s is a state in which the cluster holds the request.
func IsHeld(s LifecycleState) bool {
	for _, h := range heldStates {
		if s == h {
			return true
		}
	}
	return false
}

// validLifecycleMoves are the edges between states other than the end. Every held state also
// has an edge to every fate but not started (validLifecycleEdges), so an eviction, a timeout or
// the horizon can end a request wherever it is. Preemption sends a running stage to
// preempted, and the engine resumes the request on the stage it left.
var validLifecycleMoves = map[LifecycleState][]LifecycleState{
	LifecycleCreated:         {LifecycleArrived},
	LifecycleArrived:         {LifecycleRunning, LifecyclePrefillQueued},
	LifecycleRunning:         {LifecyclePreempted},
	LifecyclePrefillQueued:   {LifecyclePrefilling},
	LifecyclePrefilling:      {LifecyclePreempted, LifecycleWaitingDecodeKV},
	LifecycleWaitingDecodeKV: {LifecycleTransferring},
	LifecycleTransferring:    {LifecycleDecodeQueued},
	LifecycleDecodeQueued:    {LifecycleDecoding},
	LifecycleDecoding:        {LifecyclePreempted},
	LifecyclePreempted:       {LifecycleRunning, LifecyclePrefilling, LifecycleDecoding},
}

// CanMove reports whether the machine has an edge from state from to state to.
func CanMove(from, to LifecycleState) bool {
	for _, t := range validLifecycleMoves[from] {
		if t == to {
			return true
		}
	}
	return false
}

// validLifecycleEdges are the edges to a fate (SIM-8). A created request ends only not started,
// when its arrival falls after the horizon or its round is never issued. A held request ends in
// any fate but not started. An ended request has no edge.
var validLifecycleEdges = func() map[LifecycleState]map[Fate]bool {
	m := map[LifecycleState]map[Fate]bool{
		LifecycleCreated: {FateNotStarted: true},
		LifecycleEnded:   {},
	}
	for _, s := range heldStates {
		m[s] = map[Fate]bool{FateCompleted: true, FateDropped: true, FateTimedOut: true,
			FateInFlightAtHorizon: true, FateQueuedAtHorizon: true}
	}
	return m
}()

// CanEnd reports whether a request in state s may end with fate f.
func CanEnd(s LifecycleState, f Fate) bool { return validLifecycleEdges[s][f] }

// HorizonFate is the fate of a request the cluster holds when the horizon stops the run, read
// from its state: one an engine is running, or whose KV is being pulled, is in flight; one
// waiting anywhere (the gateway, an engine's queue, the decode engine's waiting handoffs, or
// preempted) is queued. A state that is not held has no horizon fate.
func HorizonFate(s LifecycleState) (Fate, bool) {
	switch s {
	case LifecycleRunning, LifecyclePrefilling, LifecycleTransferring, LifecycleDecoding:
		return FateInFlightAtHorizon, true
	case LifecycleArrived, LifecyclePrefillQueued, LifecycleWaitingDecodeKV, LifecycleDecodeQueued,
		LifecyclePreempted:
		return FateQueuedAtHorizon, true
	}
	return "", false
}

type fateEntry struct {
	state  LifecycleState
	record FateRecord
	order  int // creation order, for reports
	// planned marks a round the workload planned and never issued. It is not a generated
	// request: no count of created requests or of fates includes it (CreatePlanned).
	planned bool
}

// FateLedger records each request's lifecycle and fate. It is not safe for concurrent use; the
// cluster's event loop is its only writer.
type FateLedger struct {
	entries map[string]*fateEntry
	created int
	ended   int
	// plannedEnded counts the planned rounds never issued, each ended not started.
	plannedEnded int
	counts       map[Fate]int
	drops        map[string]int
}

// NewFateLedger returns an empty ledger.
func NewFateLedger() *FateLedger {
	return &FateLedger{entries: map[string]*fateEntry{}, counts: map[Fate]int{}, drops: map[string]int{}}
}

// Create registers a request the workload produced. A request ID is never reused within a run
// (SIM-8): a second Create of the same ID is an error.
func (l *FateLedger) Create(id string) error {
	if _, ok := l.entries[id]; ok {
		return fmt.Errorf("request %q created twice: a request identifier is never reused within a run (SIM-8)", id)
	}
	l.entries[id] = &fateEntry{state: LifecycleCreated, order: len(l.entries)}
	l.created++
	return nil
}

// CreatePlanned registers a round the workload planned and never issued. It has a ledger entry,
// so it ends not started and has a row, but it is not a generated request: Created, the fates'
// counts and the conservation check leave it out, and Tally counts it as PlannedNotIssued.
func (l *FateLedger) CreatePlanned(id string) error {
	if _, ok := l.entries[id]; ok {
		return fmt.Errorf("request %q created twice: a request identifier is never reused within a run (SIM-8)", id)
	}
	l.entries[id] = &fateEntry{state: LifecycleCreated, order: len(l.entries), planned: true}
	return nil
}

// Arrive records that request id reached the gateway.
func (l *FateLedger) Arrive(id string) error {
	if _, ok := l.entries[id]; !ok {
		return fmt.Errorf("request %q arrived but was never created", id)
	}
	return l.Advance(id, LifecycleArrived)
}

// Advance moves request id along an edge of the machine to state to. A move the machine has no
// edge for is an error: the request keeps its state.
func (l *FateLedger) Advance(id string, to LifecycleState) error {
	e, ok := l.entries[id]
	if !ok {
		return fmt.Errorf("request %q moved to %s but was never created", id, to)
	}
	if !CanMove(e.state, to) {
		return fmt.Errorf("request %q: no edge from %s to %s (SIM-8)", id, e.state, to)
	}
	e.state = to
	return nil
}

// End writes request id's fate. It is the one place a fate is written; a second write, a fate
// the request's state has no edge to, or a drop without a reason is an error.
func (l *FateLedger) End(id string, f Fate, reason string, at int64) error {
	e, ok := l.entries[id]
	if !ok {
		return fmt.Errorf("fate %s written for request %q, which was never created", f, id)
	}
	if e.state == LifecycleEnded {
		return fmt.Errorf("request %q: fate %s written after its fate %s (at %d µs): a fate is written once (DEV-9)",
			id, f, e.record.Fate, e.record.At)
	}
	if !CanEnd(e.state, f) {
		return fmt.Errorf("request %q: no edge from %s to fate %s", id, e.state, f)
	}
	if (f == FateDropped) != (reason != "") {
		return fmt.Errorf("request %q: fate %s with reason %q: a drop states its reason, and no other fate has one", id, f, reason)
	}
	e.state = LifecycleEnded
	e.record = FateRecord{Fate: f, Reason: reason, At: at}
	if e.planned {
		if f != FateNotStarted {
			return fmt.Errorf("planned round %q ended %s: a round never issued ends not started", id, f)
		}
		l.plannedEnded++
		return nil
	}
	l.ended++
	l.counts[f]++
	if f == FateDropped {
		l.drops[reason]++
	}
	return nil
}

// State returns request id's lifecycle state, and false for an ID never created.
func (l *FateLedger) State(id string) (LifecycleState, bool) {
	e, ok := l.entries[id]
	if !ok {
		return "", false
	}
	return e.state, true
}

// Fate returns request id's fate, and false while it has none.
func (l *FateLedger) Fate(id string) (FateRecord, bool) {
	e, ok := l.entries[id]
	if !ok || e.state != LifecycleEnded {
		return FateRecord{}, false
	}
	return e.record, true
}

// Created is the number of requests the workload produced.
func (l *FateLedger) Created() int { return l.created }

// Held returns the IDs of the requests the cluster holds (arrived, not ended), sorted.
func (l *FateLedger) Held() []string {
	var ids []string
	for id, e := range l.entries {
		if IsHeld(e.state) {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids
}

// Unended returns the IDs of the requests in state s, sorted.
func (l *FateLedger) Unended(s LifecycleState) []string {
	var ids []string
	for id, e := range l.entries {
		if e.state == s {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids
}

// Tally is the number of requests that ended in each fate, and of drops by reason.
type Tally struct {
	Created int
	// PlannedNotIssued counts the planned rounds never issued; they are not in Created or ByFate.
	PlannedNotIssued int
	ByFate           map[Fate]int
	Drops            map[string]int
}

// Total is the number of fates written.
func (t Tally) Total() int {
	n := 0
	for _, f := range fateOrder {
		n += t.ByFate[f]
	}
	return n
}

// Tally returns a copy of the ledger's counts.
func (l *FateLedger) Tally() Tally {
	t := Tally{Created: l.created, PlannedNotIssued: l.plannedEnded, ByFate: make(map[Fate]int, len(l.counts)), Drops: make(map[string]int, len(l.drops))}
	for f, n := range l.counts {
		t.ByFate[f] = n
	}
	for r, n := range l.drops {
		t.Drops[r] = n
	}
	return t
}

// Check is DEV-9's arrival-count check: the number of requests created equals the number of
// fates written. It names the first requests without a fate.
func (l *FateLedger) Check() error {
	if l.created == l.ended {
		return nil
	}
	var missing []string
	missing = append(missing, l.Unended(LifecycleCreated)...)
	missing = append(missing, l.Held()...)
	if len(missing) > 5 {
		missing = append(missing[:5], "...")
	}
	return fmt.Errorf("request conservation (DEV-9): %d requests created, %d fates written; without a fate: %v",
		l.created, l.ended, missing)
}

// Records returns every request's ID and fate in creation order. A request without a fate has
// the zero FateRecord.
func (l *FateLedger) Records() []IDFate {
	out := make([]IDFate, 0, len(l.entries))
	for id, e := range l.entries {
		out = append(out, IDFate{ID: id, FateRecord: e.record, order: e.order})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].order < out[j].order })
	return out
}

// IDFate is one request's ID and fate.
type IDFate struct {
	ID string
	FateRecord
	order int
}

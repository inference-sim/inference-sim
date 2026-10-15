package workload

import (
	"fmt"
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// AgentXReplay drives aiperf 0.12.0's --scenario inferencex-agentx-mvp against the
// simulated cluster: C session-tree lanes snapshotted mid-trace, a warm-up that primes
// every live stream, then a closed-loop profiling window in which subagents fan out
// beside their parent, joins gate later parent turns, recorded replay barriers order
// streams, a 10 s global idle cap pulls in timers, and drained trees recycle into the next
// root. It is a closed-loop driver with the same contract as SessionManager and
// SessionPoolDriver: OnComplete returns the follow-ups a completion causes (wired as the
// cluster's onRequestDone). Because aiperf moves and cancels timers after it sets them, a
// follow-up's ArrivalTime is the send it is scheduled for, and Gate (the cluster's
// ArrivalGate) confirms the send when it comes due.
//
// The aiperf behaviour each step reproduces is cited inline (paths under src/aiperf/).
//
// Determinism (INV-6): no map is iterated where order matters; everything is driven by
// the cluster's event order and the corpus order.
type AgentXReplay struct {
	cfg     AgentXReplayConfig
	sampler *agxSampler
	lanes   []*agxLane

	phase       agxPhase
	profileUs   int64 // profiling start: the warm-up's last return
	windowEndUs int64 // profileUs + duration: no send at or after it
	graceEndUs  int64 // windowEndUs + grace: in-flight requests cancelled
	completeUs  int64 // profiling phase completion (aiperf lifecycle complete_at); -1 until known

	byPlaceholder map[*sim.Request]*agxDispatch
	timers        []*agxDispatch // scheduler timers (LoopScheduler): shiftable by the idle cap
	inFlight      int
	idleSinceUs   int64 // -1: not idle
	warmupLeft    int
	// released are barrier-released sends not yet on the wire: aiperf issues them from a
	// task (ReplayBarrierCoordinator._dispatch_pending), after the return that released
	// them has finished, including its idle-cap check.
	released []*agxDispatch

	instances []*agxInstance
	laneInst  []*agxInstance // the tree currently holding each lane's slot
	nextReqID int64
	streams   int
	out       []*sim.Request
	nowUs     int64

	// SetHorizon, when set, is told the simulation horizon the replay needs: the end of the
	// grace period once profiling starts, earlier if the phase completes first.
	SetHorizon func(int64)

	stats AgentXStats
}

// AgentXReplayConfig is the scenario's run choices.
type AgentXReplayConfig struct {
	Concurrency int     // session-tree lanes
	Seed        int64   // --random-seed (lane snapshot seeds; prompt content salt)
	DurationUs  int64   // benchmark duration (sending window)
	GraceUs     int64   // grace period; math.MaxInt64 = .inf
	IdleCapUs   int64   // global system-idle cap
	MinRatio    float64 // trajectory start window
	MaxRatio    float64
	// RequestTimeoutUs is the client's per-request timeout: aiperf's
	// --request-timeout-seconds default, 6 h (config/endpoint.py:52). A timed-out request
	// is an error record. 0 disables it.
	RequestTimeoutUs int64
}

// AgentXStats are the replay's own counts, reported in the results file's agentic block.
type AgentXStats struct {
	Lanes             int
	TreesStarted      int // tree instances minted: one per lane plus every recycle
	TreesCompleted    int // trees that drained (root and every descendant finished)
	WarmupRequests    int
	MainRequests      int // profiling requests of root streams
	SubagentRequests  int // profiling requests of child streams (subagents and flat chains)
	Cancelled         int // in flight when the grace period ended
	Errors            int // timed out
	ContextOverflows  int // refused by the engine's context check
	IdleCapJumps      int
	IdleCapSkippedUs  int64
	BarrierHeld       int // sends held behind a replay barrier
	JoinSuspensions   int // parent turns suspended on a SPAWN_JOIN
	ChildrenSpawned   int
	ChildrenRefused   int // child first turns refused after the window closed
	DelayedChildSpawn int
}

type agxPhase int

const (
	agxWarmup agxPhase = iota
	agxProfile
	agxClosed // sending window over; in-flight requests may finish until the grace end
	agxDone
)

type agxDispatchState int

const (
	agxTimer   agxDispatchState = iota // waiting for its due time
	agxHeld                            // fired, waiting on a replay barrier
	agxReady                           // cleared to send at dueUs
	agxIssued                          // in flight
	agxDropped                         // cancelled or superseded
)

// agxDispatch is one send the client will make: a turn of a stream, or a lane recycle.
type agxDispatch struct {
	inst      *agxInstance
	stream    *agxStream // nil for a recycle
	turn      int32
	lane      int // recycle only
	recycle   bool
	windowEnd bool // the sending window's duration timeout
	warmup    bool
	start     bool // session start (acquires the tree slot)
	childPath bool // sent via dispatch_child_turn (refusal drains the child's join)
	first     bool // a spawned child's turn 0 (refusal rolls the child back)
	shiftable bool // a LoopScheduler timer (the idle cap moves it); child spawn sleeps are not
	dueUs     int64
	state     agxDispatchState
	ph        *sim.Request // the request pushed for the current due time
}

// agxInstance is one session tree: a lane's trace minted once (its prompts share a
// cache-bust marker), with aiperf's per-tree state: the session slot (SessionTreeRegistry),
// replay barrier frontier (ReplayBarrierCoordinator) and branch joins (BranchOrchestrator).
type agxInstance struct {
	id    int
	lane  int
	trace *AgentXTrace
	salt  uint64

	open        bool // slot acquired (open_tree)
	rootPending bool
	outstanding int // descendants live or registered
	drained     bool

	barrierOn bool
	completed map[agxRef]bool
	held      []*agxDispatch

	future       map[int32]*agxJoin
	active       *agxJoin
	descendants  int
	overlapDone  map[int32]bool
	childStreams []*agxStream // tracked children (BranchOrchestrator._child_to_join)

	root *agxStream
}

type agxStream struct {
	id      int // run-wide instance number
	inst    *agxInstance
	conv    int32
	child   bool
	tracked bool
	gates   []agxGateRef // (gated turn, branch) memberships; empty = background
	done    bool
}

type agxGateRef struct{ gated, branch int32 }

type agxJoin struct {
	gated   int32
	prereqs map[int32]*agxPrereq // branch -> prereq
	order   []int32
	blocked bool
}

type agxPrereq struct {
	expected   int
	completed  map[*agxStream]bool
	registered bool
}

func (j *agxJoin) satisfied() bool {
	for _, b := range j.order {
		p := j.prereqs[b]
		if !p.registered || len(p.completed) < p.expected {
			return false
		}
	}
	return true
}

func (j *agxJoin) prereq(b int32) *agxPrereq {
	p, ok := j.prereqs[b]
	if !ok {
		p = &agxPrereq{completed: map[*agxStream]bool{}}
		j.prereqs[b] = p
		j.order = append(j.order, b)
	}
	return p
}

func msToUs(ms float64) int64 { return int64(math.Round(ms * 1000)) }

// NewAgentXReplay builds the lanes (TrajectorySource) over the corpus root pool.
func NewAgentXReplay(c *AgentXCorpus, cfg AgentXReplayConfig) (*AgentXReplay, error) {
	if cfg.Concurrency < 1 {
		return nil, fmt.Errorf("concurrency must be at least 1, got %d", cfg.Concurrency)
	}
	if cfg.DurationUs <= 0 || cfg.GraceUs < 0 || cfg.IdleCapUs < 0 {
		return nil, fmt.Errorf("duration must be positive and grace and idle cap non-negative")
	}
	if len(c.Traces) == 0 {
		return nil, fmt.Errorf("empty trace pool")
	}
	x := &AgentXReplay{
		cfg:           cfg,
		sampler:       &agxSampler{pool: c.Traces},
		byPlaceholder: map[*sim.Request]*agxDispatch{},
		idleSinceUs:   -1,
		completeUs:    -1,
	}
	x.lanes = agxBuildLanes(x.sampler, cfg.Concurrency, cfg.Seed, cfg.MinRatio, cfg.MaxRatio)
	if len(x.lanes) == 0 {
		return nil, fmt.Errorf("no spawnable trajectory in the pool")
	}
	x.stats.Lanes = len(x.lanes)
	x.laneInst = make([]*agxInstance, len(x.lanes))
	for i, l := range x.lanes {
		x.laneInst[i] = x.newInstance(i, l.trace)
	}
	return x, nil
}

func (x *AgentXReplay) newInstance(lane int, tr *AgentXTrace) *agxInstance {
	in := &agxInstance{
		id: len(x.instances), lane: lane, trace: tr,
		salt:        agxMix(uint64(x.cfg.Seed), uint64(len(x.instances))+1),
		completed:   map[agxRef]bool{},
		future:      map[int32]*agxJoin{},
		overlapDone: map[int32]bool{},
	}
	in.root = x.newStream(in, 0, false)
	x.instances = append(x.instances, in)
	x.stats.TreesStarted++
	return in
}

func (x *AgentXReplay) newStream(in *agxInstance, conv int32, child bool) *agxStream {
	x.streams++
	return &agxStream{id: x.streams - 1, inst: in, conv: conv, child: child}
}

// Lanes reports how many lanes were built (fewer than C only if the pool had too few
// spawnable traces, as in aiperf).
func (x *AgentXReplay) Lanes() int { return len(x.lanes) }

func (x *AgentXReplay) flush() []*sim.Request {
	o := x.out
	x.out = nil
	return o
}

// push schedules d's send at due by pushing a fresh placeholder request; any earlier
// placeholder of d is superseded (Gate refuses it).
func (x *AgentXReplay) push(d *agxDispatch, due int64) {
	if d.ph != nil {
		delete(x.byPlaceholder, d.ph)
	}
	d.dueUs = due
	d.ph = &sim.Request{ID: "agentx-pending", ArrivalTime: due, State: sim.StateQueued}
	x.byPlaceholder[d.ph] = d
	x.out = append(x.out, d.ph)
}

// schedule sets a timer (LoopScheduler.schedule_later when shiftable, an asyncio sleep for
// a delayed child spawn otherwise).
func (x *AgentXReplay) schedule(d *agxDispatch, due int64, shiftable bool) {
	d.state = agxTimer
	d.shiftable = shiftable
	if shiftable {
		x.timers = append(x.timers, d)
	}
	x.push(d, due)
}

// Start is the warm-up's initial dispatch at time 0 (agentic_replay._execute_warmup). It
// returns the requests to inject, in non-decreasing arrival order.
func (x *AgentXReplay) Start() []*sim.Request {
	x.nowUs = 0
	type prep struct {
		d    *agxDispatch
		lead float64
	}
	var preps []prep
	maxLead := math.Inf(-1)
	for li, l := range x.lanes {
		in := x.laneInst[li]
		for _, st := range l.states {
			if st.next < 1 {
				continue
			}
			s := x.streamForState(in, st)
			lead := l.tStarMs - l.trace.convs[st.conv].turns[st.next-1].tsMs
			preps = append(preps, prep{&agxDispatch{inst: in, stream: s, turn: st.next - 1, warmup: true, start: true}, lead})
			maxLead = math.Max(maxLead, lead)
		}
	}
	x.warmupLeft = len(preps)
	x.stats.WarmupRequests = len(preps)
	if len(preps) == 0 {
		x.startProfiling(0)
	} else {
		for _, p := range preps {
			off := (maxLead - p.lead) / 1000.0
			if off > 0 {
				x.schedule(p.d, int64(math.Round(off*1e6)), true)
			} else {
				x.submit(p.d)
			}
		}
	}
	x.enforceIdleCap(false)
	out := x.flush()
	sort.SliceStable(out, func(i, j int) bool { return out[i].ArrivalTime < out[j].ArrivalTime })
	return out
}

// streamForState returns the lane tree's stream for a snapshot state, creating children.
func (x *AgentXReplay) streamForState(in *agxInstance, st agxState) *agxStream {
	if st.conv == 0 {
		return in.root
	}
	for _, s := range in.childStreams {
		if s.conv == st.conv {
			return s
		}
	}
	s := x.newStream(in, st.conv, true)
	in.childStreams = append(in.childStreams, s)
	return s
}

// startProfiling is the PROFILING phase setup and initial dispatch at now
// (agentic_replay.setup_phase, _dispatch_snapshot_for_profiling; branch_orchestrator
// seed_snapshot).
func (x *AgentXReplay) startProfiling(now int64) {
	x.phase = agxProfile
	x.profileUs = now
	x.windowEndUs = now + x.cfg.DurationUs
	if x.cfg.GraceUs == math.MaxInt64 {
		x.graceEndUs = math.MaxInt64
	} else {
		x.graceEndUs = x.windowEndUs + x.cfg.GraceUs
	}
	if x.SetHorizon != nil {
		x.SetHorizon(x.graceEndUs)
	}
	// The end of the sending window is a timer the idle cap never moves: it is the
	// phase's duration timeout, not a scheduler timer.
	x.push(&agxDispatch{state: agxTimer, windowEnd: true}, x.windowEndUs)

	for li, l := range x.lanes {
		in := x.laneInst[li]
		in.barrierOn = true
		for c, n := range l.completed {
			for k := int32(0); k < n; k++ {
				in.completed[agxRef{c, k}] = true
			}
		}
		hasRoot := false
		var root agxState
		for _, st := range l.states {
			if st.conv == 0 {
				hasRoot, root = true, st
			}
		}
		// seed_snapshot: every live child is tracked against the root; its gate memberships
		// seed joins (a rootless lane's children are tracked with no gate).
		for _, st := range l.states {
			if st.conv == 0 {
				continue
			}
			s := x.streamForState(in, st)
			s.tracked = true
			if hasRoot && st.joinTurn >= 0 && st.branch >= 0 {
				j := x.seededJoin(in, root, st.joinTurn)
				p := j.prereq(st.branch)
				p.expected++
				p.registered = true
				s.gates = []agxGateRef{{st.joinTurn, st.branch}}
			}
			in.descendants++
			in.outstanding++
		}
		hasDispatchableRoot := hasRoot && !root.waiting
		anyDispatchable := false
		for _, st := range l.states {
			if !st.waiting {
				anyDispatchable = true
			}
		}
		if !hasDispatchableRoot && anyDispatchable {
			// acquire_lane_credit: the lane holds its slot without a root credit.
			in.open = true
			in.rootPending = hasRoot
		}
		for _, st := range l.states {
			if st.waiting {
				continue
			}
			s := x.streamForState(in, st)
			d := &agxDispatch{inst: in, stream: s, turn: st.next, start: st.conv == 0}
			if delay := st.offsetMs / 1000.0; delay > 0 {
				x.schedule(d, now+int64(math.Round(delay*1e6)), true)
			} else {
				x.submit(d)
			}
		}
	}
}

// seededJoin is BranchOrchestrator._ensure_seeded_join for the lane's root.
func (x *AgentXReplay) seededJoin(in *agxInstance, root agxState, gated int32) *agxJoin {
	if in.active != nil && in.active.gated == gated {
		return in.active
	}
	if j, ok := in.future[gated]; ok {
		return j
	}
	j := &agxJoin{gated: gated, prereqs: map[int32]*agxPrereq{}}
	tr := in.trace
	for _, b := range tr.convs[0].turns[gated].gates {
		p := j.prereq(b)
		if tr.branches[b].spawnTurn < root.next {
			p.registered = true // spawned before t*: its live children re-register below
		}
	}
	if root.waiting && root.joinTurn == gated {
		j.blocked = true
		in.active = j
		x.stats.JoinSuspensions++
	} else {
		in.future[gated] = j
	}
	return j
}

// canSend is the profiling stop checks: the duration window (DurationStopCondition) for
// every send, and recycle/new sessions likewise.
func (x *AgentXReplay) canSend() bool {
	return x.phase == agxWarmup || (x.phase == agxProfile && x.nowUs < x.windowEndUs)
}

// submit is ReplayIssueGate.submit + CreditIssuer._issue_credit_ready /
// _dispatch_child_turn_ready: hold behind the barrier, refuse after the window, else send.
func (x *AgentXReplay) submit(d *agxDispatch) {
	in := d.inst
	if x.phase != agxWarmup && in.barrierOn && !x.barrierReady(in, d) {
		d.state = agxHeld
		in.held = append(in.held, d)
		x.stats.BarrierHeld++
		return
	}
	x.issue(d)
}

func (x *AgentXReplay) barrierReady(in *agxInstance, d *agxDispatch) bool {
	for _, p := range in.trace.convs[d.stream.conv].turns[d.turn].preds {
		if !in.completed[p] {
			return false
		}
	}
	return true
}

// issue sends d now, or refuses it when the window has closed.
func (x *AgentXReplay) issue(d *agxDispatch) {
	if !x.canSend() {
		d.state = agxDropped
		x.refused(d)
		return
	}
	in := d.inst
	if d.start && d.stream == in.root && !in.open && x.phase != agxWarmup {
		in.open = true // acquire_session_slot + open_tree(root_pending=True)
		in.rootPending = true
	}
	d.state = agxReady
	x.push(d, x.nowUs)
	x.inFlight++
	if d.warmup {
		return
	}
	if d.stream.child {
		x.stats.SubagentRequests++
	} else {
		x.stats.MainRequests++
		x.overlapSpawn(d)
	}
}

// refused handles a send refused after the window closed: a spawned child's first turn
// rolls back (_rollback_failed_first_turn); a child continuation drains its join
// (on_child_stopped). Root sends are simply dropped.
func (x *AgentXReplay) refused(d *agxDispatch) {
	if d.stream == nil || !d.stream.child {
		return
	}
	if d.first {
		x.stats.ChildrenRefused++
		x.rollbackChild(d.stream)
		return
	}
	if d.childPath {
		x.childDone(d.stream)
	}
}

// Gate is the cluster ArrivalGate: it confirms a send that came due, fires a timer that
// came due, and refuses every superseded placeholder.
func (x *AgentXReplay) Gate(req *sim.Request, now int64) (bool, []*sim.Request) {
	d, ok := x.byPlaceholder[req]
	if !ok || now != d.dueUs {
		return false, x.flush()
	}
	x.nowUs = now
	switch d.state {
	case agxReady:
		delete(x.byPlaceholder, req)
		d.state = agxIssued
		x.fillRequest(d, req, now)
		x.byPlaceholder[req] = d // keyed by the live request until it completes
		return true, x.flush()
	case agxTimer:
		delete(x.byPlaceholder, req)
		d.ph = nil
		x.removeTimer(d)
		x.fire(d)
		if d.shiftable {
			// LoopScheduler's drain observer re-checks the idle cap after a scheduler task.
			x.enforceIdleCap(false)
		}
		x.issueReleased()
		x.maybeComplete()
		return false, x.flush()
	}
	return false, x.flush()
}

func (x *AgentXReplay) removeTimer(d *agxDispatch) {
	for i, t := range x.timers {
		if t == d {
			x.timers = append(x.timers[:i], x.timers[i+1:]...)
			return
		}
	}
}

// fire runs a timer that came due.
func (x *AgentXReplay) fire(d *agxDispatch) {
	switch {
	case d.windowEnd:
		x.closeWindow()
	case d.recycle:
		x.dispatchRecycle(d.lane)
	default:
		if x.phase != agxWarmup && !x.canSend() && d.shiftable {
			// LoopScheduler.cancel_all_pending ran at the window end: this timer never fires.
			d.state = agxDropped
			return
		}
		x.submit(d)
	}
}

// closeWindow is the sending-window timeout (phase/runner._wait_for_sending_complete):
// pending scheduler timers are cancelled, and barrier-held sends are cancelled with
// child refusal cleanup (cancel_pending(notify_refused=True)).
func (x *AgentXReplay) closeWindow() {
	x.phase = agxClosed
	for _, t := range x.timers {
		t.state = agxDropped
		if t.ph != nil {
			delete(x.byPlaceholder, t.ph)
		}
	}
	x.timers = nil
	for _, in := range x.instances {
		held := in.held
		in.held = nil
		for _, d := range held {
			d.state = agxDropped
			if d.stream != nil && d.stream.child && d.childPath {
				x.childDone(d.stream)
			}
		}
	}
}

// dispatchRecycle is _dispatch_recycled_on_lane: the lane's next root at turn 0, a fresh
// tree with a fresh cache-bust marker.
func (x *AgentXReplay) dispatchRecycle(lane int) {
	if !x.canSend() {
		return
	}
	tr := x.sampler.nextRecycle()
	if tr == nil {
		return
	}
	in := x.newInstance(lane, tr)
	in.barrierOn = true
	x.laneInst[lane] = in
	x.submit(&agxDispatch{inst: in, stream: in.root, turn: 0, start: true})
}

// overlapSpawn is BranchOrchestrator.on_credit_issued: a SPAWN branch that started
// before its spawning request's recorded end is spawned as the request goes on the wire,
// each child at its recorded offset from the request's recorded start.
func (x *AgentXReplay) overlapSpawn(d *agxDispatch) {
	in := d.inst
	tn := &in.trace.convs[0].turns[d.turn]
	if tn.apiMs <= 0 || len(tn.spawns) == 0 {
		return
	}
	end := tn.tsMs + tn.apiMs
	var bs []int32
	for _, b := range tn.spawns {
		if in.trace.branches[b].startMs < end {
			bs = append(bs, b)
		}
	}
	if len(bs) == 0 {
		return
	}
	x.spawn(in, bs, tn.tsMs, true)
	for _, b := range bs {
		in.overlapDone[b] = true
	}
}

// spawn is _spawn_children_and_register_gates for the root of in.
func (x *AgentXReplay) spawn(in *agxInstance, branches []int32, originMs float64, overlap bool) {
	tr := in.trace
	type kid struct {
		s      *agxStream
		offset float64
	}
	var kids []kid
	for _, b := range branches {
		if !overlap && in.overlapDone[b] {
			continue
		}
		br := &tr.branches[b]
		origin := br.startMs
		if overlap {
			origin = originMs
		}
		for _, c := range br.children {
			s := x.newStream(in, c, true)
			s.tracked = true
			if !br.background && br.joinTurn >= 0 {
				s.gates = []agxGateRef{{br.joinTurn, b}}
				j := x.futureJoin(in, br.joinTurn)
				p := j.prereq(b)
				p.expected++
				p.registered = true
			}
			in.childStreams = append(in.childStreams, s)
			kids = append(kids, kid{s, math.Max(0, tr.convs[c].turns[0].tsMs-origin)})
			x.stats.ChildrenSpawned++
		}
	}
	if len(kids) == 0 {
		return
	}
	in.descendants += len(kids)
	in.outstanding += len(kids)
	for _, k := range kids {
		d := &agxDispatch{inst: in, stream: k.s, turn: 0, childPath: true, first: true}
		if k.offset > 0 {
			x.stats.DelayedChildSpawn++
			x.schedule(d, x.nowUs+msToUs(k.offset), false)
		}
	}
	for _, k := range kids {
		if k.offset <= 0 {
			x.submit(&agxDispatch{inst: in, stream: k.s, turn: 0, childPath: true, first: true})
		}
	}
}

// futureJoin is _ensure_future_join.
func (x *AgentXReplay) futureJoin(in *agxInstance, gated int32) *agxJoin {
	if j, ok := in.future[gated]; ok {
		return j
	}
	j := &agxJoin{gated: gated, prereqs: map[int32]*agxPrereq{}}
	for _, b := range in.trace.convs[0].turns[gated].gates {
		j.prereq(b)
	}
	in.future[gated] = j
	return j
}

// OnComplete is the credit-return path (credit/callback_handler.on_credit_return) for a
// request the cluster finished: completed, timed out, or dropped by the engine.
func (x *AgentXReplay) OnComplete(req *sim.Request, now int64) []*sim.Request {
	d, ok := x.byPlaceholder[req]
	if !ok || d.state != agxIssued {
		return nil
	}
	delete(x.byPlaceholder, req)
	d.state = agxDropped
	x.nowUs = now
	x.inFlight--
	if x.inFlight == 0 {
		x.idleSinceUs = now
	}
	overflow := req.State == sim.StateQueued
	errored := req.State == sim.StateTimedOut
	switch {
	case overflow:
		req.Agentic.Outcome = sim.AgenticContextOverflow
		x.stats.ContextOverflows++
	case errored:
		req.Agentic.Outcome = sim.AgenticError
		x.stats.Errors++
	default:
		req.Agentic.Outcome = sim.AgenticCompleted
	}
	if x.phase != agxWarmup && now > x.windowEndUs {
		req.Phase = sim.PhaseGrace
	}

	if d.warmup {
		x.warmupLeft--
		if x.warmupLeft == 0 {
			x.startProfiling(now)
		}
		x.enforceIdleCap(true)
		return x.flush()
	}
	x.onReturn(d, overflow)
	x.enforceIdleCap(true)
	x.issueReleased()
	x.maybeComplete()
	return x.flush()
}

// onReturn is steps 4b-5 of on_credit_return for a profiling credit.
func (x *AgentXReplay) onReturn(d *agxDispatch, overflow bool) {
	in, s := d.inst, d.stream
	convTurns := in.trace.convs[s.conv].turns
	final := int(d.turn) == len(convTurns)-1
	// 4b: a child's final turn reports the child done (errors count as done).
	if final && s.child {
		x.childDone(s)
	}
	// observe_credit_return: the barrier records the completion and releases ready sends.
	in.completed[agxRef{s.conv, d.turn}] = true
	x.releaseHeld(in)

	overflowTerminal := overflow && !final
	if !overflowTerminal {
		// intercept: spawn the branches declared on this turn (root only), then suspend
		// the parent if its next turn is gated on unfinished children.
		if !s.child {
			if spawns := convTurns[d.turn].spawns; len(spawns) > 0 {
				x.spawn(in, spawns, 0, false)
			}
			if x.maybeSuspend(in, d.turn+1) {
				return
			}
		}
	}
	if (final || overflowTerminal) && !s.child {
		in.rootPending = false // on_root_terminal
		x.maybeDrain(in)
	}
	if !s.child {
		if !x.canSend() {
			return // strategy dispatch is gated behind can_send_any_turn
		}
		if !final && !overflowTerminal {
			x.dispatchNext(d)
		}
		return
	}
	switch {
	case overflowTerminal:
		x.childDone(s) // on_child_stopped
	case final:
	case !x.canSend():
		x.childDone(s) // on_child_stopped: the continuation will never be sent
	default:
		x.dispatchNext(d)
	}
}

// dispatchNext is _dispatch_next_turn: the stream's next turn after its recorded
// end-to-start delay, from this completion.
func (x *AgentXReplay) dispatchNext(d *agxDispatch) {
	s := d.stream
	tn := &d.inst.trace.convs[s.conv].turns[d.turn+1]
	nd := &agxDispatch{inst: d.inst, stream: s, turn: d.turn + 1, childPath: s.child}
	if tn.delayMs > 0 {
		x.schedule(nd, x.nowUs+msToUs(tn.delayMs), true)
		return
	}
	x.submit(nd)
}

// maybeSuspend is _maybe_suspend_parent for the root's next turn.
func (x *AgentXReplay) maybeSuspend(in *agxInstance, next int32) bool {
	if a := in.active; a != nil && a.gated == next && !a.satisfied() {
		return true
	}
	j, ok := in.future[next]
	if !ok {
		return false
	}
	delete(in.future, next)
	if j.satisfied() {
		return false
	}
	j.blocked = true
	in.active = j
	x.stats.JoinSuspensions++
	return true
}

// childDone is _handle_child_done: satisfy the child's gates (dispatching a blocked
// parent's join turn at once, without its delay), and count it out of its tree.
func (x *AgentXReplay) childDone(s *agxStream) {
	if !s.tracked || s.done {
		return
	}
	s.done = true
	s.tracked = false
	in := s.inst
	for _, g := range s.gates {
		var j *agxJoin
		if in.active != nil && in.active.gated == g.gated {
			j = in.active
		} else {
			j = in.future[g.gated]
		}
		if j == nil {
			continue
		}
		p, ok := j.prereqs[g.branch]
		if !ok || p.completed[s] {
			continue
		}
		p.completed[s] = true
		if !j.satisfied() {
			continue
		}
		if j.blocked {
			in.active = nil
			// dispatch_join_turn -> issue_credit for the gated turn.
			x.submit(&agxDispatch{inst: in, stream: in.root, turn: j.gated})
		} else {
			delete(in.future, g.gated)
		}
	}
	in.descendants--
	in.outstanding--
	x.maybeDrain(in)
}

// rollbackChild is _rollback_failed_first_turn + _finalize_failed_dispatches for a child
// whose first turn was refused.
func (x *AgentXReplay) rollbackChild(s *agxStream) {
	if !s.tracked {
		return
	}
	s.tracked = false
	s.done = true
	in := s.inst
	for _, g := range s.gates {
		var j *agxJoin
		if in.active != nil && in.active.gated == g.gated {
			j = in.active
		} else {
			j = in.future[g.gated]
		}
		if j == nil {
			continue
		}
		if p, ok := j.prereqs[g.branch]; ok && p.expected > 0 {
			p.expected = max(len(p.completed), p.expected-1)
		}
	}
	in.descendants--
	in.outstanding--
	for g, j := range in.future {
		if j.satisfied() {
			delete(in.future, g)
		}
	}
	if in.active != nil && in.active.satisfied() {
		j := in.active
		in.active = nil
		x.submit(&agxDispatch{inst: in, stream: in.root, turn: j.gated})
	}
	x.maybeDrain(in)
}

// maybeDrain is SessionTreeRegistry._maybe_release: a drained tree frees its lane, which
// recycles into the next root (agentic_replay._on_tree_drained, via schedule_later(0)).
func (x *AgentXReplay) maybeDrain(in *agxInstance) {
	if !in.open || in.drained || in.rootPending || in.outstanding > 0 {
		return
	}
	in.drained = true
	in.barrierOn = false
	x.stats.TreesCompleted++
	if x.laneInst[in.lane] == in {
		x.schedule(&agxDispatch{recycle: true, lane: in.lane}, x.nowUs, true)
	}
}

// releaseHeld is ReplayBarrierCoordinator.complete's release of now-ready sends, in key
// order.
func (x *AgentXReplay) releaseHeld(in *agxInstance) {
	if len(in.held) == 0 {
		return
	}
	var ready, keep []*agxDispatch
	for _, d := range in.held {
		if x.barrierReady(in, d) {
			ready = append(ready, d)
		} else {
			keep = append(keep, d)
		}
	}
	in.held = keep
	sort.SliceStable(ready, func(i, j int) bool {
		a, b := in.trace.convs[ready[i].stream.conv].sessionID, in.trace.convs[ready[j].stream.conv].sessionID
		if a != b {
			return a < b
		}
		return ready[i].turn < ready[j].turn
	})
	x.released = append(x.released, ready...)
}

// issueReleased sends the barrier-released dispatches.
func (x *AgentXReplay) issueReleased() {
	for len(x.released) > 0 {
		d := x.released[0]
		x.released = x.released[1:]
		x.issue(d)
	}
}

// enforceIdleCap is AgenticReplayStrategy.enforce_system_idle_cap with
// LoopScheduler.cap_pending_delay: when nothing is in flight, every pending scheduler
// timer moves earlier by the same amount so the system is never idle longer than the cap.
// afterReturn is true on a credit return (which starts a new idle interval).
func (x *AgentXReplay) enforceIdleCap(afterReturn bool) {
	if x.cfg.IdleCapUs <= 0 || x.phase >= agxClosed {
		return
	}
	if x.inFlight > 0 {
		x.idleSinceUs = -1
		return
	}
	now := x.nowUs
	if afterReturn || x.idleSinceUs < 0 {
		x.idleSinceUs = now
	}
	if len(x.timers) == 0 {
		return
	}
	budget := max(0, x.cfg.IdleCapUs-(now-x.idleSinceUs))
	earliest := int64(math.MaxInt64)
	for _, t := range x.timers {
		earliest = min(earliest, t.dueUs)
	}
	shift := earliest - now - budget
	if shift <= 0 {
		return
	}
	x.stats.IdleCapJumps++
	x.stats.IdleCapSkippedUs += shift
	for _, t := range x.timers {
		x.push(t, max(now, t.dueUs-shift))
	}
}

// maybeComplete records the profiling phase completion once the window has closed,
// nothing is in flight and no branch work is pending (PhaseRunner
// _wait_for_returning_complete with BranchOrchestrator.has_pending_branch_work); otherwise
// the phase completes at the grace end.
func (x *AgentXReplay) maybeComplete() {
	if x.phase != agxClosed || x.inFlight > 0 || x.completeUs >= 0 {
		return
	}
	for _, in := range x.instances {
		if in.active != nil || len(in.future) > 0 || in.descendants > 0 {
			return
		}
		for _, s := range in.childStreams {
			if s.tracked {
				return
			}
		}
	}
	x.completeUs = x.nowUs
	x.phase = agxDone
	if x.SetHorizon != nil {
		x.SetHorizon(x.nowUs)
	}
}

// Finish closes the replay after the simulation stopped at stopUs: requests still in
// flight were cancelled at the grace end and leave no aiperf record.
func (x *AgentXReplay) Finish(stopUs int64) {
	for req, d := range x.byPlaceholder {
		if d.state == agxIssued && req.Agentic != nil && req.Agentic.Outcome == "" {
			req.Agentic.Outcome = sim.AgenticCancelled
			x.stats.Cancelled++
		}
	}
	if x.completeUs < 0 && x.phase >= agxProfile {
		// Branch work still pending: aiperf waits out the grace period even when nothing
		// is in flight, and completes at its end.
		x.completeUs = x.graceEndUs
		if x.completeUs == math.MaxInt64 {
			x.completeUs = max(stopUs, x.windowEndUs)
		}
	}
}

// Window returns the profiling phase's start and completion (aiperf lifecycle
// started_at and complete_at), the measured window's bounds.
func (x *AgentXReplay) Window() (startUs, endUs int64) { return x.profileUs, x.completeUs }

// Stats returns the replay's counts.
func (x *AgentXReplay) Stats() AgentXStats { return x.stats }

// fillRequest turns a confirmed send's placeholder into the request the cluster serves.
func (x *AgentXReplay) fillRequest(d *agxDispatch, req *sim.Request, now int64) {
	in, s := d.inst, d.stream
	tr := in.trace
	tn := &tr.convs[s.conv].turns[d.turn]
	x.nextReqID++
	out := int(tn.maxTokens)
	if d.warmup {
		out = 1
	}
	req.ID = fmt.Sprintf("agentx_%d", x.nextReqID)
	req.ArrivalTime = now
	req.InputTokens = x.promptTokens(in, s.conv, d.turn, tn)
	req.OutputTokens = agxOutputTokens(out)
	req.MaxOutputLen = out
	req.State = sim.StateQueued
	req.Streaming = true
	req.SessionID = fmt.Sprintf("%s#%d", tr.convs[s.conv].sessionID, in.id)
	req.RoundIndex = int(d.turn)
	req.Phase = sim.PhaseProfile
	if d.warmup {
		req.Phase = sim.PhaseWarmup
	}
	if x.cfg.RequestTimeoutUs > 0 {
		req.Deadline = now + x.cfg.RequestTimeoutUs
	}
	kind := "root"
	if s.child {
		kind = "flat"
		if tr.convs[s.conv].scope != 0 {
			kind = "subagent"
		}
	}
	req.Agentic = &sim.AgenticTag{
		Lane: in.lane, Tree: in.id, TraceID: tr.ID, Stream: tr.convs[s.conv].sessionID,
		Kind: kind, StreamInstance: s.id, Turn: int(d.turn),
	}
}

// promptTokens lays out a turn's prompt as aiperf's ConversationReconstructor does
// (dataset/loader/weka_synth_buf.py compute_turn_block_geometry): exactly `in` tokens,
// the first min(len(hash_ids), in/64) blocks from the trace's hash ids, the rest a tail
// unique to this request. A block's content is a function of the tree instance and the
// hash id only, so identical hash prefixes within one tree share KV blocks (parent with
// subagents, turn k with k+1) and nothing is shared across trees (the per-tree cache-bust
// marker). The model is the chat template's identity map: template tokens are not added.
func (x *AgentXReplay) promptTokens(in *agxInstance, conv, turn int32, tn *agxTurn) []sim.TokenID {
	n := int(tn.in)
	toks := make([]sim.TokenID, n)
	covered := min(int(tn.nHash), n/agxBlockSize)
	hashes := in.trace.hashPrefix(tn.hashNode, tn.nHash, covered, nil)
	for b, h := range hashes {
		base := agxMix(in.salt, uint64(h))
		for i := 0; i < agxBlockSize; i++ {
			toks[b*agxBlockSize+i] = agxToken(base, uint64(i))
		}
	}
	// The tail (partial last block plus any blocks the recording did not hash) is unique
	// to this request: seeded by the request number, which no other request shares.
	tail := agxMix(agxMix(in.salt^0x7461696c, uint64(x.nextReqID)), uint64(conv)<<32|uint64(turn))
	for i := covered * agxBlockSize; i < n; i++ {
		toks[i] = agxToken(tail, uint64(i))
	}
	return toks
}

func agxOutputTokens(n int) []sim.TokenID {
	t := make([]sim.TokenID, n)
	for i := range t {
		t[i] = sim.TokenID(i + 1)
	}
	return t
}

// agxMix is splitmix64 over a combination of a and b.
func agxMix(a, b uint64) uint64 {
	z := a ^ (b + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2))
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func agxToken(base, i uint64) sim.TokenID {
	return sim.TokenID(agxMix(base, i) >> 33)
}

// AgentXRequestTimeoutUs is aiperf's --request-timeout-seconds default, 6 h
// (config/endpoint.py:52, EndpointDefaults.TIMEOUT): a request still unfinished after it
// is an error record.
const AgentXRequestTimeoutUs = int64(6 * 3600 * 1_000_000)

// AgentXSummary is the results file's agentic block.
type AgentXSummary struct {
	Scenario                   string  `json:"scenario"`
	Lanes                      int     `json:"lanes"`
	TreesStarted               int     `json:"trees_started"`
	TreesCompleted             int     `json:"trees_completed"`
	MainRequests               int     `json:"main_requests"`
	SubagentRequests           int     `json:"subagent_requests"`
	WarmupRequests             int     `json:"warmup_requests"`
	CancelledRequests          int     `json:"cancelled_requests"`
	ErrorRequests              int     `json:"error_requests"`
	ContextOverflowRequests    int     `json:"context_overflow_requests"`
	TracesInCorpus             int     `json:"traces_in_corpus"`
	TracesFilteredByMaxContext int     `json:"traces_filtered_by_max_context"`
	MaxContextLength           int     `json:"max_context_length"`
	CorpusSHA256               string  `json:"corpus_sha256"`
	CorpusSHA256Verified       bool    `json:"corpus_sha256_verified"`
	ProfileStartUs             int64   `json:"profile_start_us"`
	WindowEndUs                int64   `json:"window_end_us"`
	ProfileCompleteUs          int64   `json:"profile_complete_us"`
	IdleCapJumps               int     `json:"idle_cap_jumps"`
	IdleCapSkippedS            float64 `json:"idle_cap_skipped_s"`
	BarrierHeld                int     `json:"barrier_holds"`
	JoinSuspensions            int     `json:"join_suspensions"`
	ChildrenSpawned            int     `json:"children_spawned"`
}

// Summary assembles the agentic block once the run has finished.
func (x *AgentXReplay) Summary(c *AgentXCorpus, maxContextLength int, verified bool) AgentXSummary {
	s := x.stats
	return AgentXSummary{
		Scenario: "inferencex-agentx-mvp", Lanes: s.Lanes, TreesStarted: s.TreesStarted, TreesCompleted: s.TreesCompleted,
		MainRequests: s.MainRequests, SubagentRequests: s.SubagentRequests, WarmupRequests: s.WarmupRequests,
		CancelledRequests: s.Cancelled, ErrorRequests: s.Errors, ContextOverflowRequests: s.ContextOverflows,
		TracesInCorpus: c.Rows, TracesFilteredByMaxContext: c.FilteredByMaxCtx, MaxContextLength: maxContextLength,
		CorpusSHA256: c.SHA256, CorpusSHA256Verified: verified,
		ProfileStartUs: x.profileUs, WindowEndUs: x.windowEndUs, ProfileCompleteUs: x.completeUs,
		IdleCapJumps: s.IdleCapJumps, IdleCapSkippedS: float64(s.IdleCapSkippedUs) / 1e6,
		BarrierHeld: s.BarrierHeld, JoinSuspensions: s.JoinSuspensions, ChildrenSpawned: s.ChildrenSpawned,
	}
}

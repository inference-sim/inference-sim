package workload

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
)

// AgentXReplay drives aiperf 0.12.0's --scenario inferencex-agentx-mvp against the
// simulated cluster: C session-tree lanes snapshotted mid-trace, a warm-up that primes
// every live stream (optionally followed by an accelerated cache-pressure substage that
// replays the live trajectories under a per-lane request quota before the measured window),
// then a closed-loop profiling window in which subagents fan out beside their parent, joins
// gate later parent turns, recorded replay barriers order streams, a 10 s global idle cap
// pulls in timers (and, when a run states one, a per-trace idle cap pulls in a single tree's
// timers while others stay busy), and drained trees recycle into the next root. It is a
// closed-loop driver with the same contract as SessionManager and
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

	byPlaceholder map[*sim.Request]*agxDispatch // pending sends, by the request pushed for them
	byID          map[string]*agxDispatch       // sends in flight, by request ID
	timers        []*agxDispatch                // scheduler timers (LoopScheduler): shiftable by the idle cap
	inFlight      int
	idleSinceUs   int64 // -1: not idle
	warmupLeft    int
	// accelOn is the accelerated cache-pressure warmup substage (aiperf
	// _start_accelerated_warmup): once the baseline primers return, the live trajectories are
	// replayed through the real closed loop with zero idle delay and max_tokens 1, gated by a
	// per-lane wire-request quota. accelBudgetReached latches once every lane has admitted its
	// quota, after which no further accelerated turn is admitted and the substage drains.
	accelOn            bool
	accelBudgetReached bool
	// accelAdmitted counts each lane's admitted accelerated cache-pressure wire requests,
	// keyed by lane index so it survives a recycle mid-substage (aiperf
	// _cache_warmup_requests_by_lane, keyed by the stable trajectory lane). len == lanes.
	accelAdmitted []int
	// released are barrier-released sends not yet on the wire: aiperf issues them from a
	// task (ReplayBarrierCoordinator._dispatch_pending), after the return that released
	// them has finished, including its idle-cap check.
	released []*agxDispatch

	instances []*agxInstance
	laneInst  []*agxInstance // the tree currently holding each lane's slot
	nextReqID int64
	streams   int
	horizonUs int64 // the horizon last asked for; 0 = none yet
	out       []*sim.Request
	nowUs     int64

	// SetHorizon, when set, is told the simulation horizon the replay needs: the end of the
	// grace period once profiling starts, earlier if the phase completes first.
	SetHorizon func(int64)

	stats AgentXStats
}

// AgentXReplayConfig is the scenario's run choices.
type AgentXReplayConfig struct {
	Concurrency int   // session-tree lanes
	Seed        int64 // --random-seed (lane snapshot seeds; prompt content salt)
	DurationUs  int64 // benchmark duration (sending window)
	GraceUs     int64 // grace period; math.MaxInt64 = .inf
	IdleCapUs   int64 // global system-idle cap
	// TraceIdleCapUs is the per-trace (per-session-tree) idle cap, aiperf's
	// --trace-idle-gap-cap-seconds. 0 = unset (the scenario default): no per-tree watchdog.
	TraceIdleCapUs int64
	// WarmupRequestsPerLane is aiperf's --warmup-requests-per-lane: after the snapshot
	// primers, each lane sends this many extra one-token cache-pressure warmup requests
	// before profiling starts. 0 = synthesized primers only (the scenario default).
	WarmupRequestsPerLane int
	MinRatio              float64 // trajectory start window
	MaxRatio              float64
	// RequestTimeoutUs is the client's per-request timeout: aiperf's
	// --request-timeout-seconds default, 6 h (config/endpoint.py:52). A timed-out request
	// is an error record. 0 disables it.
	RequestTimeoutUs int64
	// MaxModelLen is the deployment's resolved max_model_len, which decides whether an
	// engine refusal is a context overflow; 0 = unknown (refusals are errors).
	MaxModelLen int64
	// HorizonCapUs is a user --horizon; 0 = none. The run never extends past it.
	HorizonCapUs int64
}

// AgentXStats are the replay's own counts, reported in the results file's agentic block.
type AgentXStats struct {
	Lanes int
	// UnspawnableSkipped counts sampled traces skipped while building lanes because no
	// stream was dispatchable at their t* (aiperf skips them too and samples again).
	UnspawnableSkipped int
	TreesStarted       int // tree instances minted: one per lane plus every recycle
	TreesCompleted     int // trees that drained (root and every descendant finished)
	WarmupRequests     int
	MainRequests       int // profiling requests of root streams
	SubagentRequests   int // profiling requests of child streams (subagents and flat chains)
	Cancelled          int // in flight when the grace period ended
	// CancelledRefused counts the cancelled requests the server had already refused, whose
	// error response had not reached the client when the run stopped (a round trip).
	CancelledRefused int
	Completed        int // ended with a response
	Errors           int // timed out, or refused by the cluster or the engine (not overflow)
	WarmupFailures   int // warm-up requests that ended in an error (aiperf aborts such a run)
	ContextOverflows int // refused by the engine's context check
	IdleCapJumps     int
	IdleCapSkippedUs int64
	// TraceIdleCapJumps/SkippedUs are the per-trace watchdog's activity, apart from the
	// global cap's: a tree idled past the per-trace cap and had its own timers advanced.
	TraceIdleCapJumps     int
	TraceIdleCapSkippedUs int64
	BarrierHeld           int // sends held behind a replay barrier
	JoinSuspensions       int // parent turns suspended on a SPAWN_JOIN
	ChildrenSpawned       int
	ChildrenRefused       int // child first turns refused after the window closed
	DelayedChildSpawn     int
}

type agxPhase int

const (
	agxWarmup agxPhase = iota
	agxAccel           // accelerated cache-pressure warmup: live trajectories replayed, quota-gated
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
	// refusal, when set, is the client receiving the error response for that send.
	refusal   *agxDispatch
	outcome   string
	refusing  bool // its error response is on the way
	warmup    bool
	accel     bool // an accelerated cache-pressure warmup turn (zero delay, max_tokens 1, quota-gated)
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
	// originMs is the tree's sampled replay origin on its shared per-trajectory clock (aiperf
	// _replay_origin_ms_by_root): the lane's snapshot t* for the original tree, or the fresh
	// root's own first turn timestamp for a tree recycled during the run. The warmup-to-
	// profiling handoff places each surviving turn at max(0, ts[next] - originMs).
	originMs float64

	open        bool // slot acquired (open_tree)
	rootPending bool
	outstanding int // descendants live or registered
	drained     bool

	// inFlight and idleSinceUs are this tree's share of the global idle accounting, for the
	// per-trace idle cap (replay_dependencies.py's per-root watchdog): sends on the wire for
	// this tree's root and every descendant stream, and when the last of them returned.
	// idleSinceUs is -1 while the tree is busy.
	inFlight    int
	idleSinceUs int64

	// handoffTurns are turns deferred at this lane's quota boundary (aiperf
	// _quota_handoff_turns): the next turn of a stream the accelerated substage stopped short
	// of, which profiling resumes from. Keyed by stream to keep at most one per stream.
	handoffTurns map[*agxStream]*agxDispatch

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
	recon   *agxRecon    // the stream's message list as of its last rendered turn
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
	if cfg.DurationUs <= 0 || cfg.GraceUs < 0 || cfg.IdleCapUs < 0 || cfg.TraceIdleCapUs < 0 {
		return nil, fmt.Errorf("duration must be positive and grace and idle caps non-negative")
	}
	if len(c.Traces) == 0 {
		return nil, fmt.Errorf("empty trace pool")
	}
	x := &AgentXReplay{
		cfg:           cfg,
		sampler:       &agxSampler{pool: c.Traces},
		byPlaceholder: map[*sim.Request]*agxDispatch{},
		byID:          map[string]*agxDispatch{},
		idleSinceUs:   -1,
		completeUs:    -1,
	}
	x.lanes, x.stats.UnspawnableSkipped = agxBuildLanes(x.sampler, cfg.Concurrency, cfg.Seed, cfg.MinRatio, cfg.MaxRatio)
	if len(x.lanes) == 0 {
		return nil, fmt.Errorf("no spawnable trajectory in the pool")
	}
	x.stats.Lanes = len(x.lanes)
	x.laneInst = make([]*agxInstance, len(x.lanes))
	for i, l := range x.lanes {
		in := x.newInstance(i, l.trace)
		in.originMs = l.tStarMs // the lane's sampled snapshot instant
		x.laneInst[i] = in
	}
	return x, nil
}

func (x *AgentXReplay) newInstance(lane int, tr *AgentXTrace) *agxInstance {
	in := &agxInstance{
		id: len(x.instances), lane: lane, trace: tr,
		salt:         agxMix(uint64(x.cfg.Seed), uint64(len(x.instances))+1),
		completed:    map[agxRef]bool{},
		future:       map[int32]*agxJoin{},
		overlapDone:  map[int32]bool{},
		handoffTurns: map[*agxStream]*agxDispatch{},
		idleSinceUs:  -1,
	}
	in.root = x.newStream(in, 0, false)
	x.instances = append(x.instances, in)
	x.stats.TreesStarted++
	return in
}

// setHorizon tells the cluster where the run must end, never past a user --horizon. A
// value set before the cluster is bound (profiling can start inside Start) is kept and
// applied by BindHorizon.
func (x *AgentXReplay) setHorizon(h int64) {
	if x.cfg.HorizonCapUs > 0 {
		h = min(h, x.cfg.HorizonCapUs)
	}
	x.horizonUs = h
	if x.SetHorizon != nil {
		x.SetHorizon(h)
	}
}

// BindHorizon connects the replay to the cluster's horizon setter, applying any horizon the
// replay already asked for.
func (x *AgentXReplay) BindHorizon(set func(int64)) {
	x.SetHorizon = set
	if x.horizonUs > 0 && set != nil {
		set(x.horizonUs)
	}
}

// InFlight is the number of sends not yet ended.
func (x *AgentXReplay) InFlight() int { return x.inFlight }

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
		// No request precedes t*, so there is no baseline priming to wait for. With the
		// accelerated cache-pressure warmup enabled, jump straight to it (aiperf
		// _execute_warmup: `if not prepared: _start_accelerated_warmup`); otherwise go to
		// profiling.
		if x.cacheWarmupEnabled() {
			x.startAccel(0)
		} else {
			x.startProfiling(0)
		}
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

// cacheWarmupEnabled reports whether a request-bounded cache-pressure warmup runs after the
// baseline primers (aiperf _cache_warmup_enabled; only the per-lane request count is modelled,
// not the time-bounded duration).
func (x *AgentXReplay) cacheWarmupEnabled() bool { return x.cfg.WarmupRequestsPerLane > 0 }

// startAccel is aiperf _start_accelerated_warmup + _dispatch_accelerated_trajectory: replay
// every lane from its post-snapshot state through the real closed loop with zero idle delay
// and max_tokens 1, gated by the per-lane request quota. It is the profiling dispatch run
// early -- the same seed as startProfiling, but as accelerated warmup turns.
func (x *AgentXReplay) startAccel(now int64) {
	x.phase = agxAccel
	x.accelOn = true
	x.nowUs = now
	x.accelAdmitted = make([]int, len(x.lanes))
	for li, l := range x.lanes {
		in := x.laneInst[li]
		in.barrierOn = true
		for c, n := range l.completed {
			for k := int32(0); k < n; k++ {
				in.completed[agxRef{c, k}] = true
			}
		}
		x.seedSnapshot(in, l)
		for _, st := range l.states {
			if st.waiting {
				continue
			}
			s := x.streamForState(in, st)
			// Zero idle delay: the accelerated substage ignores st.offsetMs. Quota-gated.
			x.dispatchAccel(in, &agxDispatch{inst: in, stream: s, turn: st.next, start: st.conv == 0})
		}
	}
	// If the substage put nothing on the wire and holds nothing behind the barrier (every
	// lane was undispatchable, e.g. all roots waiting on children), there is nothing to drain,
	// so hand straight off to profiling rather than stalling.
	if !x.accelWorkPending() {
		x.startProfiling(now)
	}
}

// accelWorkPending reports whether the accelerated substage still has work on the wire. The
// phase completes once wire in_flight reaches zero (aiperf drains requests then hands off);
// paused barrier/DAG work is NOT drained here -- it becomes the profiling handoff payload, so
// a held turn whose prerequisite was deferred at the quota cannot deadlock the substage.
func (x *AgentXReplay) accelWorkPending() bool {
	return x.inFlight > 0 || len(x.released) > 0
}

// dispatchAccel submits one accelerated-warmup turn, enforcing the per-lane quota: a lane that
// has already admitted its quota defers the turn (stashed for the profiling handoff) instead
// of sending it (aiperf _admit_cache_warmup_turn DEFER).
func (x *AgentXReplay) dispatchAccel(in *agxInstance, d *agxDispatch) {
	d.accel = true
	d.warmup = true
	if x.accelAdmitted[in.lane] >= x.cfg.WarmupRequestsPerLane {
		in.handoffTurns[d.stream] = d // DEFER: profiling resumes this stream from here
		x.latchAccelBudget()
		return
	}
	x.accelAdmitted[in.lane]++
	x.stats.WarmupRequests++ // accelerated extras remain warmup requests
	x.latchAccelBudget()
	x.submit(d)
}

// dispatchTurn submits a turn, routing it through the accelerated-warmup quota gate while the
// accel substage is live and as a normal profiling send otherwise. Used by the join-turn
// dispatch sites, which fire a gated root turn the instant its children complete.
func (x *AgentXReplay) dispatchTurn(in *agxInstance, d *agxDispatch) {
	if x.phase == agxAccel {
		x.dispatchAccel(in, d)
		return
	}
	x.submit(d)
}

// latchAccelBudget sets accelBudgetReached once every lane has admitted its quota (aiperf
// _cache_warmup_request_budget_reached): after that, in-flight accel turns drain and profiling
// starts. Iterates the ordered laneInst slice (INV-6).
func (x *AgentXReplay) latchAccelBudget() {
	if x.accelBudgetReached {
		return
	}
	for _, n := range x.accelAdmitted {
		if n < x.cfg.WarmupRequestsPerLane {
			return
		}
	}
	x.accelBudgetReached = true
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
	x.setHorizon(x.graceEndUs)
	// The end of the sending window is a timer the idle cap never moves: it is the
	// phase's duration timeout, not a scheduler timer.
	x.push(&agxDispatch{state: agxTimer, windowEnd: true}, x.windowEndUs)

	if x.accelOn {
		// The accelerated substage already seeded every lane and advanced its streams, so
		// profiling resumes from those advanced positions (the deferred quota turns) rather
		// than re-seeding the snapshot.
		x.accelOn = false
		x.dispatchHandoff(now)
	} else {
		for li, l := range x.lanes {
			in := x.laneInst[li]
			in.barrierOn = true
			for c, n := range l.completed {
				for k := int32(0); k < n; k++ {
					in.completed[agxRef{c, k}] = true
				}
			}
			x.seedSnapshot(in, l)
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
	// observe_idle_root: aiperf arms the per-trace watchdog once every lane's initial timers
	// are installed, so a tree whose first profiling request is scheduled past the cap is
	// pulled in without waiting for a return. A tree with a request dispatched now is in
	// flight and the cap no-ops on it (idleSinceUs stays -1 until that request returns).
	for _, in := range x.instances {
		x.enforceTraceIdleCap(in, false)
	}
}

// seedSnapshot is BranchOrchestrator.seed_snapshot for one lane: track every live child
// against the root, seed its join gate memberships, count descendants, and hold a lane credit
// for a rootless-but-dispatchable lane. Shared by the accelerated substage and profiling.
func (x *AgentXReplay) seedSnapshot(in *agxInstance, l *agxLane) {
	hasRoot := false
	var root agxState
	for _, st := range l.states {
		if st.conv == 0 {
			hasRoot, root = true, st
		}
	}
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
}

// dispatchHandoff resumes profiling from the positions the accelerated substage advanced each
// stream to: both the turns deferred at a lane's quota (aiperf _quota_handoff_turns) and the
// turns paused behind the replay barrier when the substage drained (aiperf's pending DAG/
// barrier payload). Each resumes on its tree's shared replay clock -- offset
// max(0, ts[next] - in.originMs) -- with one phase-wide minimum subtracted so the earliest
// survivor starts at profiling t0 while cross-lane spacing is preserved
// (_profiling_phase_t0_offset_ms). Iterates the ordered instances/childStreams slices (INV-6).
func (x *AgentXReplay) dispatchHandoff(now int64) {
	type pending struct {
		in       *agxInstance
		d        *agxDispatch
		held     bool
		offsetMs float64
	}
	var ps []pending
	minOffset := math.Inf(1)
	for _, in := range x.instances {
		streams := append([]*agxStream{in.root}, in.childStreams...)
		for _, s := range streams {
			d := in.handoffTurns[s]
			if d == nil {
				continue
			}
			delete(in.handoffTurns, s)
			off := agxOffset(in.trace.convs[s.conv].turns[d.turn].tsMs, in.originMs)
			ps = append(ps, pending{in, d, false, off})
			minOffset = math.Min(minOffset, off)
		}
		// Barrier-held turns carry over as paused work: they stay held until their recorded
		// predecessors complete in profiling. Their anchor is their own dispatch offset too.
		for _, d := range in.held {
			off := agxOffset(in.trace.convs[d.stream.conv].turns[d.turn].tsMs, in.originMs)
			ps = append(ps, pending{in, d, true, off})
			minOffset = math.Min(minOffset, off)
		}
		in.held = nil
	}
	if math.IsInf(minOffset, 1) {
		minOffset = 0
	}
	for _, p := range ps {
		d := p.d
		d.accel = false
		d.warmup = false
		d.childPath = d.stream.child
		// submit re-checks the barrier, so a still-blocked held turn returns to in.held and a
		// cleared one (or a deferred quota turn) issues; the anchored offset only delays when
		// the turn's own timestamp sits past the earliest survivor's.
		if delay := (p.offsetMs - minOffset) / 1000.0; delay > 0 {
			x.schedule(d, now+int64(math.Round(delay*1e6)), true)
		} else {
			x.submit(d)
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
// every send, and recycle/new sessions likewise. Warmup (baseline primers and the
// accelerated substage) has no window.
func (x *AgentXReplay) canSend() bool {
	return x.phase == agxWarmup || x.phase == agxAccel || (x.phase == agxProfile && x.nowUs < x.windowEndUs)
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
	in.inFlight++ // this tree's share, for the per-trace watchdog
	// A baseline primer is a single isolated send: no overlap spawn, not counted as a
	// profiling request. An accelerated-warmup turn drives the real loop (so it spawns like a
	// profiling root), but is still warmup and not counted as a profiling request.
	if d.warmup && !d.accel {
		return
	}
	if d.stream.child {
		if !d.accel {
			x.stats.SubagentRequests++
		}
	} else {
		if !d.accel {
			x.stats.MainRequests++
		}
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
	if !ok {
		// A request already in flight that the cluster re-injects (a drained instance's
		// queued request redirected, Request.Redirected) is the same send: let it through
		// with no change of state.
		if d, ok := x.byID[req.ID]; ok && d.ph == req && d.state == agxIssued {
			return true, nil
		}
		return false, x.flush()
	}
	if now != d.dueUs {
		return false, x.flush()
	}
	x.nowUs = now
	switch d.state {
	case agxReady:
		delete(x.byPlaceholder, req)
		d.state = agxIssued
		x.fillRequest(d, req, now)
		x.byID[req.ID] = d
		return true, x.flush()
	case agxTimer:
		delete(x.byPlaceholder, req)
		d.ph = nil
		in := d.inst
		x.removeTimer(d)
		x.fire(d)
		if d.shiftable {
			// LoopScheduler's drain observer re-checks the idle cap after a scheduler task.
			x.enforceIdleCap(false)
			x.enforceTraceIdleCap(in, false)
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
	case d.refusal != nil:
		x.end(d.refusal, d.outcome, x.nowUs)
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
	// A recycled tree's replay origin is its own root's first timestamp, not the lane's
	// original t* (which belongs to a different trace): aiperf sets _replay_origin_ms_by_root
	// to the recycled root's first turn. Turn 0 exists because nextRecycle skips empty roots.
	in.originMs = tr.convs[0].turns[0].tsMs
	x.laneInst[lane] = in
	// A recycle during the accelerated substage is itself quota-gated: the fresh root's turn 0
	// counts against (and is stopped by) the lane's warmup request quota, so a lane whose trees
	// drain under compression cannot recycle unboundedly.
	x.dispatchTurn(in, &agxDispatch{inst: in, stream: in.root, turn: 0, start: true})
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
	// During the accelerated substage every spawned child starts at once (zero idle delay)
	// and is quota-gated, like every other accelerated turn.
	if x.phase == agxAccel {
		for _, k := range kids {
			x.dispatchAccel(in, &agxDispatch{inst: in, stream: k.s, turn: 0, childPath: true, first: true})
		}
		return
	}
	for _, k := range kids {
		d := &agxDispatch{inst: in, stream: k.s, turn: 0, childPath: true, first: true}
		if k.offset > 0 {
			x.stats.DelayedChildSpawn++
			x.schedule(d, x.nowUs+msToUs(k.offset), false)
		}
	}
	immediate := false
	for _, k := range kids {
		immediate = immediate || k.offset <= 0
	}
	if immediate {
		// aiperf dispatches the immediate children under asyncio.gather, which yields to
		// the loop: barrier-released sends already queued as tasks go out first.
		x.issueReleased()
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
	d := x.issued(req)
	if d == nil || d.refusing {
		return nil
	}
	if req.ID != d.ph.ID && req.State == sim.StateCompleted {
		// A P/D sub-request finished its stage; the parent's completion is reported
		// separately, with the parent's ID.
		return nil
	}
	outcome := sim.AgenticCompleted
	switch req.State {
	case sim.StateTimedOut:
		outcome = sim.AgenticError // aiperf records a client timeout as an error
	case sim.StateQueued:
		// Refused at enqueue. The engine's context check (prompt, or prompt plus the
		// output budget, past max_model_len) is aiperf's context-overflow error; the other
		// refusals (KV budget, invalid budget) are plain errors.
		outcome = sim.AgenticError
		if x.cfg.MaxModelLen > 0 && (req.InputLen() >= x.cfg.MaxModelLen ||
			(req.MaxOutputLen > 0 && req.InputLen()+int64(req.MaxOutputLen) > x.cfg.MaxModelLen)) {
			outcome = sim.AgenticContextOverflow
		}
		d.refusing = true
		return x.respondError(d, outcome, now)
	}
	return x.terminal(d, outcome, now)
}

// agxRefusalRoundTripUs is when the client sees an error response after the server refused
// the request: one HTTP round trip, which aiperf's credit never skips. Without it a refusal
// with no admission or routing latency would come back at the instant of the send, and a
// stream or a lane recycle refused at every send would never let the clock advance.
const agxRefusalRoundTripUs = 1000

// respondError delivers a refusal's error response a round trip later.
func (x *AgentXReplay) respondError(d *agxDispatch, outcome string, now int64) []*sim.Request {
	x.nowUs = now
	x.push(&agxDispatch{state: agxTimer, refusal: d, outcome: outcome}, now+agxRefusalRoundTripUs)
	return x.flush()
}

// OnUnserved is the cluster's unserved hook: a request the cluster refused or dropped
// without completing it (admission or routing rejection, flow-control rejection or shed,
// queue TTL expiry, in-flight eviction, a P/D decode-KV drop). The serving stack answers
// such a request with an HTTP error, which aiperf records as an error and follows with the
// stream's next turn.
func (x *AgentXReplay) OnUnserved(req *sim.Request, now int64) []*sim.Request {
	d := x.issued(req)
	if d == nil || d.refusing {
		return nil
	}
	d.refusing = true
	return x.respondError(d, sim.AgenticError, now)
}

// issued resolves a request -- or a P/D sub-request of one (<id>_prefill, <id>_decode) --
// to the send in flight it belongs to.
func (x *AgentXReplay) issued(req *sim.Request) *agxDispatch {
	if req == nil {
		return nil
	}
	if d, ok := x.byID[req.ID]; ok && d.state == agxIssued {
		return d
	}
	for _, suf := range []string{"_prefill", "_decode"} {
		if base, ok := strings.CutSuffix(req.ID, suf); ok {
			if d, ok := x.byID[base]; ok && d.state == agxIssued {
				return d
			}
		}
	}
	return nil
}

// terminal is a credit return: the send d ended with outcome at now. It returns the sends
// the return causes.
func (x *AgentXReplay) terminal(d *agxDispatch, outcome string, now int64) []*sim.Request {
	x.end(d, outcome, now)
	return x.flush()
}

// end applies a credit return, leaving the sends it causes in x.out.
func (x *AgentXReplay) end(d *agxDispatch, outcome string, now int64) {
	req := d.ph
	delete(x.byID, req.ID)
	d.state = agxDropped
	x.nowUs = now
	x.inFlight--
	if x.inFlight == 0 {
		x.idleSinceUs = now
	}
	if in := d.inst; in != nil {
		in.inFlight--
		if in.inFlight == 0 {
			in.idleSinceUs = now
		}
	}
	req.Agentic.Outcome = outcome
	switch outcome {
	case sim.AgenticCompleted:
		x.stats.Completed++
	case sim.AgenticContextOverflow:
		x.stats.ContextOverflows++
	case sim.AgenticError:
		x.stats.Errors++
		if d.warmup {
			x.stats.WarmupFailures++
		}
	}
	if x.phase == agxProfile && now > x.windowEndUs {
		req.Phase = sim.PhaseGrace
	}
	// A baseline primer return (synthesized warmup): once every primer is back, either start
	// the accelerated cache-pressure substage or go straight to profiling.
	if d.warmup && !d.accel {
		x.warmupLeft--
		if x.warmupLeft == 0 {
			if x.cacheWarmupEnabled() {
				x.startAccel(now)
			} else {
				x.startProfiling(now)
			}
		}
		x.enforceIdleCap(true)
		return
	}
	// An accelerated-warmup turn return: advance the live trajectory through the real loop
	// (zero delay, quota-gated in dispatch), then hand off to profiling once every lane has
	// hit its quota and the substage has drained.
	if d.accel {
		x.onReturn(d, outcome == sim.AgenticContextOverflow)
		x.issueReleased()
		if x.accelOn && x.accelBudgetReached && !x.accelWorkPending() {
			x.startProfiling(now)
		}
		x.enforceIdleCap(true)
		return
	}
	x.onReturn(d, outcome == sim.AgenticContextOverflow)
	x.enforceIdleCap(true)
	x.enforceTraceIdleCap(d.inst, true)
	x.issueReleased()
	x.maybeComplete()
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
// end-to-start delay, from this completion. During the accelerated substage the delay is
// dropped and the turn is quota-gated (aiperf _handle_accelerated_warmup_return).
func (x *AgentXReplay) dispatchNext(d *agxDispatch) {
	s := d.stream
	nd := &agxDispatch{inst: d.inst, stream: s, turn: d.turn + 1, childPath: s.child}
	if x.phase == agxAccel {
		x.dispatchAccel(d.inst, nd)
		return
	}
	tn := &d.inst.trace.convs[s.conv].turns[d.turn+1]
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
			x.dispatchTurn(in, &agxDispatch{inst: in, stream: in.root, turn: j.gated})
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
		x.dispatchTurn(in, &agxDispatch{inst: in, stream: in.root, turn: j.gated})
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

// enforceTraceIdleCap is aiperf's per-trace idle watchdog (replay_dependencies.py:295-327),
// the global cap scoped to one session tree: when nothing of this tree is in flight, its own
// pending scheduler timers move earlier by the same amount so the tree is never idle longer
// than the per-trace cap, while every other tree's timing is untouched. At concurrency >= 2 a
// tree can idle while others are busy, which the global cap (keyed on the whole system being
// idle) never catches; this is what carries the offered concurrency the run saw.
// afterReturn is true on a credit return (which starts a new idle interval for the tree).
func (x *AgentXReplay) enforceTraceIdleCap(in *agxInstance, afterReturn bool) {
	// Profiling only: aiperf arms the per-trace watchdog after warmup, and never once the
	// sending window has closed.
	if x.cfg.TraceIdleCapUs <= 0 || x.phase != agxProfile || in == nil {
		return
	}
	if in.inFlight > 0 {
		in.idleSinceUs = -1
		return
	}
	now := x.nowUs
	if afterReturn || in.idleSinceUs < 0 {
		in.idleSinceUs = now
	}
	// This tree's shiftable timers, in schedule order (INV-6: the x.timers slice, never a
	// map). A recycle timer has no owning instance (t.inst == nil) and belongs to a drained
	// tree's lane, not this live tree, so t.inst == in excludes it.
	budget := max(0, x.cfg.TraceIdleCapUs-(now-in.idleSinceUs))
	earliest := int64(math.MaxInt64)
	any := false
	for _, t := range x.timers {
		if t.inst == in {
			earliest = min(earliest, t.dueUs)
			any = true
		}
	}
	if !any {
		return
	}
	shift := earliest - now - budget
	if shift <= 0 {
		return
	}
	x.stats.TraceIdleCapJumps++
	x.stats.TraceIdleCapSkippedUs += shift
	for _, t := range x.timers {
		if t.inst == in {
			x.push(t, max(now, t.dueUs-shift))
		}
	}
}

// maybeComplete records the profiling phase completion once the window has closed,
// nothing is in flight and no branch work is pending (PhaseRunner
// _wait_for_returning_complete with BranchOrchestrator.has_pending_branch_work); otherwise
// the phase completes at the grace end.
func (x *AgentXReplay) maybeComplete() {
	if x.phase != agxClosed || x.inFlight > 0 || x.completeUs >= 0 || x.pendingBranchWork() {
		return
	}
	x.completeUs = x.nowUs
	x.phase = agxDone
	x.setHorizon(x.nowUs)
}

// pendingBranchWork is BranchOrchestrator.has_pending_branch_work.
func (x *AgentXReplay) pendingBranchWork() bool {
	for _, in := range x.instances {
		if in.active != nil || len(in.future) > 0 || in.descendants > 0 {
			return true
		}
		for _, s := range in.childStreams {
			if s.tracked {
				return true
			}
		}
	}
	return false
}

// Finish closes the replay after the simulation stopped at stopUs: requests still in
// flight were cancelled at the grace end and leave no aiperf record.
func (x *AgentXReplay) Finish(stopUs int64) {
	cancelled := 0
	for _, d := range x.byID {
		if d.state == agxIssued && d.ph.Agentic != nil && d.ph.Agentic.Outcome == "" {
			d.ph.Agentic.Outcome = sim.AgenticCancelled
			cancelled++
			if d.refusing {
				x.stats.CancelledRefused++
			}
		}
	}
	x.stats.Cancelled += cancelled
	if x.completeUs < 0 && x.phase >= agxProfile {
		// The grace period ran out (PhaseRunner._wait_for_returning_complete): aiperf
		// cancels what is in flight and, when branch work is still pending, waits out
		// CANCEL_DRAIN_TIMEOUT (10 s) for the all-returned signal the pending work keeps
		// deferring; with nothing in flight it completes at the grace end.
		x.completeUs = x.graceEndUs
		if cancelled > 0 && x.pendingBranchWork() {
			x.completeUs += agxCancelDrainTimeoutUs
		}
		if x.graceEndUs == math.MaxInt64 {
			x.completeUs = max(stopUs, x.windowEndUs)
		}
		if x.cfg.HorizonCapUs > 0 && x.completeUs > x.cfg.HorizonCapUs {
			x.completeUs = x.cfg.HorizonCapUs // a --horizon cut the run short
		}
	}
}

// agxCancelDrainTimeoutUs is aiperf's Environment.TIMING.CANCEL_DRAIN_TIMEOUT default.
const agxCancelDrainTimeoutUs = 10 * 1_000_000

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
	req.InputTokens = x.promptTokens(in, s, d.turn)
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

// promptTokens renders the wire prompt of turn `turn` of stream s: the stream's message
// list after that turn (agxRecon, aiperf's ConversationReconstructor; aiperf re-sends the
// full history every turn), through the chat template, with the tree's cache-bust marker at
// the head of the first user message. Message content is exactly the recorded `in` tokens:
// hash blocks, whose tokens depend only on the trace and the hash id (so equal blocks are
// equal tokens across the tree's conversations), and synthetic tails unique to their
// (stream, turn). The marker differs per tree instance, so nothing is shared across trees,
// and template tokens at each message boundary end cross-conversation sharing where the
// real wire does: at the first message boundary the two conversations place differently.
func (x *AgentXReplay) promptTokens(in *agxInstance, s *agxStream, turn int32) []sim.TokenID {
	conv := &in.trace.convs[s.conv]
	if s.recon == nil || s.recon.turn >= turn {
		s.recon = newAgxRecon()
	}
	for k := s.recon.turn + 1; k <= turn; k++ {
		s.recon.advance(conv, k)
	}
	return agxRender(in.trace, in.salt, s.conv, s.recon.segs, conv.turns[turn].in, true)
}

// agxRender renders a message list: with the template, as promptTokens describes; without
// it, the bare concatenated message content (the hash-id-only model, for comparison).
func agxRender(tr *AgentXTrace, salt uint64, conv int32, segs []agxSeg, in int32, template bool) []sim.TokenID {
	cv := &tr.convs[conv]
	toks := make([]sim.TokenID, 0, int(in)+agxTemplateOverhead(segs))
	if template {
		toks = append(toks, agxTokBOS)
	}
	marker := false
	hashes := map[int32][]int64{}
	for _, sg := range segs {
		if template {
			switch sg.role {
			case 'a':
				toks = append(toks, agxTokAssistant)
			case 's':
				toks = append(toks, agxTokSystem)
			default:
				toks = append(toks, agxTokUser)
				if !marker {
					m := agxMix(salt, 0x6d61726b6572)
					for i := 0; i < agxMarkerTokens; i++ {
						toks = append(toks, agxToken(m, uint64(i)))
					}
					marker = true
				}
			}
		}
		for _, r := range sg.runs {
			if r.tail {
				base := agxMix(agxMix(salt^0x7461696c, uint64(conv)), uint64(r.turn))
				for t := r.a; t < r.b; t++ {
					toks = append(toks, agxToken(base, uint64(t)))
				}
				continue
			}
			hl, ok := hashes[r.turn]
			if !ok {
				tn := &cv.turns[r.turn]
				hl = tr.hashPrefix(tn.hashNode, tn.nHash, int(tn.nHash), nil)
				hashes[r.turn] = hl
			}
			for t := r.a; t < r.b; t++ {
				toks = append(toks, agxBlockToken(tr, hl[t/agxBlockSize], t%agxBlockSize))
			}
		}
		if template && sg.role == 'a' {
			toks = append(toks, agxTokEOS)
		}
	}
	if template {
		toks = append(toks, agxTokAssistant)
	}
	return toks
}

// agxBlockToken is token i of hash block h of the trace: the trace's hash-id namespace is
// local (hash_id_scope "local"), so the trace id seeds it.
func agxBlockToken(tr *AgentXTrace, h int64, i int32) sim.TokenID {
	return agxToken(agxMix(tr.blockSalt(), uint64(h)), uint64(i))
}

func (t *AgentXTrace) blockSalt() uint64 { return t.salt }

func agxTraceSalt(id string) uint64 {
	s := agxMix(0x626c6f636b, uint64(len(id)))
	for _, c := range []byte(id) {
		s = agxMix(s, uint64(c))
	}
	return s
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

// agxToken is a content token: above the reserved template ids.
func agxToken(base, i uint64) sim.TokenID {
	return sim.TokenID(agxContentBase + agxMix(base, i)>>34)
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
	TraceIdleCapJumps          int     `json:"trace_idle_cap_jumps"`
	TraceIdleCapSkippedS       float64 `json:"trace_idle_cap_skipped_s"`
	BarrierHeld                int     `json:"barrier_holds"`
	JoinSuspensions            int     `json:"join_suspensions"`
	ChildrenSpawned            int     `json:"children_spawned"`
	UnspawnableSkipped         int     `json:"unspawnable_traces_skipped"`
	WarmupFailures             int     `json:"warmup_failures"`
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
		TraceIdleCapJumps: s.TraceIdleCapJumps, TraceIdleCapSkippedS: float64(s.TraceIdleCapSkippedUs) / 1e6,
		BarrierHeld: s.BarrierHeld, JoinSuspensions: s.JoinSuspensions, ChildrenSpawned: s.ChildrenSpawned,
		UnspawnableSkipped: s.UnspawnableSkipped, WarmupFailures: s.WarmupFailures,
	}
}

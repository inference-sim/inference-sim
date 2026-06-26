package workload

import (
	"container/heap"
	"errors"
	"fmt"
	"math/rand"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// ErrLazyUnsupportedTimeVarying signals that lazy generation cannot handle
// a spec with per-window parameter overrides. The caller (cmd/root.go) is
// expected to log a warning and fall back to GenerateWorkload. The error
// path is intentionally a value, not a Fatalf — library code does not
// terminate the process (R6, #1441).
var ErrLazyUnsupportedTimeVarying = errors.New("lazy generation: time-varying workloads (per-window parameters) not supported in alpha")

// ErrLazyUnsupportedConcurrency signals fallback for specs containing
// concurrency clients (Concurrency > 0). Concurrency-driven workloads
// generate their seed requests through a separate path in GenerateWorkload
// (concurrencyRNG); lazy mode in this PR's alpha scope does not replicate
// that path. Caller should fall back to GenerateWorkload.
var ErrLazyUnsupportedConcurrency = errors.New("lazy generation: concurrency clients not supported in alpha")

// ErrLazyUnsupportedMultiSession signals fallback for specs containing
// reasoning multi-turn clients with SingleSession=false. In multi-session
// mode the eager generator interleaves sessions of a single client by
// arrival time across the merge-sort, and reproducing that interleaving
// in a single-entry-per-client streaming source would require pushing
// every session's anchor onto the heap up front — defeating the memory
// savings. SingleSession=true clients (the inference-perf shape and the
// issue's reproducer) ARE supported. Caller falls back to GenerateWorkload
// for multi-session specs.
var ErrLazyUnsupportedMultiSession = errors.New("lazy generation: reasoning clients with SingleSession=false not supported in alpha")

// clientStreamState holds per-client streaming production state.
// One per "live" client in the original allClients order (the order matters:
// it is the priority-queue tie-break for identical arrival times, and it
// determines blueprint enumeration order — see GenerateWorkloadLazy).
type clientStreamState struct {
	clientIdx       int          // position in original allClients slice
	client          *ClientSpec  // pointer for field access; never mutated
	arrivalSampler  ArrivalSampler
	inputSampler    LengthSampler
	outputSampler   LengthSampler
	clientRNG       *rand.Rand
	prefix          []int
	horizon         int64
	isReasoning     bool
	isSingleSession bool
	isClosedLoop    bool

	// Single-shot mode: monotonic IAT accumulator (matches generator.go:281).
	currentTime int64

	// Reasoning mode: pendingSession holds one session's pre-computed rounds.
	// Yielded one at a time via pendingSessionIdx, then drained by drawing the
	// next IAT and calling GenerateReasoningRequests again.
	pendingSession    []*sim.Request
	pendingSessionIdx int
	// singleSessionDone short-circuits SingleSession=true reasoning clients
	// after they emit their only session.
	singleSessionDone bool

	// perClientSeq increments on every successful produceNext yield. Used as
	// the tertiary heap tie-breaker so that even when two queue entries from
	// the same client (theoretically impossible) would collide, the heap is
	// deterministic.
	perClientSeq int64

	// exhausted is sticky — once true, produceNext returns (nil, 0, false)
	// forever. Matches the cluster's RequestSource exhaustion contract.
	exhausted bool
}

// produceNext returns the next request this client would emit (if any),
// advancing the client's RNG and perClientSeq. Returns (nil, 0, false)
// when exhausted.
//
// CRITICAL: This method must consume RNG draws in the same order as the
// eager loop in GenerateRequests for the corresponding client kind.
// Any reordering breaks INV-6 (byte-identical stdout across modes).
func (s *clientStreamState) produceNext() (*sim.Request, int64, bool) {
	if s.exhausted {
		return nil, 0, false
	}
	if s.isReasoning {
		return s.produceNextReasoning()
	}
	return s.produceNextSingleShot()
}

func (s *clientStreamState) produceNextSingleShot() (*sim.Request, int64, bool) {
	for {
		if s.currentTime >= s.horizon {
			s.exhausted = true
			return nil, 0, false
		}
		iat := s.arrivalSampler.SampleIAT(s.clientRNG)
		if iat == 0 {
			// Stateful sampler exhausted (matches eager generator.go:288).
			s.exhausted = true
			return nil, 0, false
		}
		s.currentTime += iat
		if s.currentTime >= s.horizon {
			s.exhausted = true
			return nil, 0, false
		}
		// Lifecycle window filtering (matches generator.go:299-304).
		if s.client.Lifecycle != nil && !isInActiveWindow(s.currentTime, s.client.Lifecycle) {
			if s.currentTime >= lastWindowEndUs(s.client.Lifecycle) {
				s.exhausted = true
				return nil, 0, false
			}
			continue
		}

		// Token generation — must match generator.go:306-325 exactly,
		// including the RNG-draw order for multimodal vs standard paths.
		var inputTokens, outputTokens []int
		var textCount, imageCount, audioCount, videoCount int
		if s.client.Multimodal != nil {
			var err error
			inputTokens, textCount, imageCount, audioCount, videoCount, err = GenerateMultimodalTokens(s.clientRNG, s.client.Multimodal)
			if err != nil {
				// Multimodal validation runs at GenerateWorkloadLazy time, so this
				// shouldn't happen for valid specs. Treat as terminal failure rather
				// than panicking inside library code (R6); the resulting empty
				// stream surfaces the issue via downstream "no requests" warning.
				s.exhausted = true
				return nil, 0, false
			}
			outputLen := s.outputSampler.Sample(s.clientRNG)
			outputTokens = sim.GenerateRandomTokenIDs(s.clientRNG, outputLen)
		} else {
			inputLen := s.inputSampler.Sample(s.clientRNG)
			outputLen := s.outputSampler.Sample(s.clientRNG)
			inputTokens = sim.GenerateRandomTokenIDs(s.clientRNG, inputLen)
			outputTokens = sim.GenerateRandomTokenIDs(s.clientRNG, outputLen)
		}
		var prefixLength int
		if len(s.prefix) > 0 {
			inputTokens = append(append([]int{}, s.prefix...), inputTokens...)
			prefixLength = len(s.prefix)
		}
		req := &sim.Request{
			ID:               "", // assigned at heap-pop by lazyRequestSource.Next
			ArrivalTime:      s.currentTime,
			InputTokens:      inputTokens,
			OutputTokens:     outputTokens,
			MaxOutputLen:     len(outputTokens),
			State:            sim.StateQueued,
			ScheduledStepIdx: 0,
			FinishedStepIdx:  0,
			TenantID:         s.client.TenantID,
			SLOClass:         s.client.SLOClass,
			Model:            s.client.Model,
			TextTokenCount:   textCount,
			ImageTokenCount:  imageCount,
			AudioTokenCount:  audioCount,
			VideoTokenCount:  videoCount,
			Deadline:         computeDeadline(s.currentTime, s.client.Timeout, isClosedLoop(s.client)),
			SLOTargetUs:      derefInt64(s.client.SLOTargetUs),
			ClientID:         s.client.ID,
			PrefixGroup:      s.client.PrefixGroup,
			PrefixLength:     prefixLength,
			Streaming:        s.client.Streaming,
		}
		s.perClientSeq++
		return req, s.currentTime, true
	}
}

func (s *clientStreamState) produceNextReasoning() (*sim.Request, int64, bool) {
	for {
		// First, drain any pending session rounds. We yield ALL rounds here,
		// including non-round-0 rounds of closed-loop sessions, so that each
		// counts toward maxRequests at the lazyRequestSource level — matching
		// eager mode's "produce all rounds, sort+truncate, then filter round-0"
		// sequencing in GenerateWorkload. The round-0-only filter for
		// closed-loop sessions is applied at lazyRequestSource.Next, AFTER
		// the cap-counting heap pop.
		for s.pendingSessionIdx < len(s.pendingSession) {
			req := s.pendingSession[s.pendingSessionIdx]
			s.pendingSessionIdx++
			if req.ArrivalTime >= s.horizon {
				// Rounds are chronological; remaining are past horizon too.
				// Matches generator.go:264 / generator.go:203 break.
				s.pendingSession = nil
				s.pendingSessionIdx = 0
				break
			}
			if s.client.Lifecycle != nil && !isInActiveWindow(req.ArrivalTime, s.client.Lifecycle) {
				// Skip rounds outside lifecycle windows (matches generator.go:268).
				continue
			}
			s.perClientSeq++
			return req, req.ArrivalTime, true
		}
		// No more pending rounds — release the session slice.
		s.pendingSession = nil
		s.pendingSessionIdx = 0

		// Single-session reasoning client: only one session per client.
		if s.isSingleSession && s.singleSessionDone {
			s.exhausted = true
			return nil, 0, false
		}

		// Draw the next session's start time.
		iat := s.arrivalSampler.SampleIAT(s.clientRNG)
		if iat == 0 {
			s.exhausted = true
			return nil, 0, false
		}
		var startTime int64
		if s.isSingleSession {
			startTime = iat
			if s.client.Lifecycle != nil && len(s.client.Lifecycle.Windows) > 0 {
				startTime = s.client.Lifecycle.Windows[0].StartUs + iat
			}
			s.singleSessionDone = true
			if startTime >= s.horizon {
				s.exhausted = true
				return nil, 0, false
			}
			if s.client.Lifecycle != nil && !isInActiveWindow(startTime, s.client.Lifecycle) {
				s.exhausted = true
				return nil, 0, false
			}
		} else {
			s.currentTime += iat
			startTime = s.currentTime
			if startTime >= s.horizon {
				s.exhausted = true
				return nil, 0, false
			}
			if s.client.Lifecycle != nil && !isInActiveWindow(startTime, s.client.Lifecycle) {
				if startTime >= lastWindowEndUs(s.client.Lifecycle) {
					s.exhausted = true
					return nil, 0, false
				}
				continue // try next IAT (matches generator.go:233-238)
			}
		}

		// Build this session.
		reasoningReqs, err := GenerateReasoningRequests(
			s.clientRNG, s.client.Reasoning,
			s.inputSampler, s.outputSampler,
			startTime,
			s.client.ID, s.client.TenantID, s.client.SLOClass, s.client.Model,
		)
		if err != nil {
			s.exhausted = true
			return nil, 0, false
		}
		// Prepend shared prefix (matches generator.go:191-196 / 248-254).
		if len(s.prefix) > 0 {
			for _, req := range reasoningReqs {
				req.InputTokens = append(append([]int{}, s.prefix...), req.InputTokens...)
				req.PrefixLength = len(s.prefix)
			}
		}
		// Set Deadline + SLOTargetUs on every round (matches generator.go:198-201 / 256-259).
		for _, req := range reasoningReqs {
			req.Deadline = computeDeadline(req.ArrivalTime, s.client.Timeout, true)
			req.SLOTargetUs = derefInt64(s.client.SLOTargetUs)
		}
		s.pendingSession = reasoningReqs
		s.pendingSessionIdx = 0
		// Loop back to yield the first round.
	}
}

// heapEntry is one priority-queue entry. The ordering key is
// (arrivalUs, clientIdx, perClientSeq) — the second component matches
// sort.SliceStable's tie-break behavior in the eager path (lower
// allClients index emitted first on identical arrivals), and the third
// breaks any residual ties deterministically.
type heapEntry struct {
	arrivalUs    int64
	clientIdx    int
	perClientSeq int64
	req          *sim.Request
	state        *clientStreamState
}

type heapByArrival []heapEntry

func (h heapByArrival) Len() int { return len(h) }
func (h heapByArrival) Less(i, j int) bool {
	a, b := h[i], h[j]
	if a.arrivalUs != b.arrivalUs {
		return a.arrivalUs < b.arrivalUs
	}
	if a.clientIdx != b.clientIdx {
		return a.clientIdx < b.clientIdx
	}
	return a.perClientSeq < b.perClientSeq
}
func (h heapByArrival) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *heapByArrival) Push(x interface{}) { *h = append(*h, x.(heapEntry)) }
func (h *heapByArrival) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// lazyRequestSource is the streaming RequestSource implementation. It
// satisfies cluster.RequestSource via structural typing (Go duck-typing);
// sim/workload does not import sim/cluster. The cluster's Run() loop
// pulls requests one at a time via Next() until exhausted.
//
// Concurrency: not safe for concurrent use. Single-goroutine consumer.
type lazyRequestSource struct {
	h           *heapByArrival
	popped      int64 // total heap pops (counts toward maxRequests cap, including suppressed intermediate closed-loop rounds)
	yielded     int64 // requests actually returned from Next; used for sequential ID assignment
	maxRequests int64
}

// Next returns the next request and true, or (nil, false) when exhausted.
// Exhaustion is sticky — once the source returns (nil, false) once, it
// must continue to do so on subsequent calls.
//
// Each pop:
//  1. Pop the lowest-(arrival, clientIdx, perClientSeq) entry.
//  2. Ask that entry's state to produce its next candidate; if it does,
//     push the new entry back onto the heap.
//  3. If the popped request belongs to a closed-loop session AND its
//     RoundIndex > 0, it is a non-emitted intermediate round (matches
//     GenerateWorkload's round-0-only filter at generator.go:531-542).
//     Such rounds DO count toward maxRequests (so the eager/lazy total
//     budget matches) but are NOT yielded to the cluster — we loop and
//     pop the next entry instead.
//  4. Assign req.ID = "request_<emitted>" in pop order, then increment
//     the emit counter.
//
// Stopping conditions: heap empty, or emitted >= maxRequests (when > 0).
func (l *lazyRequestSource) Next() (*sim.Request, bool) {
	if l.h == nil {
		return nil, false
	}
	for {
		if l.maxRequests > 0 && l.popped >= l.maxRequests {
			return nil, false
		}
		if l.h.Len() == 0 {
			return nil, false
		}
		e := heap.Pop(l.h).(heapEntry)
		// Re-push the state with its next candidate, if any. Always push
		// regardless of whether THIS entry will be yielded to the cluster
		// (intermediate closed-loop rounds advance state but don't emit).
		if nextReq, t, ok := e.state.produceNext(); ok {
			heap.Push(l.h, heapEntry{
				arrivalUs:    t,
				clientIdx:    e.state.clientIdx,
				perClientSeq: e.state.perClientSeq,
				req:          nextReq,
				state:        e.state,
			})
		}
		// Count this pop against the global maxRequests budget regardless
		// of whether it surfaces to the cluster — this preserves eager/lazy
		// parity, where GenerateWorkload produces all rounds, truncates to
		// maxRequests, then filters round-0 for closed-loop sessions.
		l.popped++
		// Intermediate closed-loop rounds are suppressed; loop to pop the
		// next candidate. The cluster only ever sees round-0 of closed-loop
		// sessions (the SessionManager generates the follow-up rounds).
		if e.state.isClosedLoop && e.req.RoundIndex != 0 {
			continue
		}
		// Sequential IDs are assigned in yield order, matching
		// GenerateWorkload's post-filter renumbering at generator.go:619-621.
		e.req.ID = fmt.Sprintf("request_%d", l.yielded)
		l.yielded++
		return e.req, true
	}
}

// clientPrep captures the per-client information collected during the
// construction prelude of GenerateWorkloadLazy. clientSeed is drawn
// from workloadRNG in allClients order before any client begins
// producing requests — matching the eager generator (generator.go:119).
type clientPrep struct {
	idx        int
	client     *ClientSpec
	clientSeed int64
	rate       float64
	prefix     []int
}

// GenerateWorkloadLazy mirrors GenerateWorkload's setup but returns a
// streaming RequestSource instead of materializing the request slice.
// Memory consumption while the cluster drains the source is
// O(num_clients + max_session_rounds), independent of total request count.
//
// Returns:
//   - source: implements cluster.RequestSource via structural typing.
//   - sessions: deterministic session blueprints for closed-loop clients,
//     in the same (client-order, sorted-session-IDs) as GenerateWorkload.
//   - followUpBudget: -1 in lazy mode (concurrency clients force eager fallback).
//   - err: ErrLazyUnsupported* signals the caller to fall back; other errors
//     are real spec/validation failures.
//
// Determinism: same seed produces the same source byte-identically to
// GenerateWorkload's request stream (BC-3).
func GenerateWorkloadLazy(spec *WorkloadSpec, horizon int64, maxRequests int64) (
	*lazyRequestSource, []SessionBlueprint, int64, error) {

	if horizon <= 0 {
		// Empty workload: return an immediately-exhausted source.
		return &lazyRequestSource{h: &heapByArrival{}, maxRequests: maxRequests}, nil, -1, nil
	}
	if maxRequests < 0 {
		return nil, nil, 0, fmt.Errorf("maxRequests must be non-negative, got %d", maxRequests)
	}

	// Shared spec prelude (mutual-exclusion, inference-perf expansion,
	// servegen load, v1→v2 upgrade, Validate). Mutates spec.Clients in place.
	if err := validateAndExpandSpec(spec); err != nil {
		return nil, nil, 0, err
	}

	// Build working client list (matches generator.go:74-79).
	allClients := append([]ClientSpec{}, spec.Clients...)
	if len(spec.Cohorts) > 0 {
		allClients = append(allClients, ExpandCohorts(spec.Cohorts, spec.Seed)...)
	}

	// Fallback gates. The caller (cmd/root.go) catches these sentinel errors
	// and switches to the eager generator with a one-line warning.
	if hasPerWindowParameters(allClients) {
		return nil, nil, 0, ErrLazyUnsupportedTimeVarying
	}
	for i := range allClients {
		if allClients[i].Concurrency > 0 {
			return nil, nil, 0, ErrLazyUnsupportedConcurrency
		}
		if allClients[i].Reasoning != nil && allClients[i].Reasoning.MultiTurn != nil &&
			!allClients[i].Reasoning.MultiTurn.SingleSession {
			return nil, nil, 0, ErrLazyUnsupportedMultiSession
		}
	}

	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(spec.Seed))
	workloadRNG := rng.ForSubsystem(sim.SubsystemWorkloadGen)
	clientRates := normalizeRateFractions(allClients, spec.AggregateRate)
	prefixes := generatePrefixTokens(allClients, workloadRNG)

	// Phase 1: prelude — draw clientSeeds in allClients order, mirroring the
	// eager loop's draw order (generator.go:114-120). Zero-rate clients are
	// skipped BEFORE the draw, matching generator.go:114-116.
	preps := make([]clientPrep, 0, len(allClients))
	for i := range allClients {
		if clientRates[i] <= 0 {
			continue
		}
		clientSeed := workloadRNG.Int63()
		preps = append(preps, clientPrep{
			idx:        i,
			client:     &allClients[i],
			clientSeed: clientSeed,
			rate:       clientRates[i],
			prefix:     prefixes[allClients[i].PrefixGroup],
		})
	}

	// Phase 2: blueprint pre-pass (closed-loop reasoning clients only).
	// We replay each closed-loop client's RNG to enumerate session IDs
	// without retaining any request slices (each session's slice is
	// generated and discarded). This advances a temporary clientRNG; the
	// real streaming pass below uses a *fresh* clientRNG seeded from the
	// same clientSeed, so both passes see identical RNG streams.
	blueprintRNG := rand.New(rand.NewSource(spec.Seed + 7919))
	var sessions []SessionBlueprint
	for _, p := range preps {
		if !isClosedLoop(p.client) || p.client.Reasoning == nil || p.client.Reasoning.MultiTurn == nil {
			continue
		}
		sessIDs, err := enumerateReasoningSessionIDs(p, horizon)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("client %q: %w", p.client.ID, err)
		}
		// Sort session IDs to match GenerateWorkload's deterministic
		// blueprint-construction order (generator.go:504-508).
		sort.Strings(sessIDs)
		// Build samplers once per client (mirrors generator.go:447-455).
		inputSampler, err := NewLengthSampler(p.client.InputDist)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("client %q input distribution for blueprint: %w", p.client.ID, err)
		}
		outputSampler, err := NewLengthSampler(p.client.OutputDist)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("client %q output distribution for blueprint: %w", p.client.ID, err)
		}
		// Per-session prefix: GenerateWorkload extracts from req.InputTokens,
		// but the value is exactly prefixes[client.PrefixGroup] (eager prepends
		// it in the same way before populating req.InputTokens). We use it
		// directly to avoid materializing requests just to read them back.
		var prefixTokens []int
		if p.client.PrefixGroup != "" {
			prefixTokens = prefixes[p.client.PrefixGroup]
		}
		mt := p.client.Reasoning.MultiTurn
		for _, sessID := range sessIDs {
			sessSeed := blueprintRNG.Int63()
			sessions = append(sessions, SessionBlueprint{
				SessionID:     sessID,
				ClientID:      p.client.ID,
				MaxRounds:     mt.MaxRounds,
				ContextGrowth: mt.ContextGrowth,
				ThinkTimeUs:   mt.ThinkTimeUs,
				Timeout:       p.client.Timeout,
				Horizon:       horizon,
				InputSampler:  inputSampler,
				OutputSampler: outputSampler,
				RNG:           rand.New(rand.NewSource(sessSeed)),
				Prefix:        prefixTokens,
				TenantID:      p.client.TenantID,
				SLOClass:      p.client.SLOClass,
				Model:         p.client.Model,
				SLOTargetUs:   derefInt64(p.client.SLOTargetUs),
			})
		}
	}

	// Phase 3: build the streaming states with fresh RNGs seeded from the
	// same clientSeeds. Seeded from scratch — `rand.New(rand.NewSource(s))`
	// is deterministic, so the streaming RNG produces the same sequence as
	// the pre-pass RNG that enumerated session IDs.
	h := &heapByArrival{}
	heap.Init(h)
	for _, p := range preps {
		state, err := buildClientStreamState(p, horizon)
		if err != nil {
			return nil, nil, 0, err
		}
		if firstReq, t, ok := state.produceNext(); ok {
			heap.Push(h, heapEntry{
				arrivalUs:    t,
				clientIdx:    state.clientIdx,
				perClientSeq: state.perClientSeq,
				req:          firstReq,
				state:        state,
			})
		}
	}

	// FollowUpBudget: lazy mode rejects concurrency specs above, so the
	// eager codepath's "totalConcurrencyUsers > 0" condition is always
	// false — budget stays at -1 (no cap), matching GenerateWorkload:671.
	return &lazyRequestSource{h: h, maxRequests: maxRequests}, sessions, int64(-1), nil
}

// buildClientStreamState constructs one client's streaming state. The
// RNG is seeded from p.clientSeed; sampler construction mirrors the
// eager generator (generator.go:122-143) including the CustomSamplerFactory
// sub-RNG draw.
func buildClientStreamState(p clientPrep, horizon int64) (*clientStreamState, error) {
	clientRNG := newRandFromSeed(p.clientSeed)
	var arrivalSampler ArrivalSampler
	if p.client.CustomSamplerFactory != nil {
		subSeed := clientRNG.Int63()
		subRNG := newRandFromSeed(subSeed)
		arrivalSampler = p.client.CustomSamplerFactory(subRNG)
	} else {
		arrivalSampler = NewArrivalSampler(p.client.Arrival, p.rate)
	}
	inputSampler, err := NewLengthSampler(p.client.InputDist)
	if err != nil {
		return nil, fmt.Errorf("client %q input distribution: %w", p.client.ID, err)
	}
	outputSampler, err := NewLengthSampler(p.client.OutputDist)
	if err != nil {
		return nil, fmt.Errorf("client %q output distribution: %w", p.client.ID, err)
	}
	state := &clientStreamState{
		clientIdx:      p.idx,
		client:         p.client,
		arrivalSampler: arrivalSampler,
		inputSampler:   inputSampler,
		outputSampler:  outputSampler,
		clientRNG:      clientRNG,
		prefix:         p.prefix,
		horizon:        horizon,
	}
	if p.client.Reasoning != nil && p.client.Reasoning.MultiTurn != nil {
		state.isReasoning = true
		state.isSingleSession = p.client.Reasoning.MultiTurn.SingleSession
		state.isClosedLoop = isClosedLoop(p.client)
	}
	return state, nil
}

// enumerateReasoningSessionIDs replays a closed-loop reasoning client's
// RNG to discover the SessionIDs it would produce, without retaining
// session slices. Used by the blueprint pre-pass in GenerateWorkloadLazy.
//
// Implementation strategy: build a temporary clientStreamState (with a
// fresh RNG seeded from p.clientSeed) and call produceNext repeatedly,
// discarding requests but recording the SessionID of each new session.
// We detect "new session" by tracking SessionID transitions.
//
// Memory: bounded by the per-session slice held inside the state's
// pendingSession field (one session at a time, ~MaxRounds requests).
func enumerateReasoningSessionIDs(p clientPrep, horizon int64) ([]string, error) {
	state, err := buildClientStreamState(p, horizon)
	if err != nil {
		return nil, err
	}
	// Keep isClosedLoop = true so only round 0 of each session is yielded
	// (one per session). We read SessionID from each yielded round-0 to
	// enumerate every session without retaining the full per-session
	// request slice — pendingSession is overwritten on each refill.
	state.isClosedLoop = true

	var ids []string
	seen := make(map[string]bool)
	for {
		req, _, ok := state.produceNext()
		if !ok {
			break
		}
		if req.SessionID != "" && !seen[req.SessionID] {
			seen[req.SessionID] = true
			ids = append(ids, req.SessionID)
		}
	}
	return ids, nil
}

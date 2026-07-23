package workload

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
)

// The lazy generator streams EVERY workload class — there is no remaining
// ErrLazyUnsupported* fallback sentinel as of #1460:
//
//   - Concurrency clients (Concurrency > 0, #1459): seed requests +
//     SessionBlueprints are produced by the shared
//     generateConcurrencySeedsAndBlueprints helper (also used by the eager
//     GenerateWorkload), and each seed is pushed onto the global arrival heap as
//     its own entry so the merge reproduces eager's stable-sort-by-arrival
//     byte-for-byte. The seed set is O(N virtual users), present up front in both
//     modes — no horizon-length pre-materialization to stream away.
//
//   - Multi-session reasoning (SingleSession=false, #1458): each such client is a
//     traffic source spawning many independent, overlapping sessions across the
//     horizon. Eager interleaves them via its global merge-sort; the lazy
//     generator reproduces the interleaving with a small per-client internal merge
//     over its LIVE sessions (see clientStreamState.liveSessions and
//     produceNextReasoning), bounding resident sessions to the concurrent working
//     set (≈ arrival_rate × session_duration, Little's law) — independent of
//     horizon.
//
//   - Time-varying / per-window workloads (#1460): clients whose lifecycle windows
//     carry per-window trace_rate/arrival/input_distribution/output_distribution
//     overrides. Eager (generateTimeVaryingRequests) generates each window as a
//     self-contained batch in spec order, concatenates all, and stable-sorts by
//     arrival; the lazy path (generateTimeVaryingWorkloadLazy) reuses the eager
//     generateRequestsForWindow one window at a time, merging built window batches
//     through a per-client live-window heap with a suffix-min emit gate (see
//     clientStreamState.liveWindows and produceNextTimeVarying). Resident memory is
//     the concurrent-window working set rather than all windows of all clients — a
//     win for the many-small-windows layout; a single huge window still materializes
//     one full batch (generateRequestsForWindow is all-at-once), so it yields no
//     memory advantage over eager.

// clientStreamState holds per-client streaming production state.
// One per "live" client in the original allClients order (the order matters:
// it is the priority-queue tie-break for identical arrival times, and it
// determines blueprint enumeration order — see GenerateWorkloadLazy).
type clientStreamState struct {
	clientIdx       int         // position in original allClients slice
	client          *ClientSpec // pointer for field access; never mutated
	arrivalSampler  ArrivalSampler
	inputSampler    LengthSampler
	outputSampler   LengthSampler
	clientRNG       *rand.Rand
	prefix          []sim.TokenID
	horizon         int64
	isReasoning     bool
	isSingleSession bool
	isClosedLoop    bool

	// Single-shot mode: monotonic IAT accumulator (matches the
	// `currentTime` accumulator in GenerateRequests' non-reasoning loop).
	currentTime int64

	// Reasoning single-session mode: pendingSession holds the client's one
	// session's pre-computed rounds. Yielded one at a time via
	// pendingSessionIdx, then the client exhausts (singleSessionDone).
	pendingSession    []*sim.Request
	pendingSessionIdx int
	// singleSessionDone short-circuits SingleSession=true reasoning clients
	// after they emit their only session.
	singleSessionDone bool

	// Reasoning multi-session mode (SingleSession=false, #1458): a client is a
	// traffic source that spawns many independent, OVERLAPPING sessions over
	// the horizon. Because session N+1 can start before session N's later
	// rounds arrive, we cannot drain one session before the next and still feed
	// the global heap an arrival-monotonic sub-stream. Instead we keep an
	// internal min-heap of "live" sessions (built but not fully drained), keyed
	// by each session's next-round arrival time. Resident sessions are bounded
	// by the concurrent working set (≈ arrival_rate × session_duration, Little's
	// law) — independent of horizon.
	liveSessions   *liveSessionHeap
	nextSessionIdx int   // monotonic per-client session-build index (emit tie-break)
	msClientCap    int64 // per-client build cap = 2*maxRequests (0 = unbounded); mirrors eager's perClientCap
	msReqCount     int64 // count of ALL rounds built for this client (for msClientCap)
	msBuildDone    bool  // true once no further session can be built (horizon/sampler/lifecycle/cap)

	// Time-varying mode (per-window parameter overrides, #1460): a client whose
	// lifecycle windows carry per-window trace_rate/arrival/input_distribution/
	// output_distribution overrides. The eager path (generateTimeVaryingRequests)
	// generates each window as a self-contained batch (all IATs sampled up front,
	// rescaled to fill the window, then content) IN SPEC ORDER, concatenates all
	// clients' all windows, and stable-sorts by arrival. Because windows can be
	// overlapping or out-of-order in spec, the per-client concatenation is NOT
	// arrival-monotonic. The lazy path reuses generateRequestsForWindow verbatim
	// (one window at a time), keeps an internal min-heap of built-but-not-drained
	// window batches, and applies an emit-safety gate keyed on the suffix-minimum
	// window start — yielding the client's requests in arrival order so it feeds
	// the global heap a monotonic sub-stream byte-identical to eager's global sort.
	isTimeVarying  bool
	allClients     []ClientSpec // shared read-only; needed by computeProportionalRate inside generateRequestsForWindow
	aggregateRate  float64      // spec.AggregateRate (0 = absolute-rate mode)
	windows        []ActiveWindow
	windowBuildIdx int             // next window (spec order) to build
	suffixMinStart []int64         // suffixMinStart[k] = min StartUs over windows[k:]; [len]=MaxInt64 sentinel
	liveWindows    *liveWindowHeap // built-but-not-drained window batches
	twBuildDone    bool            // true once every window has been built (or skipped)

	// perClientSeq increments on every successful produceNext yield. Used as
	// the tertiary heap tie-breaker so that even when two queue entries from
	// the same client (theoretically impossible) would collide, the heap is
	// deterministic.
	perClientSeq int64

	// exhausted is sticky — once true, produceNext returns (nil, 0, false)
	// forever. Matches the cluster's RequestSource exhaustion contract.
	exhausted bool

	// lastErr captures the terminal error from a failed sampler /
	// generator call (multimodal token generation, reasoning session
	// generation). Set alongside `exhausted = true` on the failure
	// path. Surfaced up via lazyRequestSource.Err() so cmd/root.go
	// can Fatalf and match the eager path's abort-on-invalid-spec
	// behavior — instead of silently reducing traffic and exiting 0
	// with misleading capacity numbers (#1453 self-review round 3).
	lastErr error

	// dryRun disables user-facing logs on sampler errors — set true by
	// the blueprint pre-pass in enumerateSurvivingSessionsPerClient so
	// the same error is not double-logged (once in the pre-pass, once
	// in Phase 3 streaming). The Phase 3 pass is authoritative for
	// user feedback; the pre-pass's copy-of-the-state runs the same
	// samplers and would surface the same error immediately after.
	dryRun bool
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
	// Time-varying is checked FIRST: generateRequestsForWindow handles reasoning
	// (single- and multi-session) INTERNALLY per window, so a TV+reasoning client
	// must route here, not to produceNextReasoning. buildClientStreamState leaves
	// isReasoning unset for TV clients, but the explicit ordering documents intent.
	if s.isTimeVarying {
		return s.produceNextTimeVarying()
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
			// Stateful sampler exhausted (mirrors the `iat == 0 → break`
			// guard in GenerateRequests' non-reasoning per-client loop).
			s.exhausted = true
			return nil, 0, false
		}
		s.currentTime += iat
		if s.currentTime >= s.horizon {
			s.exhausted = true
			return nil, 0, false
		}
		// Lifecycle window filtering (mirrors the lifecycle/lastWindowEnd
		// check in GenerateRequests' non-reasoning loop).
		if s.client.Lifecycle != nil && !isInActiveWindow(s.currentTime, s.client.Lifecycle) {
			if s.currentTime >= lastWindowEndUs(s.client.Lifecycle) {
				s.exhausted = true
				return nil, 0, false
			}
			continue
		}

		// Token generation — must match GenerateRequests' single-shot
		// path exactly, including the RNG-draw order for multimodal vs
		// standard paths (multimodal: tokens → outputLen → outputTokens;
		// standard: inputLen → outputLen → input → output).
		var inputTokens, outputTokens []sim.TokenID
		var textCount, imageCount, audioCount, videoCount int
		if s.client.Multimodal != nil {
			var err error
			inputTokens, textCount, imageCount, audioCount, videoCount, err = GenerateMultimodalTokens(s.clientRNG, s.client.Multimodal)
			if err != nil {
				// spec.Validate() does NOT validate MultimodalSpec distribution
				// fields, so this path IS reachable for invalid multimodal specs.
				// Record the error on the state and sticky-exhaust; the
				// user-facing log happens on the Phase 3 pass only (skipped
				// during the blueprint pre-pass to avoid double-logging).
				// lazyRequestSource.Err() surfaces this to cmd/root.go so it
				// can Fatalf and match the eager path's abort-on-invalid-spec.
				s.recordError(fmt.Errorf("multimodal token generation: %w", err))
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
			inputTokens = append(append([]sim.TokenID{}, s.prefix...), inputTokens...)
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
			Adapter:          s.client.Adapter,
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
	if s.isSingleSession {
		return s.produceNextSingleSession()
	}
	return s.produceNextMultiSession()
}

// produceNextSingleSession streams a SingleSession=true reasoning client: one
// session per client, cycling through its rounds. Mirrors the SingleSession
// branch of GenerateRequests' reasoning path (generator.go). No per-client cap
// applies (eager's single-session branch does not use perClientCap).
func (s *clientStreamState) produceNextSingleSession() (*sim.Request, int64, bool) {
	for {
		// Drain the (only) session's pending rounds. All rounds are yielded,
		// including non-round-0 rounds of closed-loop sessions, so each counts
		// toward maxRequests at the lazyRequestSource level — matching eager's
		// "produce all rounds, sort+truncate, then filter round-0" sequencing.
		// The round-0-only filter for closed-loop sessions is applied at
		// lazyRequestSource.Next, after the cap-counting heap pop.
		for s.pendingSessionIdx < len(s.pendingSession) {
			req := s.pendingSession[s.pendingSessionIdx]
			s.pendingSessionIdx++
			if req.ArrivalTime >= s.horizon {
				// Rounds are chronological; remaining are past horizon too.
				// Mirrors the `break` in GenerateRequests' reasoning round loop.
				s.pendingSession = nil
				s.pendingSessionIdx = 0
				break
			}
			if s.client.Lifecycle != nil && !isInActiveWindow(req.ArrivalTime, s.client.Lifecycle) {
				continue // skip rounds outside lifecycle windows
			}
			s.perClientSeq++
			return req, req.ArrivalTime, true
		}
		s.pendingSession = nil
		s.pendingSessionIdx = 0

		// Only one session per client.
		if s.singleSessionDone {
			s.exhausted = true
			return nil, 0, false
		}

		iat := s.arrivalSampler.SampleIAT(s.clientRNG)
		if iat == 0 {
			s.exhausted = true
			return nil, 0, false
		}
		startTime := iat
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

		reasoningReqs, err := s.buildSession(startTime)
		if err != nil {
			return nil, 0, false // recordError already set exhausted + lastErr
		}
		s.pendingSession = reasoningReqs
		s.pendingSessionIdx = 0
		// Loop back to yield the first round.
	}
}

// produceNextMultiSession streams a SingleSession=false reasoning client (#1458):
// a traffic source that spawns many independent, OVERLAPPING sessions across the
// horizon. It keeps a min-heap of live (built-but-not-drained) sessions keyed by
// each session's next-round arrival, and yields the client's rounds in arrival
// order so the client feeds the global heap an arrival-monotonic sub-stream
// (BC-2), byte-identical to the set eager's global merge-sort produces (BC-1).
//
// Emit-safety gate: session start times increase strictly (currentTime grows by
// each positive IAT), so any not-yet-built session starts strictly after
// s.currentTime. A live head with arrival <= s.currentTime therefore cannot be
// preceded by any future round and is safe to emit. A head with arrival >
// s.currentTime might be preceded by a future session's round 0, so we build the
// next session first and re-check. Once building is done (msBuildDone), every
// remaining live head is safe to drain. This bounds resident sessions to the
// concurrent working set (Little's law), independent of horizon.
func (s *clientStreamState) produceNextMultiSession() (*sim.Request, int64, bool) {
	for {
		canEmit := s.liveSessions.Len() > 0
		if canEmit && !s.msBuildDone && (*s.liveSessions)[0].head().ArrivalTime > s.currentTime {
			// The earliest live round could still be preceded by a session we
			// have not built yet — build more before emitting it.
			canEmit = false
		}
		if canEmit {
			top := (*s.liveSessions)[0]
			req := top.head()
			if req.ArrivalTime >= s.horizon {
				// Rounds are chronological within a session; this and every
				// later round of this session are past the horizon. Drop the
				// whole session (mirrors eager's per-session round-loop break).
				heap.Pop(s.liveSessions)
				continue
			}
			// Consume this round from its session.
			top.cursor++
			if top.cursor >= len(top.rounds) {
				heap.Pop(s.liveSessions)
			} else {
				heap.Fix(s.liveSessions, 0)
			}
			if s.client.Lifecycle != nil && !isInActiveWindow(req.ArrivalTime, s.client.Lifecycle) {
				continue // suppress rounds outside lifecycle windows
			}
			s.perClientSeq++
			return req, req.ArrivalTime, true
		}
		// Cannot emit yet. Build the next session if any remain.
		if s.msBuildDone {
			// Nothing safe to emit and nothing left to build — with the gate
			// above this means the heap is empty. Client exhausted.
			s.exhausted = true
			return nil, 0, false
		}
		if !s.buildNextSession() && s.exhausted {
			return nil, 0, false // buildNextSession recorded a terminal error
		}
		// Loop: re-evaluate emit eligibility (currentTime may have advanced,
		// or a new session may now be the min head).
	}
}

// buildNextSession replicates one iteration of eager's multi-session build loop
// (GenerateRequests, generator.go): honor the per-client cap and horizon, draw
// one IAT, apply lifecycle gating, build one session, and push it onto the live
// heap. Returns false when no session was built. On the terminal no-build cases
// (horizon reached, cap hit, sampler exhausted, lifecycle past the last window)
// it sets msBuildDone; on a lifecycle-skip it returns false WITHOUT msBuildDone
// (the caller loops and tries the next IAT); on a generator error it records the
// error (sticky-exhausts) so lazyRequestSource.Err() can surface it.
func (s *clientStreamState) buildNextSession() bool {
	// Loop-top guards, in the same order as eager's build loop.
	if s.currentTime >= s.horizon { // `for currentTime < horizon`
		s.msBuildDone = true
		return false
	}
	if s.msClientCap > 0 && s.msReqCount >= s.msClientCap { // perClientCap (R19)
		s.msBuildDone = true
		return false
	}
	iat := s.arrivalSampler.SampleIAT(s.clientRNG)
	if iat == 0 {
		s.msBuildDone = true
		return false
	}
	s.currentTime += iat
	if s.currentTime >= s.horizon {
		s.msBuildDone = true
		return false
	}
	if s.client.Lifecycle != nil && !isInActiveWindow(s.currentTime, s.client.Lifecycle) {
		if s.currentTime >= lastWindowEndUs(s.client.Lifecycle) {
			s.msBuildDone = true
			return false
		}
		return false // lifecycle-skip: try the next IAT (no msBuildDone)
	}
	reasoningReqs, err := s.buildSession(s.currentTime)
	if err != nil {
		return false // recordError already set exhausted + lastErr
	}
	// Count ALL built rounds toward the per-client cap, matching eager's
	// `clientReqCount += len(reasoningReqs)` (before horizon/lifecycle filter).
	s.msReqCount += int64(len(reasoningReqs))
	// Only push sessions that actually produced rounds. A zero-round session is
	// unreachable today (Validate() enforces MaxRounds >= 1, so
	// GenerateReasoningRequests always returns >= 1 round), but pushing an empty
	// session would panic in liveSessionHeap.Less via head()→rounds[0]. Eager
	// is a no-op on an empty slice (it ranges over reasoningReqs); mirror that
	// robustness. The session still counts toward the cap above (matching eager).
	if len(reasoningReqs) > 0 {
		heap.Push(s.liveSessions, &liveSession{rounds: reasoningReqs, cursor: 0, sessionIdx: s.nextSessionIdx})
		s.nextSessionIdx++
	}
	return true
}

// computeSuffixMinStart returns a slice of length len(windows)+1 where
// element k is the minimum StartUs over windows[k:], and element len(windows)
// is math.MaxInt64 (the sentinel read by produceNextTimeVarying's emit gate
// after the last window is built, when windowBuildIdx == len(windows)). This is
// the lower bound on all not-yet-built windows' earliest arrivals: windows are
// built in spec order, so the unbuilt set is always a suffix.
func computeSuffixMinStart(windows []ActiveWindow) []int64 {
	n := len(windows)
	suffix := make([]int64, n+1)
	suffix[n] = math.MaxInt64
	for k := n - 1; k >= 0; k-- {
		suffix[k] = windows[k].StartUs
		if suffix[k+1] < suffix[k] {
			suffix[k] = suffix[k+1]
		}
	}
	return suffix
}

// produceNextTimeVarying streams a time-varying client (#1460): one whose
// lifecycle windows carry per-window parameter overrides. It keeps a min-heap of
// live (built-but-not-drained) window batches keyed by each batch's next-request
// arrival, and yields the client's requests in arrival order so the client feeds
// the global heap an arrival-monotonic sub-stream (BC-2), byte-identical to the
// set eager's global merge-sort produces (BC-1).
//
// Emit-safety gate: windows are built in SPEC order, so the not-yet-built windows
// are exactly the suffix windows[windowBuildIdx:], and each such window's earliest
// possible arrival is its StartUs (first request = StartUs + iats[0], iats[0] >= 0
// after rescale). Therefore suffixMinStart[windowBuildIdx] is a valid lower bound
// on every future request's arrival. A live head with arrival <= that bound cannot
// be preceded by any future window's request and is safe to emit; the `<=` is
// correct on ties because the future window is later in spec order (higher
// windowIdx) and eager's stable sort orders the already-built (lower-windowIdx)
// request first. A head with arrival > the bound might be preceded by a future
// window's first request, so we build the next window first and re-check. Once
// building is done (twBuildDone), every remaining live head is safe to drain.
func (s *clientStreamState) produceNextTimeVarying() (*sim.Request, int64, bool) {
	for {
		canEmit := s.liveWindows.Len() > 0
		if canEmit && !s.twBuildDone &&
			(*s.liveWindows)[0].head().ArrivalTime > s.suffixMinStart[s.windowBuildIdx] {
			// The earliest live request could still be preceded by a window we
			// have not built yet — build more before emitting it.
			canEmit = false
		}
		if canEmit {
			top := (*s.liveWindows)[0]
			req := top.head()
			// Consume this request from its window batch.
			top.cursor++
			if top.cursor >= len(top.batch) {
				heap.Pop(s.liveWindows)
			} else {
				heap.Fix(s.liveWindows, 0)
			}
			s.perClientSeq++
			return req, req.ArrivalTime, true
		}
		// Cannot emit yet. Build the next window if any remain.
		if s.twBuildDone {
			// Nothing safe to emit and nothing left to build — with the gate above
			// this means the heap is empty. Client exhausted.
			s.exhausted = true
			return nil, 0, false
		}
		if !s.buildNextTimeVaryingWindow() && s.exhausted {
			return nil, 0, false // buildNextTimeVaryingWindow recorded a terminal error
		}
		// Loop: re-evaluate emit eligibility (windowBuildIdx advanced, so the
		// suffix-min bound may have relaxed, or a new batch may now be the min head).
	}
}

// buildNextTimeVaryingWindow builds the next lifecycle window (spec order) via the
// reused eager generateRequestsForWindow, mirroring generateTimeVaryingRequests'
// per-window loop: skip windows starting at/after the horizon (no RNG draw), clamp
// the window end to the horizon, generate the window's request batch, stable-sort
// it by arrival (multi-session batches are not arrival-monotonic — eager relies on
// its global sort), and push it onto the live-window heap. Returns false when no
// window was built this call. Sets twBuildDone once every window has been consumed;
// a zero-request or skipped window returns false WITHOUT twBuildDone so the caller
// loops to the next window. On a generator error it records the error
// (sticky-exhausts) so lazyRequestSource.Err() can surface it (BC-6).
func (s *clientStreamState) buildNextTimeVaryingWindow() bool {
	for s.windowBuildIdx < len(s.windows) {
		w := s.windows[s.windowBuildIdx]
		s.windowBuildIdx++
		// Skip windows that start beyond the horizon (mirrors eager's
		// `if window.StartUs >= horizon { continue }` — no RNG draw).
		if w.StartUs >= s.horizon {
			continue
		}
		// Clamp window end to the horizon (mirrors eager's effectiveWindow clamp).
		effWindow := w
		if effWindow.EndUs > s.horizon {
			effWindow.EndUs = s.horizon
		}
		batch, err := generateRequestsForWindow(
			*s.client, effWindow, s.allClients, s.aggregateRate, s.clientRNG, s.prefix,
		)
		if err != nil {
			s.recordError(fmt.Errorf("time-varying window [%d-%d]: %w",
				effWindow.StartUs, effWindow.EndUs, err))
			return false
		}
		if len(batch) == 0 {
			// Zero-request window (rate too low, or every request past the window
			// boundary): eager consumes no clientRNG entropy for it beyond what
			// generateRequestsForWindow already did. Loop to the next window.
			continue
		}
		// Multi-session reasoning appends session rounds in build order, not arrival
		// order (session N+1's round 0 can precede session N's later rounds). Eager
		// fixes this only in its global sort; reproduce that with a per-batch stable
		// sort. No-op for single-shot (currentTime += iat is monotonic) and
		// single-session (rounds are chronological).
		sort.SliceStable(batch, func(i, j int) bool {
			return batch[i].ArrivalTime < batch[j].ArrivalTime
		})
		heap.Push(s.liveWindows, &liveWindow{batch: batch, cursor: 0, windowIdx: s.windowBuildIdx - 1})
		return true
	}
	s.twBuildDone = true
	return false
}

// buildSession generates one reasoning session at startTime and sets the
// per-round Deadline/SLOTargetUs (mirrors GenerateRequests' reasoning path).
// GenerateReasoningRequests seeds/prepends the prefix internally (#1445).
// On error it records the terminal error on the state and returns it; callers
// stop producing so lazyRequestSource.Err() surfaces it to cmd for a Fatalf
// matching eager's abort-on-invalid-spec behavior.
func (s *clientStreamState) buildSession(startTime int64) ([]*sim.Request, error) {
	reasoningReqs, err := GenerateReasoningRequests(
		s.clientRNG, s.client.Reasoning,
		s.inputSampler, s.outputSampler,
		startTime,
		s.client.ID, s.client.TenantID, s.client.SLOClass, s.client.Model, s.client.Adapter,
		s.prefix,
	)
	if err != nil {
		// spec.Validate() does NOT validate ReasoningSpec's distribution fields
		// (e.g. ReasonRatioDist), so this path IS reachable. The user-facing log
		// happens on the Phase 3 pass only (dryRun suppresses it in the pre-pass).
		s.recordError(fmt.Errorf("reasoning session generation at t=%d: %w", startTime, err))
		return nil, err
	}
	for _, req := range reasoningReqs {
		req.Deadline = computeDeadline(req.ArrivalTime, s.client.Timeout, true)
		req.SLOTargetUs = derefInt64(s.client.SLOTargetUs)
	}
	return reasoningReqs, nil
}

// recordError marks the state as exhausted with a terminal error,
// logging it at the Errorf level (client-scoped) unless the state is
// running in dryRun mode — the blueprint pre-pass runs the same
// samplers as Phase 3 and would double-log the same error otherwise.
// lazyRequestSource.Err() aggregates these errors across states so
// cmd/root.go can Fatalf on invalid-spec failures (matching eager).
func (s *clientStreamState) recordError(err error) {
	s.exhausted = true
	s.lastErr = err
	if !s.dryRun {
		logrus.Errorf("[workload] client %q: %v (stream exhausted)", s.client.ID, err)
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

// liveSession is one in-flight session of a multi-session reasoning client
// (#1458). rounds holds that session's pre-computed rounds (already filtered
// for nothing — filtering happens at emit time to mirror eager); cursor points
// at the next round to emit. sessionIdx is the client's monotonic build index,
// used as the deterministic tie-break when two sessions' next rounds share an
// arrival time (matches eager's stable-sort-by-append-order).
type liveSession struct {
	rounds     []*sim.Request
	cursor     int
	sessionIdx int
}

// head returns the session's next round to emit. Callers must ensure cursor is
// in range (the heap only holds sessions with a pending round).
func (ls *liveSession) head() *sim.Request { return ls.rounds[ls.cursor] }

// liveSessionHeap orders live sessions by (next-round arrival, sessionIdx).
// The arrival key reproduces the global sort; the sessionIdx key reproduces
// eager's within-client stable-sort tie-break (earlier-built session first).
type liveSessionHeap []*liveSession

func (h liveSessionHeap) Len() int { return len(h) }
func (h liveSessionHeap) Less(i, j int) bool {
	a, b := h[i].head(), h[j].head()
	if a.ArrivalTime != b.ArrivalTime {
		return a.ArrivalTime < b.ArrivalTime
	}
	return h[i].sessionIdx < h[j].sessionIdx
}
func (h liveSessionHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *liveSessionHeap) Push(x interface{}) { *h = append(*h, x.(*liveSession)) }
func (h *liveSessionHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// liveWindow is one built-but-not-fully-drained lifecycle window batch of a
// time-varying client (#1460). batch holds that window's requests, produced by
// the reused eager generateRequestsForWindow and stable-sorted by ArrivalTime
// (multi-session reasoning within a window is NOT arrival-monotonic — eager only
// fixes this in its global sort — so the per-batch sort is load-bearing). cursor
// points at the next request to emit. windowIdx is the client's window spec index,
// used as the deterministic tie-break when two windows' next requests share an
// arrival time (matches eager's stable-sort-by-concat-order: earlier spec window
// first).
type liveWindow struct {
	batch     []*sim.Request
	cursor    int
	windowIdx int
}

// head returns the window's next request to emit. Callers must ensure cursor is
// in range (the heap only holds windows with a pending request).
func (lw *liveWindow) head() *sim.Request { return lw.batch[lw.cursor] }

// liveWindowHeap orders live windows by (next-request arrival, windowIdx). The
// arrival key reproduces eager's global sort; the windowIdx key reproduces
// eager's within-client stable-sort tie-break (earlier-spec window first).
type liveWindowHeap []*liveWindow

func (h liveWindowHeap) Len() int { return len(h) }
func (h liveWindowHeap) Less(i, j int) bool {
	a, b := h[i].head(), h[j].head()
	if a.ArrivalTime != b.ArrivalTime {
		return a.ArrivalTime < b.ArrivalTime
	}
	return h[i].windowIdx < h[j].windowIdx
}
func (h liveWindowHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *liveWindowHeap) Push(x interface{}) { *h = append(*h, x.(*liveWindow)) }
func (h *liveWindowHeap) Pop() interface{} {
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
	// states retains a reference to every per-client streaming state
	// (populated at construction) so Err() can surface a sampler failure
	// that terminated a state after it stopped being on the heap. Without
	// this, once a state exhausted its heap entry got dropped and the
	// error would be unreachable from cmd.
	states []*clientStreamState
}

// Err returns the first terminal sampler / generator error recorded on any
// per-client state after the cluster has finished draining the source
// (i.e. after cluster.Run returns). Callers MUST invoke Err() post-Run
// and Fatalf on a non-nil value to match the eager path's
// abort-on-invalid-spec behavior (issue #1441, PR #1453 review round 3).
// Returns nil when every state exhausted cleanly.
//
// Scan order is per-client-index so the surfaced error is deterministic
// across runs (any parallel drainers would still see the first client's
// error first).
func (l *lazyRequestSource) Err() error {
	for _, s := range l.states {
		if s.lastErr != nil {
			return fmt.Errorf("client %q: %w", s.client.ID, s.lastErr)
		}
	}
	return nil
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
//     GenerateWorkload's round-0-only filter for closed-loop sessions).
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
		// GenerateWorkload's post-filter sequential-ID renumbering pass
		// (where it reassigns `req.ID = fmt.Sprintf("request_%d", i)`
		// over the filtered + sorted slice).
		e.req.ID = fmt.Sprintf("request_%d", l.yielded)
		l.yielded++
		return e.req, true
	}
}

// clientPrep captures the per-client information collected during the
// construction prelude of GenerateWorkloadLazy. clientSeed is drawn
// from workloadRNG in allClients order before any client begins
// producing requests — matching the eager generator's
// `clientSeed := workloadRNG.Int63()` draw at the top of its per-client
// loop in GenerateRequests.
type clientPrep struct {
	idx        int
	client     *ClientSpec
	clientSeed int64
	rate       float64
	prefix     []sim.TokenID

	// Time-varying context (#1460), populated only by the TV prelude in
	// generateTimeVaryingWorkloadLazy. When isTimeVarying is set,
	// buildClientStreamState constructs a time-varying state (per-window
	// generation via the reused generateRequestsForWindow) instead of the
	// single-shot/reasoning state; rate is unused (per-window proportional
	// allocation uses allClients + aggregateRate instead). allClients is a
	// shared read-only slice header (computeProportionalRate reads it).
	isTimeVarying bool
	allClients    []ClientSpec
	aggregateRate float64
}

// GenerateWorkloadLazy mirrors GenerateWorkload's setup but returns a
// streaming RequestSource instead of materializing the request slice.
// Memory consumption while the cluster drains the source is
// O(num_clients + max_session_rounds), independent of total request count.
//
// Returns:
//   - source: implements cluster.RequestSource via structural typing.
//   - sessions: deterministic session blueprints for closed-loop reasoning AND
//     concurrency clients (#1459), in the same order as GenerateWorkload.
//   - followUpBudget: the shared concurrencyFollowUpBudget value — -1 when
//     unbounded (maxRequests<=0) or there are no concurrency users; >=0 (the cap
//     on follow-ups) for concurrency specs under a finite maxRequests (#1459).
//   - err: a real spec/validation failure. As of #1460 there is NO
//     ErrLazyUnsupported* sentinel — every spec class (single-shot, single- and
//     multi-session reasoning #1458, concurrency #1459, time-varying #1460) is
//     streamed. Time-varying specs dispatch to generateTimeVaryingWorkloadLazy.
//
// Determinism: same seed produces the same source byte-identically to
// GenerateWorkload's request stream (BC-3).
func GenerateWorkloadLazy(spec *WorkloadSpec, horizon int64, maxRequests int64) (
	*lazyRequestSource, []SessionBlueprint, int64, error) {

	// NOTE: the horizon<=0 check MUST come before the maxRequests<0 check to match
	// eager's guard order exactly (GenerateRequests checks horizon<=0 first,
	// returning before its maxRequests<0 check — generator.go). Reversing them
	// would make lazy reject a horizon<=0 && maxRequests<0 spec that eager accepts.
	if horizon <= 0 {
		// Empty workload: return an immediately-exhausted source — UNLESS the
		// spec has concurrency clients. Eager's concurrency seed loop is
		// horizon-independent: GenerateRequests returns nil at horizon<=0 (before
		// validateAndExpandSpec AND before its maxRequests<0 check), but
		// GenerateWorkload still emits the seed set (treating maxRequests<=0 as
		// unbounded). To match (BC-1/INV-6) we must too. Concurrency is a
		// spec.Clients-only field (cohorts never carry it), so this needs no expansion.
		if !specHasConcurrencyClient(spec) {
			return &lazyRequestSource{h: &heapByArrival{}, maxRequests: maxRequests}, nil, -1, nil
		}
		// Concurrency at horizon<=0: mirror eager's zero-horizon sequence, which
		// does NOT run validateAndExpandSpec/UpgradeV1ToV2 and does NOT reject a
		// negative maxRequests (the seed cap's `maxRequests > 0` guard treats it as
		// unbounded). Assemble allClients from the raw spec, emit only the
		// (horizon-independent) concurrency seeds with keptOpen=0, and return.
		// See GenerateWorkloadLazy's "Validation symmetry" note in the plan (#1459).
		return generateConcurrencyOnlyLazyAtZeroHorizon(spec, horizon, maxRequests)
	}
	if maxRequests < 0 {
		return nil, nil, 0, fmt.Errorf("maxRequests must be non-negative, got %d", maxRequests)
	}

	// Shared spec prelude (mutual-exclusion, inference-perf expansion,
	// servegen load, v1→v2 upgrade, Validate). Mutates spec.Clients in place.
	if err := validateAndExpandSpec(spec); err != nil {
		return nil, nil, 0, err
	}

	// Build working client list (mirrors the same allClients assembly
	// in GenerateRequests: copy spec.Clients, then append cohort-
	// expanded clients).
	allClients := append([]ClientSpec{}, spec.Clients...)
	if len(spec.Cohorts) > 0 {
		allClients = append(allClients, ExpandCohorts(spec.Cohorts, spec.Seed)...)
	}

	// Time-varying dispatch. Clients with per-window parameter overrides
	// (trace_rate/arrival/input_distribution/output_distribution) take a distinct
	// generation path in eager (generateTimeVaryingRequests); the lazy path mirrors
	// it in generateTimeVaryingWorkloadLazy. As of #1460 there is NO remaining
	// unsupported class — multi-session reasoning (#1458), concurrency clients
	// (#1459), and time-varying workloads (#1460) are all streamed.
	if hasPerWindowParameters(allClients) {
		return generateTimeVaryingWorkloadLazy(spec, horizon, maxRequests, allClients)
	}

	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(spec.Seed))
	workloadRNG := rng.ForSubsystem(sim.SubsystemWorkloadGen)
	clientRates := normalizeRateFractions(allClients, spec.AggregateRate)
	prefixes := generatePrefixTokens(allClients, workloadRNG)

	// Phase 1: prelude — draw clientSeeds in allClients order, mirroring the
	// eager loop's draw order in GenerateRequests' per-client loop.
	// Zero-rate clients are skipped BEFORE the workloadRNG.Int63() draw,
	// matching the eager loop's `if clientRate <= 0 { continue }` guard.
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

	// Phases 2–4 (blueprint pre-pass, streaming states, concurrency seeds) are
	// identical for the time-varying and non-time-varying paths — they operate
	// purely on preps/prefixes/allClients — so they live in one shared helper.
	return assembleLazySourceFromPreps(preps, prefixes, allClients, spec.Seed, horizon, maxRequests)
}

// assembleLazySourceFromPreps runs the shared Phases 2–4 of lazy workload
// construction (blueprint pre-pass, per-client streaming states, concurrency
// seeds) given the Phase-1 preps. Both GenerateWorkloadLazy's non-time-varying
// path and generateTimeVaryingWorkloadLazy call it — only the Phase-1 prelude
// (which clients get a clientSeed, and whether each prep is time-varying)
// differs between the two. buildClientStreamState branches on prep.isTimeVarying,
// so the pre-pass and streaming pass transparently handle both kinds.
func assembleLazySourceFromPreps(
	preps []clientPrep,
	prefixes map[string][]sim.TokenID,
	allClients []ClientSpec,
	specSeed int64,
	horizon int64,
	maxRequests int64,
) (*lazyRequestSource, []SessionBlueprint, int64, error) {
	// Phase 2: blueprint pre-pass (closed-loop reasoning clients only).
	//
	// Mirrors GenerateWorkload's blueprint construction at
	// GenerateWorkload's closed-loop blueprint construction loop — which scans the truncated `reqs` slice for
	// round-0 SessionIDs per closed-loop client, sorts them, and draws
	// one `blueprintRNG.Int63()` per surviving session.
	//
	// To match this RNG-draw budget exactly, the lazy pre-pass simulates
	// the full streaming `Next()` loop with separate cloned per-client
	// states (RNGs re-seeded from the same clientSeeds — same sequence)
	// up to the same `maxRequests` cap that Phase 3 will impose. This
	// way we discover the EXACT set of round-0 emissions that survive
	// the cap, and only draw `blueprintRNG` for those sessions.
	//
	// Without `maxRequests` awareness, a binding cap that drops a later
	// session's round-0 would still see that session enumerated here,
	// consume an extra `blueprintRNG.Int63()`, and shift every
	// subsequent blueprint RNG seed — breaking INV-6 byte-identity and
	// INV-13 run/replay parity for closed-loop reasoning under a tight
	// cap. (Bug found in PR #1453 self-review.)
	survivingPerClient, keptOpen, err := enumerateSurvivingSessionsPerClient(preps, prefixes, horizon, maxRequests)
	if err != nil {
		return nil, nil, 0, err
	}
	blueprintRNG := rand.New(rand.NewSource(specSeed + 7919))
	var sessions []SessionBlueprint
	for _, p := range preps {
		if !isClosedLoop(p.client) || p.client.Reasoning == nil || p.client.Reasoning.MultiTurn == nil {
			continue
		}
		sessIDs := survivingPerClient[p.idx]
		if len(sessIDs) == 0 {
			continue // no round-0 of this client's sessions survived the cap
		}
		// Sort session IDs to match GenerateWorkload's deterministic
		// blueprint-construction order (sort.Strings on the map of seen IDs).
		sort.Strings(sessIDs)
		// Build samplers once per client (mirrors the eager blueprint loop).
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
		var prefixTokens []sim.TokenID
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
				Adapter:       p.client.Adapter,
				SLOTargetUs:   derefInt64(p.client.SLOTargetUs),
			})
		}
	}

	// Phase 3: build the streaming states with fresh RNGs seeded from the
	// same clientSeeds. The pre-pass RNGs ran to completion and are
	// discarded; Phase 3's RNGs start anew but, because
	// `rand.New(rand.NewSource(s))` is deterministic, both produce the
	// SAME starting sequence — giving the streaming pass byte-identical
	// emissions to what the pre-pass simulated.
	h := &heapByArrival{}
	heap.Init(h)
	states := make([]*clientStreamState, 0, len(preps))
	for _, p := range preps {
		state, err := buildClientStreamState(p, horizon, maxRequests)
		if err != nil {
			return nil, nil, 0, err
		}
		states = append(states, state)
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

	// Phase 4: concurrency seed phase (#1459). Concurrency clients have
	// RateFraction == 0, so they never appear in `preps` (Phase 1 skips them,
	// exactly as GenerateRequests does), and `keptOpen` — the number of open-loop
	// requests the source will emit under the cap — is the pop count from the
	// Phase 2 dry-run. Push the seeds as individual heap entries (see the shared
	// helper) so the global merge reproduces eager's stable-sort-by-arrival.
	sessions, followUpBudget, err := appendConcurrencySeedsToHeap(
		h, allClients, prefixes, specSeed, horizon, maxRequests, keptOpen, sessions)
	if err != nil {
		return nil, nil, 0, err
	}
	return &lazyRequestSource{h: h, maxRequests: maxRequests, states: states}, sessions, followUpBudget, nil
}

// generateTimeVaryingWorkloadLazy is the lazy counterpart of eager's
// generateTimeVaryingRequests (generator.go): it streams a workload whose clients
// carry per-window parameter overrides. It mirrors that function's RNG-draw order
// exactly — generatePrefixTokens on the workload RNG, then one clientSeed draw per
// LIFECYCLE client (in allClients order; clients WITHOUT lifecycle windows are
// warned and skipped WITHOUT a clientSeed draw, matching eager's
// mixed-always-on-and-windowed handling) — then delegates Phases 2–4 to the shared
// assembleLazySourceFromPreps. Per-client streaming is handled by
// produceNextTimeVarying (see buildClientStreamState's time-varying branch), which
// reuses the eager generateRequestsForWindow one window at a time.
//
// Determinism (INV-6): the only lazy-authored RNG draws are the prefix generation
// and the per-lifecycle-client clientSeed — both identical to eager's TV path. The
// per-window IAT/content draws happen inside the reused generateRequestsForWindow,
// so they cannot diverge.
func generateTimeVaryingWorkloadLazy(
	spec *WorkloadSpec, horizon int64, maxRequests int64, allClients []ClientSpec,
) (*lazyRequestSource, []SessionBlueprint, int64, error) {
	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(spec.Seed))
	workloadRNG := rng.ForSubsystem(sim.SubsystemWorkloadGen)
	// generatePrefixTokens draws first — same as eager's TV path (generator.go),
	// which calls it inside generateTimeVaryingRequests before the per-client loop.
	prefixes := generatePrefixTokens(allClients, workloadRNG)

	// Phase 1 (time-varying prelude): draw clientSeeds in allClients order. Unlike
	// the non-TV prelude (gated on clientRate > 0), eager's TV path draws a
	// clientSeed for every client WITH lifecycle windows and skips windowless
	// clients with a warning BEFORE the draw (generateTimeVaryingRequests). We
	// mirror that gating exactly so the workloadRNG draw sequence matches.
	preps := make([]clientPrep, 0, len(allClients))
	for i := range allClients {
		client := &allClients[i]
		if client.Lifecycle == nil || len(client.Lifecycle.Windows) == 0 {
			// Windowless clients generate nothing on the TV path (mixed always-on +
			// windowed clients are not supported). Warn and skip WITHOUT a clientSeed
			// draw, matching generateTimeVaryingRequests.
			logrus.Warnf("generateTimeVaryingWorkloadLazy: client %q has no lifecycle windows and will generate no requests (mixed always-on + windowed clients are not supported)", client.ID)
			continue
		}
		clientSeed := workloadRNG.Int63()
		preps = append(preps, clientPrep{
			idx:           i,
			client:        client,
			clientSeed:    clientSeed,
			prefix:        prefixes[client.PrefixGroup],
			isTimeVarying: true,
			allClients:    allClients,
			aggregateRate: spec.AggregateRate,
		})
	}

	return assembleLazySourceFromPreps(preps, prefixes, allClients, spec.Seed, horizon, maxRequests)
}

// exhaustedSentinelState is a single shared always-exhausted clientStreamState
// carried by concurrency seed heap entries (#1459). It makes lazyRequestSource.Next
// panic-free at both `e.state` dereference sites with zero Next() changes:
//   - produceNext() returns (nil,0,false) immediately (sticky exhausted) — no
//     successor is re-pushed for a seed (the seed set is fixed and fully pushed).
//   - isClosedLoop is false (zero value) — the intermediate-round suppression
//     `e.state.isClosedLoop && e.req.RoundIndex != 0` is skipped, so the seed
//     (RoundIndex == 0 anyway) emits directly.
//
// It holds no per-seed state (the seed is the heapEntry's req), so one instance
// is safely shared across all seed entries; the exhausted path performs no writes.
var exhaustedSentinelState = &clientStreamState{exhausted: true}

// appendConcurrencySeedsToHeap generates the concurrency seeds + blueprints via
// the shared generateConcurrencySeedsAndBlueprints helper and pushes each seed
// onto the global arrival heap as its own entry keyed
// (arrival, len(allClients), generationIndex). len(allClients) is strictly
// greater than every real client/cohort index (allClients is already fully
// cohort-expanded here), so at equal arrival open-loop pops before seeds; the
// generation index orders seeds among themselves — reproducing eager's
// sort.SliceStable over [open-loop…, seed_0, seed_1, …] exactly.
//
// It returns the updated blueprint slice (concurrency blueprints appended after
// any closed-loop reasoning blueprints — a concurrency spec has none, matching
// eager) and the follow-up budget (shared formula with eager). This single site
// is used by both the normal path and the horizon<=0 concurrency path so the
// heap-wiring cannot drift.
func appendConcurrencySeedsToHeap(
	h *heapByArrival,
	allClients []ClientSpec,
	prefixes map[string][]sim.TokenID,
	specSeed int64,
	horizon int64,
	maxRequests int64,
	keptOpen int64,
	sessions []SessionBlueprint,
) ([]SessionBlueprint, int64, error) {
	seeds, blueprints, totalUsers, err :=
		generateConcurrencySeedsAndBlueprints(allClients, prefixes, specSeed, horizon, maxRequests, keptOpen)
	if err != nil {
		return nil, 0, err
	}
	for g, seed := range seeds {
		heap.Push(h, heapEntry{
			arrivalUs:    seed.ArrivalTime,
			clientIdx:    len(allClients),
			perClientSeq: int64(g),
			req:          seed,
			state:        exhaustedSentinelState,
		})
	}
	sessions = append(sessions, blueprints...)
	// keptOpen open-loop requests + len(seeds) seeds will be emitted (the
	// no-displacement invariant guarantees the popped-cap is non-binding
	// whenever a seed exists), so the emitted total is keptOpen+len(seeds).
	budget := concurrencyFollowUpBudget(maxRequests, keptOpen+int64(len(seeds)), totalUsers)
	return sessions, budget, nil
}

// generateConcurrencyOnlyLazyAtZeroHorizon mirrors eager's horizon<=0 behavior
// for specs with concurrency clients (#1459): eager emits the horizon-independent
// concurrency seed set even at horizon<=0, WITHOUT running
// validateAndExpandSpec/UpgradeV1ToV2 (GenerateRequests returns first). We match
// that exactly — no validate/expand — assembling allClients from the raw spec,
// with keptOpen=0 (no open-loop requests exist at horizon<=0).
func generateConcurrencyOnlyLazyAtZeroHorizon(spec *WorkloadSpec, horizon int64, maxRequests int64) (
	*lazyRequestSource, []SessionBlueprint, int64, error) {
	allClients := append([]ClientSpec{}, spec.Clients...)
	if len(spec.Cohorts) > 0 {
		allClients = append(allClients, ExpandCohorts(spec.Cohorts, spec.Seed)...)
	}
	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(spec.Seed))
	workloadRNG := rng.ForSubsystem(sim.SubsystemWorkloadGen)
	prefixes := generatePrefixTokens(allClients, workloadRNG)

	h := &heapByArrival{}
	heap.Init(h)
	sessions, followUpBudget, err := appendConcurrencySeedsToHeap(
		h, allClients, prefixes, spec.Seed, horizon, maxRequests, 0, nil)
	if err != nil {
		return nil, nil, 0, err
	}
	return &lazyRequestSource{h: h, maxRequests: maxRequests}, sessions, followUpBudget, nil
}

// specHasConcurrencyClient reports whether spec.Clients contains a concurrency
// client (Concurrency > 0). Concurrency is a spec.Clients-only field; cohorts
// never carry it, so no expansion is needed. Used by the horizon<=0 fast path.
func specHasConcurrencyClient(spec *WorkloadSpec) bool {
	for i := range spec.Clients {
		if spec.Clients[i].Concurrency > 0 {
			return true
		}
	}
	return false
}

// buildClientStreamState constructs one client's streaming state. The
// RNG is seeded from p.clientSeed; sampler construction mirrors the
// eager generator's per-client sampler construction (the block
// immediately after `clientRNG := newRandFromSeed(clientSeed)` in
// GenerateRequests), including the CustomSamplerFactory
// sub-RNG draw.
//
// maxRequests seeds the multi-session per-client build cap (2*maxRequests),
// mirroring eager's perClientCap. It is only consulted by multi-session
// reasoning clients — single-shot and single-session lazy rely on the global
// popped-cap in lazyRequestSource.Next (build order == arrival order there),
// but multi-session build order != arrival order, so the per-client cap must
// bound building exactly as eager does to keep byte-identity (#1458).
func buildClientStreamState(p clientPrep, horizon int64, maxRequests int64) (*clientStreamState, error) {
	clientRNG := newRandFromSeed(p.clientSeed)

	// Time-varying branch (#1460): the eager generateTimeVaryingRequests draws
	// clientSeed then passes clientRNG STRAIGHT to generateRequestsForWindow with
	// NO intervening draw — no client-level NewArrivalSampler and no
	// CustomSamplerFactory sub-RNG (per-window samplers are built inside
	// generateRequestsForWindow). Constructing them here would consume an extra
	// clientRNG.Int63() and break byte-identity (INV-6). So the TV state carries
	// no client-level samplers; it only needs the per-window context.
	if p.isTimeVarying {
		windows := p.client.Lifecycle.Windows
		state := &clientStreamState{
			clientIdx:      p.idx,
			client:         p.client,
			clientRNG:      clientRNG,
			prefix:         p.prefix,
			horizon:        horizon,
			isTimeVarying:  true,
			allClients:     p.allClients,
			aggregateRate:  p.aggregateRate,
			windows:        windows,
			liveWindows:    &liveWindowHeap{},
			suffixMinStart: computeSuffixMinStart(windows),
			// isClosedLoop drives Next()'s round-0 suppression for closed-loop
			// reasoning windows; generateRequestsForWindow sets SessionID/RoundIndex.
			isClosedLoop: isClosedLoop(p.client),
		}
		return state, nil
	}

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
		if !state.isSingleSession {
			state.liveSessions = &liveSessionHeap{}
			// Per-client build cap = 2*maxRequests (eager's perClientCap, with
			// the same int64-overflow guard). 0 means unbounded (maxRequests<=0).
			if maxRequests > 0 {
				perClientCap := 2 * maxRequests
				if perClientCap < maxRequests { // overflow → treat as unbounded
					perClientCap = math.MaxInt64
				}
				state.msClientCap = perClientCap
			}
		}
	}
	return state, nil
}

// enumerateSurvivingSessionsPerClient simulates the streaming source's
// global heap-pop order up to maxRequests pops and returns (1) for each
// closed-loop reasoning client (keyed by allClients index), the set of
// SessionIDs whose round-0 emission survived the cap, and (2) keptOpen —
// the total number of pops (= the number of open-loop / non-concurrency
// requests the source will emit under the cap). This mirrors the eager flow's
// "produce all rounds, sort+truncate to maxRequests, then scan the truncated
// slice for SessionIDs per closed-loop client" sequence.
//
// keptOpen == eager's len(round0Only): concurrency clients have RateFraction == 0
// and are never in `preps`, so the dry-run heap holds only open-loop states and
// its pop count is exactly min(genOpen, maxRequests). The concurrency seed phase
// (#1459) consumes keptOpen to reproduce eager's seed cap
// (alreadyKept + len(seeds) >= maxRequests).
//
// LOAD-BEARING INVARIANT (INV-6): keptOpen counts ALL pops, including any
// intermediate closed-loop reasoning rounds (RoundIndex > 0), whereas eager's
// alreadyKept = len(round0Only) EXCLUDES those intermediate rounds. These two
// counts are equal only because a spec can never contain both concurrency clients
// and multi-turn/reasoning clients (spec.Validate hard-errors that mix — see
// spec.go, `hasConcurrency && hasMultiTurn`). So whenever concurrency seeds are
// being generated there are provably zero intermediate closed-loop rounds, and
// keptOpen == len(round0Only). If that mutual-exclusion validation is ever
// relaxed, this equality breaks and the seed cap would bind earlier in lazy than
// eager — revisit the concurrency seed phase here and in GenerateWorkload.
//
// Uses CLONED per-client states with RNGs re-seeded from the same
// clientSeeds, so the simulation does not advance the Phase 3
// streaming-pass RNGs. The dry-run yields rounds and counts toward
// the cap exactly as `lazyRequestSource.Next` will — including
// intermediate closed-loop rounds (`RoundIndex > 0`), which pop and
// count but do not yield to the cluster (and are not recorded here).
//
// Memory: bounded by the per-session pending slice inside each cloned
// state (one session at a time, MaxRounds entries).
func enumerateSurvivingSessionsPerClient(
	preps []clientPrep,
	prefixes map[string][]sim.TokenID,
	horizon int64,
	maxRequests int64,
) (map[int][]string, int64, error) {
	// Clone per-client states (fresh RNGs from the same clientSeed).
	// dryRun=true so sampler-error paths don't user-log twice (the Phase 3
	// pass runs the same samplers and is authoritative for user feedback).
	cloneStates := make([]*clientStreamState, 0, len(preps))
	for _, p := range preps {
		// Mirror the prefix-binding rule of Phase 3 so the clone consumes
		// identical RNG draws for prefix-prepending; p.prefix is already
		// the per-prefix-group slice resolved in Phase 1. maxRequests is
		// threaded so the clone's multi-session per-client cap matches Phase 3.
		state, err := buildClientStreamState(p, horizon, maxRequests)
		if err != nil {
			return nil, 0, fmt.Errorf("client %q: %w", p.client.ID, err)
		}
		state.dryRun = true
		cloneStates = append(cloneStates, state)
	}
	// Build the clone heap with each state's first candidate, matching
	// Phase 3's heap construction.
	h := &heapByArrival{}
	heap.Init(h)
	for _, state := range cloneStates {
		if first, t, ok := state.produceNext(); ok {
			heap.Push(h, heapEntry{
				arrivalUs:    t,
				clientIdx:    state.clientIdx,
				perClientSeq: state.perClientSeq,
				req:          first,
				state:        state,
			})
		}
	}
	// Dry-run the streaming pop loop with the SAME stopping rule as
	// lazyRequestSource.Next: stop at popped >= maxRequests (when > 0)
	// or when the heap is empty.
	surviving := make(map[int][]string)
	seen := make(map[int]map[string]bool)
	var popped int64
	for h.Len() > 0 {
		if maxRequests > 0 && popped >= maxRequests {
			break
		}
		e := heap.Pop(h).(heapEntry)
		// Re-push state's next candidate, matching Next().
		if nextReq, t, ok := e.state.produceNext(); ok {
			heap.Push(h, heapEntry{
				arrivalUs:    t,
				clientIdx:    e.state.clientIdx,
				perClientSeq: e.state.perClientSeq,
				req:          nextReq,
				state:        e.state,
			})
		}
		popped++
		// Only round-0 entries are observed by GenerateWorkload's
		// blueprint-building loop (it filters round-0 only when keying
		// the SessionID lookup). Intermediate rounds advance state and
		// count toward the cap but produce no blueprint.
		if e.req.RoundIndex != 0 || e.req.SessionID == "" {
			continue
		}
		idx := e.state.clientIdx
		if seen[idx] == nil {
			seen[idx] = make(map[string]bool)
		}
		if !seen[idx][e.req.SessionID] {
			seen[idx][e.req.SessionID] = true
			surviving[idx] = append(surviving[idx], e.req.SessionID)
		}
	}
	// popped is the number of open-loop / closed-loop-round-0-and-intermediate
	// requests emitted under the cap == eager's len(reqs) after truncation.
	// (Concurrency clients are absent from preps, so they never count here.)
	return surviving, popped, nil
}

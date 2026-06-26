// Package workload provides session management for closed-loop multi-turn workloads.
// The SessionManager tracks active sessions and generates follow-up rounds
// on request completion, enabling realistic multi-turn arrival patterns.

package workload

import (
	"fmt"
	"math/rand"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
)

// sessionState tracks a session's lifecycle.
type sessionState string

const (
	sessionActive             sessionState = "active"
	sessionCompleted          sessionState = "completed"
	sessionCancelled          sessionState = "cancelled"
	sessionHorizonInterrupted sessionState = "horizon_interrupted"
	sessionBudgetExhausted    sessionState = "budget_exhausted"
)

// SessionBlueprint describes a session's full shape. Created during workload generation,
// immutable after creation. Each session has its own deterministic RNG (INV-6).
type SessionBlueprint struct {
	SessionID        string
	ClientID         string
	MaxRounds        int
	UnlimitedRounds  bool   // when true, session continues past MaxRounds until budget/horizon/timeout/drop
	ContextGrowth    string // "accumulate" or ""
	ThinkTimeUs      int64
	Timeout          *int64 // per-request timeout from ClientSpec (nil = default 300s)
	Horizon          int64  // simulation horizon for BC-19 guard
	InputSampler     LengthSampler
	OutputSampler    LengthSampler
	RNG              *rand.Rand    // per-session, seeded deterministically from client RNG
	ThinkTimeSampler LengthSampler // optional: per-round think time in µs; nil = use constant ThinkTimeUs
	Prefix           []int         // shared system prompt tokens
	TenantID         string
	SLOClass         string
	Model            string
	SLOTargetUs      int64 // per-request SLO TTFT target in µs; 0 = no target
}

// activeSession tracks mutable per-session lifecycle state.
type activeSession struct {
	blueprint    *SessionBlueprint
	currentRound int
	// buf holds the session's growable shared token buffer for accumulate mode.
	// Layout: [prefix | r0_conversation | r0_output | r1_conversation | r1_output | ... | rN_newInput].
	// Each follow-up round's Request.InputTokens is a flat slice into this
	// buffer's underlying array — replaces the legacy O(R²) eager copy (#1445).
	// nil when ContextGrowth != "accumulate".
	buf    *sessionTokenBuffer
	seeded bool // false until first OnComplete seeds prefix + round-0 conversation
	state  sessionState
}

// SessionManager tracks active sessions and generates follow-up rounds on completion.
// Single-threaded: assumes invocation only from the DES event loop.
type SessionManager struct {
	sessions       map[string]*activeSession
	idCounter      int64 // monotonic counter for follow-up request IDs
	followUpBudget int64 // max follow-ups to generate (only meaningful when budgetEnabled)
	followUpCount  int64 // follow-ups generated so far
	budgetEnabled  bool  // true once SetFollowUpBudget has been called
}

// NewSessionManager creates a SessionManager from pre-generated session blueprints.
// Panics if any blueprint has MaxRounds < 1.
func NewSessionManager(blueprints []SessionBlueprint) *SessionManager {
	sm := &SessionManager{sessions: make(map[string]*activeSession, len(blueprints))}
	for i := range blueprints {
		bp := &blueprints[i]
		if bp.MaxRounds < 1 && !bp.UnlimitedRounds {
			panic(fmt.Sprintf("NewSessionManager: session %s has MaxRounds=%d, must be >= 1", bp.SessionID, bp.MaxRounds))
		}
		var buf *sessionTokenBuffer
		if bp.ContextGrowth == "accumulate" {
			buf = newSessionTokenBuffer()
		}
		sm.sessions[bp.SessionID] = &activeSession{
			blueprint: bp,
			buf:       buf,
			state:     sessionActive,
		}
	}
	return sm
}

// SetFollowUpBudget sets a global cap on the number of follow-up requests
// the SessionManager will generate. Zero means no follow-ups allowed.
// The budget is only active once this method is called; the default
// (budgetEnabled=false) means unlimited follow-ups.
func (sm *SessionManager) SetFollowUpBudget(budget int64) {
	sm.followUpBudget = budget
	sm.budgetEnabled = true
}

// OnComplete is called when a request reaches a terminal state. It determines
// whether to generate a follow-up round or terminate the session.
//
// Returns follow-up requests to inject, or nil.
// Session termination paths: timeout (cancelled), dropped (cancelled),
// length-capped (continues), final round (completed), past horizon (horizon-interrupted).
func (sm *SessionManager) OnComplete(req *sim.Request, tick int64) []*sim.Request {
	if req.SessionID == "" {
		return nil // non-session request
	}
	sess, ok := sm.sessions[req.SessionID]
	if !ok {
		logrus.Warnf("SessionManager.OnComplete: request %s has SessionID %q not found in sessions — possible blueprint mismatch",
			req.ID, req.SessionID)
		return nil
	}
	if sess.state != sessionActive {
		return nil // session already terminal (duplicate completion guard)
	}

	// Session cancellation on timeout (BC-7)
	if req.State == sim.StateTimedOut {
		sess.state = sessionCancelled
		return nil
	}

	// Dropped-unservable follow-up cancels session (BC-17).
	// Dropped requests still have State == StateQueued (never transitioned by the
	// enqueue guards). This detection is safe because OnRequestDone is only invoked at:
	//   1. processCompletions (req.State == StateCompleted) — handled above
	//   2. TimeoutEvent.Execute (req.State == StateTimedOut) — handled above
	//   3. EnqueueRequest guard drops (req.State == StateQueued) — handled here
	//   4. detectDecodeCompletions (cluster.go) — req.State set to StateCompleted
	//      before invocation; not a drop path (issue #884)
	// A legitimately queued request never triggers this callback.
	// If a future code path invokes OnRequestDone for a queued request that is
	// NOT dropped, this detection would incorrectly cancel the session. Review
	// all OnRequestDone call sites when adding new invocation points.
	if req.State == sim.StateQueued {
		sess.state = sessionCancelled
		return nil
	}

	// Length-capped: continues session (BC-16) — State is StateCompleted

	// Final round check
	if !sess.blueprint.UnlimitedRounds && sess.currentRound >= sess.blueprint.MaxRounds-1 {
		sess.state = sessionCompleted
		return nil
	}

	// Budget check: stop generating follow-ups once global budget is exhausted
	if sm.budgetEnabled && sm.followUpCount >= sm.followUpBudget {
		sess.state = sessionBudgetExhausted
		return nil
	}

	bp := sess.blueprint

	// Horizon guard (BC-19): don't generate follow-ups past horizon
	var thinkTime int64
	if bp.ThinkTimeSampler != nil {
		thinkTime = int64(bp.ThinkTimeSampler.Sample(bp.RNG))
	} else {
		thinkTime = bp.ThinkTimeUs
	}
	arrivalTime := tick + thinkTime
	if arrivalTime > bp.Horizon {
		sess.state = sessionHorizonInterrupted
		return nil
	}

	// Generate round N+1
	inputLen := bp.InputSampler.Sample(bp.RNG)
	outputLen := bp.OutputSampler.Sample(bp.RNG)
	newInputTokens := sim.GenerateRandomTokenIDs(bp.RNG, inputLen)
	outputTokens := sim.GenerateRandomTokenIDs(bp.RNG, outputLen)

	// Context accumulation (BC-8): use ACTUAL generated output, not oracle OutputTokens.
	// For length-capped requests, ProgressIndex - len(InputTokens) gives actual output count.
	actualOutputLen := max(int(req.ProgressIndex)-int(req.InputLen()), 0)

	var inputTokens []int
	if bp.ContextGrowth == "accumulate" {
		// Shared-buffer accumulation (#1445). The buffer's layout is
		// [prefix | r0_conversation | r0_output | r1_conversation | r1_output | ... | rN_newInput],
		// observationally identical to the legacy [prefix | accumulated context | newInput]
		// concatenation but stored as one growable slice instead of fresh copies per round.
		//
		// Seed on the first call: append prefix (if any), then the conversation
		// portion of round 0's input. Mirrors the legacy guard at the strip site:
		// if round 0's input is shorter than the prefix (defensive — e.g. malformed
		// trace replay), treat the entire input as conversation to avoid a slice
		// bounds panic.
		if !sess.seeded {
			if len(bp.Prefix) > 0 {
				sess.buf.Append(bp.Prefix)
			}
			rawConversation := req.FullInputTokens()
			if int64(len(bp.Prefix)) <= req.InputLen() {
				rawConversation = req.InputTokenSlice(int64(len(bp.Prefix)), req.InputLen())
			} else {
				// Defensive fallback: round 0's input is shorter than the prefix
				// (malformed trace replay or pathological sampler). Match the
				// legacy behavior — treat the entire input as conversation. This
				// preserves byte-for-byte equivalence with the pre-PR session.go
				// path. Error severity (not warn): the condition indicates
				// upstream data corruption that operators must investigate.
				logrus.Errorf("SessionManager.OnComplete: session %s round 0 input length %d < prefix length %d (malformed trace?); treating full input as conversation",
					req.SessionID, req.InputLen(), len(bp.Prefix))
			}
			sess.buf.Append(rawConversation)
			sess.seeded = true
		} else if req.InputLen() != sess.buf.Len() {
			// Subsequent call: the previous OnComplete returned this round's
			// InputTokens as a flat slice spanning buf[0:buf.Len()], so the
			// contract is req.InputLen() == buf.Len() exactly. Any divergence
			// (longer OR shorter) means a caller has reassigned
			// req.InputTokens to a different slice — appending from buf.Len()
			// in that case either loses tokens (too long) or seeds extra
			// tokens that the request never carried (too short). Both
			// directions are programming errors.
			panic(fmt.Sprintf("SessionManager.OnComplete: session %s req.InputLen=%d != buf.Len=%d — buffer continuity broken; expected req.InputTokens to alias buf[0:buf.Len()]",
				req.SessionID, req.InputLen(), sess.buf.Len()))
		}
		// Append the round's actual output and the new round's input.
		if actualOutputLen > 0 && len(req.OutputTokens) > 0 {
			outTokens := req.OutputTokens
			switch {
			case actualOutputLen > len(outTokens):
				// Over-cap defense: ProgressIndex accounting should never produce
				// an actualOutputLen exceeding the oracle output length. If it
				// does, log loudly and fall through to appending the full
				// oracle output — the upstream computation has drifted and the
				// resulting metrics are suspect. (outTokens is already the full
				// slice; no trim needed.)
				logrus.Errorf("SessionManager.OnComplete: session %s round %d actualOutputLen=%d > len(OutputTokens)=%d — appending full oracle output; ProgressIndex accounting may be incorrect",
					req.SessionID, req.RoundIndex, actualOutputLen, len(outTokens))
			case actualOutputLen < len(outTokens):
				outTokens = outTokens[:actualOutputLen]
				logrus.Debugf("SessionManager.OnComplete: session %s round %d length-capped — accumulating %d/%d output tokens",
					req.SessionID, req.RoundIndex, actualOutputLen, len(req.OutputTokens))
			}
			sess.buf.Append(outTokens)
		}
		_, inputEnd := sess.buf.Append(newInputTokens)
		inputTokens = sess.buf.Slice(0, inputEnd)
	} else {
		inputTokens = newInputTokens
		// Non-accumulate: prepend prefix freshly (no shared buffer in this mode).
		if len(bp.Prefix) > 0 {
			inputTokens = append(append([]int{}, bp.Prefix...), inputTokens...)
		}
	}

	sess.currentRound++
	sm.idCounter++
	nextReq := &sim.Request{
		ID:           fmt.Sprintf("session_%s_round_%d_%d", bp.SessionID, sess.currentRound, sm.idCounter),
		ArrivalTime:  arrivalTime,
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		MaxOutputLen: len(outputTokens),
		State:        sim.StateQueued,
		Deadline:     computeDeadline(arrivalTime, bp.Timeout, true), // session follow-up always gets default timeout
		SLOTargetUs:  bp.SLOTargetUs,
		TenantID:     bp.TenantID,
		SLOClass:     bp.SLOClass,
		Model:        bp.Model,
		ClientID:     bp.ClientID,
		SessionID:    bp.SessionID,
		RoundIndex:   sess.currentRound,
	}
	if sm.budgetEnabled {
		sm.followUpCount++
	}
	return []*sim.Request{nextReq}
}

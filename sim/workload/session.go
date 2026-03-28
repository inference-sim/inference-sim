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
	SessionID       string
	ClientID        string
	MaxRounds       int
	UnlimitedRounds bool   // when true, session continues past MaxRounds until budget/horizon/timeout/drop
	ContextGrowth   string // "accumulate" or ""
	ThinkTimeUs   int64
	Timeout       *int64 // per-request timeout from ClientSpec (nil = default 300s)
	Horizon       int64  // simulation horizon for BC-19 guard
	InputSampler  LengthSampler
	OutputSampler LengthSampler
	RNG           *rand.Rand // per-session, seeded deterministically from client RNG
	Prefix        []int      // shared system prompt tokens
	TenantID      string
	SLOClass      string
	Model         string
}

// activeSession tracks mutable per-session lifecycle state.
type activeSession struct {
	blueprint     *SessionBlueprint
	currentRound  int
	contextTokens []int        // accumulated input + output from prior rounds
	state         sessionState
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
		sm.sessions[bp.SessionID] = &activeSession{
			blueprint: bp,
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
	arrivalTime := tick + bp.ThinkTimeUs
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
	actualOutputLen := max(int(req.ProgressIndex)-len(req.InputTokens), 0)

	var inputTokens []int
	if bp.ContextGrowth == "accumulate" {
		// Accumulate: prepend prior context + this round's actual input/output
		sess.contextTokens = append(sess.contextTokens, req.InputTokens...)
		if actualOutputLen > 0 && len(req.OutputTokens) > 0 {
			outTokens := req.OutputTokens
			if actualOutputLen < len(outTokens) {
				outTokens = outTokens[:actualOutputLen]
			}
			sess.contextTokens = append(sess.contextTokens, outTokens...)
		}
		inputTokens = append(append([]int{}, sess.contextTokens...), newInputTokens...)
	} else {
		inputTokens = newInputTokens
	}

	// Prepend prefix
	if len(bp.Prefix) > 0 {
		inputTokens = append(append([]int{}, bp.Prefix...), inputTokens...)
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

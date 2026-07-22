package cmd

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// validateObserveCorpusFlags enforces the spec-mode vs corpus-mode split for
// blis observe. Corpus-mode is selected by --concurrent-sessions > 0 (or a
// corpus file). It is mutually exclusive with every spec-mode input, because a
// corpus IS the workload — there is nothing to generate. Returns "" when the
// flag combination is valid, else a human-readable error for logrus.Fatalf.
func validateObserveCorpusFlags(
	concurrentSessions, totalSessions int,
	corpusHeader, corpusData string,
	workload, workloadSpec string,
	rateChanged bool,
	concurrency int,
) string {
	corpusFilesSet := corpusHeader != "" || corpusData != ""
	corpusMode := concurrentSessions > 0

	// --total-sessions is meaningless without a pool.
	if totalSessions != 0 && !corpusMode {
		return "--total-sessions requires --concurrent-sessions > 0"
	}
	// Corpus files supplied but pool not enabled.
	if corpusFilesSet && !corpusMode {
		return "--corpus-header/--corpus-data require --concurrent-sessions > 0"
	}
	if !corpusMode {
		return "" // spec-mode: no corpus constraints apply
	}

	// Corpus-mode: both files required.
	if corpusHeader == "" || corpusData == "" {
		return "corpus-mode (--concurrent-sessions > 0) requires both --corpus-header and --corpus-data"
	}
	// Corpus-mode: reject every spec-mode input.
	if concurrency > 0 {
		return "--concurrency is invalid with --concurrent-sessions (spec-mode vs corpus-mode): a corpus IS the workload; use --concurrent-sessions alone"
	}
	if workload != "" {
		return "--workload is invalid with --concurrent-sessions (spec-mode vs corpus-mode)"
	}
	if workloadSpec != "" {
		return "--workload-spec is invalid with --concurrent-sessions (spec-mode vs corpus-mode)"
	}
	if rateChanged {
		return "--rate is invalid with --concurrent-sessions (spec-mode vs corpus-mode)"
	}
	return ""
}

// buildObserveCorpusPool loads a TraceV2 corpus and constructs the session pool
// that drives corpus-mode observe. The horizon passed to the blueprint loader is
// unbounded (math.MaxInt64): observe's dispatch loop drains on active-session
// count, not a wall-clock horizon, so the pool self-drains all `total` sessions
// (mirrors blis replay's self-draining default). Returns the driver and the
// initial `concurrent` round-0 requests to seed the dispatch loop.
func buildObserveCorpusPool(
	corpusHeader, corpusData string,
	concurrentSessions, totalSessions int,
	seed int64,
) (*workload.SessionPoolDriver, []*sim.Request, error) {
	trace, err := workload.LoadTraceV2(corpusHeader, corpusData)
	if err != nil {
		return nil, nil, fmt.Errorf("loading corpus: %w", err)
	}
	// nil thinkTimeSampler → derive think time from the corpus's inter-round
	// arrival gaps (same as replay's default closed-loop path).
	r0Requests, blueprints, err := workload.LoadTraceV2SessionBlueprints(trace, seed, nil, math.MaxInt64)
	if err != nil {
		return nil, nil, fmt.Errorf("building session blueprints: %w", err)
	}
	if len(blueprints) == 0 {
		return nil, nil, fmt.Errorf("corpus has no session records (need SessionID + RoundIndex rows)")
	}
	driver, initial, err := workload.BuildSessionPool(blueprints, r0Requests, concurrentSessions, totalSessions, seed)
	if err != nil {
		return nil, nil, fmt.Errorf("building session pool: %w", err)
	}
	return driver, initial, nil
}

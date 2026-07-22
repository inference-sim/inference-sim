package cmd

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

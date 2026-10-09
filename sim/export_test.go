package sim

// Test-only exports for the external sim_test package (compiled only with tests).

// RegisterCompletionScorerForTest registers, under name, a scorer whose constructor
// returns the given score, routing observer and request-completion observer (any of
// the observers may be nil). It returns a restore func that puts the original scorer
// registry back. Mutates the package-global registry: callers must not run in
// parallel with other registry users.
func RegisterCompletionScorerForTest(
	name string,
	score func(*Request, []RoutingSnapshot) map[string]float64,
	observe func(*Request, string),
	onComplete func(*Request, string, int64),
) (restore func()) {
	old := scorerRegistry
	scorerRegistry = cloneRegistry(old)
	parts := scorerParts{score: score}
	if observe != nil {
		parts.observe = observe
	}
	if onComplete != nil {
		parts.onComplete = onComplete
	}
	registerScorer(name, func(_ int, _ cacheQueryFn) scorerParts { return parts })
	return func() { scorerRegistry = old }
}

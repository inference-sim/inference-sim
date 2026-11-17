package workload

import "fmt"

// MaxContextMode says what a convert command does with a session that has a round the target
// cannot serve (inference-sim#1766). Context accumulates, so dropping one round would break the
// prefix chain of the next: a round is never dropped alone. The two modes bias the corpus
// differently, so a ceiling always states one.
type MaxContextMode string

const (
	// MaxContextDropSession discards any session in which a round would exceed the ceiling.
	MaxContextDropSession MaxContextMode = "drop-session"
	// MaxContextTruncate keeps a session up to the first round that would exceed the ceiling
	// and discards the rest.
	MaxContextTruncate MaxContextMode = "truncate"
)

// ValidateMaxContext checks a ceiling and its mode: a negative ceiling is refused, a positive one
// needs a mode, and a mode without a ceiling is refused rather than ignored.
func ValidateMaxContext(maxContext int, mode MaxContextMode) error {
	switch {
	case maxContext < 0:
		return fmt.Errorf("--max-context must be >= 0, got %d", maxContext)
	case maxContext == 0 && mode != "":
		return fmt.Errorf("--max-context-mode %q needs --max-context", mode)
	case maxContext > 0 && mode != MaxContextDropSession && mode != MaxContextTruncate:
		return fmt.Errorf("--max-context %d needs --max-context-mode %q or %q (they bias the corpus differently, so neither is a default)",
			maxContext, MaxContextDropSession, MaxContextTruncate)
	}
	return nil
}

// boundContext applies a context ceiling to a session's rounds. A round exceeds it when its
// recorded prompt plus its output is more than maxContext tokens: the context the target must
// hold to serve it. It returns the rounds to keep, or nil to drop the session. A zero ceiling
// keeps every round.
func boundContext(rounds []NormalizedRound, maxContext int, mode MaxContextMode) []NormalizedRound {
	if maxContext <= 0 {
		return rounds
	}
	for i, r := range rounds {
		if r.InputTokensAbs+r.OutputTokens <= maxContext {
			continue
		}
		if mode == MaxContextTruncate {
			return rounds[:i]
		}
		return nil
	}
	return rounds
}

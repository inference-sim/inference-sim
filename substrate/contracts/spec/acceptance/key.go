package acceptance

import (
	"regexp"

	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// DraftSampler is how a drafter samples its drafts (inference-sim/blis-substrate#174).
type DraftSampler string

const (
	// DraftGreedy: the drafter proposes its most likely token.
	DraftGreedy DraftSampler = "greedy"
	// DraftProbabilistic: the drafter samples its proposal from its distribution.
	DraftProbabilistic DraftSampler = "probabilistic"
)

var draftSamplers = map[DraftSampler]bool{DraftGreedy: true, DraftProbabilistic: true}

// Valid reports whether s is a recognized draft sampler. Empty is unstated.
func (s DraftSampler) Valid() bool { return s == "" || draftSamplers[s] }

// AllDraftSamplers returns every draft sampler, sorted, for error text.
func AllDraftSamplers() []string { return vocab.Enumerate(draftSamplers) }

// Verifier is how the target verifies drafts, in vLLM v0.33.0's vocabulary: token-by-token
// rejection sampling (standard), block verification (block), or fly verification (fly)
// (inference-sim/blis-substrate#174, design D8). Synthetic rejection sampling is a realisation
// (forced acceptance), not a verifier.
type Verifier string

const (
	VerifierStandard Verifier = "standard"
	VerifierBlock    Verifier = "block"
	VerifierFly      Verifier = "fly"
)

var verifiers = map[Verifier]bool{VerifierStandard: true, VerifierBlock: true, VerifierFly: true}

// enginePattern is the spelling of an engine name: lower case, as a deployment's engine block
// states it.
var enginePattern = regexp.MustCompile(`^[a-z][a-z0-9_-]*$`)

// ValidEngine reports whether e is unstated or a well-formed engine name.
func ValidEngine(e string) bool { return e == "" || enginePattern.MatchString(e) }

// Valid reports whether v is a recognized verifier. Empty is unstated.
func (v Verifier) Valid() bool { return v == "" || verifiers[v] }

// AllVerifiers returns every verifier, sorted, for error text.
func AllVerifiers() []string { return vocab.Enumerate(verifiers) }

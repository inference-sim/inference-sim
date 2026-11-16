package acceptance

import "github.com/inference-sim/blis-substrate/contracts/vocab"

// DraftSampler is how a drafter samples its drafts (inference-sim/blis-substrate#174).
type DraftSampler string

const (
	// DraftGreedy: the drafter proposes its most likely token.
	DraftGreedy DraftSampler = "greedy"
	// DraftProbabilistic: the drafter samples its proposal from its distribution.
	DraftProbabilistic DraftSampler = "probabilistic"
)

var draftSamplers = map[DraftSampler]bool{DraftGreedy: true, DraftProbabilistic: true}

// Valid reports whether s is a recognized draft sampler. Empty is the engine's default.
func (s DraftSampler) Valid() bool { return s == "" || draftSamplers[s] }

// AllDraftSamplers returns every draft sampler, sorted, for error text.
func AllDraftSamplers() []string { return vocab.Enumerate(draftSamplers) }

// Verifier is how the target verifies drafts: token-by-token rejection sampling (standard),
// block verification, or relaxed acceptance (inference-sim/blis-substrate#174).
type Verifier string

const (
	VerifierStandard Verifier = "standard"
	VerifierBlock    Verifier = "block"
	VerifierRelaxed  Verifier = "relaxed"
)

var verifiers = map[Verifier]bool{VerifierStandard: true, VerifierBlock: true, VerifierRelaxed: true}

// Valid reports whether v is a recognized verifier. Empty is the engine's default.
func (v Verifier) Valid() bool { return v == "" || verifiers[v] }

// AllVerifiers returns every verifier, sorted, for error text.
func AllVerifiers() []string { return vocab.Enumerate(verifiers) }

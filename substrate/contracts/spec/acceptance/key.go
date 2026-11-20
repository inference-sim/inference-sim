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

// DraftEngineDefault, in a profile, marks a sampler the source left at its engine's default: it
// matches the default the requesting engine states (Request.DefaultDraftSampler), when the
// profile's engine is the request's. A request states resolved values and never the marker.
const DraftEngineDefault DraftSampler = "default"

var draftSamplers = map[DraftSampler]bool{DraftGreedy: true, DraftProbabilistic: true}

// Valid reports whether s is a recognized draft sampler, or the engine-default marker. Empty is
// unstated.
func (s DraftSampler) Valid() bool { return s == "" || s == DraftEngineDefault || draftSamplers[s] }

// Resolved reports whether s is unstated or a resolved sampler, as a request states it.
func (s DraftSampler) Resolved() bool { return s == "" || draftSamplers[s] }

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

// VerifierEngineDefault, in a profile, marks a verifier the source left at its engine's default,
// as DraftEngineDefault does for the sampler.
const VerifierEngineDefault Verifier = "default"

var verifiers = map[Verifier]bool{VerifierStandard: true, VerifierBlock: true, VerifierFly: true}

// Resolved reports whether v is unstated or a resolved verifier, as a request states it.
func (v Verifier) Resolved() bool { return v == "" || verifiers[v] }

// enginePattern is the spelling of an engine name: lower case, as a deployment's engine block
// states it.
var enginePattern = regexp.MustCompile(`^[a-z][a-z0-9_-]*$`)

// ValidEngine reports whether e is unstated or a well-formed engine name.
func ValidEngine(e string) bool { return e == "" || enginePattern.MatchString(e) }

// Valid reports whether v is a recognized verifier, or the engine-default marker. Empty is
// unstated.
func (v Verifier) Valid() bool { return v == "" || v == VerifierEngineDefault || verifiers[v] }

// AllVerifiers returns every verifier, sorted, for error text.
func AllVerifiers() []string { return vocab.Enumerate(verifiers) }

// withEngineDefaults returns sets with each "default" marker of a profile measured by the
// request's engine replaced by the default that engine states in the request, so the lookup only
// compares. A marker of another engine, or one the request states no default for, stays, and so
// matches no resolved value. The sets themselves are not changed.
func withEngineDefaults(sets []*ProfileSet, req Request) []*ProfileSet {
	if req.Engine == "" || (req.DefaultDraftSampler == "" && req.DefaultVerifier == "") {
		return sets
	}
	out := make([]*ProfileSet, len(sets))
	for i, s := range sets {
		out[i] = s
		if s == nil {
			continue
		}
		var copied *ProfileSet
		for j, p := range s.Profiles {
			if p.Engine != req.Engine {
				continue
			}
			sub := p
			if sub.DraftSampler == DraftEngineDefault && req.DefaultDraftSampler != "" {
				sub.DraftSampler = req.DefaultDraftSampler
			}
			if sub.Verifier == VerifierEngineDefault && req.DefaultVerifier != "" {
				sub.Verifier = req.DefaultVerifier
			}
			if sub.DraftSampler == p.DraftSampler && sub.Verifier == p.Verifier {
				continue
			}
			if copied == nil {
				c := *s
				c.Profiles = append([]Profile(nil), s.Profiles...)
				copied = &c
			}
			copied.Profiles[j] = sub
		}
		if copied != nil {
			out[i] = copied
		}
	}
	return out
}

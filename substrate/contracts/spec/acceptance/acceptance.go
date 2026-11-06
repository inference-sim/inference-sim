// Package acceptance describes how far speculative decoding advances a request per
// verification step: a profile set of measured acceptance lengths hosted in blis-registry,
// the inline values a workload, a scenario or a command line may state instead, and the
// resolution that picks one of them for a run (blis-schemas#65).
//
// The canonical quantity is the mean acceptance length AL: the tokens a request gains per
// verification step, counting the target's bonus token, so AL lies in [1, K+1] for K
// speculative tokens. It is what the authoritative sources publish (InferenceX's golden
// SPEED-Bench curves) and what the engines take (a synthetic acceptance length, SGLang
// SGLANG_SIMULATE_ACC_LEN).
//
// Two derived quantities have one meaning each here:
//
//   - The acceptance rate is the accepted fraction of drafts, (AL-1)/K. It matches the engine's
//     logged "Avg Draft acceptance rate" and inference-sim's --speculative-acceptance-rate.
//     The two agree only when every step drafts all K: with a drafter that verifies
//     fewer on some steps, such as DSpark under enable_adaptive_verification, the engine's
//     logged rate, accepted over drafted tokens (v1/spec_decode/metrics.py:107-110),
//     divides by the drafts actually proposed and diverges from (AL-1)/K.
//     The per-token acceptance probability p of the speculative-decoding papers is a
//     different quantity and is never called a rate in this schema.
//   - Per-position rates p_1..p_K follow the engine's synthetic_acceptance_rates
//     (config/speculative.py:538-544 in the engine): p_i is the unconditional probability
//     that the first i drafts are all accepted, so the list is non-increasing, each entry
//     lies in [0, 1], and it sums to AL-1.
//
// Acceptance is the simulator's input, not the kernel's: K and the method stay in the
// deployment, because they change what a step costs, while acceptance changes only how far
// requests advance.
package acceptance

import (
	"fmt"
	"regexp"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// Kind is the kind key of an acceptance-profile set.
const Kind = "AcceptanceProfileSet"

// Thinking is whether the target model ran with its reasoning ("thinking") mode on.
// Acceptance differs materially between the two: DeepSeek-V4.1-Flash with DSpark at K=5
// accepts 4.07 tokens per step with thinking off and 3.51 with it on.
type Thinking string

const (
	ThinkingOn  Thinking = "on"
	ThinkingOff Thinking = "off"
)

var thinkingModes = map[Thinking]bool{ThinkingOn: true, ThinkingOff: true}

// Valid reports whether t is a recognized thinking mode.
func (t Thinking) Valid() bool { return thinkingModes[t] }

// AllThinkingModes returns every thinking mode, sorted, for error text.
func AllThinkingModes() []string { return vocab.Enumerate(thinkingModes) }

// categoryPattern is the spelling of a workload category: lower case, so "Coding" cannot
// silently miss a curve keyed "coding".
var categoryPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)

// ValidCategory reports whether c is a well-formed workload category. The set is open
// (SPEED-Bench alone has eleven), so only the spelling is checked.
func ValidCategory(c string) bool { return categoryPattern.MatchString(c) }

// ProfileSet is one immutable, registry-hosted collection of acceptance profiles, the
// acceptance counterpart of a coefficient set. A scenario names one by its file name.
type ProfileSet struct {
	Kind string `yaml:"kind"`
	Name string `yaml:"name"`
	// Caveats state what the set does not cover and how far its values travel: which
	// categories have no curve, which hardware they were measured on, which K the image
	// refused.
	Caveats  []string  `yaml:"caveats,omitempty"`
	Profiles []Profile `yaml:"profiles"`
}

// Profile is one measured acceptance length, keyed by (model, speculative method, K,
// thinking mode, workload category).
type Profile struct {
	// Model is the catalog model name, not the source's spelling of it.
	Model string `yaml:"model"`
	// SpeculativeMethod is the engine's method name, as a deployment's
	// engine.speculative.method states it. Which names an engine accepts is a rules
	// pack's to say.
	SpeculativeMethod string   `yaml:"speculative_method"`
	NumSpecTokens     int      `yaml:"num_spec_tokens"`
	Thinking          Thinking `yaml:"thinking"`
	Category          string   `yaml:"category"`

	AcceptanceLength float64 `yaml:"acceptance_length"`
	// PerPositionRates is optional: no source in scope publishes them, and a consumer
	// that needs a distribution derives one with MinimumVarianceRates.
	PerPositionRates []float64 `yaml:"per_position_rates,omitempty"`

	// Method is how the value was obtained, in the registry's evidence vocabulary.
	Method vocab.Method `yaml:"method"`
	// Provenance is required for every method but assumed.
	Provenance *Provenance `yaml:"provenance,omitempty"`
	// Rationale explains a value that was not measured.
	Rationale string `yaml:"rationale,omitempty"`
	// CopiedFrom names the profile a copied value came from, as its Key renders.
	CopiedFrom string `yaml:"copied_from,omitempty"`
}

// Key is a profile's identity.
type Key struct {
	Model             string
	SpeculativeMethod string
	NumSpecTokens     int
	Thinking          Thinking
	Category          string
}

// String renders a key as model/method/k<K>/thinking-<mode>/category, the form
// copied_from and error messages use.
func (k Key) String() string {
	return fmt.Sprintf("%s/%s/k%d/thinking-%s/%s",
		k.Model, k.SpeculativeMethod, k.NumSpecTokens, k.Thinking, k.Category)
}

// Key returns the profile's identity.
func (p Profile) Key() Key {
	return Key{Model: p.Model, SpeculativeMethod: p.SpeculativeMethod,
		NumSpecTokens: p.NumSpecTokens, Thinking: p.Thinking, Category: p.Category}
}

// Rates returns the stated per-position rates, or the minimum-variance schedule for the
// acceptance length when none are stated, and whether they were derived.
func (p Profile) Rates() ([]float64, bool, error) {
	if len(p.PerPositionRates) > 0 {
		return append([]float64(nil), p.PerPositionRates...), false, nil
	}
	r, err := MinimumVarianceRates(p.AcceptanceLength, p.NumSpecTokens)
	return r, true, err
}

// Provenance records where a profile was measured, so a consumer can report a value
// reused on other hardware, under other sampling, or from another image.
type Provenance struct {
	// Source is the document the value was read from, such as the golden-curve file's URL.
	Source string `yaml:"source"`
	// Run is the job that measured it, such as a GitHub Actions run URL.
	Run string `yaml:"run,omitempty"`
	// Benchmark names the prompt set, such as "SPEED-Bench Qualitative".
	Benchmark string `yaml:"benchmark,omitempty"`
	// Prompts is how many prompts the measurement covered. Zero means unstated.
	Prompts int `yaml:"prompts,omitempty"`
	// Hardware is the catalog chip the value was measured on.
	Hardware string `yaml:"hardware,omitempty"`
	// EngineImage is the engine image or release that ran.
	EngineImage string `yaml:"engine_image,omitempty"`
	// TP is the tensor-parallel width it ran at. Zero means unstated.
	TP int `yaml:"tp,omitempty"`
	// Sampling is the decoding configuration. Acceptance depends on it.
	Sampling *Sampling `yaml:"sampling,omitempty"`
	// Settings are other engine settings that bear on acceptance, as the source states
	// them, such as rejection_sample_method: block verification accepts more than
	// token-by-token rejection sampling.
	Settings map[string]string `yaml:"settings,omitempty"`
}

// Sampling is the decoding configuration a profile was measured under. Temperature is a
// pointer because 0, greedy decoding, is a real setting distinct from unstated.
type Sampling struct {
	Temperature *float64 `yaml:"temperature"`
	TopP        *float64 `yaml:"top_p,omitempty"`
	// TopK follows the engine: -1 or 0 disables it.
	TopK      *int `yaml:"top_k,omitempty"`
	MaxTokens int  `yaml:"max_tokens,omitempty"`
}

// Value is an acceptance stated inline: by a workload that measured its own, by a
// scenario that pins one, or by a command line. It names the K it was stated for, so a
// value stated for another K is refused rather than reinterpreted.
//
// Acceptance length and acceptance rate are two spellings of one number and are mutually
// exclusive. Per-position rates may stand alone, which states AL as 1 + their sum, or
// accompany either, which they must then agree with.
type Value struct {
	NumSpecTokens    int       `yaml:"num_spec_tokens"`
	AcceptanceLength *float64  `yaml:"acceptance_length,omitempty"`
	AcceptanceRate   *float64  `yaml:"acceptance_rate,omitempty"`
	PerPositionRates []float64 `yaml:"per_position_rates,omitempty"`
	// Source cites where the value came from, such as the server log of the run it
	// reproduces. It is recorded with the resolution.
	Source string `yaml:"source,omitempty"`
}

// UnmarshalYAML restores strict unknown-key checking, which a Value nested under a type
// with its own unmarshaller (a workload shape) would otherwise lose.
func (v *Value) UnmarshalYAML(node *yaml.Node) error {
	if err := validate.RejectUnknownKeys(node, Value{}, nil); err != nil {
		return err
	}
	type shape Value
	var raw shape
	if err := validate.DecodeNode(node, &raw); err != nil {
		return err
	}
	*v = Value(raw)
	return nil
}

// Length returns the acceptance length the value states, from whichever field states it.
// It assumes a valid value.
func (v Value) Length() float64 {
	switch {
	case v.AcceptanceLength != nil:
		return *v.AcceptanceLength
	case v.AcceptanceRate != nil:
		return LengthFromRate(*v.AcceptanceRate, v.NumSpecTokens)
	default:
		return 1 + sum(v.PerPositionRates)
	}
}

// Rates returns the stated per-position rates, or the minimum-variance schedule for the
// value's length when none are stated, and whether they were derived.
func (v Value) Rates() ([]float64, bool, error) {
	if len(v.PerPositionRates) > 0 {
		return append([]float64(nil), v.PerPositionRates...), false, nil
	}
	r, err := MinimumVarianceRates(v.Length(), v.NumSpecTokens)
	return r, true, err
}

// Pin is a scenario's acceptance slot. It either names a profile set that replaces the
// registry default for the run, or states a value inline; FallbackCategory, with a set or
// alone, asks for a category to substitute when the workload's has no curve, which the
// resolution then records.
type Pin struct {
	Profiles         string `yaml:"profiles,omitempty"`
	Value            *Value `yaml:"value,omitempty"`
	FallbackCategory string `yaml:"fallback_category,omitempty"`
}

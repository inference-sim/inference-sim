// Package routing describes how unevenly a mixture-of-experts router loads the ranks of an
// expert-parallel group: a profile set of measured imbalance curves hosted in blis-registry,
// the inline value a workload, a scenario or a command line may state instead, and the
// resolution that picks one of them for a run. It follows the acceptance package's shape:
// the quantity depends on the model (its router and how its experts specialise) and on the
// traffic, so it is keyed by model and workload category, not stored as a hardware
// coefficient.
//
// The canonical quantity is the engine's own. In the engine, a step's load on rank r in layer l is the
// count of routed (token, expert) assignments that land on r's physical experts
// (distributed/eplb/eplb_state.py:601-608), and the engine logs balancedness as
// sum_l mean_r T[l,r] / sum_l max_r T[l,r] (:66-71, :623-636). TokenImbalance is its
// reciprocal: the busiest rank's routed tokens over the mean rank's, aggregated over layers.
// It is at least 1. Every layer ends in a combine that waits for its slowest rank, so a
// layer's routed compute is paid at the busiest rank's load.
//
// ExpertImbalance is the companion for weight reads: the busiest rank's DISTINCT experts
// touched in a step, over the count expected per rank under uniform routing. A decode-sized
// grouped GEMM is bound by reading the weights of the experts it touches, so this ratio
// scales that read. It may be below 1, because a skewed router reaches fewer distinct
// experts than a uniform one.
//
// Both depend on the expert-parallel width and on the tokens in the step (the finite-sample
// term falls as the batch grows), and on placement: the values here are for the engine's default
// linear placement (config/parallel.py:192) with EPLB off (enable_eplb defaults False,
// :188).
package routing

import (
	"fmt"
	"regexp"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// Kind is the kind key of a routing-profile set.
const Kind = "RoutingProfileSet"

// Phase is which kind of step a curve describes. A decode step mixes one token from each of
// many requests; a prefill chunk is many consecutive tokens of one request, which routes
// more alike and so loads ranks more unevenly at the same token count.
type Phase string

const (
	PhaseDecode  Phase = "decode"
	PhasePrefill Phase = "prefill"
)

// Valid reports whether p is a recognised phase.
func (p Phase) Valid() bool { return p == PhaseDecode || p == PhasePrefill }

// Role is how a registry set may be used. A default set is consulted for every run whose
// key it carries; a sensitivity set only when a scenario names it. Values without
// authoritative corroboration ship as a sensitivity, never as a silent default.
type Role string

const (
	RoleDefault     Role = "default"
	RoleSensitivity Role = "sensitivity"
)

// Placement is how logical experts are laid out over ranks. Only the engine's default is
// described; EPLB's rearranged placement would need its own curves.
const PlacementLinear = "linear"

var categoryPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)

// ValidCategory reports whether c is a well-formed workload category, spelled as the
// acceptance package spells one.
func ValidCategory(c string) bool { return categoryPattern.MatchString(c) }

// ProfileSet is one immutable, registry-hosted collection of routing profiles.
type ProfileSet struct {
	Kind string `yaml:"kind"`
	Name string `yaml:"name"`
	Role Role   `yaml:"role"`
	// Caveats state how far the values travel: the hardware and engine they were captured
	// on, the prompts, the phases not captured.
	Caveats  []string  `yaml:"caveats,omitempty"`
	Profiles []Profile `yaml:"profiles"`
}

// Profile is one measured imbalance curve, keyed by (model, workload category, phase,
// placement, expert-parallel width), over the tokens in a step.
type Profile struct {
	// Model is the catalog model name.
	Model     string `yaml:"model"`
	Category  string `yaml:"category"`
	Phase     Phase  `yaml:"phase"`
	Placement string `yaml:"placement"`
	// ExpertParallel is the width the ratios were computed at. A curve at one width says
	// nothing exact about another, so a lookup at a width with no curve is refused.
	ExpertParallel int `yaml:"expert_parallel"`
	// Points are the curve, by strictly increasing BatchTokens. Between points a lookup
	// interpolates in log(batch); beyond the ends it holds the end value and says so.
	Points []Point `yaml:"points"`

	Method     string      `yaml:"method"`
	Provenance *Provenance `yaml:"provenance,omitempty"`
	Rationale  string      `yaml:"rationale,omitempty"`
}

// Point is the curve at one step size. The P10/P90 band is the step-to-step spread, not a
// confidence interval on the median.
type Point struct {
	BatchTokens     int      `yaml:"batch_tokens"`
	TokenImbalance  float64  `yaml:"token_imbalance"`
	ExpertImbalance float64  `yaml:"expert_imbalance"`
	TokenP10        *float64 `yaml:"token_p10,omitempty"`
	TokenP90        *float64 `yaml:"token_p90,omitempty"`
}

// Key is a profile's identity.
type Key struct {
	Model          string
	Category       string
	Phase          Phase
	Placement      string
	ExpertParallel int
}

// String renders a key as model/category/phase/placement/ep<N>.
func (k Key) String() string {
	return fmt.Sprintf("%s/%s/%s/%s/ep%d", k.Model, k.Category, k.Phase, k.Placement,
		k.ExpertParallel)
}

// Key returns the profile's identity.
func (p Profile) Key() Key {
	return Key{Model: p.Model, Category: p.Category, Phase: p.Phase,
		Placement: p.Placement, ExpertParallel: p.ExpertParallel}
}

// Provenance records where a profile was derived from.
type Provenance struct {
	// Source is the dataset the routes were read from, with its revision.
	Source string `yaml:"source"`
	// Derivation names the script and its output digest.
	Derivation string `yaml:"derivation,omitempty"`
	Hardware   string `yaml:"hardware,omitempty"`
	Engine     string `yaml:"engine,omitempty"`
	// Requests is how many requests the routes cover. Zero means unstated.
	Requests int `yaml:"requests,omitempty"`
	// Tier is the source's reputation tier: 1 authoritative, 2 with caveats.
	Tier int `yaml:"tier,omitempty"`
}

// Value is an imbalance stated inline, flat in the step size: by a workload that measured
// its own, by a scenario that pins one, or by a command line. 1 and 1 mean evenly loaded.
type Value struct {
	TokenImbalance float64 `yaml:"token_imbalance"`
	// ExpertImbalance defaults to 1 (the uniform-routing weight read) when omitted.
	ExpertImbalance *float64 `yaml:"expert_imbalance,omitempty"`
	Source          string   `yaml:"source,omitempty"`
}

// UnmarshalYAML restores strict unknown-key checking for a Value nested under a type with
// its own unmarshaller.
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

// Expert returns the stated weight-read ratio, 1 when unstated.
func (v Value) Expert() float64 {
	if v.ExpertImbalance == nil {
		return 1
	}
	return *v.ExpertImbalance
}

// Pin is a scenario's routing slot: a named profile set (a sensitivity set is usable only
// this way), or an inline value.
type Pin struct {
	Profiles string `yaml:"profiles,omitempty"`
	Value    *Value `yaml:"value,omitempty"`
	// FallbackCategory, with Profiles, names the category whose curves to use when the
	// workload's has none. The substitution is recorded.
	FallbackCategory string `yaml:"fallback_category,omitempty"`
}

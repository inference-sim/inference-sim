// Package routertable describes the router table: for each router BLIS does not host, the
// policy of llm-d's router that stands closest to it, with the evidence for the match
// (design §11.3 and §11.4).
//
// BLIS hosts one router, llm-d's (X4), and never simulates another. A benchmark run measured
// behind another router, such as Dynamo's, could not be scored. The design scores it G2 when
// this table maps the router to the nearest policy llm-d's router offers, and the point names
// the router the run used, the policy BLIS simulated and the evidence (point-record kind
// router_differs). A router with no entry, or an entry that says no policy is close enough, stays
// not scored (G5 router_unsupported). The simulation is never presented as if llm-d had routed
// the run.
//
// An entry is a file catalog/routers/<name>.yaml. It is added only with evidence: a citation
// of each router's own code or documentation at a pinned revision, for each field the entry
// states (SUB-18). The mapping is a claim about behaviour, so the entry also says in words why
// the policy is close enough, or why none is.
package routertable

import "strconv"

// Kind is the value of Entry.Kind.
const Kind = "RouterMapping"

// Entry maps one router, as a run records it, to the nearest policy of llm-d's router.
//
// The key is the router, its mode and its session-affinity setting. A router's behaviour is
// set by its mode, not by its name: a run behind Dynamo's frontend is round-robin, KV-aware or
// least-loaded by the recipe's setting, and the framework's label says none of them.
type Entry struct {
	Kind string `yaml:"kind"`
	// Name is the entry's identity, the filename without its extension. It is a filename
	// fact, not a field.
	Name string `yaml:"-"`
	// Router names the router the run was measured behind, in lower case: for example
	// "dynamo". It is never "llm-d", which BLIS hosts.
	Router string `yaml:"router"`
	// Mode is the router's mode or algorithm as the run's configuration names it, for example
	// "round-robin". The router's default, when the configuration sets none, is the mode its
	// code defaults to, and a citation says so.
	Mode string `yaml:"mode"`
	// SessionAffinityTTLS is the router's session-affinity idle timeout in seconds, when the
	// configuration sets one. Absent means affinity is not configured. Two runs that differ in
	// it are different keys.
	SessionAffinityTTLS *int `yaml:"session_affinity_ttl_s,omitempty"`
	// Revisions are the builds of the router the run used for which the entry's mode, policy and
	// reason are established, each a full 40-character commit, sorted, each once (design §11.4:
	// "a citation of both routers' code or documentation at the revisions the runs used"). A
	// mapping is a claim about the behaviour of a build: an entry shown for one build says
	// nothing of another with the same mode, so a run on a revision not listed here has no
	// mapping (Resolve returns OutcomeRevisionNotCovered, and the point stays router_unsupported).
	// Every citation on the router's side is at one of them, and each has one for the mode and
	// one for the behaviour.
	Revisions []string `yaml:"revisions"`
	// Policy is the llm-d policy BLIS simulates in its place, or PolicyNone when no policy is
	// close enough and the router stays unsupported.
	Policy Policy `yaml:"policy"`
	// Reason says why Policy is close enough to the router's mode, naming what the two do
	// alike and what they do differently, or, for PolicyNone, why none is.
	Reason string `yaml:"reason"`
	// Difference lists each way the router's mode and Policy differ, one statement per aspect,
	// each pointing at the citations that show it. Required for an entry whose Policy is not
	// PolicyNone: a mapping is a claim of nearness, and the entry says exactly how near. Absent
	// for PolicyNone, where Reason says why no policy is near.
	Difference []Difference `yaml:"difference,omitempty"`
	// Bound says how far the choice of policy can move each figure, measured on BLIS: the
	// largest ratio between Policy and the alternatives, with the regime it was measured in and
	// the evidence. Required for an entry whose Policy is not PolicyNone, absent otherwise.
	Bound *Bound `yaml:"bound,omitempty"`
	// Evidence cites both routers' code or documentation at pinned revisions. Each citation
	// supports one field of the entry.
	Evidence []Citation `yaml:"evidence"`
}

// Citation is one passage of one router's code or documentation at a pinned revision.
type Citation struct {
	// Side says whose code it is: the router the run used, or llm-d's router.
	Side Side `yaml:"side"`
	// Field is the field of the entry the passage supports: mode, session_affinity_ttl_s or
	// policy.
	Field Field `yaml:"field"`
	// Repository is "<owner>/<name>" on GitHub.
	Repository string `yaml:"repository"`
	// Commit is the full 40-character commit the passage is read at.
	Commit string `yaml:"commit"`
	// Path is the file, from the repository's root.
	Path string `yaml:"path"`
	// Lines is a line number, or a range "<first>-<last>", at Commit.
	Lines string `yaml:"lines"`
	// Claim says what the passage shows.
	Claim string `yaml:"claim"`
}

// Side is whose code a citation is.
type Side string

const (
	// SideRouter: the router the run used, which BLIS does not host.
	SideRouter Side = "router"
	// SideLLMD: llm-d's router.
	SideLLMD Side = "llm-d"
)

// Field is a field of an entry that a citation supports.
type Field string

const (
	// FieldMode: the router's mode, and that a run which sets none runs it.
	FieldMode Field = "mode"
	// FieldSessionAffinityTTLS: the session-affinity setting.
	FieldSessionAffinityTTLS Field = "session_affinity_ttl_s"
	// FieldPolicy: the behaviour of the router's mode (SideRouter) or of the llm-d policy
	// (SideLLMD) that makes the two alike.
	FieldPolicy Field = "policy"
)

// Policy is the name of a policy llm-d's router offers: a composition of its scheduling plugins
// that the router runs without code of BLIS's own. The list is closed and holds names only. What a
// policy is made of (plugins, weights, tie handling, source files) changes with the router's
// release, so it is not stated here: the router fork owns it, and each entry of the table states
// the behaviour it relies on in its evidence, with a citation at the router build it names.
type Policy string

const (
	// PolicyRoundRobin is the policy that treats every endpoint alike.
	PolicyRoundRobin Policy = "round-robin"
	// PolicyRandomPicker is the policy that picks uniformly at random among the candidates.
	PolicyRandomPicker Policy = "random-picker"
	// PolicyLeastActiveRequests is the policy that prefers the endpoint with the fewest
	// in-flight requests.
	PolicyLeastActiveRequests Policy = "least-active-requests"
	// PolicyDefault is the router's built-in policy when none is given.
	PolicyDefault Policy = "default"
	// PolicyNone: no llm-d policy is close enough. The router stays unsupported (G5
	// router_unsupported). The entry's reason says why; the table records the absence so that a
	// router mode is known, not missing.
	PolicyNone Policy = "none"
)

// Policies lists the policies llm-d's router offers, in the order they are documented. It does
// not include PolicyNone, which is the absence of one.
func Policies() []Policy {
	return []Policy{PolicyRoundRobin, PolicyRandomPicker, PolicyLeastActiveRequests, PolicyDefault}
}

// PolicyNames lists the names of the policies llm-d's router offers.
func PolicyNames() []string {
	var out []string
	for _, p := range Policies() {
		out = append(out, string(p))
	}
	return out
}

// LLMDRepositories are the repositories whose code is llm-d's router: the upstream and the fork
// BLIS hosts. A citation on the llm-d side names one of them.
var LLMDRepositories = []string{"llm-d/llm-d-router", "inference-sim/llm-d-router"}

// Offered reports whether p is a policy llm-d's router offers or PolicyNone.
func (p Policy) Offered() bool {
	if p == PolicyNone {
		return true
	}
	for _, q := range Policies() {
		if q == p {
			return true
		}
	}
	return false
}

// WantName is the name an entry with this key must have: the router and the mode, and the
// session-affinity setting when there is one. The file name cannot then say another key.
func (e *Entry) WantName() string {
	n := e.Router + "-" + e.Mode
	if e.SessionAffinityTTLS != nil {
		n += "-affinity-" + strconv.Itoa(*e.SessionAffinityTTLS)
	}
	return n
}

// Key is what identifies an entry's router: the router, its mode and its session-affinity
// setting. Two entries with one key are a duplicate.
func (e *Entry) Key() string {
	k := e.Router + "/" + e.Mode
	if e.SessionAffinityTTLS != nil {
		k += "/affinity-ttl-" + strconv.Itoa(*e.SessionAffinityTTLS)
	}
	return k
}

// Covers reports whether revision, a full commit of the router the run used, is one the entry
// establishes.
func (e *Entry) Covers(revision string) bool {
	for _, r := range e.Revisions {
		if r == revision {
			return true
		}
	}
	return false
}

// Outcome is what the table says about one run's router.
type Outcome string

const (
	// OutcomeMapped: an entry with the run's router, mode and affinity setting covers the run's
	// revision and names a policy llm-d's router offers. The point can be G2 (router_differs).
	OutcomeMapped Outcome = "mapped"
	// OutcomeNoPolicy: an entry covers the revision and says no policy is close enough. The point
	// stays router_unsupported (G5).
	OutcomeNoPolicy Outcome = "no_policy"
	// OutcomeRevisionNotCovered: an entry for the router, mode and affinity setting exists and
	// does not list the run's revision. What it shows is not shown for this build, so it
	// is not inherited: the point stays router_unsupported (G5).
	OutcomeRevisionNotCovered Outcome = "revision_not_covered"
	// OutcomeNoEntry: no entry has the run's router, mode and affinity setting. The point stays
	// router_unsupported (G5).
	OutcomeNoEntry Outcome = "no_entry"
	// OutcomeRegimeNotCovered: the entry maps the mode, and none of its bounds is for the
	// point's regime (an unsampled client count, another deployment). What it shows is not shown
	// for this regime, so the point stays router_unsupported (G5), with the reason NoBoundForRegime.
	OutcomeRegimeNotCovered Outcome = "regime_not_covered"
)

// NoBoundForRegime is the reason a point carries when its entry maps the mode and no bound
// covers the point's regime.
const NoBoundForRegime = "no bound for this regime"

// BoundFor returns the entry's bound restricted to the ratios whose regime the point lies in,
// with the outcome the table gives that point: OutcomeMapped when some ratio covers it,
// OutcomeNoPolicy for a mode that maps none, and OutcomeRegimeNotCovered when none does. Only
// OutcomeMapped scores the point, so a count or a deployment the runs did not sample cannot
// carry a bound.
func (e *Entry) BoundFor(p Point) (*Bound, Outcome) {
	if e.Policy == PolicyNone {
		return nil, OutcomeNoPolicy
	}
	if b := e.Bound.For(p); b != nil {
		return b, OutcomeMapped
	}
	return nil, OutcomeRegimeNotCovered
}

// Scored reports whether a point with this outcome can be graded through the table.
func (o Outcome) Scored() bool { return o == OutcomeMapped }

// Resolve finds what the table says about a run measured behind router in mode with the given
// session-affinity setting (nil when none is configured), on the router build revision (a full
// commit). The entry is returned whenever one has the run's key, covered or not, so that a
// reader can name it; only OutcomeMapped scores the point.
func Resolve(entries []*Entry, router, mode string, affinityTTLS *int, revision string) (*Entry, Outcome) {
	for _, e := range entries {
		if e.Router != router || e.Mode != mode {
			continue
		}
		if (e.SessionAffinityTTLS == nil) != (affinityTTLS == nil) ||
			(e.SessionAffinityTTLS != nil && *e.SessionAffinityTTLS != *affinityTTLS) {
			continue
		}
		switch {
		case !e.Covers(revision):
			return e, OutcomeRevisionNotCovered
		case e.Policy == PolicyNone:
			return e, OutcomeNoPolicy
		}
		return e, OutcomeMapped
	}
	return nil, OutcomeNoEntry
}

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
	// Policy is the llm-d policy BLIS simulates in its place, or PolicyNone when no policy is
	// close enough and the router stays unsupported.
	Policy Policy `yaml:"policy"`
	// Reason says why Policy is close enough to the router's mode, naming what the two do
	// alike and what they do differently, or, for PolicyNone, why none is.
	Reason string `yaml:"reason"`
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

// Policy is a policy llm-d's router offers: a composition of its scheduling plugins that the
// router runs without code of BLIS's own. The list is closed. A policy is added with the
// router code that makes it what it is.
type Policy string

const (
	// PolicyRoundRobin: no scorer, so every endpoint scores alike, and the max-score picker
	// rotates through the endpoints with a counter. The counter is one per runtime and is drawn
	// at every pick of every profile (picker.RandFor, NextCounter), not once per request: a
	// configuration whose request runs two profiles that each pick, such as a prefill and a decode
	// profile, draws two, and a pool of even size then sees only every other endpoint. The policy
	// is strict round-robin over the endpoints, listed in ID order in BLIS's hosted router, for a
	// configuration with one pick per request.
	PolicyRoundRobin Policy = "round-robin"
	// PolicyLeastActiveRequests: the router's own count of in-flight requests per endpoint
	// (inflight-load-producer), scored by active-request-scorer so the endpoint with the
	// fewest scores highest, and the max-score picker; ties are rotated, not drawn at random.
	PolicyLeastActiveRequests Policy = "least-active-requests"
	// PolicyDefault: the router's built-in configuration when none is given: queue-scorer
	// (weight 2), kv-cache-utilization-scorer (2) and prefix-cache-scorer (3).
	PolicyDefault Policy = "default"
	// PolicyNone: no llm-d policy is close enough. The router stays unsupported (G5
	// router_unsupported). The entry's reason says why; the table records the absence so that a
	// router mode is known, not missing.
	PolicyNone Policy = "none"
)

// PolicyInfo describes a policy.
type PolicyInfo struct {
	Policy Policy
	// Plugins is the composition, as the plugin types of llm-d's router with their weights.
	Plugins string
	// Sources are the files of llm-d's router, from its root, that define the composition. An
	// entry that maps to the policy cites at least one of them.
	Sources []string
}

// Policies lists the policies llm-d's router offers, in the order they are documented. It does
// not include PolicyNone, which is the absence of one.
func Policies() []PolicyInfo {
	return []PolicyInfo{
		{PolicyRoundRobin, "max-score-picker; no scorer",
			[]string{"pkg/epp/framework/plugins/scheduling/picker/maxscore/picker.go",
				"pkg/epp/framework/plugins/scheduling/picker/common.go"}},
		{PolicyLeastActiveRequests, "inflight-load-producer; active-request-scorer; max-score-picker",
			[]string{"pkg/epp/framework/plugins/scheduling/scorer/activerequest/active_request.go"}},
		{PolicyDefault, "queue-scorer (2); kv-cache-utilization-scorer (2); prefix-cache-scorer (3); max-score-picker",
			[]string{"pkg/epp/config/loader/defaults.go"}},
	}
}

// PolicyNames lists the names of the policies llm-d's router offers.
func PolicyNames() []string {
	var out []string
	for _, i := range Policies() {
		out = append(out, string(i.Policy))
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
	for _, i := range Policies() {
		if i.Policy == p {
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

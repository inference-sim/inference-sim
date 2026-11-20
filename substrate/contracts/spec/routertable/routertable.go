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
	// Builds are the builds of the router that a run's recipe names by a version, not by a commit
	// (a wheel such as "1.3.0.dev20260601"), with the commits each could have been made from. A
	// build is covered only when the files that decide the mode are byte-identical at every one of
	// those commits, so a citation at any of them shows the behaviour of the build. Absent when the
	// entry is established for the commits in Revisions only. Resolving a build string is
	// ResolveBuild.
	Builds []Build `yaml:"builds,omitempty"`
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

// Build sources: how the commits of a build were found.
const (
	// BuildNightly: a development wheel, built from the head of the main branch on the date in its
	// version, at a time the recipe does not record. Its commits are every first-parent commit of
	// main from the head at the start of that day (UTC) to the head at its end.
	BuildNightly = "nightly"
	// BuildRelease: a release wheel, built from the commit its release tag points to.
	BuildRelease = "release"
)

// Build is one build of a router that a run names by version, with every commit it could be.
//
// A recipe pins the router by a wheel version, which is no commit. A nightly wheel is made from
// whatever the main branch held when the nightly ran, so a single version stands for a window of
// commits. The entry covers the build only if what decides the mode did not change inside the
// window: each file in Files has one blob identifier at every commit of Commits. Where it changes,
// the build is not listed, and a run on it stays unmapped.
type Build struct {
	// Name is the version as the recipe states it, for example "1.3.0.dev20260601" or "1.2.1".
	// Each build of an entry once.
	Name string `yaml:"name"`
	// Source says how Commits were found: "nightly" or "release".
	Source string `yaml:"source"`
	// Window says in words how Commits were found: the dates and the rule, so a reader can redo
	// it from the router's history.
	Window string `yaml:"window"`
	// Commits are every commit the build could have been made from, each a full 40-character
	// commit, sorted, each once. For a release, the one commit of the tag.
	Commits []string `yaml:"commits"`
	// Files are the files that decide the mode, each with its blob identifier, the same at every
	// commit of Commits. Every router-side citation at one of Commits is of a file listed here.
	Files []BuildFile `yaml:"files"`
}

// BuildFile is one file that decides the mode and the git blob identifier it has at every commit
// of its build.
type BuildFile struct {
	// Path is the file, from the router repository's root.
	Path string `yaml:"path"`
	// Blob is the file's git blob identifier (40 lower-case hex digits), as `git rev-parse
	// <commit>:<path>` prints it at every commit of the build.
	Blob string `yaml:"blob"`
	// Rule names the rewrite set that makes the file's other blobs the same code, when it has
	// more than one blob in the window: NormalFormLoggingOnly for a Rust file, NormalFormPythonRouting for a Python file. Absent when Blob is the file's
	// only blob at every commit.
	Rule string `yaml:"rule,omitempty"`
	// NormalForm is the SHA-256 (64 lowercase hex digits) of Blob's normal form under Rule, as
	// catalog/scripts/router_normal_form.py prints it. Present exactly when Variants is.
	NormalForm string `yaml:"normal_form,omitempty"`
	// Variants are the file's other blobs in the window, each with the SHA-256 of its normal
	// form, which equals NormalForm. The file then differs between the window's commits only in
	// what Rule removes, which changes no value the router computes.
	Variants []BuildVariant `yaml:"variants,omitempty"`
}

// BuildVariant is another blob a deciding file has inside a build's window.
type BuildVariant struct {
	// Blob is the git blob identifier (40 lowercase hex digits).
	Blob string `yaml:"blob"`
	// NormalForm is the SHA-256 of the blob's normal form under the file's Rule.
	NormalForm string `yaml:"normal_form"`
}

// NormalFormLoggingOnly is the rule set "rust-logging-only" of catalog/scripts/router_normal_form.py:
// delete logging statements, inline a binding that only carries a value to a deleted statement,
// drop an unread element of a tuple binding, ignore blank lines. A difference left after these is
// a difference in what the code computes, and the build is not covered.
const NormalFormLoggingOnly = "rust-logging-only"

// NormalFormPythonRouting is the rule set "python-routing-statements" of the same script, for a
// Python deciding file: compare only the statements that name a router (with the definitions
// around them), so a difference in code that names no router is not a difference. A difference
// in a statement that names one is, and the build is not covered.
const NormalFormPythonRouting = "python-routing-statements"

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
	for _, b := range e.Builds {
		for _, c := range b.Commits {
			if c == revision {
				return true
			}
		}
	}
	return false
}

// CoversBuild reports whether the entry is established for the build a run's recipe names by
// version (a wheel, for example "1.3.0.dev20260601"). An empty name is a recipe that states no
// build, which no entry covers.
func (e *Entry) CoversBuild(build string) bool {
	return e.BuildOf(build) != nil
}

// BuildOf returns the entry's build of that name, or nil.
func (e *Entry) BuildOf(build string) *Build {
	if build == "" {
		return nil
	}
	for i := range e.Builds {
		if e.Builds[i].Name == build {
			return &e.Builds[i]
		}
	}
	return nil
}

// Revision returns the commit a point record names for the build: the newest commit of its
// window, which the entry covers and at which the build's files are those the entry cites. Empty
// when b is nil.
func (b *Build) Revision() string {
	if b == nil || len(b.Commits) == 0 {
		return ""
	}
	return b.Commits[len(b.Commits)-1]
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
	return resolve(entries, router, mode, affinityTTLS, func(e *Entry) bool { return e.Covers(revision) })
}

// ResolveBuild is Resolve for a run whose recipe names the router build by version, as a wheel
// (for example "1.3.0.dev20260601"), not by a commit. The build is covered when an entry with the
// run's key lists it among its Builds; a version the entry does not list, and a recipe that
// states none (build ""), are OutcomeRevisionNotCovered: what the entry shows is not shown for
// that build, so it is not inherited.
func ResolveBuild(entries []*Entry, router, mode string, affinityTTLS *int, build string) (*Entry, Outcome) {
	return resolve(entries, router, mode, affinityTTLS, func(e *Entry) bool { return e.CoversBuild(build) })
}

func resolve(entries []*Entry, router, mode string, affinityTTLS *int, covers func(*Entry) bool) (*Entry, Outcome) {
	for _, e := range entries {
		if e.Router != router || e.Mode != mode {
			continue
		}
		if (e.SessionAffinityTTLS == nil) != (affinityTTLS == nil) ||
			(e.SessionAffinityTTLS != nil && *e.SessionAffinityTTLS != *affinityTTLS) {
			continue
		}
		switch {
		case !covers(e):
			return e, OutcomeRevisionNotCovered
		case e.Policy == PolicyNone:
			return e, OutcomeNoPolicy
		}
		return e, OutcomeMapped
	}
	return nil, OutcomeNoEntry
}

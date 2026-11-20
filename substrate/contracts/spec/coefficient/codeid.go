package coefficient

import (
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/substrate/contracts/check"
)

// The engine-path lookup. A host, scheduler or graph-launch cost is a property of the code the
// engine runs to do that work, not of a kernel and not of the engine's version string: a release
// that leaves the code alone keeps its coefficient and one that changes it does not. An entry
// names the code it is valid for by CodeID (an opaque identity an engine derives from the
// closure of functions that cost depends on, engines/*/code_paths.json), and a lookup matches it
// only when the engine states that CodeID for the path the entry's name prices. There is no
// fallback to a sibling CodeID, as there is none to a sibling kernel build (VAL-8).

// The engine-path code paths a coefficient is bound to. An engine states one CodeID per path in
// its plan (plan.HostPlan.CodeID, plan.GraphPolicy.CodeID).
const (
	// CodePathHost is the host and scheduler path: request admission, per-token and completion
	// work, and the scheduler's per-request cost.
	CodePathHost = "host"
	// CodePathGraphs is the launch path: eager, per-kernel and graph-replay launch costs.
	CodePathGraphs = "graphs"
)

// CodePathOf returns the engine-path code a coefficient of this name prices, or "" for a name
// that prices none (kernels, memory sizes, fabric, the host link and the UVA gather, which are
// hardware rates). It is the one place the binding is decided, as PrimitiveOf is for kernels.
func CodePathOf(name string) string {
	switch {
	case strings.HasPrefix(name, "host_admission_"), name == "host_output_token", name == "host_completion",
		name == "host_step_per_request":
		return CodePathHost
	case strings.HasPrefix(name, "host_launch_"), name == "host_replay_graph_per_step":
		return CodePathGraphs
	}
	return ""
}

// CodeMatch says how an entry met the engine-path code that runs.
type CodeMatch string

const (
	// CodeUnkeyed: the entry states no CodeID. It applies to every code, and nothing shows it
	// was fitted on the code that runs.
	CodeUnkeyed CodeMatch = "unkeyed"
	// CodeExact: the entry lists the CodeID the engine states for the entry's path.
	CodeExact CodeMatch = "exact"
	// CodeRefused: the entry lists CodeIDs and none is the one that runs. A lookup does not
	// return it.
	CodeRefused CodeMatch = "refused"
)

// MatchCode says how the entry meets the CodeIDs that run, by code path. An entry that lists none
// is unkeyed. A keyed entry is exact only when its name binds to a code path and the engine
// states that path's CodeID and the entry lists it. A different CodeID, no CodeID stated, or a
// name that prices no engine-path code is refused.
func (e Entry) MatchCode(running map[string]string) CodeMatch {
	if len(e.CodeIDs) == 0 {
		return CodeUnkeyed
	}
	path := CodePathOf(e.Name)
	if path == "" || running[path] == "" {
		return CodeRefused
	}
	for _, id := range e.CodeIDs {
		if id == running[path] {
			return CodeExact
		}
	}
	return CodeRefused
}

// CodesCompatible reports whether one set of running CodeIDs can make both entries match:
// either is unkeyed, or they list a CodeID in common. The registry validator uses it, with
// ImplsCompatible, to decide whether two entries can be chosen between (X15).
func CodesCompatible(a, b Entry) bool {
	if len(a.CodeIDs) == 0 || len(b.CodeIDs) == 0 {
		return true
	}
	for _, x := range a.CodeIDs {
		for _, y := range b.CodeIDs {
			if x == y {
				return true
			}
		}
	}
	return false
}

// DescribeCodeIDs renders CodeIDs for an error or a report, sorted so the text is stable.
func DescribeCodeIDs(ids []string) string {
	out := append([]string(nil), ids...)
	sort.Strings(out)
	return strings.Join(out, ", ")
}

// ValidateCodeIDs checks a list of CodeIDs: each a lower-case hexadecimal digest of at least 12
// digits, and none repeated. at names the list in a message.
func ValidateCodeIDs(p *validate.Problems, at string, ids []string) {
	seen := map[string]bool{}
	for _, id := range ids {
		if !hexHash(id) {
			p.Field(at, "%q must be a lower-case hexadecimal digest of at least 12 digits", id)
		}
		if seen[id] {
			p.Field(at, "duplicate code id %s", id)
		}
		seen[id] = true
	}
}

func (e Entry) validateCodeIDs(p *validate.Problems, at string) {
	ValidateCodeIDs(p, at+".code_ids", e.CodeIDs)
	if len(e.CodeIDs) > 0 && CodePathOf(e.Name) == "" && e.Name != "" {
		p.Field(at+".code_ids", "%q prices no engine-path code (it binds to no code path, "+
			"coefficient.CodePathOf), so a code key on it could never match", e.Name)
	}
}

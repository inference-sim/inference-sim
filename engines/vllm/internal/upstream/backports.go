// Original to blis-vllm.

package upstream

import (
	"fmt"
	"path"
	"regexp"
	"strings"
)

// A Backport is one vLLM commit, newer than Commit, whose model code a catalog model needs and
// the build carries before the next upgrade moves the pin past it (DC1, LIF-6). It carries model
// code only: the model directories, config parsers and the architecture registry of the
// release. A model that lands upstream between upgrades is added as a Backport, not by moving
// the pin.
//
// The build still models the one commit named by Commit. A backport changes the plan and the
// trace of the models that use it (Architectures) and of no other: the equivalence check of
// core/kernelmodel (backport_equivalence_test.go) holds that for every catalog model.
type Backport struct {
	// Commit is the full 40-hex id of the vLLM commit, on vLLM's main.
	Commit string
	// Subject is the commit's subject line, as vLLM's history states it.
	Subject string
	// Architectures are the architectures (the catalog graph's global.architecture) whose model
	// code the commit adds or changes. A catalog model whose architecture is not listed does not
	// use the backport.
	Architectures []string
	// Paths are the vLLM paths the commit changes, relative to the repository root.
	Paths []string
}

// URL is the commit's page in vLLM's repository, the source a derived value cites.
func (b Backport) URL() string { return "https://github.com/vllm-project/vllm/commit/" + b.Commit }

// Uses reports whether a model of the given architecture uses the backport.
func (b Backport) Uses(architecture string) bool {
	for _, a := range b.Architectures {
		if a == architecture {
			return true
		}
	}
	return false
}

// backports is the Upstream record's list of backported commits. The pin carries none: no
// catalog model needs a commit past Commit (a real one is added here, with the patch it makes to
// the rules, when the first model lands upstream between two upgrades).
var backports = []Backport{}

// Backports returns the backported commits the build carries, in the order they were added.
func Backports() []Backport { return cloneBackports(backports) }

func cloneBackports(bs []Backport) []Backport {
	out := make([]Backport, len(bs))
	for i, b := range bs {
		b.Architectures = append([]string(nil), b.Architectures...)
		b.Paths = append([]string(nil), b.Paths...)
		out[i] = b
	}
	return out
}

// backportScope is the model code a backport may change, relative to the vLLM repository root:
// the model directories and architecture registry (vllm/model_executor/models/), the config
// parsers and the model-architecture config convertor (vllm/transformers_utils/). A name ending
// in "/" is a directory, any other name one file. A commit that changes anything else is not a
// backport: moving the pin carries it (design 12.5, FID-14).
var backportScope = []string{
	"vllm/model_executor/models/",
	"vllm/transformers_utils/configs/",
	"vllm/transformers_utils/config.py",
	"vllm/transformers_utils/config_parser_base.py",
	"vllm/transformers_utils/model_arch_config_convertor.py",
}

// BackportScope returns the paths a backport may change; see backportScope.
func BackportScope() []string { return append([]string(nil), backportScope...) }

func inBackportScope(p string) bool {
	for _, s := range backportScope {
		if p == s || (strings.HasSuffix(s, "/") && strings.HasPrefix(p, s)) {
			return true
		}
	}
	return false
}

var hexCommit = regexp.MustCompile(`^[0-9a-f]{40}$`)

// BackportProblems lists what is wrong with a list of backports recorded against the pinned
// commit: a commit that is not a full 40-hex id, the pin itself or a commit listed twice, an empty
// subject, no architecture or no path, an architecture listed twice, and a path that is empty,
// absolute, not clean, outside vllm/ or outside the model code a backport carries
// (BackportScope). An empty result means the record is well formed.
// Whether a commit is newer than the pin and changes only the paths listed is checked against a
// vLLM checkout, not here.
func BackportProblems(bs []Backport, pin string) []string {
	var bad []string
	seen := map[string]bool{}
	for i, b := range bs {
		at := fmt.Sprintf("backport %d (%s)", i, b.Commit)
		switch {
		case !hexCommit.MatchString(b.Commit):
			bad = append(bad, fmt.Sprintf("backport %d: commit %q is not a full 40-hex commit id", i, b.Commit))
		case b.Commit == pin:
			bad = append(bad, at+": the commit is the pin, which the build already models")
		case seen[b.Commit]:
			bad = append(bad, at+": the commit is listed twice")
		}
		seen[b.Commit] = true
		if strings.TrimSpace(b.Subject) == "" {
			bad = append(bad, at+": the subject is empty")
		}
		if len(b.Architectures) == 0 {
			bad = append(bad, at+": no architecture, so no model could be said to use it")
		}
		archs := map[string]bool{}
		for _, a := range b.Architectures {
			if strings.TrimSpace(a) == "" || a != strings.TrimSpace(a) {
				bad = append(bad, fmt.Sprintf("%s: architecture %q is empty or padded", at, a))
			}
			if archs[a] {
				bad = append(bad, fmt.Sprintf("%s: architecture %q is listed twice", at, a))
			}
			archs[a] = true
		}
		if len(b.Paths) == 0 {
			bad = append(bad, at+": no path, so the change it makes is not recorded")
		}
		for _, p := range b.Paths {
			switch {
			case p == "" || path.IsAbs(p) || path.Clean(p) != p || !strings.HasPrefix(p, "vllm/"):
				bad = append(bad, fmt.Sprintf("%s: path %q is not a clean path under vllm/", at, p))
			case !inBackportScope(p):
				bad = append(bad, fmt.Sprintf("%s: path %q is outside the model code a backport carries (%s)",
					at, p, strings.Join(backportScope, ", ")))
			}
		}
	}
	return bad
}

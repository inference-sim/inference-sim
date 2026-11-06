package main

import (
	"fmt"
	"go/version"
	"regexp"
	"sort"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/semver"
)

// The module paths X11 names.
const (
	upstreamRouter = "github.com/llm-d/llm-d-router"
	forkRouter     = "github.com/inference-sim/llm-d-router"
	simulator      = "github.com/inference-sim/inference-sim"
	blisPrefix     = "github.com/inference-sim/"
)

// engineModules are the modules whose plain tags inference-sim may exclude (X11, design
// §14.1): the engine modules, which never tag a plain v<upstream> under the current rule.
var engineModules = map[string]bool{
	"github.com/inference-sim/blis-vllm":   true,
	"github.com/inference-sim/blis-sglang": true,
}

// blisTag matches the tag forms of LIF-5: v<upstream>-blis.N, v<upstream>-post.K.blis.N
// and v<upstream>-rev.W.blis.N, with N, K and W from 1.
var blisTag = regexp.MustCompile(`^v[0-9]+\.[0-9]+\.[0-9]+-(?:(?:post|rev)\.[1-9][0-9]*\.)?blis\.[1-9][0-9]*$`)

// Module is one entry of go list -m -json all.
type Module struct {
	Path      string
	Version   string
	Main      bool
	GoVersion string
	Replace   *Module
}

// Input is what the lint reads: the main module's go.mod, the modules the build selects,
// the module graph's edges (go mod graph), whether a go.work file is committed, and the
// toolchain contracts/toolchain names.
type Input struct {
	GoMod     []byte
	Modules   []Module
	Graph     [][2]string // each edge "from@version" -> "to@version"
	GoWork    []string    // committed go.work and go.work.sum paths
	Toolchain string      // as contracts/toolchain writes it, go1.N.M
	GoVersion string      // what go env GOVERSION reports
}

// Lint returns every violation of X11 and LIF-2 in the input, sorted.
func Lint(in Input) ([]string, error) {
	mf, err := modfile.Parse("go.mod", in.GoMod, nil)
	if err != nil {
		return nil, err
	}
	var bad []string
	add := func(format string, args ...any) { bad = append(bad, fmt.Sprintf(format, args...)) }
	self := mf.Module.Mod.Path
	isSim := self == simulator

	for _, w := range in.GoWork {
		add("%s is committed; no repository commits a go.work file (X11)", w)
	}

	// replace directives.
	switch {
	case !isSim && len(mf.Replace) > 0:
		for _, r := range mf.Replace {
			add("go.mod:%d: replace %s; only inference-sim may carry a replace directive (X11)", r.Syntax.Start.Line, r.Old.Path)
		}
	case isSim:
		if len(mf.Replace) != 1 {
			add("go.mod has %d replace directives; inference-sim carries exactly one, for the router fork (X11)", len(mf.Replace))
		}
		for _, r := range mf.Replace {
			switch {
			case r.Old.Path != upstreamRouter || r.Old.Version != "":
				add("go.mod:%d: replace %s %s; the one replace substitutes %s, every version (X11)", r.Syntax.Start.Line, r.Old.Path, r.Old.Version, upstreamRouter)
			case r.New.Path != forkRouter:
				add("go.mod:%d: replace target %s; it must be %s (X11)", r.Syntax.Start.Line, r.New.Path, forkRouter)
			case !blisTag.MatchString(r.New.Version):
				add("go.mod:%d: replace target version %q is not a -blis.N tag (X11, LIF-5)", r.Syntax.Start.Line, r.New.Version)
			}
		}
	}

	// exclude directives.
	for _, e := range mf.Exclude {
		switch {
		case !isSim:
			add("go.mod:%d: exclude %s %s; only inference-sim may carry exclude directives (X11)", e.Syntax.Start.Line, e.Mod.Path, e.Mod.Version)
		case !engineModules[e.Mod.Path]:
			add("go.mod:%d: exclude %s; an exclude may name only an engine module (X11)", e.Syntax.Start.Line, e.Mod.Path)
		case semver.Prerelease(e.Mod.Version) != "" || semver.Build(e.Mod.Version) != "" || !semver.IsValid(e.Mod.Version):
			add("go.mod:%d: exclude %s %s; an exclude may name only a plain tag (X11, design §14.1)", e.Syntax.Start.Line, e.Mod.Path, e.Mod.Version)
		}
	}

	// No module other than inference-sim requires the router.
	for _, edge := range in.Graph {
		from, to := modPath(edge[0]), modPath(edge[1])
		if (to == upstreamRouter || to == forkRouter) && from != simulator {
			add("%s requires %s; only inference-sim may require the router (X11)", edge[0], to)
		}
	}

	// The versions the build selects (LIF-2): every BLIS module at an exact release tag,
	// and the router through its replace at a -blis.N tag.
	tc := in.Toolchain
	if !version.IsValid(tc) {
		return nil, fmt.Errorf("contracts/toolchain holds %q, not a Go version of the form go1.N.M", tc)
	}
	for _, m := range in.Modules {
		if m.GoVersion != "" && version.Compare("go"+m.GoVersion, tc) > 0 {
			add("%s declares go %s, above contracts/toolchain's %s (LIF-2)", m.Path, m.GoVersion, tc)
		}
		if m.Main {
			continue
		}
		if m.Path == upstreamRouter {
			if m.Replace == nil {
				add("%s %s is selected without the router fork's replace (X11)", m.Path, m.Version)
			} else if m.Replace.Path != forkRouter || !blisTag.MatchString(m.Replace.Version) {
				add("%s resolves to %s %s; it must resolve to %s at a -blis.N tag (LIF-2)", m.Path, m.Replace.Path, m.Replace.Version, forkRouter)
			}
			continue
		}
		if strings.HasPrefix(m.Path, blisPrefix) {
			v := m.Version
			if m.Replace != nil {
				add("%s is replaced by %s %s; a BLIS module resolves to its own release tag (X11)", m.Path, m.Replace.Path, m.Replace.Version)
				continue
			}
			switch {
			case !semver.IsValid(v) || isPseudo(v):
				add("%s resolves to %s, which is not a release tag (X11, LIF-2)", m.Path, v)
			case engineModules[m.Path] || m.Path == forkRouter:
				// Engines and the router fork never tag a plain v<upstream> (design §14.1).
				if !blisTag.MatchString(v) {
					add("%s resolves to %s; it must be a -blis.N tag (X11, LIF-5, design §14.1)", m.Path, v)
				}
			default:
				// The substrate and the simulator tag plain vX.Y.Z.
				if semver.Prerelease(v) != "" || semver.Build(v) != "" {
					add("%s resolves to %s; it must be a plain vX.Y.Z tag (X11, LIF-2)", m.Path, v)
				}
			}
		}
	}
	if in.GoVersion != "" && in.GoVersion != tc {
		add("go version reports %s; contracts/toolchain names %s, and every CI job installs exactly it with GOTOOLCHAIN=local (LIF-2)", in.GoVersion, tc)
	}
	sort.Strings(bad)
	return bad, nil
}

func modPath(s string) string {
	if i := strings.LastIndex(s, "@"); i >= 0 {
		return s[:i]
	}
	return s
}

var pseudo = regexp.MustCompile(`-(?:0\.)?[0-9]{14}-[0-9a-f]{12}$|\.0\.[0-9]{14}-[0-9a-f]{12}$`)

// isPseudo reports whether v is a Go pseudo-version, the form go get writes for a commit
// with no tag.
func isPseudo(v string) bool {
	return pseudo.MatchString(strings.TrimSuffix(v, "+incompatible"))
}

package stack

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/semver"
)

// CheckGoMod is the go.mod half of the selected-version lint (X11, LIF-2): exactly one replace,
// from RouterModule to RouterFork at a version (never a local path); every exclude names a
// plain tag of an engine module; no go.work beside it. It returns one message per finding.
func CheckGoMod(root string) ([]string, error) {
	path := filepath.Join(root, "go.mod")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	f, err := modfile.Parse(path, b, nil)
	if err != nil {
		return nil, err
	}
	return checkModFile(root, f), nil
}

func checkModFile(root string, f *modfile.File) []string {
	var out []string
	if len(f.Replace) != 1 {
		out = append(out, fmt.Sprintf("go.mod has %d replace directives; X11 allows exactly one, %s => %s",
			len(f.Replace), RouterModule, RouterFork))
	}
	for _, r := range f.Replace {
		switch {
		case r.Old.Path != RouterModule || r.New.Path != RouterFork:
			out = append(out, fmt.Sprintf("replace %s => %s: the only replace is %s => %s (X11)",
				r.Old.Path, r.New.Path, RouterModule, RouterFork))
		case r.New.Version == "":
			out = append(out, fmt.Sprintf("replace %s => %s names a directory, not a version (X11)", r.Old.Path, r.New.Path))
		case r.Old.Version != "":
			out = append(out, fmt.Sprintf("replace %s %s => ...: replace every version, not one", r.Old.Path, r.Old.Version))
		case !strings.Contains(r.New.Version, "-blis."):
			out = append(out, fmt.Sprintf("replace %s => %s %s: the fork is pinned at a -blis.N tag or a commit built on one",
				r.Old.Path, r.New.Path, r.New.Version))
		}
	}
	for _, e := range f.Exclude {
		if !isEngineModule(e.Mod.Path) || semver.Prerelease(e.Mod.Version) != "" || semver.Build(e.Mod.Version) != "" {
			out = append(out, fmt.Sprintf("exclude %s %s: an exclude may name only a plain tag of an engine module (X11)",
				e.Mod.Path, e.Mod.Version))
		}
	}
	if _, err := os.Stat(filepath.Join(root, "go.work")); err == nil {
		out = append(out, "go.work exists; no repository commits one (X11)")
	}
	return out
}

func isEngineModule(path string) bool {
	return path == VLLMModule || path == "github.com/inference-sim/blis-sglang"
}

// CheckGoVersions is LIF-2's go-directive check: no module in the build declares a go version
// above the substrate's contracts/toolchain. toolchain is go1.N.M.
func CheckGoVersions(root, toolchain string) ([]string, error) {
	mods, err := listModules(root)
	if err != nil {
		return nil, err
	}
	limit := "v" + strings.TrimPrefix(toolchain, "go")
	if !semver.IsValid(limit) {
		return nil, fmt.Errorf("toolchain %q is not go1.N.M", toolchain)
	}
	var out []string
	for _, m := range mods {
		gv := m.GoVersion
		if m.Replace != nil {
			gv = m.Replace.GoVersion
		}
		if gv == "" {
			continue
		}
		if semver.Compare("v"+gv, limit) > 0 {
			out = append(out, fmt.Sprintf("%s declares go %s, above contracts/toolchain %s (LIF-2)", m.Path, gv, toolchain))
		}
	}
	return out, nil
}

// Diff compares a committed record with the one the build computes and returns one line per
// difference. SIM-3's release check is Diff of the tagged build's stack.json against Build.
func Diff(committed, built *Stack) []string {
	var out []string
	if committed.Toolchain != built.Toolchain {
		out = append(out, fmt.Sprintf("toolchain: stack.json %s, build %s", committed.Toolchain, built.Toolchain))
	}
	if committed.GuidesCommit != built.GuidesCommit {
		out = append(out, fmt.Sprintf("guides_commit: stack.json %s, guides.lock %s", committed.GuidesCommit, built.GuidesCommit))
	}
	if committed.Router != built.Router {
		out = append(out, fmt.Sprintf("router: stack.json %+v, build %+v", committed.Router, built.Router))
	}
	seen := map[string]bool{}
	for _, m := range built.Modules {
		seen[m.Path] = true
		c, ok := committed.Module(m.Path)
		switch {
		case !ok:
			out = append(out, fmt.Sprintf("%s %s: selected by the build, missing from stack.json", m.Path, m.Version))
		case c != m:
			out = append(out, fmt.Sprintf("%s: stack.json %+v, build %+v", m.Path, c, m))
		}
	}
	for _, m := range committed.Modules {
		if !seen[m.Path] {
			out = append(out, fmt.Sprintf("%s: in stack.json, not selected by the build", m.Path))
		}
	}
	engines := map[string]Upstream{}
	for _, e := range built.Engines {
		engines[e.Kind] = e
	}
	for _, e := range committed.Engines {
		b, ok := engines[e.Kind]
		if !ok || b != e {
			out = append(out, fmt.Sprintf("engine %s: stack.json %+v, build %+v", e.Kind, e, b))
		}
		delete(engines, e.Kind)
	}
	for k, e := range engines {
		out = append(out, fmt.Sprintf("engine %s %s: run by the build, missing from stack.json", k, e.Version))
	}
	return out
}

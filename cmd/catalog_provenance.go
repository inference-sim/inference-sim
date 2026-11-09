package cmd

import (
	"runtime/debug"

	sim "github.com/inference-sim/inference-sim/sim"
)

// kernelProvenance is each opened pool's kernel evidence (#1942), keyed by pool role: the
// colocated pool, or the prefill and decode pools of a disaggregated run; empty when no
// kernel was opened.
func kernelProvenance(colocated *engineModel, pd *kernelPools) map[string]sim.KernelPoolProvenance {
	out := map[string]sim.KernelPoolProvenance{}
	if pd != nil {
		out[string(pd.prefill.Role())] = provenanceOf(pd.prefill.Provenance())
		out[string(pd.decode.Role())] = provenanceOf(pd.decode.Provenance())
	} else if colocated != nil {
		out[string(colocated.Role())] = provenanceOf(colocated.Provenance())
	}
	return out
}

// buildRevision is the commit a binary was built from, set with
//
//	go build -ldflags "-X github.com/inference-sim/inference-sim/cmd.buildRevision=$(git rev-parse HEAD)$(git diff --quiet HEAD || echo -dirty)"
//
// for a build the toolchain cannot stamp: Go stamps vcs.revision only when the repository's
// .git is a directory, so a git worktree (whose .git is a file) builds with no VCS settings.
// A stamped vcs.revision wins over it.
var buildRevision string

// versionsFrom extracts the build pins from build information. A local build's main module
// is "(devel)"; its VCS revision, when the toolchain stamped one, is appended so the file
// still names the commit. A version the build information does not carry (a test binary
// lists no dependencies) is "unknown", never guessed.
func versionsFrom(info *debug.BuildInfo) *sim.BuildVersions {
	v := &sim.BuildVersions{BLIS: "unknown", Kernel: "unknown", Schemas: "unknown"}
	if info == nil {
		return v
	}
	if info.Main.Version != "" {
		v.BLIS = info.Main.Version // empty when built from a file list (go build main.go)
	}
	var revision, dirty string
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			revision = s.Value
		case "vcs.modified":
			if s.Value == "true" {
				dirty = "-dirty"
			}
		}
	}
	if revision != "" {
		v.BLIS += "+" + revision + dirty
	} else if buildRevision != "" {
		v.BLIS += "+" + buildRevision
	}
	for _, dep := range info.Deps {
		switch dep.Path { // the module's own path; a replacement keeps it here
		case "github.com/inference-sim/blis-substrate":
			// One module holds the cost model and the contracts (design §3.2), so the
			// kernel and the schemas are built at the same version.
			v.Kernel = depVersion(dep)
			v.Schemas = v.Kernel
		}
	}
	return v
}

// depVersion is the version a dependency was built at: its replacement's when replaced by
// another module version, "replace:<dir>" when replaced by a filesystem directory (which has
// no version), else its own.
func depVersion(dep *debug.Module) string {
	if r := dep.Replace; r != nil {
		if r.Version != "" {
			return r.Version
		}
		return "replace:" + r.Path
	}
	return dep.Version
}

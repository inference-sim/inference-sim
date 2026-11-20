// Original to blis-sglang.

// Package blissglang is a Go model of the SGLang inference engine's scheduling, KV-cache,
// hierarchical-cache, disaggregation and metrics behaviour, for a discrete-event simulator to call
// through the engine-neutral contract in github.com/inference-sim/blis-substrate/contracts/engine.
//
// One build of this module models one SGLang release, Upstream. Its behaviour is checked against
// that release's own code (oracle/): the same inputs, fed to the real SGLang classes in Python and
// to this module in Go, must make the same decisions step by step.
//
// The simulator owns time. Nothing in this module reads a clock or starts a goroutine: every call
// carries the simulated instant as a value.
package blissglang

import (
	"fmt"
	"runtime/debug"

	"github.com/inference-sim/blis-substrate/contracts/harness"
)

// Kind is the engine kind this module registers with the contracts' engine factory.
const Kind = "sglang"

// Upstream identifies the SGLang release this build models. The tag sits on SGLang's
// release/v0.5.23 branch and is not an ancestor of its main, so an upgrade compares release tag
// with release tag.
var Upstream = harness.Upstream{
	Version: "0.5.23",
	Tag:     "v0.5.23",
	Commit:  "297e54fd105e974cdf401e5b4d20253a22c16f90",
}

// Provenance is what a run must record to be reproduced: the engine kind, the upstream release it
// models and this module's own revision. A simulator writes it into its results beside the
// kernel's provenance.
type Provenance struct {
	EngineKind      string `json:"engine_kind" yaml:"engine_kind"`
	UpstreamVersion string `json:"upstream_version" yaml:"upstream_version"`
	UpstreamCommit  string `json:"upstream_commit" yaml:"upstream_commit"`
	// Module is this module's version as the build recorded it: a release tag, or a
	// pseudo-version naming the commit; "(devel)" in a build of this repository itself.
	Module string `json:"module" yaml:"module"`
}

// ModulePath is this module's import path.
const ModulePath = "github.com/inference-sim/blis-sglang"

// moduleVersion reports this module's version from the running binary's build info.
func moduleVersion() string {
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown"
	}
	if bi.Main.Path == ModulePath {
		return bi.Main.Version
	}
	for _, d := range bi.Deps {
		if d.Path == ModulePath {
			if d.Replace != nil {
				return fmt.Sprintf("%s => %s %s", d.Version, d.Replace.Path, d.Replace.Version)
			}
			return d.Version
		}
	}
	return "unknown"
}

// ProvenanceOf returns the provenance of this build.
func ProvenanceOf() Provenance {
	return Provenance{
		EngineKind:      Kind,
		UpstreamVersion: Upstream.Version,
		UpstreamCommit:  Upstream.Commit,
		Module:          moduleVersion(),
	}
}

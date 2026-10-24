// Original to blis-vllm.

// Package blisvllm is a Go model of the vLLM inference engine's scheduling, KV-cache,
// connector and metrics behaviour, for a discrete-event simulator to call through the
// engine-neutral contract in github.com/inference-sim/blis-schemas/engine.
//
// One build of this module models one vLLM release, Upstream. Its behaviour is checked
// against that release's own code in CI (see docs/testing.md): the same inputs, fed to the
// real vLLM classes in Python and to this module in Go, must make the same decisions step by
// step.
//
// The simulator owns time. Nothing in this module reads a clock or starts a goroutine: every
// call carries the simulated instant as a value, and where a vLLM rule makes work wait until
// a later instant, the engine reports that instant through NextWake.
package blisvllm

import (
	"fmt"
	"runtime/debug"
)

// Kind is the engine kind this module registers with blis-schemas' engine factory.
const Kind = "vllm"

// Upstream identifies the vLLM release this build models.
var Upstream = UpstreamRelease{
	Version: "0.32.0",
	Tag:     "v0.32.0",
	Commit:  "5df63a7203cdb9328d824a27b682e77c7b9928be",
}

// UpstreamRelease is one vLLM release.
type UpstreamRelease struct {
	Version string
	Tag     string
	Commit  string
}

// Provenance is what a run must record to be reproduced: the engine kind, the upstream
// release it models and this module's own revision. A simulator writes it into its results
// beside the kernel's provenance.
type Provenance struct {
	EngineKind      string `json:"engine_kind" yaml:"engine_kind"`
	UpstreamVersion string `json:"upstream_version" yaml:"upstream_version"`
	UpstreamCommit  string `json:"upstream_commit" yaml:"upstream_commit"`
	// Module is this module's version as the build recorded it: a release tag, or a
	// pseudo-version naming the commit; "(devel)" in a build of this repository itself.
	Module string `json:"module" yaml:"module"`
	// Behaviour names the behaviour the engine executes: "vllm-0.32.0", or a compatibility
	// behaviour a scenario selected (see Compat).
	Behaviour string `json:"behaviour" yaml:"behaviour"`
}

// ModulePath is this module's import path.
const ModulePath = "github.com/inference-sim/blis-vllm"

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

// ProvenanceFor returns the provenance of an engine executing behaviour.
func ProvenanceFor(behaviour string) Provenance {
	return Provenance{
		EngineKind:      Kind,
		UpstreamVersion: Upstream.Version,
		UpstreamCommit:  Upstream.Commit,
		Module:          moduleVersion(),
		Behaviour:       behaviour,
	}
}

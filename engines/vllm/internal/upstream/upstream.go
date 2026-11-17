// Original to blis-vllm.

// Package upstream names the vLLM release this build models, in one place. provenance.go's
// Upstream record, the engine factory's Version and every oracle diff test read it, so moving
// the build to another release (design §12.4) changes these constants and nothing else.
package upstream

const (
	// Version is the release's version, as a scenario's engine.version states it.
	Version = "0.33.0"
	// Tag is the release's git tag.
	Tag = "v0.33.0"
	// Commit is the full commit the tag names; every golden trace records it.
	Commit = "4ba787ec9959e566d3b587a796dff07673324da5"
)

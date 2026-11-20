// Original to blis-vllm.

// Package upstream names the vLLM release this build models, in one place. provenance.go's
// Upstream record, the engine factory's Version and every oracle diff test read it, so moving
// the build to another release (design §12.4) changes these constants and nothing else.
package upstream

const (
	// Version is the release's version, as stack.json and the Result's engine.version state it.
	Version = "0fda86a35419"
	// Tag is the release's git tag, or the first 12 digits of the commit when the build models a commit that has none.
	Tag = "0fda86a35419"
	// Commit is the full commit the tag names; every golden trace records it.
	Commit = "0fda86a35419c9c70dbfada224c838c10fbb62ad"
)

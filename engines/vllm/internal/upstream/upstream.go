// Original to blis-vllm.

// Package upstream names the vLLM release this build models, in one place. provenance.go's
// Upstream record, the engine factory's Version and every oracle diff test read it, so moving
// the build to another release (design §12.4) changes these constants and nothing else.
package upstream

const (
	// Version is the release's version, as stack.json and the Result's engine.version state it.
	Version = "0.34.0+0fda86a35419"
	// Tag names the release in a ported file's provenance header. vLLM has cut no release at this
	// commit (it is 1043 commits past v0.34.0rc0), so the tag is the next release's number with
	// the commit's first 12 digits as its build label, not a git tag.
	Tag = "v0.34.0+0fda86a35419"
	// Commit is the full commit the tag names; every golden trace records it.
	Commit = "0fda86a35419c9c70dbfada224c838c10fbb62ad"

	// OracleVersion and OracleCommit are the release the committed oracle recordings (golden
	// traces under core/*/testdata/golden*, oracle/metrics) were taken at. The engine models one
	// commit (Commit), so a recording is evidence of the release it was run against, which is
	// not the pin until a maintainer re-records it (oracle/README.md). A golden that was derived
	// from the pinned source instead says derived_from_source and is held to Tag and Commit.
	OracleVersion = "0.33.0"
	OracleTag     = "v0.33.0"
	OracleCommit  = "4ba787ec9959e566d3b587a796dff07673324da5"
)

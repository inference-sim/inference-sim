// Package rules is the engine-neutral contract for validation that depends on an engine
// release.
//
// The separation from field-level validation is the point of this package. A field check
// asks whether a document is structurally a scenario: are the counts positive, does the
// arithmetic close, is the enumerated value one this schema knows. Those questions have the
// same answer in every engine release, so the answers live with the types and change only
// when the schema does.
//
// A rule asks whether a document describes a deployment a PARTICULAR engine release would
// run. Backend names appear and disappear, defaults flip, a feature gains a fourth
// disqualifying condition. Each of those is a fact about a release, so it lives with the
// engine model that owns the release, and this schema names only the shape such rules take:
// EngineRules, which a caller supplies for the release a scenario names, and the Input and
// Rule a release's checks are written in. A pool's engine block is opaque here
// (deployment.EngineBlock); the release's rules open it.
package rules

// ArchDeepseekV41 and ArchDeepseekV4 are DeepSeek-V4.1's and DeepSeek-V4's architecture
// names, the values of their model configurations' architectures[0], which several releases'
// rules key on.
const (
	ArchDeepseekV41 = "DeepseekV41ForCausalLM"
	ArchDeepseekV4  = "DeepseekV4ForCausalLM"
)

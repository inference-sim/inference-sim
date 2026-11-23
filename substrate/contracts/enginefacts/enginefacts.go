// Package enginefacts is the seam through which the cost kernel asks an engine release for the
// facts it does not own: EngineRules, which the caller supplies for the release a scenario
// names, and the KV-cache grouping and CUDA graph capture types its answers are written in.
//
// It holds no validation. A document's field checks are the schema's, and what a release
// accepts is the engine module's own refusal (contracts.EngineChecker, the engine layer of
// SUB-12); no rule layer lives here. What the interface carries is lowering, and it is
// implemented by the engine module that owns the release (the vLLM and SGLang engine modules).
package enginefacts

// ArchDeepseekV41 and ArchDeepseekV4 are DeepSeek-V4.1's and DeepSeek-V4's architecture
// names, the values of their model configurations' architectures[0], which several releases'
// rules key on.
const (
	ArchDeepseekV41 = "DeepseekV41ForCausalLM"
	ArchDeepseekV4  = "DeepseekV4ForCausalLM"
)

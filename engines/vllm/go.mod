module github.com/inference-sim/blis-vllm

go 1.27.2

require (
	// TEMPORARY: the head of inference-sim/blis-schemas#102 (the engine contract and
	// rules.EngineRules). Moves to a blis-schemas release tag before anything merges.
	github.com/inference-sim/blis-schemas v0.6.1-0.20261025010642-7b94c7c25d3f
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

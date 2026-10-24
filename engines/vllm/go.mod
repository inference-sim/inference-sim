module github.com/inference-sim/blis-vllm

go 1.27.2

require (
	// TEMPORARY: the head of inference-sim/blis-schemas#102 (the engine contract and
	// rules.EngineRules). Moves to a blis-schemas release tag before anything merges.
	github.com/inference-sim/blis-schemas v0.6.1-0.20261024105647-9a71eeb47a9e
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

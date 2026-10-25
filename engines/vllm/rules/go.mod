module github.com/inference-sim/blis-vllm/rules

go 1.27.2

// TEMPORARY: the head of inference-sim/blis-schemas#102 (rules.EngineRules and the
// exported Problems aliases these packs need). Moves to a blis-schemas release tag before
// anything merges.
require github.com/inference-sim/blis-schemas v0.6.1-0.20261025010642-7b94c7c25d3f

require gopkg.in/yaml.v3 v3.0.1 // indirect

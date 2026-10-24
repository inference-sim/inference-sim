module github.com/inference-sim/blis-vllm/rules

go 1.27.2

// TEMPORARY: the head of inference-sim/blis-schemas#102 (rules.EngineRules and the
// exported Problems aliases these packs need). Moves to a blis-schemas release tag before
// anything merges.
require github.com/inference-sim/blis-schemas v0.6.1-0.20261024062559-1571a115227a

require gopkg.in/yaml.v3 v3.0.1 // indirect

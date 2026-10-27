module github.com/inference-sim/blis-vllm/rules

go 1.24

// TEMPORARY: inference-sim/blis-schemas#102 at dd46797 (https://github.com/inference-sim/blis-schemas/pull/102:
// rules.EngineRules, the Problems aliases and deployment.EngineBlock). Moves to a blis-schemas release tag before
// anything merges (blis-latency-kernel#88, which gives the tags' order).
require github.com/inference-sim/blis-schemas v0.7.1-0.20261027022131-dd4679743fde

require gopkg.in/yaml.v3 v3.0.1 // indirect

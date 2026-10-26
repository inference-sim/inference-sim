module github.com/inference-sim/blis-vllm/rules

go 1.24

// TEMPORARY: inference-sim/blis-schemas#102 at 630937e (rules.EngineRules and the
// exported Problems aliases these packs need; #102's later commits change only comments). Moves to a blis-schemas release tag before
// anything merges (blis-latency-kernel#88, which gives the tags' order).
require github.com/inference-sim/blis-schemas v0.6.1-0.20261025121631-630937e6f21f

require gopkg.in/yaml.v3 v3.0.1 // indirect

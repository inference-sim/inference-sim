module github.com/inference-sim/blis-sglang

go 1.27.2

// TEMPORARY: the head of inference-sim/blis-substrate#95 (contracts/harness), pinned by pseudo-version
// until it merges; then re-pin to its merge commit or a tag, regenerate the harness copies and rerun CI.
require github.com/inference-sim/blis-substrate v0.14.2-0.20261117064636-bfa2cb3b20b0

require gopkg.in/yaml.v3 v3.0.1 // indirect

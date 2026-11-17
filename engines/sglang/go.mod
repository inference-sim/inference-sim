module github.com/inference-sim/blis-sglang

go 1.27.2

// TEMPORARY: the head of inference-sim/blis-substrate#95 (contracts/harness), pinned by pseudo-version
// until it merges; then re-pin to its merge commit or a tag, regenerate the harness copies and rerun CI.
require github.com/inference-sim/blis-substrate v0.14.1-0.20261116203115-5c16d8682c35

module github.com/inference-sim/blis-vllm

go 1.27.2

require (
	// TEMPORARY: the head of inference-sim/blis-substrate#95 (contracts/harness), pinned by pseudo-version
	// until it merges; then re-pin to its merge commit or a tag, regenerate the harness copies and rerun CI.
	github.com/inference-sim/blis-substrate v0.14.1-0.20261116203115-5c16d8682c35
	github.com/sirupsen/logrus v1.9.4
	github.com/stretchr/testify v1.12.1
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

require (
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/sys v0.13.0 // indirect
)

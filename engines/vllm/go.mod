module github.com/inference-sim/blis-vllm

go 1.27.2

require (
	// TEMPORARY: the head of inference-sim/blis-latency-kernel#82 (the kernel takes its rules
	// from the caller), which compat/kernelmodel builds against. Moves to a release tag before
	// anything merges (blis-latency-kernel#88).
	github.com/inference-sim/blis-latency-kernel v0.3.1-0.20261025021202-515d766bbac2
	// TEMPORARY: the head of inference-sim/blis-schemas#102 (the engine contract and
	// rules.EngineRules). Moves to a blis-schemas release tag before anything merges.
	github.com/inference-sim/blis-schemas v0.6.1-0.20261025010642-7b94c7c25d3f
	// The rules packs, a separate module in this repository (rules/); compat/kernelmodel hands
	// them to the kernel. Pinned at this repository's own commit f43fad16.
	github.com/inference-sim/blis-vllm/rules v0.0.0-20261025013633-f43fad16b584
	github.com/sirupsen/logrus v1.9.4
	github.com/stretchr/testify v1.12.1
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

require (
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/sys v0.13.0 // indirect
)

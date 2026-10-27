module github.com/inference-sim/blis-vllm

go 1.27.2

require (
	// TEMPORARY: inference-sim/blis-latency-kernel#82 at 4a10ea9 (the kernel takes its rules and its
	// engine settings from the caller), which compat/kernelmodel builds against. Moves to a release
	// tag before anything merges (blis-latency-kernel#88).
	github.com/inference-sim/blis-latency-kernel v0.3.1-0.20261027011803-4a10ea956ce2
	// TEMPORARY: inference-sim/blis-schemas#102 at ddfd45b (the engine contract, rules.EngineRules,
	// rules.CUDAGraphCaptureInput and deployment.EngineBlock). Moves to a blis-schemas release tag
	// before anything merges (blis-latency-kernel#88, which gives the tags' order).
	github.com/inference-sim/blis-schemas v0.6.1-0.20261027005726-ddfd45b9e944
	// TEMPORARY: the rules packs, a separate module in this repository (rules/); compat/kernelmodel
	// hands them to the kernel, and the v0.32 factory applies rules/v0_32; rules/engineconfig decodes
	// a pool's engine block. Pinned at this repository's own commit 480d434 (rules/engineconfig). A
	// module requiring this one selects at least this pin. Moves to a rules/ release tag
	// (blis-latency-kernel#88).
	github.com/inference-sim/blis-vllm/rules v0.0.0-20261027010410-480d4340c353
	github.com/sirupsen/logrus v1.9.4
	github.com/stretchr/testify v1.12.1
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

require (
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/sys v0.13.0 // indirect
)

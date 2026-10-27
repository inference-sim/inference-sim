module github.com/inference-sim/blis-vllm

go 1.27.2

require (
	// TEMPORARY: inference-sim/blis-latency-kernel#82 at 625cd1d (the kernel takes its rules and its
	// engine settings from the caller), which compat/kernelmodel builds against. Moves to a release
	// tag before anything merges (blis-latency-kernel#88).
	github.com/inference-sim/blis-latency-kernel v0.3.1-0.20261027020405-625cd1de5208
	// TEMPORARY: inference-sim/blis-schemas#102 at 2c38342 (the engine-neutral contract: a pool's
	// engine block held unopened as deployment.EngineBlock, and the engine's rules from the caller).
	// Moves to a blis-schemas release tag before anything merges (blis-latency-kernel#88, which gives
	// the tags' order).
	github.com/inference-sim/blis-schemas v0.6.1-0.20261027013518-2c38342bcf34
	// TEMPORARY: the rules packs, a separate module in this repository (rules/); compat/kernelmodel
	// hands them to the kernel, and the v0.32 factory applies rules/v0_32; rules/engineconfig decodes
	// a pool's engine block and rules/pack is the pack type. Pinned at this repository's own commit
	// 0f892e3 (rules/pack). A module requiring this one selects at least this pin. Moves to a rules/
	// release tag (blis-latency-kernel#88).
	github.com/inference-sim/blis-vllm/rules v0.0.0-20261027014353-0f892e3ecfc5
	github.com/sirupsen/logrus v1.9.4
	github.com/stretchr/testify v1.12.1
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

require (
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/sys v0.13.0 // indirect
)

module github.com/inference-sim/blis-vllm

go 1.27.2

require (
	// TEMPORARY: inference-sim/blis-latency-kernel's decoupling PR at 549af8f (stacked on #82: the kernel
	// takes its rules and its engine settings from the caller, and imports nothing from this module),
	// which compat/kernelmodel builds against. Moves to a release tag before anything merges
	// (blis-latency-kernel#88).
	github.com/inference-sim/blis-latency-kernel v0.3.1-0.20261028170927-549af8fd2474
	// TEMPORARY: inference-sim/blis-schemas#106 at bd0c5d0 (https://github.com/inference-sim/blis-schemas/pull/106,
	// stacked on #102: the engine-neutral contract, a pool's engine block and an offload's connector
	// options held unopened as deployment.EngineBlock, the engine's rules from the caller, the P/D
	// handoff, Scheduled.PrefixLookup; and #106's kernel/enginesettings, the settings the kernel prices from).
	// Moves to a blis-schemas release tag before anything merges (blis-latency-kernel#88, which gives
	// the tags' order).
	github.com/inference-sim/blis-schemas v0.7.1-0.20261028162459-bd0c5d015b01
	github.com/sirupsen/logrus v1.9.4
	github.com/stretchr/testify v1.12.1
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

require (
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/sys v0.13.0 // indirect
)

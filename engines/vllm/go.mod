module github.com/inference-sim/blis-vllm

go 1.27.2

require (
	// TEMPORARY: the head of inference-sim/blis-latency-kernel#82 (the kernel takes its rules
	// from the caller), which compat/kernelmodel builds against. Moves to a release tag before
	// anything merges (blis-latency-kernel#88).
	github.com/inference-sim/blis-latency-kernel v0.3.1-0.20261025124457-4f8110fb2dba
	// TEMPORARY: inference-sim/blis-schemas#102 at 630937e (the engine contract and
	// rules.EngineRules; #102's later commits change only comments). Moves to a blis-schemas release tag before anything merges
	// (blis-latency-kernel#88, which gives the tags' order).
	github.com/inference-sim/blis-schemas v0.6.1-0.20261025121631-630937e6f21f
	// TEMPORARY: the rules packs, a separate module in this repository (rules/); compat/kernelmodel
	// hands them to the kernel, and the v0.32 factory applies rules/v0_32. Pinned at this
	// repository's own commit 4e838b6 (mla-prefill-backend-known: an error for a name no plugin can
	// register, a warning for one a plugin could). A module requiring
	// this one selects at least this pin. Moves to a rules/ release tag (blis-latency-kernel#88).
	github.com/inference-sim/blis-vllm/rules v0.0.0-20261026132615-4e838b6382cd
	github.com/sirupsen/logrus v1.9.4
	github.com/stretchr/testify v1.12.1
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

require (
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/sys v0.13.0 // indirect
)

module github.com/inference-sim/inference-sim

go 1.24.0

toolchain go1.24.4

require (
	github.com/google/uuid v1.6.0
	github.com/sirupsen/logrus v1.9.3
	github.com/spf13/cobra v1.9.1
	github.com/stretchr/testify v1.7.0
	gopkg.in/yaml.v3 v3.0.1
)

require (
	github.com/inference-sim/blis-latency-kernel v0.0.0 // indirect
	github.com/inference-sim/blis-schemas v0.0.0 // indirect
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/spf13/pflag v1.0.6
	golang.org/x/sys v0.0.0-20220715151400-c0bba94af5f8 // indirect
)

// Local link for the kernel-exclusive experiment (worktree-only; see
// docs/kernel-exclusive/PLAN.md). Points at the working copies on disk rather than
// vendoring or copying, so the experiment measures the kernel as committed elsewhere.
replace github.com/inference-sim/blis-latency-kernel => /Users/sri/Documents/Projects/blis-latency-kernel

replace github.com/inference-sim/blis-schemas => /Users/sri/Documents/Projects/blis-schemas

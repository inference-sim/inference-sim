module github.com/inference-sim/inference-sim

go 1.24.0

toolchain go1.24.4

require (
	github.com/google/uuid v1.6.0
	github.com/inference-sim/blis-latency-kernel v0.2.0
	github.com/inference-sim/blis-schemas v0.3.1-0.20261015030426-1a9978ea32ad
	github.com/sirupsen/logrus v1.9.3
	github.com/spf13/cobra v1.9.1
	github.com/stretchr/testify v1.7.0
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/spf13/pflag v1.0.6
	golang.org/x/sys v0.0.0-20220715151400-c0bba94af5f8 // indirect
)

// TODO(#1939): development pin to a local checkout of blis-schemas dsv41/acceptance-profile
// (d4c35bc). The final PR pins the pushed schemas PR branch commit instead.
replace github.com/inference-sim/blis-schemas => ../blis-schemas-accept-pin

// TODO(#1941): development pin to a local checkout of blis-latency-kernel
// dsv41/h100-capacity-fidelity (4918440; SequenceAdmissionBytes from blis-latency-kernel#46) for SequenceAdmissionBytes.
// The final PR pins the pushed kernel branch commit instead.
replace github.com/inference-sim/blis-latency-kernel => ../blis-latency-kernel-accept-pin

module github.com/inference-sim/blis-vllm

go 1.27.2

require (
	github.com/inference-sim/blis-substrate v0.14.1-0.20261116200733-2ce0e12e1b81 // TEMPORARY: the pushed head of inference-sim/blis-substrate#104, until it is released
	github.com/sirupsen/logrus v1.9.4
	github.com/stretchr/testify v1.12.1
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

require (
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/sys v0.13.0 // indirect
)

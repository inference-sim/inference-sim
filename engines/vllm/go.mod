module github.com/inference-sim/blis-vllm

go 1.27.2

require (
	github.com/inference-sim/blis-substrate v0.14.1-0.20261116211210-c4fef7d9092f // TEMPORARY: inference-sim/blis-substrate branch m3/vllm-v033-integration, with blis-substrate#98 merged into it, until those are released
	github.com/sirupsen/logrus v1.9.4
	github.com/stretchr/testify v1.12.1
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

require (
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/sys v0.13.0 // indirect
)

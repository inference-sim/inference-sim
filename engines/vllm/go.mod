module github.com/inference-sim/blis-vllm

go 1.27.2

require (
	github.com/inference-sim/blis-substrate v0.14.2-0.20261116211815-f14a5415c547 // TEMPORARY: inference-sim/blis-substrate branch m3/integration (f14a5415c5479a2d8ba995b787ed9a4312f1d5b3), until it is released
	github.com/sirupsen/logrus v1.9.4
	github.com/stretchr/testify v1.12.1
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

require (
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/sys v0.13.0 // indirect
)

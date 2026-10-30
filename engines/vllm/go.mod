module github.com/inference-sim/blis-vllm

go 1.27.2

require (
	github.com/inference-sim/blis-latency-kernel v0.4.2-0.20261030162513-1fbc78e0bd53 // TEMPORARY: the pushed head of inference-sim/blis-latency-kernel#99, until it is released
	github.com/inference-sim/blis-schemas v0.9.2-0.20261030162312-3465d9d16dfc // TEMPORARY: the pushed head of inference-sim/blis-schemas#112, until it is released
	github.com/sirupsen/logrus v1.9.4
	github.com/stretchr/testify v1.12.1
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

require (
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/sys v0.13.0 // indirect
)

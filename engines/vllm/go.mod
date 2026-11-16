module github.com/inference-sim/blis-vllm

go 1.27.2

require (
	github.com/inference-sim/blis-substrate v0.14.2-0.20261116230853-feee8e581ef2 // TEMPORARY: inference-sim/blis-substrate#208 (branch m3/skip-drafts, feee8e581ef2bb31858fcadfbb203bc9a30cc539, on m3/integration), until it is released
	github.com/sirupsen/logrus v1.9.4
	github.com/stretchr/testify v1.12.1
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

require (
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/sys v0.13.0 // indirect
)

module github.com/inference-sim/inference-sim

go 1.25.0

require (
	github.com/llm-inferno/model-tuner v0.5.1
	github.com/llm-inferno/queue-analysis v0.5.0
	github.com/sirupsen/logrus v1.9.3
	github.com/spf13/cobra v1.9.1
	github.com/stretchr/testify v1.7.0
	gopkg.in/yaml.v3 v3.0.1
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/spf13/pflag v1.0.6 // indirect
	golang.org/x/sys v0.37.0 // indirect
)

replace (
	github.com/llm-inferno/model-tuner => ../../llm-inferno/model-tuner
	github.com/llm-inferno/queue-analysis => ../../llm-inferno/queue-analysis
)

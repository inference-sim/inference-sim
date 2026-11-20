package contracts_test

// The examples in this file are the Go snippets the documentation shows. They run under
// `go test`, and their Output comments are checked, so a snippet on the site cannot show a
// call that no longer compiles or a message the code no longer prints. The `--8<--` markers
// delimit the part each page includes.

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/inference-sim/inference-sim/substrate/contracts"
	"github.com/inference-sim/inference-sim/substrate/contracts/check"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"gopkg.in/yaml.v3"
)

// exampleRules are an engine release's checker as an engine module supplies it, for the
// examples: it opens a pool's engine block, checks one bound and refuses one value. A real
// caller passes its engine module's checker (the vLLM engine module's for vLLM).
type exampleRules struct {
	version string
}

type exampleEngine struct {
	BlockSize            int     `yaml:"block_size"`
	MaxNumSeqs           int     `yaml:"max_num_seqs"`
	MaxNumBatchedTokens  int     `yaml:"max_num_batched_tokens"`
	GPUMemoryUtilization float64 `yaml:"gpu_memory_utilization"`
	CacheDType           string  `yaml:"cache_dtype"`
}

func (r exampleRules) EngineVersion() string { return r.version }
func (exampleRules) DraftTokens(deployment.EngineBlock) int {
	return 0
}
func (exampleRules) CheckConnectorOptions(string, deployment.EngineBlock) *validate.Problems {
	return nil
}
func (r exampleRules) CheckEngineBlock(at string, b deployment.EngineBlock) *validate.Problems {
	p := &validate.Problems{}
	var e exampleEngine
	if err := b.Decode(&e); err != nil {
		p.Field(at+".engine", "%v", err)
		return p
	}
	if e.GPUMemoryUtilization < 0 || e.GPUMemoryUtilization > 1 {
		p.Field(at+".engine.gpu_memory_utilization", "must lie in [0, 1] (0 means unset), got %v", e.GPUMemoryUtilization)
	}
	if e.CacheDType != "" && e.CacheDType != "auto" && e.CacheDType != "fp8" {
		p.Field(at+".engine.cache_dtype", "%q is not accepted by engine %s", e.CacheDType, r.version)
	}
	return p
}
func exampleRulesFor(version string) contracts.EngineChecker { return exampleRules{version: version} }

// setEngineKey sets one key of a pool's engine block, as an author edits the file.
func setEngineKey(pool *deployment.Pool, key, value string) {
	n := pool.Engine.Node()
	if n == nil {
		n = &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	}
	for i := 0; i+1 < len(n.Content); i += 2 {
		if n.Content[i].Value == key {
			n.Content[i+1] = &yaml.Node{Kind: yaml.ScalarNode, Value: value}
			pool.Engine = deployment.NewEngineBlock(n)
			return
		}
	}
	n.Content = append(n.Content, &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: key},
		&yaml.Node{Kind: yaml.ScalarNode, Value: value})
	pool.Engine = deployment.NewEngineBlock(n)
}

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

// ExampleValidate validates a scenario and a deployment together with the catalog
// documents they name.
func ExampleValidate() {
	// --8<-- [start:validate]
	s := must(contracts.LoadScenario("testdata/examples/scenario.yaml"))
	d := must(contracts.LoadDeployment("testdata/examples/deployment.yaml"))
	g := must(contracts.LoadModelGraph("../catalog/models/llama-3.1-8b-instruct/graph.yaml"))
	c := must(contracts.LoadChip("../catalog/hardware/h100.yaml"))

	rep := contracts.Validate(contracts.Bundle{
		Scenario: s, Deployment: d, Model: g, Chip: c,
		Engine: exampleRulesFor, // the release's checker, from its engine module
	})
	for _, p := range rep.Problems() {
		fmt.Println(p)
	}
	fmt.Println("ok:", rep.OK())
	// --8<-- [end:validate]

	// Output:
	// ok: true
}

// ExampleValidate_problems shows what a failing bundle reports: every problem at once,
// each at the path of the field that is wrong.
func ExampleValidate_problems() {
	s := must(contracts.LoadScenario("testdata/examples/scenario.yaml"))
	d := must(contracts.LoadDeployment("testdata/examples/deployment.yaml"))
	// --8<-- [start:problems]
	pool := &d.Pools[0]
	pool.Parallel.TP = 8  // 8 x 4 data-parallel ranks needs 32 GPUs
	pool.Parallel.DCP = 3 // does not divide tp
	setEngineKey(pool, "gpu_memory_utilization", "1.5")

	rep := contracts.Validate(contracts.Bundle{Scenario: s, Deployment: d, Engine: exampleRulesFor})
	for _, p := range rep.Problems() {
		fmt.Println(p)
	}
	// --8<-- [end:problems]

	// Output:
	// error: deployment.pools[0].engine.gpu_memory_utilization: must lie in [0, 1] (0 means unset), got 1.5
	// error: deployment.pools[0].parallel: needs 32 GPUs (pp 1 x tp 8 x pcp 1 x dp 4, one per rank) but the pool's 1 node(s) of 8 GPUs provide 8
	// error: deployment.pools[0].parallel.dcp: decode-context parallelism reuses the tensor-parallel ranks when prefill-context parallelism is off, so tp 8 must be divisible by dcp 3
}

// ExampleValidate_rules shows an engine problem: the documents are well formed, but the engine
// release the build runs would refuse them.
func ExampleValidate_rules() {
	s := must(contracts.LoadScenario("testdata/examples/scenario.yaml"))
	d := must(contracts.LoadDeployment("testdata/examples/deployment.yaml"))
	s.Engine.Version = "0.29.0" // the release the build runs; a scenario states none
	// --8<-- [start:rules]
	setEngineKey(&d.Pools[0], "cache_dtype", "fp8_e4m3fn") // not a name the release accepts

	rep := contracts.Validate(contracts.Bundle{Scenario: s, Deployment: d, Engine: exampleRulesFor})
	fmt.Println("ok:", rep.OK())
	for _, p := range rep.Problems() {
		fmt.Println(p)
	}
	// --8<-- [end:rules]

	// Output:
	// ok: false
	// error: deployment.pools[0].engine.cache_dtype: "fp8_e4m3fn" is not accepted by engine 0.29.0
}

// ExampleValidateAgainstCatalog resolves a scenario's by-name references against a catalog
// checkout and a registry checkout.
func ExampleValidateAgainstCatalog() {
	// A stand-in registry: resolution checks only that each named set's file exists.
	registry := must(os.MkdirTemp("", "registry"))
	defer os.RemoveAll(registry)
	must(0, os.MkdirAll(filepath.Join(registry, "coefficients"), 0o755))
	for _, name := range []string{"cost-model-primitives", "cost-model-collectives",
		"cost-model-host-overheads", "cost-model-attention", "cost-model-recurrent"} {
		must(0, os.WriteFile(filepath.Join(registry, "coefficients", name+".yaml"), nil, 0o644))
	}

	// --8<-- [start:resolve]
	s := must(contracts.LoadScenario("testdata/examples/scenario.yaml"))
	s.Cluster.Hardware = "nvidia-h100" // the catalog calls it h100

	rep := contracts.ValidateAgainstCatalog(
		contracts.Bundle{Scenario: s}, "../catalog", registry)
	for _, p := range rep.Problems() {
		fmt.Println(p)
	}
	// --8<-- [end:resolve]

	// Output:
	// error: references.hardware: "nvidia-h100": looked for hardware/nvidia-h100.yaml; the catalog has [a100-80 a100-sxm b200 b300 gb200-nvl72 gb300 h100 h200 l40s rtx6000pro-server]
}

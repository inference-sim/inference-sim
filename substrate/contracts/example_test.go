package contracts_test

// The examples in this file are the Go snippets the documentation shows. They run under
// `go test`, and their Output comments are checked, so a snippet on the site cannot show a
// call that no longer compiles or a message the code no longer prints. The `--8<--` markers
// delimit the part each page includes.

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"gopkg.in/yaml.v3"
)

// exampleRules are an engine release's rules as an engine model supplies them, for the
// examples: they open a pool's engine block, check one bound and refuse one value by a named
// rule. A real caller passes its engine model's rules (blis-vllm/rules for vLLM).
type exampleRules struct {
	rules.EngineRules
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
func (exampleRules) RuleNames() []string     { return []string{"enum-values-known"} }
func (exampleRules) DraftTokens(deployment.EngineBlock) int {
	return 0
}
func (exampleRules) CheckConnectorOptions(string, deployment.EngineBlock) *rules.Problems {
	return nil
}
func (exampleRules) CheckEngineBlock(at string, b deployment.EngineBlock) *rules.Problems {
	p := &rules.Problems{}
	var e exampleEngine
	if err := b.Decode(&e); err != nil {
		p.Field(at+".engine", "%v", err)
		return p
	}
	if e.GPUMemoryUtilization < 0 || e.GPUMemoryUtilization > 1 {
		p.Field(at+".engine.gpu_memory_utilization", "must lie in [0, 1] (0 means unset), got %v", e.GPUMemoryUtilization)
	}
	return p
}
func (r exampleRules) Check(in rules.Input) *rules.Problems {
	p := &rules.Problems{}
	for i, pool := range in.Deployment.Pools {
		var e exampleEngine
		if pool.Engine.Decode(&e) == nil && e.CacheDType != "" && e.CacheDType != "auto" && e.CacheDType != "fp8" {
			p.RuleErrorf("enum-values-known", "pools[%d].engine.cache_dtype: %q is not accepted by engine %s", i, e.CacheDType, r.version)
		}
	}
	return p
}

func exampleRulesFor(version string) rules.EngineRules { return exampleRules{version: version} }

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
	s := must(contracts.LoadScenario("../docs/contracts/examples/scenario.yaml"))
	d := must(contracts.LoadDeployment("../docs/contracts/examples/deployment.yaml"))
	g := must(contracts.LoadModelGraph("../catalog/models/llama-3.1-8b-instruct/graph.yaml"))
	c := must(contracts.LoadChip("../catalog/hardware/h100.yaml"))

	rep := contracts.Validate(contracts.Bundle{
		Scenario: s, Deployment: d, Model: g, Chip: c,
		Rules: exampleRulesFor, // the release's rules, from its engine model
	})
	for _, p := range rep.Problems() {
		fmt.Println(p)
	}
	fmt.Println("ok:", rep.OK(), "rules applied:", rep.RulesApplied)
	// --8<-- [end:validate]

	// Output:
	// ok: true rules applied: 0.29.0
}

// ExampleValidate_problems shows what a failing bundle reports: every problem at once,
// each at the path of the field that is wrong.
func ExampleValidate_problems() {
	s := must(contracts.LoadScenario("../docs/contracts/examples/scenario.yaml"))
	d := must(contracts.LoadDeployment("../docs/contracts/examples/deployment.yaml"))
	// --8<-- [start:problems]
	pool := &d.Pools[0]
	pool.Parallel.TP = 8  // 8 x 4 data-parallel ranks needs 32 GPUs
	pool.Parallel.DCP = 3 // does not divide tp
	setEngineKey(pool, "gpu_memory_utilization", "1.5")

	rep := contracts.Validate(contracts.Bundle{Scenario: s, Deployment: d, Rules: exampleRulesFor})
	for _, p := range rep.Problems() {
		fmt.Println(p)
	}
	// --8<-- [end:problems]

	// Output:
	// error: deployment.pools[0].engine.gpu_memory_utilization: must lie in [0, 1] (0 means unset), got 1.5
	// error: deployment.pools[0].parallel: needs 32 GPUs (pp 1 x tp 8 x pcp 1 x dp 4, one per rank) but the pool's 1 node(s) of 8 GPUs provide 8
	// error: deployment.pools[0].parallel.dcp: decode-context parallelism reuses the tensor-parallel ranks when prefill-context parallelism is off, so tp 8 must be divisible by dcp 3
}

// ExampleValidate_rules shows a rule problem: the documents are well formed, but the engine
// version they name would refuse them.
func ExampleValidate_rules() {
	s := must(contracts.LoadScenario("../docs/contracts/examples/scenario.yaml"))
	d := must(contracts.LoadDeployment("../docs/contracts/examples/deployment.yaml"))
	// --8<-- [start:rules]
	setEngineKey(&d.Pools[0], "cache_dtype", "fp8_e4m3fn") // not a name the release accepts

	rep := contracts.Validate(contracts.Bundle{Scenario: s, Deployment: d, Rules: exampleRulesFor})
	fmt.Println("field layer ok:", rep.Field.OK())
	for _, p := range rep.Rule.All() {
		fmt.Println(p)
	}
	// --8<-- [end:rules]

	// Output:
	// field layer ok: true
	// error [enum-values-known]: : pools[0].engine.cache_dtype: "fp8_e4m3fn" is not accepted by engine 0.29.0
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
	s := must(contracts.LoadScenario("../docs/contracts/examples/scenario.yaml"))
	s.Cluster.Hardware = "nvidia-h100" // the catalog calls it h100

	rep := contracts.ValidateAgainstCatalog(
		contracts.Bundle{Scenario: s}, "../catalog", registry)
	for _, p := range rep.Problems() {
		fmt.Println(p)
	}
	// --8<-- [end:resolve]

	// Output:
	// error: references.hardware: "nvidia-h100": looked for hardware/nvidia-h100.yaml; the catalog has [a100-80 a100-sxm b200 b300 gb200-nvl72 gb300 h100 h200 l40s]
}

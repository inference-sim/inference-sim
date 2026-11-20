package contracts

import (
	"slices"

	"github.com/inference-sim/inference-sim/substrate/contracts/check"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"gopkg.in/yaml.v3"
)

// testEngine is the engine block the test engine's rules open: a stand-in for an engine model's
// own configuration, which this schema does not know.
type testEngine struct {
	BlockSize            int     `yaml:"block_size"`
	GPUMemoryUtilization float64 `yaml:"gpu_memory_utilization"`
	Backend              string  `yaml:"backend"`
	DraftTokens          int     `yaml:"draft_tokens"`
}

// testRules is an engine release's checker as a caller supplies it (Bundle.Engine): it opens a
// pool's engine block (CheckEngineBlock), refuses a backend outside the release's set there,
// and reports how many tokens the block drafts (DraftTokens).
type testRules struct {
	version  string
	backends []string
}

func (r testRules) EngineVersion() string { return r.version }

func (r testRules) CheckEngineBlock(at string, b deployment.EngineBlock) *validate.Problems {
	p := &validate.Problems{}
	var e testEngine
	if err := b.Decode(&e); err != nil {
		p.Field(at+".engine", "%v", err)
		return p
	}
	if e.BlockSize < 0 {
		p.Field(at+".engine.block_size", "must not be negative")
	}
	if p.FiniteField(at+".engine.gpu_memory_utilization", e.GPUMemoryUtilization) &&
		(e.GPUMemoryUtilization < 0 || e.GPUMemoryUtilization > 1) {
		p.Field(at+".engine.gpu_memory_utilization", "must lie in [0, 1] (0 means unset), got %v", e.GPUMemoryUtilization)
	}
	if e.Backend != "" && !slices.Contains(r.backends, e.Backend) {
		p.Field(at+".engine.backend", "%q is not accepted by engine %s (accepted: %v)", e.Backend, r.version, r.backends)
	}
	return p
}

func (r testRules) CheckConnectorOptions(at string, b deployment.EngineBlock) *validate.Problems {
	p := &validate.Problems{}
	var o struct {
		BlocksPerChunk int `yaml:"blocks_per_chunk"`
	}
	if err := b.Decode(&o); err != nil {
		p.Field(at+".connector_options", "%v", err)
		return p
	}
	if o.BlocksPerChunk < 0 {
		p.Field(at+".connector_options.blocks_per_chunk", "must not be negative")
	}
	return p
}

func (r testRules) DraftTokens(b deployment.EngineBlock) int {
	var e testEngine
	if b.Decode(&e) != nil {
		return 0
	}
	return e.DraftTokens
}

// testRulesFor is a caller's lookup over the test engine's releases.
func testRulesFor(version string) EngineChecker {
	switch version {
	case "1.0.0", "1.1.0":
		return testRules{version: version, backends: []string{"fast", "slow"}}
	}
	return nil
}

// testEngineVersions is the releases testRulesFor answers.
func testEngineVersions() []string { return []string{"1.0.0", "1.1.0"} }

// engineBlock is a pool's engine block as a document states it.
func engineBlock(doc string) deployment.EngineBlock {
	var n yaml.Node
	if err := yaml.Unmarshal([]byte(doc), &n); err != nil {
		panic(err)
	}
	return deployment.NewEngineBlock(n.Content[0])
}

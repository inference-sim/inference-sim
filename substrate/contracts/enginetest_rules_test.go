package contracts

import (
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
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

// testRules is an engine release's rules as a caller supplies them (Bundle.Rules): they open
// a pool's engine block (CheckEngineBlock), report how many tokens it drafts (DraftTokens), and
// refuse a backend outside the release's set as a named rule. Every other method is the
// embedded nil interface's, which a test that reached it would fail on.
type testRules struct {
	rules.EngineRules
	version  string
	backends []string
}

func (r testRules) EngineVersion() string { return r.version }

func (r testRules) RuleNames() []string { return []string{"backend-known"} }

func (r testRules) CheckEngineBlock(at string, b deployment.EngineBlock) *rules.Problems {
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
	return p
}

func (r testRules) CheckConnectorOptions(at string, b deployment.EngineBlock) *rules.Problems {
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

func (r testRules) Check(in rules.Input) *rules.Problems {
	p := &validate.Problems{}
	if in.Deployment == nil {
		return p
	}
	known := map[string]bool{}
	for _, b := range r.backends {
		known[b] = true
	}
	for i, pool := range in.Deployment.Pools {
		var e testEngine
		if pool.Engine.Decode(&e) != nil || e.Backend == "" || known[e.Backend] {
			continue
		}
		names := append([]string(nil), r.backends...)
		sort.Strings(names)
		p.RuleErrorf("backend-known", "pools[%d].engine.backend: %q is not accepted by engine %s (accepted: %v)",
			i, e.Backend, r.version, names)
	}
	return p
}

// testRulesFor is a caller's lookup over the test engine's releases.
func testRulesFor(version string) rules.EngineRules {
	switch version {
	case "1.0.0", "1.1.0":
		return testRules{version: version, backends: []string{"fast", "slow"}}
	}
	return nil
}

// testRulesVersions is the releases testRulesFor answers.
func testRulesVersions() []string { return []string{"1.0.0", "1.1.0"} }

// engineBlock is a pool's engine block as a document states it.
func engineBlock(doc string) deployment.EngineBlock {
	var n yaml.Node
	if err := yaml.Unmarshal([]byte(doc), &n); err != nil {
		panic(err)
	}
	return deployment.NewEngineBlock(n.Content[0])
}

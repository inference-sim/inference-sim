// Original to blis-vllm.

// Package vllmengine is how a simulator gets blis-vllm's vLLM v0.32.0 engine (core/vllm) without
// naming any of its packages: Register makes Kind "vllm" available in blis-schemas' engine
// registry, built from nothing but an engine.Spec, and the simulator drives it through
// engine.Engine.
//
// The factory registered here is the production one. The simulator passes only the paths the
// engine reads (Roots) and which pool it is (Pool); the factory opens the scenario itself, so the
// simulator opens no vLLM configuration (inference-sim#2021 B2, B3). For that pool it:
//
//   - prices every step with blis-latency-kernel, the same kernel the compatibility engine's
//     compat/kernelmodel opens, so a 0.32.0 run and a 0.31.0 run of one deployment are priced by
//     one cost model. The kernel's engine behaviour is vLLM 0.31.0's; it records that as an
//     assumption for a 0.32.0 scenario, and the engine discloses it (Assumptions);
//   - sizes the KV cache from the kernel's memory model (compat/kernelmodel KVBudget), so the
//     engine holds exactly the blocks the compatibility engine holds for the same deployment;
//   - fills in the facts the engine block does not carry: the KV block count, the pool's
//     pipeline_parallel_size, and the model runner, which vLLM picks from the image (see
//     modelRunner).
package vllmengine

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	vllm "github.com/inference-sim/blis-vllm/core/vllm"
)

// The Roots keys the factory reads.
const (
	// RootCatalog is a blis-catalog checkout or a pinned copy of one.
	RootCatalog = "catalog"
	// RootRegistry is a blis-registry checkout or a pinned copy of one.
	RootRegistry = "registry"
	// RootScenario is the scenario file's path: its Scenario document and its Deployment.
	RootScenario = "scenario"
)

// Register makes Kind "vllm" available in blis-schemas' engine registry, built as this package's
// doc comment says. A 0.29.0 or 0.31.0 spec is refused: a simulator runs those releases on the
// compatibility engine itself. Later calls are no-ops.
func Register() { vllm.Register(Factory()) }

// Factory is the production factory Register registers.
func Factory() vllm.Factory {
	return vllm.Factory{Pricer: pricerFor, Prepare: prepare}
}

// pool is one opened pool: the kernel that prices it, its KV block count and its engine block.
type pool struct {
	model  *kernelmodel.Model
	blocks int64
	engine yaml.Node
	pp     int
}

// open opens the spec's pool through compat/kernelmodel, as the compatibility engine opens it.
func open(s engine.Spec) (pool, error) {
	path := s.Roots[RootScenario]
	if path == "" {
		return pool{}, fmt.Errorf("vllmengine: Roots[%q] must name the scenario file", RootScenario)
	}
	for _, k := range []string{RootCatalog, RootRegistry} {
		if s.Roots[k] == "" {
			return pool{}, fmt.Errorf("vllmengine: Roots[%q] must name a directory", k)
		}
	}
	m, err := kernelmodel.OpenPool(filepath.Base(path), kernelmodel.Repos{
		Scenarios: filepath.Dir(path), Catalog: s.Roots[RootCatalog], Registry: s.Roots[RootRegistry],
	}, s.Pool)
	if err != nil {
		return pool{}, fmt.Errorf("vllmengine: %w", err)
	}
	if missing := m.V032Pieces(); len(missing) > 0 {
		// Refused rather than run without the piece: a run that dropped a sliding-window group,
		// a drafter's lookahead or a rank's wave would still produce numbers.
		return pool{}, fmt.Errorf("vllmengine: %s needs %s, which blis-vllm's v0.32.0 engine does not build yet",
			filepath.Base(path), strings.Join(missing, ", "))
	}
	st, err := m.Settings()
	if err != nil {
		return pool{}, fmt.Errorf("vllmengine: %w", err)
	}
	node, pp, err := poolBlock(path, s.Pool)
	if err != nil {
		return pool{}, err
	}
	return pool{model: m, blocks: st.KVBlocks, engine: node, pp: pp}, nil
}

// poolBlock reads the pool's engine block, as the document gave it, and its parallel.pp.
func poolBlock(path string, index int) (yaml.Node, int, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return yaml.Node{}, 0, fmt.Errorf("vllmengine: %w", err)
	}
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	for {
		var doc struct {
			Kind  string `yaml:"kind"`
			Pools []struct {
				Parallel deployment.Parallelism `yaml:"parallel"`
				Engine   yaml.Node              `yaml:"engine"`
			} `yaml:"pools"`
		}
		if err := dec.Decode(&doc); err == io.EOF {
			return yaml.Node{}, 0, fmt.Errorf("vllmengine: %s has no Deployment document", path)
		} else if err != nil {
			return yaml.Node{}, 0, fmt.Errorf("vllmengine: %s: %w", path, err)
		}
		if doc.Kind != "Deployment" {
			continue
		}
		if index < 0 || index >= len(doc.Pools) {
			return yaml.Node{}, 0, fmt.Errorf("vllmengine: %s has %d pools, not a pool %d", path, len(doc.Pools), index)
		}
		return doc.Pools[index].Engine, max(doc.Pools[index].Parallel.PP, 1), nil
	}
}

// prepare completes a spec's engine block with what the kernel and the deployment know: the KV
// block count, pipeline_parallel_size, the model runner and an MTP drafter's layer count. A block the spec carries is used as
// given; an empty one is the pool's own, read from the scenario, so a simulator need not open it.
func prepare(s engine.Spec) (engine.Spec, error) {
	p, err := open(s)
	if err != nil {
		return s, err
	}
	s.Config = p.complete(s.Config)
	return s, nil
}

// complete is the engine block given, or the pool's own when none is, with what the pool knows
// added where the block does not state it.
func (p pool) complete(given yaml.Node) yaml.Node {
	block := given
	if block.Kind == 0 {
		block = p.engine
	}
	if block.Kind == yaml.DocumentNode && len(block.Content) == 1 {
		block = *block.Content[0]
	}
	if block.Kind != yaml.MappingNode {
		block = yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	}
	block = *cloneNode(&block)
	setIfAbsent(&block, "num_gpu_blocks", fmt.Sprint(p.blocks))
	setIfAbsent(&block, "pipeline_parallel_size", fmt.Sprint(p.pp))
	runner, _ := p.model.ModelRunner()
	setIfAbsent(&block, "use_v2_model_runner", fmt.Sprint(runner == "v2"))
	if n := p.model.MTPDraftLayers(); n > 0 {
		// An MTP drafter's prefill lookahead turns on its checkpoint's layer count
		// (vllm/config/vllm.py:644-664@5df63a72), which the catalog's model graph holds.
		setIfAbsent(&block, "mtp_nextn_predict_layers", fmt.Sprint(n))
	}
	return block
}

// pricerFor is the kernel's pricing for the spec's pool.
func pricerFor(s engine.Spec) (vllm.Pricer, error) {
	p, err := open(s)
	if err != nil {
		return nil, err
	}
	return kernelPricer{m: p.model}, nil
}

// kernelPricer prices an engine.Batch with the latency kernel, as compat/kernelmodel prices the
// compatibility engine's batches (Model.StepTime): the kernel's StepTime of the batch, its
// NoOverlap figure, in microseconds, at least one.
type kernelPricer struct{ m *kernelmodel.Model }

// StepTime implements core/vllm.Pricer.
func (k kernelPricer) StepTime(b engine.Batch) engine.Duration {
	return engine.Duration(k.m.StepTimeOfShapes(shapes(b)))
}

// Assumptions are the kernel's recorded assumptions for this pool, each as "name = value:
// reason". For a 0.32.0 scenario they include that the kernel prices vLLM 0.31.0's engine
// behaviour.
func (k kernelPricer) Assumptions() []string {
	var out []string
	for _, a := range k.m.Provenance().Assumptions {
		out = append(out, a.Name+" = "+a.Scope)
	}
	return out
}

// shapes is the batch in the kernel's request shapes. The mapping is compat/kernelmodel's
// (shapeOf): Scheduled is the tokens the forward pass runs, which under speculation is the verify
// width, and the contract's Tokens already counts the drafts; Computed and PromptLen pass
// through; CachedTokens is zero because a prefix hit is already in Computed, and the kernel
// subtracts CachedTokens from Scheduled only for a caller that has not.
func shapes(b engine.Batch) []kernelmodel.ReqShape {
	out := make([]kernelmodel.ReqShape, len(b.Reqs))
	for i, r := range b.Reqs {
		out[i] = kernelmodel.ReqShape{Scheduled: r.Tokens, Computed: r.Computed, PromptLen: r.PromptLen}
	}
	return out
}

// setIfAbsent adds key: value to a mapping node that does not state key.
func setIfAbsent(m *yaml.Node, key, value string) {
	for i := 0; i+1 < len(m.Content); i += 2 {
		if m.Content[i].Value == key {
			return
		}
	}
	m.Content = append(m.Content,
		&yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: key},
		&yaml.Node{Kind: yaml.ScalarNode, Value: value})
}

// cloneNode copies a node tree, so completing a block never edits the caller's.
func cloneNode(n *yaml.Node) *yaml.Node {
	c := *n
	c.Content = make([]*yaml.Node, len(n.Content))
	for i, ch := range n.Content {
		c.Content[i] = cloneNode(ch)
	}
	return &c
}

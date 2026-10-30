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
//   - sizes the KV cache from the kernel's memory model (compat/kernelmodel KVBudget): the
//     compatibility engine's block count for a model of one full-attention group, and for a
//     hybrid model the memory left for KV over its widest KV cache group's bytes per block, as
//     vLLM sizes its shared pool (poolBlocks);
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
// kvBytes is the memory a hybrid model's pool was sized from (poolBlocks), 0 for one sized as
// the compatibility engine sizes it.
type pool struct {
	model   *kernelmodel.Model
	blocks  int64
	kvBytes int64
	engine  yaml.Node
	pp, dp  int
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
	// The engine runs vLLM's start-up check over the KV cache groups (core/vllm checkKVLayersFit).
	st, err := m.SettingsCheckedByGroups()
	if err != nil {
		return pool{}, fmt.Errorf("vllmengine: %w", err)
	}
	node, par, err := poolBlock(path, s.Pool)
	if err != nil {
		return pool{}, err
	}
	blocks, kvBytes, err := poolBlocks(m, st)
	if err != nil {
		return pool{}, fmt.Errorf("vllmengine: %s: %w", filepath.Base(path), err)
	}
	return pool{model: m, blocks: blocks, kvBytes: kvBytes, engine: node, pp: max(par.PP, 1), dp: max(par.DP, 1)}, nil
}

// poolBlocks is the KV pool's block count. vLLM sizes the pool over its KV cache groups: the
// memory left for KV over the widest group's bytes per block (get_kv_cache_config_from_groups,
// vllm/v1/core/kv_cache_utils.py:1622-1628@5df63a72), because every group draws its blocks from
// one shared pool. A model of one full-attention group is the compatibility engine's count, the
// kernel's allocatable bytes over its bytes for one block of every layer; a hybrid model's
// groups each hold a share of the layers, so its pool holds more blocks than that
// (gpt-oss-120b's two groups of 18 layers, twice as many).
//
// For a hybrid model it also returns the memory the pool was sized from, which the engine block
// states as kv_available_bytes so vLLM's start-up check reports its numbers in GiB.
//
// Under pipeline parallelism each stage's rank sizes a pool over its own layers (the drafter's
// on the last), and the engine keeps the smallest (get_kv_cache_configs step 5,
// vllm/v1/core/kv_cache_utils.py:2530-2660@5df63a72). The kernel's per-rank budget and bytes per
// block are the binding stage's, so each stage's layers are sized against them and the
// smallest count is the pool's.
func poolBlocks(m *kernelmodel.Model, st kernelmodel.Settings) (int64, int64, error) {
	runs, err := m.KVLayers()
	if err != nil || len(runs) == 0 {
		return st.KVBlocks, 0, err
	}
	b, err := m.KVBudget()
	if err != nil {
		return 0, 0, err
	}
	e, err := m.Engine()
	if err != nil {
		return 0, 0, err
	}
	disable := e.DisableHybridKVCacheManager != nil && *e.DisableHybridKVCacheManager
	pp := m.PipelineStages()
	if pp == 1 {
		blocks, err := vllm.KVPoolBlocks(vllmRuns(runs), disable, e.BlockSize, b.PerBlockBytes, b.AllocatableBytes)
		return blocks, b.AllocatableBytes, err
	}
	// The kernel's bytes a block are the binding stage's: the stage holding the most KV
	// elements a token (liftPipelineMemory). Every layer is at one cache dtype, so its bytes
	// over that stage's elements is the bytes an element, which sizes every stage's page.
	stages := make([][]kernelmodel.KVLayerRun, pp)
	var widest int64
	for s := range pp {
		if stages[s], err = m.KVLayersOfStage(s); err != nil {
			return 0, 0, err
		}
		widest = max(widest, kvElems(stages[s]))
	}
	if widest == 0 {
		return st.KVBlocks, 0, nil
	}
	pool := int64(-1)
	for _, stage := range stages {
		if len(stage) == 0 {
			continue
		}
		perBlock := b.PerBlockBytes * kvElems(stage) / widest
		blocks, err := vllm.KVPoolBlocks(vllmRuns(stage), disable, e.BlockSize, perBlock, b.AllocatableBytes)
		if err != nil {
			return 0, 0, err
		}
		if pool < 0 || blocks < pool {
			pool = blocks
		}
	}
	return pool, b.AllocatableBytes, nil
}

// kvElems is the KV elements one token of these layers holds: what scales a block's bytes
// between layer sets at one cache dtype.
func kvElems(runs []kernelmodel.KVLayerRun) int64 {
	var n int64
	for _, r := range runs {
		n += int64(r.Count) * int64(r.NumKVHeads) * int64(r.HeadSize)
	}
	return n
}

// poolBlock reads the pool's engine block, as the document gave it, and its parallel block.
func poolBlock(path string, index int) (yaml.Node, deployment.Parallelism, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return yaml.Node{}, deployment.Parallelism{}, fmt.Errorf("vllmengine: %w", err)
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
			return yaml.Node{}, deployment.Parallelism{}, fmt.Errorf("vllmengine: %s has no Deployment document", path)
		} else if err != nil {
			return yaml.Node{}, deployment.Parallelism{}, fmt.Errorf("vllmengine: %s: %w", path, err)
		}
		if doc.Kind != "Deployment" {
			continue
		}
		if index < 0 || index >= len(doc.Pools) {
			return yaml.Node{}, deployment.Parallelism{}, fmt.Errorf("vllmengine: %s has %d pools, not a pool %d", path, len(doc.Pools), index)
		}
		return doc.Pools[index].Engine, doc.Pools[index].Parallel, nil
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
	if p.kvBytes > 0 {
		// The memory the pool's blocks were sized from, for vLLM's start-up check's message.
		setIfAbsent(&block, "kv_available_bytes", fmt.Sprint(p.kvBytes))
	}
	setIfAbsent(&block, "pipeline_parallel_size", fmt.Sprint(p.pp))
	if p.dp > 1 {
		// The pool's parallel.dp: the factory builds the data-parallel group behind one endpoint.
		setIfAbsent(&block, "data_parallel_size", fmt.Sprint(p.dp))
	}
	runner, _ := p.model.ModelRunner()
	setIfAbsent(&block, "use_v2_model_runner", fmt.Sprint(runner == "v2"))
	if n := p.model.MTPDraftLayers(); n > 0 {
		// An MTP drafter's prefill lookahead turns on its checkpoint's layer count
		// (vllm/config/vllm.py:644-664@5df63a72), which the catalog's model graph holds.
		setIfAbsent(&block, "mtp_nextn_predict_layers", fmt.Sprint(n))
	}
	if runs, err := p.model.KVLayers(); err == nil && len(runs) > 0 {
		// The model's attention layers by the KV cache spec vLLM builds for each, which the
		// factory groups as get_kv_cache_groups does (vllm/v1/core/kv_cache_utils.py:2159@5df63a72).
		setNodeIfAbsent(&block, "kv_layers", kvLayersNode(runs))
	}
	return block
}

// kvLayersNode is the kernel's KV layer runs as an engine block's kv_layers.
func kvLayersNode(runs []kernelmodel.KVLayerRun) *yaml.Node {
	var n yaml.Node
	if err := n.Encode(vllmRuns(runs)); err != nil {
		panic(err) // a slice of plain structs always encodes
	}
	return &n
}

// vllmRuns is the kernel's KV layer runs in core/vllm's terms.
func vllmRuns(runs []kernelmodel.KVLayerRun) []vllm.KVLayerRun {
	out := make([]vllm.KVLayerRun, len(runs))
	for i, r := range runs {
		out[i] = vllm.KVLayerRun{Count: r.Count, Spec: r.Spec, SlidingWindow: r.SlidingWindow,
			NumKVHeads: r.NumKVHeads, HeadSize: r.HeadSize, BlockSize: r.BlockSize, TokensPerState: r.TokensPerState,
			StateBytes: r.StateBytes, ElemBytes: r.ElemBytes, Alignment: r.Alignment,
			StrideAlignment: r.StrideAlignment, BoundedReplay: r.BoundedReplay}
	}
	return out
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

// StageTimes implements core/vllm.StagePricer: the batch's price split by pipeline stage
// (compat/kernelmodel.StageTimesOfShapes), summing to StepTime.
func (k kernelPricer) StageTimes(b engine.Batch) []engine.Duration {
	got := k.m.StageTimesOfShapes(shapes(b))
	out := make([]engine.Duration, len(got))
	for i, d := range got {
		out[i] = engine.Duration(d)
	}
	return out
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
// StepTimesAcrossDP implements core/vllm.DPPricer: one lockstep step of a data-parallel group,
// each rank priced with the group's padding (compat/kernelmodel.StepTimesAcrossDPOfShapes, the
// rule the compatibility engine prices its lockstep with), an empty rank as vLLM's dummy batch.
func (k kernelPricer) StepTimesAcrossDP(ranks []engine.Batch) []engine.Duration {
	in := make([][]kernelmodel.ReqShape, len(ranks))
	for i, b := range ranks {
		in[i] = shapes(b)
	}
	got := k.m.StepTimesAcrossDPOfShapes(in)
	out := make([]engine.Duration, len(got))
	for i, d := range got {
		out[i] = engine.Duration(d)
	}
	return out
}

func shapes(b engine.Batch) []kernelmodel.ReqShape {
	out := make([]kernelmodel.ReqShape, len(b.Reqs))
	for i, r := range b.Reqs {
		out[i] = kernelmodel.ReqShape{Scheduled: r.Tokens, Computed: r.Computed, PromptLen: r.PromptLen}
	}
	return out
}

// setIfAbsent adds key: value to a mapping node that does not state key.
func setIfAbsent(m *yaml.Node, key, value string) {
	setNodeIfAbsent(m, key, &yaml.Node{Kind: yaml.ScalarNode, Value: value})
}

// setNodeIfAbsent adds key: value to a mapping node that does not state key.
func setNodeIfAbsent(m *yaml.Node, key string, value *yaml.Node) {
	for i := 0; i+1 < len(m.Content); i += 2 {
		if m.Content[i].Value == key {
			return
		}
	}
	m.Content = append(m.Content, &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: key}, value)
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

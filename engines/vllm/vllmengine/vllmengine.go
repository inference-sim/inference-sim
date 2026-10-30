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
	"reflect"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	vllm "github.com/inference-sim/blis-vllm/core/vllm"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
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
	// role is the pool's role and connector the deployment's pd_transfer.connector, which a
	// prefill or decode pool runs as kv_producer or kv_consumer.
	role      deployment.Role
	connector string
	// offload is the deployment's offload block as the engine block's kv_offload, nil when it
	// states none (offloadOf).
	offload *yaml.Node
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
	shape, err := kernelmodel.ShapeOf(filepath.Base(path), kernelmodel.Repos{
		Scenarios: filepath.Dir(path), Catalog: s.Roots[RootCatalog], Registry: s.Roots[RootRegistry]})
	if err != nil {
		return pool{}, fmt.Errorf("vllmengine: %w", err)
	}
	blocks, kvBytes, err := poolBlocks(m, st)
	if err != nil {
		return pool{}, fmt.Errorf("vllmengine: %s: %w", filepath.Base(path), err)
	}
	off, err := offloadOf(m, shape.Offload)
	if err != nil {
		return pool{}, fmt.Errorf("vllmengine: %s: %w", filepath.Base(path), err)
	}
	return pool{model: m, blocks: blocks, kvBytes: kvBytes, engine: node, pp: max(par.PP, 1), dp: max(par.DP, 1),
		role: m.Role(), connector: shape.PDConnector, offload: off}, nil
}

// offloadIssue tracks the offload hierarchies the engine does not model; secondaryTierIssue the
// tiers below the CPU one.
const (
	offloadIssue       = "https://github.com/inference-sim/blis-vllm/issues/14"
	secondaryTierIssue = "https://github.com/inference-sim/blis-vllm/issues/13"
)

// cpuTier is the catalog device class of host DRAM, the OffloadingConnector's CPU tier.
const cpuTier = "cpu_dram"

// offloadOf maps a deployment's offload block onto the engine block's kv_offload, or refuses what
// the engine does not model. The OffloadingConnector's kv_connector_extra_config is its CPU tier:
// cpu_bytes_to_use is the tier's bytes, the engine's, which vLLM divides among its workers'
// copies (vllm/v1/kv_offload/cpu/spec.py:103-121@5df63a72), as the compatibility engine reads the
// same tier (compat/build KVOffloadBlockFromDeployment), so one deployment's 0.31.0 and 0.32.0
// runs hold one CPU tier. eviction_policy and spec_name pass through, and blocks_per_chunk comes from the connector options. What vLLM derives at start-up
// the kernel states: one worker's KV bytes for one GPU block over the offloaded groups
// (worker_kv_bytes_per_block, offloading/config.py:136-151), the engine's world size (num_workers,
// offloading/scheduler.py:335) and how many workers' copies of a block the tier holds
// (OffloadCopies: one for the replicated latent layout, offloading/config.py:158-171).
func offloadOf(m *kernelmodel.Model, o *deployment.Offload) (*yaml.Node, error) {
	if o == nil {
		return nil, nil
	}
	if o.Connector != "" && o.Connector != "OffloadingConnector" {
		return nil, fmt.Errorf("offload.connector %q is not built by the v0.32.0 engine, which runs vLLM's "+
			"OffloadingConnector (%s)", o.Connector, offloadIssue)
	}
	if o.PrefetchDepth != 0 {
		return nil, fmt.Errorf("offload.prefetch_depth %d is not a vLLM setting: the OffloadingConnector loads a hit "+
			"when the scheduler admits the request (%s)", o.PrefetchDepth, offloadIssue)
	}
	var cpu int64
	for i, t := range o.Tiers {
		if t.Device != cpuTier || i > 0 {
			return nil, fmt.Errorf("offload.tiers[%d] (%q): only one %s tier, the OffloadingConnector's CPU tier, is "+
				"modelled; secondary tiers (TieringOffloadingSpec's) are not (%s)", i, t.Device, cpuTier, secondaryTierIssue)
		}
		cpu = t.Bytes
	}
	if cpu <= 0 {
		return nil, fmt.Errorf("offload states no %s tier: the OffloadingConnector's tier is host DRAM "+
			"(cpu_bytes_to_use, required by vLLM, cpu/spec.py:97-101@5df63a72)", cpuTier)
	}
	opts, err := engineconfig.DecodeConnectorOptions(o.ConnectorOptions)
	if err != nil {
		return nil, fmt.Errorf("offload.connector_options: %w", err)
	}
	bpc := 0
	if opts != nil {
		bpc = opts.BlocksPerChunk
		*opts = engineconfig.ConnectorOptions{BlocksPerChunk: opts.BlocksPerChunk}
		if again, _ := engineconfig.DecodeConnectorOptions(o.ConnectorOptions); again != nil && !reflect.DeepEqual(*again, *opts) {
			return nil, fmt.Errorf("offload.connector_options: only blocks_per_chunk is an OffloadingConnector option (%s)",
				offloadIssue)
		}
	}
	perBlock, err := offloadBytesPerBlock(m)
	if err != nil {
		return nil, err
	}
	e, err := m.Engine()
	if err != nil {
		return nil, err
	}
	c := map[string]any{
		"connector":       "OffloadingConnector",
		"cpu_bytes":       cpu,
		"bytes_per_block": perBlock,
		"workers":         m.WorldSize(),
		"copies":          m.OffloadCopies(),
	}
	if o.Spec != "" {
		c["spec"] = o.Spec
	}
	if o.EvictionPolicy != "" {
		c["eviction_policy"] = o.EvictionPolicy
	}
	if bpc > 0 {
		c["block_size"] = bpc * e.BlockSize
	}
	var n yaml.Node
	if err := n.Encode(c); err != nil {
		return nil, err
	}
	return &n, nil
}

// offloadBytesPerBlock is one worker's KV bytes for one GPU block: the kernel's per-block KV bytes
// for a model of one full-attention group, and the pool's widest KV cache group's bytes for a
// hybrid model, as vLLM's KV tensor holds them (poolBlocks). Under pipeline parallelism it is
// pipeline rank 0's, not the binding stage's: the scheduler's KV cache config, which its
// OffloadingConnector sizes the CPU tier from (worker_kv_bytes_per_block, offloading/config.py:
// 137-147), is a copy of worker 0's (generate_scheduler_kv_cache_config, vllm/v1/core/
// kv_cache_utils.py:2289; vllm/v1/engine/core.py:339@5df63a72), whose KV tensor is the shared
// block count times rank 0's own bytes a block (get_kv_cache_configs step 5, :2664-2675).
func offloadBytesPerBlock(m *kernelmodel.Model) (int64, error) {
	b, err := m.KVBudget()
	if err != nil {
		return 0, err
	}
	runs, err := m.KVLayers()
	if err != nil || len(runs) == 0 {
		return b.PerBlockBytes, err
	}
	p, err := stagePoolsOf(m, runs)
	if err != nil {
		return 0, err
	}
	if p.perBlock[0] <= 0 {
		return 0, fmt.Errorf("vllmengine: pipeline rank 0 holds no KV layer, so vLLM's scheduler sizes no CPU tier " +
			"(worker_kv_bytes_per_block 0, offloading/config.py:137-152@5df63a72)")
	}
	return p.perBlock[0], nil
}

// offloadGroupBytes is, for each pipeline rank and each of the scheduler's KV cache groups, the
// bytes the rank's worker copies for one block of the group when it offloads it
// (vllm.KVStageGroupCopyBytes); nil for a model of one full-attention group, whose block is
// offloadBytesPerBlock whole.
func offloadGroupBytes(m *kernelmodel.Model) ([][]int64, error) {
	runs, err := m.KVLayers()
	if err != nil || len(runs) == 0 {
		return nil, err
	}
	p, err := stagePoolsOf(m, runs)
	if err != nil {
		return nil, err
	}
	e, err := m.Engine()
	if err != nil {
		return nil, err
	}
	disable := e.DisableHybridKVCacheManager != nil && *e.DisableHybridKVCacheManager
	return vllm.KVStageGroupCopyBytes(vllmRuns(runs), p.spans, disable, e.BlockSize, p.page)
}

// stagePools is a pool's KV cache groups sized per pipeline rank: each rank's bytes a block and
// the memory its pool may take, the rank whose pool binds, and its block count. At pp 1 it is
// the one rank's.
type stagePools struct {
	perBlock, avail []int64
	binding         int
	blocks          int64
	// spans are each rank's [lo, hi) of the model's KV layers, and page the bytes for one block
	// of every layer the element sizes are resolved from (KVStagePoolBytesPerBlock's kernelPerBlock).
	spans [][2]int
	page  int64
}

// stagePoolsOf sizes a pool over its KV cache groups. vLLM forms the groups once over every
// layer (get_kv_cache_groups) and sizes each worker's pool from its own memory over the groups
// projected onto its layers (_project_kv_cache_groups_to_worker), keeping the smallest
// (get_kv_cache_configs, vllm/v1/core/kv_cache_utils.py:2486-2528, 2595-2603, 2650-2680@5df63a72).
// A rank's memory is the kernel's for that rank (kernelmodel StageBudgets): its own budget less
// its own fixed occupancy. The layers' element bytes come from the kernel's bytes for one block of
// every KV layer (KVLayerPageBytes), window layers included, not from SequenceVariableBytes,
// which is only the share that grows with context.
func stagePoolsOf(m *kernelmodel.Model, runs []kernelmodel.KVLayerRun) (stagePools, error) {
	b, err := m.KVBudget()
	if err != nil {
		return stagePools{}, err
	}
	e, err := m.Engine()
	if err != nil {
		return stagePools{}, err
	}
	page, err := m.KVLayerPageBytes()
	if err != nil {
		return stagePools{}, err
	}
	disable := e.DisableHybridKVCacheManager != nil && *e.DisableHybridKVCacheManager
	pp := m.PipelineStages()
	if pp == 1 {
		per, err := vllm.KVPoolBytesPerBlock(vllmRuns(runs), disable, e.BlockSize, page)
		if err != nil {
			return stagePools{}, err
		}
		n := 0
		for _, r := range runs {
			n += r.Count
		}
		return stagePools{perBlock: []int64{per}, avail: []int64{b.AllocatableBytes},
			blocks: b.AllocatableBytes / per, spans: [][2]int{{0, n}}, page: page}, nil
	}
	budgets, err := m.StageBudgets()
	if err != nil {
		return stagePools{}, err
	}
	// Every stage's span of the model's layers, in model order. The kernel's page bytes are the
	// binding rank's; every layer is at one cache dtype, so its bytes over that rank's elements
	// is the bytes an element, which sizes the whole model's block of every layer.
	spans := make([][2]int, pp)
	avail := make([]int64, pp)
	at, bindElems := 0, int64(0)
	for s := range pp {
		stage, err := m.KVLayersOfStage(s)
		if err != nil {
			return stagePools{}, err
		}
		n := 0
		for _, r := range stage {
			n += r.Count
		}
		spans[s] = [2]int{at, at + n}
		at += n
		avail[s] = budgets[s].AllocatableBytes
		if budgets[s].PageBytes == page && bindElems == 0 {
			bindElems = kvElems(stage)
		}
	}
	whole := 0
	for _, r := range runs {
		whole += r.Count
	}
	if at != whole || bindElems == 0 {
		return stagePools{}, fmt.Errorf("vllmengine: the pipeline stages hold %d KV layers of the model's %d", at, whole)
	}
	modelPage := page * kvElems(runs) / bindElems
	per, err := vllm.KVStagePoolBytesPerBlock(vllmRuns(runs), spans, disable, e.BlockSize, modelPage)
	if err != nil {
		return stagePools{}, err
	}
	blocks, binding, err := vllm.KVStagePoolBlocks(vllmRuns(runs), spans, disable, e.BlockSize, modelPage, avail)
	if err != nil {
		return stagePools{}, err
	}
	return stagePools{perBlock: per, avail: avail, binding: binding, blocks: blocks, spans: spans, page: modelPage}, nil
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
// Under pipeline parallelism each stage's rank sizes its own pool from its own memory over the
// model's groups projected onto its layers, and the engine keeps the smallest (stagePoolsOf);
// the memory returned is the binding rank's.
func poolBlocks(m *kernelmodel.Model, st kernelmodel.Settings) (int64, int64, error) {
	runs, err := m.KVLayers()
	if err != nil || len(runs) == 0 {
		return st.KVBlocks, 0, err
	}
	p, err := stagePoolsOf(m, runs)
	if err != nil {
		return 0, 0, err
	}
	return p.blocks, p.avail[p.binding], nil
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
	if role := kvRoleOf(p.role); role != "" && p.connector != "" {
		// A P/D pool's kv_transfer_config: the deployment's pd_transfer.connector, and the pool's
		// role as vLLM's kv_role (a prefill pool produces KV, a decode pool consumes it).
		kt := yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
		setIfAbsent(&kt, "connector", p.connector)
		setIfAbsent(&kt, "kv_role", role)
		setNodeIfAbsent(&block, "kv_transfer", &kt)
	}
	if p.offload != nil {
		// The deployment's offload block as the OffloadingConnector's configuration (offloadOf).
		setNodeIfAbsent(&block, "kv_offload", cloneNode(p.offload))
	}
	if runs, err := p.model.KVLayers(); err == nil && len(runs) > 0 {
		// The model's attention layers by the KV cache spec vLLM builds for each, which the
		// factory groups as get_kv_cache_groups does (vllm/v1/core/kv_cache_utils.py:2159@5df63a72).
		setNodeIfAbsent(&block, "kv_layers", kvLayersNode(runs))
	}
	return block
}

// kvRoleOf is a pool role's kv_role, "" for a pool that runs no P/D connector.
func kvRoleOf(r deployment.Role) string {
	switch r {
	case deployment.RolePrefill:
		return "kv_producer"
	case deployment.RoleDecode:
		return "kv_consumer"
	}
	return ""
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
	k := kernelPricer{m: p.model}
	if per, err := offloadBytesPerBlock(p.model); err == nil {
		k.blockBytes = per
	}
	if g, err := offloadGroupBytes(p.model); err == nil {
		k.groupBytes = g
	}
	return k, nil
}

// kernelPricer prices an engine.Batch with the latency kernel, as compat/kernelmodel prices the
// compatibility engine's batches (Model.StepTime): the kernel's StepTime of the batch, its
// NoOverlap figure, in microseconds, at least one.
type kernelPricer struct {
	m *kernelmodel.Model
	// blockBytes is a worker's KV bytes a GPU block, the pool's own (offloadBytesPerBlock), which
	// an offload transfer moves per block when its groups are not known; 0 prices at the kernel's
	// per-block bytes.
	blockBytes int64
	// groupBytes are, by pipeline rank and KV cache group, the bytes a worker copies for one
	// block of the group (offloadGroupBytes); nil prices every block at blockBytes.
	groupBytes [][]int64
}

// StepTime implements core/vllm.Pricer.
func (k kernelPricer) StepTime(b engine.Batch) engine.Duration {
	return engine.Duration(k.m.StepTimeOfShapes(shapes(b)))
}

// VerifiedDrafts implements core/vllm.VerifyingPricer: how many of each request's scheduled
// drafts the verifier runs under DSpark adaptive verification, from the same kernel and the same
// shapes StepTime priced (compat/kernelmodel.VerifiedDraftsOfShapes); nil when the pool does not
// verify adaptively, so the engine caps acceptance at the scheduled drafts as before.
func (k kernelPricer) VerifiedDrafts(b engine.Batch) []int {
	return k.m.VerifiedDraftsOfShapes(shapes(b))
}

// TransferTime implements core/vllm.TransferPricer: an offload transfer of t.Blocks GPU blocks
// to or from the CPU tier, as the kernel prices it (TierTime at cpu_dram, one at a time, as
// vLLM's offload handler runs its copies on one stream chain: kv_offload/cpu/gpu_worker.py).
func (k kernelPricer) TransferTime(t engine.Transfer) engine.Duration {
	return engine.Duration(k.m.TierTransferTicksOf("cpu_dram", t.Kind == engine.TransferStore, int64(t.Blocks)*k.blockBytes,
		int64(t.Blocks)))
}

// GroupTransferTime implements core/vllm.GroupTransferPricer: an offload job of groupBlocks[g]
// GPU blocks of each KV cache group moves, on each worker, those blocks' pages of the group's
// layers on that worker (vLLM's CPU offload worker copies one (block, layer) page at a time,
// _fill_direct_ops, vllm/v1/kv_offload/cpu/gpu_worker.py:395-416@5df63a72), and the job ends
// when the worker moving the most bytes does. A group of fewer layers than the widest moves
// fewer bytes a block than the pool's stride.
func (k kernelPricer) GroupTransferTime(t engine.Transfer, groupBlocks []int) engine.Duration {
	if len(k.groupBytes) == 0 || len(groupBlocks) != len(k.groupBytes[0]) {
		return k.TransferTime(t)
	}
	var bytes int64
	for _, per := range k.groupBytes {
		var b int64
		for g, n := range groupBlocks {
			b += int64(n) * per[g]
		}
		bytes = max(bytes, b)
	}
	return engine.Duration(k.m.TierTransferTicksOf("cpu_dram", t.Kind == engine.TransferStore, bytes, int64(t.Blocks)))
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

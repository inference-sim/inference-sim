package latencykernel

import (
	"fmt"
	"maps"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/vocab"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// Inputs are the documents a kernel is built from. They arrive already parsed: loading
// belongs to blis-schemas, and New takes documents rather than paths so a caller holding
// structs — a config search producing deployment variants never written to disk, a test, a
// generator — does not have to serialize them first. Open is the path-based half of the
// pair.
//
// New DOES validate them, at the field layer, which is the half of blis-schemas' two-layer
// validation that is version-independent and always the author's to fix. See New.
type Inputs struct {
	Scenario *scenario.Scenario
	// Deployment is the tunable configuration applied to the scenario: the pools that lay
	// the model out. It is a separate document because a scenario fixes the immutable
	// problem and an optimizer sweeps deployments against it.
	Deployment *deployment.Deployment
	// PoolIndex selects which pool of a disaggregated deployment this kernel prices. Each
	// pool runs its own engine with its own settings, so one kernel per pool.
	PoolIndex    int
	Model        *model.Graph
	Chip         *hardware.Chip
	Fabric       *hardware.Fabric
	Devices      []*hardware.StorageDevice
	Coefficients []*coefficient.Set
	// Rules carry the engine-version behaviour the resolver needs.
	Rules resolve.EngineRules
	// Engine is the priced pool's engine settings, decoded by the caller: the engine block is
	// the engine model's to open, not this cost model's, so it comes in already read, as Rules
	// does (for vLLM, blis-latency-kernel/vllmsettings). Required.
	Engine *enginesettings.Settings
	// Draft is the graph of the catalog model the deployment names as its drafter
	// (speculative.draft_model), a checkpoint of its own; nil when the deployment names none.
	// It states itself as a speculator and is priced as the target's (withDrafter).
	Draft *model.Graph
	// Routing is the run's resolved expert-parallel routing imbalance (blis-schemas
	// routing.Resolve), which depends on the model and the traffic and so is resolved by
	// the caller, as speculative acceptance is. Nil prices an expert-parallel group as
	// evenly loaded and records that as an assumption.
	Routing *routing.Resolution
	// Implementations are the kernel implementations the pool runs, by the lowered primitive each
	// executes: the library, variant and code hash the engine lowered it to, which is
	// plan.PoolPlan.ImplementationsByPrimitive() for a caller that holds the pool's plan (design
	// VAL-8). A coefficient that lists kernel implementations is used only for the primitive it
	// prices (coefficient.PrimitiveOf), and only when every implementation supplied for that
	// primitive is one it lists, by library, variant and code hash together; a different hash of
	// the same kernel, or a kernel of another primitive, is refused. A primitive with nothing
	// supplied uses only coefficients that list no implementation, as every coefficient of the
	// registry did before any listed one.
	Implementations map[plan.Primitive][]plan.KernelImpl
	// Operations are the pool's lowered operations, each with the layer it belongs to, its
	// primitive and the kernel that executes it: plan.PoolPlan.Operations() for a caller that
	// holds the pool's plan (design VAL-8, 6.2). A coefficient is then matched against the
	// kernels of the operations that consume it, not against every kernel that runs its
	// primitive: a value valid for the kernel that runs the sliding-window layers prices them
	// whatever runs the full-attention ones. A name that states a class (coefficient.ClassOf)
	// is consumed by the operations of that class; any other by every operation of its
	// primitive. When Operations is given, Implementations is derived from it and may be left
	// empty; if both are given they must name the same implementations.
	Operations []plan.Operation
	// DraftFirstPassRows and DraftPassRequests state how an autoregressive drafter's passes run,
	// as plan.PoolPlan.Drafts.FirstPassRows and PassRequests for a caller that holds the pool's
	// plan: the engine resolves them, in neutral words, and the kernel prices what it is told
	// (SUB-6). A deployment that speculates through an autoregressive drafter, named or the
	// model graph's own, with either empty is refused (draft_passes_unstated), and an unknown word
	// is refused (draft_passes_unknown). A parallel-block drafter needs neither; both are ignored
	// while the deployment does not speculate.
	DraftFirstPassRows plan.FirstPassRows
	DraftPassRequests  plan.PassRequests
	// Acceptance is the run's resolved speculative acceptance (blis-schemas acceptance.Resolve),
	// resolved by the caller as Routing is. It is read only when the engine enables adaptive
	// verification (enginesettings.Speculative.EnableAdaptiveVerification), where it is
	// required: the verify budget the engine chooses depends on it (adaptive.go).
	Acceptance *acceptance.Resolution
	// PoolFirstNode is the cluster node the pool's first rank sits on, when it is not where the
	// deployment's order puts it: pools take consecutive nodes in declaration order, so nil
	// means the sum of the node counts of the pools before PoolIndex. The pool's instances are
	// laid out one after another from there, instance GPUs each. It is read only when the
	// scenario states switch groups and an instance spans nodes, where it decides whether a
	// collective stays under one switch. A collective that crosses a switch boundary is not
	// priced (SUB-17 asks for its path and its per-link demand), so such a pool is refused with
	// switch_group_collective_crosses.
	PoolFirstNode *int
}

// New resolves the inputs into a kernel.
//
// Everything that can be decided once is decided here: which coefficients apply, which
// requested settings the layout overrides, and the model graph flattened to a per-kind
// plan. The result holds no mutable state, so its methods are pure and safe to call
// concurrently.
//
// Construction is deliberately strict. A missing coefficient, an unresolvable condition or
// a non-positive rate is an error rather than a default, because each would otherwise
// produce a step time that looks plausible and is wrong by whatever the term contributes.
// validateDocuments extends that strictness to the documents themselves.
func New(in Inputs) (*Kernel, error) {
	if in.Scenario == nil || in.Deployment == nil || in.Model == nil || in.Chip == nil {
		return nil, fmt.Errorf(
			"a kernel needs a scenario, a deployment, a model graph and a chip")
	}
	if in.PoolIndex < 0 || in.PoolIndex >= len(in.Deployment.Pools) {
		return nil, fmt.Errorf("pool index %d is outside the deployment's %d pool(s)",
			in.PoolIndex, len(in.Deployment.Pools))
	}
	if isNil(in.Rules) {
		return nil, fmt.Errorf("a kernel needs engine rules for version %q",
			in.Scenario.EngineRelease())
	}
	if in.Engine == nil {
		return nil, fmt.Errorf("a kernel needs the pool's engine settings, decoded by the " +
			"engine model (Inputs.Engine)")
	}
	if err := rulesAnswer(in.Rules); err != nil {
		return nil, fmt.Errorf("the engine rules for version %q cannot answer: %w",
			in.Scenario.EngineRelease(), err)
	}
	return buildGuarded(in)
}

// buildGuarded is New past its nil checks. The rules value is the caller's code, so a rules method
// that panics while the kernel is built is reported as a refusal naming the rules, not let
// out of New: a configuration search gets an error it can report and move past. It covers
// construction only; a rules method that panics later, inside a pricing call, is a defect
// in the caller's rules and panics there.
func buildGuarded(in Inputs) (k *Kernel, err error) {
	defer func() {
		if p := recover(); p != nil {
			if _, ok := p.(rulesPanic); !ok {
				panic(p)
			}
			k, err = nil, fmt.Errorf("the engine rules for version %q cannot answer: %v",
				in.Scenario.EngineRelease(), p.(rulesPanic).v)
		}
	}()
	in.Rules = panicTagged{in.Rules}
	k, err = construct(in)
	if k != nil {
		k.rules = k.rules.(panicTagged).EngineRules
	}
	return k, err
}

// construct builds the kernel. It runs only under buildGuarded, whose rules are tagged so a panic
// they raise is told apart from one the kernel raises.
func construct(in Inputs) (*Kernel, error) {
	if err := validateDocuments(in); err != nil {
		return nil, err
	}
	running := runningImpls(in.Implementations)
	if err := coefficient.ValidateRunning(running); err != nil {
		return nil, fmt.Errorf("the kernel implementations that run: %w", err)
	}
	if len(in.Operations) > 0 {
		pooled := runningImpls(plan.ImplementationsOf(in.Operations))
		if err := coefficient.ValidateRunning(pooled); err != nil {
			return nil, fmt.Errorf("the kernel implementations that run: %w", err)
		}
		if len(running) > 0 && !sameImplementations(running, pooled) {
			return nil, fmt.Errorf("Inputs.Implementations and Inputs.Operations name different kernel " +
				"implementations; give the operations alone, and the implementations are derived from them")
		}
	}
	// A deep copy: the engine block's optional settings are pointers, and a kernel that
	// shared them with the caller's documents would change its answers when the caller
	// edited a deployment it had already priced -- the config-search shape Inputs exists
	// for. The kernel's purity is a promise about its own state, so it owns that state.
	pool := clonePool(in.Deployment.Pools[in.PoolIndex])
	eng := in.Engine.Clone()

	fab := resolve.Fabric{IntraNodeBwGBps: in.Chip.IntraNodeBwGBps}
	fabricName := ""
	if in.Fabric != nil {
		fab.InterNodeBwGBps = in.Fabric.InterNodeBwGBps
		fabricName = in.Fabric.Name
	}
	// The cluster's packaging, resolved from the scenario and the chip's catalog entry: the
	// one source of the node and rack sizes the kernel prices (SUB-3). validateDocuments has
	// already refused a scenario that contradicts the entry.
	pk, err := in.Scenario.Cluster.Packaging(in.Chip)
	if err != nil {
		return nil, fmt.Errorf("resolving the cluster's packaging: %w", err)
	}
	sw, err := in.Scenario.Cluster.Switching()
	if err != nil {
		return nil, fmt.Errorf("resolving the cluster's switch groups: %w", err)
	}
	// A rack is one NVLink domain where the chip says so: the catalog records a rack tier
	// only on parts whose NVLink fabric spans chassis, and the resolved rack is the entry's.
	fab.RackIsOneDomain = pk.GPUsPerRack > 0 && in.Chip.IntraRackBwGBps > 0

	// The graph as served on this chip. A graph that states facts by GPU architecture --
	// DeepSeek-V4.1's resident widths and KV records differ between SM90 and SM100 --
	// resolves through blis-schemas' one overlay, and one that does not is the same graph on
	// every chip. A chip that states no compute capability cannot run a graph that
	// distinguishes architectures, and ForCapability refuses it rather than giving it the
	// base's facts.
	family, _ := in.Chip.CapabilityMajor()
	g, err := in.Model.ForCapability(family)
	if err != nil {
		return nil, fmt.Errorf("resolving the model graph for %s: %w", in.Chip.Name, err)
	}
	// The target's own graph, before any drafter joins it: the layout the engine says it serves
	// is the target's, and the drafter's layers (its own checkpoint, at its own widths) are not
	// part of it.
	targetGraph := g
	// A drafter named by the deployment, a checkpoint of its own, joins the target graph as
	// its speculator; one named without its graph is refused rather than priced as free.
	if name := draftModelName(eng); name != "" || in.Draft != nil {
		switch {
		case in.Draft == nil:
			return nil, fmt.Errorf("speculative.draft_model %q names a drafter whose graph was "+
				"not given (Inputs.Draft), so its passes cannot be priced", name)
		case name != in.Draft.Name:
			return nil, fmt.Errorf("speculative.draft_model %q, and the drafter graph given is %q",
				name, in.Draft.Name)
		}
		d, err := in.Draft.ForCapability(family)
		if err != nil {
			return nil, fmt.Errorf("resolving the drafter graph for %s: %w", in.Chip.Name, err)
		}
		if g, err = withDrafter(g, d); err != nil {
			return nil, err
		}
		if p := g.Validate(); !p.OK() {
			return nil, fmt.Errorf("the target graph with drafter %s does not validate:\n%w",
				d.Name, p)
		}
	}
	in.Model = g

	layout, err := resolve.ResolveLayoutFor(pk, pool, eng, fab, in.Rules,
		g.Global.Architecture)
	if err != nil {
		return nil, fmt.Errorf("resolving the layout: %w", err)
	}

	ops, err := loweredOperations(g, in.Operations, *in.Chip, family)
	if err != nil {
		return nil, err
	}
	coeffs, err := resolve.Load(in.Coefficients, resolve.Scope{
		Hardware: in.Chip.Name, Model: in.Model.Name,
		TP: layout.TP, EP: layout.ExpertWidth, NodesSpanned: layout.NodesSpanned,
		Fabric:          fabricName,
		Implementations: running, Operations: ops,
	})
	if err != nil {
		return nil, fmt.Errorf("resolving coefficients: %w", err)
	}
	// The fabric's measured effective fraction of its nominal line rate (blis-registry
	// fabric_bandwidth_efficiency, scoped by fabric). The catalog states only the nominal
	// figure, and two clusters of comparable silicon on RoCE and on InfiniBand differ by
	// what this corrects; without it both price at their datasheet rates.
	fabricEfficiency, fabricEfficiencyStated, err := resolveFabricEfficiency(coeffs, in.Fabric)
	if err != nil {
		return nil, err
	}
	fab.InterNodeEfficiency = fabricEfficiency

	k := &Kernel{layout: layout, fabric: fab, pool: pool, engine: eng, chip: *in.Chip, rules: in.Rules,
		family: family, architecture: g.Global.Architecture, graph: g,
		switching:           sw,
		statedFirstPassRows: in.DraftFirstPassRows, statedPassRequests: in.DraftPassRequests}
	if err := k.checkCollectivesStayInAGroup(poolFirstNode(in)); err != nil {
		return nil, err
	}
	if in.Fabric != nil && !fabricEfficiencyStated {
		k.assume("fabric_bandwidth_efficiency", "1",
			"the scenario's coefficient sets carry no "+fabricEfficiencyName+" scoped to fabric "+
				in.Fabric.Name+", so every byte that crosses it is priced at the catalog's "+
				"nominal line rate")
	}
	k.tp = float64(max(layout.TP, 1))
	k.kvConnector = (in.Deployment.Offload != nil && in.Deployment.Offload.Connector != "") ||
		(in.Deployment.PDTransfer != nil && in.Deployment.PDTransfer.Connector != "")

	// The served weight format, which the scenario may override. A bf16 checkpoint
	// served fp8 reads half the weight bytes, runs against a different compute peak and
	// sits on a different efficiency envelope, so this has to be resolved before
	// anything is priced.
	quantization := eng.Quantization
	// A DeepSeek-V4-family FP8 checkpoint is served by its own quantization method whatever
	// --quantization names: DeepseekV4Fp8Config's override claims it
	// (vllm/models/deepseek_v41/quant_config.py:139-167) and ModelConfig replaces the
	// requested method with it (vllm/config/model.py:1382-1403). So the checkpoint's own
	// formats are what run.
	if deepseekV4Family(in.Model) && in.Model.Global.WeightDType == model.DTypeFP8 &&
		quantization != "" {
		if quantization != "deepseek_v4_fp8" {
			k.layout.Overrides = append(k.layout.Overrides, resolve.Override{
				Field: "quantization", Requested: quantization, Resolved: "deepseek_v4_fp8",
				Reason: "vLLM v0.31.0 serves a DeepSeek-V4-family FP8 checkpoint with its own " +
					"quantization method whatever is requested",
			})
		}
		quantization = ""
	}
	// The layout the engine says it serves is shown against the target's own graph, before a
	// drafter joins it: the drafter's layers are another checkpoint's.
	if err := checkServedWeights(targetGraph, eng.ServedWeights); err != nil {
		return nil, err
	}
	served, err := servedDType(in.Model.Global.WeightDType, quantization)
	if err != nil {
		return nil, err
	}
	k.servedDType = served

	if err := checkCacheDTypes(eng.CacheDType, eng.MambaCacheDType,
		attentionKinds(in.Model)); err != nil {
		return nil, err
	}
	cacheBytes := cacheDTypeBytes(eng.CacheDType, served)
	// The cache dtype a stated cache record is looked up by: the requested one, as the
	// architecture's attention backend resolves it. DeepSeek-V4.1 rewrites auto and fp8 to
	// its packed record (fp8_ds_mla under FlashMLA, nvfp4_ds_mla under SM100's mega
	// attention), which is what Cache.EntryBytes is keyed by.
	k.cacheDType = in.Rules.ResolveCacheDType(k.architecture, family, layout.TP,
		eng.CacheDType)
	if stated := eng.CacheDType; stated != "" && k.cacheDType != stated {
		k.layout.Overrides = append(k.layout.Overrides, resolve.Override{
			Field: "cache_dtype", Requested: stated, Resolved: k.cacheDType,
			Reason: "the attention backend of " + k.architecture + " serves this cache as " +
				k.cacheDType + " (rules pack CacheDTypeResolutions)",
		})
	}
	stateBytes := cacheDTypeBytes(eng.MambaCacheDType, model.DTypeFP32)

	// Pipeline parallelism is priced per stage (pipeline.go). What it is not combined with
	// here is refused rather than priced half-way: a stage-level split of data or context
	// parallelism, which no stage split here describes (DP with PP is refused by
	// blis-vllm's v0.32.0 engine as well, core/vllm/config.go). A tensor-parallel width that
	// does not divide a node is priced: liftPipeline places a stage's ranks one by one.
	if pp := pool.Parallel.PP; pp > 1 {
		switch {
		case pool.Parallel.DP > 1:
			return nil, fmt.Errorf("pipeline parallelism (pp %d) with data parallelism "+
				"(dp %d) is not priced: each data-parallel rank would be a pipeline of its "+
				"own, which no stage split here describes", pp, pool.Parallel.DP)
		case pool.Parallel.PCP > 1 || pool.Parallel.DCP > 1:
			return nil, fmt.Errorf("pipeline parallelism (pp %d) with context parallelism "+
				"(pcp %d, dcp %d) is not priced", pp, pool.Parallel.PCP, pool.Parallel.DCP)
		}
	}

	if err := refuseUnsplittableKVHeads(in.Model, layout.TP); err != nil {
		return nil, err
	}
	if err := refuseV4Layouts(in, k.cacheDType); err != nil {
		return nil, err
	}
	if in.Model.Modality == model.ModalityTextDecoderOfMultimodal {
		k.assume("modality", string(in.Model.Modality),
			"the graph prices the text decoder of a multimodal model: its vision or audio "+
				"tower, and the encoder work a multimodal request adds, are not priced")
	}

	plan, err := price.BuildPlanWith(in.Model, emitter{layout}, price.PlanOptions{
		WeightBytesPerParam: func(n model.Node) float64 {
			return weightBytesPerParam(in.Model, n, served)
		},
		Format:     func(n model.Node) price.Format { return weightFormat(in.Model, n, served) },
		EntryBytes: func(c model.Cache) int { return c.EntryBytes(k.cacheDType) },
		StateBytes: stateBytes,
	})
	if err != nil {
		return nil, fmt.Errorf("planning the model graph: %w", err)
	}
	k.plan = plan

	// The recurrent cache mode as it runs. The config's default is "none"
	// (vllm/config/cache.py:190), and for a hybrid model vLLM v0.31.0 turns "none" into
	// "align" whenever prefix caching is on (vllm/model_executor/models/config.py:640-642,
	// run for every hybrid model from vllm/config/vllm.py:2797-2798) -- a STATED "none"
	// included, since the hook cannot tell it from the default. Prefix caching is on unless
	// disabled (vllm/config/cache.py:142).
	hybrid := false
	for _, l := range plan.Layers {
		hybrid = hybrid || l.RecurrentKind != ""
	}
	prefixCaching := true
	if pc := eng.EnablePrefixCaching; pc != nil && !*pc {
		prefixCaching = false
	}
	stated := price.RecurrentCacheMode(eng.MambaCacheMode)
	k.recurrentCacheMode = stated
	if stated == "" {
		k.recurrentCacheMode = price.RecurrentCacheNone
	}
	if hybrid && prefixCaching && k.recurrentCacheMode == price.RecurrentCacheNone {
		k.recurrentCacheMode = price.RecurrentCacheAlign
	}
	if hybrid {
		switch {
		case stated == "":
			k.assume("mamba_cache_mode", string(k.recurrentCacheMode),
				"vLLM v0.31.0's choice for a hybrid model: align with prefix caching on, "+
					"none with it off (vllm/model_executor/models/config.py:640-642)")
		case stated != k.recurrentCacheMode:
			k.layout.Overrides = append(k.layout.Overrides, resolve.Override{
				Field: "mamba_cache_mode", Requested: string(stated),
				Resolved: string(k.recurrentCacheMode),
				Reason: "with prefix caching on, vLLM v0.31.0 runs a hybrid model's " +
					"recurrent cache in align mode whatever none was stated " +
					"(vllm/model_executor/models/config.py:640-642)",
			})
		}
	}

	k.resolveMaxModelLen(in.Model)
	k.blockSize = eng.BlockSize
	dsa := isDSA(in.Model)
	v4 := deepseekV4Family(in.Model)
	k.blockSizeFinal = !hybrid && (k.blockSize > 0 || dsa || v4)
	switch {
	case v4 && k.blockSize > 0 && k.blockSize%v4BlockSize(in.Model.Global.Architecture, k.hopper) != 0:
		return nil, fmt.Errorf("block_size %d on a DeepSeek-V4-family model: its sparse "+
			"attention and indexer run %d-token kernel blocks on this part, so a stated size "+
			"must be a multiple of it, and the engine refuses this layout at startup",
			k.blockSize, v4BlockSize(in.Model.Global.Architecture, k.hopper))
	case v4 && k.blockSize <= 0:
		k.blockSize = v4BlockSize(in.Model.Global.Architecture, k.hopper)
		k.assume("block_size", strconv.Itoa(k.blockSize),
			"what vLLM v0.31.0 picks for a DeepSeek-V4-family model on this part, the only "+
				"kernel block its sparse backends accept (vllm/models/deepseek_v41/"+
				"sparse_mla.py:89-90; vllm/models/deepseek_v4/sparse_mla.py:59-61)")
	case dsa && k.blockSize > 0 && k.blockSize%dsaBlockSize != 0:
		return nil, fmt.Errorf("block_size %d on a DSA sparse-MLA model: its indexer runs "+
			"on %d-token kernel blocks, so a stated size must be a multiple of %d, and the "+
			"engine refuses this layout at startup (see dsaBlockSize)",
			k.blockSize, dsaBlockSize, dsaBlockSize)
	case dsa && k.blockSize <= 0:
		k.blockSize = dsaBlockSize
		k.assume("block_size", strconv.Itoa(dsaBlockSize),
			"what vLLM v0.31.0 picks for a DSA sparse-MLA model, whose indexer accepts "+
				"only 64-token blocks (vllm/v1/attention/backends/mla/indexer.py:203-204; "+
				"vllm/platforms/interface.py:666-689)")
	case k.blockSize <= 0:
		k.blockSize = 16
		k.assume("block_size", "16",
			"CacheConfig.DEFAULT_BLOCK_SIZE (vllm/config/cache.py:71 at v0.31.0); the "+
				"platform substitutes a size every attention backend supports where the "+
				"default is not one, and aligns a hybrid model's block to its recurrent "+
				"page even over a stated size (vllm/platforms/interface.py:683-702), "+
				"neither of which the kernel can see")
	}
	if err := k.resolveContextParallel(in); err != nil {
		return nil, err
	}

	if err := k.resolveAsyncScheduling(in.Rules); err != nil {
		return nil, err
	}
	if err := k.lift(coeffs, in.Model, cacheBytes); err != nil {
		return nil, err
	}
	if err := k.liftPipeline(in.Model); err != nil {
		return nil, err
	}
	k.noteShardedProjections()
	k.adoptRouting(in.Routing, k.totalExperts > 0)
	if err := k.liftTiers(coeffs, in); err != nil {
		return nil, err
	}
	// The engine behaviour priced here is vLLM v0.31.0's, whatever release the scenario
	// declares; the rules pack follows the declared release. Where they differ, a behaviour
	// that changed between the two is priced at v0.31.0, and that is disclosed rather than
	// left for a reader to infer from the version string.
	if v := in.Scenario.EngineRelease(); v != EngineBehaviourVersion {
		k.assume("engine_version", EngineBehaviourVersion,
			"the scenario declares vLLM "+v+" and its rules pack is that release's, but "+
				"the engine behaviour this kernel encodes -- backend selection, defaults, "+
				"refusals -- is v"+EngineBehaviourVersion+"'s (README, Sources of truth)")
	}
	// Last before provenance: the adaptive budget's cost curves are this kernel's own step
	// prices, so every pricing term is settled first (adaptive.go).
	if err := k.resolveAdaptive(in.Acceptance); err != nil {
		return nil, err
	}
	k.buildProvenance(coeffs)
	k.selected = coeffs.Selected()
	k.running = coeffs.Implementations()
	k.trust = buildTrust(k.selected)
	k.fixed = k.computeFixedBytes(in.Model)
	if err := k.refuseUnallocatableDeepEP(); err != nil {
		return nil, err
	}
	return k, nil
}

// resolveContextParallel settles the decode-context knobs and refuses the context-parallel
// layouts vLLM v0.31.0 cannot run for THIS model. blis-schemas refuses the layouts that are
// inadmissible for every model (parallel.py:563-578 and the world size); what remains
// depends on the model's attention, which only the graph knows.
func (k *Kernel) resolveContextParallel(in Inputs) error {
	pl := k.pool.Parallel

	// PCP RUNS ONLY WHERE EVERY LAYER'S BACKEND SUPPORTS IT, which at v0.31.0 means a
	// stack of latent attention and nothing else. The engine asserts it per layer at startup
	// (vllm/v1/worker/cp_utils.py:35-38, "PCP requires attention backend support"), and a
	// backend reports support through its implementation class:
	//
	//   - attention implementations default to no (AttentionImplBase.supports_pcp = False,
	//     vllm/v1/attention/backend.py:843) and only MLAAttentionImpl says yes (:1048);
	//   - recurrent backends -- Mamba1, Mamba2, GDN, linear attention, and Kimi-K3's KDA,
	//     which subclasses GDN -- declare no implementation class at all, so the lookup
	//     raises and support reads as no (:230-235).
	//
	// So a stack with any full-attention, sliding-window or recurrent layer does not start
	// under pcp > 1, whichever backend is chosen, and pricing it would describe nothing the
	// engine runs. Model Runner V2, which runs PCP, says the same in one line: "MRV2 PCP
	// currently supports MLA models only" (vllm/v1/worker/gpu/pcp_manager.py:132-133).
	//
	// Admitting every latent stack is the other edge, and it is not exact: a latent layer
	// served by a backend that declares no implementation class refuses PCP too, and
	// DeepSeek-V4's sparse-MLA backends are such (vllm/models/deepseek_v4/sparse_mla.py).
	// The graph cannot name the backend, so that refusal is the engine's to make.
	if pl.PCP > 1 {
		for _, l := range k.plan.Layers {
			var kind string
			switch {
			case l.AttnQHeads > 0 && !latentAttention(l.AttnKind):
				kind = string(l.AttnKind) + " attention"
			case l.RecurrentKind != "":
				kind = "a " + string(l.RecurrentKind) + " recurrent mixer"
			default:
				continue
			}
			return fmt.Errorf(
				"prefill-context parallelism (pcp %d) needs every layer's backend to support "+
					"it, and layer kind %q has %s, whose backend does not: the engine "+
					"refuses this layout at startup", pl.PCP, l.ID, kind)
		}
	}

	// PCP WITH DCP RUNS ONLY ON DSA SPARSE-MLA LAYERS. An MLA attention layer refuses the
	// combination unless its class opts in -- "Under PCP+DCP only the decode rows carry an
	// LSE; the base forward merges a full-batch LSE, so subclasses opt in"
	// (MLAAttention.supports_pcp_dcp = False, vllm/model_executor/layers/attention/
	// mla_attention.py:440-442, raised at :684-687) -- and the one class that opts in is
	// DeepseekV32Attention (vllm/models/deepseek_v32/attention.py:121-124), the DSA layer
	// the catalog states as sparse_mla. A plain mla layer therefore does not start.
	//
	// The converse is not certain: a sparse_mla layer served by another class may refuse
	// too, and the kernel cannot see the class. It is admitted here because the DSA
	// family -- DeepSeek-V3.2, GLM-5 -- is what the kind describes, and it is stated
	// rather than assumed silent.
	if pl.PCP > 1 && pl.DCP > 1 {
		for _, l := range k.plan.Layers {
			if l.AttnQHeads > 0 && l.AttnKind != model.AttentionSparseMLA {
				return fmt.Errorf(
					"prefill- and decode-context parallelism together (pcp %d, dcp %d) run "+
						"only on DSA sparse-MLA attention, and layer kind %q is %s: its "+
						"attention class does not support PCP with DCP, so the engine "+
						"refuses this layout at startup", pl.PCP, pl.DCP, l.ID, l.AttnKind)
			}
		}
	}

	// A REPLICATED QUERY PROJECTION IS NOT PRICED, so it is refused where it would take
	// effect rather than priced half-way. dcp_q_replicate skips the decode query
	// all-gather, and pays for it by building the MLA query projection over tp/dcp ranks
	// instead of tp (DCPGroupColumnParallelLinear, vllm/model_executor/layers/linear.py:
	// 632-659), so each rank holds and computes dcp times that projection. Crediting the
	// skipped gather without the replicated GEMM would make the knob look free; charging
	// the GEMM needs to know which projection is the query's, and the graph identifies
	// GEMMs by shape, not purpose. It takes effect only with dcp > 1, pcp <= 1
	// (vllm/model_executor/models/deepseek_v2.py:1072-1076) and on a latent layer, so
	// everywhere else the request is inert and is accepted.
	if q := k.engine.DCPQReplicate; q != nil && *q && pl.DCP > 1 && pl.PCP <= 1 {
		for _, l := range k.plan.Layers {
			if l.AttnQHeads > 0 && latentAttention(l.AttnKind) {
				return fmt.Errorf(
					"dcp_q_replicate is requested with dcp %d on a latent-attention model; "+
						"the replicated query projection it costs is not priced by this "+
						"kernel, so the knob is refused rather than credited for the "+
						"collective it saves", pl.DCP)
			}
		}
	}

	// The release's own list of backend names. A rules value that states none leaves the
	// release's check undone, and that is disclosed rather than silent: the name is still
	// checked against what this kernel prices, but not against what the engine release
	// accepts.
	accepted := in.Rules.DCPCommBackendNames()
	if b := k.engine.DCPCommBackend; b != "" && accepted == nil {
		k.assume("dcp_comm_backend", b,
			"stated, and checked against the backends this kernel prices, but not against "+
				"the names the engine release accepts: the rules value carries no list")
	}
	dc, overrides, err := resolve.ResolveDecodeContext(
		k.pool, k.engine, in.Deployment, k.blockSize, k.blockSizeFinal, accepted)
	if err != nil {
		return fmt.Errorf("resolving decode-context parallelism: %w", err)
	}
	k.decodeContext = dc
	k.layout.Overrides = append(k.layout.Overrides, overrides...)
	if pl.DCP > 1 && k.engine.DCPCommBackend == "" {
		k.assume("dcp_comm_backend", dc.CommBackend,
			"the engine's stock default (ParallelConfig.set_dcp_defaults, "+
				"vllm/config/parallel.py:581-594 at v0.31.0), with the query projection not "+
				"replicated; a model's configuration hook runs first and may choose "+
				"otherwise -- GlmMoeDsaForCausalLM selects a2a with dcp_q_replicate "+
				"(vllm/model_executor/models/config.py:43-50) -- and the kernel cannot see "+
				"which model class serves this graph, so state both to price it exactly. On "+
				"that model, where FlashMLA-sparse serves the layout, the difference "+
				"decides whether it starts: that backend runs DCP only with ag_rs "+
				"(flashmla_sparse.py:416-422)")
	}
	return nil
}

// resolveMaxModelLen settles the context bound the engine runs with. blis-schemas v0.3.0
// states it in three ways (deployment.MaxModelLen): a count, auto, or nothing.
//
//   - A count is the bound.
//   - Unstated, vLLM v0.31.0 derives it from the model configuration, which the graph
//     carries as global.context_window (its default_context_window where that differs).
//   - auto fits the bound to the KV memory the engine finds at startup, capped at that
//     same derived length (vllm/config/model.py _get_and_verify_max_len, then the KV
//     cache sizing in vllm/v1/core/kv_cache_utils.py). The kernel cannot run the engine's
//     memory profile, so auto is taken at the cap, the most it can resolve to, and that
//     is recorded.
//
// Zero means neither the deployment nor the graph states one, and every consumer of the
// bound then says what it does without it.
func (k *Kernel) resolveMaxModelLen(g *model.Graph) {
	mml := k.engine.MaxModelLen
	if n, ok := mml.Stated(); ok {
		k.maxModelLen = n
		return
	}
	k.maxModelLen = g.Global.DefaultWindow()
	switch {
	case mml.Auto() && k.maxModelLen > 0:
		k.assume("max_model_len", strconv.Itoa(k.maxModelLen),
			"auto fits the bound to the KV memory the engine finds at startup, capped at the "+
				"model's derived window (vllm/config/model.py _get_and_verify_max_len); the "+
				"kernel cannot profile memory, so it takes the cap, which the engine lowers "+
				"where the cache cannot hold one sequence of it")
	case mml.Auto():
		k.assume("max_model_len", "auto",
			"auto, on a graph that states no context_window: the bound is unknown, and the "+
				"terms that read it say what they take instead")
	case k.maxModelLen > 0:
		k.assume("max_model_len", strconv.Itoa(k.maxModelLen),
			"unstated, so the engine derives it from the model configuration, which the "+
				"graph states as its context window")
	}
}

// refuseV4Layouts refuses the DeepSeek-V4-family layouts vLLM v0.31.0 refuses at startup,
// which no field validation catches and which would otherwise price as a deployment that
// runs.
//
//   - V4.1's packed attention serves only fp8 caches: a cache dtype that resolves to neither
//     packed record is a ValueError ("DeepseekV4 packed KV layouts only support fp8
//     kv-cache", vllm/models/deepseek_v41/attention.py:147-155).
//   - A mega-MoE backend needs expert parallelism ("DeepSeek V4 MegaMoE currently requires
//     expert parallel", vllm/models/deepseek_v41/nvidia/model.py:630-635; the V4 model is
//     the same).
//   - V4.1 with Engram layers refuses dual-batch overlap ("DeepSeek V4.1 Engram does not
//     support DBO or microbatching", vllm/config/vllm.py).
//   - V4.1 refuses MTP drafting: its drafter is DSpark (the rules pack's
//     deepseek-v41-refuses-mtp).
//
// refuseUnsplittableKVHeads refuses a tensor-parallel width that does not divide the KV heads
// of an attention whose projection cannot replicate a head across ranks
// (model.Node.KVHeadsNotReplicated): the engine asserts n_kv % tp == 0 when it builds the
// layer, as Muse Spark's fused [Q, K, V, O-gate] projection does ("KV-head padding is not
// supported", vllm/models/muse_spark/common/layers.py:368-371@0f6b2619). The drafter's layers
// are checked too: they are built at the same width.
func refuseUnsplittableKVHeads(g *model.Graph, tp int) error {
	if tp <= 1 {
		return nil
	}
	var err error
	g.EachNode(func(at string, n model.Node) {
		if err == nil && n.Op == model.OpAttention && n.KVHeadsNotReplicated &&
			n.NumKVHeads%tp != 0 {
			err = fmt.Errorf("tensor parallelism %d does not divide the %d KV heads of %s, whose "+
				"projection cannot replicate a head across ranks, so vLLM refuses the layer when "+
				"it builds it (n_kv %% tp == 0)", tp, n.NumKVHeads, at)
		}
	})
	return err
}

func refuseV4Layouts(in Inputs, cacheDType string) error {
	arch := in.Model.Global.Architecture
	if arch != rules.ArchDeepseekV4 && arch != rules.ArchDeepseekV41 {
		return nil
	}
	pool := in.Deployment.Pools[in.PoolIndex]
	family, _ := in.Chip.CapabilityMajor()
	if in.Rules.MegaMoEBackend(in.Engine.MoEBackend) {
		if !pool.Parallel.EnableExpertParallel {
			return fmt.Errorf("moe_backend %q on %s needs expert parallelism: vLLM v0.31.0 "+
				"refuses mega-MoE without it", in.Engine.MoEBackend, arch)
		}
		if family != 10 {
			return fmt.Errorf("moe_backend %q on %s needs an SM100 part: \"DeepGEMM MegaMoE "+
				"requires SM100 GPUs\" (vllm/models/deepseek_v4/nvidia/model.py:366-369)",
				in.Engine.MoEBackend, in.Chip.Name)
		}
	}
	// The family's sparse attention backends declare no implementation class -- their
	// get_impl_cls raises (vllm/models/deepseek_v41/sparse_mla.py:96-104, deepseek_v4/
	// sparse_mla.py:67-75) -- so PCP support reads as no and the engine's assertion stops it
	// (vllm/v1/worker/cp_utils.py:31-38). Nor does the family's attention implement
	// decode-context sharding: its window cache's manager asserts dcp_world_size == 1 on the
	// prefix-cache lookup (vllm/v1/core/single_type_kv_cache_manager.py:989), and nothing in
	// vllm/models/deepseek_v4* shards its compressed caches by DCP.
	if pool.Parallel.PCP > 1 {
		return fmt.Errorf("prefill-context parallelism (pcp %d) on %s: its sparse attention "+
			"backends do not support PCP, and vLLM v0.31.0 refuses it at startup",
			pool.Parallel.PCP, arch)
	}
	if pool.Parallel.DCP > 1 {
		return fmt.Errorf("decode-context parallelism (dcp %d) on %s: its attention "+
			"implements no DCP sharding, and its window cache asserts dcp_world_size == 1",
			pool.Parallel.DCP, arch)
	}
	if arch != rules.ArchDeepseekV41 {
		return nil
	}
	// The packed records are SM90's and SM100's; another family's attention class, which
	// no rules-pack resolution describes, is not judged here.
	if family == 9 && cacheDType == "nvfp4_ds_mla" {
		return fmt.Errorf("cache_dtype nvfp4_ds_mla on %s on SM90: \"nvfp4_ds_mla needs the "+
			"V4.1 KV records, which FlashMLA decodes only on SM100\" "+
			"(vllm/models/deepseek_v41/attention.py:513-517)", arch)
	}
	if (family == 9 || family == 10) && !strings.HasSuffix(cacheDType, "_ds_mla") {
		return fmt.Errorf("cache_dtype %q on %s: its packed attention serves only fp8 "+
			"caches, as fp8_ds_mla or nvfp4_ds_mla, and vLLM v0.31.0 refuses any other",
			in.Engine.CacheDType, arch)
	}
	if s := in.Engine.Speculative; s != nil && s.NumSpecTokens > 0 &&
		strings.Contains(strings.ToLower(s.Method), "mtp") {
		return fmt.Errorf("speculative method %q on %s: its drafter is DSpark, and vLLM "+
			"v0.31.0 refuses MTP for it", s.Method, arch)
	}
	if d := in.Engine.DBO; d != nil && d.Enabled && graphHasEngram(in.Model) {
		return fmt.Errorf("dual-batch overlap on %s with Engram layers: vLLM v0.31.0 "+
			"refuses it at startup", arch)
	}
	return nil
}

// graphHasEngram reports whether a graph runs an Engram lookup.
func graphHasEngram(g *model.Graph) bool {
	for _, lk := range g.LayerKinds {
		for _, n := range lk.Nodes {
			if n.Op == model.OpGather && n.ResidencyControl == model.ResidencyEngramCPUOffload {
				return true
			}
		}
	}
	return false
}

// weightBytesPerParam is the resident bytes of one parameter of a GEMM or GroupedGEMM
// node, scales included (blis-schemas Graph.WeightBytesPerParam), as the deployment serves
// it. A --quantization that serves the checkpoint in another format replaces the global
// dtype for every node that does not state its own, and with it the global scale layout,
// which belongs to the checkpoint's format; the served format's intrinsic scale, if it
// has one, is counted instead (NVFP4's one FP8 byte per 16, MXFP4's one per 32).
//
// An alias reads its target's weights at the width they are resident at, the same rule
// applied to the target.
func weightBytesPerParam(g *model.Graph, n model.Node, served model.DType) float64 {
	if n.Aliases != "" {
		if t, ok := aliasTarget(g, n.Aliases); ok && t.Op != model.OpGather &&
			t.Aliases == "" {
			return weightBytesPerParam(g, t, served)
		}
		return g.WeightBytesPerParam(n)
	}
	if n.WeightDType == "" && served != g.Global.WeightDType {
		return model.StoredBytes(served, nil)
	}
	return g.WeightBytesPerParam(n)
}

// weightFormat is the format a GEMM or GroupedGEMM node computes from: its own dtype and
// scale where it states them, the served one otherwise. The scale is what tells MXFP8
// (one E8M0 byte per 32 elements) from block-scaled FP8.
//
// An alias computes from the weights it reads, so it takes its target's format: DSpark's
// LM head reads the target's BF16 one, and a tied LM head reads an unquantized embedding
// table, which vLLM keeps at BF16 whatever the checkpoint (see computeFixedBytes).
func weightFormat(g *model.Graph, n model.Node, served model.DType) price.Format {
	if n.Aliases != "" {
		target, ok := aliasTarget(g, n.Aliases)
		switch {
		case ok && target.Op == model.OpGather:
			return price.Format{DType: model.DTypeBF16}
		case ok && target.Aliases == "":
			return weightFormat(g, target, served)
		}
	}
	d, s := n.WeightDType, n.WeightScale
	if d == "" {
		d = served
		if served == g.Global.WeightDType && s == nil {
			s = g.Global.WeightScale
		}
	}
	if s == nil {
		if is, ok := d.IntrinsicScale(); ok {
			s = &is
		}
	}
	f := price.Format{DType: d}
	if s != nil {
		f.ScaleElements = s.Elements
	}
	return f
}

// aliasTarget resolves "<layer kind id>.<role>", "head.<role>" or "embedding.<role>" to the
// node it names, with the resolver validation uses.
func aliasTarget(g *model.Graph, ref string) (model.Node, bool) { return g.AliasTarget(ref) }

// attentionKinds is every attention kind a layer of the target stack reads the engine's
// cache with. A speculator's layers are not the target's, and a kind declared but not
// stacked runs nowhere.
func attentionKinds(g *model.Graph) map[model.AttentionKind]bool {
	byID := map[string]model.LayerKind{}
	for _, lk := range g.LayerKinds {
		byID[lk.ID] = lk
	}
	kinds := map[model.AttentionKind]bool{}
	for _, id := range g.Stack.Expand() {
		for _, n := range byID[id].Nodes {
			if n.ReadsEngineKV() {
				kinds[n.AttentionKind] = true
			}
		}
	}
	return kinds
}

// dsMLAStateBytes is the packed page cell, in bytes per token per layer, of the cache
// layouts that state their own size.
var dsMLAStateBytes = map[string]float64{"fp8_ds_mla": 656, "nvfp4_ds_mla": 352}

// latentAttention reports whether an attention kind keeps a latent (MLA) cache, which is
// what the engine's MLA attention implementations serve.
func latentAttention(kind model.AttentionKind) bool {
	return kind == model.AttentionMLA || kind == model.AttentionSparseMLA
}

// clonePool copies a pool so the copy shares no memory with the original. The engine block is
// held unopened and immutable (deployment.EngineBlock: a copy of it is a copy of the node, and its
// Node returns another copy); the parallel block is flat; the labels are copied.
func clonePool(p deployment.Pool) deployment.Pool {
	p.Labels = maps.Clone(p.Labels)
	return p
}

// emitter adapts a layout to the plan builder's interface.
type emitter struct{ l resolve.Layout }

func (e emitter) Emits(c model.EmitCondition) bool      { return e.l.Emits(c) }
func (e emitter) Recognizes(c model.EmitCondition) bool { return resolve.Recognizes(c) }

// lift copies the coefficients the hot path needs into fields, so a step reads a struct
// rather than hashing strings. Each lookup that has no default is required: a rate the
// kernel cannot resolve is an error, not a zero.
func (k *Kernel) lift(c *resolve.Coefficients, g *model.Graph, cacheBytes float64) error {
	derate, err := c.Value("hbm_derate")
	if err != nil {
		return err
	}
	k.hbmBytesPerSecond = k.chip.MemoryBandwidthTBs * 1e12 * derate
	if err := mustPositive("HBM bandwidth", k.hbmBytesPerSecond); err != nil {
		return err
	}

	// Compute peak and the efficiency ramp are selected by the weight dtype: the three
	// formats in the source data reach materially different asymptotes, so a single ramp
	// would misprice two of them.
	served, err := k.liftFit(c, dtypeFitSuffix(k.servedDType, k.chip))
	if err != nil {
		return fmt.Errorf("served dtype %s: %w", k.servedDType, err)
	}
	k.computeFLOPsPerSecond, k.epsMax, k.mHalf = served.peak, served.epsMax, served.mHalf
	k.nHalf, k.kHalf, k.gemmShapeAware = served.nHalf, served.kHalf, served.shapeAware

	// Every GEMM is priced at the rate of the format it computes from, not the served one
	// alone. A model's GEMMs do not all run one kernel: DeepSeek-V4.1's mHC projections are
	// FP32 parameters run at TF32, its router and wo_a are BF16, and on SM90 its MXFP8
	// projections and MXFP4 experts run weight-only Marlin kernels on BF16 activations.
	// Pricing all of them at the served format's peak priced each at a rate its kernel
	// does not reach.
	k.fits = map[fitKey]gemmFit{}
	addFits := func(layers ...price.PlannedLayer) error {
		for _, l := range layers {
			for _, g := range l.DenseGEMMs {
				if err := k.addFit(c, g.Format, false); err != nil {
					return err
				}
			}
			if l.ExpertFLOPsPerTokenPerExpert > 0 {
				if err := k.addFit(c, l.RoutedFormat, true); err != nil {
					return err
				}
			}
			if l.SharedExpertFLOPsPerToken > 0 {
				if err := k.addFit(c, l.SharedFormat, false); err != nil {
					return err
				}
			}
		}
		return nil
	}
	all := append(append([]price.PlannedLayer{}, k.plan.Layers...), k.plan.Head,
		k.plan.Embedding)
	if d := k.plan.Draft; d != nil {
		all = append(append(append(all, d.Layers...), d.Context...), d.Embedding, d.Head,
			d.SequentialHead)
	}
	if err := addFits(all...); err != nil {
		return err
	}
	k.layerFits = make([][]gemmFit, len(k.plan.Layers))
	for i, l := range k.plan.Layers {
		for _, g := range l.DenseGEMMs {
			k.layerFits[i] = append(k.layerFits[i], k.fits[fitKey{g.Format, false}])
		}
	}

	k.nvlinkBytesPerSecond = k.chip.IntraNodeBwGBps * 1e9
	k.nicBytesPerSecond = k.fabric.EffectiveInterNodeBwGBps() * 1e9
	if k.nicBytesPerSecond <= 0 {
		// No fabric: a single-node deployment, where no collective leaves the node. The
		// rate is unused, and setting it to the on-node rate keeps a stray cross-node term
		// finite rather than infinite.
		k.nicBytesPerSecond = k.nvlinkBytesPerSecond
	}
	k.adoptSwitching()

	// Collective floors are per operation, dtype and rank count. All three axes
	// matter and the measurements say so: on H200 an all-reduce floor is 15.98 us
	// against an all-to-all's 7.39, and an all-reduce floor grows 6.0 / 9.4 / 16.0
	// with 2 / 4 / 8 ranks where an all-to-all's barely moves. A single floor for
	// every collective is wrong in both directions at once, so each op resolves its
	// own and they are looked up once here rather than per step.
	if err := k.liftCollectiveFloors(c); err != nil {
		return err
	}

	// The CPU-to-GPU link rate, per part. NVIDIA's descriptor states it, and it is
	// not a constant across platforms: a Grace-Blackwell part connects over
	// NVLink-C2C where an HGX part uses PCIe, so an offload tier that is a liability
	// on one host can be affordable on the other.
	// Recurrent families, one entry per kind the registry measures. A kind with no entries
	// is left out of the map rather than defaulted to zero cost, so the difference between
	// "measured as cheap" and "not measured" stays visible.
	//
	// Every coefficient read from here to the host terms is optional: the kernel prices
	// without it. Each one the deployment needs and the scenario's sets do not carry is
	// recorded as an assumption (optional), so Provenance and Evidence show what the
	// prediction lacks. A per-kind entry is needed only where the model has a layer of that
	// kind.
	usedAttn, usedRec := map[model.AttentionKind]bool{}, map[model.RecurrentKind]bool{}
	moe := false
	for _, l := range k.plan.Layers {
		if l.AttnQHeads > 0 {
			usedAttn[l.AttnKind] = true
		}
		if l.RecurrentKind != "" {
			usedRec[l.RecurrentKind] = true
		}
		moe = moe || l.ExpertWeightBytesPerExpert > 0
	}
	k.recurrent = map[model.RecurrentKind]floorRate{}
	for kind, suffix := range map[model.RecurrentKind]string{
		model.RecurrentMamba2: "mamba2",
		model.RecurrentKDA:    "kda",
		model.RecurrentGDN:    "gdn",
	} {
		without := "no measured decode form for the " + suffix + " mixer, so its decode " +
			"term is not charged"
		floor := k.optional(c, "recurrent_decode_floor_"+suffix, usedRec[kind], without)
		rate := k.optional(c, "recurrent_decode_rate_"+suffix, usedRec[kind], without)
		if floor <= 0 || rate <= 0 {
			continue
		}
		k.recurrent[kind] = floorRate{
			floor: time.Duration(floor * float64(time.Microsecond)),
			rate:  rate * 1e6, // tokens per microsecond to tokens per second
		}
	}

	// The attention primitive's measured floor and rate. Absent for a part with no
	// generation-attention sweep, in which case the KV read is charged at HBM bandwidth
	// with no floor, and the absence is recorded.
	k.attentionFloor = time.Duration(k.optional(c, "attention_decode_floor",
		len(usedAttn) > 0, "the per-layer attention decode floor is not charged") *
		float64(time.Microsecond))
	k.attentionRate = k.optional(c, "attention_decode_rate", len(usedAttn) > 0,
		"no measured attention decode rate, so the KV read is charged at HBM "+
			"bandwidth and no attention floor is charged") * 1e6

	// Per-attention-kind decode terms, where the registry carries them. A kind absent from the
	// registry falls back to the unsuffixed pair above, so a deployment with no per-kind fit
	// prices exactly as it did before this map existed.
	//
	// The kinds are different KERNELS, not one kernel at different settings: full attention
	// reads the whole context, a sliding window reads a bounded slice of it. NVIDIA's own
	// sweeps collect them separately for that reason, and fitted separately a windowed kernel
	// sustains 1.1x to 2.8x less effective bandwidth than a full-attention one on the same
	// silicon (0.26 against 0.58 of peak on H200). Pricing both with the full-attention rate
	// makes every windowed layer too cheap, and gpt-oss-120b is half windowed layers.
	//
	// Naming follows the recurrent terms, which already carry a per-kind suffix. That is why
	// no schema change is needed: coefficient.Scope has no kind axis and does not need one.
	// MLA joins the map for the same reason SWA did: its kernel reads a different number
	// of bytes per token and sustains a different fraction of peak, so pricing it with the
	// full-attention rate is wrong by construction rather than by a little. The WIDTH comes
	// from the graph -- blis-catalog declares an MLA node with `n_kv: 1` and d_h the latent
	// width (576 = kv_lora_rank + qk_rope_head_dim for the DeepSeek-V3 family and Kimi, 512
	// for DeepSeek-V4), so a layer's NumKVHeads*HeadDim is that width --
	// and KVBytesPerToken charges it ONCE per token, since a latent cache has no value
	// tensor. The RATE is the other half: an MLA decode sustains 0.61-0.80 of datasheet
	// bandwidth against full attention's 0.52-0.88.
	//
	// The FLOOR is not what distinguishes them, which an earlier version of this comment
	// had backwards: it claimed an MLA floor "several times larger (51.5-89.5us against
	// 9.5-19.5us) because the per-call setup reads a latent cache". The registry says
	// otherwise -- attention_decode_floor_mla is 9.5-14.5us against the part-wide
	// 9.5-19.0us -- because that entry IS the part's own measured attention floor reused
	// for the kind ("this is this part's own measured attention-kernel decode floor,
	// reused for the MLA kind", cost-model-attention.yaml). The larger module-derived
	// floors, 4.7x-6.2x those, were measured and REJECTED: they cover the whole MLA block
	// including down-projections this kernel prices separately as GEMM nodes, and charging
	// them took kimi-k2.5's TPOT error from 6.38% to 14.57%.
	//
	// Nine models in the pinned catalog declare an mla or sparse_mla layer, and two of them
	// (deepseek-v4-pro, kimi-k3) appear in both the FPM dataset and the InferenceX corpus,
	// so this is on the scored path rather than hypothetical.
	//
	// sparse_mla is now mapped too, and its pair is asymmetric: the registry ships a
	// RATE and deliberately no floor.
	//
	// The reason is recorded in blis-registry scripts/fit_attention_sparse_mla.py.
	// Searching a sparse floor freely lands at 12.6-14.4us on every part -- a band that
	// does not track the part-wide floors it would replace -- and pinning it to each
	// part's own measured attention floor costs under 8% of fit. Adding the parameter
	// would re-open the failure that `attention_decode_floor_mla` shipped with: a floor
	// fitted from a MODULE table, charging projections the catalog already prices as
	// separate GEMM nodes, which cost 2.45 points of end-to-end TPOT before
	// correct_mla_floor.py undid it.
	//
	// So a kind may supply a rate alone and inherit the part-wide floor. A floor without its
	// rate is not used -- the kind is priced with the part-wide pair -- and the missing rate
	// is recorded, so the floor's listing in Provenance is not mistaken for its use.
	k.attentionByKind = map[model.AttentionKind]floorRate{}
	for kind, suffix := range map[model.AttentionKind]string{
		model.AttentionSWA:       "swa",
		model.AttentionMLA:       "mla",
		model.AttentionSparseMLA: "sparse_mla",
	} {
		floor := c.ValueOr("attention_decode_floor_"+suffix, 0)
		rate := k.optional(c, "attention_decode_rate_"+suffix, usedAttn[kind],
			"no per-kind decode rate for "+suffix+" attention, so it is priced with the "+
				"part-wide attention pair and any per-kind floor is not used")
		if rate <= 0 {
			continue
		}
		if floor <= 0 {
			// Inherit the part-wide measured floor, which is what the sparse fit was
			// conditioned on. Lifted after k.attentionFloor is set, so this is the same
			// number the fitter pinned to.
			k.attentionByKind[kind] = floorRate{
				floor: k.attentionFloor, rate: rate * 1e6,
			}
			continue
		}
		k.attentionByKind[kind] = floorRate{
			floor: time.Duration(floor * float64(time.Microsecond)),
			rate:  rate * 1e6, // bytes per microsecond to bytes per second
		}
	}
	k.attentionPrefillFloor = time.Duration(k.optional(c, "attention_prefill_floor",
		len(usedAttn) > 0, "the per-layer attention prefill floor is not charged") *
		float64(time.Microsecond))
	k.attentionPrefillScale = k.optional(c, "attention_prefill_work_scale",
		len(usedAttn) > 0, "no measured attention prefill scale, so prefill attention "+
			"is charged on the unmodified GEMM efficiency ramp with no floor")
	if err := k.applyAttentionLane(c, usedAttn); err != nil {
		return err
	}

	k.hostBytesPerSecond = k.optional(c, "host_link_bandwidth", true,
		"no host-link rate, so an offload transfer is not capped at it") * 1e6
	k.moeImbalance = 1
	if v := k.optional(c, "moe_routing_imbalance_median", moe,
		"no measured routing imbalance, so experts are priced as evenly loaded"); v > 0 {
		k.moeImbalance = v
	}

	hostWithout := func(what string) string {
		return "no measured " + what + ", so that host cost is charged at zero"
	}
	k.admissionPerToken = time.Duration(k.optional(c, "host_admission_per_token", true,
		hostWithout("admission cost per token")) * float64(time.Microsecond))
	// The frontend tokenizes every prompt before the engine admits it, and the Rust one
	// (engine.frontend: rust, vLLM's VLLM_USE_RUST_FRONTEND=1; vllm/envs.py:169 and
	// vllm/entrypoints/cli/serve.py:64 at v0.31.0) does it through fastokens at its own
	// rate. A deployment on it takes that rate where the registry states one, and keeps
	// the Python path's, recorded as an assumption, where it does not.
	if k.engine.RustFrontend() {
		const rust = "host_admission_per_token_rust_frontend"
		if c.Has(rust) {
			k.admissionPerToken = time.Duration(c.ValueOr(rust, 0) * float64(time.Microsecond))
		} else {
			k.assume(rust, "absent", "not in the scenario's coefficient sets: the Rust "+
				"frontend's admission is charged at the Python frontend's per-token rate")
		}
	}
	k.admissionPerRequest = time.Duration(k.optional(c, "host_admission_per_request", true,
		hostWithout("admission cost per request")) * float64(time.Microsecond))
	k.outputTokenCost = time.Duration(k.optional(c, "host_output_token", true,
		hostWithout("per-output-token cost")) * float64(time.Microsecond))
	k.completionCost = time.Duration(k.optional(c, "host_completion", true,
		hostWithout("completion cost")) * float64(time.Microsecond))
	// One eager kernel launch. The registry's historical name is per layer, and it defines
	// the value as "multiplied by a layer's kernel count at the call site" (blis-registry
	// cost-model-host-overheads.yaml, host_launch_eager_per_layer), so it is charged per
	// kernel: an earlier form charged it once per layer, which made eager about four
	// percent dearer than a full graph and cheaper than piecewise replay.
	// host_launch_eager_per_kernel names the same per-launch cost by what it is, and is
	// taken where a set carries it -- unless only the per-layer name is scoped to this
	// model, since a model's own constant is the narrower claim (see resolve.Load). Either
	// may be scoped to a model, which matters because the cost with no graph is Python
	// dispatch per op on a model vLLM runs uncompiled (DeepseekV41ForCausalLM "cannot
	// torch.compile", vllm/config/vllm.py:120) rather than a CUDA launch, and so is a
	// property of the model's code.
	perKernel, hasPerKernel := c.Entry("host_launch_eager_per_kernel")
	perLayer, hasPerLayer := c.Entry("host_launch_eager_per_layer")
	if hasPerKernel && (len(perKernel.Entry.Scope.Model) > 0 || !hasPerLayer ||
		len(perLayer.Entry.Scope.Model) == 0) {
		k.launchEager = time.Duration(c.ValueOr("host_launch_eager_per_kernel", 0) *
			float64(time.Microsecond))
	} else {
		k.launchEager = time.Duration(k.optional(c, "host_launch_eager_per_layer", true,
			hostWithout("eager launch cost")) * float64(time.Microsecond))
	}
	// The scheduler's per-request CPU cost (schedulerHost). Not needed: no registry carries
	// a fitted value yet, and zero is how every deployment was priced before it existed.
	k.schedPerRequest = time.Duration(k.optional(c, "host_step_per_request", false, "") *
		float64(time.Microsecond))
	// Per-kernel dispatch, which dominates a single-request decode step. Charged at zero
	// when the registry carries none, like the other host terms here -- which removes the
	// term two published runs say is most of a small step's cost, so a registry that
	// carries the host set is what a scored deployment needs, and its absence is recorded.
	k.launchPerKernel = time.Duration(k.optional(c, "host_launch_per_kernel", true,
		hostWithout("per-kernel launch cost")) * float64(time.Microsecond))
	k.replayPerStep = time.Duration(k.optional(c, "host_replay_graph_per_step", true,
		hostWithout("graph replay cost")) * float64(time.Microsecond))
	// How much of a step a captured graph covers, which sets the launch count rather
	// than only whether capture happens. The mode decides it PER BATCH: vLLM v0.31.0
	// documents each (vllm/config/compilation.py:604-640) and dispatches each step by
	// whether it is a uniform decode batch (vllm/v1/cudagraph_dispatcher.py:233-310):
	//
	//	PIECEWISE           piecewise for every step
	//	FULL                one full graph for every step
	//	FULL_AND_PIECEWISE  full for a uniform decode batch, piecewise otherwise
	//	FULL_DECODE_ONLY    full for a uniform decode batch, NO graph otherwise
	//	NONE                no graph
	//
	// FULL_AND_PIECEWISE is v0.31.0's default ("(v1 default)", compilation.py:614), so
	// that is what an unstated mode resolves to, and the kernel says so in Provenance:
	// the engine downgrades it where an attention backend cannot capture a full graph,
	// which the kernel cannot see.
	mode := strings.ToUpper(k.engine.CUDAGraphMode)
	if mode == "" {
		mode = "FULL_AND_PIECEWISE"
		k.assume("cudagraph_mode", mode,
			"vLLM v0.31.0's default (vllm/config/compilation.py:614); the engine "+
				"downgrades it where an attention backend does not support full graphs, "+
				"which the kernel cannot see")
	}
	// Two MoE dispatch backends cannot be captured, and the engine turns graphs off for
	// them whatever was stated: DeepEP high-throughput with dp above one
	// (vllm/config/compilation.py:1237-1253), and MoonEP wherever the all-to-all runs --
	// dp above one, sequence-parallel MoE, or expert parallelism with pcp above one
	// (vllm/config/vllm.py:2098-2106; use_all2all, vllm/config/parallel.py:738-743).
	backend := k.rules.EffectiveAll2AllBackend(k.engine.All2AllBackend)
	pl := k.pool.Parallel
	useAll2All := k.layout.DP > 1 || k.layout.SequenceParallelMoE ||
		(pl.EnableExpertParallel && k.layout.PCP > 1)
	if mode != "NONE" && ((backend == "deepep_high_throughput" && k.layout.DP > 1) ||
		(backend == "moonep" && useAll2All)) {
		k.layout.Overrides = append(k.layout.Overrides, resolve.Override{
			Field: "cudagraph_mode", Requested: mode, Resolved: "NONE",
			Reason: "the " + backend + " all-to-all cannot be captured, so vLLM v0.31.0 " +
				"runs this layout with no CUDA graphs",
		})
		mode = "NONE"
	}
	switch mode {
	case "PIECEWISE":
		k.graphDecode, k.graphOther = graphModePiecewise, graphModePiecewise
	case "FULL":
		k.graphDecode, k.graphOther = graphModeFull, graphModeFull
	case "FULL_AND_PIECEWISE":
		k.graphDecode, k.graphOther = graphModeFull, graphModePiecewise
	case "FULL_DECODE_ONLY":
		k.graphDecode, k.graphOther = graphModeFull, graphModeEager
	case "NONE":
		k.graphDecode, k.graphOther = graphModeEager, graphModeEager
	default:
		return fmt.Errorf(
			"cudagraph_mode %q is not one this kernel prices; add it with its launch "+
				"count rather than letting it fall through",
			k.engine.CUDAGraphMode)
	}
	// The capture's memory is keyed by vLLM's own mode name, lowercased: the modes hold
	// different graph sets, so FULL_AND_PIECEWISE and FULL do not share a figure even
	// where they share a decode launch count.
	k.graphModeName = strings.ToLower(mode)
	k.uniformDecodeWidth = 1
	if s := k.engine.Speculative; s != nil && s.NumSpecTokens > 0 {
		// A uniform decode batch verifies 1 + num_spec_tokens tokens per request
		// (uniform_decode_query_len, vllm/v1/worker/gpu_model_runner.py:883).
		k.uniformDecodeWidth = 1 + s.NumSpecTokens
	}

	// The largest batch a captured graph covers. vLLM v0.31.0 captures token counts up to
	// max_cudagraph_capture_size and runs anything larger with no graph at all, whatever the
	// mode (vllm/v1/worker/gpu/cudagraph_utils.py:499-529; V1's
	// vllm/v1/cudagraph_dispatcher.py:270-279). A deployment that states the maximum or the
	// sizes is resolved as the engine resolves them, through the rules pack
	// (rules.Pack.CUDAGraphCaptureSizes, blis-schemas#72; VllmConfig._set_cudagraph_sizes,
	// vllm/config/vllm.py:2388-2660): capped at max_num_batched_tokens, and expanded on the
	// 1, 2, 4, step-8, step-16 ladder, so the ceiling is the largest size that runs.
	// Unstated, it is min(max_num_seqs x decode_query_len x 2, 512), or 1024 on a
	// data-center Blackwell part (:2442-2460). With speculation the uniform decode sizes are
	// appended only up to that same ceiling (`n * query_len <= max_cudagraph_capture_size`,
	// :2531-2538), so a decode step has no higher one. The default is then capped at
	// max_num_batched_tokens (:2541-2542), which no step exceeds, so that cap changes no
	// verdict here.
	//
	// The catalog records no compute capability, so a data-center Blackwell part is
	// recognised by native NVFP4 support, which among the catalog's parts is exactly the
	// SM100 family (b200, b300, gb200-nvl72, gb300).
	seqs := k.engine.MaxNumSeqs
	if seqs <= 0 {
		seqs = defaultMaxNumSeqs(k.chip)
		k.assume("max_num_seqs", strconv.Itoa(seqs),
			"vLLM v0.31.0's OpenAI-API-server default for this part's memory "+
				"(EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:2858-2887); it sets "+
				"the cudagraph capture ceiling and the MLA context-gather chunk")
	}
	k.maxNumSeqs = seqs
	if k.graphDecode != graphModeEager || k.graphOther != graphModeEager {
		e := graphCaptureRequest{Mode: mode, MaxCaptureSize: k.engine.MaxCUDAGraphCaptureSize,
			CaptureSizes: k.engine.CUDAGraphCaptureSizes, MaxNumBatchedTokens: k.engine.MaxNumBatchedTokens}
		if e.MaxNumBatchedTokens <= 0 {
			e.MaxNumBatchedTokens = k.defaultBatchedTokens()
		}
		capture, stated, err := k.rules.CUDAGraphCaptureSizes(e)
		if err != nil {
			return fmt.Errorf("resolving the CUDA graph capture sizes: %w", err)
		}
		if stated {
			k.captureTokens = capture.Max
			k.captureSizes = capture.Sizes
			if m := k.engine.MaxCUDAGraphCaptureSize; m != nil && *m != capture.Max {
				k.layout.Overrides = append(k.layout.Overrides, resolve.Override{
					Field: "max_cudagraph_capture_size", Requested: strconv.Itoa(*m),
					Resolved: strconv.Itoa(capture.Max),
					Reason: "vLLM v0.31.0 caps the capture at max_num_batched_tokens and " +
						"captures only the sizes on its ladder (1, 2, 4, every 8 below 256, " +
						"every 16 from it), so the largest graph that runs is this",
				})
			}
			if len(capture.Dropped) > 0 {
				k.layout.Overrides = append(k.layout.Overrides, resolve.Override{
					Field: "cudagraph_capture_sizes", Requested: fmt.Sprint(
						k.engine.CUDAGraphCaptureSizes), Resolved: fmt.Sprint(capture.Sizes),
					Reason: "vLLM v0.31.0 drops the sizes above max_num_batched_tokens",
				})
			}
		} else {
			platform := 512
			if k.chip.NVFP4Peak > 0 {
				platform = 1024
			}
			k.captureTokens = min(seqs*k.uniformDecodeWidth*2, platform)
			defaulted := e
			defaulted.MaxCaptureSize = &k.captureTokens
			if d, _, err := k.rules.CUDAGraphCaptureSizes(defaulted); err == nil {
				k.captureSizes = d.Sizes
			}
			k.assume("max_cudagraph_capture_size", strconv.Itoa(k.captureTokens),
				"vLLM v0.31.0's default, min(max_num_seqs x decode_query_len x 2, "+
					strconv.Itoa(platform)+") (VllmConfig._set_cudagraph_sizes, "+
					"vllm/config/vllm.py:2442-2460), as the deployment states neither "+
					"max_cudagraph_capture_size nor cudagraph_capture_sizes; a step above it "+
					"runs with no graph")
		}
	}

	// The rows one chunk of an MLA prefill's context gather covers:
	// min(max(8 x max_model_len, 4 x max_num_seqs x block_size), 65536)
	// (determine_chunked_prefill_workspace_size, vllm/model_executor/layers/attention/
	// mla_attention.py:2297-2320 at v0.31.0). Where neither the deployment nor the graph
	// states a context length (resolveMaxModelLen), it is taken at the 65,536 cap: exact for
	// any window of 8,192 tokens or more, which every latent model in the catalog has.
	chunk := 65536
	if mml := k.maxModelLen; mml > 0 {
		chunk = min(max(8*mml, 4*seqs*k.blockSize), 65536)
	} else if k.layout.DCP > 1 {
		for _, l := range k.plan.Layers {
			if latentAttention(l.AttnKind) {
				k.assume("mla_context_chunk", strconv.Itoa(chunk),
					"neither the deployment nor the graph states a context length, so the MLA "+
						"context-gather chunk is taken at "+
						"its 65,536-row cap, exact for a model window of 8,192 tokens or more")
				break
			}
		}
	}
	// Then rounded up to a whole number of blocks -- of lcm(block, dcp x interleave) blocks
	// under DCP, so every rank's share stays interleave-aligned
	// (align_mla_chunked_context_workspace_size, mla_attention.py:1993-2005, applied at
	// :2319).
	align := max(k.blockSize, 1)
	if k.layout.DCP > 1 {
		align = lcm(align, k.layout.DCP*max(k.decodeContext.Interleave, 1))
	}
	chunk = (max(chunk, align) + align - 1) / align * align
	k.mlaContextChunk = max(chunk, 1)

	// Expert geometry, derived once.
	// The routed experts' count: a shared expert stated as its own node is not one of them.
	experts := 0
	for _, l := range k.plan.Layers {
		if l.TopK > 0 {
			experts = max(experts, l.Experts)
		}
	}
	// A stated dispatch backend that cannot take effect: with expert parallelism on and
	// none of dp > 1, pcp > 1 or a sequence-parallel MoE input, use_all2all_kernels is false
	// and no backend's kernels are built (vllm/model_executor/layers/fused_moe/
	// config.py:1024-1027, all2all_utils.py:188-214), so every backend prices the same.
	if b := k.engine.All2AllBackend; b != "" && experts > 0 &&
		k.pool.Parallel.EnableExpertParallel && !k.layout.MoEDispatches() {
		k.layout.Overrides = append(k.layout.Overrides, resolve.Override{
			Field: "all2all_backend", Requested: b, Resolved: "none",
			Reason: "expert parallelism with dp 1, pcp 1 and no sequence-parallel MoE input " +
				"runs no dispatch in vLLM v0.31.0: the experts run on the replicated tokens " +
				"and the tensor-parallel all-reduce combines them",
		})
	}
	// How an expert's weights are divided, which depends on whether expert parallelism
	// is on. vLLM gives a rank whole experts when it is (ep_size the group width, tp_size 1)
	// and a tensor slice of every expert when it is not (ep_size 1)
	// (FusedMoEParallelConfig.make, vllm/model_executor/layers/fused_moe/config.py:1186-1214
	// at v0.31.0). So exactly one of the two axes divides an expert, never both.
	//
	// The slice is over every rank the MoE spans, not the tensor-parallel group alone: with
	// expert parallelism off vLLM flattens tp across dp and pcp, "so we shard across all
	// devices" (flatten_tp_across_dp_and_pcp, config.py:1090-1098). An earlier form divided
	// by tp, which priced a dp > 1 deployment's experts as dp independent replicas, each rank
	// holding dp times its true share.
	//
	// COVERAGE LIMIT: the flattening runs over pcp as well, and the shard here does not. With
	// pcp > 1 vLLM gives a rank a 1/(dp x pcp x tp) slice of each expert, and this charges
	// 1/(dp x tp). The token gather over those ranks is priced where it runs -- with dp > 1
	// the dispatch spans dp x pcp x tp ranks (resolve.Layout.MoEGroupWidth); with dp = 1 there
	// is no dispatch (all2all_utils.py:202-214) -- but the reduction a tensor-parallel MoE
	// runs over the wider slice is not, so dividing by pcp alone would make a PCP step's
	// experts cheaper than the engine runs them.
	k.localExpertShare = 1
	k.expertTensorShards = 1
	if k.layout.ExpertWidth <= 1 {
		// No expert parallelism: every rank holds every expert, tensor-sharded across
		// the dp x tp ranks.
		k.expertTensorShards = k.tp * float64(max(k.layout.DP, 1))
	}
	if experts > 0 {
		redundant := 0
		if k.engine.EPLB != nil && k.engine.EPLB.Enabled {
			redundant = k.engine.EPLB.NumRedundantExperts
		}
		// With expert parallelism off the width is 1, so every rank holds every expert
		// — which is right: the division is by tensor shard, applied at pricing time.
		base, heaviest, _, _ := price.ExpertsPerRank(experts+redundant, k.layout.ExpertWidth)
		if base == 0 {
			return fmt.Errorf(
				"expert-parallel width %d exceeds the model's %d physical experts, so some "+
					"ranks would hold none", k.layout.ExpertWidth, experts+redundant)
		}
		// The HEAVIEST rank is the one priced. An uneven split gives the first
		// experts % width ranks one expert more (vllm/model_executor/layers/fused_moe/
		// expert_map_manager.py:67-69 at v0.31.0), and that rank bounds both answers: its
		// memory fills first, and the dispatch and combine wait for its experts to finish.
		// An earlier form priced the lighter rank, understating minimax-m2.5 at ep72 --
		// 256 experts, 40 ranks holding 4 -- by a quarter of its expert weights.
		k.expertsPerRank = float64(heaviest)
		// The share of the model's experts that rank holds, from its count rather than as
		// 1/width.
		k.localExpertShare = float64(heaviest) / float64(experts+redundant)
		k.totalExperts = experts + redundant
		k.expertGeoms = map[int]expertGeom{experts: {
			total: k.totalExperts, perRank: k.expertsPerRank, share: k.localExpertShare}}
	}

	// KV, PER LAYER KIND of the target stack. Both the cache dtype and the tensor-parallel
	// width divide it, and the head count floors at one because a head is never split
	// across ranks.
	//
	// A layer whose attention states its caches (blis-schemas v0.3.0) is sized from them
	// alone: which caches it owns, each one's record under the cache dtype that runs, how
	// many tokens an entry covers, and whether a window or a fixed count bounds it
	// (cacheTerms). Every other KV-holding layer keeps the inference it always had, one
	// full-length cache of its primary attention's read width -- taken from ITS OWN
	// primary, not from one geometry standing for the model. An earlier form took the
	// geometry, and the attention kind, from the last-declared layer kind, so DeepSeek-
	// V4.1-Flash was sized and gated by a DSpark draft layer outside its target stack.
	//
	// The packed DeepSeek sparse-MLA layouts state their own page cell: fp8_ds_mla packs the
	// latent with its scales into 656 bytes a token and nvfp4_ds_mla into 352, against 576
	// elements by the per-element formula (state_content_bytes,
	// vllm/model_executor/layers/attention/mla_attention.py:1361-1363 at v0.31.0). On a DSA
	// model the layout is the one its backend allocates, which can differ from the stated
	// dtype, and a layout no backend serves is refused (sparseMLABackend). Elsewhere it is
	// taken where the deployment states one.
	layout := k.engine.CacheDType
	if isDSA(g) {
		heads := 0
		for _, l := range k.plan.Layers {
			if l.AttnKind == model.AttentionSparseMLA && l.AttnQHeads > 0 {
				heads = max(heads, l.AttnQHeads)
			}
		}
		preHopper, err := k.preHopper()
		if err != nil {
			return err
		}
		backend, dsaLayout, err := sparseMLABackend(sparseMLABackendRequest{
			cache:        k.engine.CacheDType,
			headsPerRank: heads / max(k.layout.TP, 1),
			sm100:        k.chip.NVFP4Peak > 0,
			preHopper:    preHopper,
			tp:           k.layout.TP,
			dcp:          k.layout.DCP,
			pcp:          k.layout.PCP,
			dcpComm:      k.decodeContext.CommBackend,
		})
		if err != nil {
			return err
		}
		if backend == backendFlashInferSM90 {
			k.assume("sparse_mla_backend", backend,
				"chosen as vLLM v0.31.0 would choose it, which needs FlashInfer 0.6.18 or "+
					"later installed (flashinfer_mla_sparse_sm90.py:194-199)")
		}
		if stated := k.engine.CacheDType; dsaLayout != stated && stated != "" {
			k.layout.Overrides = append(k.layout.Overrides, resolve.Override{
				Field: "cache_dtype", Requested: stated, Resolved: dsaLayout,
				Reason: backend + " serves this cache as " + dsaLayout +
					" (_canonicalize_sparse_mla_kv_cache_dtype, " +
					"vllm/model_executor/layers/attention/mla_attention.py:358-375)",
			})
		}
		layout = dsaLayout
	}
	inferred := func(l price.PlannedLayer) float64 {
		latent := latentAttention(l.AttnKind)
		if b, ok := dsMLAStateBytes[layout]; ok && latent {
			return b
		}
		return price.KVBytesPerToken(l.AttnKVHeads, k.layout.TP, l.AttnHeadDim, 1, cacheBytes,
			latent)
	}
	k.layerKV = make([]layerKV, len(k.plan.Layers))
	layers := 0
	for i, l := range k.plan.Layers {
		if l.AttnQHeads <= 0 {
			continue
		}
		per := inferred(l)
		k.layerKV[i].inferred = per
		k.caches = append(k.caches, k.cacheTerms(layerCaches(l), l.Count)...)
		if len(l.AttnCaches) > 0 {
			k.layerKV[i].stated = true
			continue
		}
		k.layerKV[i].perToken = per
		if ringed(l) {
			// A sliding-window layer's cache is a ring of its window, bounded per sequence
			// (ringBytes), not one that grows with context (SlidingWindowSpec,
			// vllm/v1/kv_cache_interface.py:840-907@v0.32.0).
			k.layerKV[i].window = l.AttnWindow
			continue
		}
		k.kvBytesPerToken += per * float64(l.Count)
		layers += l.Count
	}
	// The layer count kvBytesPerToken is spread over: only the layers whose cache it infers.
	// On a hybrid stack it is not the total layer count, and the two differ ninefold on
	// Nemotron-3-Ultra (12 KV layers of 108).
	k.kvLayers = layers
	// Whether EVERY inferred KV-holding layer is a kind decode-context parallelism shards. A
	// whole-model verdict, because kvBytesPerToken is a whole-model figure: see
	// SequenceVariableBytes for what that costs and why it costs nothing real.
	k.dcpShardsAllKVLayers = true
	for i, l := range k.plan.Layers {
		if l.AttnQHeads > 0 && !k.layerKV[i].stated && !dcpShardsKV(l.AttnKind) {
			k.dcpShardsAllKVLayers = false
			break
		}
	}
	if err := k.liftDraft(c, inferred); err != nil {
		return err
	}
	if err := k.checkCaches(); err != nil {
		return err
	}
	if err := k.liftStepTerms(c, g); err != nil {
		return err
	}
	k.liftDeepEP(c)
	return k.liftMemory(c, g)
}

// liftMemory resolves the per-rank occupancy outside the KV budget, every magnitude from
// the registry. computeFixedBytes composes them.
//
// THE COMPOSITION IS NVIDIA'S, as blis-registry's cost-model-memory set states it
// (blis-registry v0.1.1, methodology section 11): per rank,
//
//	weights + activation + nccl_communicator_bytes_<N>rank + engine_workspace_bytes
//	        + cudagraph_capture_bytes_<mode>
//
// where every term ADDS -- the communicator figure is the whole collective-buffer
// reservation, the workspace is a separate allowance beside it, and neither contains the
// graph capture -- and
//
//	activation = max(activation_buffer_count_<dense|moe>_<N>rank
//	                     * max_num_batched_tokens * h * 2,
//	                 activation_scratch_floor_bytes)
//
// with N the tensor-parallel width snapped to a declared width (snapDeclared). Each
// magnitude is REQUIRED: the registry carries all of them for every catalog part, so an
// absent one means the scenario did not list cost-model-memory, and charging zero for it
// would overstate how many sequences fit by up to several gigabytes per rank.
//
// They replace three constants this kernel carried -- 512 MiB of capture, 392 MiB of
// communicator, four buffers of activation -- which no Provenance entry could see, and the
// engine workspace, which was loaded and never read.
func (k *Kernel) liftMemory(c *resolve.Coefficients, g *model.Graph) error {
	need := func(name string) (float64, error) {
		v, err := c.Value(name)
		if err != nil {
			return 0, fmt.Errorf("memory occupancy needs %s; list cost-model-memory and "+
				"cost-model-primitives in the scenario's coefficients: %w", name, err)
		}
		return v, nil
	}
	// A communicator exists wherever some collective group is wider than one rank. The
	// figure is keyed on the tensor-parallel width, as NVIDIA's descriptor keys it, and
	// snapped to a declared width; a layout with tp=1 but another group (expert, prefill-
	// or decode-context) still holds a communicator, and is charged the narrowest declared
	// one rather than none.
	if k.layout.TP > 1 || k.layout.MoEGroup() > 1 || k.layout.PCP > 1 || k.layout.DCP > 1 {
		comm, err := need(fmt.Sprintf("nccl_communicator_bytes_%drank",
			snapDeclared(c, "nccl_communicator_bytes_%drank", k.layout.TP)))
		if err != nil {
			return err
		}
		k.commBytes = int64(comm)
	}
	workspace, err := need("engine_workspace_bytes")
	if err != nil {
		return err
	}
	k.workspaceBytes = int64(workspace)

	capture, err := need("cudagraph_capture_bytes_" + k.graphModeName)
	if err != nil {
		return err
	}
	k.captureBytes = k.captureMemory(c, capture)
	k.draftCaptureTokens = k.uniformCaptureTokens(k.numSpecTokens())

	// The activation multiple is keyed by whether the model routes tokens to experts,
	// which is the only family distinction the kernel can draw without a model name.
	family := "dense"
	if k.totalExperts > 0 {
		family = "moe"
	}
	format := "activation_buffer_count_" + family + "_%drank"
	if k.activationBuffers, err = need(fmt.Sprintf(format,
		snapDeclared(c, format, min(k.layout.TP, 8)))); err != nil {
		return err
	}
	if k.activationFloor, err = need("activation_scratch_floor_bytes"); err != nil {
		return err
	}
	k.activationWidth = activationWidth(k.plan, g)

	k.batchedTokens = k.engine.MaxNumBatchedTokens
	if k.batchedTokens <= 0 {
		k.batchedTokens = k.defaultBatchedTokens()
		why := "vLLM v0.31.0's OpenAI-API-server default for this part's memory " +
			"(EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:2858-2887); " +
			"the offline LLM class defaults higher on every part below 160 GiB"
		if k.batchedDPMoE() {
			why = "vLLM v0.31.0's default under a batched data-parallel MoE " +
				"(DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP, vllm/config/scheduler.py:43; " +
				"vllm/engine/arg_utils.py:3046-3051), which also sizes the DeepEP " +
				"low-latency buffer"
		}
		k.assume("max_num_batched_tokens", strconv.Itoa(k.batchedTokens), why)
	}
	return nil
}

// captureMemory is the CUDA-graph memory vLLM v0.31.0 takes from the KV budget: the
// estimate determine_available_memory subtracts (vllm/v1/worker/gpu_worker.py:628-676),
// which profile_cudagraph_memory measures by capturing every PIECEWISE and drafter graph and
// the two largest FULL graphs, extrapolating the rest (vllm/v1/worker/gpu/
// cudagraph_utils.py:854-968). Each family's cost grows with its largest captured shape: the
// mixed (PIECEWISE) family's is the capture ceiling, the uniform-decode FULL family's is
// bounded by max_num_seqs as well (decodeCaptureTokens), and the drafter's FULL graphs follow
// the decode family's request counts (dflash/speculator.py:126-150).
//
// The form is
//
//	cudagraph_capture_floor_bytes_<mode>
//	    + cudagraph_capture_bytes_per_mixed_token_<mode>  x mixed ceiling
//	    + cudagraph_capture_bytes_per_decode_token_<mode> x decode ceiling
//
// with a family's term present only where the mode captures it. It is all or none: with
// any needed entry absent the flat cudagraph_capture_bytes_<mode> stands, as before, and a
// partial set is recorded as a fallback.
func (k *Kernel) captureMemory(c *resolve.Coefficients, flat float64) int64 {
	if k.captureTokens == 0 {
		return int64(flat)
	}
	mode := k.graphModeName
	floorName := "cudagraph_capture_floor_bytes_" + mode
	terms := map[string]float64{}
	if k.graphOther != graphModeEager {
		terms["cudagraph_capture_bytes_per_mixed_token_"+mode] = float64(k.captureTokens)
	}
	if k.graphDecode == graphModeFull && k.graphOther != graphModeFull {
		terms["cudagraph_capture_bytes_per_decode_token_"+mode] =
			float64(k.decodeCaptureTokens())
	}
	names := []string{floorName}
	for n := range terms {
		names = append(names, n)
	}
	sort.Strings(names)
	var have, missing []string
	for _, n := range names {
		if c.Has(n) {
			have = append(have, n)
		} else {
			missing = append(missing, n)
		}
	}
	if len(missing) > 0 {
		if len(have) > 0 {
			k.assume(floorName, "fallback", strings.Join(missing, ", ")+" absent beside "+
				strings.Join(have, ", ")+", so the capture is charged the flat "+
				"cudagraph_capture_bytes_"+mode)
		}
		return int64(flat)
	}
	bytes := c.ValueOr(floorName, 0)
	for _, n := range names {
		if n != floorName {
			bytes += c.ValueOr(n, 0) * terms[n]
		}
	}
	return int64(bytes)
}

// decodeCaptureTokens is the largest uniform-decode FULL graph vLLM v0.31.0 captures: each
// capture size rounded up to a multiple of the decode query length, kept where it is within
// max_num_seqs requests and the capture ceiling (_init_candidates,
// vllm/v1/worker/gpu/cudagraph_utils.py:313-327).
func (k *Kernel) decodeCaptureTokens() int { return k.uniformCaptureTokens(k.uniformDecodeWidth) }

// uniformCaptureTokens is decodeCaptureTokens at a uniform query width of q tokens a request.
func (k *Kernel) uniformCaptureTokens(q int) int {
	q = max(q, 1)
	best := 0
	for _, s := range k.captureSizes {
		r := (s + q - 1) / q * q
		if r <= k.captureTokens && r <= k.maxNumSeqs*q {
			best = max(best, r)
		}
	}
	return best
}

// samplerPeakBytes is the profile run's dummy sampler pass (gpu/model_runner.py:942-954,
// sample/sampler.py:239): logits for one row per dummy request in the head dtype, the
// model's, and the sampler's FP32 copy of them, over the whole vocabulary, which
// compute_logits gathers on every rank. The dummy batch holds min(max_num_batched_tokens,
// max_num_seqs) requests, fewer where a parallel-block drafter's query rows would overrun
// the token budget (_dummy_run, :785-830).
func (k *Kernel) samplerPeakBytes(g *model.Graph) float64 {
	reqs := min(k.batchedTokens, k.maxNumSeqs)
	if d := k.plan.Draft; d != nil && k.speculating && d.Drafting == model.DraftingParallelBlock {
		reqs = min(reqs, k.batchedTokens/max(k.numSpecTokens(), 1))
	}
	return float64(reqs) * float64(g.Global.VocabSize) * (price.ActivationBytes + 4)
}

// activationWidth is h in the activation form: num_attention_heads x head_dim.
//
// That is NVIDIA's definition (AISimulate sdk/backends/base_backend.py: h =
// model._num_heads * model._head_size), where head_size is the config's head_dim, or
// hidden_size / heads when the config states none (sdk/utils.py). It equals hidden_size
// unless a config declares a head_dim that differs, and several do: qwen3-30b-a3b is
// 32 x 128 = 4,096 over a 2,048 hidden size.
//
// The graph carries it for a full or windowed attention layer, whose d_h is the config's
// head_dim. A LATENT layer's d_h is not a head_dim but the cache width (kv_lora_rank +
// qk_rope_head_dim), so for one the width is hidden_size. That is exact for every
// DeepSeek-V3-derived config, which states no head_dim, and it is the one place this can
// differ from NVIDIA's figure: a latent config that DOES state a head_dim (glm-5's is 64,
// deepseek-v4's 512) has an h the graph does not record.
//
// Taken as the widest over the stack, so a model whose attention kinds disagree is
// charged its widest scratch.
func activationWidth(p *price.Plan, g *model.Graph) float64 {
	var h int
	for _, l := range p.Layers {
		if l.AttnQHeads <= 0 {
			continue
		}
		w := l.AttnQHeads * l.AttnHeadDim
		if latentAttention(l.AttnKind) {
			w = g.Global.HiddenSize
		}
		h = max(h, w)
	}
	if h == 0 {
		h = g.Global.HiddenSize
	}
	return float64(h)
}

// lcm is the least common multiple of two positive integers.
func lcm(a, b int) int {
	x, y := a, b
	for y != 0 {
		x, y = y, x%y
	}
	return a / x * b
}

// a100Parts are the catalog's A100 parts, by catalog key. vLLM v0.31.0's batch defaults
// exclude an A100 by its device name (EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:
// 2858-2887: the 70 GiB tier applies to a device not named A100), an identity the compute
// capability does not carry -- SM80 is every A100, and SM89 parts such as the L40S are not
// A100s but are not SM80 either -- so the parts are named exactly rather than matched by a
// substring of the name.
var a100Parts = map[string]bool{"a100-80": true, "a100-sxm": true}

// isA100 reports whether the chip is one of the catalog's A100 parts.
func isA100(chip hardware.Chip) bool { return a100Parts[chip.Name] }

// defaultMaxNumSeqs is vLLM v0.31.0's default sequence cap for a part in the
// OpenAI-API-server context (EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:
// 2858-2887): 1,024 on a part with at least 70 GiB that is not an A100, 256 otherwise.
func defaultMaxNumSeqs(chip hardware.Chip) int {
	if chip.MemoryGiB >= 70 && !isA100(chip) {
		return 1024
	}
	return 256
}

// batchedDPMoE is ParallelConfig.use_batched_dp_moe (vllm/config/parallel.py:746-755 at
// v0.31.0): a DeepEP low-latency or NIXL-EP dispatch with expert parallelism over dp > 1.
func (k *Kernel) batchedDPMoE() bool {
	switch k.rules.EffectiveAll2AllBackend(k.engine.All2AllBackend) {
	case "deepep_low_latency", "nixl_ep":
		return k.pool.Parallel.EnableExpertParallel && k.layout.DP > 1
	}
	return false
}

// defaultBatchedTokens is the token budget an unstated max_num_batched_tokens resolves to:
// 256 under a batched data-parallel MoE (vllm/engine/arg_utils.py:3046-3051), the part's
// API-server default otherwise.
func (k *Kernel) defaultBatchedTokens() int {
	if k.batchedDPMoE() {
		return 256
	}
	return defaultMaxNumBatchedTokens(k.chip)
}

// defaultMaxNumBatchedTokens is vLLM v0.31.0's default token budget for a part, in the
// OpenAI-API-server usage context an llm-d deployment serves through
// (EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:2858-2887): 16,384 on a part
// with at least 160 GiB, 8,192 on one with at least 70 GiB that is not an A100, 2,048
// otherwise.
func defaultMaxNumBatchedTokens(chip hardware.Chip) int {
	switch {
	case chip.MemoryGiB >= 160:
		return 16384
	case chip.MemoryGiB >= 70 && !isA100(chip):
		return 8192
	}
	return 2048
}

// liftCollectiveFloors resolves one floor and one rate per collective operation.
//
// The entry names carry operation, dtype and rank count because the coefficient
// schema's scope has no member for any of the three. A deployment whose exact
// combination was not measured is an error rather than a substitution: borrowing
// another width's floor would misprice by up to 2.7x in the measured set, and doing
// so silently is what this refuses.
func (k *Kernel) liftCollectiveFloors(c *resolve.Coefficients) error {
	// The dtype the collectives move, which selects which measured sweep prices them: the
	// 16-bit one, always. Collectives carry activations, which are bf16 whatever the weights
	// are served in (price.ActivationBytes has the citation, and the one exception this does
	// not model: a quantized MoE dispatch); an int8-served model's
	// all-reduce is a bf16 all-reduce, and the int8 sweep describes a payload no layer here
	// sends. An earlier form picked the int8 sweep for an int8-served model.
	const dtype = "fp16"
	chip := strings.ReplaceAll(k.chip.Name, "-", "_")
	k.collectiveFloors = map[collKey]time.Duration{}
	k.collectiveRates = map[collKey]float64{}
	k.collectiveTransitions = map[collKey]float64{}
	names := map[model.Op]string{
		model.OpAllReduce:     "all_reduce",
		model.OpAllGather:     "all_gather",
		model.OpReduceScatter: "reduce_scatter",
		model.OpAll2All:       "alltoall",
	}
	// One triple per (op, group), because a group is what picks the width and two groups
	// can run the same op. The tensor-parallel and expert-parallel axes cover everything
	// a model graph emits; the decode-context axis is resolved ONLY when dcp exceeds one,
	// so a deployment without DCP asks the registry for exactly the coefficients it
	// always did and cannot newly fail construction.
	keys := []collKey{
		{Op: model.OpAllReduce, Group: price.GroupTP},
		{Op: model.OpAllGather, Group: price.GroupTP},
		{Op: model.OpReduceScatter, Group: price.GroupTP},
		{Op: model.OpAll2All, Group: price.GroupExpert},
	}
	if k.layout.MoEDispatches() && !k.routedAll2All() && k.layout.MoEGroup() > 1 {
		// allgather_reducescatter's two phases over the expert group (agrsDispatchSeconds).
		keys = append(keys,
			collKey{Op: model.OpAllGather, Group: price.GroupExpert},
			collKey{Op: model.OpReduceScatter, Group: price.GroupExpert})
	}
	if k.layout.DCP > 1 {
		// Exactly the collectives the resolved combine launches over the decode-context
		// group (dcpDecodeCollectives has the table and the citations): an all-gather
		// always -- the query with PCP off, and the log-sum-exp under ag_rs -- then a
		// reduce-scatter for ag_rs, an all-reduce for ag_rs under PCP, or one all-to-all
		// carrying output and log-sum-exp together for a2a. Resolving only those means a part
		// missing an unused op is not refused.
		combine := model.OpReduceScatter
		switch {
		case k.decodeContext.CommBackend == resolve.DCPAllToAll:
			combine = model.OpAll2All
		case k.layout.PCP > 1:
			combine = model.OpAllReduce
		}
		keys = append(keys,
			collKey{Op: model.OpAllGather, Group: price.GroupDCP},
			collKey{Op: combine, Group: price.GroupDCP})
	}
	if k.layout.PCP > 1 {
		// One KV all-gather per layer that holds KV, so every rank keeps a full cache
		// replica of what the split prefill wrote.
		keys = append(keys, collKey{Op: model.OpAllGather, Group: price.GroupPCP})
	}
	for _, key := range keys {
		op := key.Op
		measured := names[op]
		// The rank count a collective spans, snapped to a width this chip was
		// actually measured at. The widths come from the registry rather than from a
		// constant here, so a part swept at 2 and 4 ranks only (a Grace-Blackwell
		// tray is four GPUs) is not asked for an 8-rank figure that does not exist.
		ranks, err := k.groupWidth(c, key, measured, dtype, chip)
		if err != nil {
			return err
		}
		stem := fmt.Sprintf("%s_%s_%drank_%s", measured, dtype, ranks, chip)
		floor, err := c.Value("collective_floor_" + stem)
		if err != nil {
			return fmt.Errorf("no measured floor for a %d-rank %s on %s: %w",
				ranks, measured, k.chip.Name, err)
		}
		rate, err := c.Value("collective_peak_rate_" + stem)
		if err != nil {
			return fmt.Errorf("no measured rate for a %d-rank %s on %s: %w",
				ranks, measured, k.chip.Name, err)
		}
		k.collectiveFloors[key] = time.Duration(floor * float64(time.Microsecond))
		k.collectiveRates[key] = rate * 1e6 // bytes per microsecond to bytes per second
		// The transition rate is what actually prices a forward pass's collectives; the
		// peak above is only a ceiling. Absent, the pricing falls back to the
		// two-parameter form, which is optimistic in the transition region — so its
		// absence is reported rather than silently tolerated.
		transition, err := c.Value("collective_transition_rate_" + stem)
		if err != nil {
			return fmt.Errorf(
				"no measured transition rate for a %d-rank %s on %s; the two-parameter "+
					"form understates a collective by up to 3.8x in the message range a "+
					"forward pass produces: %w", ranks, measured, k.chip.Name, err)
		}
		k.collectiveTransitions[key] = transition * 1e6
	}
	return nil
}

// snapDeclared snaps a width to one the registry declares for a family of per-width
// entries, named by format with a %d for the width: the widest declared width at or below
// it, or the narrowest declared where it is below them all.
//
// Separate from groupWidth because these families are vendor descriptor figures --
// NVIDIA's communicator sizes and activation multiples -- not measured collective sweeps,
// and a slightly mis-sized memory allowance is not a mispriced latency term, so snapping is
// right here where groupWidth refuses. NVIDIA's own model indexes its tables at
// min(tp, 8) (AISimulate base_backend.py). The registry declares both families at widths
// up to 8, so snapping a wider tp already lands on 8; the activation caller also clamps to
// 8 explicitly, and the communicator caller would take a wider entry if one were declared.
func snapDeclared(c *resolve.Coefficients, format string, width int) int {
	var declared []int
	for _, w := range []int{1, 2, 4, 8, 16} {
		if c.Has(fmt.Sprintf(format, w)) {
			declared = append(declared, w)
		}
	}
	if len(declared) == 0 {
		return width
	}
	best := declared[0]
	for _, w := range declared {
		if w <= width {
			best = w
		}
	}
	return best
}

// candidateRankWidths are the group widths any AISimulate comm sweep in this project has
// ever carried. It is the SEARCH SPACE groupWidth discriminates against — the widths for
// which "some part was swept here, so fit it for yours" is a fair demand — and nothing
// else. Which widths a given part actually carries is read from the registry by
// measuredWidths, so this list does not gate what can be used; a sweep at a width absent
// from it is still found and still priced.
//
// It is a constant because it encodes a claim about the DATA LANDSCAPE rather than about
// any registry state: 2, 4, 8 and 16 are the widths NVIDIA's comm sweeps cover across the
// SKUs this project reads. Extend it when a sweep at a new width appears upstream, which
// turns "past all available data, clamp" into "this part has a gap, fit it" for that
// width. Leaving it stale is the safe direction — a wider group clamps with the
// approximation visible in provenance rather than erroring.
var candidateRankWidths = []int{2, 4, 8, 16}

// measuredWidths returns the widths this chip carries a complete coefficient triple
// for, ascending. Completeness matters: liftCollectiveFloors needs floor, peak rate
// AND transition rate, and a width holding only some of the three cannot price a
// collective, so it is not a measured width for this purpose.
//
// Discovered from the coefficient names the registry actually carries, rather than probed
// against a list of widths this file knows about. The difference matters the moment the
// registry grows: a sweep at a width no constant here mentions would otherwise be
// invisible — present in the registry, never looked for, and the deployment that needs it
// priced by clamping to a narrower figure or refused outright. Reading the names means a
// width becomes usable by being committed upstream, with no change here.
func measuredWidths(c *resolve.Coefficients, measured, dtype, chip string) []int {
	// collective_floor_<op>_<dtype>_<N>rank_<chip>
	prefix := "collective_floor_" + measured + "_" + dtype + "_"
	suffix := "rank_" + chip

	var out []int
	for _, name := range c.Names() {
		digits, ok := strings.CutPrefix(name, prefix)
		if !ok {
			continue
		}
		digits, ok = strings.CutSuffix(digits, suffix)
		if !ok {
			continue
		}
		w, err := strconv.Atoi(digits)
		if err != nil || w < 1 {
			continue
		}
		stem := fmt.Sprintf("%s_%s_%drank_%s", measured, dtype, w, chip)
		if c.Has("collective_peak_rate_"+stem) &&
			c.Has("collective_transition_rate_"+stem) {
			out = append(out, w)
		}
	}
	sort.Ints(out)
	return out
}

// groupWidth returns the rank count a collective spans, snapped to a width this chip
// was measured at.
//
// Two cases, and the discriminator is the SEARCH SPACE rather than this part's own
// widths. A group wider than every width any sweep in this project covers
// (candidateRankWidths) is past all available data: it is priced at the widest
// measured figure, which understates its floor, and the clamp is recorded in
// provenance because refusing would decline to price a deployment the data merely
// does not reach. A group whose width IS in the search space but was not swept for
// this part is an error. That is the rack-scale case — a Grace-Blackwell tray is four
// GPUs, so GB200 and GB300 carry 2- and 4-rank sweeps while an 8-rank group is a
// routine tp=8 deployment — and substituting the 4-rank floor there understates an
// 8-rank all-reduce by 1.53x to 1.91x across the seven parts measured at both widths.
// Making that substitution silently is what liftCollectiveFloors' doc comment
// promises not to do.
func (k *Kernel) groupWidth(
	c *resolve.Coefficients, key collKey, measured, dtype, chip string,
) (int, error) {
	// The group decides the width, not the op: an all-gather spans tp ranks on the
	// tensor-parallel axis and dcp ranks on the decode-context-parallel one. An axis with
	// no width is refused here rather than resolved against a substitute, which is the
	// same standard this function already holds for an unmeasured width.
	width, ok := k.groupSize(key.Group)
	if !ok {
		return 0, fmt.Errorf(
			"collective %s names parallelism axis %d, whose width this kernel cannot "+
				"determine; pricing it at the tensor-parallel width would understate a "+
				"narrower group's floor by up to 1.59x", key.Op, key.Group)
	}
	widths := measuredWidths(c, measured, dtype, chip)
	if len(widths) == 0 {
		return 0, fmt.Errorf(
			"no measured %s widths at all for %s on %s: the registry carries no "+
				"complete floor/peak/transition triple for any rank count",
			measured, dtype, k.chip.Name)
	}
	// Narrower than anything measured: the narrowest is the only defensible figure,
	// and a 1-rank group does no collective the caller reaches this path for.
	if width <= widths[0] {
		return widths[0], nil
	}
	// Exact hit on a width this part was swept at.
	for _, w := range widths {
		if w == width {
			return width, nil
		}
	}
	// Past the whole search space: clamp to this part's widest measured figure.
	if width > candidateRankWidths[len(candidateRankWidths)-1] {
		return widths[len(widths)-1], nil
	}
	return 0, fmt.Errorf(
		"no %d-rank %s measurement for %s: the registry carries %v-rank figures for "+
			"this part, and borrowing a narrower width's floor understates an 8-rank "+
			"all_reduce by 1.53x to 1.91x across the parts measured at both widths. "+
			"Fit a %d-rank %s coefficient for %s, or run this deployment at a "+
			"measured width",
		width, measured, k.chip.Name, widths, width, measured, k.chip.Name)
}

// dtypeFit maps a weight dtype to its coefficient-name suffix and the chip's peak rate for
// it. A format the chip does not support natively has no peak here: pricing it at a
// nominal rate the hardware reaches only through a dequantize path would overstate it.
func dtypeFit(d model.DType, c hardware.Chip) (suffix string, peak float64) {
	switch d {
	case model.DTypeBF16, model.DTypeFP16:
		return "bf16", c.BF16Peak * 1e12
	case model.DTypeFP8:
		if c.FP8Peak > 0 {
			return "fp8", c.FP8Peak * 1e12
		}
		return "bf16", c.BF16Peak * 1e12
	case model.DTypeNVFP4, model.DTypeMXFP4:
		if c.NVFP4Peak > 0 {
			// The nvfp4 suffix, not fp8. This returned "fp8" while the registry carried
			// six fitted gemm_*_nvfp4 entries, so those were never read and a four-bit
			// deployment was priced with the fp8 asymptote -- 0.717 against the 0.504
			// fitted for nvfp4 on B200 and 0.441 on B300, over-pricing efficiency by
			// 1.42x and 1.73x on top of the NVFP4 peak those fractions are taken of.
			return "nvfp4", c.NVFP4Peak * 1e12
		}
		// Four-bit weights on a part without native support are dequantized to a wider
		// format before the matmul, so the achievable rate is that wider format's.
		if c.FP8Peak > 0 {
			return "fp8", c.FP8Peak * 1e12
		}
		return "bf16", c.BF16Peak * 1e12
	case model.DTypeINT8:
		return "fp8", c.FP8Peak * 1e12
	case model.DTypeINT4:
		// W4A16: the weights are four-bit but the activations are not quantized at all
		// (compressed-tensors leaves input_activations null), so the matmul runs at the
		// compute dtype's rate after dequantizing the weights. That is BF16, not a
		// four-bit rate -- the narrow storage buys memory traffic, not FLOPs. Stated
		// rather than left to the fallthrough, because the two agree only by coincidence
		// and a reader should see which one is intended.
		return "bf16", c.BF16Peak * 1e12
	}
	return "bf16", c.BF16Peak * 1e12
}

// gemmFit is one compute format's peak and efficiency ramp.
type gemmFit struct {
	peak, epsMax, mHalf float64
	// nHalf and kHalf extend the ramp to the GEMM's own shape, where the registry carries
	// both for the format; shapeAware says whether it does.
	nHalf, kHalf float64
	shapeAware   bool
	// weightOnly is a weight-only kernel's own measured ramp (Marlin W8A16 or W4A16), which
	// the registry validated with experts divided by their tensor-parallel slice.
	weightOnly bool
}

// fitKey is a format and whether it is a grouped (MoE) GEMM, which can select a ramp of
// its own.
type fitKey struct {
	f       price.Format
	grouped bool
}

// fitSpec is how a format is priced: the ramp's coefficient suffix and the peak it is a
// fraction of. fallback is the suffix to use when the registry carries no ramp under
// suffix, and why is the assumption recorded if it is used, or if the rate itself is one.
type fitSpec struct {
	suffix, fallback string
	peak             float64
	why              string
}

// dtypeFitSuffix is the served dtype's fit, as dtypeFit has always chosen it.
func dtypeFitSuffix(d model.DType, chip hardware.Chip) fitSpec {
	suffix, peak := dtypeFit(d, chip)
	return fitSpec{suffix: suffix, peak: peak}
}

// formatFit chooses the kernel a GEMM of this format runs on this chip, which is what its
// rate is.
//
//   - FP32 parameters (DeepSeek-V4.1's mHC) run at the TF32 tensor-core rate. The catalog
//     states no TF32 peak; NVIDIA's data-center datasheets put dense TF32 at half the dense
//     BF16 figure on every part the catalog carries, which is what is taken.
//   - MXFP8 on SM90: the FlashInfer MXFP8 kernels ranked ahead of Marlin require SM100, so
//     vLLM v0.31.0 serves it with MarlinMxfp8LinearKernel, "MXFP8 W8A16 GEMM via Marlin"
//     (vllm/model_executor/kernels/linear/__init__.py:529-538, mxfp8/flashinfer.py:47-53,
//     134-141, mxfp8/marlin.py:13-14): BF16 compute, on the registry's w8a16_mxfp8 ramp.
//   - Four-bit weights on a part without a four-bit peak are weight-only too: Marlin
//     W4A16 for the MXFP4 MoE backends on SM90 (fused_moe/oracle/mxfp4.py:381-390, the
//     TRTLLM and DeepGEMM backends being SM100/SM120 only), at the BF16 rate. An earlier
//     form priced them at the FP8 peak, a rate the dequantize-then-BF16 path never reaches.
//   - Everything else is dtypeFit's choice.
func formatFit(f price.Format, grouped bool, chip hardware.Chip, hopper bool) fitSpec {
	bf16 := chip.BF16Peak * 1e12
	switch {
	case f.DType == model.DTypeFP32:
		return fitSpec{suffix: "bf16", peak: bf16 / 2,
			why: "FP32 weights run at the TF32 tensor-core rate, taken as half the dense BF16 " +
				"peak as NVIDIA's datasheets state it for every catalog part; the catalog states " +
				"no TF32 peak, and the ramp is BF16's"}
	case f.DType == model.DTypeFP8 && f.MX() && hopper:
		return fitSpec{suffix: "w8a16_mxfp8", fallback: "bf16", peak: bf16}
	case (f.DType == model.DTypeMXFP4 || f.DType == model.DTypeNVFP4) && chip.NVFP4Peak <= 0:
		if grouped {
			// NVFP4 on Hopper is Marlin W4A16 too ("MuseSpark NVFP4: Marlin W4A16 GEMMs",
			// vllm/models/muse_spark/nvidia/muse_moe_runner.py:540-551@0f6b2619), with
			// 16-element FP8 scales where MXFP4 has 32-element E8M0 ones; its own grouped
			// ramp is read where the registry carries one, the MXFP4 one otherwise.
			if f.DType == model.DTypeNVFP4 {
				return fitSpec{suffix: "w4a16_nvfp4_grouped", fallback: "w4a16_mxfp4_grouped",
					peak: bf16}
			}
			return fitSpec{suffix: "w4a16_mxfp4_grouped", fallback: "bf16", peak: bf16}
		}
		return fitSpec{suffix: "bf16", peak: bf16}
	case f.DType == model.DTypeNVFP4 && grouped:
		// A native-FP4 part's NVFP4 fused MoE (FlashInfer TRT-LLM, CUTLASS NvFp4) runs its
		// own grouped kernels, whose efficiency is not the dense FlashInfer CuTe-DSL linear's
		// (blis-registry#51). Its own grouped ramp is read where the registry carries one,
		// the dense NVFP4 ramp otherwise, which is recorded.
		spec := dtypeFitSuffix(f.DType, chip)
		if spec.suffix == "nvfp4" {
			spec.fallback, spec.suffix = spec.suffix, "nvfp4_grouped"
		}
		return spec
	}
	return dtypeFitSuffix(f.DType, chip)
}

// addFit lifts the fit for a format, once.
func (k *Kernel) addFit(c *resolve.Coefficients, f price.Format, grouped bool) error {
	key := fitKey{f, grouped}
	if _, ok := k.fits[key]; ok {
		return nil
	}
	hopper := false
	if f.DType == model.DTypeFP8 && f.MX() {
		hopper = k.hopper()
	}
	spec := formatFit(f, grouped, k.chip, hopper)
	if spec.fallback != "" && !c.Has("gemm_eps_max_"+spec.suffix) {
		k.assume("gemm_eps_max_"+spec.suffix, spec.fallback,
			"no efficiency ramp for the "+spec.suffix+" kernel on "+k.chip.Name+", so it is "+
				"priced on the "+spec.fallback+" ramp at the peak it computes at")
		spec.suffix = spec.fallback
		// A fallback may need one of its own: NVFP4 W4A16 on Hopper falls back to the MXFP4
		// Marlin ramp, which a registry may not carry either.
		if spec.suffix == "w4a16_mxfp4_grouped" && !c.Has("gemm_eps_max_"+spec.suffix) {
			k.assume("gemm_eps_max_"+spec.suffix, "bf16",
				"no efficiency ramp for the "+spec.suffix+" kernel on "+k.chip.Name+", so it "+
					"is priced on the bf16 ramp at the BF16 peak it computes at")
			spec.suffix = "bf16"
		}
	}
	if spec.why != "" {
		k.assume("compute_rate_"+string(f.DType), spec.suffix, spec.why)
	}
	fit, err := k.liftFit(c, spec)
	if err != nil {
		return fmt.Errorf("weights in %s: %w", f.DType, err)
	}
	fit.weightOnly = strings.HasPrefix(spec.suffix, "w8a16_") ||
		strings.HasPrefix(spec.suffix, "w4a16_")
	k.fits[key] = fit
	return nil
}

// liftFit reads one fit's ramp. The asymptote and half-max are required; the shape-aware
// pair is opt-in by its presence, and both are required together: a k factor without an n
// factor is a different law, not a partial one, and fitting one of the two leaves the
// other's work absorbed into epsMax.
func (k *Kernel) liftFit(c *resolve.Coefficients, spec fitSpec) (gemmFit, error) {
	if err := mustPositive("compute peak", spec.peak); err != nil {
		return gemmFit{}, err
	}
	fit := gemmFit{peak: spec.peak}
	var err error
	if fit.epsMax, err = c.Value("gemm_eps_max_" + spec.suffix); err != nil {
		return gemmFit{}, fmt.Errorf("no efficiency asymptote: %w", err)
	}
	if fit.mHalf, err = c.Value("gemm_m_half_" + spec.suffix); err != nil {
		return gemmFit{}, fmt.Errorf("no efficiency half-max: %w", err)
	}
	if err := mustPositive("gemm_eps_max_"+spec.suffix, fit.epsMax); err != nil {
		return gemmFit{}, err
	}
	nHalf, nErr := c.Value("gemm_n_half_" + spec.suffix)
	kHalf, kErr := c.Value("gemm_k_half_" + spec.suffix)
	switch {
	case nErr == nil && kErr == nil:
		fit.nHalf, fit.kHalf, fit.shapeAware = nHalf, kHalf, true
	case nErr == nil || kErr == nil:
		return gemmFit{}, fmt.Errorf("%s has one of gemm_n_half/gemm_k_half but not both; "+
			"the shape-aware ramp needs both or neither", spec.suffix)
	}
	return fit, nil
}

// fit returns the lifted fit for a format.
func (k *Kernel) fit(f price.Format, grouped bool) gemmFit {
	return k.fits[fitKey{f, grouped}]
}

// validateDocuments runs blis-schemas' FIELD validation over the documents New was given.
//
// Why here rather than in Open: both constructors end in New, so putting it here means a
// caller that assembles Inputs directly cannot skip it. Why at all: nothing in this
// repository called any Validate() before, while Inputs' doc comment claimed the documents
// "arrive already validated" — so a caller was promised a check that did not happen.
// LoadModelGraph only deserializes. Strict decoding catches an unparseable file or a
// misspelled key, not a cyclic layer graph, a self-edge, a duplicate layer-kind id or an
// unrecognized emit condition; model.Graph.Validate holds 47 such checks and was called by
// nothing.
//
// It also closes a gap the v0.2.0 document split opened. Before the split, Scenario carried
// Pools, so one Validate covered the pool/cluster coupling. After it, those checks live in
// deployment.ValidateAgainstCluster — that a deployment's pools fill the cluster its
// scenario declares, that each local data-parallel width divides a node, that offload tiers
// draw from the declared storage inventory — and contracts.Validate is what supplies the
// cluster to them. Nothing ran it outside this repository's own fixture tests, so an
// external caller could build a kernel for a deployment that does not fit its hardware.
//
// THE FIELD LAYER ONLY, and that boundary is deliberate. blis-schemas separates field
// problems, which are version-independent and always the author's to fix, from
// version-scoped RULE problems, where a finding may mean the document is right and the
// engine version is wrong. Measured over the committed corpus: every one of the 687 bundles
// and all 32 catalog model graphs pass the field layer, while the rules layer rejects 213 of
// them — the 0.29.0 pack does not accept minimax_m3_mtp as a speculative method. Enforcing
// rules here would refuse a quarter of the deployments this project scores against measured
// data, for a reason that belongs in the catalog or the rules pack rather than in a
// constructor.
//
// Warnings are not failures: Problems.OK ignores them by design, so an out-of-tree
// quantization name reported as "not in-tree for 0.29.0" still builds.
//
// The rule layer is still computed, so it is computed with the caller's rules (Bundle.Rules,
// inference-sim#2021 B4), never the packs blis-schemas registers for itself: the kernel
// consults no rules but its caller's, even in a layer it does not enforce.
func validateDocuments(in Inputs) error {
	rep := contracts.Validate(contracts.Bundle{
		Scenario: in.Scenario, Deployment: in.Deployment, Model: in.Model,
		Chip: in.Chip, Fabric: in.Fabric, Devices: in.Devices,
		Coefficients: in.Coefficients,
		Rules:        func(string) rules.EngineRules { return in.Rules },
	})
	if !rep.Field.OK() {
		return fmt.Errorf("the documents do not validate:\n%w", rep.Field)
	}
	return nil
}

// servedDType resolves the format the linear layers are served in.
//
// A quantization the kernel does not recognize is an error rather than a fallback to the
// checkpoint's width: silently serving a request for, say, awq at bf16 would report weight
// bytes and a compute peak that are both wrong, with nothing saying so.
func servedDType(checkpoint model.DType, quantization string) (model.DType, error) {
	switch quantization {
	case "":
		return checkpoint, nil
	case "fp8", "ptpc_fp8", "fbgemm_fp8", "modelopt":
		return model.DTypeFP8, nil
	case "modelopt_fp4", "nvfp4":
		return model.DTypeNVFP4, nil
	case "mxfp4":
		return model.DTypeMXFP4, nil
	case "bitsandbytes", "awq", "awq_marlin", "gptq", "gptq_marlin", "compressed-tensors":
		// These are weight-only integer formats whose packed width depends on a
		// per-checkpoint bit count the flag does not carry. A checkpoint serving one of
		// them records the width in its own quantization_config, which the graph already
		// read, so the flag adds nothing and the graph governs.
		return checkpoint, nil
	}
	return "", fmt.Errorf(
		"quantization %q is not one this kernel prices; add it to servedDType with its "+
			"weight width rather than letting it fall through to %s",
		quantization, checkpoint)
}

// cacheDTypeBytes resolves the KV or state cache width.
//
// "auto" follows the model's COMPUTE dtype: vLLM sizes an auto cache at
// model_config.dtype (vllm/platforms/interface.py:861-862, vllm/utils/torch_utils.py:527-529
// at v0.31.0), which is bf16 or fp16 whatever the weights are stored in. That is not the
// weight storage width on a quantized checkpoint, and the difference is not cosmetic. A
// W4A16 checkpoint -- Kimi-K2.5 is compressed-tensors int4 at group_size 32 -- stores
// weights at four bits and computes in bf16, so following the weight width gave a
// half-byte KV element and, once paged, a per-block figure of zero. An fp8-served model is
// the same case: its linears return out_dtype=x.dtype, so its activations and an auto
// cache stay bf16, and an earlier form that sized it at one byte halved its cache.
//
// WHAT THIS CANNOT SEE. "auto" is resolved against the checkpoint before it reaches the
// cache: a quantization config declaring a KV algorithm (a ModelOpt checkpoint with
// kv_cache_quant_algo) turns it into that dtype (resolve_kv_cache_dtype_string,
// vllm/utils/torch_utils.py:503-518), and DeepSeek-V4 turns it into fp8_ds_mla when its
// backend uses that layout (vllm/models/deepseek_v4/attention.py:107-120). (DeepSeek-V3.2
// keeps "auto" at the model dtype, vllm/model_executor/models/config.py:37-39, whatever the
// cache.py:127 docstring says.) The graph carries neither, so such a deployment should
// state its cache dtype; "auto" here is the model dtype.
func cacheDTypeBytes(declared string, fallback model.DType) float64 {
	switch declared {
	case "", "auto":
		// A weight format narrower than 16 bits is a storage width, not a compute width:
		// the model computes in bf16 (or fp16) and the cache follows that. A 16- or 32-bit
		// format is its own compute width and passes through.
		if fallback.Bytes() < 2 {
			return model.DTypeBF16.Bytes()
		}
		return fallback.Bytes()
	case "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "fp8_ds_mla":
		return 1
	case "bfloat16", "float16":
		return 2
	case "nvfp4", "nvfp4_4over6":
		// Four-bit data plus an fp8 scale per 16 elements: head_size/2 + head_size/16
		// bytes a head (nvfp4_kv_cache_full_dim, vllm/utils/torch_utils.py:543-545).
		return 0.5 + 1.0/16
	case "nvfp4_ds_mla":
		// Its 352-byte cell is applied where it is served (dsMLAStateBytes); this is only
		// the per-element width a latent model never prices with.
		return 0.5
	case "float32":
		return 4
	}
	// checkCacheDTypes has refused every other name, so this is unreachable from New.
	return fallback.Bytes()
}

// The KV cache dtypes vLLM v0.31.0 accepts (CacheDType, vllm/config/cache.py:39-58), split by
// whether this kernel prices them. The turboquant and per-token-head formats are real but
// carry per-token scales or packings whose page size this kernel does not compute
// (MLACommonBackend.customize_spec, mla_attention.py:1580-1593, for the per-token-head
// ones), so they are refused rather than priced at a width they do not have. Some of the
// priced ones are refused for a model as well (checkCacheDTypes).
var (
	pricedCacheDTypes = map[string]bool{
		"": true, "auto": true, "float16": true, "bfloat16": true, "fp8": true,
		"fp8_e4m3": true, "fp8_e5m2": true, "fp8_inc": true, "fp8_ds_mla": true,
		"nvfp4_ds_mla": true, "nvfp4": true, "nvfp4_4over6": true,
	}
	unpricedCacheDTypes = map[string]bool{
		"turboquant_k8v4": true, "turboquant_4bit_nc": true, "turboquant_k3v4_nc": true,
		"turboquant_3bit_nc": true, "int4_per_token_head": true,
		"int8_per_token_head": true, "fp8_per_token_head": true,
	}
	// The recurrent state dtypes (MambaDType, vllm/config/cache.py:61).
	mambaCacheDTypes = map[string]bool{
		"": true, "auto": true, "float32": true, "float16": true, "bfloat16": true,
	}
)

// checkCacheDTypes refuses a cache dtype the engine does not accept, one it accepts that
// this kernel does not price, and one no backend serves for this model's attention kind:
//
//   - fp8_inc is listed by no CUDA attention backend (it appears only in
//     vllm/config/cache.py:46 and vllm/utils/torch_utils.py:46);
//   - fp8_ds_mla and nvfp4_ds_mla are listed only by the sparse-MLA backends;
//   - plain nvfp4 on a latent model is refused outright (validate_nvfp4_kv_cache_with_mla,
//     vllm/config/vllm.py:3414-3430).
//
// An unchecked name used to fall through to the weight width, which on an nvfp4-served
// model charged an unknown cache at half a byte.
//
// The question is asked of every kind the target stack holds, not of one kind standing for
// the model: a dtype must be served for each of them. An earlier form took the kind from
// the LAST-declared layer kind, so DeepSeek-V4.1-Flash, whose last declaration is a DSpark
// draft layer outside the target stack, was refused fp8_ds_mla as if it were plain MLA.
func checkCacheDTypes(cache, mamba string, kinds map[model.AttentionKind]bool) error {
	latent := false
	var notSparse model.AttentionKind
	for _, kind := range sortedKinds(kinds) {
		latent = latent || latentAttention(kind)
		if kind != model.AttentionSparseMLA && notSparse == "" {
			notSparse = kind
		}
	}
	switch {
	case cache == "fp8_inc":
		return fmt.Errorf("cache_dtype fp8_inc is served by no CUDA attention backend in " +
			"vLLM v0.31.0")
	case strings.HasSuffix(cache, "_ds_mla") && (len(kinds) == 0 || notSparse != ""):
		return fmt.Errorf("cache_dtype %q is served only by the sparse-MLA backends, and "+
			"this model's attention includes %s", cache, notSparse)
	case strings.HasPrefix(cache, "nvfp4") && !strings.HasSuffix(cache, "_ds_mla") &&
		latent:
		return fmt.Errorf("cache_dtype %q on a latent-attention model: vLLM v0.31.0 "+
			"refuses plain nvfp4 with MLA (vllm/config/vllm.py:3414-3430)", cache)
	case unpricedCacheDTypes[cache]:
		return fmt.Errorf("cache_dtype %q is a vLLM v0.31.0 format this kernel does not "+
			"price: its page carries scales or a packing the kernel does not size", cache)
	case !pricedCacheDTypes[cache]:
		return fmt.Errorf("cache_dtype %q is not a vLLM v0.31.0 cache dtype "+
			"(vllm/config/cache.py:39-58); the engine refuses it", cache)
	case !mambaCacheDTypes[mamba]:
		return fmt.Errorf("mamba_cache_dtype %q is not a vLLM v0.31.0 recurrent state "+
			"dtype (vllm/config/cache.py:61); the engine refuses it", mamba)
	}
	return nil
}

// sortedKinds returns a set's kinds in a fixed order, so a refusal names the same kind
// every time.
func sortedKinds(kinds map[model.AttentionKind]bool) []model.AttentionKind {
	out := make([]model.AttentionKind, 0, len(kinds))
	for kind := range kinds {
		out = append(out, kind)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// computeFixedBytes sums occupancy independent of the request set.
//
// COVERAGE LIMIT: on DeepSeek-V4.1-Flash with expert parallelism, vLLM v0.31.0's
// "Model loading took" figure exceeds what this charges by about 9 GiB a rank, at TP 8 and
// TP 4 alike. our calibration runs' H200 sizing gives 45.81 and 81.49 GiB, so a fit of R + S/tp has
// R = 10.1 GiB replicated and S = 285.4 GiB sharded, against this kernel's 1.0 and 286.1.
// The sharded weights agree, and the gap does not scale with TP; about 2 GiB of it comes
// with DSpark (42.64 GiB without it). The parameter and buffer paths read at v0.31.0 --
// routed experts, Marlin's MXFP4 repack and its round-up, the shared expert, the
// drafter's MoE, the embedding and LM head, Engram's host tables, the init-time buffers
// and the rotary caches -- do not account for it. Until a per-module memory dump on the
// part localizes it, capacity under expert parallelism is overstated by that much a rank
// (TestInternalH200ArmsAreTheEnginesFigures).
func (k *Kernel) computeFixedBytes(g *model.Graph) kernel.MemoryBreakdown {
	// Under pipeline parallelism a rank holds its own stage's share (liftPipelineMemory),
	// and the binding stage's is the one a single pool is sized against.
	if len(k.stages) > 0 {
		m := k.stages[k.ppKVStage].memory
		return kernel.MemoryBreakdown{
			Weights:        int64(m.weights - m.eplbRedundant),
			ActivationPeak: int64(m.activation),
			CUDAGraph:      k.captureBytes,
			EPLBRedundant:  int64(m.eplbRedundant),
			CommBuffers:    k.deepEPBufferBytes(g),
		}
	}
	var weights float64
	for _, l := range k.plan.Layers {
		weights += float64(l.Count) * k.residentBytes(l)
	}
	weights += k.residentBytes(k.plan.Head)
	weights += k.embeddingResidentBytes(g)
	// The drafter's weights, while the deployment speculates.
	weights += k.draftResidentBytes()
	eplb := k.eplbRedundantBytes(k.plan.Layers)
	return kernel.MemoryBreakdown{
		Weights:        int64(weights - eplb),
		ActivationPeak: int64(k.activationBytes(g, true)),
		CUDAGraph:      k.captureBytes,
		EPLBRedundant:  int64(eplb),
		// The NCCL communicator reservation and the engine workspace are not here: vLLM
		// holds them before the memory snapshot its budget is measured from, so they are
		// outside gpu_memory_utilization (PreSnapshotBytes). DeepEP's buffers are made
		// after it, so they are (deepEPBufferBytes).
		CommBuffers: k.deepEPBufferBytes(g),
	}
}

// embeddingResidentBytes is the target's embedding table on a rank that holds it. A graph
// that states it (blis-schemas v0.3.0 Graph.Embedding) is sized from its Gather's table; one
// that does not is inferred as a vocab x hidden table. Either is tensor-parallel sharded but
// not expert-parallel sharded.
//
// Embeddings are not quantized by --quantization in vLLM: the flag applies to the linear
// layers, and the embedding table stays at the checkpoint's width. Nor is an embedding
// quantized by a quantized CHECKPOINT: the FP8 quant config returns no method for one, and
// vLLM falls back to UnquantizedEmbeddingMethod (vllm/model_executor/layers/quantization/
// fp8.py:177-209, vocab_parallel_embedding.py:314-317 at v0.31.0), so an inferred table is
// sized at the checkpoint's width where that is a compute width and at BF16 where it is a
// storage one. Sizing DeepSeek-V4.1's at its FP8 global dtype halved a 1.32 GB table.
func (k *Kernel) embeddingResidentBytes(g *model.Graph) float64 {
	if k.plan.Embedding.Count > 0 {
		return k.residentBytes(k.plan.Embedding)
	}
	width := g.Global.WeightDType.Bytes()
	if width < model.DTypeBF16.Bytes() {
		width = model.DTypeBF16.Bytes()
	}
	return float64(g.Global.VocabSize) * float64(g.Global.HiddenSize) * width /
		float64(max(k.layout.TP, 1))
}

// eplbRedundantBytes is EPLB's redundant replicas over these MoE layers, reported apart from
// the weights rather than on top of them. expertsPerRank is the PHYSICAL count -- the model's
// experts plus the redundant ones, divided over the group -- so the expert term of
// residentBytes already holds them; the replicas' share is moved out of Weights into
// EPLBRedundant, leaving Total what a rank actually holds. An earlier form added them as
// well, counting every replica twice. Under pipeline parallelism a rank holds its own
// stage's MoE layers, so their replicas are summed over those (each MoE layer replicates its
// own experts, vllm/distributed/eplb/eplb_state.py; num_redundant_experts is per layer).
func (k *Kernel) eplbRedundantBytes(layers []price.PlannedLayer) float64 {
	if k.engine.EPLB == nil || !k.engine.EPLB.Enabled || k.layout.ExpertWidth <= 0 {
		return 0
	}
	var eplb float64
	for _, l := range layers {
		if l.ExpertWeightBytesPerExpert == 0 {
			continue
		}
		eplb += float64(l.Count) * l.ExpertWeightBytesPerExpert *
			float64(k.engine.EPLB.NumRedundantExperts) /
			float64(k.layout.ExpertWidth) / k.expertTensorShards
	}
	return eplb
}

// activationBytes is the profile run's activation peak on a rank: the forward's scratch at
// the batched-token bound -- NVIDIA's live-buffer multiple of one bf16 activation row per
// token, floored (see liftMemory) -- and, on a rank that samples, the dummy sampler's, which
// runs after the forward has released its activations (samplerPeakBytes). Under pipeline
// parallelism only the last stage samples (vllm/v1/worker/gpu/model_runner.py:1010-1015@
// v0.32.0).
//
// The 2 is the ACTIVATION width, which is the model's compute dtype and does not follow the
// served weight format: vLLM's fp8 and fp4 linears quantize their input transiently and
// return out_dtype=x.dtype (vllm/model_executor/kernels/linear/scaled_mm/cutlass.py at
// v0.31.0), so an fp8 deployment's residual stream and inter-layer buffers stay bf16. Sizing
// it at the served width would halve this term on an fp8 deployment and understate its
// occupancy.
func (k *Kernel) activationBytes(g *model.Graph, samples bool) float64 {
	const activationBytes = price.ActivationBytes
	act := math.Max(k.activationBuffers*float64(k.batchedTokens)*k.activationWidth*activationBytes,
		k.activationFloor)
	if samples {
		act = math.Max(act, k.samplerPeakBytes(g))
	}
	return act + k.engineBufferBytes(g)
}

// draftResidentBytes is the drafter's weights while the deployment speculates. A node
// aliasing a target weight -- DSpark's LM head and embedding read the target's -- adds none.
func (k *Kernel) draftResidentBytes() float64 {
	d := k.plan.Draft
	if d == nil || !k.speculating {
		return 0
	}
	var b float64
	for _, l := range append(append([]price.PlannedLayer{}, d.Layers...), d.Context...) {
		b += float64(l.Count) * k.residentBytes(l)
	}
	for _, l := range []price.PlannedLayer{d.Embedding, d.Head, d.SequentialHead} {
		b += k.residentBytes(l)
	}
	return b
}

// DeepSeek-V4.1 sizes three of its engine buffers from config.json values its graph does not
// state: hc_mult, which widens the residual stream to hc_mult x hidden_size
// (vllm/models/deepseek_v41/nvidia/model.py:639), and candidate_topk_blocks, the width of the
// candidate-block buffer (model.py:665-675). The prefill chunk is the attention class's own
// constant (vllm/models/deepseek_v41/attention.py:197). The indexer's gather holds 40 x
// max_model_len entries before the compress ratio divides it
// (get_max_prefill_buffer_size, vllm/v1/attention/backends/mla/indexer.py:791-801), and its
// profile run allocates VLLM_SPARSE_INDEXER_MAX_LOGITS_MB, 512 MiB by default, to stand for
// its logits (vllm/envs.py:61).
const (
	dsv41HCMult              = 4
	dsv41CandidateTopKBlocks = 2048
	dsv41PrefillChunkSize    = 4
	dsv41IndexerGatherFactor = 40
	dsv41IndexerLogitsBytes  = 512 << 20
	dsv41RadixTopKWorkspace  = 1 << 20
)

// engineBufferBytes is what DeepSeek-V4.1's model code allocates beside its weights at the
// token budget and the context bound, none of which the registry's generic activation
// multiple can hold, since NVIDIA's coefficient knows no model's buffers. vLLM v0.31.0 counts
// all of it against the KV cache: the profiler's non-KV figure is everything consumed since
// the instance was created plus the profile run's transient peak (memory_profiling,
// vllm/utils/mem_utils.py:300-330; determine_available_memory, vllm/v1/worker/
// gpu_worker.py:612-676). The buffers below are allocated in the model's __init__, inside
// load_model; the gather workspaces are reserved by the profile run itself, which both model
// runners run with no attention metadata (vllm/v1/worker/gpu_model_runner.py:6468-6470,
// gpu/model_runner.py:973-977), so the attention's forward_mqa and the indexer each take
// their warmup branch and ask the workspace manager for exactly what a prefill would, and
// the manager keeps it (vllm/v1/worker/workspace.py:178-260).
//
// Per rank, none divided by tensor parallelism:
//   - the target's workspace: the manager holds one buffer per lane, grown to the largest
//     request made in it (workspace.py:121-128, 233-266), so the larger of two requests.
//     The attention's: PREFILL_CHUNK_SIZE x (cdiv(max_model_len, compress_ratio) + window +
//     max_num_batched_tokens) x head_dim in BF16, beside max_num_batched_tokens x
//     round_up(index_topk + window, 128) and max_num_batched_tokens int32 indices, each
//     rounded to 256 B (vllm/models/deepseek_v41/nvidia/flashmla.py:94-117 and
//     flash_mla_mega_attn.py:305-319 on SM100); a window-only layer gathers no compressed
//     context and selects no top-k. The indexer's: 40 x max_model_len / compress_ratio
//     entries of head_dim FP8 values and 4 scale bytes, and the 1 MiB radix top-k workspace
//     (vllm/model_executor/layers/sparse_attn_indexer.py:363-381, 279-297;
//     indexer_topk.py:15). At a 1M context the indexer's is the larger;
//   - DSpark's own workspace: the V2 model runner gives the drafter a lane of its own
//     (_num_workspace_lanes, gpu_worker.py:107-112; gpu/model_runner.py:195-197), and its
//     dummy run proposes in that lane (gpu/model_runner.py:888-913), so its window-only
//     layers' request is held beside the target's rather than inside it;
//   - the indexer's stand-in for its logits, a transient of 512 MiB at the profile peak
//     (sparse_attn_indexer.py:383-388);
//   - the indexers' shared top-k buffer, max_num_batched_tokens x index_topk int32
//     (nvidia/model.py:656-660), and DSpark's own while it speculates (nvidia/dspark.py:
//     110-114);
//   - the candidate-block buffer, where a layer publishes candidates (model.py:665-675);
//   - the drafter's hidden-state buffer, max_num_batched_tokens x hc_mult x hidden_size in
//     the model dtype, whenever the deployment speculates, as DSpark is an EAGLE-family
//     method (model.py:773-784; use_eagle, vllm/config/speculative.py:1938-1942).
//
// KNOWN over-charge: the workspace manager's buffer is shared scratch, and the MoE modular
// kernels grow the same one (workspace.py:49-56), so the engine holds the larger of the two
// where this adds the attention's request to an activation multiple that already allows for
// MoE intermediates; and the indexer's logits stand-in is a transient that this adds to the
// activation multiple's own transients rather than to their peak.
func (k *Kernel) engineBufferBytes(g *model.Graph) float64 {
	if g.Global.Architecture != rules.ArchDeepseekV41 {
		return 0
	}
	tokens := float64(k.batchedTokens)
	aligned := func(b float64) float64 { return math.Ceil(b/256) * 256 }
	var draft []price.PlannedLayer
	if d := k.plan.Draft; d != nil && k.speculating {
		draft = d.Layers
	}
	topk := 0
	for _, l := range append(append([]price.PlannedLayer{}, k.plan.Layers...), draft...) {
		topk = max(topk, l.AttnIndexTopK)
	}
	// workspace is the largest request one lane's layers make.
	indexed := false
	workspace := func(layers []price.PlannedLayer) float64 {
		var ws float64
		for _, l := range layers {
			if l.AttnQHeads <= 0 || l.AttnHeadDim <= 0 {
				continue
			}
			context, selected := 0, 0
			if r := l.AttnCompressRatio; r > 0 {
				context, selected = (k.maxModelLen+r-1)/r, topk
			}
			// The attention's window_size is its ring's, which a parallel-block draft
			// layer's read window exceeds by its block positions (DSpark's 133 over 128).
			window := l.AttnWindow
			for _, c := range layerCaches(l) {
				if c.Kind == model.CacheWindow && c.Window > 0 {
					window = c.Window
				}
			}
			rows := float64(context + window + k.batchedTokens)
			combined := float64((selected + window + 127) / 128 * 128)
			ws = math.Max(ws, aligned(dsv41PrefillChunkSize*rows*float64(l.AttnHeadDim)*2)+
				aligned(tokens*combined*4)+aligned(tokens*4))
			for _, a := range l.Attentions {
				if a.HoldsKV || a.HeadDim <= 0 {
					continue
				}
				indexed = true
				entries := float64(dsv41IndexerGatherFactor * k.maxModelLen / max(a.CompressRatio, 1))
				ws = math.Max(ws, aligned(entries*float64(a.HeadDim))+aligned(entries*4)+
					dsv41RadixTopKWorkspace)
			}
		}
		return ws
	}
	total := workspace(k.plan.Layers)
	if k.engine.Speculative != nil && k.engine.Speculative.Method == "dspark" {
		total += workspace(draft)
	} else {
		total = math.Max(total, workspace(draft))
	}
	if indexed {
		total += dsv41IndexerLogitsBytes
	}
	indexBuffers := 1.0
	if k.speculating {
		indexBuffers++
	}
	total += indexBuffers * tokens * float64(topk) * 4
	// The candidate-block top-k buffer exists where some layer runs work over a buffer padded to
	// max_model_len (context.decode_extent: max_model_len), which is what V4.1's candidate
	// scoring and candidate mask do. The graph states that; no node's label is read.
	if paddedToMaxModelLen(g) {
		total += tokens * dsv41CandidateTopKBlocks * 4
	}
	if k.speculating {
		total += tokens * dsv41HCMult * float64(g.Global.HiddenSize) * price.ActivationBytes
	}
	return total + k.dsv41RopeCacheBytes(g)
}

// dsv41RopeCacheBytes is DeepSeek-V4.1's rotary cos/sin caches, which model init builds on
// the device inside load_model, at the model's whole position range and whatever the
// deployment's context bound: one for the window-only layers, plain RoPE at the config's
// max_position_embeddings, and one for the compressed layers, YaRN over
// original_max_position_embeddings x factor -- both the model's context window -- each of
// qk_rope_head_dim FP32 values a position (cos and sin halves;
// vllm/models/deepseek_v41/common/rope.py:9-54; DeepseekV4ScalingRotaryEmbedding,
// vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py:224-255). get_rope
// shares one instance between every layer of the same parameters, the drafter's and the
// indexer's included (rotary_embedding/__init__.py:73-83), so there are as many caches as
// layer types. At a 1,048,576-token window that is 256 MiB each.
func (k *Kernel) dsv41RopeCacheBytes(g *model.Graph) float64 {
	types := map[bool]int{}
	for _, lk := range g.LayerKinds {
		for _, n := range lk.Nodes {
			if n.ReadsEngineKV() && n.QKRopeHeadDim > 0 {
				types[n.CompressRatio > 0] = max(types[n.CompressRatio > 0], n.QKRopeHeadDim)
			}
		}
	}
	var total float64
	for _, rope := range types {
		total += float64(g.Global.ContextWindow) * float64(rope) * 4
	}
	return total
}

// paddedToMaxModelLen reports whether any layer kind states work launched over a buffer padded
// to the engine's max_model_len.
func paddedToMaxModelLen(g *model.Graph) bool {
	for _, lk := range g.LayerKinds {
		for _, n := range lk.Nodes {
			if n.Context != nil && n.Context.DecodeExtent == model.ExtentMaxModelLen {
				return true
			}
		}
	}
	return false
}

// expertGeom is how one MoE layer's routed experts divide over the expert group: the physical
// count, the heaviest rank's count, and that rank's share.
type expertGeom struct {
	total   int
	perRank float64
	share   float64
}

// geom returns the expert geometry of a layer, by its own expert count. The target's layers
// share the geometry lift resolved, EPLB's redundant replicas included; a layer of another
// count -- DeepSeek-V4.1's DSpark draft layers route over 128 experts against the target's
// 384 (dspark_n_routed_experts, vllm/models/deepseek_v41/nvidia/model.py:126-135 at
// v0.31.0) -- divides its own experts over the same group.
func (k *Kernel) geom(l *price.PlannedLayer) expertGeom {
	if g, ok := k.expertGeoms[l.Experts]; ok {
		return g
	}
	_, heaviest, _, _ := price.ExpertsPerRank(l.Experts, k.layout.ExpertWidth)
	if heaviest <= 0 || l.Experts <= 0 {
		return expertGeom{total: k.totalExperts, perRank: k.expertsPerRank,
			share: k.localExpertShare}
	}
	return expertGeom{total: l.Experts, perRank: float64(heaviest),
		share: float64(heaviest) / float64(l.Experts)}
}

// residentBytes is one layer's weights resident on a rank, divided as each node is laid
// out.
//
//   - A dense GEMM divides over tp, except a replicated one (tp: replicated), which every
//     rank holds whole, and, under the model's own sequence parallelism, a token-sharded
//     one, whose rank holds its whole weight to run its token shard.
//   - Routed experts: exactly one axis divides an expert's weights. Expert parallelism gives
//     a rank whole experts (expertTensorShards 1, expertsPerRank a fraction of the total),
//     and without it every rank holds every expert as a tensor slice (expertTensorShards
//     tp x dp, expertsPerRank the full count). Dividing by expertTensorShards is what makes
//     this a PER-RANK figure in the second case; omitting it reported the whole model's
//     expert weights on every rank, which on GLM-5 at tp=8 was 675 GiB against a 141 GiB
//     part. The step-time path applies the same division (step.go), and the two must
//     agree: they are the same bytes, read once per step and resident throughout.
//   - Shared experts divide like a projection, or are whole under the model's sequence
//     parallelism where they run on a token shard.
//   - A table lookup holds its table where it is device-resident: split by row over tp, by
//     hash head over its shards, or replicated. A host-resident table holds no HBM, and an
//     aliased one holds its target's.
func (k *Kernel) residentBytes(l price.PlannedLayer) float64 {
	sp := k.layout.ModelSequenceParallel
	var b float64
	for _, g := range l.DenseGEMMs {
		if g.Replicated || (g.TokenSharded && sp) {
			b += g.WeightBytes
		} else {
			b += g.WeightBytes / k.tp
		}
	}
	if l.ExpertWeightBytesPerExpert > 0 {
		b += l.ExpertWeightBytesPerExpert * k.geom(&l).perRank / k.expertTensorShards *
			k.expertPadding(&l)
	}
	if l.SharedTokenSharded && sp {
		b += l.SharedExpertWeightBytes
	} else {
		b += l.SharedExpertWeightBytes / k.tp
	}
	for _, gt := range l.Gathers {
		if gt.Aliased || k.gatherOnHost(gt) {
			continue
		}
		table := float64(gt.Rows) * float64(gt.RowBytes)
		switch gt.Shard {
		case model.ShardRows:
			table /= k.tp
		case model.ShardHeads:
			table /= float64(k.gatherShards(gt))
		}
		b += table
	}
	return b
}

// runningImpls converts the engine's implementations to the registry's key: library, variant and
// code hash. The toolkit is recorded by the engine and is not part of the key (design 6.2).
func runningImpls(in map[plan.Primitive][]plan.KernelImpl) map[string][]coefficient.KernelImpl {
	out := map[string][]coefficient.KernelImpl{}
	for prim, impls := range in {
		seen := map[coefficient.KernelImpl]bool{}
		for _, k := range impls {
			c := coefficient.KernelImpl{Library: k.Library, Variant: k.Variant, CodeHash: k.CodeHash}
			if !seen[c] {
				seen[c] = true
				out[string(prim)] = append(out[string(prim)], c)
			}
		}
		if _, ok := out[string(prim)]; !ok {
			out[string(prim)] = nil
		}
	}
	return out
}

// buildProvenance records every coefficient used, so a prediction can state its evidence.
//
// It records the kernel's own assumptions as well, after the registry's entries: see
// assume. A figure the kernel supplied itself would otherwise be invisible to Provenance
// and Evidence however its comment described it, which is how 1.3 GB of hardcoded memory
// occupancy once went unreported beside "101 measured of 133".
func (k *Kernel) buildProvenance(c *resolve.Coefficients) {
	for _, name := range c.Names() {
		r, ok := c.Admitted(name)
		if !ok {
			continue
		}
		// A tier the deployment does not offload to is admitted but unread; listing its eighteen
		// entries on every scenario would dilute what the evidence summary counts (liftTiers).
		if isOffloadTierEntry(name) && !k.tierEntries[name] {
			continue
		}
		k.origins = append(k.origins, kernel.CoefficientOrigin{
			Name: name, Set: r.Set, Method: string(r.Entry.Method),
			Scope:       scopeString(r.Entry, k.chip.Name),
			KernelMatch: string(r.Match), Primitive: r.Primitive, Running: coefficient.DescribeImpls(r.Running),
		})
	}
	k.origins = append(k.origins, k.assumptions...)
	var overrides []kernel.Override
	for _, o := range k.layout.Overrides {
		overrides = append(overrides, kernel.Override{
			Field: o.Field, Requested: o.Requested, Resolved: o.Resolved, Reason: o.Reason,
		})
	}
	backend := "nccl"
	if k.layout.CustomAllReduce {
		backend = "custom"
	}
	cascade := false
	if k.engine.DisableCascadeAttn != nil {
		cascade = !*k.engine.DisableCascadeAttn
	}
	async := k.asyncScheduling
	k.resolution = kernel.Resolution{
		// The layout this kernel priced, each width floored at one as the schema requires:
		// a consumer multiplies by these, so zero must never stand for "unset".
		TensorParallelWidth: max(k.layout.TP, 1),
		DataParallelWidth:   max(k.layout.DP, 1),
		ExpertParallelWidth: max(k.layout.ExpertWidth, 1),
		AllReduceBackend:    backend,
		AsyncScheduling:     async,
		CascadeAttention:    cascade,
		SequenceParallelMoE: k.layout.SequenceParallelMoE || k.layout.ModelSequenceParallel,
		Overrides:           overrides,
	}
}

// KernelAssumptionSet is the Set a Provenance entry carries when the kernel supplied the
// value itself rather than reading it from a registry set or the deployment. No registry
// set can take this name, since registry sets are named for their coefficient family, so a
// consumer can tell the two apart without parsing anything else.
const KernelAssumptionSet = "blis-latency-kernel"

// EngineBehaviourVersion is the vLLM release whose engine behaviour this kernel encodes.
const EngineBehaviourVersion = "0.31.0"

// EngineBehaviourCommit is the commit of vLLM's EngineBehaviourVersion tag. A comment citing
// vllm/<path>:<lines> means those lines at this commit.
const EngineBehaviourCommit = "db9527a46873454610df6dbedf79a36d6bf1a7f6"

// optional reads a coefficient the kernel can price without, returning zero when the
// scenario's sets do not carry it. Where the deployment needs it, its absence is recorded as
// an assumption, naming what is priced without it, so a registry that drops an entry moves
// Evidence rather than silently cheapening the step.
func (k *Kernel) optional(c *resolve.Coefficients, name string, needed bool,
	without string) float64 {
	if c.Has(name) {
		return c.ValueOr(name, 0)
	}
	if needed {
		k.assume(name, "absent", "not in the scenario's coefficient sets: "+without)
	}
	return 0
}

// adoptRouting takes the run's cross-rank routing imbalance. moe_routing_imbalance_median
// above is a different quantity: one rank's grouped-GEMM slowdown when ITS OWN experts are
// unevenly loaded, measured on one GPU. Cross-rank imbalance is which rank holds the most
// routed work; each MoE layer ends in a combine that waits for that rank, so the layer pays
// the busiest rank's share. The two compose.
//
// Without expert parallelism there is no cross-rank term. With it and no resolution, the
// group is priced as evenly loaded and Provenance says so.
func (k *Kernel) adoptRouting(r *routing.Resolution, moe bool) {
	if !moe || k.layout.ExpertWidth <= 1 {
		return
	}
	const name = "routing_imbalance"
	if r == nil || r.Layer == routing.LayerDefault {
		reason := "no routing resolution was supplied"
		if r != nil && r.Reason != "" {
			reason = r.Reason
		}
		k.assume(name, "1 (evenly loaded)", reason+": every expert-parallel rank is priced "+
			"at the mean rank's routed tokens and experts")
		return
	}
	k.routing = r
	method, scope := string(vocab.MethodMeasured), r.Source
	if r.Value != nil {
		method = string(vocab.MethodAssumed)
		scope = fmt.Sprintf("token %.4g, expert %.4g: %s", r.Value.TokenImbalance,
			r.Value.Expert(), r.Source)
	} else {
		for _, p := range []*routing.Profile{r.Decode, r.Prefill} {
			if p != nil {
				method = p.Method
				scope = p.Key().String() + ": " + r.Source
				break
			}
		}
	}
	set := string(r.Layer)
	if r.Set != "" {
		set += ":" + r.Set
	}
	k.assumptions = append(k.assumptions, kernel.CoefficientOrigin{
		Name: name, Set: set, Method: method, Scope: scope,
	})
}

// routingAt is the busiest-rank factors for a step routing tokens through the expert
// group: decode and prefill tokens read their own curves, weighted by their share of
// the step. Evenly loaded without a resolution.
func (k *Kernel) routingAt(s *stepShape, tokens float64) (tok, exp float64) {
	if k.routing == nil {
		return 1, 1
	}
	b := max(int(tokens+0.5), 1)
	d, p := float64(s.decodeTokens), float64(s.prefillTokens)
	if d+p <= 0 {
		d = 1
	}
	ld := k.routing.At(routing.PhaseDecode, b)
	lp := k.routing.At(routing.PhasePrefill, b)
	w := d / (d + p)
	return w*ld.TokenImbalance + (1-w)*lp.TokenImbalance,
		w*ld.ExpertImbalance + (1-w)*lp.ExpertImbalance
}

// assume records a value the kernel filled in on the deployment's behalf -- an engine
// default for a setting the deployment left unstated, where the default is a judgement the
// kernel made rather than a fact it read. It reaches Provenance as an entry from
// KernelAssumptionSet with method "assumed", so Evidence counts it in the total and lists
// it among the assumptions instead of leaving the prediction's footing overstated.
//
// name is the setting; scope carries the value and the reason, since a reader of the
// provenance trail needs both and CoefficientOrigin has no other free-text field.
func (k *Kernel) assume(name, value, reason string) {
	k.assumptions = append(k.assumptions, kernel.CoefficientOrigin{
		Name: name, Set: KernelAssumptionSet, Method: string(vocab.MethodAssumed),
		Scope: value + ": " + reason,
	})
}

// scopeString is an entry's scope as Provenance records it: the part (the deployment's, where
// the entry lists several), its model where it states one, and the scope a copied value came
// from, so a reader can tell a model's own value, and a copy of another part's, from a
// part-wide one.
func scopeString(e coefficient.Entry, chip string) string {
	s := e.Scope
	part := ""
	if len(s.Hardware) > 0 {
		part = s.Hardware[0]
		for _, h := range s.Hardware {
			if h == chip {
				part = h
			}
		}
	}
	var out string
	switch {
	case part != "" && len(s.Model) > 0:
		out = "hardware: " + part + ", model: " + s.Model[0]
	case part != "":
		out = "hardware: " + part
	case len(s.Model) > 0:
		out = "model: " + s.Model[0]
	default:
		out = "unscoped"
	}
	if e.TransferredFrom != "" {
		out += "; transferred from " + e.TransferredFrom
	}
	return out
}

// Evidence reports how much of the resolved coefficient set rests on measurement, so a
// caller can state a prediction's footing.
func (k *Kernel) Evidence() (measured, total int, assumed []string) {
	for _, o := range k.origins {
		total++
		if vocab.Method(o.Method).Evidenced() {
			measured++
		} else if vocab.Method(o.Method) == vocab.MethodAssumed {
			assumed = append(assumed, o.Name)
		}
	}
	return measured, total, assumed
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// instanceGPUs is the GPUs one instance of the pool occupies: the engine lays ranks out
// DP x PP x PCP x TP.
func (k *Kernel) instanceGPUs() int {
	pl := k.pool.Parallel
	return max(pl.TP, 1) * max(pl.PP, 1) * max(pl.DP, 1) * max(pl.PCP, 1)
}

// uplinkRate is the capacity of a switch group's uplink, in bytes per second:
//
//	n_g x gpusPerNode x nic / r
//
// n_g nodes each offer gpusPerNode rails at the NIC rate, and the uplink carries 1/r of that.
// It does not depend on the pool that asks: a physical link has one rate, and the ranks of an
// instance that pull through it at once are the demand's Streams.
func uplinkRate(nodes, gpusPerNode int, nic, ratio float64) float64 {
	return float64(nodes) * float64(gpusPerNode) * nic / ratio
}

// adoptSwitching sizes each switch group's uplink and refuses a pool whose collective would
// cross a group boundary, which the model does not price.
func (k *Kernel) adoptSwitching() {
	if k.switching == nil {
		return
	}
	for g := range k.switching.Groups {
		k.uplinkBytesPerSecond = append(k.uplinkBytesPerSecond, uplinkRate(
			k.switching.GroupNodes(g), k.layout.GPUsPerNode, k.nicBytesPerSecond,
			k.switching.Oversubscription))
	}
	k.uplinkStreams = k.instanceGPUs()
}

// collectiveCrossesGroups reports the first instance of the pool whose ranks sit under more
// than one switch, as the instance's index and the groups it spans. Instances are laid out
// one after another from the pool's first node, instanceGPUs ranks each, gpusPerNode ranks to
// a node. An instance on one node, or inside one group, has none.
func collectiveCrossesGroups(sw *scenario.Switching, firstNode, poolNodes, gpusPerNode, instanceGPUs int) (instance int, groups [2]int, crosses bool) {
	if sw == nil || gpusPerNode < 1 || instanceGPUs <= gpusPerNode {
		return 0, groups, false
	}
	instances := poolNodes * gpusPerNode / instanceGPUs
	for i := 0; i < max(instances, 1); i++ {
		lo := firstNode + i*instanceGPUs/gpusPerNode
		hi := firstNode + (i*instanceGPUs+instanceGPUs-1)/gpusPerNode
		g0, ok0 := sw.GroupOf(lo)
		for n := lo; n <= hi; n++ {
			if g, ok := sw.GroupOf(n); ok0 && ok && g != g0 {
				return i, [2]int{g0, g}, true
			}
		}
	}
	return 0, groups, false
}

// poolFirstNode is the node the pool starts on: the stated one, or after the pools before it.
func poolFirstNode(in Inputs) int {
	if in.PoolFirstNode != nil {
		return *in.PoolFirstNode
	}
	n := 0
	for _, p := range in.Deployment.Pools[:in.PoolIndex] {
		n += p.Nodes
	}
	return n
}

// checkCollectivesStayInAGroup refuses a pool under switch groups whose instance spans nodes
// unless every instance sits under one switch. The step collectives of such an instance run
// over the nodes' NICs at the node rate; one that crossed a group's uplink would be priced over
// that tier and would place a demand on it that transfers must share what remains of, and
// neither is modelled, so accepting the placement would price the step and the concurrent
// handoffs wrongly.
func (k *Kernel) checkCollectivesStayInAGroup(firstNode int) error {
	if k.switching == nil || k.instanceGPUs() <= k.layout.GPUsPerNode {
		return nil
	}
	if i, g, crosses := collectiveCrossesGroups(k.switching, firstNode, k.pool.Nodes,
		k.layout.GPUsPerNode, k.instanceGPUs()); crosses {
		return fmt.Errorf("switch_group_collective_crosses: instance %d of the pool, from node %d, "+
			"spans switch groups %d and %d; a collective that leaves its switch is priced over the "+
			"uplink with its demand on it, which this cost model does not yet do. Place the pool's "+
			"instances each inside one group", i, firstNode, g[0], g[1])
	}
	return nil
}

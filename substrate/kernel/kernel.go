// Package latencykernel implements the blis-schemas Kernel interface: it prices GPU work
// for one resolved deployment.
//
// Construction resolves everything — catalog facts, registry coefficients, the engine
// settings a layout overrides, and the model graph flattened to a per-kind plan. After
// that every method is a pure function of its arguments, so a discrete-event simulator can
// call StepTime at any simulated instant, from any goroutine, and get the same answer.
//
// Two constructors, for the two shapes a caller comes in. Open takes a scenario filename
// and the roots its names resolve against, which is what a simulator or a scoring command
// has. New takes the documents already parsed, which is what a configuration search
// producing deployment variants that were never written to disk has. Open ends in New, so
// both validate identically and neither can price a deployment the other would refuse.
//
// Speed is a requirement rather than a nicety: a simulator calls StepTime once per
// simulated step, millions of times over a run. So the hot path does arithmetic over a
// short pre-aggregated slice and allocates little: the per-resource breakdown map, which
// StepTimeInto lets a caller reuse, and a few scratch slices that grow with the number of
// requests. The plan holds one entry per distinct layer kind rather than one per layer, which
// for the deepest model in the catalog is three entries instead of 108.
package latencykernel

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/routing"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// scorerKernel names the block-index scorer's kernel in the registry's coefficient names
// (attention_decode_rate_block_index_scores and its siblings). Every attention node that
// states holds_engine_kv false is that kernel, so the name is a constant of the registry and
// not read from the node's role label.
const scorerKernel = "block_index_scores"

// Kernel prices one deployment. Obtain one from New; the zero value is not usable.
type Kernel struct {
	// Resolved configuration, frozen at construction.
	layout resolve.Layout
	fabric resolve.Fabric
	plan   *price.Plan
	pool   deployment.Pool
	chip   hardware.Chip
	// engine is the pool's engine settings, the caller's decoding of its engine block
	// (Inputs.Engine), a deep copy the kernel owns.
	engine enginesettings.Settings

	// servedDType is the weight format the linear layers actually run in, after the
	// scenario's --quantization is applied over the checkpoint's own format.
	servedDType model.DType

	// rules is the engine release's rules, kept for the per-request window sizing, the
	// effective MoE dispatch backend and the Engram defaults.
	// family is the chip's compute-capability major version (zero when the catalog states
	// none), and architecture the model class the graph names.
	rules  resolve.EngineRules
	family int
	// kvConnector is a deployment that runs a KV connector, for offload or a P/D handoff.
	kvConnector bool
	// familyAssumed is that an unstated compute capability's inference has been recorded.
	familyAssumed bool
	architecture  string
	// cacheDType is the cache dtype that runs, which a stated cache record is looked up by.
	cacheDType string

	// fits holds each compute format's peak and ramp, and layerFits each target layer's
	// dense GEMMs' fits in plan order, so the hot path indexes rather than hashes.
	fits      map[fitKey]gemmFit
	layerFits [][]gemmFit
	// prefillFit is the BF16 fit a recurrent prefill with no measured form is priced on.
	prefillFit gemmFit

	// layerKV is each target layer's cache view, caches every cache a layer owns with its
	// paging resolved (caches.go), and draftLayerKV the drafter's layers' views.
	layerKV []layerKV
	caches  []cacheTerm
	// kvPool is the engine's shared block pool the stated caches are drawn from, nil where
	// the graph states none or the grouping could not be resolved (caches.go).
	kvPool       *kvPool
	draftLayerKV []layerKV
	// draftContextKV is the drafter's context stack's views.
	draftContextKV []layerKV
	// draftKVBytesPerToken is the drafter's inferred caches, per context token, held while
	// the deployment speculates.
	draftKVBytesPerToken float64
	// speculating is a deployment that drafts with a speculator the graph states.
	speculating bool
	// adaptive is adaptive verification's resolved state, nil when the engine does not enable
	// it (adaptive.go).
	adaptive *adaptiveVerify

	// attentionByRole holds a block-index scorer's measured form where the registry carries
	// one under the kernel's name (scorerKernel), as it does for DeepSeek-V4.1's.
	attentionByRole map[string]floorRate
	// attentionPrefillByKind and attentionPrefillByRole hold a measured prefill form for a
	// primary attention kind and for a secondary kernel's role, where the registry carries
	// the whole family (see liftPrefillForms); otherwise prefill is priced with the
	// part-wide pair.
	attentionPrefillByKind map[model.AttentionKind]prefillForm
	attentionPrefillByRole map[string]prefillForm
	// mhcPre and mhcPost are the measured mHC components' forms, zero where the registry
	// carries neither member of the pair (see liftPrefillForms).
	mhcPre, mhcPost floorRate
	// hiddenSize is the model's hidden width, which the mHC form's work is counted in.
	hiddenSize int
	// recurrentPrefill holds a recurrent kind's measured prefill form, where one exists.
	recurrentPrefill map[model.RecurrentKind]floorRate
	// padFlashMLA is a DeepSeek-V4-family model on SM90 under tensor parallelism, whose
	// FlashMLA kernels pad each rank's query heads to 64 or 128 (attentionHeads).
	padFlashMLA bool
	// hostGatherBytesPerSecond is the rate a host-resident table's rows are read at.
	hostGatherBytesPerSecond float64
	// engramOnHost is where an engram_cpu_offload table resolves to, and engramShards how
	// many ways its heads split.
	engramOnHost bool
	engramShards int
	// captureBreaks is how many regions a breakable capture leaves eager, zero for a model
	// split at its attention ops instead (modeHost).
	captureBreaks int
	// breakableCapture is whether the model runs under a breakable capture at all, which a
	// pipeline stage with no attention layer of its own (captureBreaks 0) still does.
	breakableCapture bool

	// Constants lifted out of the coefficient set once, so the hot path reads fields
	// rather than hashing strings.
	hbmBytesPerSecond     float64
	computeFLOPsPerSecond float64
	// nHalf and kHalf extend the efficiency ramp to the GEMM's own shape. Zero means the
	// registry carries no shape-aware entry for this part and dtype, and the kernel keeps
	// the token-count ramp -- so a partially migrated registry prices as it did before
	// rather than pricing a GEMM at zero.
	nHalf, kHalf         float64
	gemmShapeAware       bool
	epsMax, mHalf        float64
	nvlinkBytesPerSecond float64
	nicBytesPerSecond    float64
	hostBytesPerSecond   float64
	moeImbalance         float64
	// asyncScheduling is the engine's resolved async_scheduling (resolveAsyncScheduling).
	asyncScheduling bool
	// schedPerRequest is the engine-core scheduler CPU cost per scheduled request
	// (host_step_per_request), exposed per step as schedulerHost says.
	schedPerRequest time.Duration
	// routing is the run's cross-rank routing imbalance; nil prices the expert-parallel
	// group as evenly loaded (adoptRouting).
	routing *routing.Resolution

	// Per-operation collective floors and peak rates, resolved once. Keyed by op
	// because an all-reduce floor and an all-to-all floor differ by more than 2x on
	// the same part, and by rank count inside the lookup because a ring-shaped
	// collective's floor grows with group width where a shuffle's barely does.
	//
	// Keyed by GROUP as well as op, because the op alone stopped identifying a triple
	// once decode-context parallelism arrived: a tensor-parallel all-gather at tp=8 and
	// a DCP all-gather at dcp=2 are the same primitive at two widths, and the registry
	// carries a separate floor/peak/transition for each. One entry per op would have
	// priced one of the two at the other's width. For the ops the context-parallel axes
	// add, the 8-rank floor is 1.35x to 1.60x the 4-rank figure across the nine parts
	// this registry carries at both; see price.GroupAxis for the per-op spread.
	collectiveFloors      map[collKey]time.Duration
	collectiveRates       map[collKey]float64
	collectiveTransitions map[collKey]float64

	// deepEP prices the MoE dispatch and combine where a measured DeepEP backend runs on
	// one node; nil leaves the generic routed all-to-all.
	deepEP *deepEPA2A

	// Recurrent families' measured floor and rate, keyed by kind. A kind absent from the
	// map has no measurement for this part and contributes nothing.
	recurrent map[model.RecurrentKind]floorRate

	// The attention primitive's own floor and rate, measured per part. A zero rate means
	// this part has no measured entries and the FLOPs fallback runs.
	attentionFloor time.Duration
	attentionRate  float64
	// attentionByKind holds per-kind decode terms where the registry carries them. A kind
	// absent from the map uses attentionFloor and attentionRate, so the unsuffixed pair stays
	// the default and adding a kind never changes a deployment that has no fit for it.
	attentionByKind map[model.AttentionKind]floorRate
	// Prefill attention's own floor and the fraction of the dense ramp it reaches. A zero
	// scale means this part has no measured prefill entries and the FLOPs fallback runs.
	attentionPrefillFloor time.Duration
	attentionPrefillScale float64

	// Memory occupancy outside the KV budget, every magnitude from the registry (see
	// liftMemory): the communicator reservation, the engine workspace, the graph capture,
	// and the activation form's multiple, floor, width and token bound.
	commBytes         int64
	workspaceBytes    int64
	captureBytes      int64
	activationBuffers float64
	activationFloor   float64
	activationWidth   float64
	batchedTokens     int

	// The heaviest rank's expert count, derived once, and every MoE layer's geometry by its
	// expert count (geom).
	expertsPerRank float64
	expertGeoms    map[int]expertGeom

	// totalExperts is the model's physical expert count, which the expected-coverage
	// term needs alongside the local count.
	totalExperts int
	// expertTensorShards is how many ranks one expert's weights are split across: tp x dp
	// when expert parallelism is off, since vLLM flattens the expert's tensor shard across
	// the data-parallel ranks (see lift), and 1 when it is on, because an expert-parallel
	// rank owns whole experts.
	expertTensorShards float64

	// tp and localExpertShare are the per-rank divisors, held as floats so the hot path
	// multiplies rather than converting. localExpertShare is the fraction of the model's
	// experts one rank holds, which is what divides routed FLOPs.
	tp               float64
	localExpertShare float64

	// graph is the model graph priced, as Graph returns it.
	graph *model.Graph
	// stages is the pipeline-stage split of the plan, nil at pp 1 (liftPipeline).
	stages []pipelineStage
	// ppKVBytesPerToken is the binding stage's KV per token at pp > 1, drafter included where
	// it is the last stage's, and ppKVStage which stage that is (liftPipelineMemory).
	ppKVBytesPerToken float64
	ppKVStage         int

	// KV geometry, derived once from the graph and the cache dtype.
	kvBytesPerToken float64
	blockSize       int
	// blockSizeFinal is whether blockSize is the size the engine will run: stated by the
	// deployment, or the block a DSA or DeepSeek-V4-family model always picks, on a model
	// with no recurrent layer
	// (whose block the platform may re-align even over a stated size). A check against the
	// block size runs only when it is final.
	blockSizeFinal bool
	// dcpShardsAllKVLayers is whether every layer whose cache is inferred is a kind DCP
	// shards (layers that state their caches are sized in caches.go), which
	// is what lets a whole-model capacity figure carry the shard at all.
	dcpShardsAllKVLayers bool
	// kvLayers is how many layers' caches are inferred, the count kvBytesPerToken is summed
	// over and so the divisor for one layer's share of the cache. Not TotalLayers: on a
	// hybrid stack the two differ, and Nemotron-3-Ultra holds KV on 12 of 108 layers.
	kvLayers int

	// Host overheads.
	admissionPerToken time.Duration
	// admissionPerRequest is the length-INDEPENDENT host cost before a request can be
	// scheduled. Separate from admissionPerToken because the two have different
	// dimensions: InferenceX's c=1 anchors scale as L^0.575 between 1k and 8k inputs,
	// so the deficit is not proportional to prompt length and charging it per token
	// adds steepness to a prefill term that is already linear in L.
	admissionPerRequest time.Duration
	outputTokenCost     time.Duration
	completionCost      time.Duration
	// launchEager is one kernel's launch with no graph captured.
	launchEager     time.Duration
	launchPerKernel time.Duration
	replayPerStep   time.Duration
	// graphDecode is the graph mode a uniform decode batch runs under, graphOther the
	// mode every other batch does; they differ under FULL_AND_PIECEWISE and
	// FULL_DECODE_ONLY. graphModeName is the resolved mode in vLLM's lowercase spelling,
	// which keys the capture's memory. uniformDecodeWidth is the tokens per request that
	// make a decode batch uniform.
	graphDecode        graphMode
	graphOther         graphMode
	graphModeName      string
	uniformDecodeWidth int
	// captureTokens is the largest batch a captured graph covers, decode or not; a step
	// with more tokens runs with no graph. Zero when nothing is captured.
	captureTokens int
	// captureSizes are the captured step sizes, ascending, and maxNumSeqs the resolved
	// max_num_seqs; together they set the capture memory's decode family (captureMemory).
	captureSizes []int
	maxNumSeqs   int
	// draftCaptureTokens is the largest graph a parallel-block drafter captures (draftHost).
	draftCaptureTokens int
	// mlaContextChunk is how many context tokens one chunk of an MLA prefill's context
	// gather covers (see dcpPrefillContextGathers).
	mlaContextChunk int

	// maxModelLen is the context bound as resolved (resolveMaxModelLen); zero when neither
	// the deployment nor the graph states one.
	maxModelLen int

	// recurrentCacheMode is how a hybrid model's recurrent state is cached, as resolved.
	recurrentCacheMode price.RecurrentCacheMode

	// decodeContext is how the decode-context-parallel combine runs: its backend and the
	// stripe each rank holds. Meaningful only when the layout's DCP exceeds one.
	decodeContext resolve.DecodeContext

	// Fixed occupancy, computed once.
	fixed kernel.MemoryBreakdown

	// Provenance, assembled once. assumptions are the values the kernel supplied itself,
	// appended to origins after the registry's entries.
	origins []kernel.CoefficientOrigin
	// selected is every registry entry a lookup returned while the kernel was built, for
	// the selected-entry report (RecordSelected). Pricing reads no coefficient after
	// construction, so the list is complete when New returns.
	selected    []resolve.Resolved
	assumptions []kernel.CoefficientOrigin
	// trust is the selected entries as terms of a price's trust (trust.go, VAL-3).
	trust      []trustTerm
	resolution kernel.Resolution

	// tiers maps a tier name to its device facts, for TierTime.
	tiers map[string]hardware.StorageDevice
}

// collKey identifies one measured collective triple: the primitive AND the group it runs
// across.
//
// Op alone is not enough. Two different groups can run the same primitive -- a
// tensor-parallel all-gather spans tp ranks while a decode-context-parallel one spans dcp
// -- and each width has its own measured floor, peak rate and transition rate in the
// registry. Collapsing them would silently charge one group the other's width.
type collKey struct {
	Op    model.Op
	Group price.GroupAxis
}

// floorRate is a primitive's two-parameter measured form: a fixed setup cost plus a rate.
// prefillForm is a measured prefill form for one attention kernel, per layer, rank and step
// that holds a prefill: floor + perRequest x prefill requests + perToken x prefill query
// tokens + bytes / rate, the bytes counted as the form was fitted (see the callers). The
// terms are in seconds and rate in bytes per second: float, not time.Duration, because a
// per-token term of a few nanoseconds (the indexer's is 4.66 ns) would truncate.
type prefillForm struct {
	floor, perRequest, perToken, rate float64
}

// seconds is the form's time for a step with requests prefill requests of tokens query
// tokens that read bytes.
func (f prefillForm) seconds(requests, tokens, bytes float64) float64 {
	return f.floor + requests*f.perRequest + tokens*f.perToken + bytes/f.rate
}

type floorRate struct {
	floor time.Duration
	// rate is tokens per second, converted from the registry's tokens-per-microsecond.
	rate float64
}

// Compile-time proof that this type satisfies the published interface. If the interface
// gains a method, this line fails to build rather than the mismatch surfacing at a call
// site.
var _ kernel.Kernel = (*Kernel)(nil)

// FixedBytes returns occupancy independent of the request set, per rank.
func (k *Kernel) FixedBytes() kernel.MemoryBreakdown { return k.fixed }

// SequenceFixedBytes returns per-sequence bytes that do not vary with prompt length:
// recurrent state, where the model has it and the cache mode keeps it bounded, and the
// bounded caches a graph states -- window rings and fixed state (cacheFixedBytes).
func (k *Kernel) SequenceFixedBytes() int64 {
	return k.sequenceBoundedBytes(k.cacheFixedBytes())
}

// SequenceAdmissionBytes returns the per-sequence bytes the engine reserves for a request's
// bounded state at admission: SequenceFixedBytes with each window ring at vLLM's admission
// cap rather than its steady decode hold -- cdiv(window - 1 + extra_retained +
// max_in_flight_tokens, block) + 1 pages, max_in_flight_tokens being the concurrent batches
// times max_num_batched_tokens (vllm/v1/kv_cache_interface.py:840-866;
// vllm/config/vllm.py:592-613) -- in whole blocks of the shared pool.
//
// It is what vLLM sizes a request at when it reports its KV capacity: the engine's
// "GPU KV cache size" is the pool's blocks over SequenceAdmissionBytes plus
// SequenceVariableBytes(max_model_len), in blocks, times max_model_len
// (get_kv_cache_capacity, vllm/v1/core/kv_cache_utils.py:2444-2471), and its runtime
// admission gate clamps a request's window reservation to the same cap
// (vllm/v1/core/single_type_kv_cache_manager.py:216-229). It grows with
// max_num_batched_tokens: on DeepSeek-V4.1-Flash at 8,192 tokens, 517 pages in each of eight
// window groups against the six SequenceFixedBytes holds. A caller charging every resident
// sequence the engine's reservation charges this; one charging the steady decode hold
// charges SequenceFixedBytes. It equals SequenceFixedBytes for a model with no window cache.
func (k *Kernel) SequenceAdmissionBytes() int64 {
	return k.sequenceBoundedBytes(k.cacheAdmissionBytes())
}

// SequenceBoundedBytes returns the per-sequence bytes of a request's bounded state with
// inFlightTokens scheduled but not yet settled: each window ring at
// cdiv(window - 1 + extra_retained + inFlightTokens, block) + 1 pages in whole pool blocks
// (SlidingWindowSpec.max_admission_blocks_per_request, vllm/v1/kv_cache_interface.py:840-866
// at v0.31.0), the state caches and the recurrent state as SequenceFixedBytes holds them.
// At 1 + num_spec_tokens it is SequenceFixedBytes, the steady decode hold, wherever the
// drafter retains no extra tokens (all but multi-module MTP); at the engine's
// max_in_flight_tokens it is SequenceAdmissionBytes. A simulator that grows a sequence's
// window reservation with the tokens it actually has in flight, as vLLM's scheduler does,
// charges this per step instead of holding the admission cap for the sequence's life.
func (k *Kernel) SequenceBoundedBytes(inFlightTokens int) int64 {
	return k.sequenceBoundedBytes(k.cacheBoundedBytes(max(inFlightTokens, 0),
		k.extraRetainedTokens()))
}

// SequenceBoundedBytesAt is SequenceBoundedBytes for a request that has computed only
// computedTokens: a window ring holds no page for a token it has not seen, so each ring
// pages min(computedTokens + inFlightTokens, window - 1 + extra_retained +
// inFlightTokens) tokens, plus the rules pack's page for a window that does not start on a
// page edge. That is the hold vLLM's allocate_slots reaches after remove_skipped_blocks:
// blocks wholly before the window's trailing edge are freed (get_num_skipped_tokens,
// SlidingWindowManager, vllm/v1/core/single_type_kv_cache_manager.py:1153-1175, applied by
// remove_skipped_blocks, :687-725), and new ones are allocated only for the tokens
// scheduled (get_num_blocks_to_allocate, :178-238). Past the window it equals
// SequenceBoundedBytes.
//
// KNOWN over-charge: a ring that has not yet outgrown its window starts on a page edge,
// so its trailing page is charged though vLLM holds none, at most one page a ring.
func (k *Kernel) SequenceBoundedBytesAt(computedTokens, inFlightTokens int) int64 {
	inFlight := max(inFlightTokens, 0)
	seen := max(computedTokens, 0) + inFlight
	if k.maxModelLen > 0 {
		seen = min(seen, k.maxModelLen)
	}
	if seen == 0 {
		seen = -1
	}
	return k.sequenceBoundedBytes(k.cacheBoundedBytesUpTo(inFlight, k.extraRetainedTokens(),
		seen))
}

// sequenceBoundedBytes adds the recurrent state a request holds whatever its length to the
// stated bounded caches' bytes.
func (k *Kernel) sequenceBoundedBytes(caches float64) int64 {
	total := caches
	mode := k.recurrentCacheMode
	if mode.ProportionalToContext() {
		// Under this mode the state scales with the context bound, so it belongs in the
		// variable term and this one reports none of it.
		return int64(total)
	}
	// The rank's own layers: under pipeline parallelism a worker holds the recurrent state of
	// its stage's layers only (the binding stage's, rankMemory), as it holds their KV.
	for _, l := range k.rankMemory().layers {
		if l.RecurrentStateBytes == 0 {
			continue
		}
		spec := 0
		if k.engine.Speculative != nil {
			spec = k.engine.Speculative.NumSpecTokens
		}
		per := price.RecurrentStateBytes(k.recurrentRankBytes(l), mode,
			k.maxModelLen, k.blockSize, spec, 0)
		// The state is replicated across context-parallel ranks and sharded by tensor
		// parallelism (recurrentRankBytes).
		total += per * float64(l.Count)
	}
	return int64(total)
}

// recurrentRankBytes is one rank's share of a recurrent layer's state, which the graph
// states for the whole model. vLLM divides both the convolutional and the temporal state by
// the tensor-parallel width -- the conv channels and the heads -- for every recurrent family
// (MambaStateShapeCalculator.mamba2_state_shape, gated_delta_net_state_shape and
// kda_state_shape, vllm/model_executor/layers/mamba/mamba_utils.py:195-326 at v0.31.0), as
// the step's recurrent scan already does with the heads. A head is never split, so the
// divisor is at most the head count.
//
// COVERAGE LIMIT: where tp exceeds a Mamba-2 layer's group count, vLLM widens the groups
// to cover every head's shard (extra_groups_for_head_shards, mamba_utils.py:206-210), which
// this division leaves out.
func (k *Kernel) recurrentRankBytes(l price.PlannedLayer) float64 {
	shards := k.tp
	if h := float64(l.RecurrentHeads); h > 0 && h < shards {
		shards = h
	}
	return l.RecurrentStateBytes / shards
}

// SequenceVariableBytes returns per-sequence KV bytes for a token count, quantized to the
// engine's page size.
func (k *Kernel) SequenceVariableBytes(tokens int) int64 {
	if tokens <= 0 {
		return 0
	}
	// Decode-context parallelism shards the TOKEN COUNT, and the paging happens after the
	// division rather than before it. That ordering is the engine's:
	// FullAttentionSpec.max_memory_usage_bytes divides max_model_len by dcp_world_size and
	// THEN rounds to blocks (vllm/v1/kv_cache_interface.py:578-583 at v0.31.0), and
	// max_num_blocks_per_req is cdiv(max_len, block_size * kv_shard_count) (:545-550).
	// page_size_bytes never sees dcp.
	//
	// Dividing kvBytesPerToken instead would be the smaller edit and it would be wrong in
	// two ways: it would make a page fractional, understating occupancy by up to dcp below
	// block_size*dcp tokens (8x at 16 tokens and dcp 8, 2x at 64, exact from 128 up), and
	// the same field prices PDTransferTime, which DCP does not shard at all.
	//
	// COVERAGE LIMIT, STATED RATHER THAN PAPERED OVER. kvBytesPerToken is ONE whole-model
	// scalar, summed over the inferred layers of every attention kind, and this function
	// has no per-kind context to spend. So the shard applies only when every KV-holding
	// layer is a kind DCP shards. A hybrid stack -- gpt-oss-120b alternates a 128-token
	// swa layer with a full gqa one -- is left UNSHARDED, which overstates what it needs.
	// That is the conservative direction, and under the engine's default hybrid KV-cache
	// manager it costs nothing real: the engine refuses the deployment, asserting
	// dcp == 1 with "DCP not support sliding window" (kv_cache_interface.py:868-872). With
	// the hybrid manager disabled -- explicitly, or by a KV connector that does not support
	// it (vllm/config/vllm.py) -- the windowed layers are promoted to full attention and
	// sharded, which this overstates. Expressing the hybrid case exactly would mean
	// splitting kvBytesPerToken per kind -- a larger refactor, for configurations that
	// either do not start or run only with the hybrid manager off.
	//
	// AND THIS DISAGREES WITH THE DECODE READ ON THAT SAME STACK, deliberately. The read
	// is applied per layer kind, so on a hybrid it shards the full-attention layers and
	// leaves the windowed ones whole; capacity, having no per-kind context, shards
	// nothing. A consumer reading both therefore gets two models of one cache for a
	// configuration the engine will not run. Each term is individually defensible -- the
	// read is exact where it can be, capacity errs high where it cannot -- and making
	// them agree would mean either coarsening the read to a whole-model verdict, which
	// loses accuracy on every deployment that DOES run, or the per-kind refactor above.
	// Neither is worth doing for an inadmissible layout, so the disagreement is recorded
	// rather than resolved.
	shardedTokens := tokens
	if k.layout.DCP > 1 && k.dcpShardsAllKVLayers {
		shardedTokens = (tokens + k.layout.DCP - 1) / k.layout.DCP
	}
	// The inferred caches that grow with context, the drafter's while the deployment
	// speculates (a draft layer holds a cache of its own), and the growing caches a graph
	// states, each paged as its own owner pages it. A pipeline-parallel rank holds its own
	// stage's layers' KV (rankMemory, the binding stage's). A window ring is bounded and is
	// not here (ringBytes).
	perToken := k.rankMemory().kvPerToken
	total := k.pagedKVBytes(shardedTokens, perToken) + int64(k.cacheVariableBytes(tokens))

	// Under the all-positions recurrent cache mode the state is proportional to the
	// context bound, so it is a variable cost rather than a fixed one.
	//
	// Sized from the UNSHARDED token count, which is deliberate: "Mamba state is
	// replicated across DCP/PCP ranks, never sharded"
	// (vllm/v1/kv_cache_interface.py:1097-1098 at v0.31.0). A context-parallel rank holds
	// every position's recurrent state even while holding only its shard of the KV cache.
	mode := k.recurrentCacheMode
	if mode.ProportionalToContext() {
		spec := 0
		if k.engine.Speculative != nil {
			spec = k.engine.Speculative.NumSpecTokens
		}
		for _, l := range k.rankMemory().layers {
			if l.RecurrentStateBytes == 0 {
				continue
			}
			total += int64(price.RecurrentStateBytes(k.recurrentRankBytes(l), mode,
				tokens, k.blockSize, spec, 0) * float64(l.Count))
		}
	}
	return total
}

// pagedKVBytes is tokens of perToken KV, quantized to the engine's page.
func (k *Kernel) pagedKVBytes(tokens int, perToken float64) int64 {
	return price.PagedBytes(tokens, k.blockSize, perToken)
}

// StepTime prices one forward pass. This is the hot path.
//
// The returned StepEstimate carries a freshly allocated PerResource map. A simulator that
// calls this per simulated step, millions of times, can use StepTimeInto instead and reuse
// one map across calls, which removes that allocation.
func (k *Kernel) StepTime(b kernel.Batch) kernel.StepEstimate {
	if len(k.stages) > 0 && b.PaddedTokens() > 0 {
		return k.stagedStepTime(b, nil)
	}
	return k.stepTime(b, nil)
}

// StepTimeInto prices one forward pass, writing the per-resource breakdown into per
// instead of allocating a map.
//
// per is cleared first, so a caller reuses one map for every step. Passing nil is the same
// as calling StepTime: a new map is allocated.
//
// The map is the allocation a caller can remove. The step still allocates a few scratch
// slices that grow with the number of requests -- the prefill chunks, the decode contexts,
// the prefills a context gather packs -- and BenchmarkStepTimeIntoDecode reports how many.
func (k *Kernel) StepTimeInto(b kernel.Batch,
	per map[kernel.Resource]time.Duration) kernel.StepEstimate {
	for r := range per {
		delete(per, r)
	}
	if len(k.stages) > 0 && b.PaddedTokens() > 0 {
		return k.stagedStepTime(b, per)
	}
	return k.stepTime(b, per)
}

// collectiveTime prices one collective's bytes against its own measured floor and
// rate. Both are per operation: charging an all-to-all the all-reduce floor would
// overstate it by 2.2x on H200, and the rates differ by a similar factor.
func (k *Kernel) collectiveSeconds(key collKey, bytes float64) float64 {
	if k.deepEP != nil && key.Op == model.OpAll2All && key.Group == price.GroupExpert {
		return k.deepEP.seconds(bytes)
	}
	peak, ok := k.collectiveRates[key]
	if !ok || peak <= 0 {
		// Every op the plan can emit is resolved at construction, so this is
		// unreachable for a kernel New returned. Returning infinity rather than zero
		// keeps an unpriced collective visible if that ever changes.
		return math.Inf(1)
	}
	return price.CollectiveTime(bytes, k.collectiveTransitions[key], peak,
		k.collectiveFloors[key].Seconds())
}

// collectiveTime prices one invocation, for callers outside the step loop.
func (k *Kernel) collectiveTime(key collKey, bytes float64) time.Duration {
	return seconds(k.collectiveSeconds(key, bytes))
}

// crossesNodes reports whether a collective's own group spans more than one node.
//
// Per collective rather than per deployment: a tensor-parallel group of 8 fits inside an
// 8-GPU node however wide expert parallelism is, and an expert-parallel group of 72 does
// not. Using the deployment's widest group for both would move every reduction onto the
// NIC in a wide-EP layout, where in fact it never leaves the node.
//
// And by PLACEMENT rather than by width alone, because a group's members are not always
// neighbours. See groupPerNode: a two-rank prefill-context group at tp=8 has one member on
// each of two nodes, which a width test reads as fitting comfortably inside one.
func (k *Kernel) crossesNodes(key collKey) bool {
	if k.layout.GPUsPerNode <= 0 {
		return false
	}
	width, ok := k.groupSize(key.Group)
	if !ok {
		return false
	}
	if width <= k.groupPerNode(key.Group) {
		return false
	}
	// A rack that is one NVLink domain -- an NVL72-class part, where the NVSwitch fabric
	// spans every tray at the full per-GPU rate -- is crossed only at its own boundary. A
	// group whose rank footprint fits inside it never reaches the inter-node fabric,
	// however many trays it spans.
	if k.fabric.RackIsOneDomain {
		rack := k.layout.GPUsPerRack // the resolved rack, which is the chip's
		if footprint := width * k.groupStride(key.Group); rack > 0 && footprint <= rack {
			return false
		}
	}
	return true
}

// groupPerNode is how many of one group's ranks share a node, which is what decides both
// whether the group crosses a node boundary and how much of its traffic does.
//
// THE ENGINE'S RANK LAYOUT DECIDES IT. vLLM numbers ranks ExternalDP x DP x PP x PCP x TP
// with the tensor-parallel axis innermost (vllm/distributed/parallel_state.py:2045,
// :2054-2060 at v0.31.0), and a node holds consecutive ranks. So a group's members sit
// `stride` ranks apart, where the stride is the product of the axes inside the one the
// group runs along. Within one pipeline stage (pp_rank x tp + tp_rank, :2045-2060, so a
// stage is a contiguous block of tp ranks and its groups are the ones below):
//
//	tensor-parallel     contiguous                                     stride 1
//	expert-parallel     one contiguous DP x PCP x TP block (:2212-2220)   stride 1
//	prefill-context     the PCP axis, at a fixed TP rank (:2155-2160)     stride tp
//	decode-context      within TP when pcp is 1; along the PCP axis when  stride 1, tp
//	                    dcp == pcp; the whole TP x PCP block when          or 1
//	                    dcp == tp*pcp (:2139-2144)
//
// and a node of g GPUs holds g/stride of them. At tp=8 on 8-GPU nodes a prefill-context
// group therefore has ONE member per node: every hop of its ring crosses the fabric. A
// width test priced exactly that gather on NVLink.
//
// A stride that does not divide the node (tp=6 on 8-GPU nodes) has no uniform answer, and
// is floored -- which charges more of the group to the fabric than some nodes carry, the
// conservative direction.
//
// A rack that is one NVLink domain is consulted in crossesNodes, not here: inside such a
// rack a tray boundary is not a bandwidth change, so it decides WHETHER a group crosses,
// and this per-node count only matters for a group that does.
func (k *Kernel) groupPerNode(group price.GroupAxis) int {
	gpn := k.layout.GPUsPerNode
	if gpn <= 0 {
		return math.MaxInt
	}
	return max(gpn/k.groupStride(group), 1)
}

// groupStride is how many ranks apart a group's members sit in the engine's rank layout;
// see groupPerNode for the table and its citations.
func (k *Kernel) groupStride(group price.GroupAxis) int {
	switch group {
	case price.GroupPCP:
		return max(k.layout.TP, 1)
	case price.GroupDCP:
		// dcp is 1, pcp or tp*pcp once pcp exceeds one (parallel.py:571-578); of those,
		// only dcp == pcp runs along the PCP axis rather than over a contiguous block.
		if k.layout.PCP > 1 && k.layout.DCP <= k.layout.PCP {
			return max(k.layout.TP, 1)
		}
	}
	return 1
}

// groupSize returns the rank count a collective's group spans.
//
// Read from the GROUP rather than inferred from the op, because the op no longer decides
// it: an all-gather spans tp ranks on the tensor-parallel axis and dcp ranks on the
// decode-context-parallel one.
func (k *Kernel) groupSize(group price.GroupAxis) (int, bool) {
	switch group {
	case price.GroupTP:
		return max(k.layout.TP, 1), true
	case price.GroupExpert:
		return max(k.layout.MoEGroup(), 1), true
	case price.GroupDCP:
		return max(k.layout.DCP, 1), true
	case price.GroupPCP:
		return max(k.layout.PCP, 1), true
	}
	// EXHAUSTIVE RATHER THAN DEFAULTING TO THE TENSOR-PARALLEL WIDTH. An axis this
	// function cannot name has no width, and pricing it at tp anyway is the exact failure
	// the composite key exists to prevent: an all-gather's 8-rank floor is 1.35x to 1.59x
	// its 4-rank figure across the nine parts measured at both, so a wrong axis is not a
	// rounding error.
	// Pipeline parallelism has no collective of its own: a stage boundary is a
	// point-to-point send priced apart (pipelineSendSeconds), and the receiving stage's
	// reassembly is a TP all-gather (pipelineRecvGatherSeconds). A future pipeline-wide
	// collective should arrive here as a refusal rather than as a silently tensor-parallel
	// price.
	//
	// This is the half of the resolve.Emits/Recognizes pair that a bare default would
	// have left out: that pair exists so an unrecognized condition can be refused rather
	// than quietly answered, and a width is no different.
	return 0, false
}

// spanFor returns the cross-node scaling a collective pays. A ring reduces as it travels
// and so crosses only a fraction of the fabric; a routed all-to-all must reach every peer.
func (k *Kernel) spanFor(key collKey) float64 {
	ratio := k.fabric.Ratio()
	group, _ := k.groupSize(key.Group)
	perNode := k.groupPerNode(key.Group)
	// An all-to-all is point to point unless it is the MoE dispatch over a backend that
	// moves ring-shaped volume. The decode-context a2a is all_to_all_single over its group
	// (dcp.py:1000-1005), every rank sending a slice to every other.
	if key.Op == model.OpAll2All && (key.Group != price.GroupExpert || k.routedAll2All()) {
		return price.All2AllSpan(group, perNode, ratio)
	}
	return price.RingSpan(group, perNode, ratio)
}

// denseDispatch reports whether this collective is the MoE dispatch and combine run as
// allgather_reducescatter: the default backend, and what pplx and naive resolve to.
func (k *Kernel) denseDispatch(key collKey) bool {
	return key.Op == model.OpAll2All && key.Group == price.GroupExpert &&
		k.layout.MoEDispatches() && !k.routedAll2All()
}

// agrsDispatchSeconds prices allgather_reducescatter's dispatch and combine for a rank
// holding `bytes` of hidden state. AgRsAll2AllManager gathers EVERY DP rank's tokens to
// every rank with all_gatherv and returns them with reduce_scatterv over the same group
// (vllm/distributed/device_communicators/all2all.py:84-157 at v0.31.0), so each phase's
// tensor is the whole group's tokens -- the DP funnel the expert GEMM already charges
// (moeDPFunnel) -- not one rank's. Each phase is priced as the NCCL collective it is, on
// the tensor's full size as the TP all-gather is.
func (k *Kernel) agrsDispatchSeconds(bytes float64) float64 {
	ag := collKey{Op: model.OpAllGather, Group: price.GroupExpert}
	rs := collKey{Op: model.OpReduceScatter, Group: price.GroupExpert}
	full := bytes * k.moeDPFunnel()
	if k.crossesNodes(ag) {
		full *= k.spanFor(ag)
	}
	return k.collectiveSeconds(ag, full) + k.collectiveSeconds(rs, full)
}

// routedAll2All reports whether the resolved MoE backend moves top-k-selected bytes point
// to point, rather than dense per-token bytes in two ring phases. The distinction changes
// both the volume and the cross-node scaling, and it is a property of the backend rather
// than of the primitive's name.
func (k *Kernel) routedAll2All() bool {
	if !k.pool.Parallel.EnableExpertParallel {
		// With expert parallelism off, a data-parallel MoE always falls back to the
		// all-gather/reduce-scatter dispatch, whatever backend is named
		// (vllm/model_executor/layers/fused_moe/all2all_utils.py:202-214 at v0.31.0).
		return false
	}
	if !k.layout.MoEDispatches() {
		// use_all2all_kernels is false, so no backend's kernels run
		// (vllm/model_executor/layers/fused_moe/config.py:1024-1027).
		return false
	}
	// The backend that runs: the release's default for an unstated one, and the
	// replacement for a renamed one -- pplx and naive both run as allgather_reducescatter
	// in v0.31.0 (vllm/config/parallel.py; rules.Pack.EffectiveAll2AllBackend).
	switch k.rules.EffectiveAll2AllBackend(k.engine.All2AllBackend) {
	case "", "naive", "allgather_reducescatter":
		return false
	}
	return true
}

// chunk is one prefill request's shape within a step: `sched` new tokens resuming on a
// prefix of `prefix` already-computed ones.
type chunk struct {
	sched  int
	prefix int
}

// windowedCausalFLOPs is the attention FLOPs a sliding-window layer pays for these
// prefill chunks. A query at absolute position p attends keys in [p-window+1, p], so it
// reads min(p+1, window) of them rather than the whole prefix.
//
// For a chunk of `s` tokens resuming at prefix `c`, the queries whose own position still
// fits inside the window are the first max(0, min(s, window-c)); each later query reads
// exactly `window` keys. Summing the two regimes:
//
//	nSmall = max(0, min(s, window-c))
//	pairs  = nSmall*(c+1) + nSmall*(nSmall-1)/2 + (s-nSmall)*window
//
// The 2*2 factor matches the unwindowed term above: two FLOPs per multiply-accumulate,
// and two matmuls (scores, then the value-weighted sum).
//
// The bound SATURATES at s*window however long the prefix grows, where the unwindowed
// count grows linearly in the prefix without bound. The over-charge this corrects is
// therefore governed by PREFIX length, not chunk size: about 4x at an 8,192-token prefix
// for a 1,024-token chunk with a 2,176 window, 15x at 32,768 and 60x at 131,072.
// TestWindowedCausalFLOPsMatchesBruteForce checks it against a count made one query at a time.
func windowedCausalFLOPs(chunks []chunk, window int) float64 {
	if window <= 0 {
		return 0
	}
	w := float64(window)
	var pairs float64
	for _, c := range chunks {
		s, prefix := float64(c.sched), float64(c.prefix)
		nSmall := math.Min(s, w-prefix)
		if nSmall < 0 {
			nSmall = 0
		}
		// Queries still inside the window read their whole prefix plus themselves;
		// the rest read exactly `window`. The 0.5*nSmall subtraction drops the
		// diagonal's half-token so this matches the unwindowed term's continuum
		// convention (s*c + s^2/2) rather than exceeding it by s/2 when the window
		// is wide enough to be inactive.
		pairs += nSmall*(prefix+1) + nSmall*(nSmall-1)*0.5 - nSmall*0.5
		pairs += (s - nSmall) * w
	}
	return 2 * 2 * pairs
}

// selectedKVTokens is how many KV tokens a BOUNDED layer's decode actually reads, summed
// over the batch. It returns -1 for a layer that reads its whole context, so a caller can
// tell "no bound" from "a bound that happens to equal the context".
//
// A layer's read is bounded by what it states, and the bounds compose (selectedKVTokensFor):
//
//	selected = min(ctx, window) + min(ctx / ratio, topk)
//
// The first term is a sliding window, read at full resolution; the second the positions the
// indexer selects from a stream compressed by ratio (one where nothing is compressed),
// capped at the top-k and flat in context once it is reached. A layer that compresses and
// selects nothing reads the whole compressed stream, ctx / ratio, beside its window.
// DeepSeek-V4's ratio-128 layers are that case; its ratio-4 layers and every compressed
// DeepSeek-V4.1 layer read their window plus at most index_topk positions
// (vllm/models/deepseek_v4/nvidia/flashmla.py:187-262, deepseek_v41/nvidia/flashmla.py:
// 181-246 at v0.31.0), and nothing of theirs grows with context.
//
// An earlier form read min(ctx, topk) + (ctx - topk) / ratio, which charged a top-k layer
// the compressed REMAINDER beyond its selection, and dropped the window wherever a top-k
// was stated: DeepSeek-V4-Pro's csa4 layers paid 1024 + (ctx - 1024)/4 against a real
// 128 + at most 1024. It also bounded only sparse_mla, so a sliding-window layer of any
// other kind -- swa, or an mla layer with a window -- was charged its whole context, which
// disagreed with the registry's own swa rate, fitted with reads capped at the window
// (blis-registry scripts/fit_attention_by_kind.py:98-109).
//
// WHY THIS IS NOT OPTIONAL. Without it the only available count is the full context, which
// overstates a 1M-token decode read on a compressed architecture by orders of magnitude.
// Fitting a rate against that reading forces it to 1.00 of datasheet peak -- physically
// impossible, and the clamp is how the wrong byte count announces itself.
//
// Summed per request rather than over the batch total, because min(sum) != sum(min): a
// batch holding one request below the top-k and one far above it gets both wrong if the
// bound is applied to the aggregate.
//
// NOT ON THE PRICING PATH. dcpDecodeTokens composes this bound with the decode-context
// shard per request and is what the pricer calls (decodeReadBytes, from priceLayers); this
// remains as the selection bound stated
// on its own, which is what the sparse-MLA tests pin. The two share readBounded and
// selectedKVTokensFor, so what those tests establish still holds for the pricer.
func selectedKVTokens(l *price.PlannedLayer, contexts []int) float64 {
	if !readBounded(l) {
		return -1
	}
	var total float64
	for _, ctx := range contexts {
		total += selectedKVTokensFor(l, ctx)
	}
	return total
}

// selectedKVTokensFor is one request's selected count, which selectedKVTokens sums. Split
// out so a per-request shard can be applied to each request's own selected count rather
// than to the batch total: the two differ once a sharded count is rounded per request,
// and the engine itself keeps a per-request vector (dcp_local_seq_lens,
// vllm/v1/attention/backend.py:427-428, built per request in flash_attn.py:884-893).
func selectedKVTokensFor(l *price.PlannedLayer, ctx int) float64 {
	c := float64(ctx)
	var sel float64
	if l.AttnWindow > 0 {
		sel += math.Min(c, float64(l.AttnWindow))
	}
	ratio := float64(max(l.AttnCompressRatio, 1))
	switch {
	case l.AttnIndexTopK > 0:
		sel += math.Min(c/ratio, float64(l.AttnIndexTopK))
	case ratio > 1:
		sel += c / ratio
	}
	return sel
}

// readBounded reports whether a layer's decode read is bounded below its context: by a
// window, a top-k selection or a compression ratio, on any attention kind.
//
// The single source of that decision: both selectedKVTokens and dcpDecodeTokens call it,
// so neither can disagree with the other about which layers are bounded.
func readBounded(l *price.PlannedLayer) bool {
	return l.AttnWindow > 0 || l.AttnIndexTopK > 0 || l.AttnCompressRatio > 1
}

// pcpLocalTokens is how many scheduled tokens the BUSIEST prefill-context-parallel rank
// computes, for one prefill chunk of `sched` tokens. It returns -1 when prefill-context
// parallelism is off, so a caller keeps the whole chunk.
//
// PCP SPLITS THE PREFILL ITSELF, which is its defining purpose: vLLM calls it the "Number
// of ranks that split prefill sequence computation" (vllm/config/parallel.py:131-133 at
// v0.31.0). A rank builds a LOCAL batch of its own share and runs the forward on that
// (num_local_tokens, vllm/v1/worker/gpu/pcp_manager.py:485), so every term proportional to
// scheduled tokens falls with the split -- not only attention.
//
// THE SPLIT IS A ZIGZAG, NOT A SLICE, and that is the whole reason a single divisor is
// defensible here. _iter_rank_chunks (pcp_manager.py:235-273) cuts each prefill into
// 2*pcp chunks and gives rank r chunks r and 2*pcp-1-r, which its own docstring draws:
//
//	full:  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     (pcp = 4)
//	rank 0:  0                           7
//	rank 1:      1                   6
//	rank 2:          2           5
//	rank 3:              3   4
//
// Pairing a low chunk with a high one is what balances CAUSAL work: an early chunk has few
// keys to its left and a late one has many, and the two sum to the same total on every
// rank. Verified by enumeration: at pcp 2, 4 and 8 the per-rank causal pair counts are
// EXACTLY equal -- max over ranks divided by the ideal share is 1.000000 with zero spread
// -- for unchunked prefills and for chunked ones alike (tested at 2,048 tokens on no
// prefix, 4,096 on 8,192, 1,024 on 131,072, 8,192 on none). A plain contiguous 1/pcp slice
// would NOT balance, because the last rank would carry the whole upper triangle.
//
// THE BUSIEST RANK BINDS, as it does for DCP: the ranks must gather before the next layer,
// so the step waits for the one with the most work. Where 2*pcp divides the query every
// rank holds 2*ceil(sched/(2*pcp)) tokens and the shards sum to sched exactly; where it
// does not, the ragged remainder lands unevenly and the maximum is taken over the ranks
// rather than assumed. The overhead over a nominal sched/pcp is bounded and small at any
// realistic chunk -- under 1.5% above 1,024 tokens and under 0.4% above 4,096, for pcp up
// to 8 -- and it is an OVERSTATEMENT, which is the safe direction.
//
// A SHORT PREFILL IS REPLICATED RATHER THAN SPLIT, BUT ONLY ALONGSIDE DCP. That gate is
// easy to miss and changes the answer, so it is taken from the engine verbatim:
// replicated_requests (pcp_manager.py:222-233) computes `drops_a_chunk` only inside
// `if self.dcp_world_size > 1`, so with PCP alone every prefill is partitioned however
// short it is, and a rank holding no chunk simply contributes nothing.
//
// The condition itself is not "shorter than 2*pcp" either, which is the obvious guess and
// wrong: it is `(2*pcp - 1) * ceil(sched/(2*pcp)) >= sched`, which also fires at lengths
// well above 2*pcp -- at pcp 8 it is true at 17 tokens and false at 16 -- because what it
// detects is a partition in which some chunk comes out empty.
//
// `dcp` is therefore a parameter rather than an assumption: it decides whether the
// replication branch exists at all.
func pcpLocalTokens(sched, pcp, dcp int) int {
	if pcp <= 1 || sched <= 0 {
		return -1
	}
	numChunks := 2 * pcp
	chunk := (sched + numChunks - 1) / numChunks
	if dcp > 1 && (numChunks-1)*chunk >= sched {
		// A chunk would come out empty, so the engine replicates instead: every rank
		// computes the whole query.
		return sched
	}
	// Rank r holds chunks r and 2*pcp-1-r, each truncated where it runs past the query.
	// Maximised over the ranks rather than taken as 2*chunk: that shortcut is right only
	// when every chunk is full, and it overstates a ragged partition -- at 2 tokens over
	// 2 ranks the chunks are one token each and the busiest rank holds ONE, where 2*chunk
	// capped at the query would say two.
	most := 0
	for rank := 0; rank < pcp; rank++ {
		local := 0
		for _, idx := range [...]int{rank, numChunks - 1 - rank} {
			lo := min(idx*chunk, sched)
			hi := min(lo+chunk, sched)
			if hi > lo {
				local += hi - lo
			}
		}
		if local > most {
			most = local
		}
	}
	return most
}

// appendPrefillRuns appends the prefill runs this rank computes for one request, reading
// the split widths from the resolved layout.
//
// A method rather than a bare call so that the chunk geometry stepTime depends on is
// reachable from outside stepTime: the quantity is a local, and the step time it feeds is
// too insensitive to pin the geometry through (a dense GEMM dominates a windowed attention
// term roughly five to one, so the difference between a positional reading and a
// by-token-count one is a few percent of SM). Exercising the same path the pricer takes is
// what makes that geometry testable at all.
func (k *Kernel) appendPrefillRuns(dst []chunk, sched, prefix int) []chunk {
	return appendPCPChunks(dst, sched, prefix, k.layout.PCP, k.layout.DCP)
}

// appendPCPChunks appends the prefill runs THIS RANK computes, each at its true global
// offset, so a positional cost law sees where the tokens sit rather than only how many
// there are.
//
// With prefill-context parallelism off, or for a request the engine replicates, that is
// the one run the caller would have appended anyway: the whole query at its own prefix.
// With the split on, it is the two zigzag chunks rank r owns -- chunk r near the cheap
// start and chunk 2*pcp-1-r near the expensive end -- at offsets `prefix + lo`.
//
// WHICH RANK, and why it is the same one the token count comes from: pcpLocalTokens
// maximises the token count over the ranks, so the chunks appended here are that rank's.
// For a sliding-window layer the busiest rank by token count is not necessarily the
// costliest by key count, but the two agree wherever the partition is even, and a window
// term driven by the wrong rank of an uneven partition is a second-order error against
// the 7.9x one this function exists to remove.
func appendPCPChunks(dst []chunk, sched, prefix, pcp, dcp int) []chunk {
	local := pcpLocalTokens(sched, pcp, dcp)
	if local < 0 || local == sched {
		// Not split, or replicated whole onto every rank.
		return append(dst, chunk{sched: sched, prefix: prefix})
	}
	numChunks := 2 * pcp
	size := (sched + numChunks - 1) / numChunks
	// The rank holding the most tokens, matching pcpLocalTokens' own aggregation.
	busiest, most := 0, 0
	for rank := 0; rank < pcp; rank++ {
		held := 0
		for _, idx := range [...]int{rank, numChunks - 1 - rank} {
			lo := min(idx*size, sched)
			hi := min(lo+size, sched)
			if hi > lo {
				held += hi - lo
			}
		}
		if held > most {
			busiest, most = rank, held
		}
	}
	for _, idx := range [...]int{busiest, numChunks - 1 - busiest} {
		lo := min(idx*size, sched)
		hi := min(lo+size, sched)
		if hi > lo {
			dst = append(dst, chunk{sched: hi - lo, prefix: prefix + lo})
		}
	}
	return dst
}

// pcpCausalPairs is the attention work the BUSIEST prefill-context-parallel rank does for
// one prefill chunk, in the same units the unsharded count uses: four times the
// query-key pairs, since there are two FLOPs per multiply-accumulate and two matmuls.
//
// Counted over the engine's own partition rather than scaled by a ratio, because no ratio
// is right in general. A query at chunk-local offset j attends its whole prefix plus its
// causal share within the chunk, so it reads prefix + j + 1 keys; rank r owns chunks r and
// 2*pcp-1-r. Where 2*pcp divides the query the zigzag pairing makes every rank's total
// EXACTLY equal and this reduces to the whole count over pcp -- verified equal to
// 1.000000 of the ideal share with zero spread across ranks, at pcp 2, 4 and 8, with and
// without a prefix. Where it does not divide, the busiest rank carries more than either
// 1/pcp or its token share: at 100 tokens over 8 ranks it carries 0.1568 of the pairs
// against 0.125 and 0.14 respectively, because a ragged partition hands one rank two short
// chunks from the expensive end.
//
// A replicated request keeps its whole count, since every rank computes all of it. The
// loop below produces that naturally: pcpLocalTokens returning the full length means the
// chunk indices cover the query.
func pcpCausalPairs(sched, prefix, pcp, dcp int) float64 {
	if pcp <= 1 || sched <= 0 {
		return 0
	}
	if pcpLocalTokens(sched, pcp, dcp) == sched {
		// Replicated: every rank attends the whole query, so this is the unsharded count.
		s := float64(sched)
		return 2 * 2 * (s*float64(prefix) + s*s*0.5)
	}
	numChunks := 2 * pcp
	chunk := (sched + numChunks - 1) / numChunks
	// Maximised over the ranks rather than read off one of them. WHICH rank is busiest
	// depends on where the ragged remainder falls -- enumerated over 40,000 random
	// (pcp, sched, prefix) draws it is rank 1 in 85% of cases, rank 0 in 14%, and some
	// other rank in the rest -- so assuming an index is wrong. The arithmetic below is
	// closed-form per rank and pcp is at most the tensor-parallel width, so this is a
	// handful of iterations rather than a walk over the query.
	var most float64
	for rank := 0; rank < pcp; rank++ {
		var pairs float64
		for _, idx := range [...]int{rank, numChunks - 1 - rank} {
			lo := min(idx*chunk, sched)
			hi := min(lo+chunk, sched)
			if n := hi - lo; n > 0 {
				// The keys each query in [lo, hi) reads: prefix + j + 1, summed in
				// closed form. The final -n/2 drops the diagonal's half-token so this
				// matches the unsharded term's CONTINUUM convention (s*c + s^2/2)
				// rather than exceeding it by s/2 -- the same correction
				// windowedCausalFLOPs makes, and for the same reason: without it an
				// inactive split would not reproduce the figure it replaces.
				pairs += float64(n)*float64(prefix+1) +
					float64(lo+hi-1)*float64(n)/2 - float64(n)/2
			}
		}
		if pairs > most {
			most = pairs
		}
	}
	return 2 * 2 * most
}

// dcpShardsKV reports whether decode-context parallelism shards THIS attention kind's
// cache.
//
// Per kind rather than per deployment, and vLLM states each case separately
// (vllm/v1/kv_cache_interface.py at v0.31.0):
//
//   - full attention, MLA and sparse MLA shard. AttentionSpec carries
//     `dcp_sharded: bool = True` (:488) and FullAttentionSpec.max_memory_usage_bytes
//     divides max_model_len by dcp_world_size (:578-583).
//   - a sliding window does NOT, and the engine refuses the combination outright:
//     SlidingWindowSpec.max_memory_usage_bytes asserts dcp == 1 with the message
//     "DCP not support sliding window" (:868-872). ChunkedLocalAttentionSpec likewise
//     divides by nothing.
//   - recurrent state does not, for a reason the engine also states: "Mamba state is
//     replicated across DCP/PCP ranks, never sharded" (:1097-1098). Nothing here has to
//     act on that, because a recurrent layer is priced through RecurrentStateBytes and
//     never reaches the KV term this function gates.
//
// The distinction is load-bearing rather than tidy: gpt-oss-120b ALTERNATES a 128-token
// swa layer kind with a full gqa one, so a deployment-wide divisor would shard half its
// stack against an assertion the engine would not have started under.
//
// An unrecognized kind returns false, which over-prices rather than under-prices. That is
// the same direction resolve.Emits takes for a condition it cannot answer: silently
// dropping a cost is the failure mode worth refusing.
func dcpShardsKV(kind model.AttentionKind) bool {
	switch kind {
	case model.AttentionGQA, model.AttentionMLA, model.AttentionSparseMLA:
		return true
	}
	return false
}

// dcpLocalTokens is how many KV tokens the SLOWEST decode-context-parallel rank holds,
// summed over these requests. It returns -1 when DCP shards nothing here, so a caller
// keeps the unsharded count -- the same sentinel contract selectedKVTokens uses.
//
// THREE PROPERTIES, each of which changes the answer by a large factor if dropped.
//
// IT DIVIDES TOKENS, NOT BYTES PER TOKEN. vLLM shards the cache by token position and
// leaves a page's size alone: max_num_blocks_per_req is
// `cdiv(max_len, block_size * kv_shard_count)` (kv_cache_interface.py:545-550) and
// page_size_bytes never sees dcp. Dividing a per-token byte figure instead would also
// divide PDTransferTime, which DCP does not shard, and would understate a paged capacity
// by up to dcp below block_size*dcp tokens -- 8x at 16 tokens and dcp 8, 2x at 64, exact
// from 128 up.
//
// THE SLOWEST RANK BINDS, NOT THE MEAN. The decode combine is a collective
// (cp_lse_ag_out_rs at vllm/v1/attention/ops/dcp.py:471), so every rank waits for the
// one holding the most tokens. Charging the mean would make a step cheaper than any rank
// can actually deliver.
//
// THERE IS NO REPLICATION FLOOR, which is exactly where this differs from the KV-head
// division next door. price.KVBytesPerToken floors at one head because a head is never
// split across ranks; tokens carry no such constraint and the shards sum to the whole
// context. That is also why DCP is the only axis that shards an MLA cache: a latent
// cache has one head, so no tensor-parallel width reduces it.
//
// THE STRIPING QUESTION, answered rather than left to omission. Tokens are striped across
// ranks in runs of `interleave` (cp_kv_cache_interleave_size, default 1), and the per-rank
// length is
//
//	base = (L / interleave / dcp) * interleave        // integer division
//	local(rank) = base + clip(L - base*dcp - rank*interleave, 0, interleave)
//
// (vllm/v1/attention/backends/utils.py:1143-1156). Rank 0 takes the largest clip, so the
// slowest rank holds base + min(L - base*dcp, interleave). Verified against the engine's
// own per-rank form over 18,000 (dcp, interleave, L) combinations: this equals the maximum
// across ranks exactly, and the shards sum to L exactly. So the interleave contributes a
// BOUNDED ADDITIVE term of at most one run, not a multiplicative one -- at interleave 32,
// dcp 8 and a 1,000-token context it is 128 against ceil(1000/8)=125, under 3%. Bounded is
// not small on a context short against the run: at interleave 64 a 100-token context puts
// 64 tokens on the slowest rank against ceil(100/8)=13, which is why the resolved interleave
// is passed in rather than assumed. At the default interleave of 1 the form reduces to
// ceil(L/dcp).
//
// PER REQUEST, not over the batch total, for the same reason selectedKVTokens is: the
// remainder term is per sequence, so sum-then-shard and shard-then-sum disagree whenever
// a context is not a multiple of interleave*dcp, which in a real batch is the normal case.
func dcpLocalTokens(contexts []int, dcp, interleave int) float64 {
	if dcp <= 1 {
		// Not sharded: one rank holds everything. -1 rather than the unsharded sum, so a
		// caller cannot confuse "no shard" with "a shard that happens to be the whole
		// context".
		return -1
	}
	if interleave < 1 {
		// The engine's own default. Guarded rather than trusted, because a zero here would
		// divide by zero below.
		interleave = 1
	}
	var total float64
	for _, ctx := range contexts {
		if ctx <= 0 {
			continue
		}
		base := ctx / interleave / dcp * interleave
		rest := ctx - base*dcp
		if rest > interleave {
			rest = interleave
		}
		local := base + rest
		if local < 1 {
			// A context shorter than the group still occupies one token on some rank.
			// Unreachable given the arithmetic above (rest is at least 1 when base is 0
			// and ctx is positive), asserted rather than assumed because a zero here
			// would make a decode read free.
			local = 1
		}
		total += float64(local)
	}
	return total
}

// dcpDecodeTokens is the KV token count a decode read charges for THIS layer, after both
// bounds that can narrow it: what a bounded layer selects (selectedKVTokensFor), and the
// shard a decode-context-parallel group holds. It returns -1 when neither applies, so a caller
// keeps the raw context sum -- the sentinel contract selectedKVTokens established.
//
// BOTH BOUNDS ARE APPLIED PER REQUEST, and the order is selection then sharding. Selection
// is a property of the sequence -- which positions this layer looks at -- and sharding a
// property of the layout: which of those positions live on this rank. So a rank reads its
// share of what the layer selected, not its share of the whole context.
//
// PER REQUEST RATHER THAN ON THE BATCH TOTAL, because the shard rounds. The engine keeps a
// per-request vector of local lengths for exactly this reason (dcp_local_seq_lens,
// vllm/v1/attention/backend.py:427-428, built per request at flash_attn.py:884-893 of
// v0.31.0), and sums it in the kernel. Scaling a batch total by one ratio instead agrees
// in the common case and does not in general: searched over 200,000 random (width, top-k,
// compression, context) draws, the largest disagreement is 4.3% -- at a heavily compressed
// sparse layer whose selected count is short enough that each request's rounding matters.
// Small, but it is the kind of error that compounds silently, so this computes the thing
// itself rather than an approximation of it.
func (k *Kernel) dcpDecodeTokens(l *price.PlannedLayer, kind model.AttentionKind,
	contexts []int) float64 {
	bounded := readBounded(l)
	dcp := k.layout.DCP
	shards := dcp > 1 && dcpShardsKV(kind)
	if !bounded && !shards {
		return -1
	}
	// The stripe as resolved: stated, pinned to the block size under NIXL, or the
	// engine's default of 1 (resolve.ResolveDecodeContext).
	interleave := k.decodeContext.Interleave
	var total float64
	for _, ctx := range contexts {
		if ctx <= 0 {
			continue
		}
		read := float64(ctx)
		if bounded {
			read = selectedKVTokensFor(l, ctx)
		}
		if shards {
			// Round before sharding: a rank holds whole cached positions, and the
			// selected count of a compressed layer is already a continuous expectation.
			read = dcpLocalTokens([]int{int(read + 0.5)}, dcp, interleave)
		}
		total += read
	}
	return total
}

// dcpDecodeCollectives prices one layer's decode-context-parallel query gather and combine,
// returning the on-node and cross-node seconds. rows is the query rows that take part, one
// per token: a latent layer's decode tokens (the MQA rows, num_mqa_tokens), or a
// full-attention layer's whole step when any request reads context (see the call site).
//
// WHAT vLLM v0.31.0 LAUNCHES PER DECODE LAYER, by backend and by whether prefill-context
// parallelism is on. Three collectives at most, which is the "3 NCCL calls" its own
// documentation counts for ag_rs (vllm/config/parallel.py:371-380):
//
//	                    query gather            log-sum-exp gather   output combine
//	ag_rs, pcp off      all-gather over DCP     all-gather over DCP  reduce-scatter over DCP
//	a2a,   pcp off      all-gather over DCP     -- packed into one all-to-all over DCP --
//	ag_rs, pcp on       all-gather over TP,     all-gather over DCP  all-reduce over DCP
//	(latent only)       only when dcp = tp*pcp
//
//	query gather, pcp off  MLADCPManager._gather_query, dcp.py:1592-1596; flash_attn.py:1591
//	query gather, pcp on   deepseek_v32/attention.py:560-563 (none when dcp == pcp)
//	LSE gather             _cp_lse_common, dcp.py:458
//	ag_rs combine          cp_lse_ag_out_rs :493, or cp_lse_ag_out_ar :526 under PCP,
//	                       chosen by MLADCPManager._init_combine, dcp.py:1525-1531
//	a2a combine            dcp_a2a_lse_reduce, dcp.py:939-1010
//
// a2a with pcp on is refused at construction (pcp_manager.py:188-194), and so is pcp on a
// non-latent model ("MRV2 PCP currently supports MLA models only", pcp_manager.py:132-133)
// and pcp with dcp on anything but a DSA sparse-MLA layer (mla_attention.py:684-687), so
// none of those reaches here.
//
// THE PAYLOAD IS THE TENSOR THE COLLECTIVE MOVES, which is the convention every collective
// in this kernel prices against: an all-gather's output, a reduce-scatter's input, an
// all-reduce's buffer, an all-to-all's send buffer. A rank's tensor-parallel shard holds
// heads/tp query heads; the DCP query gather widens that to heads*dcp/tp, and under PCP the
// attention runs over heads/tp heads (dcp == pcp) or all heads (dcp == tp*pcp, after the
// tensor-parallel gather). The query crosses at the full head width -- for a latent layer
// kv_lora_rank + qk_rope_head_dim, which the catalog's d_h is -- and the output at the value
// width: kv_lora_rank for a latent layer (MLADCPManager's output_head_dim,
// mla_attention.py:697-708) and the head dimension otherwise. The a2a packs the fp32 LSE
// into two 16-bit lanes beside each output row (_dcp_a2a_lse_pack_dim, dcp.py:604-610).
//
// ACTIVATIONS CROSS AT 16 BITS. The query and the partial output are activations in the
// model's compute dtype, which stays bf16 under an fp8 or fp4 weight format: vLLM's
// quantized linears return out_dtype=x.dtype (vllm/model_executor/kernels/linear/scaled_mm/
// cutlass.py). The one exception is a backend that takes a quantized query alongside a
// quantized cache (query_dtype, mla_attention.py:690-696), which would move the query at
// one byte; this prices it at two, an overstatement of one of the three collectives.
//
// A latent prefill's context is read by its own gather, dcpPrefillContextGathers.
func (k *Kernel) dcpDecodeCollectives(l *price.PlannedLayer,
	rows int) (onNode, crossNode float64) {
	const activationBytes = price.ActivationBytes
	tp := float64(max(k.layout.TP, 1))
	dcp := float64(k.layout.DCP)
	pcpOn := k.layout.PCP > 1
	heads := float64(l.AttnQHeads)
	queryWidth := float64(l.AttnHeadDim)
	outWidth := queryWidth
	if latentAttention(l.AttnKind) && l.AttnKVLoRARank > 0 {
		outWidth = float64(l.AttnKVLoRARank)
	}
	r := float64(rows)

	charge := func(key collKey, bytes float64) {
		if k.crossesNodes(key) {
			crossNode += k.collectiveSeconds(key, bytes*k.spanFor(key))
		} else {
			onNode += k.collectiveSeconds(key, bytes)
		}
	}

	// The heads each rank's attention runs over, after any query gather.
	attnHeads := heads / tp * dcp
	if pcpOn {
		attnHeads = heads / tp
		if k.layout.DCP > k.layout.PCP {
			// dcp == tp*pcp: the query is gathered over the tensor-parallel group first.
			attnHeads = heads
			charge(collKey{Op: model.OpAllGather, Group: price.GroupTP},
				r*attnHeads*queryWidth*activationBytes)
		}
	} else {
		charge(collKey{Op: model.OpAllGather, Group: price.GroupDCP},
			r*attnHeads*queryWidth*activationBytes)
	}

	if k.decodeContext.CommBackend == resolve.DCPAllToAll {
		const lsePack = 2 // an fp32 LSE in two 16-bit lanes
		charge(collKey{Op: model.OpAll2All, Group: price.GroupDCP},
			r*attnHeads*(outWidth+lsePack)*activationBytes)
		return onNode, crossNode
	}
	const lseBytes = 4 // fp32
	charge(collKey{Op: model.OpAllGather, Group: price.GroupDCP}, dcp*r*attnHeads*lseBytes)
	combine := model.OpReduceScatter
	if pcpOn {
		combine = model.OpAllReduce
	}
	charge(collKey{Op: combine, Group: price.GroupDCP}, r*attnHeads*outWidth*activationBytes)
	return onNode, crossNode
}

// pcpPrefillGathers prices one layer's prefill-context all-gathers, returning the on-node
// and cross-node seconds. localTokens is the prefill tokens THIS rank computed.
//
// ONE ALL-GATHER PER TENSOR, and a latent layer gathers two. _gather_prefill_cache_inputs
// launches one all_gather per tensor it is given (pcp.py:31-35), and the MLA cache write
// gives it kv_c_normed and k_pe separately (maybe_gather_mla_latent_cache_inputs,
// pcp.py:56-74; called before the cache update, mla_attention.py:781-800, and on a DSA layer
// deepseek_v32/attention.py:511-529). A layer with a sparse indexer gathers the indexer's
// key as well, in its own launch
// (maybe_gather_indexer_k, pcp.py:77-87, from sparse_attn_indexer.py:440). Each launch pays
// its own floor, which at a prefill chunk's payload is most of the cost.
//
// AT THE MODEL DTYPE, NOT THE CACHE'S. The gather runs before do_kv_cache_update quantizes
// into the cache (mla_attention.py:781-800; deepseek_v32/attention.py:511-529), so an fp8
// cache does not halve it: these are bf16 activations.
//
// SIZED AT THE GATHERED OUTPUT, pcp times what this rank contributes, which is the
// convention every all-gather in this kernel prices against (an all-gather's payload is
// the tensor it produces). An earlier form sized it at the rank's own contribution, which
// understated the bytes by the PCP width.
//
// PCP runs only where every layer is latent attention (resolveContextParallel), so the
// widths here are a latent layer's: kv_lora_rank for kv_c_normed and the remainder of the
// head width for k_pe. A latent node that states no kv_lora_rank is gathered as one tensor
// of its head width.
func (k *Kernel) pcpPrefillGathers(l *price.PlannedLayer,
	localTokens int) (onNode, crossNode float64) {
	const activationBytes = price.ActivationBytes
	rows := float64(k.layout.PCP) * float64(localTokens)
	key := collKey{Op: model.OpAllGather, Group: price.GroupPCP}
	gather := func(width float64) {
		bytes := rows * width * activationBytes
		if k.crossesNodes(key) {
			crossNode += k.collectiveSeconds(key, bytes*k.spanFor(key))
		} else {
			onNode += k.collectiveSeconds(key, bytes)
		}
	}
	if lora := l.AttnKVLoRARank; lora > 0 && lora < l.AttnHeadDim {
		gather(float64(lora))                 // kv_c_normed
		gather(float64(l.AttnHeadDim - lora)) // k_pe
	} else {
		gather(float64(l.AttnHeadDim))
	}
	for _, a := range l.Attentions {
		if !a.HoldsKV && a.KVHeads > 0 && a.HeadDim > 0 {
			gather(float64(a.KVHeads * a.HeadDim)) // the indexer's key
		}
	}
	return onNode, crossNode
}

// dcpPrefillContextGathers prices the all-gathers a decode-context-parallel latent layer
// runs to read a prefill's cached context, returning on-node and cross-node seconds.
//
// The context is sharded across the DCP group like any cached sequence, but an MLA prefill
// attends it with the MHA kernel over the WHOLE context, so each rank gathers its shard and
// the group all-gathers the rest (MLACommonImpl chunked context:
// vllm/model_executor/layers/attention/mla_attention.py:3090-3104 at v0.31.0). It runs in
// workspace-sized chunks, one all-gather per chunk: the workspace holds
// min(max(8 x max_model_len, 4 x max_num_seqs x block_size), 65536) rows
// (determine_chunked_prefill_workspace_size, :2297-2320), and a chunk covers up to that many
// context tokens (:2060-2080). The gathered rows are the latent width at the model dtype --
// the cache is dequantized or upconverted into the workspace first (:3060-3089).
//
// Chunks are packed across requests: plan_mla_context_chunks (:1922-1990) fills each chunk
// with consecutive context-reading prefills until the workspace is full, splitting the
// request that overflows it, and starts a new run at a prefill with no context. So a run
// of prefills whose contexts sum to S pays ceil(S / chunk) gathers, not one or more per
// request.
//
// The runs are formed in the order the engine lays the batch out, which is not the order a
// caller lists it. Model Runner V2 -- v0.31.0's runner unless a configuration needs V1
// (VllmConfig.use_v2_model_runner, vllm/config/vllm.py:701-761), and the only one that runs
// PCP -- sorts requests by (no drafts, not a decode-sized query, scheduled tokens), a stable
// sort (sort_batch_req_ids, vllm/v1/worker/gpu/model_runner.py:2398-2410). So prefills run in
// ascending scheduled-token order, ties in arrival order, and that is the order sorted here.
// The V1 runner orders differently (every context-reading prefill ahead of every fresh one,
// vllm/v1/attention/backends/utils.py:936-945), which would join runs this order splits.
//
// Two simplifications, both toward more launches than counted here: a split is rounded
// down to the cache's block alignment, and under DCP each rank's share is padded to the
// interleave block (:2051-2064), so a chunk can hold slightly less context than the
// workspace and a run can take one launch more than this counts.
func (k *Kernel) dcpPrefillContextGathers(l *price.PlannedLayer,
	prefills []prefillShape) (onNode, crossNode float64) {
	key := collKey{Op: model.OpAllGather, Group: price.GroupDCP}
	width := float64(l.AttnHeadDim) * price.ActivationBytes
	gatherRun := func(context int) {
		if context <= 0 {
			return
		}
		chunks := (context + k.mlaContextChunk - 1) / k.mlaContextChunk
		bytes := float64(context) * width / float64(chunks)
		for c := 0; c < chunks; c++ {
			if k.crossesNodes(key) {
				crossNode += k.collectiveSeconds(key, bytes*k.spanFor(key))
			} else {
				onNode += k.collectiveSeconds(key, bytes)
			}
		}
	}
	ordered := append([]prefillShape(nil), prefills...)
	sort.SliceStable(ordered, func(i, j int) bool {
		return ordered[i].scheduled < ordered[j].scheduled
	})
	run := 0
	for _, p := range ordered {
		prefix := p.computed
		if prefix <= 0 {
			gatherRun(run)
			run = 0
			continue
		}
		run += prefix
	}
	gatherRun(run)
	return onNode, crossNode
}

// prefillShape is one prefill request as the context gather sees it.
type prefillShape struct{ scheduled, computed int }

// moeDPFunnel is how many replica groups' tokens reach one rank's experts.
//
// Attention data parallelism replicates attention across DP ranks but SHARES the expert
// pool: each rank routes its own tokens and then every rank's tokens are gathered before
// the grouped GEMM, so the experts see dp times one rank's token count.
//
// Expert parallelism need not be on: vLLM v0.31.0 shares the MoE across data-parallel
// ranks either way -- with it off it shards every expert tensor-parallel over the dp x pcp
// x tp ranks (flatten_tp_across_dp_and_pcp, vllm/model_executor/layers/fused_moe/
// config.py:1090-1098, 1186-1200; "MoE layers will be sharded according to the product of
// the tensor, prefill-context, and data parallel sizes", vllm/config/parallel.py:134-136)
// and gathers every rank's tokens to it (all2all_utils.py:202-214). So the funnel is dp
// whenever there is an MoE group to gather into.
//
// COVERAGE LIMIT: prefill-context ranks are not counted here. A PCP rank's prefill tokens
// are its own share while its decode rows are replicated across the group, and how the MoE
// gathers that mixture is not modelled.
//
// Returns 1 rather than 0 when DP is unset, so a scenario that omits it prices as it did
// before this term existed.
func (k *Kernel) moeDPFunnel() float64 {
	if k.layout.MoEGroup() <= 1 || k.layout.DP <= 1 {
		return 1
	}
	return float64(k.layout.DP)
}

// hostPerStep returns the host cost of one step.
//
// The graph mode that runs decides how many launches a step makes, and it is chosen per
// batch: under FULL_AND_PIECEWISE -- vLLM v0.31.0's default -- and FULL_DECODE_ONLY a uniform
// decode batch replays one full graph, and every other batch runs piecewise or with no graph
// respectively. Those are each mode's decode and mixed halves (CUDAGraphMode.decode_mode
// and mixed_mode, vllm/config/compilation.py:62-69), which the default Model Runner V2
// builds its graph candidates from (vllm/v1/worker/gpu/cudagraph_utils.py:250-330).
//
//	NONE       every kernel launched eagerly, each at the eager launch cost.
//	PIECEWISE  the graph is split at every attention op, because attention takes a
//	           varying-shape KV cache that a captured graph cannot hold. So a step
//	           replays one segment per split point and runs attention eagerly between
//	           them: 73 launches on a 72-layer model, not one.
//	FULL       one replay for the whole step, which is what a single per-step cost
//	           describes.
//
// Charging one replay for a PIECEWISE step understates its host cost by the layer count.
// On a 72-layer model at the conventional 5-10 microsecond replay cost that is half a
// millisecond against seven microseconds — which matters at decode, where the whole step
// is a few milliseconds.
//
// The split points are vLLM's `_attention_ops` (vllm/config/compilation.py:764-782): one
// per attention or recurrent mixer, and a second on a sparse-MLA layer, whose indexer is
// its own split op (vllm::sparse_attn_indexer). The segment count is the split points plus
// one; see price.Plan.PiecewiseSegments.
//
// EAGER IS PER KERNEL. Running with no graph launches every kernel the step runs, so the
// eager cost scales with the kernel count, not the layer count: an earlier form charged it
// once per layer, which priced cudagraph_mode NONE about four percent above a full graph
// and BELOW piecewise replay on any model with more segments than layers.
//
// And a batch of more tokens than the capture ceiling runs with no graph at all, whatever
// the mode (captureTokens), so a large prefill step is charged eager launches rather than
// replays it does not make. That step is never charged less than the same step captured:
// running a step with no graph cannot be cheaper on the host than replaying its pieces, and
// a price that fell as a request joined the step across the ceiling would say it was. Per
// kernel, eager exceeds the replay on every graph the catalog states; the bound makes it a
// law rather than a property of the coefficients.
func (k *Kernel) hostPerStep(b kernel.Batch) time.Duration { return k.hostPerStepWith(b, 0) }

// hostPerStepWith is hostPerStep for a step that also runs draftKernels drafter launches.
// A drafter with its own graphs (draftHasOwnGraphs) is charged apart from the target's,
// under its own mode and ceiling (draftHost); any other drafter's launches run under the
// target's.
func (k *Kernel) hostPerStepWith(b kernel.Batch, draftKernels int) time.Duration {
	return k.hostPerStepSplit(b, b, draftKernels)
}

// hostPerStepSplit is hostPerStepWith for a step whose target forward runs over b and whose
// drafter runs over drafted, which differ only under adaptive verification (adaptive.go).
func (k *Kernel) hostPerStepSplit(b, drafted kernel.Batch, draftKernels int) time.Duration {
	if draftKernels > 0 && k.draftHasOwnGraphs() {
		return k.targetHost(b, 0) + k.draftHost(drafted, draftKernels)
	}
	return k.targetHost(b, draftKernels)
}

// targetHost is the target forward's launch-and-replay cost, with draftKernels drafter
// launches run under the same mode.
func (k *Kernel) targetHost(b kernel.Batch, draftKernels int) time.Duration {
	mode := k.graphOther
	// An empty step runs no forward; it is charged as the decode loop's idle step.
	if k.decodeGraphed(b) {
		mode = k.graphDecode
	}
	kernels := k.stepKernels(b) + draftKernels
	captured := k.modeHost(mode, kernels)
	// The graph is dispatched at the padded count (gpu_model_runner.py:4407-4413).
	if b.PaddedTokens() > k.captureTokens && mode != graphModeEager {
		// Past every captured size: the engine runs this step with no graph.
		eager := k.modeHost(graphModeEager, kernels)
		if eager < captured {
			eager = captured
		}
		return eager
	}
	return captured
}

// modeHost is one step's launch-and-replay host cost under a graph mode, for a step that
// launches kernels kernels (stepKernels plus the drafter's).
func (k *Kernel) modeHost(mode graphMode, kernels int) time.Duration {
	switch mode {
	case graphModeFull:
		return k.replayPerStep
	case graphModePiecewise:
		// A model on vLLM's breakable CUDA graphs is captured whole with eager regions cut
		// out of it, rather than split at every attention op: piecewise dispatch replays
		// one piece per region between breaks (BreakableCUDAGraphWrapper, "Scoped to
		// PIECEWISE dispatch; FULL cudagraphs (below) are unaffected",
		// vllm/v1/worker/gpu_model_runner.py:5396-5407 at v0.31.0; the default
		// architectures, vllm/config/vllm.py:79-106). The break regions' own launches are
		// dispatch, which launchSeconds charges per kernel in every mode.
		if k.captureBreaks > 0 {
			return time.Duration(float64(k.replayPerStep) * float64(1+k.captureBreaks))
		}
		return time.Duration(float64(k.replayPerStep) * float64(k.plan.PiecewiseSegments()))
	default:
		return time.Duration(float64(k.launchEager) * float64(kernels))
	}
}

// launchSeconds returns the per-kernel dispatch cost a step pays regardless of how much
// work each kernel does, for the target's kernels and draftKernels drafter launches.
//
// This is the term that dominates a single-request decode step, and an earlier version of
// this model omitted it entirely. Two published runs of one 230B MoE model made the case:
// ITL at one concurrent request was 6.02 ms on H200 at 8k context and 6.13 ms on H100 at
// 707 tokens. Those parts differ in memory bandwidth by 1.43x and the contexts by 10x, and
// the measurement barely moved — so the step is dominated by something independent of
// both, which is dispatch. The residual against the rest of the model was 43.7 and 46.0
// microseconds per layer respectively, agreeing to 5% across that variation. Those reports
// are no longer in the corpus; the reasoning is recorded here because it is why the term
// exists, and the term is still checked by TestHostTermIsChargedOnEveryStep.
//
// Charged per KERNEL rather than per layer, because that is the quantity a deeper or
// shallower model scales with: the plan counts each layer's launches from its surviving
// graph nodes. Over roughly 20 launches per MoE layer the implied per-launch cost is about
// 2.3 microseconds, which is the conventional captured-graph dispatch figure the registry
// already cites for a single launch.
//
// IT IS CHARGED UNDER A FULL REPLAY TOO, and that is not a second charge of the replay. A
// FULL replay makes no per-kernel CPU launch: vLLM v0.31.0 calls one graph.replay() for the
// whole step (CudaGraphManager.run_fullgraph, vllm/v1/worker/gpu/cudagraph_utils.py:531-544),
// which replayPerStep prices. What this term carries is the per-node cost the GPU still pays
// inside a replayed graph, the gap between one node's end and the next's start: the registry
// defines and would measure it that way ("the gap between kernel end and next kernel start,
// summed over the step", blis-registry cost-model-host-overheads.yaml,
// host_launch_per_kernel), and backed it out of single-request decode steps that ran FULL
// graphs. Dropping it under FULL would remove the term from the steps it was fitted on.
// Under NONE it is charged beside the eager launch, additively, which is the composition
// DeepSeek-V4.1's model-scoped eager constant was derived on.
//
// It does NOT overlap with device work, and that is the point rather than a simplification:
// a dispatch that has not happened cannot have its kernel running. At large batch the
// device work per kernel dwarfs it and it vanishes into the noise, which is why a model
// fitted only at high concurrency would miss it.
func (k *Kernel) launchSeconds(b kernel.Batch, draftKernels int) float64 {
	return k.launchPerKernel.Seconds() * float64(k.stepKernels(b)+draftKernels)
}

// stepKernels is how many of the target's kernels a step launches: every planned kernel
// when a request in it prefills, and all but the graph's prefill-only ones otherwise
// (price.PlannedLayer.PrefillOnlyKernels). A request prefills on the rule shapeOf uses:
// more scheduled tokens than the batch's decode threshold.
func (k *Kernel) stepKernels(b kernel.Batch) int {
	if stepPrefills(b) {
		return k.plan.TotalKernels
	}
	return k.plan.TotalKernels - k.plan.PrefillOnlyKernels
}

// stepPrefills reports whether a request in the batch prefills, on the rule shapeOf uses.
func stepPrefills(b kernel.Batch) bool {
	for _, r := range b.Reqs {
		if r.Computed+r.Scheduled > 0 && r.Scheduled > b.DecodeThreshold {
			return true
		}
	}
	return false
}

// graphMode is how much of a step a captured graph covers.
type graphMode int

const (
	// graphModeEager captures nothing: every kernel is launched.
	graphModeEager graphMode = iota
	// graphModePiecewise captures between attention boundaries, so a step replays one
	// segment per layer and runs attention eagerly.
	graphModePiecewise
	// graphModeFull captures the whole step as one graph.
	graphModeFull
)

// String is the mode's vLLM name.
func (m graphMode) String() string {
	switch m {
	case graphModePiecewise:
		return "PIECEWISE"
	case graphModeFull:
		return "FULL"
	}
	return "NONE"
}

// TierTime prices one transfer between GPU and an offload tier.
func (k *Kernel) TierTime(tier string, dir kernel.Direction, bytes int64,
	queueDepth int) time.Duration {
	dev, ok := k.tiers[tier]
	if !ok {
		// An unknown tier is not a free transfer. Returning zero would make an offload
		// hit look costless, which is the opposite of what an unmodelled tier means.
		return time.Duration(math.MaxInt64)
	}
	rate := dev.ReadBandwidthMBs
	if dir == kernel.DirectionToTier {
		rate = dev.WriteBandwidthMBs
	}
	rate *= 1e6 // MB/s to bytes per second
	// A CPU tier crosses the host link as well as host memory, and the slower of the two
	// binds. Which one that is depends on the pairing rather than on either alone.
	if k.hostBytesPerSecond > 0 && rate > k.hostBytesPerSecond {
		rate = k.hostBytesPerSecond
	}
	// Storage bandwidth saturates under concurrency. The coefficient behind this is an
	// assumption, so the effect is linear in queue depth beyond the first request rather
	// than a fitted curve.
	if queueDepth > 1 {
		rate /= float64(queueDepth)
	}
	return seconds(price.FloorAndRate(float64(bytes), rate, dev.BaseLatencyUs*1e-6))
}

// PDTransferTime prices moving one request's KV between pools: its demand, run alone.
//
// Computed from PDTransferDemand rather than beside it, so the two cannot drift: the
// blis-schemas contract is that a demand alone costs exactly this price, for every argument.
func (k *Kernel) PDTransferTime(tokens int, from, to kernel.Placement) time.Duration {
	return k.PDTransferDemand(tokens, from, to).AloneTime()
}

// PDTransferDemand is the handoff PDTransferTime prices, separated into payload, floor and
// the links it crosses, so a simulator running several handoffs at once can arbitrate the
// links they share. The kernel prices one transfer; dividing a shared link among several is
// the simulator's work.
//
// Decode-context parallelism deliberately does NOT divide the payload. A prefill pool holds
// the whole cache for a request -- DCP shards the DECODE cache, and vLLM's own wording is
// "Number of ranks that shard the decode KV cache" (vllm/config/parallel.py:359-362 at
// v0.31.0) -- so the bytes that cross between pools are the whole request's, whatever the
// decode pool's dcp width is.
//
// The links mirror the three placements a handoff can have:
//
//	same node                          that node's NVLink domain, at the NVLink rate
//	same rack, one-domain NVLink rack  that rack's NVLink domain, at the NVLink rate
//	otherwise                          the sender's NIC egress and the receiver's NIC
//	                                   ingress, each at the NIC rate
//
// Link IDs are derived from the placement alone, so every kernel of one deployment names a
// physical link the same way: a prefill pool's kernel and a decode pool's kernel describe
// one handoff with the same IDs, and two handoffs that cross one NIC name it alike.
//
// The floor is the expert-group all-to-all floor, which is what PDTransferTime has always
// charged. Whether that is the right floor for a KV handoff is a separate question: changing
// it would move every P/D price, so it is not changed here.
func (k *Kernel) PDTransferDemand(tokens int, from, to kernel.Placement) kernel.TransferDemand {
	// A handoff of no tokens moves nothing and is no transfer: no floor, no links.
	if tokens <= 0 {
		return kernel.TransferDemand{}
	}
	d := kernel.TransferDemand{
		Floor: seconds(k.collectiveFloors[collKey{Op: model.OpAll2All,
			Group: price.GroupExpert}].Seconds()),
	}
	d.Bytes = int64(math.Round(k.pdTransferBytes(tokens)))
	switch {
	case from.Node == to.Node:
		d.Links = []kernel.LinkUse{{
			Link:           kernel.LinkID(fmt.Sprintf("nvlink/node/%d", from.Node)),
			BytesPerSecond: k.nvlinkBytesPerSecond,
		}}
	case k.fabric.RackIsOneDomain && from.Rack == to.Rack:
		// Multi-node NVLink makes a rack one domain, so an in-rack transfer moves at near
		// on-node speed over the NVSwitch fabric rather than either node's NIC. That fabric
		// is non-blocking across the rack: what two handoffs share is a tray's own NVLink
		// ports, not one rack-wide link, so the links are the sender tray's egress and the
		// receiver tray's ingress.
		d.Links = []kernel.LinkUse{
			{Link: kernel.LinkID(fmt.Sprintf("nvlink/egress/node/%d", from.Node)),
				BytesPerSecond: k.nvlinkBytesPerSecond},
			{Link: kernel.LinkID(fmt.Sprintf("nvlink/ingress/node/%d", to.Node)),
				BytesPerSecond: k.nvlinkBytesPerSecond},
		}
	default:
		d.Links = []kernel.LinkUse{
			{Link: kernel.LinkID(fmt.Sprintf("nic/egress/node/%d", from.Node)),
				BytesPerSecond: k.nicBytesPerSecond},
			{Link: kernel.LinkID(fmt.Sprintf("nic/ingress/node/%d", to.Node)),
				BytesPerSecond: k.nicBytesPerSecond},
		}
	}
	return d
}

// pdTransferBytes is the payload of one request's handoff: the inferred caches at their
// per-token width, and the caches a graph states as NIXL moves them (cacheTransferBytes) --
// owned caches only, window rings clipped. It is one rank's: a decode rank pulls its own
// shard, and for a latent cache, which tensor parallelism does not divide, each decode rank
// pulls a whole copy ("since cache is duplicated", vllm/distributed/kv_transfer/
// kv_connector/v1/nixl/tp_mapping.py:88-97 at v0.31.0), so the fabric carries this payload
// once per decode rank.
//
// COVERAGE LIMIT: each pull is priced as one rank's demand at one rank's link rate. That is
// exact when the pools' tensor-parallel widths match. Where the decode pool is wider, several
// decode ranks pull from one prefill rank (tp_mapping.py:97), whose link then carries
// decode_tp / prefill_tp payloads, which a placement that names a node and not a rank
// cannot express.
//
// A sliding-window layer whose cache is inferred is a ring, which NIXL clips as it clips a
// stated window cache (ringTransferBytes), not a cache that grows with the prompt.
func (k *Kernel) pdTransferBytes(tokens int) float64 {
	return float64(tokens)*k.kvBytesPerToken + k.ringTransferBytes(tokens) +
		k.cacheTransferBytes(tokens)
}

// AdmissionOverhead returns host time before a request can be scheduled.
func (k *Kernel) AdmissionOverhead(promptTokens int) time.Duration {
	if promptTokens <= 0 {
		return 0
	}
	return k.admissionPerRequest +
		time.Duration(float64(k.admissionPerToken)*float64(promptTokens))
}

// OutputTokenOverhead returns host time per emitted token.
func (k *Kernel) OutputTokenOverhead() time.Duration { return k.outputTokenCost }

// CompletionOverhead returns fixed host time at request completion.
func (k *Kernel) CompletionOverhead() time.Duration { return k.completionCost }

// Provenance reports every coefficient this kernel used and its evidence.
func (k *Kernel) Provenance() []kernel.CoefficientOrigin { return k.origins }

// Resolved reports the configuration after resolution, including overridden requests.
func (k *Kernel) Resolved() kernel.Resolution { return k.resolution }

// Deployment returns the pool this kernel prices, as the document stated it.
//
// The request, not the resolution: a caller wanting a width or a backend reads Resolved,
// and reads this for the settings resolution does not touch -- the admission settings a
// scheduler sizes itself from. It is the pool New was given at PoolIndex, so a caller
// holding several kernels of one disaggregated deployment needs no index of its own to
// know which pool each prices.
//
// A deep copy, so a caller writing through one of the engine block's pointer fields
// changes its copy and not the kernel. blis-schemas permits returning a shallow copy and
// asking callers not to write; this is not a hot path, and a promise the type enforces is
// worth more than one a reader has to keep.
func (k *Kernel) Deployment() deployment.Pool { return clonePool(k.pool) }

// Engine returns the engine settings the kernel prices with: the caller's decoding of the
// pool's engine block (Inputs.Engine), as a deep copy for the same reason Deployment is one.
// Graph is the model graph this kernel priced: the scenario's graph as served on the chip
// (its architecture overlay resolved), with a drafter named by speculative.draft_model grafted
// on as its speculator. An engine model sizing its KV cache groups reads the same layers, so it
// does not rebuild a second answer to which layers run. A copy the caller may keep; its slices
// are shared, so treat it as read-only.
func (k *Kernel) Graph() model.Graph { return *k.graph }

func (k *Kernel) Engine() enginesettings.Settings { return k.engine.Clone() }

// DecodeContextParallelWidth returns how many ranks shard the decode KV cache.
//
// A consumer sizing its own KV budget needs it and cannot derive it from anything else
// this kernel exposes: it is the only axis that shards a LATENT cache, because
// KVBytesPerToken floors at one KV head and a latent cache has exactly one, so no
// tensor-parallel width reduces it. A simulator that read Resolution's TensorParallelWidth
// alone would size an MLA deployment's cache as if DCP did nothing.
//
// From the resolved layout rather than the deployment document: a Parallelism states a
// request and resolution settles it.
//
// A method on this type rather than a field of kernel.Resolution only because blis-schemas
// v0.3.0's Resolution carries the tensor-, data- and expert-parallel widths and not the two
// context-parallel ones. The argument that put those three in Resolution -- a consumer
// that can read one width and not the others has to re-derive the rest -- applies to these
// two as well, so this method and its companion are interim, pending the same fields
// upstream.
func (k *Kernel) DecodeContextParallelWidth() int { return max(k.layout.DCP, 1) }

// PrefillContextParallelWidth returns how many ranks split a prefill sequence.
//
// The companion to DecodeContextParallelWidth, and the two are genuinely independent: PCP
// splits prefill computation and expands the process world size while leaving the KV cache
// replicated, where DCP shards the cache without expanding the world size -- it reuses the
// tensor-parallel ranks when pcp is 1, and spans the PCP axis or the whole TP x PCP block
// otherwise (vllm/config/parallel.py:359-362 at v0.31.0). A consumer
// sizing a deployment needs both, and neither can be derived from the other.
//
// Interim for the same reason as DecodeContextParallelWidth.
func (k *Kernel) PrefillContextParallelWidth() int { return max(k.layout.PCP, 1) }

// PreSnapshotBytes returns what a rank holds before vLLM measures the memory its
// gpu_memory_utilization budget is drawn from: the collective library's communicator
// reservation and the engine workspace -- NVIDIA's lumped allowance for the CUDA context,
// cuBLAS and allocator slack (blis-registry nccl_communicator_bytes_<N>rank,
// engine_workspace_bytes). vLLM v0.31.0 initializes the device and the distributed
// environment, NCCL included, and only then takes the snapshot both its requested memory
// (total x gpu_memory_utilization) and its non-KV figure are measured against
// (Worker.init_device, vllm/v1/worker/gpu_worker.py:466-493; request_memory,
// vllm/v1/worker/utils.py:539-557; memory_profiling, vllm/utils/mem_utils.py:300-330). So
// these take their room from the part's memory beyond the budget, not from the KV cache,
// and FixedBytes leaves them out of its Total. A consumer checking that the deployment
// starts at all compares gpu_memory_utilization x memory + PreSnapshotBytes with the part.
//
// KNOWN DIVERGENCE: the workspace allowance is one lumped figure, and the part of it that
// arises after the snapshot -- cuBLAS and other library workspaces made at the first
// forward -- is charged against the KV cache by vLLM and is left out here with the rest;
// the registry states no split.
func (k *Kernel) PreSnapshotBytes() int64 { return k.commBytes + k.workspaceBytes }

// CaptureTokens returns the largest step, in tokens, a captured CUDA graph covers: the
// deployment's max_cudagraph_capture_size or cudagraph_capture_sizes as vLLM v0.31.0 resolves
// them, or its default where neither is stated (see Provenance). A step of more tokens runs
// with no graph whatever the mode. Zero when the resolved mode captures nothing.
func (k *Kernel) CaptureTokens() int { return k.captureTokens }

// GraphModes returns the CUDA graph mode the engine runs a uniform decode step under and the
// one it runs every other step under, as vLLM names them -- "FULL", "PIECEWISE" or "NONE" --
// after the overrides Resolved reports. A step above CaptureTokens runs with no graph under
// either.
func (k *Kernel) GraphModes() (decode, other string) {
	return k.graphDecode.String(), k.graphOther.String()
}

// Resource indices for the hot path's fixed array. A map allocation per step would show
// up in a simulator calling this millions of times, so the accumulation is positional and
// converted to the published map once at the end.
const (
	idxSM = iota
	idxHBM
	idxNVLink
	idxNIC
	idxHost
	numResources
)

// summarize fills per from the positional totals and returns the serialized total and the
// busiest resource. Separated from StepTime so both the allocating and the caller-supplied
// forms share one definition of what the summary means.
func summarize(totals *[numResources]float64,
	per map[kernel.Resource]time.Duration) (time.Duration, kernel.Resource) {
	var noOverlap time.Duration
	var largest float64
	bottleneck := kernel.ResourceSM
	for i, elapsed := range totals {
		if elapsed <= 0 {
			continue
		}
		d := seconds(elapsed)
		if per != nil {
			per[resourceForIndex(i)] = d
		}
		noOverlap += d
		if elapsed > largest {
			largest, bottleneck = elapsed, resourceForIndex(i)
		}
	}
	return noOverlap, bottleneck
}

func resourceForIndex(i int) kernel.Resource {
	switch i {
	case idxSM:
		return kernel.ResourceSM
	case idxHBM:
		return kernel.ResourceHBM
	case idxNVLink:
		return kernel.ResourceNVLink
	case idxNIC:
		return kernel.ResourceNIC
	default:
		return kernel.ResourceHost
	}
}

func seconds(s float64) time.Duration {
	if math.IsNaN(s) || s < 0 {
		return 0
	}
	if math.IsInf(s, 1) {
		return time.Duration(math.MaxInt64)
	}
	return time.Duration(s * float64(time.Second))
}

func mustPositive(name string, v float64) error {
	if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
		return fmt.Errorf("%s resolved to %v, which cannot price anything", name, v)
	}
	return nil
}

package latencykernel

import (
	"math"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/vocab"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
)

// stepShape is one step's batch, reduced to what the per-layer pricing reads. It is built
// once per step and priced against the target's layers and, when the deployment
// speculates, against the drafter's.
type stepShape struct {
	// tokens is the token count this rank runs a forward over, raw the batch's own
	// requests' tokens. tokens includes the data-parallel lockstep pad (Batch.PadTokens),
	// which runs every layer's GEMMs, collectives and experts; attnTokensF is tokens
	// without it, the rows attention runs.
	tokens, raw          int
	tokensF, attnTokensF float64
	// spRows is the rows a token-sharded node sees under the model's own sequence
	// parallelism: ceil(tokens / tp), since sp_shard pads the count to a multiple of tp.
	spRows float64
	// causalFLOPs is the unwindowed prefill attention work, decodeKVTokens the context the
	// decode rows read, and decodeQueryCtx those rows times the context each reads.
	causalFLOPs, decodeKVTokens, decodeQueryCtx float64
	prefillChunks                               []chunk
	decodeContexts                              []int
	// decodeRows is each decode request's query rows, aligned with decodeContexts.
	decodeRows []int
	// logitsRows is the rows the LM head runs over: every decode row, and one per prefill
	// request, its last scheduled token (input_batch.logits_indices,
	// vllm/v1/worker/gpu/model_runner.py:1578-1579 at v0.31.0).
	logitsRows                      float64
	prefills                        []prefillShape
	prefillRequests, decodeRequests int
	decodeTokens, prefillTokens     int
	anyContext                      bool
	smDerate                        float64
	// graphed is whether the step replays a captured graph rather than launching eagerly
	// (stepGraphed), which picks DeepEP low-latency's price (deepEPA2A.seconds).
	graphed bool
}

// shapeOf reduces a batch to the step shape.
func (k *Kernel) shapeOf(b kernel.Batch) stepShape {
	tokens := b.Tokens()
	// Attention work depends on each request's own context length, so it is the one term
	// that walks the batch. Everything else multiplies a pre-aggregated scalar.
	//
	// Prefill and decode are separated because they are different regimes with their own
	// measured constants: prefill is compute-bound and quadratic in prompt length, decode is
	// bandwidth-bound in context. A request is in the prefill regime when it schedules more
	// than the batch's decode threshold -- the cost model's batch-region classifier. vLLM's
	// MLA and FlashInfer backends make the same split (split_decodes_and_prefills,
	// vllm/v1/attention/backends/utils.py:799-870 at v0.31.0), each at its own threshold;
	// FlashAttention does not split, and runs one varlen kernel over the mixed batch
	// (flash_attn.py:1450-1476).
	//
	// causalFLOPs halves the naive count because attention attends only to earlier
	// positions. decodeKVTokens is the context those requests read.
	var causalFLOPs, decodeKVTokens float64
	// decodeQueryCtx is the decode rows times the context each reads, which work charged
	// per query token over the context scales with.
	var decodeQueryCtx float64
	// Each prefill chunk's shape is kept so a sliding-window layer can be charged its
	// own bounded pair count. causalFLOPs is the unwindowed total, which is what a
	// full-attention layer pays.
	var prefillChunks []chunk
	// Each decode request's own context, kept for the same reason prefillChunks is: a
	// bounded layer's read is bounded PER REQUEST (min(ctx, window) + min(ctx / ratio,
	// topk); see selectedKVTokensFor), and a bound cannot be applied to a batch-wide sum
	// without changing the answer. min(sum) != sum(min) whenever any request sits on either side of the bound,
	// which in a mixed batch is the normal case.
	var decodeContexts, decodeRows []int
	var prefillRequests, decodeRequests int
	// decodeTokens is the query rows the decode regime runs, which is what crosses in a
	// decode-context combine: one row per scheduled token, so a speculative decode
	// verifying four tokens moves four rows, not one.
	var decodeTokens int
	// prefills is each prefill request's scheduled tokens and already-computed context, which
	// a decode-context-parallel latent layer gathers back before attending (see
	// dcpPrefillContextGathers). anyContext is whether any request in the step reads cached
	// context at all, which decides whether a full-attention DCP layer runs its context
	// pass.
	var prefills []prefillShape
	anyContext := false
	// prefillTokens is the scheduled prefill tokens THIS RANK computes, which the KV
	// gather's payload is sized from. withheldByPCP is how many of the batch's scheduled
	// tokens the prefill split takes off this rank, which is what the step's token count
	// loses -- tracked as a DELTA rather than by re-summing the batch, so a request the
	// loop below skips keeps contributing exactly what it contributed before.
	var prefillTokens, withheldByPCP int
	for i := range b.Reqs {
		r := &b.Reqs[i]
		ctx := r.Computed + r.Scheduled
		if ctx <= 0 {
			continue
		}
		anyContext = anyContext || r.Computed > 0
		if r.Scheduled > b.DecodeThreshold {
			prefillRequests++
			// Attention pairs for a chunk of `Scheduled` tokens resuming on a prefix of
			// `Computed` already-computed ones. Every chunk token attends to the WHOLE
			// prefix -- causal masking does not reduce that, since the prefix is entirely
			// earlier -- and to its causal share within the chunk:
			//
			//	pairs = Scheduled*Computed + Scheduled^2 / 2
			//
			// This read `Scheduled * (Computed+Scheduled) * 0.5`, which halves the
			// prefix term. At Computed == 0 the two agree, so an unchunked prefill
			// cannot reveal it, and every prefill check in this repository was
			// unchunked. FPM's mixed rows are what exposed it: at a 1025-token chunk on
			// a 203,760-token prefix the old count is 1.995x low, and the measured
			// whole-forward latency grows 5.39x across that context sweep where the
			// prediction grew 2.85x.
			sched := float64(r.Scheduled)
			// Prefill-context parallelism splits this chunk across its ranks, and the
			// split is balanced in BOTH quantities the chunk contributes.
			//
			// The causal pairs divide by exactly pcp. That is the zigzag pairing's doing,
			// not an approximation: pairing chunk r with chunk 2*pcp-1-r gives every rank
			// the same pair count, verified equal to 1.000000 of the ideal share with zero
			// spread across the ranks. See pcpLocalTokens for the enumeration.
			//
			// The TOKEN count divides by the busiest rank's share, which carries the
			// ragged remainder when 2*pcp does not divide the chunk. Both are per request,
			// since each prefill is partitioned on its own.
			pairs := 2 * 2 * (sched*float64(r.Computed) + sched*sched*0.5)
			localSched := r.Scheduled
			if local := pcpLocalTokens(r.Scheduled, k.layout.PCP, k.layout.DCP); local >= 0 {
				localSched = local
				// The causal work of the BUSIEST rank, computed from the engine's own
				// partition rather than scaled by a ratio.
				//
				// Where 2*pcp divides the query the zigzag makes every rank's pair count
				// exactly equal, so this is the whole count over pcp. Where it does not,
				// neither 1/pcp nor the token share is right: at 100 tokens over 8 ranks
				// the busiest rank carries 0.1568 of the pairs where 1/pcp is 0.125 and
				// its token share is 0.14. A ragged partition loads one rank with two
				// short chunks from the EXPENSIVE end, and only counting them says by how
				// much.
				//
				// A replicated request -- one the engine hands whole to every rank -- is
				// left at its full count by pcpCausalPairs, since every rank computes all
				// of it.
				pairs = pcpCausalPairs(r.Scheduled, r.Computed,
					k.layout.PCP, k.layout.DCP)
			}
			causalFLOPs += pairs
			prefillTokens += localSched
			withheldByPCP += r.Scheduled - localSched
			// The chunks this rank owns, at their TRUE GLOBAL OFFSETS.
			//
			// A sliding-window layer's cost is positional, not a function of the token
			// count: a query at global offset 100 under a 128-token window reads 101 keys,
			// one at offset 100,000 reads 128. So the windowed term needs where this
			// rank's tokens SIT, not merely how many it has -- and under the zigzag a
			// rank owns two runs, one near the start and one near the end, which cannot
			// be described as a single run at the request's own prefix.
			//
			// Collapsing them to one run at `prefix` charges the cheap early end twice
			// and understates a windowed prefill by up to 7.9x (a 4,096-token chunk on no
			// prefix at pcp 8, window 2,176). The error vanishes once the prefix
			// saturates the window, since every query then reads exactly `window` keys
			// wherever it sits -- which is why a long-context fixture cannot reveal it.
			prefillChunks = k.appendPrefillRuns(prefillChunks,
				r.Scheduled, r.Computed)
			// Every prefill is recorded, a fresh one too: it ends a run of
			// context-reading prefills, which is what decides how they pack into chunks.
			prefills = append(prefills, prefillShape{r.Scheduled, r.Computed})
			continue
		}
		decodeRequests++
		decodeTokens += r.Scheduled
		decodeKVTokens += float64(ctx)
		decodeQueryCtx += float64(r.Scheduled) * float64(ctx)
		decodeContexts = append(decodeContexts, ctx)
		decodeRows = append(decodeRows, r.Scheduled)
		// A decode row is REPLICATED across prefill-context-parallel ranks, not split:
		// _iter_rank_chunks gives every rank chunk_indices = (0,) for a non-prefilling
		// request, so each computes the whole one-token query. PCP divides prefill only,
		// which is why nothing is withheld here.
	}
	// The token count this rank actually runs a forward over: the batch's own count less
	// whatever the prefill split withheld. Subtracting a delta rather than re-summing the
	// loop is deliberate -- the loop skips a request whose context is non-positive, so a
	// re-sum would silently drop tokens that request still contributes to every term
	// derived from the step's token count. That would reprice a batch holding one, which
	// a re-summed draft of this did: a two-token request resuming on a negative prefix
	// priced 11.64 ms against 12.87 ms before.
	//
	// THIS IS THE RANK-LOCAL COUNT, AND THE EFFICIENCY RAMP THEREFORE SEES IT. That is a
	// decision rather than a side effect, and the engine settles it: a prefill-context
	// rank fills its input_ids and positions buffers with only num_local_tokens_padded
	// rows (vllm/v1/worker/gpu/pcp_manager.py:519-532 at v0.31.0) and runs the forward on
	// that, so every GEMM in the layer genuinely sees a shorter m. A narrower GEMM reaches
	// a lower fraction of peak, so a PCP rank is LESS efficient per token than an
	// unsplit one -- measured here at 3.4% above the ideal half at pcp 2 and 10.1% above
	// the ideal quarter at pcp 4, on a 4,096-token chunk.
	//
	// Evaluating the ramp at the batch-wide count instead would credit a split rank with
	// an efficiency its kernel does not reach, and would make PCP look exactly linear when
	// it is not. The argument the ramp takes is unchanged for every deployment that does
	// not split -- this is the same scalar it always was -- so the finding recorded at the
	// ramp's own call site, that it is evaluated at the step's token count rather than a
	// per-term one, still holds: what changed is how many tokens the step HAS.
	tokens -= withheldByPCP
	attnTokensF := float64(tokens)
	// The data-parallel lockstep pad. vLLM runs every rank's forward at the group's
	// largest token count (_post_process_dp_padding, vllm/v1/worker/dp_utils.py:99-111),
	// passed to the forward as num_tokens (vllm/v1/worker/gpu_model_runner.py:4407-4413),
	// so the pad runs the GEMMs, the MoE dispatch and combine and the experts. Attention
	// runs only the scheduled requests' tokens, and the LM head, the vocabulary all-gather
	// and sampling only their logits rows (logits_indices, :2231), so the pad enters no
	// term that walks the requests and no logits row.
	tokens += max(b.PadTokens, 0)

	smBudget := b.SMBudget
	if smBudget <= 0 || k.chip.SMCount == 0 {
		smBudget = k.chip.SMCount
	}
	smDerate := 1.0
	if k.chip.SMCount > 0 && smBudget > 0 && smBudget < k.chip.SMCount {
		// Fewer SMs available means the compute term takes proportionally longer.
		smDerate = float64(k.chip.SMCount) / float64(smBudget)
	}

	tokensF := float64(tokens)
	return stepShape{
		tokens: tokens, raw: b.Tokens(), tokensF: tokensF, attnTokensF: attnTokensF,
		spRows:      math.Ceil(tokensF / k.tp),
		causalFLOPs: causalFLOPs, decodeKVTokens: decodeKVTokens,
		decodeQueryCtx: decodeQueryCtx,
		prefillChunks:  prefillChunks, decodeContexts: decodeContexts, decodeRows: decodeRows, prefills: prefills,
		prefillRequests: prefillRequests, decodeRequests: decodeRequests,
		decodeTokens: decodeTokens, prefillTokens: prefillTokens,
		anyContext: anyContext, smDerate: smDerate,
		logitsRows: float64(decodeTokens + prefillRequests),
		graphed:    k.stepGraphed(b),
	}
}

func (k *Kernel) stepTime(b kernel.Batch,
	into map[kernel.Resource]time.Duration) kernel.StepEstimate {
	if k.adaptive != nil {
		// The target's forward runs the verify budget the engine chose; the drafter, as
		// scheduled (adaptive.go).
		return k.priceStep(k.adaptive.trim(k, b), b, into, true)
	}
	return k.priceStep(b, b, into, true)
}

// priceStep prices one step whose target forward runs over b and whose drafter, when draft is
// set, runs over drafted: the same batch but under adaptive verification, where the verifier
// runs a trimmed one (adaptive.go).
func (k *Kernel) priceStep(b, drafted kernel.Batch,
	into map[kernel.Resource]time.Duration, draft bool) kernel.StepEstimate {
	tokens := b.PaddedTokens()
	if tokens == 0 {
		// An empty step launches no work but still costs the host its per-step overhead.
		// A data-parallel rank with nothing scheduled but a pad to run is not empty: it
		// runs a dummy forward at the pad (_dummy_run, gpu_model_runner.py:5784, 6090).
		host := k.hostPerStep(b)
		per := into
		if per == nil {
			per = make(map[kernel.Resource]time.Duration, 1)
		}
		per[kernel.ResourceHost] = host
		return kernel.StepEstimate{
			// Both edges are the host cost on an empty step: there is no device work for
			// them to disagree about.
			Overlap: host, NoOverlap: host,
			Bottleneck:  kernel.ResourceHost,
			PerResource: per,
			// An empty step prices only the host's per-step cost, so only the entries a step
			// with no tokens reads count; the fit ranges are checked at the empty step.
			Trust: trustOf(k.trust, map[string]float64{
				string(vocab.FitTokens): float64(tokens), string(vocab.FitRequests): 0,
				string(vocab.FitContextTokens): 0}),
		}
	}
	s := k.shapeOf(b)

	// Composed PER LAYER, not once over the whole step.
	//
	// The rule is to partition a step into stages, where a stage is a
	// maximal set of operations that run concurrently; take the max over resources within
	// a stage, and SUM across stages. A layer is a stage: layer N+1 needs layer N's
	// output, so the two cannot overlap however idle a resource is.
	//
	// Applying one max over the whole step instead assumes all 72 layers overlap with
	// each other, which understates a step badly wherever no single resource dominates.
	// At single-request decode nothing dominates — every kernel is small — so that
	// understatement is largest exactly where a latency-sensitive deployment runs.
	//
	// Per-resource totals are still accumulated, because a caller wants to see where the
	// work went even though the step time is not their max.
	var perResource [numResources]float64
	overlapSeconds := k.priceTarget(b, &s, &perResource)
	if k.plan.Embedding.Count > 0 {
		overlapSeconds += k.priceLayers([]price.PlannedLayer{k.plan.Embedding}, nil,
			[]layerKV{{}}, &s, &perResource)
	}
	overlapSeconds += k.priceHead(k.plan.Head, &s, &perResource)

	// The drafter's passes, when the deployment speculates (see priceDraft).
	draftKernels := 0
	if k.speculating && draft {
		var d float64
		d, draftKernels = k.priceDraft(drafted, &s, &perResource)
		overlapSeconds += d
	}

	// The host is additive against everything: the CPU is a resource that cannot overlap
	// with itself, and its work is not hidden behind a stage the way device work is. Per
	// kernel dispatch joins it, for the same reason — a kernel cannot run before it has
	// been dispatched.
	host := k.hostPerStepSplit(b, drafted, draftKernels) + seconds(k.launchSeconds(b, draftKernels))
	perResource[idxHost] = host.Seconds()

	// The breakdown map is the only thing that can allocate here. A caller that supplied
	// one gets no allocation from it; one that did not gets a map sized to the resources
	// actually in play.
	per := into
	if per == nil {
		var live int
		for _, elapsed := range perResource {
			if elapsed > 0 {
				live++
			}
		}
		per = make(map[kernel.Resource]time.Duration, live)
	}
	noOverlap, _ := summarize(&perResource, nil)
	if sched := k.schedulerHost(b, noOverlap); sched > 0 {
		host += sched
		perResource[idxHost] = host.Seconds()
	}
	noOverlap, bottleneck := summarize(&perResource, per)
	overlap := seconds(overlapSeconds) + host

	if overlap > noOverlap {
		// Per-layer composition can exceed the sum of per-resource totals only through
		// rounding, since each stage's max is one of its own terms. Clamp rather than
		// report a band the wrong way round.
		overlap = noOverlap
	}
	// Which edge to believe, for a caller that wants one number: NoOverlap, on measured
	// evidence rather than preference. Over 219 points of NVIDIA's FPM whole-forward
	// dataset -- two models, two parts, five parallelism topologies -- Overlap's SIGNED
	// error is -13.45% against NoOverlap's -3.44%, and NoOverlap is closer on 158 of them.
	// A one-sided error of that size is a missing term rather than scatter, and it matches
	// the -13.10% deficit this kernel shows end to end on an independent serving corpus.
	// The physical reason is PIECEWISE cudagraph mode: attention runs eagerly between
	// captured segments, so per-layer overlap is structurally limited.
	//
	// A concentration-aware choice would refine this -- where one resource holds most of a
	// step, per-stage max IS right, and the single dissenting cell in that evidence is a
	// whole model on two GPUs. Five cells is enough to reject the optimistic edge as a
	// universal default and not enough to fit a blend.
	//
	// schemas v0.2.0 removed StepEstimate.Expected, so the kernel no longer names an edge
	// in the estimate and this paragraph is the only place the choice is recorded. It is
	// kept because it is measured rather than rederivable: a caller holding only Overlap
	// and NoOverlap cannot tell which the evidence favours, and the band is wide enough
	// that guessing picks a different number. A caller that wants one figure should read
	// NoOverlap.
	return kernel.StepEstimate{
		Overlap: overlap, NoOverlap: noOverlap,
		Bottleneck: bottleneck, PerResource: per,
		Trust: trustOf(k.trust, s.inputs()),
	}
}

// priceLayers prices one pass of a stack's layers over a step shape, adding each
// resource's time to per and returning the stage-composed time. fits, where given, are the
// layers' dense GEMM fits in plan order; kv is each layer's cache view.
func (k *Kernel) priceLayers(layers []price.PlannedLayer, fits [][]gemmFit, kv []layerKV,
	s *stepShape, perResource *[numResources]float64) float64 {
	var overlapSeconds float64
	tokensF, smDerate := s.tokensF, s.smDerate
	sp := k.layout.ModelSequenceParallel
	for i := range layers {
		l := &layers[i]
		count := float64(l.Count)

		// One layer's work. Everything here is PER RANK, and which axis divides a term is
		// a property of the term rather than one global divisor: projections shard by
		// tensor parallelism, routed experts by expert-parallel width, and a term divided
		// by the wrong one is wrong by the ratio between them.
		// routedFLOPs is kept apart from flops because the two are priced against
		// different points on the efficiency ramp.
		// attnSeconds is apart from the FLOPs terms because attention is priced by a
		// measured floor-and-rate form rather than from FLOPs.
		var flops, routedFLOPs, weightBytes, elementwise, attnSeconds float64
		// kvBytes is cache read charged at HBM bandwidth where no measured attention form
		// prices it.
		var kvBytes float64
		// denseShapedSeconds is the dense-GEMM time when each GEMM is priced at its own
		// shape's efficiency. It replaces the flops-and-one-ramp path, so exactly one of
		// the two is populated per layer.
		var denseShapedSeconds float64
		// The routed-expert term is kept apart from the dense terms because it does NOT
		// compose with its own compute the way the dense primitives do. See the composition
		// below.
		var routedWeightBytes float64
		// attnSMSeconds is prefill attention, which binds on compute where decode binds on
		// bandwidth. kvFallback records that a decode request had no measured form, so the
		// raw KV read is charged instead.
		var attnSMSeconds float64
		kvFallback := false
		// recurrentSeconds, like attnSeconds, is priced by a measured floor-and-rate form
		// rather than from FLOPs.
		var recurrentSeconds float64
		// shardGatherOn and shardGatherCross are the all-gathers that reassemble a replicated
		// GEMM's columns where the engine splits them at decode (shardsAtDecode).
		var shardGatherOn, shardGatherCross float64

		//
		// Each GEMM is priced at the rate of the format it computes from (formatFit), and
		// divided as the graph lays it out: a column- or row-parallel GEMM over tp, a
		// replicated one not at all (tp: replicated), and under the model's own sequence
		// parallelism a token-sharded one over its ceil(tokens / tp) rows with its whole
		// weight. Where the registry carries no shape-aware ramp for a format, the GEMM is
		// priced on its token ramp, which is what that format always was.
		for gi := range l.DenseGEMMs {
			g := &l.DenseGEMMs[gi]
			// A hyper-connection mixing GEMM is inside the measured mHC pre form, which
			// times the TF32 prenorm GEMM with the kernel (see mhcSeconds).
			if g.MHCFn && k.mhcPre.rate > 0 {
				continue
			}
			var fit gemmFit
			if fits != nil {
				fit = fits[i][gi]
			} else {
				fit = k.fit(g.Format, false)
			}
			// Tensor parallelism splits ONE of the two dimensions, and which one is
			// a property of the projection. The plan classifies it from the graph:
			// a column-parallel GEMM splits its output width, a row-parallel one its
			// reduction. Sharding the wrong axis would price mlp_gate_up's 6144-deep
			// reduction as 768-deep at TP=8, which is the widest GEMM in the layer.
			n, kk := float64(g.N), float64(g.K)
			rows, div := tokensF, k.tp
			sharded := g.Replicated && k.shardsAtDecode(g, s.tokens)
			switch {
			case g.TokenSharded && sp:
				rows, div = s.spRows, 1
			case sharded:
				// The engine computes this rank's n/tp output columns from its slice of the
				// whole weight, so it reads a tp-th of it, and all-gathers the columns
				// (_project_local_columns and gather_columns,
				// vllm/models/muse_spark/nvidia/model.py:158-168, 646-660@0f6b2619).
				n /= k.tp
				on, cross := k.columnGatherSeconds(tokensF * float64(g.N) * price.ActivationBytes)
				shardGatherOn += on
				shardGatherCross += cross
			case g.Replicated:
				div = 1
			case g.ShardN:
				n /= k.tp
			default:
				kk /= k.tp
			}
			weightBytes += g.WeightReadBytes / div
			eff := price.Efficiency(rows, fit.epsMax, fit.mHalf)
			if fit.shapeAware {
				eff = price.ShapeEfficiency(rows, n, kk,
					fit.epsMax, fit.mHalf, fit.nHalf, fit.kHalf)
			}
			if eff <= 0 {
				continue
			}
			denseShapedSeconds += rows * g.FLOPsPerToken / div / (fit.peak * eff)
		}

		if l.ExpertFLOPsPerTokenPerExpert > 0 {
			// A rank computes only the tokens routed to the experts it holds.
			//
			// NOT divided by expertTensorShards, and that is a KNOWN over-charge rather
			// than an oversight. Under pure tensor parallelism localExpertShare is 1 --
			// every rank does hold every expert -- and vLLM shards the intermediate
			// dimension, `intermediate_size_per_partition = intermediate_size // tp_size`
			// (vllm/model_executor/layers/fused_moe/config.py:1321-1323), so a rank's
			// routed work is its slice's. Charging the whole expert over-prices routed
			// compute by the tensor-parallel width on the 400 of 591 InferenceX scenarios
			// that are pure TP. The BYTES do divide by it (routedWeightBytes, below), so the
			// two terms disagree about the same experts.
			//
			// Dividing it was implemented, tested and measured: it improves per-step
			// accuracy against NVIDIA's FPM whole-forward set, mape 27.45% to 23.03% over
			// 9,161 held-out steps, and costs 3.57 points of TPOT mape and 12.29 of TTFT
			// mape end to end. It is retained in this form on the evaluation evidence,
			// because the over-charge cancels another term that is not yet identified --
			// the sixth such cancellation this model is known to rest on. Correcting it
			// alone makes the kernel worse, so it waits for the term it offsets.
			//
			// Re-measured after the activation-width, MLA-cache and shared-MoE corrections,
			// which changed what it offsets: dividing it takes cmd/score's in-scope MAPE
			// from 32.7% to 62.1% -- both scored arms then under-predict by more than half --
			// and cmd/shape's from 11.32% to 11.41%. The cancellation still holds.
			//
			// ATTENTION-DP FUNNEL. With attention data parallelism every DP rank's
			// tokens are concatenated before expert routing, so the grouped GEMM sees
			// the whole replica group's tokens rather than one rank's. vLLM states it
			// for the naive dispatch -- "all DP ranks' tokens are concatenated before
			// routing" (fused_moe/routed_experts_capturer.py) -- and
			// allgather_reducescatter, the default (config/parallel.py:202), gathers every
			// DP rank's tokens before the expert GEMM whether it runs as that dispatch or
			// inside the modular kernel.
			// NVIDIA's own simulator applies the same factor exactly once before its
			// perf lookup (crates/core/src/perfmodel/operators/moe.rs,
			// `num_tokens.saturating_mul(self.attention_dp_size.max(1))`).
			//
			// Omitting it under-prices the routed term by the DP width, which is what
			// FPM's mixed rows measure: on MiniMax-M2.7 h200 the signed error runs
			// -14.9% at dp=1 tep2, -34.0% at dp=2 and -46.7% at dp=4, ordering by dp.
			routedTokens := tokensF * k.moeDPFunnel()
			eg := k.geom(l)
			routedPerRank := routedTokens * float64(l.TopK) * eg.share
			// With expert parallelism off and dp above one, the funnel's dp-fold tokens
			// each reach a 1/dp slice of every expert beyond the tensor-parallel one (the
			// MoE is flattened over dp x tp, see expertTensorShards), so the work per rank
			// divides by dp again. Only the dp part is divided: the tensor-parallel part is
			// the KNOWN over-charge above, kept as recorded, and dividing it here would
			// change every pure tensor-parallel deployment in the corpus.
			dpSlice := 1.0
			if k.layout.ExpertWidth <= 1 {
				dpSlice = k.expertTensorShards / k.tp
				// EXCEPT on a weight-only Marlin ramp, where the over-charge has no
				// measurement to cancel. That ramp was fitted at moe_tp 1 and validated at
				// wider tp with each rank's slice of the expert width (blis-registry
				// scripts/fit_marlin_grouped_ramp.py terms()), which is what vLLM runs; the
				// cancellation the native ramps rest on was measured on them, not on it.
				if k.fit(l.RoutedFormat, true).weightOnly {
					dpSlice = k.expertTensorShards
				}
			}
			// Marlin pads a tensor-parallel slice of each expert's intermediate width to a
			// multiple of 128 (see expertPadding), which is work and weight it runs.
			pad := k.expertPadding(l)
			// The busiest expert-parallel rank gates the layer's combine, so its routed
			// tokens and distinct experts, not the mean rank's, are what the layer pays.
			rankTok, rankExp := k.routingAt(s, routedTokens)
			routedFLOPs += routedPerRank * l.ExpertFLOPsPerTokenPerExpert * k.moeImbalance /
				dpSlice * pad * rankTok
			// Expert weights read: the DISTINCT local experts this step touches, times
			// each one's per-rank bytes. Two things are separate and vLLM keeps them
			// separate too (`fused_moe/config.py`): WHICH experts a rank holds, and how
			// much of each. With expert parallelism a rank owns whole experts; without
			// it, every expert at a tensor-parallel slice. The total is the same and the
			// coverage is not, which a small batch exposes.
			//
			// The count that matters is how many distinct experts the step reaches. At
			// the batch sizes the "all local experts are read at decode" assumption was
			// written for that is all of them; below those it is not, since one token
			// reaches at most top_k.
			// The token count driving coverage is the SAME funnelled count that drives
			// the routed FLOPs above: with attention data parallelism a rank's experts
			// receive every DP rank's tokens, so they are reached by `routedTokens`
			// tokens, not by one rank's. Using the unfunnelled count here while the
			// FLOPs term uses the funnelled one is an internal inconsistency, and it
			// under-reads expert weights exactly where coverage is partial -- at the
			// batch 1-4 prefill rows that make up FPM's attention-DP cells.
			routedWeightBytes += l.ExpertWeightBytesPerExpert / k.expertTensorShards * pad *
				price.ExpertsTouched(int(routedTokens+0.5), eg.total, l.TopK, eg.perRank) *
				rankExp
		}
		if l.SharedExpertFLOPsPerToken > 0 {
			// A shared expert is dense: every token pays it, sharded like any projection --
			// or, under the model's own sequence parallelism where the graph says so, run on
			// the rank's token shard with its whole weight, as DeepSeek-V4.1's is.
			rows, div := tokensF, k.tp
			if l.SharedTokenSharded && sp {
				rows, div = s.spRows, 1
			}
			fit := k.fit(l.SharedFormat, false)
			if eff := price.Efficiency(rows, fit.epsMax, fit.mHalf); eff > 0 {
				denseShapedSeconds += rows * l.SharedExpertFLOPsPerToken / div /
					(fit.peak * eff)
			}
			weightBytes += l.SharedExpertWeightBytes / div
		}

		// Attention is priced by its own measured form, not from FLOPs.
		//
		// A FLOPs-and-bandwidth model of attention is wrong by an order of magnitude in both
		// regimes, and for the same two reasons in each: it charges nothing for the kernel's
		// launch and setup, and it assumes an efficiency the kernel does not reach. Decode is
		// off by 9.7x on L40S to 35.8x on GB200; prefill by 12.9x to 32.4x. So each regime
		// carries its own measured floor and scale, fitted per part.
		//
		//	decode:  floor + kv_bytes / rate            (bandwidth-bound in context)
		//	prefill: floor + causal_flops / (peak * eff * work_scale)
		//
		// A step holding both kinds of request pays both floors on the MLA and FlashInfer
		// backends, which split a mixed batch and launch a kernel per regime, and one on
		// FlashAttention, which runs one kernel over every row (see below). Decode is
		// charged to HBM and prefill to SM, which is what each regime binds on.
		if l.AttnQHeads > 0 {
			// This layer kind's own decode terms, falling back to the part-wide pair.
			floor, rate := k.attentionFloor, k.attentionRate
			if fr, ok := k.attentionByKind[l.AttnKind]; ok {
				floor, rate = fr.floor, fr.rate
			}
			if s.decodeRequests > 0 && rate > 0 {
				// A bounded layer does not read the whole context. Bounded per request,
				// because min(sum) != sum(min): in a batch holding one request below the
				// top-k and one far above it, bounding the batch-wide total gets both
				// wrong.
				read := k.decodeReadBytes(l, kv, i, s)
				// The FLOOR stays outside the shard, deliberately. It is a KERNEL
				// LAUNCH-AND-SETUP cost, paid once per rank per invocation however many
				// tokens that rank holds, and sharding a sequence does not make a launch
				// cheaper.
				//
				// The registry's own numbers are what settle this, and they settle it
				// the opposite way round from the obvious argument. An MLA decode floor
				// is 9.5-14.5us across the parts that carry one -- essentially the same
				// as the part-wide attention floor, because blis-registry's
				// attention_decode_floor_mla IS that figure reused: "this is this part's
				// own measured attention-kernel decode floor, reused for the MLA kind"
				// (cost-model-attention.yaml, attention_decode_floor_mla rationale). The
				// larger module-derived floors, 4.7x-6.2x those, were measured and
				// REJECTED, because they cover the whole MLA block including
				// down-projections this kernel already prices as separate GEMM nodes.
				//
				// So the reason not to divide is not that an MLA setup is unusually
				// expensive -- it is not -- but that a floor of 9.5-14.5us is a launch
				// cost at any shard width. Dividing it by 8 would put it at 1.2-1.8us,
				// below any measured attention floor on any part in the registry, for a
				// kernel that still has to be launched on every rank.
				attnSeconds += floor.Seconds() + read/rate
			} else if s.decodeRequests > 0 {
				// No measured decode form for this part: the KV read is charged below and
				// the floor is lost. Provenance says the entries are missing.
				kvFallback = true
			}
			// A sliding-window layer attends within its window, not over the whole
			// prefix. The window is a per-LAYER property -- gpt-oss-120b alternates a
			// 128-token swa layer with a full gqa one -- so it cannot be applied to the
			// batch-wide causalFLOPs computed before this loop.
			layerCausalFLOPs := s.causalFLOPs
			switch {
			case l.AttnIndexTopK > 0 || l.AttnCompressRatio > 1:
				// A sparse layer's prefill query attends its window and its selection, not
				// its whole prefix: min(ctx, window) + min(ctx / ratio, index_topk) keys, or
				// the whole compressed stream where it selects nothing. DeepSeek-V4.1 runs
				// round_up(index_topk + window, 128) keys a query
				// (vllm/models/deepseek_v41/nvidia/flashmla.py:292 at v0.31.0); the kernel
				// charges the unpadded count.
				layerCausalFLOPs = sparseCausalFLOPs(s.prefillChunks, l.AttnWindow,
					l.AttnIndexTopK, l.AttnCompressRatio)
			case l.AttnWindow > 0:
				layerCausalFLOPs = windowedCausalFLOPs(s.prefillChunks, l.AttnWindow)
			}
			heads := k.attentionHeads(l.AttnQHeads, l.AttnReplicated, l.AttnKind)
			// ONE LAUNCH ON A ONE-KERNEL BACKEND. Where full or windowed attention runs on
			// FlashAttention, a mixed step is one dense varlen kernel over every row
			// (vllm/v1/attention/backends/flash_attn.py:1326-1478 at v0.31.0), so it pays one
			// launch, not one per regime: the smaller of the two floors is taken back out.
			// FlashAttention is v0.31.0's default for causal non-MLA attention everywhere
			// but data-center Blackwell, where FlashInfer, which splits the batch, is
			// (vllm/platforms/cuda.py:154-179); the catalog marks that family by native NVFP4.
			//
			// Not under decode-context parallelism: there FlashAttention runs
			// _forward_with_dcp instead (flash_attn.py:1344-1357), a context pass and a query
			// pass over the whole step and a merge (:1543-1717), so a mixed step whose
			// decodes read context is two launches -- which the two floors already charge.
			if s.decodeRequests > 0 && s.prefillRequests > 0 && !latentAttention(l.AttnKind) &&
				k.layout.DCP <= 1 &&
				k.chip.NVFP4Peak <= 0 && rate > 0 && k.prefillFor(l.AttnKind).scale > 0 {
				if pf := k.prefillFor(l.AttnKind).floor; floor < pf {
					attnSeconds -= floor.Seconds()
				} else {
					attnSMSeconds -= pf.Seconds()
				}
			}
			if pf, ok := k.attentionPrefillByKind[l.AttnKind]; ok && s.prefillRequests > 0 &&
				primaryRecord(l) > 0 {
				// The kind's measured prefill form, over FlashMLA's block reads.
				read := sparsePrefillReadBytes(s.prefillChunks, l.AttnWindow, l.AttnIndexTopK,
					l.AttnCompressRatio, primaryRecord(l), heads)
				attnSMSeconds += pf.seconds(float64(s.prefillRequests),
					float64(s.prefillTokens), read)
			} else if s.prefillRequests > 0 {
				sm, fl := k.prefillAttention(layerCausalFLOPs*heads*float64(l.AttnHeadDim), s, l.AttnKind)
				attnSMSeconds += sm
				flops += fl
			}

			// Secondary attention kernels this layer launches. A block-sparse layer
			// scores which blocks to read before reading them, and that scorer is a
			// separate kernel over a separate, narrow cache: its bytes come from its own
			// head geometry rather than from the engine's KV figure, and its read is
			// bounded by context where the attention it feeds is bounded by the top-k.
			//
			// Charged to HBM like the primary decode term. Omitting it made selection
			// free, which understates every block-sparse layer -- on MiniMax-M3 the
			// dropped read is 0.94x the attention read at 8K context and 15x at 131K.
			for _, a := range l.Attentions {
				if a.HoldsKV {
					continue
				}
				// The scorer's prefill: every prefill query scores every compressed
				// position it can see, over the scorer's own heads (fp8_fp4_mqa_logits,
				// vllm/model_executor/layers/sparse_attn_indexer.py:583-625 at v0.31.0).
				// One matmul, not attention's two, so it is half attention's FLOPs a pair.
				// An earlier form skipped every secondary kernel in a step with no decode,
				// which made the scan free in prefill.
				if pf, ok := k.attentionPrefillByRole[scorerKernel]; ok && s.prefillRequests > 0 &&
					len(a.Caches) > 0 {
					// The role's measured prefill form, over the index keys each query scores.
					read := scorerPrefillReadBytes(s.prefillChunks, a.CompressRatio,
						a.Caches[0].EntryBytes, a.QHeads)
					attnSMSeconds += pf.seconds(0, float64(s.prefillTokens), read)
				} else if s.prefillRequests > 0 {
					// The window bounds the span scored and the ratio how many states that
					// span holds, as in decode below: min(p, window) / ratio a query.
					var pairs float64
					if a.Window > 0 {
						pairs = windowedCausalFLOPs(s.prefillChunks, a.Window) / 2 /
							float64(max(a.CompressRatio, 1))
					} else {
						pairs = sparseCausalFLOPs(s.prefillChunks, 0, 0,
							max(a.CompressRatio, 1)) / 2
					}
					h := k.attentionHeads(a.QHeads, a.Replicated, a.Kind)
					sm, fl := k.prefillAttention(pairs*h*float64(a.HeadDim), s, a.Kind)
					attnSMSeconds += sm
					flops += fl
				}
				if s.decodeRequests <= 0 {
					continue
				}
				floor, rate := k.attentionFloor, k.attentionRate
				if fr, ok := k.attentionByKind[a.Kind]; ok {
					floor, rate = fr.floor, fr.rate
				}
				// A scorer with a measured form of its own, keyed by its role, takes it: it
				// is a different kernel from the attention it feeds.
				if fr, ok := k.attentionByRole[scorerKernel]; ok {
					floor, rate = fr.floor, fr.rate
				}
				// With no measured form at all, the scan is charged as a raw read at HBM
				// bandwidth, as the primary's is (kvFallback), rather than dropped.
				charge := func(read float64) {
					if rate > 0 {
						attnSeconds += floor.Seconds() + read/rate
					} else {
						kvBytes += read
					}
				}
				if len(a.Caches) > 0 {
					// A scorer that states its cache reads that cache's records: every
					// entry of each request's context, ceil(ctx / tokens_per_entry).
					var read float64
					for _, ctx := range s.decodeContexts {
						read += k.statedReadBytes(a.Caches, 0, 0, ctx)
					}
					charge(read)
					continue
				}
				// Scale the primary's inferred per-token width (layerKV.inferred) by this
				// kernel's own head geometry over the primary's. Derived rather than
				// recomputed from the cache dtype so the two cannot drift: that width carries
				// the cache width, the tensor-parallel division and the replication
				// floor that a head is never split across ranks.
				primary := float64(l.AttnKVHeads) * float64(l.AttnHeadDim)
				if primary <= 0 {
					continue
				}
				ratio := float64(a.KVHeads) * float64(a.HeadDim) / primary
				perToken := kv[i].inferred * ratio
				tokens := s.decodeKVTokens
				if a.Window > 0 {
					tokens = math.Min(tokens, float64(a.Window)*float64(s.decodeRequests))
				}
				// A pooled scorer reads cached STATES, not tokens: the kpool indexer
				// compresses index_kpool tokens into one state and scores over those,
				// so the candidate count is the token count over the ratio. This is the
				// scan bound, not a width -- perToken already carries the state's own
				// head geometry. GLM-5.3-Flash pools 4:1, so an unpooled reading charges
				// four times the candidates on the one term in the layer that grows with
				// context. Applied after the window bound because the two compose: the
				// window fixes the span, the ratio how many states it holds.
				if a.CompressRatio > 1 {
					tokens /= float64(a.CompressRatio)
				}
				// This kernel's OWN kind decides whether its cache shards, not the
				// layer's primary: a block-index scorer keeps a separate cache, which is
				// what HoldsKV distinguishes. perToken is deliberately not touched -- it
				// carries head geometry and a cache width, and dividing bytes per token
				// rather than the token count is the error this shard exists to avoid.
				//
				// The scorer's bounds above are already applied to the batch total, so
				// the shard is taken on that total rather than per request. The
				// difference from per-request sharding is the rounding of one token per
				// request, on a term that is a fraction of the layer -- against the
				// primary read, where it is applied per request because that is where it
				// can matter.
				if k.layout.DCP > 1 && dcpShardsKV(a.Kind) {
					tokens = dcpLocalTokens([]int{int(tokens + 0.5)}, k.layout.DCP,
						k.decodeContext.Interleave)
				}
				charge(tokens * perToken)
			}
		}
		// Elementwise traffic divides as its node is laid out: sharded nodes over tp,
		// replicated ones not at all, and a replicated node on the model's token shard over
		// tp again (see price.PlannedLayer).
		elementwise += tokensF*l.ElementwiseBytesPerToken/k.tp +
			tokensF*l.ElementwiseReplicatedBytesPerToken
		if sp {
			elementwise += s.spRows * l.ElementwiseTokenShardedBytesPerToken
		} else {
			elementwise += tokensF * l.ElementwiseTokenShardedBytesPerToken
		}
		mhcSM, mhcBytes := k.mhcSeconds(l, s, sp)
		elementwise = math.Max(elementwise-mhcBytes, 0)
		elementwise += k.contextBytes(l, s)
		gatherBytes, hostSeconds := k.gatherTraffic(l, s)
		elementwise += gatherBytes

		// A recurrent state update, priced by its own measured floor and rate. The cost
		// model charged this at zero, which omits a real term on a hybrid stack: the
		// kernel fires on 48 of Nemotron-3-Ultra's 108 layers and on all 69 of Kimi-K3's.
		//
		// Charged to SM, because a recurrent update is a compute kernel whose cost is
		// launch and sequential arithmetic rather than a bandwidth read. It is sequential
		// in the state dimension, which is why its token rate is low against a GEMM's.
		//
		// A family with no measured entries contributes nothing and Provenance says the
		// entries are missing — the same failure the whole term used to have, but now
		// visible rather than silent.
		//
		// PER PHASE. The registry's pair is fitted on the decode kernels, a sequential
		// per-token scan, and vLLM runs a different kernel for prefill: KDA dispatches
		// FlashKDA or the chunked chunk_kda_with_fused_gate when a step has prefills, and
		// runs fused_recurrent_kda for decode rows only
		// (vllm/models/glm5next/common/kda.py:644-702 at v0.31.0; Kimi-K3's path is the
		// same). The chunked kernel parallelises within a sequence, so charging prefill
		// tokens at the decode scan's rate priced an 8,192-token GLM-5.3-Flash prefill at
		// over a second. See recurrentSeconds.
		if l.RecurrentKind != "" {
			recurrentSeconds += k.recurrentSeconds(l, s)
		}

		// One layer's KV read. Only charged here when attention has no measured form for
		// this part; otherwise the attention term already carries these bytes and adding
		// them again would double-count.
		if l.AttnQHeads > 0 && kvFallback {
			kvBytes += k.decodeReadBytes(l, kv, i, s)
		}

		// The efficiency ramp is evaluated at the STEP's token count for every term,
		// including the grouped GEMM, and that is a finding rather than a simplification.
		//
		// One argument says the grouped GEMM should see a smaller argument — the
		// rows routed to one expert rather than the whole batch — and the reasoning is
		// sound for a LOW-top_k model, one routing to 8 of a few hundred experts. Scored
		// against four published deployments the per-expert argument was worse overall: it
		// improved the two low-top_k arms (31% to 18% MAPE on one, 64% to 41% on Kimi-K3)
		// and ruined Nemotron-3-Ultra (87% to 158%), which routes each token to 22 of 512
		// experts. The aggregate went from 48% to 56%. Two of those four arms have since
		// left the corpus with their model, so the figures are kept as the record of why
		// this choice was made rather than as a re-runnable result.
		//
		// The reason the per-expert argument fails at high top_k is visible in its own
		// arithmetic: at one token routed to 22 experts it gives one row per expert and
		// prices the whole routed term at about 1% of peak, where the measurements say a
		// single-token MoE step reaches about 4%. A real grouped GEMM batches those
		// single-row problems into one kernel, so the machine sees more parallelism than
		// the per-expert view credits.
		//
		// Neither argument is measured. AISimulate's MoE tables are a shape-indexed latency
		// lookup rather than a FLOPs-and-efficiency model, so they state no grouped-GEMM
		// ramp to fit; an attempt to recover one yielded implied efficiencies above peak,
		// because the normalization those tables use is not documented with the data. So
		// this is the better of two assumptions on the evidence available, and the SM term
		// for a high-top_k MoE model is the least certain part of this kernel.
		ramp := price.Efficiency(tokensF, k.epsMax, k.mHalf)
		denseSM := flops/(k.computeFLOPsPerSecond*ramp)*smDerate +
			denseShapedSeconds*smDerate + recurrentSeconds + attnSMSeconds + mhcSM
		denseHBM := (weightBytes+kvBytes+elementwise)/k.hbmBytesPerSecond + attnSeconds +
			hostSeconds

		// The routed-expert grouped GEMM composes its compute and memory as a SUM, where
		// every other primitive in this layer composes as a max.
		//
		// That asymmetry is measured, not assumed. Priced against NVIDIA's own MoE tables --
		// 3,223 shape comparisons over roughly 70 geometries per part, on the fastest kernel
		// lane, expert parallelism off, balanced routing -- the sum fits the term's growth in
		// token count better than the max on every part tested:
		//
		//	                max(c,m)    c+m
		//	h200  fp8         64.04%  55.98%
		//	h100  fp8         53.41%  44.91%
		//	b200  nvfp4      101.49%  79.70%
		//	b300  nvfp4      116.70%  91.95%
		//
		// The mechanism is visible in the shapes: measured MoE latency grows 3.58x from 4 to
		// 256 tokens where the max-composed prediction grows 8.38x. ExpertsTouched rises 30x
		// over that range, so above about 32 tokens the weight-read term dominates the max and
		// drags the prediction up. A real grouped GEMM amortises each expert's weight read
		// against the work done on that expert; a max between the two credits the whole read
		// as free whenever compute exceeds it, and the sum is the closer of the two available
		// approximations.
		//
		// The same test on DENSE GEMM says the opposite -- max 8.75% against sum 15.91% on
		// h200 fp8 -- which is why this is scoped to the routed term rather than applied to
		// the layer. A blanket sum improves the end-to-end score more (10.07% against 13.86%)
		// and is wrong for the dense primitives, so it is not taken.
		var routedSM float64
		if routedFLOPs > 0 {
			fit := k.fit(l.RoutedFormat, true)
			if eff := price.Efficiency(tokensF, fit.epsMax, fit.mHalf); eff > 0 {
				routedSM = routedFLOPs / (fit.peak * eff) * smDerate
			}
		}
		routedHBM := routedWeightBytes / k.hbmBytesPerSecond
		routed := routedSM + routedHBM

		// Each half of the routed cost is charged to the resource that actually does it, so
		// the per-resource breakdown remains a true account of where time went: the compute
		// half to SM, the memory half to HBM. An earlier version charged the WHOLE routed cost
		// to whichever half was larger, which doubled one resource and zeroed the other --
		// four existing tests caught it, including one asserting that withholding SMs cannot
		// move an HBM-bound step and one asserting routed compute falls as the expert group
		// widens.
		//
		// The sum is then realised in the stage composition below rather than here: the
		// routed halves are excluded from the per-layer max and added to it, which is what
		// makes the MoE term compose as a sum while every other term still composes as a max.
		sm := denseSM + routedSM
		hbm := denseHBM + routedHBM

		// Collectives in this layer, each priced against its own floor and rate. Every
		// launch pays its own floor: summing bytes across layers and applying one floor
		// understates a floor-dominated step by the layer count.
		onNode, crossNode := shardGatherOn, shardGatherCross
		for _, c := range l.Collectives {
			bytes := tokensF * c.BytesPerToken
			// The MoE dispatch carries the tokens a rank holds: under the model's own
			// sequence parallelism (vllm/models/deepseek_v41/nvidia/model.py:898-904,
			// sp_shard) or sequence-parallel MoE, that is its ceil(tokens / tp) shard, not
			// the whole step.
			if c.Group == price.GroupExpert && (sp || k.layout.SequenceParallelMoE) {
				bytes = s.spRows * c.BytesPerToken
			}
			if c.RoutedByTopK && l.TopK > 0 && k.routedAll2All() {
				bytes *= float64(l.TopK)
			}
			// Whether a collective leaves the node depends on ITS OWN group, not the
			// deployment's widest: a tensor-parallel reduction across 8 ranks stays on an
			// 8-GPU node even when expert parallelism spans nine of them.
			key := collKey{Op: c.Op, Group: c.Group}
			elapsed := 0.0
			if k.denseDispatch(key) {
				elapsed = k.agrsDispatchSeconds(bytes)
				if k.crossesNodes(key) {
					crossNode += elapsed
				} else {
					onNode += elapsed
				}
			} else if k.crossesNodes(key) {
				elapsed = k.collectiveSeconds(key, bytes*k.spanFor(key))
				crossNode += elapsed
			} else if k.deepEP != nil && key.Op == model.OpAll2All &&
				key.Group == price.GroupExpert {
				elapsed = k.deepEP.secondsFor(bytes, s.graphed)
				onNode += elapsed
			} else {
				elapsed = k.collectiveSeconds(key, bytes)
				onNode += elapsed
			}
		}

		// Decode-context parallelism's own collectives, which no model graph can emit.
		//
		// Sharding a sequence by token means no rank holds the whole context, so each
		// computes a PARTIAL attention output and the group must combine them. That combine
		// is the cost DCP trades for the capacity it buys, and omitting it would make DCP
		// look free in both directions. dcpDecodeCollectives prices exactly the set vLLM
		// v0.31.0 launches for the resolved backend and layout.
		//
		// NOT planned as a PlannedCollective: the plan is per-layer-kind and
		// shape-independent, while whether this fires depends on the batch: its decode rows
		// on a latent layer, and on a full-attention layer whether any request reads
		// context. It accumulates into the same onNode/crossNode totals the graph
		// collectives use, so composition and the per-resource breakdown treat it
		// identically.
		if k.layout.DCP > 1 && dcpShardsKV(l.AttnKind) && l.AttnQHeads > 0 {
			// Which query rows take part in the combine depends on the attention.
			//
			// A latent layer runs its decode rows through the MQA path, where the combine
			// happens, and its prefill rows through the MHA path over a gathered context
			// (below), so only decode rows cross here.
			//
			// A full-attention layer runs ONE context pass over every query row -- decode
			// and prefill alike -- whenever any request reads cached context: it gathers
			// the whole step's query and combines the whole step's output
			// (FlashAttentionImpl._forward_with_dcp, vllm/v1/attention/backends/
			// flash_attn.py:1591-1686 at v0.31.0), and skips both when no request has
			// context (max_dcp_context_kv_len == 0, :1565-1589).
			rows := s.decodeTokens
			if !latentAttention(l.AttnKind) {
				rows = 0
				if s.anyContext {
					rows = s.raw
				}
			}
			if rows > 0 {
				on, cross := k.dcpDecodeCollectives(l, rows)
				onNode += on
				crossNode += cross
			}
			if latentAttention(l.AttnKind) && len(s.prefills) > 0 {
				on, cross := k.dcpPrefillContextGathers(l, s.prefills)
				onNode += on
				crossNode += cross
			}
		}

		// Prefill-context parallelism's own collectives, which no model graph emits either.
		//
		// A PCP rank computes only its share of a prefill, so it writes only its share of
		// the new KV. Every rank must still hold the WHOLE cache, because PCP "does not
		// increase the KV-cache shard count" (vllm/config/parallel.py:131-133 at v0.31.0)
		// -- that is what distinguishes it from DCP. So the ranks all-gather the cache
		// inputs they just computed, on the token dimension, carrying PREFILL tokens only:
		// "Keep replicated decode writes local and gather partitioned prefills"
		// (vllm/v1/attention/ops/pcp.py:11-52), since a replicated decode row is already
		// on every rank. See pcpPrefillGathers for what crosses.
		if s.prefillTokens > 0 && k.layout.PCP > 1 && l.AttnQHeads > 0 {
			on, cross := k.pcpPrefillGathers(l, s.prefillTokens)
			onNode += on
			crossNode += cross
		}

		// Max across resources within the layer, summed over the layers of this kind --
		// except that the routed-expert term composes as a sum rather than a max, for the
		// measured reason given above. So the max is taken over the layer WITHOUT the routed
		// halves, and the routed sum is added to it.
		stage := sm - routedSM
		for _, d := range [...]float64{hbm - routedHBM, onNode, crossNode} {
			if d > stage {
				stage = d
			}
		}
		stage += routed
		overlapSeconds += count * stage

		perResource[idxSM] += count * sm
		perResource[idxHBM] += count * hbm
		perResource[idxNVLink] += count * onNode
		perResource[idxNIC] += count * crossNode
	}
	return overlapSeconds

}

// stepGraphed reports whether a step replays a captured graph: a mode other than NONE for
// the step's kind (a uniform decode batch takes the decode mode, as targetHost does), and
// a padded token count within the capture ceiling, past which the engine runs eagerly
// (gpu_model_runner.py:4407-4413).
func (k *Kernel) stepGraphed(b kernel.Batch) bool {
	mode := k.graphOther
	if k.decodeGraphed(b) {
		mode = k.graphDecode
	}
	return mode != graphModeEager && b.PaddedTokens() <= k.captureTokens
}

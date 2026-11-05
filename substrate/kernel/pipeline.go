package latencykernel

import (
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
)

// Pipeline parallelism: a pp > 1 pool runs the model as pp stages on disjoint ranks, each
// stage tp ranks wide holding a contiguous share of the decoder layers, and hands each
// stage's output to the next over whatever link joins them.
//
// What a stage holds is the engine's split. vLLM gives each of pp ranks
// num_hidden_layers // pp layers and spreads the remainder over all but the last stage,
// counting from the second-to-last backwards (get_pp_indices,
// vllm/distributed/utils.py:126-169@5df63a72); the embedding is on the first stage and the
// final norm, LM head and sampling on the last (PPMissingLayer elsewhere,
// vllm/model_executor/models/utils.py:895-919@5df63a72). The drafter runs on the last
// stage, which holds the sampled tokens.
//
// What crosses a boundary is the model's intermediate tensors, sent from one stage's TP
// rank to the same rank of the next with each rank sending only its 1/tp slice
// (isend_tensor_dict with all_gather_group=tp, vllm/distributed/parallel_state.py:
// 1162-1230@5df63a72, gpu_worker.py:1306-1360): per token, the graph's
// pipeline_boundary_bytes_per_token -- vLLM's standard pair, hidden_states and residual, two
// hidden-width activations, when unstated (make_empty_intermediate_tensors_factory(
// ["hidden_states", "residual"]), vllm/model_executor/models/llama.py:399-401) -- plus every
// auxiliary hidden state a drafter reads that was captured upstream of the boundary and is
// carried to the last stage (Speculator.AuxHiddenStateLayers;
// pack_local_aux_hidden_states and collect_remote_aux_hidden_states,
// vllm/model_executor/models/interfaces.py:1569-1645; relay_aux_hidden_states,
// vllm/v1/worker/gpu/pp_utils.py:102-123), over tp ranks in parallel. A boundary inside one
// node or one NVLink rack crosses NVLink; one between nodes the fabric's NIC, at its
// effective rate. The receiving stage all-gathers the slices back over its TP group before
// it runs (irecv_tensor_dict's _postprocess, parallel_state.py:1332-1356; waited at
// gpu_worker.py:1316), which is charged to that stage as an all-gather over TP.
//
// How the stages compose into what a simulator sees is the contract's (blis-schemas
// kernel.StepEstimate.Stages, engine.Pipelined): each stage is priced as a step of its own
// over its layers, and the step is their sum. A stage's per-step host cost is its own --
// every stage is a separate worker process launching its own kernels -- and the scheduler's
// is the first stage's, where the engine core runs.
//
// A stage's memory is its own: its layers' weights, KV, recurrent state and window rings,
// the embedding on the first, the head, sampler and drafter on the last, and, where the
// deployment drafts with an EAGLE-family method, the target's embedding on the last stage
// too (spec_decode_needs_target_embed, vllm/model_executor/models/utils.py:863-883,
// llama.py:377-382). Every worker sizes its pool from its own free memory and the engine
// keeps the smallest (get_kv_cache_configs step 5, vllm/v1/core/kv_cache_utils.py:
// 2531-2680), so the kernel reports the BINDING stage -- the one whose memory left for KV
// over its own KV per token holds the fewest tokens -- through FixedBytes and the sequence
// sizes, and every stage through PipelineStageMemory.
//
// ASSUMED: VLLM_PP_LAYER_PARTITION is unset. A deployment cannot state it, and with it the
// split is the operator's (get_pp_indices, vllm/distributed/utils.py:142-163).
//
// What is NOT priced: the engine's PP-specific rank placement of a TP group spanning nodes
// (a TP group stays within its stage), and a stage-level split of DP or context
// parallelism, which vLLM does not combine with PP for these models.

// pipelineStage is one stage's share of the model.
type pipelineStage struct {
	// layers are the plan's layer kinds with each count the stage's share; a kind the stage
	// holds none of is absent.
	layers []price.PlannedLayer
	fits   [][]gemmFit
	kv     []layerKV
	// first and last say whether the stage holds the embedding, or the head and drafter.
	first, last bool
	// kernels is the stage's own launch count: its layers' and, on the end stages, the
	// embedding's and head's.
	kernels, prefillOnlyKernels int
	// totalLayers is how many decoder layers the stage holds.
	totalLayers int
	// sendBytesPerToken is what one TP rank sends the next stage a token: its 1/tp slice of
	// the intermediate tensors and the auxiliary states carried past the boundary (zero on
	// the last). recvBytesPerToken is what the stage receives a token, whole: what its TP
	// group all-gathers back from the previous stage's slices (zero on the first).
	sendBytesPerToken, recvBytesPerToken float64
	// crossesNode is whether some rank of the stage sends to a peer on another node.
	crossesNode bool
	// overNIC is whether the boundary send leaves the NVLink domain: it crosses a node and
	// the pool is not inside one NVLink rack domain. The send's rate and the resource it is
	// booked to follow this one test.
	overNIC bool
	// memory is the stage's per-rank memory (liftPipelineMemory).
	memory stageMemory
}

// stageMemory is one stage's per-rank memory: what a worker of that stage holds outside its
// KV pool, and what its KV costs a token. KV per token is the context-growing caches only;
// a window ring and recurrent state are per-sequence bounded terms.
type stageMemory struct {
	weights, eplbRedundant, activation float64
	// kvPerToken is the growing inferred KV a context token costs on this stage's rank, and
	// layerPerToken every inferred KV layer's, window rings included: the page sum the
	// engine's KV cache groups are cut from (PipelineStageMemory.LayerPageBytes).
	kvPerToken, layerPerToken float64
	// layers and kv are the stage's layer kinds and their cache views, which the bounded
	// per-sequence terms (recurrent state, window rings) are summed over.
	layers []price.PlannedLayer
	kv     []layerKV
	// drafter is whether the stage holds the drafter.
	drafter bool
}

// KVLayerPageBytes is one page of blockTokens tokens of every KV-holding layer the rank holds
// -- full attention and sliding window alike, the drafter's while the deployment speculates,
// and the growing caches a graph states -- at the cache dtype that runs, page-quantized. It is
// the sum an engine model's KV cache groups are cut from, where SequenceVariableBytes is only
// the share that grows with context: a window layer holds a ring of its window
// (SequenceFixedBytes), but its page is still one of the pool's. Where no layer is a window it
// equals SequenceVariableBytes(blockTokens). Under pipeline parallelism it is the binding
// rank's (PipelineStageMemory has every stage's).
func (k *Kernel) KVLayerPageBytes(blockTokens int) int64 {
	stated := int64(k.cacheVariableBytes(blockTokens))
	if len(k.stages) > 0 {
		return k.pagedKVBytes(blockTokens, k.stages[k.ppKVStage].memory.layerPerToken) + stated
	}
	var per float64
	for i, l := range k.plan.Layers {
		per += k.layerKV[i].perToken * float64(l.Count)
	}
	if k.speculating {
		per += k.draftLayerBytesPerToken()
	}
	return k.pagedKVBytes(blockTokens, per) + stated
}

// draftLayerBytesPerToken is every inferred drafter layer's KV a token, rings included.
func (k *Kernel) draftLayerBytesPerToken() float64 {
	if k.plan.Draft == nil {
		return 0
	}
	var per float64
	for i, l := range k.plan.Draft.Layers {
		if i < len(k.draftLayerKV) {
			per += k.draftLayerKV[i].perToken * float64(l.Count)
		}
	}
	return per
}

// rankMemoryView is the memory view the sequence-size methods read: the binding stage's under
// pipeline parallelism, the whole model's otherwise.
type rankMemoryView struct {
	layers     []price.PlannedLayer
	kv         []layerKV
	kvPerToken float64
	drafter    bool
}

// rankMemory is the view of the rank whose pool binds.
func (k *Kernel) rankMemory() rankMemoryView {
	if len(k.stages) > 0 {
		m := k.stages[k.ppKVStage].memory
		return rankMemoryView{layers: m.layers, kv: m.kv, kvPerToken: m.kvPerToken,
			drafter: m.drafter}
	}
	per := k.kvBytesPerToken
	if k.speculating {
		per += k.draftKVBytesPerToken
	}
	return rankMemoryView{layers: k.plan.Layers, kv: k.layerKV, kvPerToken: per, drafter: true}
}

// ppIndices is get_pp_indices (vllm/distributed/utils.py:126-169@5df63a72) without
// VLLM_PP_LAYER_PARTITION, which a deployment cannot state: each stage's [start, end).
func ppIndices(layers, pp int) [][2]int {
	parts := make([]int, pp)
	for i := range parts {
		parts[i] = layers / pp
	}
	if rem := layers % pp; rem > 0 {
		for i := 2; i < rem+2; i++ {
			parts[pp-i]++
		}
	}
	out := make([][2]int, pp)
	start := 0
	for i, n := range parts {
		out[i] = [2]int{start, start + n}
		start += n
	}
	return out
}

// liftPipeline splits the plan into stages, once. Called after lift, so every layer kind's
// fits and KV are resolved.
func (k *Kernel) liftPipeline(g *model.Graph) error {
	pp := max(k.layout.PP, 1)
	if pp <= 1 {
		return nil
	}
	// A cache a graph states -- the target's or its drafter's -- is sized per layer kind over
	// the whole stack (cacheTerms) and grouped into one pool over all of them, not per stage,
	// so a pipeline split of it would charge each rank every stage's cache. Refused rather
	// than overstated. COVERAGE LIMIT, not an engine refusal: vLLM serves DeepSeek-V4 and
	// V4.1, whose graphs state their caches, under PP (their make_empty_intermediate_tensors,
	// vllm/models/deepseek_v4/nvidia/model.py:1406-1424, deepseek_v41/nvidia/model.py:
	// 885-910@v0.32.0).
	if len(k.caches) > 0 {
		return fmt.Errorf("pipeline parallelism (pp %d) on a model whose graph states its "+
			"caches is not priced: the stated caches are sized over the whole stack, not "+
			"per stage", pp)
	}
	seq := g.Stack.Expand()
	// Not an engine refusal: get_pp_indices gives a stage no layers when there
	// are fewer layers than stages (vllm/distributed/utils.py:152-163@v0.32.0) and vLLM
	// checks nothing there, but a stage of no layers is not one this pricing describes.
	if len(seq) < pp {
		return fmt.Errorf("pipeline parallelism (pp %d) over %d decoder layers leaves a stage "+
			"with none, which is not priced", pp, len(seq))
	}
	boundary := float64(g.Global.PipelineBoundaryBytesPerToken)
	if boundary <= 0 {
		// vLLM's standard intermediate tensors: hidden_states and residual, each one
		// hidden-width activation (llama.py:399-401@v0.32.0, and deepseek_v2.py,
		// qwen3_moe.py, glm4_moe.py, minimax_m2.py, nemotron_h.py alike).
		boundary = 2 * float64(g.Global.HiddenSize) * price.ActivationBytes
	}
	aux, auxBytes, err := k.auxHiddenStates(g, len(seq))
	if err != nil {
		return err
	}
	// The stage's ranks are pp-major consecutive blocks of tp (rank = pp_rank x tp + tp_rank
	// within a DP rank, parallel_state.py:2045-2060@5df63a72), and TP rank r of one stage
	// sends to TP rank r of the next. A boundary crosses a node when some rank and its peer
	// sit on different nodes, which for a tp that does not divide the node is not every
	// rank's case: tp 6 on 8-GPU nodes puts stage 0's ranks 2-5 on node 0 and their peers
	// 8-11 on node 1, but ranks 0 and 1 beside theirs.
	tp := max(k.layout.TP, 1)
	gpn := k.layout.GPUsPerNode
	crosses := func(s int) bool {
		if gpn <= 0 {
			return false
		}
		for r := range tp {
			if (s*tp+r)/gpn != ((s+1)*tp+r)/gpn {
				return true
			}
		}
		return false
	}
	k.stages = make([]pipelineStage, pp)
	spans := ppIndices(len(seq), pp)
	for s, span := range spans {
		counts := map[string]int{}
		for _, id := range seq[span[0]:span[1]] {
			counts[id]++
		}
		st := &k.stages[s]
		st.first, st.last = s == 0, s == pp-1
		st.totalLayers = span[1] - span[0]
		for i, l := range k.plan.Layers {
			n := counts[l.ID]
			if n == 0 {
				continue
			}
			l.Count = n
			st.layers = append(st.layers, l)
			st.fits = append(st.fits, k.layerFits[i])
			st.kv = append(st.kv, k.layerKV[i])
			per := l.Kernels
			st.kernels += per * n
			st.prefillOnlyKernels += l.PrefillOnlyKernels * n
		}
		if st.first && k.plan.Embedding.Count > 0 {
			st.kernels += k.plan.Embedding.Kernels
			st.prefillOnlyKernels += k.plan.Embedding.PrefillOnlyKernels
		}
		if st.last {
			st.kernels += k.plan.Head.Kernels
			st.prefillOnlyKernels += k.plan.Head.PrefillOnlyKernels
		} else {
			whole := boundary + auxCarriedBytes(aux, span, auxBytes,
				float64(g.Global.HiddenSize)*price.ActivationBytes)
			st.sendBytesPerToken = whole / float64(tp)
			k.stages[s+1].recvBytesPerToken = whole
			st.crossesNode = crosses(s)
			st.overNIC = st.crossesNode && (!k.fabric.RackIsOneDomain || !k.sameRack())
		}
	}
	k.liftPipelineMemory()
	k.assume("pipeline_stages", strconv.Itoa(pp),
		"each stage is priced as its own step over its share of the layers (get_pp_indices, "+
			"vllm/distributed/utils.py:126-169@5df63a72), the embedding on the first, the head "+
			"and drafter on the last, plus its boundary send and the receiving stage's TP "+
			"all-gather; each stage's per-rank memory is its own, and the rank whose pool "+
			"holds the fewest tokens binds")
	k.assume("pp_layer_partition", "unset",
		"VLLM_PP_LAYER_PARTITION is taken as unset, so the stages split the layers as "+
			"get_pp_indices does by default (vllm/distributed/utils.py:142-169); a deployment "+
			"cannot state the variable")
	return nil
}

// auxHiddenStates is the capture ids of the auxiliary hidden states the drafter reads, and one
// state's bytes a token, where the deployment drafts with a speculator that states them.
//
// The engine captures them on the stage that holds the layer and carries them in the
// intermediate tensors to the last stage (EagleModelMixin, interfaces.py:1569-1645@v0.32.0;
// v2 model runner, vllm/v1/worker/gpu/model_runner.py:405-412), each a hidden-width tensor
// at the activation dtype (reserve_aux_intermediate_tensor_slots,
// eagle3_utils.py:62-95) unless the graph states a wider one.
func (k *Kernel) auxHiddenStates(g *model.Graph, layers int) ([]int, float64, error) {
	sp := g.Speculator
	if !k.speculating || sp == nil || len(sp.AuxHiddenStateLayers) == 0 {
		return nil, 0, nil
	}
	for _, id := range sp.AuxHiddenStateLayers {
		if id < 0 || id > layers {
			return nil, 0, fmt.Errorf("the speculator reads the auxiliary hidden state of "+
				"target layer %d, outside the target's %d layers", id, layers)
		}
	}
	b := float64(sp.AuxHiddenStateBytesPerToken)
	if b <= 0 {
		b = float64(g.Global.HiddenSize) * price.ActivationBytes
	}
	return sp.AuxHiddenStateLayers, b, nil
}

// auxCarriedBytes is the auxiliary hidden states a stage spanning layers [span[0], span[1])
// sends the next a token: every state captured at or before its boundary (a capture id L is
// the state entering layer L, so one at the boundary's own index is captured upstream,
// bisect_right in _aux_slot_base, vllm/model_executor/models/interfaces.py:1595@v0.32.0).
// One captured on this stage crosses at own, the width the graph states; one captured on an
// earlier stage is relayed out of the receive slot the engine allocated at the model dtype,
// one hidden-width activation, act (reserve_aux_intermediate_tensor_slots,
// vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:89-95; the slots sized at
// model_config.dtype, gpu/model_runner.py:537-541, and copied into at :1980; relayed as
// received, gpu/pp_utils.py:109-123). The two differ only where the graph states a wider state
// than an activation, and only from the third stage on.
func auxCarriedBytes(aux []int, span [2]int, own, act float64) float64 {
	var b float64
	for _, id := range aux {
		switch {
		case id > span[1]:
		case id > span[0] || span[0] == 0:
			b += own
		default:
			b += act
		}
	}
	return b
}

// pipelineSendSeconds prices one stage's boundary send for a step of tokens: the slice
// each TP rank sends to its peer in the next stage, all tp in parallel over their own
// links, floor plus bytes over the link's rate. The floor is the point-to-point
// collective floor the registry measures (the all-to-all's, a pair exchange).
func (k *Kernel) pipelineSendSeconds(st *pipelineStage, tokens float64) float64 {
	if st.sendBytesPerToken <= 0 || tokens <= 0 {
		return 0
	}
	rate := k.nvlinkBytesPerSecond
	if st.overNIC {
		rate = k.nicBytesPerSecond
	}
	floor := k.collectiveFloors[collKey{Op: model.OpAll2All, Group: price.GroupExpert}].Seconds()
	return floor + tokens*st.sendBytesPerToken/rate
}

// pipelineRecvGatherSeconds prices the receiving stage's reassembly of the previous stage's
// slices: an all-gather over its TP group of the whole intermediate tensors
// (irecv_tensor_dict's _postprocess, all_gather_group.all_gather(slice_tensor, dim=0),
// vllm/distributed/parallel_state.py:1332-1356@5df63a72, run when the stage waits on its
// inputs, gpu_worker.py:1316, AsyncIntermediateTensors.wait_for_comm :174-183). It returns the
// on-node and cross-node seconds, as a TP collective is booked.
func (k *Kernel) pipelineRecvGatherSeconds(st *pipelineStage, tokens float64) (onNode, crossNode float64) {
	if st.recvBytesPerToken <= 0 || tokens <= 0 || k.layout.TP <= 1 {
		return 0, 0
	}
	key := collKey{Op: model.OpAllGather, Group: price.GroupTP}
	bytes := tokens * st.recvBytesPerToken
	if k.crossesNodes(key) {
		return 0, k.collectiveSeconds(key, bytes*k.spanFor(key))
	}
	return k.collectiveSeconds(key, bytes), 0
}

// sameRack reports whether every stage sits in one NVLink rack domain.
func (k *Kernel) sameRack() bool {
	rack := k.layout.GPUsPerRack // the resolved rack, which is the chip's
	if rack <= 0 {
		return false
	}
	pp := max(k.layout.PP, 1)
	return pp*max(k.layout.TP, 1) <= rack
}

// stagedStepTime prices a pipeline-parallel step: each stage over its layers, with its own
// host cost and boundary send, and the step as their sum (StepEstimate.Stages).
func (k *Kernel) stagedStepTime(b kernel.Batch,
	into map[kernel.Resource]time.Duration) kernel.StepEstimate {
	s := k.shapeOf(b)
	var total [numResources]float64
	stages := make([]kernel.StageEstimate, len(k.stages))
	var overlap, noOverlap time.Duration
	for i := range k.stages {
		st := &k.stages[i]
		var per [numResources]float64
		o := k.priceLayers(st.layers, st.fits, st.kv, &s, &per)
		if st.first && k.plan.Embedding.Count > 0 {
			o += k.priceLayers([]price.PlannedLayer{k.plan.Embedding}, nil, []layerKV{{}}, &s, &per)
		}
		draftKernels := 0
		if st.last {
			o += k.priceHead(k.plan.Head, &s, &per)
			if k.speculating {
				var draft float64
				draft, draftKernels = k.priceDraft(b, &s, &per)
				o += draft
			}
		}
		send := k.pipelineSendSeconds(st, s.tokensF)
		if st.overNIC {
			per[idxNIC] += send
		} else {
			per[idxNVLink] += send
		}
		o += send
		on, cross := k.pipelineRecvGatherSeconds(st, s.tokensF)
		per[idxNVLink] += on
		per[idxNIC] += cross
		o += on + cross
		host := k.stageHost(b, st, draftKernels)
		per[idxHost] = host.Seconds()
		stageNo, _ := summarize(&per, nil)
		if st.first {
			if sched := k.schedulerHost(b, stageNo); sched > 0 {
				host += sched
				per[idxHost] = host.Seconds()
				stageNo, _ = summarize(&per, nil)
			}
		}
		stageOv := seconds(o) + host
		if stageOv > stageNo {
			stageOv = stageNo
		}
		stages[i] = kernel.StageEstimate{Overlap: stageOv, NoOverlap: stageNo}
		overlap += stageOv
		noOverlap += stageNo
		for r := range total {
			total[r] += per[r]
		}
	}
	per := into
	if per == nil {
		per = make(map[kernel.Resource]time.Duration, numResources)
	}
	_, bottleneck := summarize(&total, per)
	return kernel.StepEstimate{Overlap: overlap, NoOverlap: noOverlap, Bottleneck: bottleneck,
		PerResource: per, Stages: stages}
}

// stageHost is one stage's launch-and-replay cost, which every stage's worker pays for its
// own kernels: the target's host form (targetHost, launchSeconds) over the stage's launches.
// A piecewise-captured stage replays one segment per layer it holds; the drafter, on the
// last stage, is charged as the single-stage step charges it.
func (k *Kernel) stageHost(b kernel.Batch, st *pipelineStage, draftKernels int) time.Duration {
	mode := k.graphOther
	if k.decodeGraphed(b) {
		mode = k.graphDecode
	}
	kernels := st.kernels
	if !stepPrefills(b) {
		kernels -= st.prefillOnlyKernels
	}
	ownGraphs := draftKernels > 0 && k.draftHasOwnGraphs()
	modeKernels := kernels
	if !ownGraphs {
		modeKernels += draftKernels
	}
	replay := func(m graphMode) time.Duration {
		switch m {
		case graphModeFull:
			return k.replayPerStep
		case graphModePiecewise:
			// The stage's own pieces: a stage's worker compiles and captures only the layers
			// it holds, so its replays are its layers' segments, or its breaks' regions under
			// a breakable capture, not the whole model's.
			if k.breakableCapture {
				return time.Duration(float64(k.replayPerStep) *
					float64(1+k.captureBreaksOf(st.layers)))
			}
			return time.Duration(float64(k.replayPerStep) *
				float64(price.PiecewiseSegmentsOf(st.totalLayers, st.layers)))
		}
		return time.Duration(float64(k.launchEager) * float64(modeKernels))
	}
	host := replay(mode)
	if b.PaddedTokens() > k.captureTokens && mode != graphModeEager {
		if eager := replay(graphModeEager); eager > host {
			host = eager
		}
	}
	if ownGraphs {
		host += k.draftHost(b, draftKernels)
	}
	return host + seconds(k.launchPerKernel.Seconds()*float64(kernels+draftKernels))
}

// liftPipelineMemory sets each stage's per-rank memory and picks the binding stage.
//
// Each worker of vLLM's pipeline loads only its stage's modules (PPMissingLayer elsewhere,
// make_layers, vllm/model_executor/models/utils.py:895-919@5df63a72), profiles its own
// memory, and sizes its own KV pool over the global KV cache groups projected onto its layers
// (_project_kv_cache_groups_to_worker, kv_cache_utils.py:2486-2528); the engine then keeps the
// smallest pool (get_kv_cache_configs step 5, :2660-2680). So a stage's memory is its own:
//
//   - weights: its layers'; the embedding on the first (stated, or inferred as a vocab x
//     hidden table); the head and drafter on the last, and the target's embedding there too
//     when the drafter is an EAGLE-family one that reads it under PP
//     (spec_decode_needs_target_embed, vllm/model_executor/models/utils.py:863-883, built at
//     llama.py:377-382, deepseek_v4/nvidia/model.py:1330, kimi_k3/nvidia/model.py:1143,
//     minimax_m3/nvidia/model.py:814);
//   - EPLB's redundant replicas, over its own MoE layers;
//   - activation: every stage runs the profile forward at max_num_batched_tokens; only the
//     last runs the dummy sampler (vllm/v1/worker/gpu/model_runner.py:1010-1015);
//   - KV a token: its own layers' growing caches, the drafter's on the last.
//
// The binding stage is the one whose pool holds the fewest blocks: the memory left beside its
// fixed bytes over its KV a block. The budget is the part's memory times the deployment's
// gpu_memory_utilization (requested_memory, vllm/v1/worker/gpu_worker.py:535-540@v0.32.0),
// and where stages trade fixed bytes against KV a token the binding one depends on it, so the
// stages are compared at the deployment's value, at vLLM's default (defaultMemoryUtilization)
// only where the block states none, which is recorded. An engine model that sizes each stage's
// pool at its own budget reads PipelineStageMemory.
func (k *Kernel) liftPipelineMemory() {
	embedding := k.embeddingResidentBytes(k.graph)
	targetEmbedOnLast := k.speculating && k.plan.Draft != nil &&
		needsTargetEmbed(k.plan.Draft.Method) && len(k.stages) > 1
	for i := range k.stages {
		st := &k.stages[i]
		m := stageMemory{layers: st.layers, kv: st.kv, drafter: st.last}
		for j, l := range st.layers {
			m.weights += float64(l.Count) * k.residentBytes(l)
			m.layerPerToken += st.kv[j].perToken * float64(l.Count)
			if st.kv[j].window == 0 {
				m.kvPerToken += st.kv[j].perToken * float64(l.Count)
			}
		}
		m.eplbRedundant = k.eplbRedundantBytes(st.layers)
		if st.first || (st.last && targetEmbedOnLast) {
			m.weights += embedding
		}
		if st.last {
			m.weights += k.residentBytes(k.plan.Head) + k.draftResidentBytes()
			if k.speculating {
				m.kvPerToken += k.draftKVBytesPerToken
				m.layerPerToken += k.draftLayerBytesPerToken()
			}
		}
		st.memory = m
	}
	k.bindStage()
}

// bindStage picks the binding pipeline stage from the stages' memory (liftPipelineMemory).
func (k *Kernel) bindStage() {
	util := k.engine.GPUMemoryUtilization
	if util <= 0 {
		util = defaultMemoryUtilization
		k.assume("pipeline_binding_utilization", strconv.FormatFloat(util, 'g', -1, 64),
			"the deployment states no gpu_memory_utilization, so the pipeline stages' pools are "+
				"compared at vLLM's default (vllm/config/cache.py:102@v0.32.0)")
	}
	part := k.chip.MemoryGiB * (1 << 30) * util
	best, binding := math.Inf(1), 0
	for i := range k.stages {
		st := &k.stages[i]
		act := k.activationBytes(k.graph, st.last)
		st.memory.activation = act
		other := act + float64(k.captureBytes) + float64(k.deepEPBufferBytes(k.graph))
		left := part - st.memory.weights - other
		tokens := math.Inf(1)
		if st.memory.kvPerToken > 0 {
			tokens = left / st.memory.kvPerToken
		} else if left < 0 {
			tokens = left
		}
		// Ties go to the stage with more fixed bytes, which a smaller budget binds first.
		if tokens < best || (tokens == best && st.memory.weights > k.stages[binding].memory.weights) {
			best, binding = tokens, i
		}
	}
	if math.IsInf(best, 1) {
		// No stage holds growing KV (an all-window or recurrent model): the stage with the
		// most fixed bytes binds.
		for i := range k.stages {
			if k.stages[i].memory.weights > k.stages[binding].memory.weights {
				binding = i
			}
		}
	}
	k.ppKVStage = binding
	k.ppKVBytesPerToken = k.stages[binding].memory.kvPerToken
	k.assume("pipeline_binding_stage", strconv.Itoa(binding),
		"FixedBytes and the sequence sizes are this pipeline rank's: of the stages' pools, "+
			"each its own budget at the deployment's gpu_memory_utilization less its own fixed "+
			"bytes over its own KV a token, it holds the fewest tokens "+
			"(get_kv_cache_configs, vllm/v1/core/kv_cache_utils.py:2650-2680@v0.32.0); "+
			"PipelineStageMemory reports every stage")
}

// defaultMemoryUtilization is vLLM's default gpu_memory_utilization (CacheConfig,
// vllm/config/cache.py:102@v0.32.0), at which the kernel picks the binding pipeline stage when
// the deployment states none.
const defaultMemoryUtilization = 0.92

// needsTargetEmbed reports whether a draft method's drafter reads the target's input
// embedding on the last pipeline stage, which the target then builds there:
// spec_decode_needs_target_embed's methods (vllm/model_executor/models/utils.py:863-883@
// v0.32.0), MTP excluded as by default.
func needsTargetEmbed(method string) bool {
	switch method {
	case "eagle", "eagle3", "dflash", "dspark":
		return true
	}
	return false
}

// StageMemory is one pipeline rank's memory: what the rank holds outside its KV pool, and
// what a block of its KV pool costs. It is what an engine model sizing each stage's pool
// separately reads, as vLLM sizes each worker's (get_kv_cache_configs, kv_cache_utils.py:
// 2531-2680@5df63a72): a stage's pool is (budget - Fixed.Total()) over PerBlockBytes, and the
// engine keeps the smallest.
type StageMemory struct {
	// Fixed is the stage rank's request-independent occupancy, as FixedBytes reports the
	// binding rank's.
	Fixed kernel.MemoryBreakdown
	// PerBlockBytes is one block of the rank's growing KV, page-quantized: SequenceVariableBytes
	// of one block on that rank.
	PerBlockBytes int64
	// LayerPageBytes is one block of every KV layer the rank holds whose cache the kernel
	// infers, window rings included (KVLayerPageBytes on that rank).
	LayerPageBytes int64
	// Binding is whether this is the rank FixedBytes and the sequence sizes describe.
	Binding bool
}

// PipelineStageMemory returns every pipeline rank's memory, in stage order; nil at pp 1,
// where FixedBytes and SequenceVariableBytes are the one rank's.
//
// FixedBytes and the sequence sizes describe the binding rank -- the stage whose memory
// beside its fixed bytes holds the fewest tokens of its KV -- so a consumer that sizes one
// pool from them gets vLLM's pool wherever the binding stage's per-block bytes are the
// largest. One that sizes each stage separately, as vLLM does, reads this.
func (k *Kernel) PipelineStageMemory() []StageMemory {
	if len(k.stages) == 0 {
		return nil
	}
	out := make([]StageMemory, len(k.stages))
	for i := range k.stages {
		m := k.stages[i].memory
		out[i] = StageMemory{
			Fixed: kernel.MemoryBreakdown{
				Weights:        int64(m.weights - m.eplbRedundant),
				ActivationPeak: int64(m.activation),
				CUDAGraph:      k.captureBytes,
				EPLBRedundant:  int64(m.eplbRedundant),
				CommBuffers:    k.deepEPBufferBytes(k.graph),
			},
			PerBlockBytes:  k.pagedKVBytes(k.blockSize, m.kvPerToken),
			LayerPageBytes: k.pagedKVBytes(k.blockSize, m.layerPerToken),
			Binding:        i == k.ppKVStage,
		}
	}
	return out
}

// Original to blis-vllm.

package kernelmodel

import (
	"fmt"
	"math/bits"
	"slices"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	latencykernel "github.com/inference-sim/blis-substrate/kernel"

	"github.com/inference-sim/blis-vllm/core/vllm"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// Decoder-side SWA bounded replay: the rows the layers after the last KV-producing layer run.
//
// DeepSeek-V4.1 at vLLM v0.32.0 (not v0.31.0) writes the KV of every row in layers 0 to 20, then
// runs layer 20 and the layers after it, which own only sliding-window KV, on each request's last
// `window` scheduled tokens (vllm/models/deepseek_v41/nvidia/model.py:800-835,1160-1245,
// decoder_replay_layers.py, model_state.py:438-540@v0.32.0). The latency kernel prices every layer
// on the scheduled tokens unless the batch states a tail (kernel.Batch.TailFrom, ReqShape.TailRows,
// TailPadTokens); this file is where the engine states it, from vLLM's own rules:
//
//   - The model has the mechanism when it is DeepSeek-V4.1 and its SWA bounded replay resolved on
//     (Model.PrefixHit: the V2 model runner, no prefill context parallelism, not switched off).
//     The decoder side adds its own conditions (_decoder_replay_supported, model.py:1202-1238):
//     the stage that holds the last KV layer holds every layer after it, no model sequence
//     parallelism, prefill context parallelism or micro-batching, no Engram layer at or after the
//     cut, and a drafter, if any, with a window no wider than the target's and only
//     sliding-attention layers.
//   - A step trims when the model runner runs it with no graph, or in a piecewise graph padded to
//     decoder_replay_trim_threshold tokens or more (model_state.py:449-465); a full graph, or a
//     small piecewise one, keeps the whole batch. Each request then keeps its last
//     min(scheduled, window) tokens (model_state.py:466-470): a decode or verify row keeps all of its
//     tokens, a prefill chunk its last 128, whatever the length of the prompt or the prefix hit
//     behind it.
//   - The trimmed batch runs in a piecewise graph of its own, padded to the smallest size of
//     decoder_replay_cudagraph_capture_sizes that holds it, when cudagraph_decoder_replay is on and
//     the batch fits one; otherwise eager, at its own size (decoder_replay_cudagraph.py:21-52,
//     model_state.py:493-505).
//
// What it does not price: layer 20's KV write for the rows outside the window (the kernel prices
// layer 20 wholly on the kept rows), the launch overhead of the replay graph against eager, a
// request that asks for prompt logprobs (it keeps every row), and the data-parallel agreement of
// the replay batch across ranks (a data-parallel pool prices every layer on every token, as before).

const (
	// decoderReplayTrimThreshold is CompilationConfig.decoder_replay_trim_threshold's default
	// (vllm/config/compilation.py:571@v0.32.0). The engine block has no key for it, so it is
	// always the default.
	decoderReplayTrimThreshold = 768
)

// captureSizer is the kernel's report of the capture sizes it prices from (kernel CaptureSizes).
type captureSizer interface {
	graphResolver
	CaptureSizes() []int
}

// decoderReplay is a pool's decoder-side bounded replay as it resolved. A nil one runs none.
type decoderReplay struct {
	// from is the position, in the model's layer stack, of the last layer that owns a compressed KV
	// cache (kv_source_layer_ids' last): the first layer the trimmed batch runs.
	from int
	// window is the sliding window in tokens: the rows each request keeps.
	window int
	// otherMode is the graph mode a step that is not a uniform decode runs under ("FULL",
	// "PIECEWISE" or "NONE"), ceiling the largest step a graph covers and sizes the captured sizes.
	otherMode string
	ceiling   int
	sizes     []int
	// replaySizes are the trimmed batch's graph sizes, ascending; empty when it runs no graph.
	replaySizes []int
}

// decoderReplayOf resolves the pool's decoder replay, or the reason it does not run. A pool whose
// model class has no such mechanism returns neither. A stated decoder_replay_cudagraph_capture_sizes
// that vLLM refuses at startup is an error.
func (m *Model) decoderReplayOf(release string, rl sequenceParallelRules, pl poolShape, cuda captureSizer) (*decoderReplay, string, error) {
	g := m.graph
	if g == nil || g.Global.Architecture != rules.ArchDeepseekV41 || release != vllm.Version {
		return nil, "", nil
	}
	e := m.engine
	hit := m.PrefixHit(nil)
	switch {
	case hit.BoundedReplay == nil || !*hit.BoundedReplay:
		return nil, "SWA bounded replay is off (" + hit.BoundedReplayFrom + ")", nil
	case pl.pp > 1:
		return nil, "pipeline parallelism: the stage holding the last KV layer must hold every layer after it", nil
	case pl.dp > 1:
		return nil, "data parallelism: the replay batch is agreed across ranks, which is not priced", nil
	case pl.pcp > 1 || (e.DBO != nil && e.DBO.Enabled):
		return nil, "prefill context parallelism or micro-batching", nil
	case rl != nil && rl.ModelSequenceParallel(g.Global.Architecture, pl.parallel, e.MoEBackend):
		return nil, "the model's own sequence parallelism", nil
	}
	from, ok := lastKVLayer(g)
	if !ok || from >= g.Stack.Layers()-1 {
		return nil, "no layer follows the last KV layer", nil
	}
	if engram(g, from) {
		return nil, "an Engram layer sits at or after the last KV layer", nil
	}
	if why := drafterOutsideWindow(g, e, m.id.prefix.window); why != "" {
		return nil, why, nil
	}
	d := &decoderReplay{from: from, window: m.id.prefix.window}
	d.ceiling, d.sizes = cuda.CaptureTokens(), cuda.CaptureSizes()
	_, other := cuda.GraphModes()
	d.otherMode = strings.ToUpper(other)
	// cudagraph_decoder_replay is on (the block has no key to turn it off), and needs piecewise
	// graphs and the breakable capture V4.1 defaults to (model_state.py:246-267).
	if strings.Contains(d.otherMode, "PIECEWISE") {
		sizes, err := replayGraphSizes(e, d.window, d.ceiling, m.resolved.engine.MaxNumSeqs)
		if err != nil {
			return nil, "", err
		}
		d.replaySizes = sizes
	}
	return d, "", nil
}

// poolShape is the parallel layout the replay conditions read.
type poolShape struct {
	pp, dp, pcp int
	parallel    deployment.Parallelism
}

// sequenceParallelRules is the release rules' answer to whether the model's own sequence
// parallelism is on (EngineRules.ModelSequenceParallel).
type sequenceParallelRules interface {
	ModelSequenceParallel(architecture string, pl deployment.Parallelism, moeBackend string) bool
}

// lastKVLayer is the position in the stack of the last layer whose kind owns a compressed KV cache:
// max(config.kv_source_layer_ids), the layer that writes every row's KV before the layers that
// read it (model.py:805).
func lastKVLayer(g *model.Graph) (int, bool) {
	owns := map[string]bool{}
	for _, lk := range g.LayerKinds {
		for _, n := range lk.Nodes {
			for _, c := range n.Caches {
				if c.Owned && c.Kind == model.CacheKV {
					owns[lk.ID] = true
				}
			}
		}
	}
	last := -1
	for i, id := range g.Stack.Expand() {
		if owns[id] {
			last = i
		}
	}
	return last, last >= 0
}

// engram reports whether a layer at or after from is an Engram layer (model.py:1220).
func engram(g *model.Graph, from int) bool {
	has := map[string]bool{}
	for _, lk := range g.LayerKinds {
		for _, n := range lk.Nodes {
			if strings.HasPrefix(n.Role, "engram") {
				has[lk.ID] = true
			}
		}
	}
	for i, id := range g.Stack.Expand() {
		if i >= from && has[id] {
			return true
		}
	}
	return false
}

// drafterOutsideWindow says why a drafter reads hidden states outside the target's window, or "":
// a drafter whose layers are not all sliding-window attention, or whose window is wider
// (model.py:1224-1236). A named drafter checkpoint is not described by this graph, so it is
// taken as outside the window rather than guessed.
func drafterOutsideWindow(g *model.Graph, e engineconfig.Engine, window int) string {
	sp := e.Speculative
	if sp == nil || sp.NumSpecTokens == 0 {
		return ""
	}
	if sp.DraftModel != "" {
		return "a named drafter (" + sp.DraftModel + "), whose window and layer types this graph does not state"
	}
	if g.Speculator == nil {
		return ""
	}
	kinds := map[string]model.LayerKind{}
	for _, lk := range g.LayerKinds {
		kinds[lk.ID] = lk
	}
	for _, id := range g.Speculator.Stack.Expand() {
		for _, n := range kinds[id].Nodes {
			for _, c := range n.Caches {
				if !c.Owned {
					continue
				}
				if c.Kind != model.CacheWindow || c.Window > window {
					return fmt.Sprintf("the drafter's layer kind %s keeps a %s cache of window %d, outside the target's %d-token window",
						id, c.Kind, c.Window, window)
				}
			}
		}
	}
	return ""
}

// replayGraphSizes are the trimmed batch's graph sizes: the stated ones, or, unstated, multiples
// of the window up to min(max_cudagraph_capture_size, max_num_seqs x window), stepping by 1/8 of
// the power-of-two interval once wider than 16 windows (decoder_replay_cudagraph.py:24-40). A size
// above max_num_batched_tokens is vLLM's startup error (:42-46).
func replayGraphSizes(e engineconfig.Engine, window, ceiling, maxNumSeqs int) ([]int, error) {
	sizes := slices.Clone(e.DecoderReplayCUDAGraphCaptureSizes)
	if len(sizes) == 0 {
		bound := min(ceiling, maxNumSeqs*window)
		sizes = []int{bound}
		for s := window; s < bound; s += max(window, 1<<max(bits.Len(uint(s))-4, 0)) {
			sizes = append(sizes, s)
		}
	}
	sort.Ints(sizes)
	sizes = slices.Compact(sizes)
	if mnbt := e.MaxNumBatchedTokens; mnbt > 0 && sizes[len(sizes)-1] > mnbt {
		return nil, fmt.Errorf("decoder_replay_cudagraph_capture_sizes must not exceed max_num_batched_tokens "+
			"(%d): vLLM refuses a size of %d at startup (vllm/models/deepseek_v41/nvidia/decoder_replay_cudagraph.py:42-46@v0.32.0)",
			mnbt, sizes[len(sizes)-1])
	}
	return sizes, nil
}

// trims reports whether a step of tokens scheduled tokens trims: the model runner runs it with no
// graph (a mode with none for it, or more tokens than the largest captured size), or in a
// piecewise graph padded to the trim threshold or more.
func (d *decoderReplay) trims(tokens int) bool {
	if d.otherMode == "NONE" || d.ceiling <= 0 || tokens > d.ceiling {
		return true
	}
	if !strings.Contains(d.otherMode, "PIECEWISE") {
		return false
	}
	padded := tokens
	if i := sort.SearchInts(d.sizes, tokens); i < len(d.sizes) {
		padded = d.sizes[i]
	}
	return padded >= decoderReplayTrimThreshold
}

// tail is the batch with the decoder replay it runs: TailFrom, each request's rows, and the rows
// the trimmed batch is padded with. The batch is returned unchanged when the step does not trim, or
// trims nothing (every request keeps all its rows).
func (d *decoderReplay) tail(b kernel.Batch) kernel.Batch {
	if d == nil || !d.trimsAny(b) || !d.trims(b.Tokens()) {
		return b
	}
	b.Reqs = slices.Clone(b.Reqs)
	rows := 0
	for i := range b.Reqs {
		r := min(b.Reqs[i].Scheduled, d.window)
		if r < b.Reqs[i].Scheduled {
			b.Reqs[i].TailRows = r
		}
		rows += r
	}
	b.TailFrom = d.from
	if i := sort.SearchInts(d.replaySizes, rows); i < len(d.replaySizes) {
		b.TailPadTokens = d.replaySizes[i] - rows
	}
	return b
}

// trimsAny reports whether some request schedules more than the window, the only batch that builds
// a replay batch at all (model_state.py:480-483).
func (d *decoderReplay) trimsAny(b kernel.Batch) bool {
	for _, r := range b.Reqs {
		if r.Scheduled > d.window {
			return true
		}
	}
	return false
}

// resolveDecoderReplay resolves the pool's decoder replay and discloses it: what runs, or why a
// model that has the mechanism does not run it, or that a stated decoder_replay_cudagraph_capture_sizes
// has no effect because the model has no such layers.
func (m *Model) resolveDecoderReplay(in latencykernel.Inputs, pool deployment.Pool) error {
	sizes := m.engine.DecoderReplayCUDAGraphCaptureSizes
	cuda, ok := m.k.(captureSizer)
	if !ok {
		if len(sizes) > 0 {
			return fmt.Errorf("decoder_replay_cudagraph_capture_sizes is stated and the kernel (%T) does not report its capture sizes, so the replay graphs cannot be priced", m.k)
		}
		return nil
	}
	rl, _ := in.Rules.(sequenceParallelRules)
	pl := poolShape{pp: max(pool.Parallel.PP, 1), dp: max(pool.Parallel.DP, 1), pcp: max(pool.Parallel.PCP, 1), parallel: pool.Parallel}
	d, why, err := m.decoderReplayOf(in.Scenario.EngineRelease(), rl, pl, cuda)
	if err != nil {
		return err
	}
	switch {
	case d != nil:
		m.replay = d
		note := fmt.Sprintf("layers %d to %d run on each request's last %d tokens in steps with no graph or a piecewise graph of %d or more tokens",
			d.from, in.Model.Stack.Layers()-1, d.window, decoderReplayTrimThreshold)
		if len(d.replaySizes) > 0 {
			note += fmt.Sprintf("; the trimmed batch is padded to the replay graph sizes %v", d.replaySizes)
		} else {
			note += "; the trimmed batch runs eager at its own size"
		}
		m.assumed = append(m.assumed, "decoder_replay = on: "+note+" (vllm/models/deepseek_v41/nvidia/model_state.py:438-540@v0.32.0); "+
			"layer "+fmt.Sprint(d.from)+"'s KV write for the other rows and the replay graph's launch saving are not priced")
	case why != "":
		m.assumed = append(m.assumed, "decoder_replay = off: "+why+"; every layer is priced on every scheduled token, as vLLM 0.31.0 runs")
	case len(sizes) > 0:
		m.assumed = append(m.assumed, "decoder_replay_cudagraph_capture_sizes = "+fmt.Sprint(sizes)+
			": this model and release have no decoder replay layers, so vLLM does not read the key either")
	}
	return nil
}

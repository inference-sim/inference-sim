package model

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// Validate performs FIELD-LEVEL validation: that the document is structurally a
// model graph, that every enumerated value is recognized, and that each node
// carries exactly the parameters its op prices.
//
// It knows nothing about engine versions. Whether a speculative method name exists
// in some release, or whether a backend is still supported, is a version-scoped
// question and lives in a rules pack. Keeping the two apart is what lets this
// function stay unchanged while engines move.
func (g *Graph) Validate() *validate.Problems {
	p := &validate.Problems{}

	if g.Kind != "ModelGraph" {
		p.Field("kind", "must be %q, got %q", "ModelGraph", g.Kind)
	}
	if g.Name == "" {
		p.Field("name", "required")
	}
	if !g.Modality.Valid() {
		p.Field("modality", "%q is not a recognized modality", g.Modality)
	}
	g.validateDerivation(p)
	g.validateGlobal(p)
	g.validateLayerKinds(p)
	g.validateStack(p)
	g.validateSpeculator(p)

	for i, n := range g.Embedding {
		n.validate(p, fmt.Sprintf("embedding[%d]", i))
		validateOutsideLayers(p, fmt.Sprintf("embedding[%d]", i), n)
	}
	for i, n := range g.Head {
		n.validate(p, fmt.Sprintf("head[%d]", i))
		validateOutsideLayers(p, fmt.Sprintf("head[%d]", i), n)
	}
	g.validateCacheSharing(p, "stack", g.Stack)
	if s := g.Speculator; s != nil {
		g.validateCacheSharing(p, "speculator.stack", s.Stack)
		if s.ContextStack != nil {
			g.validateCacheSharing(p, "speculator.context_stack", *s.ContextStack)
		}
	}
	g.validateAliases(p)
	g.validateArchOverrides(p)
	g.validateWindowReads(p)
	return p
}

func (g *Graph) validateDerivation(p *validate.Problems) {
	d := g.DerivedFrom
	if d.Format == "" {
		p.Field("derived_from.format", "required: a graph must say which dialect it was derived from")
	}
	if d.Path == "" {
		p.Field("derived_from.path", "required")
	}
	// A digest is what makes a derived artifact auditable against its source. A
	// graph without one can drift silently, so its absence is an error rather than
	// a warning.
	if len(d.SHA256) != 64 {
		p.Field("derived_from.sha256",
			"must be a 64-character hex digest of the source file, got %d characters",
			len(d.SHA256))
	}
	if d.DeriverVersion < 1 {
		p.Field("derived_from.deriver_version", "must be at least 1")
	}
}

func (g *Graph) validateGlobal(p *validate.Problems) {
	if g.Global.HiddenSize < 1 {
		p.Field("global.hidden_size", "must be positive")
	}
	if g.Global.VocabSize < 1 {
		p.Field("global.vocab_size", "must be positive")
	}
	if !g.Global.WeightDType.Valid() {
		p.Field("global.weight_dtype", "%q is not a recognized dtype", g.Global.WeightDType)
	}
	validateWeightScale(p, "global.weight_scale", g.Global.WeightScale)
	if g.Global.ContextWindow < 0 {
		p.Field("global.context_window", "must not be negative")
	}
	if g.Global.DefaultContextWindow < 0 {
		p.Field("global.default_context_window", "must not be negative")
	}
	if g.Global.DefaultContextWindow > 0 && g.Global.ContextWindow > 0 &&
		g.Global.DefaultContextWindow > g.Global.ContextWindow {
		p.Field("global.default_context_window",
			"%d exceeds context_window %d, which the engine would refuse",
			g.Global.DefaultContextWindow, g.Global.ContextWindow)
	}
	// The model's sequence parallelism is decided by its own code, so a node that asks
	// about it asks a question only a named architecture lets a rules pack answer.
	if g.Global.Architecture == "" {
		g.EachNode(func(at string, n Node) {
			switch n.SequenceParallelField() {
			case "emit":
				p.Field(at+".emit", "%q needs global.architecture, whose model code decides it", n.Emit)
			case "under_sequence_parallel":
				p.Field(at+".under_sequence_parallel", "%q needs global.architecture, whose model code decides it", n.UnderSequenceParallel)
			}
		})
	}
	if g.Global.PipelineBoundaryBytesPerToken < 0 {
		p.Field("global.pipeline_boundary_bytes_per_token", "must not be negative")
	}
	if g.Global.DefaultContextWindow > 0 && g.Global.ContextWindow == 0 {
		p.Field("global.default_context_window", "stated without context_window, which it narrows")
	}
}

func validateWeightScale(p *validate.Problems, at string, s *WeightScale) {
	if s == nil {
		return
	}
	requirePositive(p, at, "elements", s.Elements)
	requirePositive(p, at, "bytes", s.Bytes)
}

func (g *Graph) validateLayerKinds(p *validate.Problems) {
	if len(g.LayerKinds) == 0 {
		p.Field("layer_kinds", "at least one layer kind is required")
		return
	}
	seen := map[string]bool{}
	for i, lk := range g.LayerKinds {
		at := fmt.Sprintf("layer_kinds[%d]", i)
		if lk.ID == "" {
			p.Field(at+".id", "required")
		} else if seen[lk.ID] {
			p.Field(at+".id", "duplicate layer-kind id %q", lk.ID)
		}
		seen[lk.ID] = true

		if len(lk.Nodes) == 0 {
			p.Field(at+".nodes", "a layer kind with no nodes prices nothing")
		}
		for j, n := range lk.Nodes {
			n.validate(p, fmt.Sprintf("%s.nodes[%d]", at, j))
		}
		validateLayerCacheNames(p, at, lk)
		validateLayerCacheRefs(p, at, lk)
		validateEdges(p, at, len(lk.Nodes), lk.Edges)
	}
}

// validateEdges checks that every edge indexes a real node, that no node is its own
// predecessor, and that the graph is acyclic. A cycle would make a layer's cost
// undefined, so it is rejected rather than traversed defensively at run time.
func validateEdges(p *validate.Problems, at string, n int, edges [][2]int) {
	adj := make([][]int, n)
	for k, e := range edges {
		from, to := e[0], e[1]
		if from < 0 || from >= n {
			p.Field(fmt.Sprintf("%s.edges[%d]", at, k),
				"from-index %d is out of range for %d nodes", from, n)
			continue
		}
		if to < 0 || to >= n {
			p.Field(fmt.Sprintf("%s.edges[%d]", at, k),
				"to-index %d is out of range for %d nodes", to, n)
			continue
		}
		if from == to {
			p.Field(fmt.Sprintf("%s.edges[%d]", at, k), "self-edge on node %d", from)
			continue
		}
		adj[from] = append(adj[from], to)
	}
	if cyc := findCycle(adj); cyc >= 0 {
		p.Field(at+".edges", "graph is cyclic; a cycle reaches node %d", cyc)
	}
}

// findCycle returns a node on a cycle, or -1. Iterative depth-first search with
// three colours; the graph is small enough that clarity beats cleverness.
func findCycle(adj [][]int) int {
	const (
		white = 0
		grey  = 1
		black = 2
	)
	colour := make([]int, len(adj))
	for start := range adj {
		if colour[start] != white {
			continue
		}
		stack := []int{start}
		for len(stack) > 0 {
			v := stack[len(stack)-1]
			switch colour[v] {
			case white:
				colour[v] = grey
				for _, w := range adj[v] {
					if colour[w] == grey {
						return w
					}
					if colour[w] == white {
						stack = append(stack, w)
					}
				}
			case grey:
				colour[v] = black
				stack = stack[:len(stack)-1]
			default:
				stack = stack[:len(stack)-1]
			}
		}
	}
	return -1
}

func (g *Graph) validateStack(p *validate.Problems) {
	s := g.Stack
	// A stack must produce at least one layer. Either a repeated pattern or a
	// literal run satisfies that; a model with no periodicity uses the run alone.
	if s.Layers() < 1 {
		p.Field("stack",
			"expands to no layers: state a pattern with a repeat, or a literal prologue")
	}
	if s.Repeat < 0 {
		p.Field("stack.repeat", "must not be negative")
	}
	if len(s.Pattern) > 0 && s.Repeat < 1 {
		p.Field("stack.repeat", "a pattern is stated but never repeated")
	}
	if len(s.Pattern) == 0 && s.Repeat > 0 {
		p.Field("stack.pattern", "a repeat is stated but there is no pattern to repeat")
	}
	ids := map[string]bool{}
	for _, lk := range g.LayerKinds {
		ids[lk.ID] = true
	}
	for field, seq := range map[string][]string{
		"prologue": s.Prologue, "pattern": s.Pattern, "epilogue": s.Epilogue,
	} {
		for i, id := range seq {
			if !ids[id] {
				p.Field(fmt.Sprintf("stack.%s[%d]", field, i),
					"%q names no layer kind", id)
			}
		}
	}
	// Every declared kind should appear, or it prices nothing and is more likely a
	// leftover than an intention. A kind the speculator's own stack uses counts as
	// used: a draft module is a second stack over the same layer kinds, so a kind that
	// only a speculator references (DeepSeek-V4-Pro's mtp_moe) is intentional, not a leftover.
	used := map[string]bool{}
	for _, id := range s.Expand() {
		used[id] = true
	}
	if g.Speculator != nil {
		for _, id := range g.Speculator.Stack.Expand() {
			used[id] = true
		}
		if g.Speculator.ContextStack != nil {
			for _, id := range g.Speculator.ContextStack.Expand() {
				used[id] = true
			}
		}
	}
	for _, lk := range g.LayerKinds {
		if !used[lk.ID] {
			p.Warnf("layer kind %q is declared but never used by the stack", lk.ID)
		}
	}
}

func (g *Graph) validateSpeculator(p *validate.Problems) {
	s := g.Speculator
	if s == nil {
		return
	}
	if s.Method == "" {
		p.Field("speculator.method", "required")
	}
	if s.NumSpec < 1 {
		p.Field("speculator.num_spec_tokens",
			"must be at least 1; omit the speculator block instead of declaring zero drafts")
	}
	if len(s.Stack.Pattern) == 0 {
		p.Field("speculator.stack.pattern",
			"required: a draft module is its own stack, because for an MoE target the draft pass is a second MoE")
	}
	if s.Stack.Repeat < 1 {
		p.Field("speculator.stack.repeat", "must be at least 1")
	}
	if !s.Drafting.Valid() {
		p.Field("speculator.drafting", "%q is not a recognized drafting mode", s.Drafting)
	}
	if len(s.SequentialHead) > 0 && s.Drafting != DraftingParallelBlock {
		p.Field("speculator.sequential_head",
			"only a parallel_block drafter has a per-position head; an autoregressive one is sequential already")
	}
	ids := map[string]bool{}
	for _, lk := range g.LayerKinds {
		ids[lk.ID] = true
	}
	// The draft stacks name layer kinds like the target stack does, and a name that
	// matches none would price nothing.
	stacks := map[string]Stack{"speculator.stack": s.Stack}
	if c := s.ContextStack; c != nil {
		stacks["speculator.context_stack"] = *c
		if c.Layers() < 1 {
			p.Field("speculator.context_stack", "expands to no layers; omit it instead")
		}
		if len(c.Pattern) > 0 && c.Repeat < 1 {
			p.Field("speculator.context_stack.repeat", "a pattern is stated but never repeated")
		}
		if len(c.Pattern) == 0 && c.Repeat > 0 {
			p.Field("speculator.context_stack.pattern", "a repeat is stated but there is no pattern to repeat")
		}
	}
	for field, st := range stacks {
		for _, part := range []struct {
			name string
			seq  []string
		}{{"prologue", st.Prologue}, {"pattern", st.Pattern}, {"epilogue", st.Epilogue}} {
			for i, id := range part.seq {
				if !ids[id] {
					p.Field(fmt.Sprintf("%s.%s[%d]", field, part.name, i), "%q names no layer kind", id)
				}
			}
		}
	}
	for i, n := range s.Embedding {
		n.validate(p, fmt.Sprintf("speculator.embedding[%d]", i))
		validateOutsideLayers(p, fmt.Sprintf("speculator.embedding[%d]", i), n)
	}
	for i, n := range s.Head {
		n.validate(p, fmt.Sprintf("speculator.head[%d]", i))
		validateOutsideLayers(p, fmt.Sprintf("speculator.head[%d]", i), n)
	}
	for i, n := range s.SequentialHead {
		n.validate(p, fmt.Sprintf("speculator.sequential_head[%d]", i))
		validateOutsideLayers(p, fmt.Sprintf("speculator.sequential_head[%d]", i), n)
	}
	// A capture id names a point in the TARGET's stack: 0 before its first layer, its
	// layer count after the last. A drafter that is a checkpoint of its own states its
	// own draft stack as the graph's, so the upper bound is checked where the drafter
	// meets its target, not here. The engine sorts and deduplicates the ids
	// (_set_aux_hidden_state_layers), so a graph states them that way.
	for i, id := range s.AuxHiddenStateLayers {
		at := fmt.Sprintf("speculator.aux_hidden_state_layers[%d]", i)
		if id < 0 {
			p.Field(at, "must not be negative, got %d", id)
		}
		if i > 0 && id <= s.AuxHiddenStateLayers[i-1] {
			p.Field(at, "%d does not follow %d: the ids are strictly increasing", id, s.AuxHiddenStateLayers[i-1])
		}
	}
	if b := s.AuxHiddenStateBytesPerToken; b < 0 {
		p.Field("speculator.aux_hidden_state_bytes_per_token", "must not be negative")
	} else if b > 0 && len(s.AuxHiddenStateLayers) == 0 {
		p.Field("speculator.aux_hidden_state_bytes_per_token",
			"sizes a captured state, and aux_hidden_state_layers captures none")
	}
}

// validate checks one node: that its op is known, that the parameters its op prices
// are present and positive, and that it carries no parameter belonging to a
// different op. The last check is what keeps the union-of-fields Node honest.
func (n Node) validate(p *validate.Problems, at string) {
	if !n.Op.Valid() {
		p.Field(at+".op", "%q is not a recognized primitive", n.Op)
		return
	}
	if !n.Emit.Valid() {
		p.Field(at+".emit", "%q is not a recognized emit condition", n.Emit)
	}
	// A collective's existence depends on the layout, so it must say under which
	// layouts it exists. An unconditional collective would be emitted even at tp=1.
	if n.Op.Collective() && n.Emit == EmitAlways {
		p.Field(at+".emit",
			"a collective node must name an emit condition; an unconditional %s would be emitted at tp=1", n.Op)
	}
	switch {
	case !n.Op.Collective() && n.Emit.SequenceParallelOnly():
		// Work runs in every layout. At tp 1 neither sequence-parallel condition holds,
		// so work stated twice, once under each, vanishes there.
		p.Warnf("%s: emit %q on a %s, which is work rather than a collective; at tp 1 neither sequence-parallel condition holds, so state the node once, with under_sequence_parallel for how the mode divides it",
			at, n.Emit, n.Op)
	case !n.Op.Collective() && n.Emit != EmitAlways:
		// Allowed, but unusual enough to flag: a non-collective conditional node
		// usually means the graph is encoding a deployment rather than a model.
		p.Warnf("%s: a condition on a non-collective node encodes deployment in the model graph", at)
	}
	if !n.UnderSequenceParallel.Valid() {
		p.Field(at+".under_sequence_parallel", "%q is not a recognized split", n.UnderSequenceParallel)
	} else if n.UnderSequenceParallel != SPAsTP && n.Op.Collective() {
		p.Field(at+".under_sequence_parallel",
			"a collective is defined by its group and its emit condition; only work is token-sharded")
	}

	if !n.TP.ValidFor(n.Op) {
		p.Field(at+".tp", "%q is not a layout a %s node accepts", n.TP, n.Op)
	}
	if n.Aliases != "" && n.Op != OpGEMM && n.Op != OpGather {
		p.Field(at+".aliases", "only a GEMM or Gather reads weights another node holds")
	}
	if n.WeightScale != nil {
		if n.Op != OpGEMM && n.Op != OpGroupedGEMM {
			p.Field(at+".weight_scale", "only a GEMM or GroupedGEMM holds parameters")
		}
		validateWeightScale(p, at+".weight_scale", n.WeightScale)
	}
	n.validateNodeCaches(p, at)
	if d := n.ShardedAtDecode; d != nil {
		at := at + ".sharded_at_decode"
		if n.Op != OpGEMM || n.TP != TPReplicated {
			p.Field(at, "only a replicated GEMM (tp: replicated) splits its columns at decode")
		}
		requirePositive(p, at, "capability_family", d.CapabilityFamily)
		requirePositive(p, at, "max_tokens", d.MaxTokens)
		if d.MinTokensAcrossNodes < 0 || d.MinTokensAcrossNodes > d.MaxTokens {
			p.Field(at+".min_tokens_across_nodes", "%d is outside [0, max_tokens %d]",
				d.MinTokensAcrossNodes, d.MaxTokens)
		}
	}
	if n.KVHeadsNotReplicated && n.Op != OpAttention {
		p.Field(at+".kv_heads_not_replicated", "only an Attention node has KV heads")
	}
	if n.PerTokens != 0 && n.Op != OpElementwise {
		p.Field(at+".per_tokens", "only an Elementwise node runs once per several tokens")
	}
	if n.PerTokens < 0 {
		p.Field(at+".per_tokens", "must not be negative")
	}
	if n.Context != nil {
		if n.Op != OpElementwise {
			p.Field(at+".context", "only an Elementwise node carries context-proportional traffic")
		}
		validateContext(p, at+".context", *n.Context)
	}
	if n.WritesCache != "" && n.Op != OpElementwise {
		p.Field(at+".writes_cache", "only an Elementwise node is a cache insert")
	}
	if n.Op != OpGather {
		if n.Rows != 0 {
			p.Field(at+".rows", "not priced by this op; remove it")
		}
		forbid(p, at, map[string]int{
			"row_bytes": n.RowBytes, "rows_per_token": n.RowsPerToken,
			"out_bytes_per_token": n.OutBytesPerToken,
		})
		for field, set := range map[string]bool{
			"residency": n.Residency != "", "residency_control": n.ResidencyControl != "",
			"shard": n.Shard != "",
		} {
			if set {
				p.Field(at+"."+field, "only a Gather has a table; remove it")
			}
		}
	}
	if n.BytesPerToken != 0 {
		switch n.Op {
		case OpElementwise, OpAllReduce, OpAllGather, OpReduceScatter:
		default:
			// An All2All's volume follows from the routed top-k and the backend, and
			// a GEMM's or an attention node's traffic from its shape.
			p.Field(at+".bytes_per_token", "not priced by this op; remove it")
		}
	}

	// A per-node weight dtype is only meaningful where the node holds parameters, and an
	// unrecognized one must be an error rather than silently falling back to the global
	// width: the whole point of the override is that the two differ.
	if n.WeightDType != "" {
		switch {
		case !n.WeightDType.Valid():
			p.Field(at+".weight_dtype", "%q is not a recognized dtype", n.WeightDType)
		case n.Op != OpGEMM && n.Op != OpGroupedGEMM:
			p.Field(at+".weight_dtype", "only a GEMM or GroupedGEMM holds parameters")
		}
	}

	// The latent-read compression factor is meaningful only on a compressed sparse-MLA
	// layer, so it is rejected on any other op or attention kind -- gated the same way as
	// the other MLA-specific parameters. The field is optional (omitempty), so a zero is
	// an absent value, not a declared one; a negative ratio is a mistake and is rejected.
	// The wrong-kind and negative faults are independent, so a node that is both reports
	// both rather than masking one, matching the validator's report-every-problem contract.
	if n.CompressRatio != 0 {
		if n.CompressRatio < 0 {
			p.Field(at+".compress_ratio", "must be positive, got %d", n.CompressRatio)
		}
		if n.Op != OpAttention || n.AttentionKind != AttentionSparseMLA {
			p.Field(at+".compress_ratio", "only priced by a sparse_mla attention node")
		}
	}

	// index_topk selects a top-k of the latent cache, so like compress_ratio it belongs
	// only on a sparse_mla attention node; a value set anywhere else is rejected. Its
	// positivity and the index_topk-or-compress_ratio requirement are checked in the
	// sparse-MLA branch below, so this guard only polices placement -- reporting the
	// negative value there too would double-report the one in the branch.
	if n.IndexTopK != 0 && (n.Op != OpAttention || n.AttentionKind != AttentionSparseMLA) {
		p.Field(at+".index_topk", "only priced by a sparse_mla attention node")
	}

	switch n.Op {
	case OpGEMM:
		requirePositive(p, at, "n", n.N)
		requirePositive(p, at, "k", n.K)
		forbid(p, at, map[string]int{
			"experts": n.Experts, "top_k": n.TopK, "n_q": n.NumQHeads,
			"n_kv": n.NumKVHeads, "state_size": n.StateSize,
		})
	case OpGroupedGEMM:
		requirePositive(p, at, "n", n.N)
		requirePositive(p, at, "k", n.K)
		requirePositive(p, at, "experts", n.Experts)
		requirePositive(p, at, "top_k", n.TopK)
		if n.TopK > n.Experts {
			p.Field(at+".top_k", "top_k %d exceeds experts %d", n.TopK, n.Experts)
		}
		if n.SharedExperts < 0 {
			p.Field(at+".shared_experts", "must not be negative")
		}
		if n.SharedIntermediateSize < 0 {
			p.Field(at+".shared_intermediate_size", "must not be negative")
		}
		if n.SharedIntermediateSize > 0 && n.SharedExperts == 0 {
			p.Field(at+".shared_intermediate_size",
				"stated without shared_experts, so it sizes nothing")
		}
		if n.LatentSize < 0 {
			p.Field(at+".latent_size", "must not be negative")
		}
		forbid(p, at, map[string]int{"n_q": n.NumQHeads, "state_size": n.StateSize})
	case OpAttention:
		if !n.AttentionKind.Valid() {
			p.Field(at+".kind", "%q is not a recognized attention kind", n.AttentionKind)
		}
		requirePositive(p, at, "n_q", n.NumQHeads)
		requirePositive(p, at, "n_kv", n.NumKVHeads)
		requirePositive(p, at, "d_h", n.HeadDim)
		if n.NumKVHeads > n.NumQHeads {
			p.Field(at+".n_kv", "n_kv %d exceeds n_q %d", n.NumKVHeads, n.NumQHeads)
		}
		if n.KVHeadsNotReplicated && n.AttentionKind.LatentKV() {
			p.Field(at+".kv_heads_not_replicated",
				"%s stores one latent vector per token, which every rank holds whole", n.AttentionKind)
		}
		if n.AttentionKind.LatentKV() && n.NumKVHeads != 1 {
			// A latent cache is one vector per token: an engine pins the head count
			// to one, so any other value describes no real deployment.
			p.Field(at+".n_kv",
				"%s stores one latent vector per token, so n_kv must be 1, got %d",
				n.AttentionKind, n.NumKVHeads)
		}
		if n.AttentionKind == AttentionSWA && n.Window < 1 {
			p.Field(at+".window", "sliding-window attention requires a positive window")
		}
		// A sparse-MLA layer must do at least one of the two things that make it sparse:
		// select a top-k of the latent cache (index_topk) or compress the latent read
		// (compress_ratio). DeepSeek-V4-Pro ships both shapes -- csa4_moe selects a top-k
		// and lightly compresses, csa128_moe compresses heavily with no top-k selection --
		// so requiring index_topk unconditionally would reject a layer the catalog ships.
		// A set index_topk must still be positive: a negative count is a mistake, not a
		// compress-only layer, so it is rejected even when compress_ratio carries the
		// sparsity.
		//
		// A window is the third bound. DeepSeek-V4.1's window-only layers (compress ratio
		// 0, and its DSpark draft layers) run the same sparse FlashMLA kernel as its
		// compressed ones, with the window's positions as the indices and no extra cache
		// (the engine's models/deepseek_v41/nvidia/flashmla.py: tile_sched_swaonly,
		// indices=swa_indices). Stating them as mla would route them to a different
		// kernel's coefficients.
		if n.AttentionKind == AttentionSparseMLA {
			switch {
			case n.IndexTopK < 0:
				p.Field(at+".index_topk", "must be positive, got %d", n.IndexTopK)
			case n.IndexTopK < 1 && n.CompressRatio < 1 && n.Window < 1:
				p.Field(at+".index_topk",
					"sparse MLA requires a positive index_topk, compress_ratio or window")
			}
		}
		forbid(p, at, map[string]int{"experts": n.Experts, "state_size": n.StateSize})
	case OpRecurrentUpdate:
		if !n.RecurrentKind.Valid() {
			p.Field(at+".recurrent_kind", "%q is not a recognized recurrent kind", n.RecurrentKind)
		}
		requirePositive(p, at, "n_heads", n.NumHeads)
		requirePositive(p, at, "state_size", n.StateSize)
		if n.RecurrentKind == RecurrentMamba2 {
			requirePositive(p, at, "n_groups", n.NumGroups)
			requirePositive(p, at, "conv_kernel", n.ConvKernel)
			requirePositive(p, at, "intermediate_size", n.IntermediateSize)
		}
		if n.StateDType != "" && !n.StateDType.Valid() {
			p.Field(at+".state_dtype", "%q is not a recognized dtype", n.StateDType)
		}
		forbid(p, at, map[string]int{"experts": n.Experts, "n_q": n.NumQHeads})
	case OpElementwise:
		// Bytes per token may be derived from hidden size, so it is optional; a
		// negative value is still a mistake.
		if n.BytesPerToken < 0 {
			p.Field(at+".bytes_per_token", "must not be negative")
		}
	case OpGather:
		if n.Rows < 1 {
			p.Field(at+".rows", "must be positive for this op, got %d", n.Rows)
		}
		requirePositive(p, at, "row_bytes", n.RowBytes)
		requirePositive(p, at, "rows_per_token", n.RowsPerToken)
		requirePositive(p, at, "out_bytes_per_token", n.OutBytesPerToken)
		if int64(n.RowsPerToken) > n.Rows && n.Rows > 0 {
			p.Field(at+".rows_per_token", "%d exceeds the table's %d rows", n.RowsPerToken, n.Rows)
		}
		if !n.Residency.Valid() {
			p.Field(at+".residency", "%q is not a recognized residency", n.Residency)
		}
		if !n.ResidencyControl.Valid() {
			p.Field(at+".residency_control", "%q is not a recognized residency control", n.ResidencyControl)
		}
		if !n.Shard.Valid() {
			p.Field(at+".shard", "%q is not a recognized shard", n.Shard)
		}
		if n.Shard.Effective() == ShardHeads && n.RowsPerToken < 2 {
			p.Field(at+".shard", "a table split by heads needs one row per head per token, so rows_per_token of at least 2")
		}
		forbid(p, at, map[string]int{
			"n": n.N, "k": n.K, "experts": n.Experts, "n_q": n.NumQHeads,
			"n_kv": n.NumKVHeads, "state_size": n.StateSize,
		})
	case OpAllReduce, OpAllGather, OpReduceScatter, OpAll2All:
		if n.BytesPerToken < 0 {
			p.Field(at+".bytes_per_token", "must not be negative")
		}
		// A collective's byte volume follows from the activation shape and the
		// layout, so it carries no shape of its own. Anything else is a mistake.
		forbid(p, at, map[string]int{
			"n": n.N, "k": n.K, "experts": n.Experts, "n_q": n.NumQHeads,
			"n_kv": n.NumKVHeads, "state_size": n.StateSize,
		})
	}
}

func requirePositive(p *validate.Problems, at, field string, v int) {
	if v < 1 {
		p.Field(at+"."+field, "must be positive for this op, got %d", v)
	}
}

// forbid reports any parameter set on a node whose op does not price it. Silence
// would let a typo — a head count on a GEMM — survive review looking deliberate.
func forbid(p *validate.Problems, at string, fields map[string]int) {
	for name, v := range fields {
		if v != 0 {
			p.Field(at+"."+name, "not priced by this op; remove it")
		}
	}
}

func validateContext(p *validate.Problems, at string, c ContextWork) {
	if c.ReadsCache == "" {
		requirePositive(p, at, "bytes_per_entry", c.BytesPerEntry)
	} else {
		if c.BytesPerEntry < 0 {
			p.Field(at+".bytes_per_entry", "must not be negative")
		}
		if c.TokensPerEntry != 0 {
			p.Field(at+".tokens_per_entry", "is the read cache's; remove it")
		}
	}
	if c.TokensPerEntry < 0 {
		p.Field(at+".tokens_per_entry", "must not be negative")
	}
	if !c.Per.Valid() {
		p.Field(at+".per", "%q is not a recognized unit; use query_token or request", c.Per)
	}
	if !c.Phase.Valid() {
		p.Field(at+".phase", "%q is not a recognized phase", c.Phase)
	}
	if !c.DecodeExtent.Valid() {
		p.Field(at+".decode_extent", "%q is not a recognized extent", c.DecodeExtent)
	}
	// Work that runs only in prefill has no decode extent to state.
	if c.Phase == PhasePrefill && c.DecodeExtent != ExtentContext {
		p.Field(at+".decode_extent", "stated on work that runs only in prefill")
	}
}

// validateLayerCacheRefs resolves the caches a layer's work names -- a cache insert's
// writes_cache, a context read's reads_cache -- against the caches the same layer states,
// and ties per-entry work to a stream the layer has. Work done once every N tokens is
// done once per compressed entry, so N must be a compress ratio the layer reads at, or,
// for an insert, the tokens per entry of the cache it writes.
func validateLayerCacheRefs(p *validate.Problems, at string, lk LayerKind) {
	caches := map[string]Cache{}
	ratios := map[int]bool{}
	for _, n := range lk.Nodes {
		for _, c := range n.Caches {
			caches[c.Name] = c
		}
		if n.CompressRatio > 0 {
			ratios[n.CompressRatio] = true
		}
	}
	for j, n := range lk.Nodes {
		nat := fmt.Sprintf("%s.nodes[%d]", at, j)
		if n.WritesCache != "" {
			c, ok := caches[n.WritesCache]
			switch {
			case !ok:
				p.Field(nat+".writes_cache", "%q names no cache of this layer", n.WritesCache)
			case !c.Owned:
				p.Field(nat+".writes_cache", "%q is read from another layer, which writes it", n.WritesCache)
			case max(n.PerTokens, 1) != c.EntryTokens():
				p.Field(nat+".per_tokens", "is %d but %q holds one entry per %d tokens", max(n.PerTokens, 1), c.Name, c.EntryTokens())
			}
		} else if n.PerTokens > 1 && !ratios[n.PerTokens] {
			p.Field(nat+".per_tokens", "%d is no compress ratio this layer reads at", n.PerTokens)
		}
		if n.Context != nil && n.Context.ReadsCache != "" {
			if _, ok := caches[n.Context.ReadsCache]; !ok {
				p.Field(nat+".context.reads_cache", "%q names no cache of this layer", n.Context.ReadsCache)
			}
		}
	}
}

// validateOutsideLayers refuses, on an embedding, head or speculator head node, the keys
// that tie work to a layer's caches: those nodes run outside every layer, so there is no cache for them to
// own, read or write, and no compressed stream for per-entry work to follow.
func validateOutsideLayers(p *validate.Problems, at string, n Node) {
	if len(n.Caches) > 0 {
		p.Field(at+".caches", "only a node in a layer kind reads a cache")
	}
	if n.WritesCache != "" {
		p.Field(at+".writes_cache", "only a node in a layer kind writes a cache")
	}
	if n.Context != nil && n.Context.ReadsCache != "" {
		p.Field(at+".context.reads_cache", "only a node in a layer kind reads a cache")
	}
	if n.PerTokens > 1 {
		p.Field(at+".per_tokens", "only a layer's per-entry work runs once per several tokens")
	}
}

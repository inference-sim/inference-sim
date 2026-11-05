package model

// Graph is a model expressed as cost primitives: a set of distinct layer kinds,
// a repetition pattern over them, and optional head and speculator stacks.
//
// The pattern is not a flat list of layers. A 72-layer uniform stack is
// {Pattern: ["attn_moe"], Repeat: 72}; a hybrid stack that interleaves three
// recurrent layers with one attention layer is
// {Pattern: ["mamba_moe","mamba_moe","mamba_moe","attn_moe"], Repeat: 10}. That
// mirrors how an engine itself resolves a hybrid model — a per-index type vector
// selecting a decoder-layer class — and keeps a 200-layer model a few lines rather
// than two hundred.
type Graph struct {
	Kind string `yaml:"kind"` // "ModelGraph"
	Name string `yaml:"name"` // matches the blis-catalog models/ directory

	// DerivedFrom records the vendor file this graph was translated from, and its
	// digest. The graph is a derived artifact, so it can fall out of step with its
	// source; the digest makes that detectable instead of silent.
	DerivedFrom Derivation `yaml:"derived_from"`

	Global     GlobalShape `yaml:"global"`
	LayerKinds []LayerKind `yaml:"layer_kinds"`
	// Embedding is the nodes that run once per scheduled token before the stack: the
	// vocabulary lookup, as a Gather. Absent means the consumer infers a vocab_size x
	// hidden_size table at global.weight_dtype, the meaning every graph had before the
	// field existed -- which is wrong for a checkpoint whose embedding is wider than the
	// rest: DeepSeek-V4.1's is BF16 beside FP8 weights, because the FP8 quant config
	// returns no method for an embedding and the engine falls back to an unquantized one
	// (the engine's vocab_parallel_embedding.py).
	Embedding  []Node      `yaml:"embedding,omitempty"`
	Stack      Stack       `yaml:"stack"`
	Head       []Node      `yaml:"head,omitempty"`
	Speculator *Speculator `yaml:"speculator,omitempty"`

	// ArchOverrides restate facts that differ by GPU architecture, applied for the
	// deployment's chip by ForCapability. See ArchOverride.
	ArchOverrides []ArchOverride `yaml:"arch_overrides,omitempty"`
	// BaseCapabilityFamilies are the CUDA compute-capability major versions the base
	// graph describes as served, without an override. Required with ArchOverrides: a
	// graph that distinguishes architectures must also say which ones its base is, so a
	// family it describes neither way is refused rather than given the base's facts.
	// DeepSeek-V4.1's base is SM90; SM120 runs a plain-row KV cache under FlashInfer
	// (models/deepseek_v41/nvidia/model.py _select_dsv4_attn_cls), so the SM90
	// records do not describe it.
	BaseCapabilityFamilies []int `yaml:"base_capability_families,omitempty"`

	// Modality records what a graph covers when its source describes more than a
	// language model. Several shipped configurations nest the text shapes under a
	// sub-object alongside a vision or audio tower; a graph derived from one of those
	// describes the decoder alone unless it says otherwise.
	//
	// The field exists so the omission is explicit. A cost model that silently
	// ignored an encoder would under-predict a multimodal request's prefill, and a
	// reader comparing a prediction against a measured run of the full model would
	// have no way to see why.
	Modality Modality `yaml:"modality,omitempty"`
}

// Modality states which towers a graph prices.
type Modality string

const (
	// ModalityTextOnly: the decoder stack only. This is the default when the field
	// is absent, because every graph this schema can currently express is a decoder.
	ModalityTextOnly Modality = "text_only"
	// ModalityTextDecoderOfMultimodal: the decoder of a model whose source also
	// describes an encoder that this graph does NOT price. A prediction from such a
	// graph is valid for text-only requests and understates any request carrying an
	// image or audio input.
	ModalityTextDecoderOfMultimodal Modality = "text_decoder_of_multimodal"
)

var modalities = map[Modality]bool{
	"": true, ModalityTextOnly: true, ModalityTextDecoderOfMultimodal: true,
}

// Valid reports whether m is a recognized modality. The empty value is valid and
// means text-only.
func (m Modality) Valid() bool { return modalities[m] }

// PricesEncoder reports whether the graph accounts for a non-text tower. It is
// always false today: the primitive set has no encoder term, so a graph can declare
// that it omits one but cannot declare that it includes one.
func (m Modality) PricesEncoder() bool { return false }

// Derivation identifies the vendor artifact a graph came from without adopting its
// vocabulary. Format names the dialect (for example "hf_config_json") so a reader
// knows which deriver produced this graph; the schema itself stays neutral.
//
// A deriver cannot assume a field exists just because it usually does. Depth is the
// clearest case: most configurations state num_hidden_layers, some spell it n_layer,
// and at least one shipped hybrid states neither — its depth is the length of a
// per-layer type vector, and a deriver reading only the usual key would produce a
// zero-layer model. The same applies to head dimension, which many configurations
// omit because it is hidden size over head count.
//
// This is why a graph carries a digest and a deriver version rather than being
// recomputed on demand: the translation has judgement in it, and a committed artifact
// can be reviewed where a silent recomputation cannot.
type Derivation struct {
	Format         string `yaml:"format"`
	Path           string `yaml:"path"`
	SHA256         string `yaml:"sha256"`
	DeriverVersion int    `yaml:"deriver_version"`
}

// GlobalShape holds the few quantities that are properties of the whole model
// rather than of one layer.
type GlobalShape struct {
	HiddenSize        int  `yaml:"hidden_size"`
	VocabSize         int  `yaml:"vocab_size"`
	TieWordEmbeddings bool `yaml:"tie_word_embeddings"`
	// WeightDType is the dtype parameters are stored in. It sets weight bytes and,
	// through them, both per-rank occupancy and decode-time HBM traffic.
	WeightDType DType `yaml:"weight_dtype"`
	// WeightScale is the resident scale layout of weights stored at WeightDType, where
	// the format does not fix one itself. See WeightScale and WeightBytesPerParam.
	WeightScale *WeightScale `yaml:"weight_scale,omitempty"`

	// ContextWindow is the longest context the engine admits for this model when the
	// deployment does not raise the limit: the derived max_model_len of the engine's
	// ModelConfig, the smallest of the configuration's length keys
	// (max_position_embeddings and its aliases), scaled by a RoPE factor for scaling
	// types other than su, longrope, llama3, yarn, deepseek_yarn and
	// deepseek_llama_scaling, for a pooling model with absolute positions capped by the
	// tokenizer's model_max_length, and capped by the sliding window when the deployment
	// disables sliding-window attention (config/model.py _get_and_verify_max_len). A
	// deployment that states a longer max_model_len is refused at startup unless the
	// configuration's model_max_length admits it or the engine's long-context override is
	// set. Zero means not stated.
	//
	// It is what an unstated or auto max_model_len starts from, and the graph carries it
	// because nothing else a consumer holds does: the vendor configuration is not a
	// schema input.
	ContextWindow int `yaml:"context_window,omitempty"`
	// DefaultContextWindow is the window an unstated or auto max_model_len takes, where
	// it differs from ContextWindow: for a LongRoPE model the engine starts from
	// original_max_position_embeddings to avoid degrading short sequences. Zero means it
	// equals ContextWindow.
	DefaultContextWindow int `yaml:"default_context_window,omitempty"`

	// Architecture is the model class the engine serves the checkpoint with, the
	// configuration's architectures[0], such as "DeepseekV41ForCausalLM". It is the one
	// vendor name a graph carries, because the engine keys model-specific behaviour on
	// it and nothing else in the graph identifies the model code: the engine refuses
	// dual-batch overlap only for a DeepseekV41ForCausalLM with Engram layers, and
	// DeepSeek-V4.1 decides its own sequence parallelism. Which architectures a release
	// treats specially is its rules pack's to say. Empty means not stated.
	Architecture string `yaml:"architecture,omitempty"`

	// PipelineBoundaryBytesPerToken is what one token costs to hand from one pipeline
	// stage to the next: the intermediate tensors the engine sends at a stage boundary.
	// Zero means vLLM's standard pair, hidden_states and residual, each one hidden-width
	// activation: 2 x hidden_size x 2 bytes a token
	// (make_empty_intermediate_tensors_factory(["hidden_states", "residual"], ...),
	// vllm/model_executor/models/llama.py:399-401 at v0.32.0). A model that sends
	// something else states it: Muse Spark folds its residual into one fp32 stream and
	// sends that alone, so a boundary moves 4 x hidden_size bytes a token
	// (_make_empty_intermediate_tensors, vllm/models/muse_spark/nvidia/model.py:1639-1650
	// at vllm-project/vllm 0f6b2619). Auxiliary hidden states a drafter reads
	// (Speculator.AuxHiddenStateLayers) are not part of this figure.
	PipelineBoundaryBytesPerToken int `yaml:"pipeline_boundary_bytes_per_token,omitempty"`
}

// DefaultWindow returns the window an unstated or auto max_model_len starts from, or zero
// when the graph states none.
func (g GlobalShape) DefaultWindow() int {
	if g.DefaultContextWindow > 0 {
		return g.DefaultContextWindow
	}
	return g.ContextWindow
}

// DType is a numeric format. Only formats that change a byte count or a peak rate
// appear; a format that merely changes rounding does not belong in a cost schema.
type DType string

const (
	DTypeBF16 DType = "bf16"
	DTypeFP16 DType = "fp16"
	DTypeFP8  DType = "fp8"
	// DTypeNVFP4 is NVIDIA's 4-bit float: a 16-element block sharing an FP8 scale,
	// with native Tensor-Core support on Blackwell.
	DTypeNVFP4 DType = "nvfp4"
	// DTypeMXFP4 is the OCP microscaling 4-bit float: a 32-element block sharing an
	// E8M0 scale. It is a distinct format from NVFP4, not a spelling of it — the block
	// size and scale type differ, so the two reach different rates on the same part,
	// and a checkpoint in one cannot be priced as the other.
	DTypeMXFP4 DType = "mxfp4"
	DTypeINT8  DType = "int8"
	// DTypeINT4 is 4-bit integer weights with a per-group scale, the W4A16 form
	// compressed-tensors emits as num_bits 4 with type "int" (Kimi-K2.5 ships this way at
	// group_size 32). It is a distinct format from NVFP4 and MXFP4, not a spelling of
	// either: an integer grid dequantizes through a different path than a float one, so
	// the three reach different rates on the same part even where the payload width
	// matches. Storage is four bits of payload, as for the 4-bit floats.
	DTypeINT4 DType = "int4"
	DTypeFP32 DType = "fp32"
)

var dtypes = map[DType]bool{
	DTypeBF16: true, DTypeFP16: true, DTypeFP8: true, DTypeNVFP4: true,
	DTypeMXFP4: true, DTypeINT8: true, DTypeINT4: true, DTypeFP32: true,
}

// Valid reports whether d is a recognized dtype.
func (d DType) Valid() bool { return dtypes[d] }

// Bytes returns the payload width of one element, excluding scales. NVFP4 is four
// bits, so it returns a fractional value expressed in bytes; callers multiply by an
// element count and round at the end rather than per element. For the bytes a weight
// occupies, scales included, use StoredBytes or Graph.WeightBytesPerParam.
func (d DType) Bytes() float64 {
	switch d {
	case DTypeFP32:
		return 4
	case DTypeBF16, DTypeFP16:
		return 2
	case DTypeFP8, DTypeINT8:
		return 1
	case DTypeNVFP4, DTypeMXFP4, DTypeINT4:
		// Four bits of payload. Each format adds a per-group scale -- one FP8 byte per
		// sixteen elements for NVFP4, one E8M0 byte per thirty-two for MXFP4, and for INT4
		// one scale per group at whatever group_size the checkpoint declares -- which
		// StoredBytes adds; this returns the payload width.
		return 0.5
	}
	return 0
}

// LayerKind is one distinct layer, as a small DAG over primitives.
type LayerKind struct {
	ID    string   `yaml:"id"`
	Nodes []Node   `yaml:"nodes"`
	Edges [][2]int `yaml:"edges"` // indices into Nodes; a DAG, checked acyclic
}

// Stack is the layer sequence. Three fields, because real stacks come in three
// shapes and forcing them into one would either lose information or bloat the file.
//
// Prologue is a literal run of layer kinds before the repeating part, and Epilogue a
// literal run after it. They exist because several shipped models are not periodic:
// one opens with three dense layers before seventy-five sparse ones, and another
// carries a 108-entry layer vector with no repeating unit at all. A schema offering
// only Pattern and Repeat would force the first into a 78-entry pattern and could
// not express the second without abandoning the compression entirely.
//
// A uniform stack is Pattern with Repeat and nothing else. A hybrid with a regular
// period is the same. A model with a prologue states it. A model with no periodicity
// puts its whole vector in Prologue and leaves Repeat at zero.
type Stack struct {
	Prologue []string `yaml:"prologue,omitempty"`
	Pattern  []string `yaml:"pattern,omitempty"`
	Repeat   int      `yaml:"repeat,omitempty"`
	Epilogue []string `yaml:"epilogue,omitempty"`
}

// Layers returns the total layer count the stack expands to.
func (s Stack) Layers() int {
	return len(s.Prologue) + len(s.Pattern)*s.Repeat + len(s.Epilogue)
}

// Expand returns the full layer-kind sequence. A cost model walks this rather than
// re-deriving the expansion, so the expansion lives in one place.
func (s Stack) Expand() []string {
	out := make([]string, 0, s.Layers())
	out = append(out, s.Prologue...)
	for i := 0; i < s.Repeat; i++ {
		out = append(out, s.Pattern...)
	}
	return append(out, s.Epilogue...)
}

// Speculator is a draft module, represented as its own stack rather than as a
// token count. For an MoE target the draft pass is a second MoE, and a scalar
// draft length hides that cost entirely.
type Speculator struct {
	// Method is the engine's speculative method name. It is carried as a string
	// because the set of methods is engine-version data, not a schema invariant;
	// a rules pack validates it against a declared engine version.
	Method string `yaml:"method"`
	// NumSpec is the tokens drafted per request per step. It is not the draft layer
	// count: DeepSeek-V4.1's DSpark drafts dspark_block_size = 5 tokens through
	// num_nextn_predict_layers = 3 layers (both from the checkpoint's configuration;
	// the engine's config/speculative.py defaults num_speculative_tokens to the draft
	// config's block_size, or dspark_block_size where it has none).
	NumSpec int `yaml:"num_spec_tokens"`
	// Drafting is how Stack runs. See Drafting.
	Drafting Drafting `yaml:"drafting,omitempty"`
	// Embedding is the nodes that run before each pass of Stack, over the same rows: the
	// draft's own token lookup and the reduction that completes it. DSpark's drafter
	// holds its own VocabParallelEmbedding, with the target's weights ("Shared with the
	// target", the engine's models/deepseek_v41/nvidia/dspark.py), so its Gather aliases
	// embedding.<role>. Like every vocab-parallel lookup it ends in a tensor-parallel
	// all-reduce (vocab_parallel_embedding.py), and the draft embeds before it shards its
	// stream for sequence parallelism (dspark.py forward), so that reduction runs in both
	// modes. Stated at the head of a draft layer kind instead, it would read as a
	// decoder-layer reduction, which the mode replaces. Absent means the draft does no
	// lookup of its own.
	Embedding []Node `yaml:"embedding,omitempty"`
	// Stack is the draft layers. Under autoregressive drafting it runs once per drafted
	// token; under parallel_block drafting it runs once per step over NumSpec positions
	// per request.
	Stack Stack `yaml:"stack"`
	// ContextStack is draft work that runs once per step over the TARGET's scheduled
	// tokens, prefill tokens included, rather than over draft positions. DSpark projects
	// three target layers' hidden states through main_proj, and each draft layer
	// derives its context KV from that projection, over every target token the step
	// schedules (v1/worker/gpu/spec_decode/dflash/speculator.py
	// _precompute_context_kv over [0, num_target_tokens)).
	// Absent means the speculator has no such work.
	ContextStack *Stack `yaml:"context_stack,omitempty"`
	// Head is the nodes after each pass of Stack, over the same rows: the draft's
	// logits. A node here that reuses the target's LM head says so with aliases.
	Head []Node `yaml:"head,omitempty"`
	// SequentialHead is the nodes that run once per draft position, in order, over one
	// row per request, after Head: DSpark's Markov head, which gathers a rank-256 row
	// for the previous token and projects it to the vocabulary to bias the next
	// position's logits, one position at a time (model_executor/models/
	// qwen3_dspark.py, "the head runs sequentially for every draft position"). Only a
	// parallel_block drafter has one; an autoregressive drafter is sequential already.
	SequentialHead []Node `yaml:"sequential_head,omitempty"`
	// AuxHiddenStateLayers are the target layers whose hidden states the drafter reads,
	// in the engine's capture convention: an id L is the state entering decoder layer L
	// (0 the embedding output, num_hidden_layers the last layer's output), as
	// set_aux_hidden_state_layers receives them. An EAGLE3, DFlash or dense-DSpark
	// drafter's configuration states its target_layer_ids one lower, and the engine adds
	// one (get_eagle3_aux_layers_from_config,
	// vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:98-135 at v0.32.0).
	//
	// Under pipeline parallelism a captured state travels with the intermediate tensors
	// from the stage that captures it to the last stage, where the drafter runs: each
	// boundary carries every state captured upstream of it (pack_local_aux_hidden_states
	// and collect_remote_aux_hidden_states, vllm/model_executor/models/interfaces.py:
	// 1569-1645; relay_aux_hidden_states, vllm/v1/worker/gpu/pp_utils.py:102-123).
	// Absent means the drafter reads none. Each id is in [0, target layers], and the
	// list is strictly increasing.
	AuxHiddenStateLayers []int `yaml:"aux_hidden_state_layers,omitempty"`
	// AuxHiddenStateBytesPerToken is one captured state's bytes a token. Zero means one
	// hidden-width activation, which is what a model that carries hidden_states and
	// residual apart captures (their sum, _maybe_add_hidden_state, interfaces.py:
	// 1600-1611). A target whose residual stream is wider states it: Muse Spark captures
	// its fp32 stream, 4 x hidden_size bytes (vllm/models/muse_spark/nvidia/
	// model.py:1705-1722 at vllm-project/vllm 0f6b2619). Only with AuxHiddenStateLayers.
	AuxHiddenStateBytesPerToken int `yaml:"aux_hidden_state_bytes_per_token,omitempty"`
}

// ShardedAtDecode is when a replicated GEMM's output columns split over the tensor-parallel
// ranks (Node.ShardedAtDecode). Both bounds are on the step's scheduled tokens.
type ShardedAtDecode struct {
	// CapabilityFamily is the CUDA compute-capability major version from which the engine
	// shards: 10 for SM100 and later (has_device_capability(100)).
	CapabilityFamily int `yaml:"capability_family"`
	// MaxTokens is the largest step that shards; a larger one runs the GEMM whole.
	MaxTokens int `yaml:"max_tokens"`
	// MinTokensAcrossNodes is the smallest step that shards when the tensor-parallel group
	// spans nodes, where the gather crosses the fabric: Muse Spark's unpooler shards from
	// 32 tokens then (_MULTI_NODE_SHARDED_UNPOOLER_MIN_TOKENS, model.py:154, 535-539). Zero
	// means from one token.
	MinTokensAcrossNodes int `yaml:"min_tokens_across_nodes,omitempty"`
}

// Drafting is how a speculator's stack produces its draft tokens.
type Drafting string

const (
	// DraftingAutoregressive is the zero value: the stack runs once per drafted token,
	// each pass feeding the next, as MTP and EAGLE drafters do.
	DraftingAutoregressive Drafting = ""
	// DraftingParallelBlock: the stack runs ONCE per step over num_spec_tokens
	// positions per request -- the anchor and num_spec_tokens - 1 noise tokens -- and
	// every position predicts one draft token. DSpark drafts this way: "each request
	// emits exactly N = num_speculative_tokens query tokens (anchor + N-1 noise), NOT
	// 1 + N" (v1/worker/gpu/spec_decode/dspark/speculator.py).
	DraftingParallelBlock Drafting = "parallel_block"
)

var draftings = map[Drafting]bool{DraftingAutoregressive: true, DraftingParallelBlock: true}

// Valid reports whether d is a recognized drafting mode.
func (d Drafting) Valid() bool { return draftings[d] }

// EmitCondition names when a conditional node is part of the graph. A resolver maps each
// member to a decision from the resolved layout; the graph states which question to ask,
// never how to answer it.
type EmitCondition string

const (
	// EmitAlways is the zero value: the node is unconditional.
	EmitAlways EmitCondition = ""
	// EmitTensorParallel: present when the tensor-parallel width exceeds one. The
	// attention-output and mixer-output reductions.
	EmitTensorParallel EmitCondition = "tensor_parallel"
	// EmitExpertParallel: present when the expert-parallel width exceeds one. The MoE
	// dispatch and combine.
	EmitExpertParallel EmitCondition = "expert_parallel"
	// EmitTensorParallelUnlessSequenceParallelMoE: present when the tensor-parallel
	// width exceeds one AND the engine has not made the MoE input sequence-parallel.
	// Under sequence-parallel MoE the layer's reduction is replaced by a
	// reduce-scatter and all-gather pair, so this node and that pair are alternatives
	// rather than both running.
	EmitTensorParallelUnlessSequenceParallelMoE EmitCondition = "tensor_parallel_unless_sp_moe"
	// EmitSequenceParallel: present when the model's own sequence parallelism is on, as a
	// rules pack decides it for the architecture global.architecture names. DeepSeek-V4.1
	// then shards the residual stream by token across the tensor-parallel ranks,
	// all-gathers each attention input and reduce-scatters its output (the engine's models/deepseek_v41/nvidia/model.py:
	// "x = sp_all_gather(x)" ... "x = sp_reduce_scatter(x)").
	EmitSequenceParallel EmitCondition = "sequence_parallel"
	// EmitTensorParallelUnlessSequenceParallel: present when the tensor-parallel width
	// exceeds one and the model's sequence parallelism is off. The all-reduces that
	// sequence parallelism replaces carry it.
	EmitTensorParallelUnlessSequenceParallel EmitCondition = "tensor_parallel_unless_sequence_parallel"
	// EmitTensorParallelWithoutExpertParallel: present when the tensor-parallel width
	// exceeds one and expert parallelism is off, so the MoE layer's experts are sharded by
	// tensor parallelism and their partial outputs need a reduction of their own. A model
	// whose MoE reduces a different tensor in each mode states one collective per mode:
	// Muse Spark all-reduces its top_k per-route rows before its post-expert norm under
	// tensor parallelism, and only the combined latent-width output under expert
	// parallelism.
	EmitTensorParallelWithoutExpertParallel EmitCondition = "tensor_parallel_without_expert_parallel"
	// EmitExpertParallelWithoutDispatch: present when expert parallelism is on, the
	// tensor-parallel width exceeds one, and the MoE layer dispatches no tokens: every rank
	// holds every token, because the expert-parallel group is the tensor-parallel group, so
	// the ranks' partial expert outputs combine by an all-reduce rather than an
	// all-to-all.
	EmitExpertParallelWithoutDispatch EmitCondition = "expert_parallel_without_dispatch"
)

var emitConditions = map[EmitCondition]bool{
	EmitAlways: true, EmitTensorParallel: true, EmitExpertParallel: true,
	EmitTensorParallelUnlessSequenceParallelMoE: true, EmitSequenceParallel: true,
	EmitTensorParallelUnlessSequenceParallel: true, EmitTensorParallelWithoutExpertParallel: true,
	EmitExpertParallelWithoutDispatch: true,
}

// SequenceParallelOnly reports whether c is one of the two conditions decided by the
// model's own sequence parallelism. They belong on the collectives the mode adds or
// replaces. Work belongs on a node stated once, with UnderSequenceParallel for how the
// mode divides it: at tp 1 neither condition holds, so work stated twice, once under
// each, would vanish there.
func (c EmitCondition) SequenceParallelOnly() bool {
	return c == EmitSequenceParallel || c == EmitTensorParallelUnlessSequenceParallel
}

// Valid reports whether c is a recognized condition.
func (c EmitCondition) Valid() bool { return emitConditions[c] }

// SequenceParallelField names the field of n that the model's own sequence parallelism
// decides -- "emit" or "under_sequence_parallel" -- or returns "" when it decides none.
func (n Node) SequenceParallelField() string {
	switch {
	case n.Emit.SequenceParallelOnly():
		return "emit"
	case n.UnderSequenceParallel != SPAsTP:
		return "under_sequence_parallel"
	}
	return ""
}

// Node is one primitive with the shape parameters it prices. The fields are a
// union across ops: a GEMM reads N and K, an Attention reads the head counts, a
// RecurrentUpdate reads the state dimensions. Validation rejects a node carrying
// parameters its op does not price, so the union cannot hide a mistake.
type Node struct {
	Op   Op     `yaml:"op"`
	Role string `yaml:"role,omitempty"` // human label, e.g. "qkv_proj"

	// Emit names the condition under which this node exists. Empty means
	// unconditional. A collective node must carry one, because whether a collective
	// runs depends on the parallelism a deployment chooses rather than on the model.
	//
	// It is a closed enum rather than an expression. An earlier draft wrote predicates
	// as strings — "tp > 1 and not sp_moe" — which would have required the kernel to
	// carry an expression parser, and would have let a graph express a condition no
	// resolver knows how to evaluate. Every condition the catalog's models need is one
	// of three, and each is decided by the collective's role rather than by arbitrary
	// arithmetic, so the enum loses nothing and the resolver becomes a switch.
	Emit EmitCondition `yaml:"emit,omitempty"`

	// TP is how the node divides across tensor-parallel ranks. Absent means the
	// consumer's inference, the meaning every graph had before the field existed. See
	// TPLayout.
	TP TPLayout `yaml:"tp,omitempty"`

	// UnderSequenceParallel is how the node divides when the model's own sequence
	// parallelism is on. Empty means as TP states, over every token the step schedules.
	// See SequenceParallelSplit.
	UnderSequenceParallel SequenceParallelSplit `yaml:"under_sequence_parallel,omitempty"`

	// Aliases, on a GEMM or Gather, names the node whose weights this one reads, as
	// "<layer kind id>.<role>", "head.<role>" or "embedding.<role>". The node is priced
	// for its compute and its weight-read traffic but adds no resident bytes, because the
	// weights it reads are already counted where they live. DSpark's draft reads the
	// target's LM head and embedding (dspark.py: "load_dspark_model always aliases the
	// target's"), so pricing the draft's own copies would count each twice.
	Aliases string `yaml:"aliases,omitempty"`

	// ShardedAtDecode, on a replicated GEMM (tp: replicated), says the engine splits its output
	// columns over the tensor-parallel ranks for a small step and gathers them back, holding the
	// whole weight on every rank: each rank computes n/tp columns and an all-gather over TP of
	// the step's rows x n output at the activation width reassembles them. Absent means the
	// GEMM runs whole on every rank at every step. Muse Spark's MoE pooler and unpooler are
	// such ReplicatedLinears on Blackwell with tensor parallelism and no data parallelism, at
	// 256 tokens or fewer (shard_pooler / shard_unpooler, _SHARDED_PROJECTION_MAX_TOKENS,
	// vllm/models/muse_spark/nvidia/model.py:151, 505-523, 635-660 at vllm-project/vllm
	// 0f6b2619). The condition is the model code's, so a consumer reads it from the
	// architecture and the layout; this field only marks the GEMMs it applies to.
	ShardedAtDecode *ShardedAtDecode `yaml:"sharded_at_decode,omitempty"`

	// GEMM and GroupedGEMM.
	N int `yaml:"n,omitempty"`
	K int `yaml:"k,omitempty"`
	// WeightScale overrides the resident scale layout for this node's weights. See
	// WeightScale.
	WeightScale *WeightScale `yaml:"weight_scale,omitempty"`
	// WeightDType overrides GlobalShape.WeightDType for this node's parameters, for a
	// checkpoint that stores one part of itself at a different width from the rest.
	// Empty means the global dtype governs, which is the common case.
	//
	// Mixed-precision MoE is why this exists. DeepSeek-V4-Pro declares expert_dtype fp4
	// beside an fp8 quantization_config: the routed experts are 4-bit and everything else
	// is 8-bit. Pricing the experts at the global width doubled them -- 1,441 GiB against
	// 720 GiB for 384 experts -- which put a per-rank figure of 180 GiB on a 141 GiB part
	// and made a deployment InferenceX actually ran look impossible. A global dtype cannot
	// express that split, and inferring it from the quantization method would be a guess
	// about a width the checkpoint states.
	WeightDType DType `yaml:"weight_dtype,omitempty"`
	// GroupedGEMM only.
	Experts int `yaml:"experts,omitempty"`
	TopK    int `yaml:"top_k,omitempty"`
	// SharedExperts are experts every token passes through, in addition to the
	// TopK routed ones. They are a distinct cost: their work is dense rather than
	// routed, so it neither scales with TopK nor contributes to routing imbalance.
	// Whether an engine fuses them into the grouped GEMM or runs them separately
	// changes the kernel but not the byte count.
	SharedExperts int `yaml:"shared_experts,omitempty"`
	// SharedIntermediateSize is the shared experts' inner dimension where it differs
	// from N, as it commonly does. Zero means it equals N.
	SharedIntermediateSize int `yaml:"shared_intermediate_size,omitempty"`
	// LatentSize reduces the expert input dimension below the model's hidden size:
	// a projection to a narrower space runs before the routed experts, so K for the
	// expert matmul is this rather than HiddenSize. Zero means no reduction. It
	// changes both the expert FLOP count and the all-to-all volume, which is why it
	// is a node parameter rather than a global one.
	LatentSize int `yaml:"latent_size,omitempty"`

	// Attention.
	AttentionKind AttentionKind `yaml:"kind,omitempty"`
	NumQHeads     int           `yaml:"n_q,omitempty"`
	NumKVHeads    int           `yaml:"n_kv,omitempty"`
	HeadDim       int           `yaml:"d_h,omitempty"`
	// MLA-specific latent dimensions; zero for other kinds.
	KVLoRARank    int `yaml:"kv_lora_rank,omitempty"`
	QKRopeHeadDim int `yaml:"qk_rope_head_dim,omitempty"`
	// SWA window, in tokens; zero for other kinds.
	Window int `yaml:"window,omitempty"`
	// SparseMLA top-k index count; zero for other kinds.
	IndexTopK int `yaml:"index_topk,omitempty"`
	// CompressRatio is the integer factor by which a compressed sparse-MLA layer shrinks
	// its latent KV read -- DeepSeek-V4-Pro's csa4_moe reads 1/4 of the latent cache and
	// csa128_moe 1/128. The catalog expresses it as a whole number (4 and 128), so it is
	// an int rather than a float; how it combines with the layer's dimensions is a pricing
	// concern for the simulator, not a cross-field invariant this field-level schema
	// checks. Being omitempty, a zero is an unset value, the convention every optional
	// shape field here follows, so a set (nonzero) ratio is what validation checks for
	// positivity and sparse-MLA placement; zero for other kinds.
	CompressRatio int `yaml:"compress_ratio,omitempty"`
	// Caches are the per-request caches this attention node reads, and which of them its
	// layer owns. Absent means the consumer infers one cache from the read shape, the
	// meaning every graph had before the field existed. See Cache.
	Caches []Cache `yaml:"caches,omitempty"`
	// KVHeadsNotReplicated, on an Attention node, says its projection cannot replicate a KV
	// head across tensor-parallel ranks: the tensor-parallel width must divide n_kv. vLLM's
	// standard QKV projection replicates a head over ranks when tp exceeds n_kv
	// (QKVParallelLinear num_kv_head_replicas, vllm/model_executor/layers/linear.py:1050-1058
	// at v0.32.0), so a deployment at tp 32 over 16 KV heads starts; a fused projection that
	// slices by head group does not, and the engine asserts n_kv % tp == 0 at load. Muse
	// Spark's fused [Q, K, V, O-gate] projection is one ("KV-head padding is not supported",
	// MuseSpark1xQKVOGateParallelLinear, vllm/models/muse_spark/common/layers.py:339-371 at
	// vllm-project/vllm 0f6b2619). Absent means the standard projection.
	KVHeadsNotReplicated bool `yaml:"kv_heads_not_replicated,omitempty"`

	// RecurrentUpdate.
	RecurrentKind    RecurrentKind `yaml:"recurrent_kind,omitempty"`
	NumHeads         int           `yaml:"n_heads,omitempty"`
	StateSize        int           `yaml:"state_size,omitempty"`
	NumGroups        int           `yaml:"n_groups,omitempty"`
	ConvKernel       int           `yaml:"conv_kernel,omitempty"`
	IntermediateSize int           `yaml:"intermediate_size,omitempty"`
	// StateDType is the dtype the recurrent state is stored in, which is often
	// wider than the weight dtype.
	StateDType DType `yaml:"state_dtype,omitempty"`

	// Gather. See OpGather.
	//
	// Rows is the table's row count and RowBytes one stored row, scales included.
	// RowsPerToken is the rows gathered per token, and OutBytesPerToken what the gather
	// writes to device memory per token after any dequantize.
	Rows             int64 `yaml:"rows,omitempty"`
	RowBytes         int   `yaml:"row_bytes,omitempty"`
	RowsPerToken     int   `yaml:"rows_per_token,omitempty"`
	OutBytesPerToken int   `yaml:"out_bytes_per_token,omitempty"`
	// Residency is where the table lives by the engine's default. Empty means device.
	Residency Residency `yaml:"residency,omitempty"`
	// ResidencyControl names the engine setting that can move the table, which a
	// deployment states. Empty means the residency is fixed.
	ResidencyControl ResidencyControl `yaml:"residency_control,omitempty"`
	// Shard is how the table divides across tensor-parallel ranks. Empty means
	// replicated.
	Shard GatherShard `yaml:"shard,omitempty"`

	// Elementwise: bytes touched per token, when it is not derivable from
	// HiddenSize alone.
	//
	// AllReduce, AllGather and ReduceScatter: the bytes per token of the collective's full
	// tensor -- what every rank holds after an all-gather or all-reduce, and before a
	// reduce-scatter -- where it is not one hidden-width activation. DeepSeek-V4.1's
	// engram row all-gather moves 24 BF16 rows of 256 per token, 12,288 B, against a
	// 10,240 B hidden activation. Absent means the hidden-width default.
	BytesPerToken int `yaml:"bytes_per_token,omitempty"`

	// PerTokens, on an Elementwise node, makes it run once every PerTokens tokens rather
	// than once per token: work done once per compressed entry, such as V4.1's pooled
	// compressor write and its index-key store, which exit early on every token but the
	// last of a compress_ratio group (fused_compress_quant_cache.py, indexer_k_store.py:
	// "if (position + 1) % COMPRESS_RATIO != 0: return"). Zero means one.
	//
	// It is not offered on a GEMM, because the engine runs none of these GEMMs per entry:
	// V4.1's compressor projection and its index wk run over every token's row and the
	// store kernel then skips the non-boundary rows (attention.py: "Rows at non-boundary
	// tokens hold garbage latent and are skipped by the store kernel"). A per-entry GEMM
	// would leave out (r - 1) / r of the FLOPs the engine spends at ratio r.
	PerTokens int `yaml:"per_tokens,omitempty"`

	// Context, on an Elementwise node, is traffic that scales with the attended context
	// rather than with the scheduled tokens. It is in addition to BytesPerToken. See
	// ContextWork.
	Context *ContextWork `yaml:"context,omitempty"`

	// WritesCache, on an Elementwise node in a layer kind, names a cache of the same layer
	// this node writes one entry of each time it runs: a cache insert. The entry's record
	// -- under the cache dtype that runs, on the resolved architecture -- adds to
	// BytesPerToken, which is then the traffic beyond the record, and PerTokens must be
	// that cache's tokens_per_entry. Naming the cache keeps the write's size with the
	// record it writes rather than restating a number an arch override would leave behind.
	WritesCache string `yaml:"writes_cache,omitempty"`
}

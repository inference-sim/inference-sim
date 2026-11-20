package model

// ContextWork is memory traffic an Elementwise node does in proportion to the attended
// context rather than to the tokens scheduled this step.
//
// One scalar cannot state it, because context-proportional work comes in two scalings
// that differ by orders of magnitude, and DeepSeek-V4.1-Flash on the engine runs both:
//
//   - Per query token. Layer 20's candidate-block scoring and the candidate mask of
//     layers 24, 28, 32 and 36 launch one row per query token over the whole logits
//     width (model_executor/kernels/attention/dsa/candidate_blocks.py), so a
//     prefill chunk of 8,192 tokens over a 128K context touches 8,192 x 128K positions.
//   - Per request. The prefill path dequantizes a request's whole compressed stream
//     into a BF16 workspace once per step, however many of its tokens the step
//     schedules (models/deepseek_v41/nvidia/flashmla.py, dequantize_and_gather_k_cache
//     over seq_lens // compress_ratio).
//
// The extent can also differ by phase. In decode the logits width is padded to the
// indexer's max_model_len // compress_ratio (utils/deep_gemm.py, "[B * next_n,
// max_model_len]"), and the mask kernel writes every column of it, so decode-time mask
// traffic follows max_model_len, not the context.
type ContextWork struct {
	// ReadsCache names a cache of the same layer whose entries this work reads. Each entry
	// then costs that cache's record -- under the cache dtype that runs, on the
	// architecture ForCapability resolved -- plus BytesPerEntry, and an entry covers that
	// cache's tokens_per_entry. Naming the cache rather than stating its record is what
	// keeps the figure right when the record moves: V4.1's prefill dequantize reads a
	// 584 B record on SM90, 528 B on SM100 and 288 B under nvfp4_ds_mla, and writes 1,024
	// BF16 bytes in every case. Empty means BytesPerEntry is the whole traffic.
	ReadsCache string `yaml:"reads_cache,omitempty"`
	// BytesPerEntry is the traffic per context entry, per unit of Per, reads and writes
	// together -- beyond the named cache's record when ReadsCache is set, where it may be
	// zero.
	BytesPerEntry int `yaml:"bytes_per_entry"`
	// TokensPerEntry is how many context tokens one entry covers, the compress ratio of
	// the stream the work walks. Zero means one. With ReadsCache it is the cache's and
	// is not stated.
	TokensPerEntry int `yaml:"tokens_per_entry,omitempty"`
	// Per is what multiplies the context. See ContextPer.
	Per ContextPer `yaml:"per"`
	// Phase restricts the work to steps where the request is prefilling. Empty means
	// every step.
	Phase Phase `yaml:"phase,omitempty"`
	// DecodeExtent names what a decode step's work covers. Empty means the request's
	// context; ExtentMaxModelLen means the engine's max_model_len, for a kernel launched
	// over a buffer padded to it.
	DecodeExtent Extent `yaml:"decode_extent,omitempty"`
}

// ContextPer is the unit a context-proportional cost is charged per.
type ContextPer string

const (
	// PerQueryToken: once per token scheduled this step, each over the context.
	PerQueryToken ContextPer = "query_token"
	// PerRequest: once per request in the step, over its context, independent of how
	// many of its tokens are scheduled.
	PerRequest ContextPer = "request"
)

var contextPers = map[ContextPer]bool{PerQueryToken: true, PerRequest: true}

// Valid reports whether c is a recognized unit.
func (c ContextPer) Valid() bool { return contextPers[c] }

// Phase restricts work to part of a request's life.
type Phase string

const (
	// PhaseAll is the zero value: every step the request is scheduled in.
	PhaseAll Phase = ""
	// PhasePrefill: only steps that schedule prompt tokens of the request, including
	// every chunk of a chunked prefill.
	PhasePrefill Phase = "prefill"
)

var phases = map[Phase]bool{PhaseAll: true, PhasePrefill: true}

// Valid reports whether p is a recognized phase.
func (p Phase) Valid() bool { return phases[p] }

// Extent is how far along the context a kernel's work reaches.
type Extent string

const (
	// ExtentContext is the zero value: the request's own context.
	ExtentContext Extent = ""
	// ExtentMaxModelLen: the engine's max_model_len, whatever the context, because the
	// kernel runs over a buffer padded to it.
	ExtentMaxModelLen Extent = "max_model_len"
)

var extents = map[Extent]bool{ExtentContext: true, ExtentMaxModelLen: true}

// Valid reports whether e is a recognized extent.
func (e Extent) Valid() bool { return extents[e] }

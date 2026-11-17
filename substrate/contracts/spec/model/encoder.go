package model

import (
	"fmt"
	"regexp"

	validate "github.com/inference-sim/blis-substrate/contracts/check"
)

var issueRef = regexp.MustCompile(`^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+#[1-9][0-9]*$`)

// Encoder is the non-text towers a graph prices: the vision tower of a multimodal
// checkpoint, which the engine runs over an image's patches before the decoder reads
// the tower's output tokens as part of the prompt. It is a section of its own because
// a tower is not a layer of the decoder stack: it runs once per image, over the image's
// patches, with bidirectional attention and no KV cache, and its output joins the
// prompt as tokens.
//
// A graph without an Encoder prices the decoder alone. With one, Modality says whether
// every tower the checkpoint describes is priced (ModalityMultimodal) or some are
// listed in Omitted (ModalityTextDecoderOfMultimodal).
type Encoder struct {
	Towers []Tower `yaml:"towers"`
	// Omitted lists each tower of the source that Towers does not price, with the open
	// issue that tracks it. A request carrying that input is refused, not priced as text.
	Omitted []OmittedTower `yaml:"omitted,omitempty"`
}

// TowerModality is the kind of input a tower encodes.
type TowerModality string

const (
	// TowerImage is a vision tower. Stills and video frames both enter it as patches.
	TowerImage TowerModality = "image"
	// TowerAudio is an audio tower. No graph prices one yet; it appears only in Omitted.
	TowerAudio TowerModality = "audio"
)

// Valid reports whether m is a recognized tower modality.
func (m TowerModality) Valid() bool { return m == TowerImage || m == TowerAudio }

// OmittedTower is a tower of the source that the graph does not price.
type OmittedTower struct {
	Modality TowerModality `yaml:"modality"`
	// Reason says what the kernels cannot express, in one sentence.
	Reason string `yaml:"reason"`
	// Issue is the open issue that tracks the gap (X28), as "owner/repo#N".
	Issue string `yaml:"issue"`
}

// EncoderParallel is how the engine divides the tower's weights over the tensor-parallel
// ranks by default.
type EncoderParallel string

const (
	// EncoderTensor: the tower's linears are column- and row-parallel, as vLLM's default
	// mm_encoder_tp_mode "weights" builds them. When the head count does not divide the
	// tensor-parallel width, or the deployment states mm_encoder_tp_mode "data", the engine
	// replicates the tower instead; that is lowering and the engine's to decide.
	EncoderTensor EncoderParallel = "tensor"
	// EncoderReplicated: the model code builds every linear of the tower replicated, so
	// every rank holds the whole tower and no deployment setting shards it (Gemma 4, whose
	// tower converts every nn.Linear to a ReplicatedLinear).
	EncoderReplicated EncoderParallel = "replicated"
)

// Valid reports whether e is a recognized tower parallel mode.
func (e EncoderParallel) Valid() bool { return e == EncoderTensor || e == EncoderReplicated }

// Tower is one vision tower: its patching, its layers and how its output joins the prompt.
type Tower struct {
	Modality TowerModality `yaml:"modality"`
	Input    TowerInput    `yaml:"input"`
	// MergeRatio is the encoder rows that become one output token: the spatial merge
	// squared (Qwen 2x2 is 4), a pooling window (Gemma 4's 3x3 is 9), or 1.
	MergeRatio int `yaml:"merge_ratio"`
	// OutputHiddenSize is the width of each output token, which is the text hidden size
	// because the decoder reads the tokens as embeddings.
	OutputHiddenSize int `yaml:"output_hidden_size"`
	// MaxOutputTokens is the most tokens one image yields when the processor shrinks a
	// larger image to fit (Gemma 4's 280, DeepSeek-V4.1's 1024). Zero means the checkpoint
	// states none.
	MaxOutputTokens int `yaml:"max_output_tokens,omitempty"`
	// Parallel is how the weights divide by default. See EncoderParallel.
	Parallel EncoderParallel `yaml:"parallel"`
	// WeightDType is the dtype the tower's tensors are stored in, from the checkpoint's
	// headers. A node may override it. A tower stays in a wider type than the decoder
	// when the engine builds it without the quantization config.
	WeightDType DType `yaml:"weight_dtype"`
	// Embedding runs once over the image's rows before the stack: patch projection and
	// position embedding.
	Embedding  []EncoderNode      `yaml:"embedding,omitempty"`
	LayerKinds []EncoderLayerKind `yaml:"layer_kinds"`
	Stack      Stack              `yaml:"stack"`
	// Projector runs after the stack: the merger or projector that ends at
	// OutputHiddenSize.
	Projector []EncoderNode `yaml:"projector,omitempty"`
}

// TowerInput says how pixels become the tower's rows.
type TowerInput struct {
	// PatchSize is the pixels on one side of a patch.
	PatchSize int `yaml:"patch_size"`
	// TemporalPatchSize is the frames one patch spans; 1 for a tower with no temporal
	// patching. A still image occupies one temporal patch.
	TemporalPatchSize int `yaml:"temporal_patch_size,omitempty"`
	Channels          int `yaml:"channels"`
	// TilePixels, when positive, says the image is cut into square tiles of this many
	// pixels a side, each encoded as its own attention sequence, with one extra tile of the
	// whole image when there is more than one (Llama 4). Zero means one grid per image.
	TilePixels int `yaml:"tile_pixels,omitempty"`
	// ClassTokens is the rows a tile or grid adds beyond its patches, such as a class token.
	// They are encoded but yield no output token.
	ClassTokens int `yaml:"class_tokens,omitempty"`
	// FixedRows, when positive, is the rows the tower encodes for every image whatever its
	// size, because the processor pads to it (Gemma 4: 280 tokens x 9 = 2520).
	FixedRows int `yaml:"fixed_rows,omitempty"`
}

// EncoderOp names a tower primitive. The tower needs three: a dense matrix multiply, a
// bidirectional attention with no KV cache, and bandwidth-bound work.
type EncoderOp string

const (
	EncoderGEMM        EncoderOp = "gemm"
	EncoderAttention   EncoderOp = "attention"
	EncoderElementwise EncoderOp = "elementwise"
)

// Valid reports whether o is a recognized tower op.
func (o EncoderOp) Valid() bool {
	return o == EncoderGEMM || o == EncoderAttention || o == EncoderElementwise
}

// EncoderRows is which row count a tower node runs over.
type EncoderRows string

const (
	// RowsPatches is the rows the tower encodes: the patches of the image, plus class
	// tokens. Everything up to the merge runs on these.
	RowsPatches EncoderRows = "patches"
	// RowsOutputTokens is the rows after the merge: the projector's GEMMs after a spatial
	// merge run on one row per output token.
	RowsOutputTokens EncoderRows = "output_tokens"
)

// Valid reports whether r is a recognized row count.
func (r EncoderRows) Valid() bool { return r == RowsPatches || r == RowsOutputTokens }

// EncoderLayerKind is one distinct tower layer.
type EncoderLayerKind struct {
	ID    string        `yaml:"id"`
	Nodes []EncoderNode `yaml:"nodes"`
}

// EncoderNode is one primitive of a tower. Like Node it is a union across ops, and
// validation refuses a node carrying parameters its op does not price.
type EncoderNode struct {
	Op   EncoderOp `yaml:"op"`
	Role string    `yaml:"role,omitempty"`
	// Rows is the row count the node runs over. Absent means patches.
	Rows EncoderRows `yaml:"rows,omitempty"`
	// TP is how the node divides when the tower is EncoderTensor. Absent means
	// replicated. A gemm is column or row, an attention is sharded by heads, an
	// elementwise is sharded or replicated.
	TP TPLayout `yaml:"tp,omitempty"`
	// WeightDType overrides Tower.WeightDType for this node.
	WeightDType DType `yaml:"weight_dtype,omitempty"`

	// gemm: an N-by-K weight, with an N-long bias when Bias is set.
	N    int  `yaml:"n,omitempty"`
	K    int  `yaml:"k,omitempty"`
	Bias bool `yaml:"bias,omitempty"`

	// attention: full bidirectional attention with n_heads heads of head_dim over each
	// attention sequence. Window, when positive, limits a sequence to blocks of that many
	// rows (Muse Spark's window layers). The tower has no KV cache.
	NumHeads int `yaml:"n_heads,omitempty"`
	HeadDim  int `yaml:"d_h,omitempty"`
	Window   int `yaml:"window,omitempty"`

	// elementwise: BytesPerRow is the traffic per row (read plus write), and Params the
	// stored elements the node owns (norm scales, a position table, a class token).
	BytesPerRow int   `yaml:"bytes_per_row,omitempty"`
	Params      int64 `yaml:"params,omitempty"`
}

// ParamCount is the stored elements the node owns.
func (n EncoderNode) ParamCount() int64 {
	switch n.Op {
	case EncoderGEMM:
		c := int64(n.N) * int64(n.K)
		if n.Bias {
			c += int64(n.N)
		}
		return c
	case EncoderElementwise:
		return n.Params
	}
	return 0
}

// ImageRows is what the tower encodes for one image: Patches rows in Segments attention
// sequences of equal length.
type ImageRows struct {
	Patches  int
	Segments int
}

func ceilDiv(a, b int) int { return (a + b - 1) / b }

// Rows evaluates the rows the tower encodes for an image of heightPx by widthPx pixels
// and frames frames (1 for a still), as the processor has already resized it. Sizes
// below one pixel are one pixel.
func (t Tower) Rows(heightPx, widthPx, frames int) ImageRows {
	grid, seg := t.gridRows(heightPx, widthPx, frames)
	if t.Input.FixedRows > 0 {
		return ImageRows{Patches: t.Input.FixedRows, Segments: 1}
	}
	if t.Input.TilePixels == 0 && t.MaxOutputTokens > 0 {
		if limit := t.MaxOutputTokens * t.MergeRatio; grid > limit {
			grid = limit
		}
	}
	return ImageRows{Patches: grid, Segments: seg}
}

// gridRows is the rows and sequences before any fixed padding or cap.
func (t Tower) gridRows(heightPx, widthPx, frames int) (rows, segments int) {
	if heightPx < 1 {
		heightPx = 1
	}
	if widthPx < 1 {
		widthPx = 1
	}
	if frames < 1 {
		frames = 1
	}
	in := t.Input
	if in.PatchSize < 1 {
		return 0, 0
	}
	if in.TilePixels > 0 {
		tiles := ceilDiv(heightPx, in.TilePixels) * ceilDiv(widthPx, in.TilePixels)
		if tiles > 1 {
			tiles++
		}
		per := (in.TilePixels/in.PatchSize)*(in.TilePixels/in.PatchSize) + in.ClassTokens
		return tiles * per, tiles
	}
	tp := in.TemporalPatchSize
	if tp < 1 {
		tp = 1
	}
	return ceilDiv(heightPx, in.PatchSize)*ceilDiv(widthPx, in.PatchSize)*ceilDiv(frames, tp) + in.ClassTokens, 1
}

// Patches is Rows(...).Patches: the rows the tower encodes for the image.
func (t Tower) Patches(heightPx, widthPx, frames int) int {
	return t.Rows(heightPx, widthPx, frames).Patches
}

// OutputTokens is the tokens the image adds to the prompt: its grid rows (class tokens
// excluded) divided by the merge ratio, rounded up and capped at MaxOutputTokens.
func (t Tower) OutputTokens(heightPx, widthPx, frames int) int {
	grid, seg := t.gridRows(heightPx, widthPx, frames)
	if grid == 0 || t.MergeRatio < 1 {
		return 0
	}
	grid -= t.Input.ClassTokens * seg
	if t.Input.TilePixels == 0 && t.MaxOutputTokens > 0 {
		if limit := t.MaxOutputTokens * t.MergeRatio; grid > limit {
			grid = limit
		}
	}
	return ceilDiv(grid, t.MergeRatio)
}

// OutputBytes is what tokens output tokens occupy as the embeddings the decoder reads
// (bf16), which is what an encoder-to-prefill handoff moves.
func (t Tower) OutputBytes(tokens int) int64 {
	return int64(tokens) * int64(t.OutputHiddenSize) * 2
}

// Params is the stored elements of the whole tower.
func (t Tower) Params() int64 {
	var c int64
	for _, n := range t.Embedding {
		c += n.ParamCount()
	}
	byID := map[string]EncoderLayerKind{}
	for _, lk := range t.LayerKinds {
		byID[lk.ID] = lk
	}
	for _, id := range t.Stack.Expand() {
		for _, n := range byID[id].Nodes {
			c += n.ParamCount()
		}
	}
	for _, n := range t.Projector {
		c += n.ParamCount()
	}
	return c
}

// WeightBytes is the bytes the tower's tensors occupy as stored (payload only), summed
// over its nodes at each node's dtype.
func (t Tower) WeightBytes() float64 {
	b := 0.0
	add := func(n EncoderNode) {
		d := n.WeightDType
		if d == "" {
			d = t.WeightDType
		}
		b += float64(n.ParamCount()) * d.Bytes()
	}
	for _, n := range t.Embedding {
		add(n)
	}
	byID := map[string]EncoderLayerKind{}
	for _, lk := range t.LayerKinds {
		byID[lk.ID] = lk
	}
	for _, id := range t.Stack.Expand() {
		for _, n := range byID[id].Nodes {
			add(n)
		}
	}
	for _, n := range t.Projector {
		add(n)
	}
	return b
}

// PricesEncoder reports whether the graph prices at least one tower.
func (g *Graph) PricesEncoder() bool { return g.Encoder != nil && len(g.Encoder.Towers) > 0 }

// ImageTower returns the graph's priced image tower, if it has one.
func (g *Graph) ImageTower() (Tower, bool) {
	if g.Encoder == nil {
		return Tower{}, false
	}
	for _, t := range g.Encoder.Towers {
		if t.Modality == TowerImage {
			return t, true
		}
	}
	return Tower{}, false
}

func (g *Graph) validateEncoder(p *validate.Problems) {
	e := g.Encoder
	if e == nil {
		if g.Modality == ModalityMultimodal {
			p.Field("modality", "%q needs an encoder section that prices the towers", g.Modality)
		}
		return
	}
	if len(e.Towers) == 0 && len(e.Omitted) == 0 {
		p.Field("encoder.towers", "at least one tower, or an omitted tower with its issue, is required")
	}
	switch {
	case len(e.Omitted) == 0 && g.Modality != ModalityMultimodal:
		p.Field("modality", "a graph with an encoder and no omitted tower must state %q, got %q",
			ModalityMultimodal, g.Modality)
	case len(e.Omitted) > 0 && g.Modality != ModalityTextDecoderOfMultimodal:
		p.Field("modality", "a graph that omits a tower must state %q, got %q",
			ModalityTextDecoderOfMultimodal, g.Modality)
	}
	seen := map[TowerModality]bool{}
	for i, t := range e.Towers {
		at := fmt.Sprintf("encoder.towers[%d]", i)
		if t.Modality != TowerImage {
			p.Field(at+".modality", "%q cannot be priced: only image towers are (list others under omitted)", t.Modality)
		}
		if seen[t.Modality] {
			p.Field(at+".modality", "duplicate %q tower", t.Modality)
		}
		seen[t.Modality] = true
		t.validate(p, at, g.Global.HiddenSize)
	}
	for i, o := range e.Omitted {
		at := fmt.Sprintf("encoder.omitted[%d]", i)
		if !o.Modality.Valid() {
			p.Field(at+".modality", "%q is not a recognized tower modality", o.Modality)
		}
		if seen[o.Modality] {
			p.Field(at+".modality", "%q is both priced and omitted", o.Modality)
		}
		if o.Reason == "" {
			p.Field(at+".reason", "required")
		}
		if !issueRef.MatchString(o.Issue) {
			p.Field(at+".issue", "must name the open issue that tracks the gap as owner/repo#N, got %q", o.Issue)
		}
	}
}

func (t Tower) validate(p *validate.Problems, at string, textHidden int) {
	in := t.Input
	requirePositive(p, at+".input", "patch_size", in.PatchSize)
	requirePositive(p, at+".input", "channels", in.Channels)
	if in.TemporalPatchSize < 0 {
		p.Field(at+".input.temporal_patch_size", "must not be negative")
	}
	if in.ClassTokens < 0 || in.FixedRows < 0 || in.TilePixels < 0 {
		p.Field(at+".input", "class_tokens, fixed_rows and tile_pixels must not be negative")
	}
	if in.TilePixels > 0 && in.TilePixels%in.PatchSize != 0 {
		p.Field(at+".input.tile_pixels", "%d is not a multiple of patch_size %d", in.TilePixels, in.PatchSize)
	}
	if in.TilePixels > 0 && in.FixedRows > 0 {
		p.Field(at+".input", "tile_pixels and fixed_rows both say how many rows an image has; state one")
	}
	if t.MergeRatio < 1 {
		p.Field(at+".merge_ratio", "must be at least 1, got %d", t.MergeRatio)
	}
	if t.MaxOutputTokens < 0 {
		p.Field(at+".max_output_tokens", "must not be negative")
	}
	if t.OutputHiddenSize != textHidden {
		p.Field(at+".output_hidden_size", "%d differs from global.hidden_size %d: the decoder reads the tower's output as embeddings",
			t.OutputHiddenSize, textHidden)
	}
	if !t.Parallel.Valid() {
		p.Field(at+".parallel", "%q is not a recognized tower parallel mode", t.Parallel)
	}
	if !t.WeightDType.Valid() {
		p.Field(at+".weight_dtype", "%q is not a recognized dtype", t.WeightDType)
	}
	for i, n := range t.Embedding {
		n.validate(p, fmt.Sprintf("%s.embedding[%d]", at, i))
	}
	for i, n := range t.Projector {
		n.validate(p, fmt.Sprintf("%s.projector[%d]", at, i))
	}
	if len(t.LayerKinds) == 0 {
		p.Field(at+".layer_kinds", "at least one layer kind is required")
	}
	ids := map[string]bool{}
	for i, lk := range t.LayerKinds {
		lat := fmt.Sprintf("%s.layer_kinds[%d]", at, i)
		if lk.ID == "" {
			p.Field(lat+".id", "required")
		} else if ids[lk.ID] {
			p.Field(lat+".id", "duplicate layer-kind id %q", lk.ID)
		}
		ids[lk.ID] = true
		if len(lk.Nodes) == 0 {
			p.Field(lat+".nodes", "a layer kind with no nodes prices nothing")
		}
		for j, n := range lk.Nodes {
			n.validate(p, fmt.Sprintf("%s.nodes[%d]", lat, j))
		}
	}
	if t.Stack.Layers() == 0 {
		p.Field(at+".stack", "expands to no layers")
	}
	if t.Stack.Repeat < 0 {
		p.Field(at+".stack.repeat", "must not be negative")
	}
	used := map[string]bool{}
	for _, id := range append(append(append([]string{}, t.Stack.Prologue...), t.Stack.Pattern...), t.Stack.Epilogue...) {
		if !ids[id] {
			p.Field(at+".stack", "names layer kind %q, which layer_kinds does not define", id)
		}
		used[id] = true
	}
	for _, lk := range t.LayerKinds {
		if lk.ID != "" && !used[lk.ID] {
			p.Field(at+".layer_kinds", "layer kind %q is not in the stack", lk.ID)
		}
	}
}

func (n EncoderNode) validate(p *validate.Problems, at string) {
	if !n.Op.Valid() {
		p.Field(at+".op", "%q is not a recognized tower op", n.Op)
		return
	}
	if n.Rows != "" && !n.Rows.Valid() {
		p.Field(at+".rows", "%q is not a recognized row count", n.Rows)
	}
	if n.WeightDType != "" && !n.WeightDType.Valid() {
		p.Field(at+".weight_dtype", "%q is not a recognized dtype", n.WeightDType)
	}
	refuse := func(field string, set bool) {
		if set {
			p.Field(at+"."+field, "is not priced by op %q", n.Op)
		}
	}
	gemm := n.N != 0 || n.K != 0 || n.Bias
	attn := n.NumHeads != 0 || n.HeadDim != 0 || n.Window != 0
	elem := n.BytesPerRow != 0 || n.Params != 0
	switch n.Op {
	case EncoderGEMM:
		requirePositive(p, at, "n", n.N)
		requirePositive(p, at, "k", n.K)
		refuse("n_heads/d_h/window", attn)
		refuse("bytes_per_row/params", elem)
		if !n.TP.ValidFor(OpGEMM) {
			p.Field(at+".tp", "%q is not a layout for a gemm", n.TP)
		}
	case EncoderAttention:
		requirePositive(p, at, "n_heads", n.NumHeads)
		requirePositive(p, at, "d_h", n.HeadDim)
		if n.Window < 0 {
			p.Field(at+".window", "must not be negative")
		}
		refuse("n/k/bias", gemm)
		refuse("bytes_per_row/params", elem)
		if !n.TP.ValidFor(OpAttention) {
			p.Field(at+".tp", "%q is not a layout for an attention", n.TP)
		}
	case EncoderElementwise:
		requirePositive(p, at, "bytes_per_row", n.BytesPerRow)
		if n.Params < 0 {
			p.Field(at+".params", "must not be negative")
		}
		refuse("n/k/bias", gemm)
		refuse("n_heads/d_h/window", attn)
		if !n.TP.ValidFor(OpElementwise) {
			p.Field(at+".tp", "%q is not a layout for an elementwise node", n.TP)
		}
	}
}

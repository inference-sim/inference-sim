package model

// WeightScale is the layout of the scales a block-scaled weight carries while it is
// resident: one scale of Bytes bytes for every Elements weight elements.
//
// DType.Bytes is payload alone, and the scales are not small enough to ignore or regular
// enough to guess. DeepSeek-V4.1-Flash's routed experts are MXFP4, one E8M0 byte per 32
// elements, so 0.53125 B per parameter rather than 0.5: 288.8 GB for the 40 target
// layers' 384 experts rather than 271.8. Its dense FP8 weights ship with one E8M0 byte
// per 32 x 32 block, and the engine expands those to one per 1 x 32 at load
// (modelopt.py KMxfp8Static, repeat_interleave over the block rows), so the resident
// overhead is 1/32 B per parameter, 32 times what the checkpoint's own scale tensors
// suggest. What is resident is what a cost model
// needs, so that is what this states.
type WeightScale struct {
	// Elements is how many weight elements share one scale: 32 for a 1 x 32 block, 16384
	// for 128 x 128.
	Elements int `yaml:"elements"`
	// Bytes is one scale's width: 1 for E8M0 or E4M3, 2 for BF16, 4 for FP32.
	Bytes int `yaml:"bytes"`
}

// PerElement returns the scale bytes one weight element carries.
func (s WeightScale) PerElement() float64 {
	if s.Elements < 1 {
		return 0
	}
	return float64(s.Bytes) / float64(s.Elements)
}

// IntrinsicScale returns the scale layout a format defines for itself, and false for a
// format that defines none. NVFP4 is a 16-element block with an FP8 (E4M3) scale, and
// MXFP4 a 32-element block with an E8M0 scale (OCP Microscaling Formats v1.0); those are
// part of what the names mean. NVFP4's single FP32 per-tensor scale is not counted, being
// four bytes per tensor. FP8, INT8 and INT4 are scaled per tensor, per channel, per group
// or per block depending on the checkpoint, so their layout is never assumed.
func (d DType) IntrinsicScale() (WeightScale, bool) {
	switch d {
	case DTypeNVFP4:
		return WeightScale{Elements: 16, Bytes: 1}, true
	case DTypeMXFP4:
		return WeightScale{Elements: 32, Bytes: 1}, true
	}
	return WeightScale{}, false
}

// StoredBytes returns the resident bytes of one element of dtype d with scale layout s,
// or with d's intrinsic layout when s is nil. A nil s on a format with no intrinsic
// layout counts payload alone, which is exact for an unscaled format and an understatement
// for a scaled one -- the reason WeightScale exists.
func StoredBytes(d DType, s *WeightScale) float64 {
	b := d.Bytes()
	if s != nil {
		return b + s.PerElement()
	}
	if is, ok := d.IntrinsicScale(); ok {
		return b + is.PerElement()
	}
	return b
}

// WeightBytesPerParam returns the resident bytes of one parameter of a GEMM or
// GroupedGEMM node, scales included: the one number the catalog, the kernel and a
// simulator should all agree on. Any other node holds no parameters, and gets zero; a
// Gather states its stored row width as row_bytes instead.
//
// The dtype is the node's override or the global one. The scale layout is the node's
// own; failing that, the global layout when the node is stored at the global dtype; and
// failing that, the dtype's intrinsic layout. The global layout does not follow a node
// stored in a different format, because a scale layout belongs to a format: DeepSeek-
// V4.1's BF16 router and FP32 mHC projections carry none of its FP8 weights' scales.
//
// An alias reads its target's weights, so it is priced at the target's format: the
// weight-read traffic of a DSpark head reading the BF16 LM head is BF16's, whatever the
// global dtype. An alias whose target does not resolve gets zero, as validation reports.
func (g *Graph) WeightBytesPerParam(n Node) float64 {
	if n.Aliases != "" {
		target, _, err := g.resolveAlias(n.Aliases)
		if err != nil || target.Aliases != "" {
			return 0
		}
		// A tied LM head reads its embedding table: k elements per row, stored in the
		// table's row width, scales included.
		if n.Op == OpGEMM && target.Op == OpGather && n.K > 0 {
			return float64(target.RowBytes) / float64(n.K)
		}
		n = target
	}
	if n.Op != OpGEMM && n.Op != OpGroupedGEMM {
		return 0
	}
	d := g.Global.WeightDType
	if n.WeightDType != "" {
		d = n.WeightDType
	}
	s := n.WeightScale
	if s == nil && d == g.Global.WeightDType {
		s = g.Global.WeightScale
	}
	return StoredBytes(d, s)
}

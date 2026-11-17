package latencykernel

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/kernel/internal/price"
)

// EncodeImage is one image as the processor has already resized it: pixels and frames (1
// for a still). The tower turns it into rows (model.Tower.Rows).
type EncodeImage struct {
	HeightPx, WidthPx, Frames int
}

// EncodeBatch is the images an engine step encodes together.
type EncodeBatch struct {
	Images []EncodeImage
}

// EncodeEstimate prices the encode work of one step.
type EncodeEstimate struct {
	// Total is the time of the batched run, GPU work only.
	Total time.Duration
	// PerImage splits Total over the images by each one's own standalone time, so the engine
	// step can place each image in the chunk it prefills. It sums to Total exactly.
	PerImage []time.Duration
	// OutputTokens is the prompt tokens each image adds.
	OutputTokens []int
	// HostPerImage is the per-image host cost (preprocess and dispatch) that the encode stage
	// charges apart from Total. It is zero when unmeasured, and HostPerImageNote says so.
	HostPerImage     time.Duration
	HostPerImageNote string
}

// encoderHostNote is what HostPerImageNote carries: no allowed source measures the host side
// of an image's preprocessing, so the kernel states none rather than inventing one.
const encoderHostNote = "unmeasured: no allowed source records a per-image preprocess or dispatch cost"

// encoderSubstitution is the named substitution the encoder is priced under (X19).
const encoderSubstitution = "the tower is priced with the decoder's GEMM efficiency ramp (BF16) and its measured " +
	"prefill-attention rate, bidirectional and with no KV cache; no allowed data measures an encoder step, so " +
	"every point with an encode step is capped, and the tower's collectives and fused kernels are not priced"

// imageTowerRun is the priced tower and how the engine runs it, nil when the kernel prices
// no encode step.
func (k *Kernel) imageTowerRun() (model.Tower, plan.EncoderMode, bool) {
	if k.encoder == nil || !k.encoder.Resident {
		return model.Tower{}, "", false
	}
	t, ok := k.graph.ImageTower()
	if !ok {
		return model.Tower{}, "", false
	}
	return t, k.encoder.Mode, true
}

// tensorShards is how many ways a node's weight divides on a rank.
func (k *Kernel) towerShards(t model.Tower, mode plan.EncoderMode, n model.EncoderNode) float64 {
	if t.Parallel == model.EncoderReplicated || mode != plan.EncoderWeightsSharded {
		return 1
	}
	switch n.TP {
	case model.TPColumn, model.TPRow, model.TPSharded:
		return math.Max(k.tp, 1)
	}
	return 1
}

// towerResidentBytes is the tower's weights on one rank.
func (k *Kernel) towerResidentBytes() float64 {
	t, mode, ok := k.imageTowerRun()
	if !ok {
		return 0
	}
	var total float64
	each := func(n model.EncoderNode) {
		d := n.WeightDType
		if d == "" {
			d = t.WeightDType
		}
		total += float64(n.ParamCount()) * d.Bytes() / k.towerShards(t, mode, n)
	}
	for _, n := range t.Embedding {
		each(n)
	}
	byID := map[string]model.EncoderLayerKind{}
	for _, lk := range t.LayerKinds {
		byID[lk.ID] = lk
	}
	for _, id := range t.Stack.Expand() {
		for _, n := range byID[id].Nodes {
			each(n)
		}
	}
	for _, n := range t.Projector {
		each(n)
	}
	return total
}

// EncodeTime prices the tower over the images of one step. A graph with no priced tower, or a
// deployment whose engine does not keep it resident (a decoder-only deployment), is refused by
// name: an image there is an error, not text.
func (k *Kernel) EncodeTime(b EncodeBatch) (EncodeEstimate, error) {
	t, mode, ok := k.imageTowerRun()
	if !ok {
		switch {
		case !k.graph.PricesEncoder():
			return EncodeEstimate{}, fmt.Errorf("model %q prices no vision tower, so an image cannot be encoded "+
				"(its graph holds the text decoder only)", k.graph.Name)
		case k.encoder == nil:
			return EncodeEstimate{}, fmt.Errorf("the engine states no encoder run for model %q, so the tower is not priced", k.graph.Name)
		default:
			return EncodeEstimate{}, fmt.Errorf("the engine serves model %q without its vision tower, so an image cannot be encoded", k.graph.Name)
		}
	}
	est := EncodeEstimate{HostPerImageNote: encoderHostNote}
	if len(b.Images) == 0 {
		return est, nil
	}
	standalone := make([]float64, len(b.Images))
	for i, im := range b.Images {
		est.OutputTokens = append(est.OutputTokens, t.OutputTokens(im.HeightPx, im.WidthPx, im.Frames))
		standalone[i] = k.towerSeconds(t, mode, []EncodeImage{im})
	}
	total := k.towerSeconds(t, mode, b.Images)
	est.Total = seconds(total)
	est.PerImage = splitDuration(est.Total, standalone)
	return est, nil
}

// splitDuration divides total over weights proportionally, exactly: the shares sum to total.
func splitDuration(total time.Duration, weights []float64) []time.Duration {
	out := make([]time.Duration, len(weights))
	var sum float64
	for _, w := range weights {
		sum += w
	}
	var given time.Duration
	for i, w := range weights {
		if i == len(weights)-1 {
			out[i] = total - given
			break
		}
		share := time.Duration(0)
		if sum > 0 {
			share = time.Duration(float64(total) * w / sum)
		} else {
			share = total / time.Duration(len(weights))
		}
		out[i] = share
		given += share
	}
	return out
}

// towerSeconds is the GPU time of the tower over these images on one rank.
func (k *Kernel) towerSeconds(t model.Tower, mode plan.EncoderMode, images []EncodeImage) float64 {
	// Under a batch split each rank encodes its share of the images; the busiest rank binds.
	if mode == plan.EncoderBatchSplit && t.Parallel != model.EncoderReplicated && k.tp > 1 {
		images = busiestShare(t, images, int(k.tp))
	}
	var patches, outRows float64
	var segs []int // attention sequence lengths, one entry per sequence
	for _, im := range images {
		r := t.Rows(im.HeightPx, im.WidthPx, im.Frames)
		patches += float64(r.Patches)
		if t.Input.FixedRows > 0 {
			outRows += float64(t.Input.FixedRows / max(t.MergeRatio, 1))
		} else {
			outRows += float64(t.OutputTokens(im.HeightPx, im.WidthPx, im.Frames))
		}
		for s := 0; s < r.Segments; s++ {
			segs = append(segs, r.Patches/max(r.Segments, 1))
		}
	}
	rowsOf := func(n model.EncoderNode) float64 {
		if n.Rows == model.RowsOutputTokens {
			return outRows
		}
		return patches
	}
	var total float64
	run := func(nodes []model.EncoderNode, times float64) {
		for _, n := range nodes {
			total += times * k.towerNodeSeconds(t, mode, n, rowsOf(n), segs)
		}
	}
	run(t.Embedding, 1)
	byID := map[string]model.EncoderLayerKind{}
	for _, lk := range t.LayerKinds {
		byID[lk.ID] = lk
	}
	for _, id := range t.Stack.Expand() {
		run(byID[id].Nodes, 1)
	}
	run(t.Projector, 1)
	return total
}

// busiestShare deals the images over ranks largest-first and returns the rank holding the
// most rows, which is the rank the step waits for.
func busiestShare(t model.Tower, images []EncodeImage, ranks int) []EncodeImage {
	if ranks <= 1 || len(images) <= 1 {
		return images
	}
	order := append([]EncodeImage(nil), images...)
	sort.SliceStable(order, func(i, j int) bool {
		return t.Patches(order[i].HeightPx, order[i].WidthPx, order[i].Frames) >
			t.Patches(order[j].HeightPx, order[j].WidthPx, order[j].Frames)
	})
	load := make([]int, ranks)
	held := make([][]EncodeImage, ranks)
	for _, im := range order {
		min := 0
		for r := 1; r < ranks; r++ {
			if load[r] < load[min] {
				min = r
			}
		}
		load[min] += t.Patches(im.HeightPx, im.WidthPx, im.Frames)
		held[min] = append(held[min], im)
	}
	busiest := 0
	for r := range load {
		if load[r] > load[busiest] {
			busiest = r
		}
	}
	return held[busiest]
}

// towerNodeSeconds prices one node over rows rows (and, for attention, the given sequences).
func (k *Kernel) towerNodeSeconds(t model.Tower, mode plan.EncoderMode, n model.EncoderNode, rows float64, segs []int) float64 {
	if rows <= 0 {
		return 0
	}
	div := k.towerShards(t, mode, n)
	switch n.Op {
	case model.EncoderGEMM:
		fit := k.encoderFit
		nn, kk := float64(n.N), float64(n.K)
		if div > 1 {
			if n.TP == model.TPColumn {
				nn /= div
			} else {
				kk /= div
			}
		}
		eff := price.Efficiency(rows, fit.epsMax, fit.mHalf)
		if fit.shapeAware {
			eff = price.ShapeEfficiency(rows, nn, kk, fit.epsMax, fit.mHalf, fit.nHalf, fit.kHalf)
		}
		if eff <= 0 || fit.peak <= 0 {
			return math.Inf(1)
		}
		compute := rows * 2 * nn * kk / (fit.peak * eff)
		d := n.WeightDType
		if d == "" {
			d = t.WeightDType
		}
		read := nn * kk * d.Bytes() / k.hbmBytesPerSecond
		return math.Max(compute, read)
	case model.EncoderAttention:
		heads := float64(n.NumHeads) / div
		var pairs float64
		for _, l := range segs {
			L := float64(l)
			if n.Window > 0 && L > float64(n.Window) {
				pairs += L * float64(n.Window)
			} else {
				pairs += L * L
			}
		}
		work := pairs * heads * float64(n.HeadDim)
		pair := k.prefillFor(model.AttentionGQA)
		if pair.scale <= 0 {
			fit := k.encoderFit
			eff := price.Efficiency(rows, fit.epsMax, fit.mHalf)
			if eff <= 0 || fit.peak <= 0 {
				return math.Inf(1)
			}
			return 4 * work / (fit.peak * eff)
		}
		scale := price.Efficiency(rows, k.epsMax, k.mHalf) * pair.scale
		floor := 0.0
		if len(segs) > 0 {
			floor = pair.floor.Seconds()
		}
		return floor + work/(k.computeFLOPsPerSecond*scale)
	case model.EncoderElementwise:
		return rows * float64(n.BytesPerRow) / div / k.hbmBytesPerSecond
	}
	return 0
}

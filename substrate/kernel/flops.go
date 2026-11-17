package latencykernel

import (
	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// StepFLOPs is the model FLOPs of one forward pass over b, counted from the plan the kernel
// prices (so there is one source for the model's shape): the whole model's work, before any
// parallel split, which is what a model-FLOPs-utilization figure divides by the devices'
// peak (inference-sim#2081).
//
// It counts, per scheduled token, every dense and expert GEMM of every layer (two FLOPs per
// multiply-accumulate; each token pays top_k experts and the shared expert), the language-model
// head over the rows that sample (every decode row, one row per prefill request, as the step's
// logits rows are), and each attention's two matmuls over the keys each query reads: four FLOPs
// per query, key, head and head-dim element, with a sliding window bounding the keys.
//
// It leaves out what is not a matmul or is priced from measured forms rather than FLOPs: the
// elementwise work, the recurrent and linear-attention mixers, a sparse layer's index scoring
// and top-k bound (its attention is counted over the window, or the whole context), the
// drafter's passes, and the data-parallel padding rows. A model that is mostly recurrent
// therefore reads low; the omission is the figure's, not an error.
func (k *Kernel) StepFLOPs(b kernel.Batch) float64 {
	var tokens, logitsRows float64
	for _, r := range b.Reqs {
		tokens += float64(r.Scheduled)
		if r.Computed >= r.PromptLen {
			logitsRows += float64(r.Scheduled)
		} else {
			logitsRows++
		}
	}
	var total float64
	for i := range k.plan.Layers {
		l := &k.plan.Layers[i]
		perToken := l.DenseFLOPsPerToken + float64(l.TopK)*l.ExpertFLOPsPerTokenPerExpert + l.SharedExpertFLOPsPerToken
		total += float64(l.Count) * perToken * tokens
		for _, a := range l.Attentions {
			if a.QHeads <= 0 || a.HeadDim <= 0 {
				continue
			}
			var pairs float64
			for _, r := range b.Reqs {
				pairs += queryKeyPairs(r.Computed, r.Scheduled, a.Window)
			}
			total += float64(l.Count) * 4 * float64(a.QHeads) * float64(a.HeadDim) * pairs
		}
	}
	total += k.plan.Head.DenseFLOPsPerToken * logitsRows
	return total
}

// queryKeyPairs is the (query, key) pairs of a chunk of n tokens that resumes after c computed
// ones: the token at position p attends to the p+1 positions up to and including its own, or
// to the window's last w of them when w is positive.
func queryKeyPairs(c, n, w int) float64 {
	if n <= 0 {
		return 0
	}
	fc, fn := float64(c), float64(n)
	if w <= 0 {
		return fn*fc + fn*(fn+1)/2
	}
	// Positions below the window attend to everything before them; the rest, to w.
	a := float64(min(max(w-c, 0), n))
	return a*fc + a*(a+1)/2 + (fn-a)*float64(w)
}

// PeakFLOPsPerGPU is the datasheet dense peak of one GPU at the format the model's linear
// layers run in, in FLOP/s. The model-FLOPs-utilization figure divides by it, not by the
// fitted ramp's asymptote the step is priced against.
func (k *Kernel) PeakFLOPsPerGPU() float64 {
	_, peak := dtypeFit(k.servedDType, k.chip)
	return peak
}

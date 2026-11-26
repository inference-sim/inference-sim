package decision

import (
	"fmt"
	"math"
)

// Layout kinds, as the stage's training data numbers them.
const (
	KindTP  = 0 // tensor parallel
	KindEP  = 1 // expert parallel
	KindDPA = 2 // data-parallel attention
	KindPD  = 3 // prefill and decode in separate pools
)

// Point is one simulated candidate of a case: a configuration at one concurrency, with the
// simulator's prediction of its objectives. Logarithms are natural. A setting the point does not
// state, or a case property not known, is NaN, which the trees route to their default side.
type Point struct {
	// ID names the point in the output.
	ID string

	// The simulator's predictions: log throughput per GPU, log interactivity (1000 over the P90
	// inter-token latency in ms) and log P90 time to first token in ms (lower is better).
	// A NaN is a value the simulator does not give.
	LogX, LogI, LogT float64

	// The configuration. Kind is one of the Kind constants. TP, EP, DP and GPUs are logarithms
	// of the parallel degrees and the GPU count; PD is 1 for a prefill/decode layout; Spec is 1
	// when speculative decoding is on; Seqs and Btok are logs of max_num_seqs and
	// max_num_batched_tokens; PDFrac is the prefill share of the GPUs of a PD layout and 0
	// otherwise; LConc is the log of the concurrency.
	Kind                         int
	TP, EP, DP, GPUs             float64
	PD, Spec, Seqs, Btok, PDFrac float64
	LConc                        float64

	// The case: log total and active parameters in billions, chip HBM, bandwidth and FP8 rate
	// (log), the engine (0 vLLM, 1 SGLang), the log input and output lengths and the source of
	// the point in the training data (0 for the simulator's points).
	MTot, MAct, CHBM, CBW, CFP8 float64
	Engine                      float64
	LISL, LOSL, Src             float64
}

// PairFeatureNames are the 29 pair features of the stage's training data, in order.
var PairFeatureNames = []string{
	"dX", "dI", "dT", "kind_a", "kind_b", "same_kind", "d_tp", "d_ep", "d_dp", "d_gpus", "d_pd", "d_spec",
	"d_seqs", "d_btok", "d_pdfrac", "lcr", "lcm", "m_tot", "m_act", "c_hbm", "c_bw", "c_fp8", "d_hbm",
	"d_bw", "d_fp8", "engine", "lisl", "losl", "src",
}

// StageFeatureNames are the 54 inputs of the fitted stage: the pair features,
// the same three simulator differences from three other simulator groups, the adjusted-TTFT
// difference, 13 columns of the calibration stage and the engine-version pair. This stage uses the
// first group's simulator differences (dX, dI, dT); the other columns it does not have are NaN.
var StageFeatureNames = func() []string {
	n := append([]string(nil), PairFeatureNames...)
	for _, g := range []string{"L1K", "L1KH", "L1F"} {
		for _, d := range []string{"dX", "dI", "dT"} {
			n = append(n, d+"_"+g)
		}
	}
	n = append(n, "dT_adj")
	n = append(n, "ref_a", "ref_b", "ddX", "ddI", "ddT", "sdX_a", "sdI_a", "sdT_a", "sdX_b", "sdI_b", "sdT_b", "pi_a", "pi_b")
	return append(n, "ver", "same_version")
}()

// notAvailable are the stage inputs that come from outside the simulator's own prediction of the
// point: other simulator groups, the adjusted TTFT, the calibration stage and the engine version.
// They are always missing here.
var notAvailable = func() map[string]bool {
	m := map[string]bool{}
	for _, n := range StageFeatureNames[len(PairFeatureNames):] {
		m[n] = true
	}
	return m
}()

// PairFeatures computes the named features of the ordered pair (a, b). A feature in
// PairFeatureNames is computed; one the stage does not have (see StageFeatureNames) is NaN;
// any other name is an error. The simulator enters only as differences, so adding a constant to
// every point's LogX, LogI or LogT changes no feature (DEC-4).
func PairFeatures(names []string, a, b Point) ([]float32, error) {
	out := make([]float32, len(names))
	for i, n := range names {
		v, ok := pairFeature(n, a, b)
		if !ok {
			if !notAvailable[n] {
				return nil, fmt.Errorf("unknown feature %q", n)
			}
			v = math.NaN()
		}
		out[i] = float32(v)
	}
	return out, nil
}

func boolF(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func pairFeature(name string, a, b Point) (float64, bool) {
	switch name {
	case "dX":
		return a.LogX - b.LogX, true
	case "dI":
		return a.LogI - b.LogI, true
	case "dT":
		return a.LogT - b.LogT, true
	case "kind_a":
		return float64(a.Kind), true
	case "kind_b":
		return float64(b.Kind), true
	case "same_kind":
		return boolF(a.Kind == b.Kind), true
	case "d_tp":
		return a.TP - b.TP, true
	case "d_ep":
		return a.EP - b.EP, true
	case "d_dp":
		return a.DP - b.DP, true
	case "d_gpus":
		return a.GPUs - b.GPUs, true
	case "d_pd":
		return a.PD - b.PD, true
	case "d_spec":
		return a.Spec - b.Spec, true
	case "d_seqs":
		return a.Seqs - b.Seqs, true
	case "d_btok":
		return a.Btok - b.Btok, true
	case "d_pdfrac":
		return a.PDFrac - b.PDFrac, true
	case "lcr":
		return a.LConc - b.LConc, true
	case "lcm":
		return (a.LConc + b.LConc) / 2, true
	case "m_tot":
		return a.MTot, true
	case "m_act":
		return a.MAct, true
	case "c_hbm":
		return a.CHBM, true
	case "c_bw":
		return a.CBW, true
	case "c_fp8":
		return a.CFP8, true
	case "d_hbm":
		return a.CHBM - b.CHBM, true
	case "d_bw":
		return a.CBW - b.CBW, true
	case "d_fp8":
		return a.CFP8 - b.CFP8, true
	case "engine":
		return a.Engine, true
	case "lisl":
		return a.LISL, true
	case "losl":
		return a.LOSL, true
	case "src":
		return a.Src, true
	}
	return 0, false
}

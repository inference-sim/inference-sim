package latency

import "fmt"

// moeCommFamily is the physical communication-volume family a vLLM MoE all-to-all
// backend belongs to. The trained-physics step-time model charges MoE dispatch/
// combine differently per family because the two move genuinely different byte
// volumes on the wire (verified against vllm@f6ec81c7):
//
//   - commFamilyAllGather: dispatch all-gathers / combine reduce-scatters the dense
//     per-token hidden states across the DP group (NaiveAll2AllManager.naive_multicast
//     and AgRsAll2AllManager all_gatherv/reduce_scatterv, all2all.py:38-175). Each
//     token's hidden state moves once per phase regardless of top_k → volume ∝
//     tokens·hidden, NO top_k factor.
//   - commFamilyAll2All: each token is routed to its top_k expert-owning ranks
//     (DeepEP/pplx point-to-point all-to-all kernels) → volume ∝ tokens·top_k·hidden.
//
// The two families share the same per-phase NVLink bus-bandwidth efficiency (NCCL
// busbw factor (n-1)/n for both all-gather/reduce-scatter and all-to-all; ring
// all-reduce is (n-1)/n×2 and is implemented as reduce-scatter+all-gather), so the
// β_EP coefficient is the same for both — only the volume basis differs.
type moeCommFamily int

const (
	// commFamilyAllGather covers the dense-hidden-state collectives. It is the
	// vLLM default (allgather_reducescatter) and the no-special-kernel "naive" path.
	commFamilyAllGather moeCommFamily = iota
	// commFamilyAll2All covers the top_k-routing point-to-point kernels.
	commFamilyAll2All
)

// DefaultMoECommBackend is vLLM's general-purpose default all-to-all backend
// (vllm@f6ec81c7 vllm/config/parallel.py:154). An empty MoECommBackend config
// value resolves to this.
const DefaultMoECommBackend = "allgather_reducescatter"

// ValidMoECommBackends is the set of accepted --moe-comm-backend values, mirroring
// vLLM's VLLM_ALL2ALL_BACKEND choices (vllm@f6ec81c7 vllm/envs.py:186). The CLI
// validator and moeCommFamilyFor share this list so they cannot disagree.
var ValidMoECommBackends = []string{
	"naive",
	"allgather_reducescatter",
	"pplx",
	"deepep_high_throughput",
	"deepep_low_latency",
	"mori",
	"flashinfer_all2allv",
}

// moeCommFamilyFor maps a vLLM backend name to its communication-volume family.
// An unrecognized name is a hard error (R1): a typo in the --moe-comm-backend flag
// must surface, not silently fall back to a default volume model.
func moeCommFamilyFor(name string) (moeCommFamily, error) {
	switch name {
	case "naive", "allgather_reducescatter":
		return commFamilyAllGather, nil
	case "pplx", "deepep_high_throughput", "deepep_low_latency", "mori", "flashinfer_all2allv":
		return commFamilyAll2All, nil
	default:
		return 0, fmt.Errorf("unknown MoE comm backend %q (valid: %v)", name, ValidMoECommBackends)
	}
}

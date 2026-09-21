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

// all2AllProfile is the PER-BACKEND-MODE step-time profile for the MoE dispatch/combine
// collective (#1548). It exists so that backend selection reaches the step-time model
// through a named per-mode parameter rather than only through the volume family: two
// backends can share a family (both DeepEP modes are commFamilyAll2All — they move the
// same top_k-routed bytes) and still cost differently, because DeepEP's high-throughput
// and low-latency kernels make opposite tradeoffs on the same wire volume.
//
// The name follows vLLM's, whose VLLM_ALL2ALL_BACKEND covers the all-gather managers too
// (naive / allgather_reducescatter), not only the genuine all-to-all kernels — so every row
// of moeCommBackends carries one of these regardless of volume family.
//
// Every backend currently ships commScale = 1.0 — a deliberate SHARED PLACEHOLDER, not a
// measurement. Differentiating DeepEP HT from LL needs its own calibration (plus the
// inter-node fabric model, #1530) and is delegated to #1568. What this PR guarantees is
// that #1568 populates THIS TABLE and nothing else: the selector, the per-role plumbing,
// and the step-time multiplication site are all in place, so no re-plumbing is required.
// A future differentiation may add fields here (e.g. a per-collective launch cost for the
// LL mode); adding a field does not touch the selector either.
type all2AllProfile struct {
	// commScale multiplies the dispatch/combine communication basis — a dimensionless
	// efficiency dial where 1.0 means "exactly the family's nominal volume ÷ effective
	// bandwidth". Values > 1 make the backend slower than nominal, < 1 faster. 1.0 is an
	// exact IEEE-754 multiplicative identity, so the shared placeholder leaves every
	// step time bit-for-bit unchanged (INV-6/INV-BC-DP1).
	commScale float64
}

// moeCommBackendEntry is one row of the backend table: a vLLM name, the physical
// communication-volume family it belongs to, and its per-mode step-time profile.
type moeCommBackendEntry struct {
	name    string
	family  moeCommFamily
	profile all2AllProfile
}

// nominalAll2AllProfile is the shared placeholder every backend resolves to today: the
// family's nominal cost, unscaled. Named (rather than repeated inline) so that #1568
// differentiating one backend is a one-line change that cannot accidentally leave the
// others at a hand-copied literal.
var nominalAll2AllProfile = all2AllProfile{commScale: 1.0}

// moeCommBackends is the single source of truth for the accepted --moe-comm-backend
// values, their volume families, and their per-mode step-time profiles, mirroring vLLM's
// VLLM_ALL2ALL_BACKEND choices (vllm@f6ec81c7 vllm/envs.py:186), in vLLM's declared
// order. ValidMoECommBackends (the display/validation list), moeCommFamilyFor (the family
// lookup) and moeCommProfileFor (the profile lookup) are all derived from this slice, so
// they cannot drift apart.
var moeCommBackends = []moeCommBackendEntry{
	{"naive", commFamilyAllGather, nominalAll2AllProfile},
	{"allgather_reducescatter", commFamilyAllGather, nominalAll2AllProfile},
	{"pplx", commFamilyAll2All, nominalAll2AllProfile},
	{"deepep_high_throughput", commFamilyAll2All, nominalAll2AllProfile},
	{"deepep_low_latency", commFamilyAll2All, nominalAll2AllProfile},
	{"mori", commFamilyAll2All, nominalAll2AllProfile},
	{"flashinfer_all2allv", commFamilyAll2All, nominalAll2AllProfile},
}

// ValidMoECommBackends is the ordered list of accepted --moe-comm-backend values,
// derived from moeCommBackends (the single source of truth). Order is deterministic
// (R2) for stable CLI help and error messages.
var ValidMoECommBackends = func() []string {
	names := make([]string, len(moeCommBackends))
	for i, b := range moeCommBackends {
		names[i] = b.name
	}
	return names
}()

// IsValidMoECommBackend reports whether name is a recognized vLLM MoE all-to-all
// backend. Used by the CLI to validate --moe-comm-backend before constructing the
// model (the constructor performs the same check via moeCommFamilyFor).
func IsValidMoECommBackend(name string) bool {
	_, err := moeCommFamilyFor(name)
	return err == nil
}

// moeCommFamilyFor maps a vLLM backend name to its communication-volume family.
// An unrecognized name is a hard error (R1): a typo in the --moe-comm-backend flag must
// surface, not silently fall back to a default volume model.
func moeCommFamilyFor(name string) (moeCommFamily, error) {
	b, err := lookupMoECommBackend(name)
	if err != nil {
		return 0, err
	}
	return b.family, nil
}

// moeCommProfileFor maps a vLLM backend name to its per-mode step-time profile (#1548).
// Same hard-error policy as moeCommFamilyFor, for the same reason: silently falling back to
// a nominal profile would hide a typo behind a plausible-looking number.
func moeCommProfileFor(name string) (all2AllProfile, error) {
	b, err := lookupMoECommBackend(name)
	if err != nil {
		return all2AllProfile{}, err
	}
	return b.profile, nil
}

// lookupMoECommBackend is the single lookup into moeCommBackends, so the family and the
// profile can never be resolved from two different rows or disagree about which names are
// valid (R23). Linear scan over seven entries, once at model construction.
func lookupMoECommBackend(name string) (moeCommBackendEntry, error) {
	for _, b := range moeCommBackends {
		if b.name == name {
			return b, nil
		}
	}
	return moeCommBackendEntry{}, fmt.Errorf("unknown MoE comm backend %q (valid: %v)", name, ValidMoECommBackends)
}

package latencykernel

import (
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// The prefill coefficients measured on the trtllm-gen kernel alone, one pair per data-center
// Blackwell part and per KV cache dtype of the sweep (registry/scripts/
// emit_attention_trtllm_gen.py). A part or a dtype without the pair has no measurement.
const (
	trtllmGenPrefillFloor = "attention_prefill_floor_trtllm_gen_kv_"
	trtllmGenPrefillScale = "attention_prefill_work_scale_trtllm_gen_kv_"
)

// prefillPair is a prefill attention floor and the fraction of the dense ramp it reaches.
type prefillPair struct {
	floor time.Duration
	scale float64
}

// prefillFor is the pair that prices one attention layer's prefill: the trtllm-gen lane's for a
// non-latent layer when the lane is priced (applyAttentionLane), the part's pooled pair
// otherwise. A latent layer is a different kernel and keeps the pooled pair.
func (k *Kernel) prefillFor(kind model.AttentionKind) prefillPair {
	if k.laneRuns && !latentAttention(kind) {
		return k.lanePair
	}
	return prefillPair{k.attentionPrefillFloor, k.attentionPrefillScale}
}

// The trtllm-gen lane is chosen by the kernel implementation the caller reports for the prefill
// primitive, and by nothing else (design X2, SUB-6, VAL-8). Which kernel an engine runs for a
// setting is the engine's to decide and to report: the cost model reads no engine setting, names
// no engine option and applies no engine default. It prices the implementation it is told runs,
// from the entry whose kernel_impls lists exactly that implementation (the lookup returns it as
// exact), and takes the part's pooled pair otherwise, recording why.

// lanePairs are the cache dtypes of the lane's entries. Which one prices is the one whose
// entries are keyed to the reported prefill implementation; the cost model does not derive it
// from the engine's cache dtype.
var lanePairs = []string{"bf16", "fp8"}

// applyAttentionLane prices the prefill of the non-latent attention layers on the trtllm-gen
// lane when, and only when, the caller reported the implementation of the attention.prefill
// primitive (Inputs.Operations or Inputs.Implementations) and one pair of the lane lists exactly
// that implementation (the lookup returns it as exact). The registry's lane is transferred from
// vLLM 0.25.0 and lists none, so on it the lane is not priced.
//
//   - A pair is exact: the lane prices the non-latent layers' prefill.
//   - The caller reported operations (Inputs.Operations), which state the cache format each
//     non-latent prefill reads: the pair of that format is the one considered, so one kernel
//     that serves both formats, and is listed by both pairs, is not ambiguous. Reported in
//     two formats, the lane is refused: it is priced from one pair for the layers of a kind.
//   - Only the kernels by primitive were reported (Inputs.Implementations), and both pairs are
//     exact: refused, because the entries do not say which cache dtype the implementation
//     serves.
//   - No pair is exact: the part's pooled pair prices the prefill. When the sets carry the lane
//     and the lookup could not show it valid for the reported implementation (it lists none, lists
//     another, or the caller reported none), the reason is recorded as an assumption, so a reader
//     of the provenance sees that the lane was not priced and what it needs.
//
// A model with no non-latent layer has nothing for the lane to price. The match is by the
// primitive the lane's names bind to (attention.prefill), so a decode-only trtllm-gen
// implementation cannot admit the pair. When the plan reports several prefill implementations, a
// pair is exact only if it lists every one of them: one value cannot price a prefill that runs
// two kernels.
func (k *Kernel) applyAttentionLane(c *resolve.Coefficients, used map[model.AttentionKind]bool) error {
	plain := false
	for kind, u := range used {
		plain = plain || (u && !latentAttention(kind))
	}
	if !plain {
		return nil
	}
	var exact []string
	var why []string
	dtypes := lanePairs
	if ops := c.Operations(); len(ops) > 0 {
		// The operations state the cache format each non-latent prefill reads, which picks the
		// pair: the bf16 pair for a bf16 cache and the fp8 pair for an fp8 one. One kernel can
		// serve both formats and so be listed by both pairs; the format, not the kernel, says
		// which pair prices.
		formats := map[string]bool{}
		for _, op := range ops {
			if op.Primitive != laneRunningPrimitive || op.Scorer {
				continue
			}
			for _, class := range op.Chain {
				if f, ok := strings.CutPrefix(class, "kv_"); ok {
					formats[f] = true
				}
			}
		}
		switch len(formats) {
		case 0:
			dtypes = nil
			if laneInSets(c) {
				why = append(why, "no non-latent prefill operation states the format of the cache it reads, so the "+
					"kernel cannot tell which cache format's pair prices it")
			}
		case 1:
			dtypes = nil
			for f := range formats {
				dtypes = []string{f}
			}
		default:
			return fmt.Errorf("the non-latent prefill operations read caches of %d formats (%s), and the trtllm-gen "+
				"lane is priced from one pair for the layers of a kind; the engine must report one cache format "+
				"for them", len(formats), strings.Join(slices.Sorted(maps.Keys(formats)), ", "))
		}
	}
	for _, dtype := range dtypes {
		floor, scale := trtllmGenPrefillFloor+dtype, trtllmGenPrefillScale+dtype
		if r := laneRefusedByImplementation(c, floor, scale); r != "" {
			why = append(why, r)
			continue
		}
		if !c.Has(floor) || !c.Has(scale) {
			continue
		}
		pairExact := true
		for _, name := range []string{floor, scale} {
			if r, _ := c.Admitted(name); r.Match != coefficient.KernelExact {
				pairExact = false
				why = append(why, fmt.Sprintf("%s states no kernel implementation it is valid for (kernel_impls), so it "+
					"applies to every build of the kernel and nothing shows it was measured on, or moved to, the one the "+
					"engine runs", name))
				break
			}
		}
		if pairExact {
			exact = append(exact, dtype)
		}
	}
	switch {
	case len(exact) > 1:
		return fmt.Errorf("the trtllm-gen prefill pairs of cache dtypes %v both list the prefill implementation the "+
			"engine reports (%s), so the kernel cannot tell which cache dtype it serves; each pair must list the "+
			"build it was moved to", exact, describeLaneRunning(c))
	case len(exact) == 1:
		floor, scale := trtllmGenPrefillFloor+exact[0], trtllmGenPrefillScale+exact[0]
		k.laneRuns = true
		k.lanePair = prefillPair{time.Duration(c.ValueOr(floor, 0) * float64(time.Microsecond)), c.ValueOr(scale, 0)}
		return nil
	}
	if len(why) == 0 {
		return nil
	}
	k.assume("trtllm_gen_prefill_lane", "pooled", "the trtllm-gen prefill pair cannot be shown valid for the prefill "+
		"implementation reported ("+describeLaneRunning(c)+"): "+strings.Join(why, "; ")+". Needed: the code hash of the "+
		"trtllm-gen prefill kernel of the engine's FlashInfer pin, which an engine release's kernel_impls.json would "+
		"publish (inference-sim/blis-registry#73), stated on the entries and reported by the engine as the "+
		"implementation of attention.prefill. The part's pooled pair prices the prefill.")
	return nil
}

// describeLaneRunning is the prefill implementations the caller reported, for a message.
func describeLaneRunning(c *resolve.Coefficients) string {
	if r := c.Implementations()[laneRunningPrimitive]; len(r) > 0 {
		return laneRunningPrimitive + " runs " + coefficient.DescribeImpls(r)
	}
	return "no implementation reported for " + laneRunningPrimitive
}

// laneRunningPrimitive is the lowered primitive the lane's coefficient names bind to.
const laneRunningPrimitive = "attention.prefill"

// laneRefusedByImplementation describes the lane entries the lookup refused because they list
// kernel implementations and the primitive's running set is not covered by them, or "" when it
// refused none.
func laneRefusedByImplementation(c *resolve.Coefficients, names ...string) string {
	var out []string
	for _, r := range c.Refused() {
		if slices.Contains(names, r.Entry.Name) {
			running := "no implementation of " + coefficient.PrimitiveOf(r.Entry.Name)
			if len(r.Running) > 0 {
				running = coefficient.PrimitiveOf(r.Entry.Name) + " runs " + coefficient.DescribeImpls(r.Running)
			}
			out = append(out, fmt.Sprintf("%s is valid for %s, and the engine reports %s", r.Entry.Name,
				coefficient.DescribeImpls(r.Entry.KernelImpls), running))
		}
	}
	if len(out) == 0 {
		return ""
	}
	return strings.Join(out, "; ") + "; a different build or kernel, or a prefill that also runs another, is not priced from it"
}

// laneInSets reports whether the sets carry the lane's entries for the deployment, admitted or
// refused for the implementations that run.
func laneInSets(c *resolve.Coefficients) bool {
	for _, dtype := range lanePairs {
		if c.Has(trtllmGenPrefillFloor + dtype) {
			return true
		}
		for _, r := range c.Refused() {
			if r.Entry.Name == trtllmGenPrefillFloor+dtype {
				return true
			}
		}
	}
	return false
}

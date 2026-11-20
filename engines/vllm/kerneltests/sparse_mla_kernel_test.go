// Original to blis-vllm.

package latencykernel_test

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	. "github.com/inference-sim/blis-substrate/kernel"

	"github.com/inference-sim/blis-vllm/core/kernelmodel"
)

// GLM-5.3-Flash on an H200 runs its sparse-MLA layers on FlashInfer's SM90 backend, whose kernel is
// BatchMLAPagedAttentionWrapper[fa3]; the lowering states that kernel, so the registry's GLM-5.3
// sparse-MLA family (keyed to that build) is the price of those layers. Without the statement the
// family is not read and the layers are priced from the unkeyed entries.
func TestGLM53FlashSparseMLAIsPricedFromTheFamilyKeyedToItsKernel(t *testing.T) {
	in := fixtureInputs(t, "glm5-h200-tp8.yaml")
	withModel(t, &in, "glm-5.3-flash")
	low, err := kernelmodel.LowerPool(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(low.Plan.Layers) == 0 {
		t.Fatal("the lowering states no kernel for a sparse-MLA model on an H200")
	}
	stated := in
	stated.Lowered = low
	stated.Implementations = kernelmodel.PlanImplementations(nil, low.Plan)
	with, err := New(stated)
	if err != nil {
		t.Fatalf("New with the kernel stated: %v", err)
	}
	bare := *low
	bare.Plan.Layers = nil
	unstated := in
	unstated.Lowered = &bare
	without, err := New(unstated)
	if err != nil {
		t.Fatalf("New without the statement: %v", err)
	}

	exact := func(k *Kernel) map[string]bool {
		out := map[string]bool{}
		for _, o := range k.Provenance() {
			if strings.HasSuffix(o.Name, "_sparse_mla") && strings.HasPrefix(o.Name, "attention_prefill_") && o.KernelMatch == "exact" {
				out[o.Name] = true
			}
		}
		return out
	}
	b := prefillOf(8192, 0)
	_ = with.StepTime(b)
	_ = without.StepTime(b)
	if got := exact(with); !got["attention_prefill_floor_sparse_mla"] || !got["attention_prefill_rate_sparse_mla"] {
		t.Errorf("with the kernel stated, the GLM-5.3 prefill family is not read as an exact match: %v", got)
	}
	if got := exact(without); len(got) != 0 {
		t.Errorf("without the statement, the family is read as an exact match: %v", got)
	}
	if a, c := with.StepTime(b).NoOverlap, without.StepTime(b).NoOverlap; a == c {
		t.Errorf("stating the sparse-MLA kernel moves no price: a prefill of 8192 tokens takes %v either way", a)
	}
	t.Logf("8192-token prefill: %v with the kernel stated, %v without", with.StepTime(b).NoOverlap, without.StepTime(b).NoOverlap)
}

// The family prices the sparse-MLA kernel of a prefill of any length (the plan states the prefill primitive;
// vLLM's dense multi-head route at or below the top-k is named on the pool, core/kernelmodel) and of every
// decode. A prefill of 1024 tokens (at most the top-k of 2048) and one of 8192 both read the prefill family as
// an exact kernel match, and a decode step reads the decode family.
func TestGLM53FlashPrefillsBelowAndAboveTheTopKAndDecodesReadTheirFamilies(t *testing.T) {
	in := fixtureInputs(t, "glm5-h200-tp8.yaml")
	withModel(t, &in, "glm-5.3-flash")
	low, err := kernelmodel.LowerPool(in)
	if err != nil {
		t.Fatal(err)
	}
	in.Lowered = low
	in.Implementations = kernelmodel.PlanImplementations(nil, low.Plan)
	exact := func(prefix string, batch kernel.Batch) map[string]bool {
		k, err := New(in)
		if err != nil {
			t.Fatal(err)
		}
		k.StepTime(batch)
		out := map[string]bool{}
		for _, o := range k.Provenance() {
			if strings.HasPrefix(o.Name, prefix) && strings.HasSuffix(o.Name, "_sparse_mla") && o.KernelMatch == "exact" {
				out[o.Name] = true
			}
		}
		return out
	}
	for _, tokens := range []int{1024, 8192} {
		if got := exact("attention_prefill_", prefillOf(tokens, 0)); !got["attention_prefill_floor_sparse_mla"] {
			t.Errorf("a prefill of %d tokens does not read the prefill family as an exact match: %v", tokens, got)
		}
	}
	if got := exact("attention_decode_", decodeBatch(4, 1, 4000)); !got["attention_decode_floor_sparse_mla"] {
		t.Errorf("a decode step does not read the decode family as an exact match: %v", got)
	}
}

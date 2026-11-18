// Original to blis-vllm.

package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// A pool whose plan states the sparse-MLA kernel for the prefill prices every prefill step from the family keyed
// to it, though vLLM runs a prefill of at most the top-k tokens as dense multi-head attention. The pool says so,
// with the top-k; a model that does not run FlashInfer's SM90 sparse-MLA backend (GLM-5 selects FlashAttention
// sparse first) states no kernel and lists nothing.
func TestASparseMLAPrefillPricedFromTheFamilyNamesTheDenseMHARoute(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "kerneltests", "testdata", "glm5-h200-tp8.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "model: glm-5\n") {
		t.Fatal("the scenario no longer names glm-5, which this test swaps for glm-5.3-flash")
	}
	open := func(t *testing.T, name string) *Model {
		t.Helper()
		r := repos()
		r.Scenarios = t.TempDir()
		body := strings.Replace(string(raw), "model: glm-5\n", "model: "+name+"\n", 1)
		if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		m, err := Open("v.yaml", r)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	listed := func(m *Model) (out []string) {
		for _, s := range m.Provenance().Substitutions {
			if s.Setting == "sparse_mla_prefill_path" {
				out = append(out, s.Reason)
			}
		}
		return
	}
	got := listed(open(t, "glm-5.3-flash"))
	if len(got) != 1 || !strings.Contains(got[0], "at most 2048 tokens") || !strings.Contains(got[0], "dense multi-head") {
		t.Errorf("GLM-5.3-Flash on an H200: substitutions %q, want one naming the dense MHA route at 2048 tokens", got)
	}
	if got := listed(open(t, "glm-5")); len(got) != 0 {
		t.Errorf("GLM-5 states no sparse-MLA kernel and lists %q", got)
	}
}

// The pool kernelmodel opens passes the plan's kernels to the cost model: GLM-5.3-Flash on an H200 reads the
// registry's sparse-MLA family as an exact match for the kernel it runs, in the prefill and in the decode.
// (Without Open handing the plan's implementations to the kernel, no entry keyed to a kernel is read.)
func TestAnOpenedGLM53FlashPoolReadsTheSparseMLAFamilyAsAnExactMatch(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "kerneltests", "testdata", "glm5-h200-tp8.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	r := repos()
	r.Scenarios = t.TempDir()
	body := strings.Replace(string(raw), "model: glm-5\n", "model: glm-5.3-flash\n", 1)
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Open("v.yaml", r)
	if err != nil {
		t.Fatal(err)
	}
	m.k.StepTime(kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 8192, PromptLen: 8192}}})
	m.k.StepTime(kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 1, Computed: 4000, PromptLen: 4001}}})
	exact := map[string]bool{}
	for _, o := range m.k.Provenance() {
		if strings.HasSuffix(o.Name, "_sparse_mla") && o.KernelMatch == "exact" {
			exact[o.Name] = true
		}
	}
	for _, want := range []string{"attention_prefill_floor_sparse_mla", "attention_decode_floor_sparse_mla"} {
		if !exact[want] {
			t.Errorf("%s is not read as an exact kernel match by the opened pool (exact: %v)", want, exact)
		}
	}
}

// A kernel the caller reports for a primitive wins over the plan's for that primitive, a primitive the caller
// reports nothing for takes the plan's, and the map the caller gave is not changed.
func TestACallerReportedKernelWinsOverThePlansForItsPrimitive(t *testing.T) {
	ours := plan.KernelImpl{Library: "caller", Variant: "kernel"}
	theirs := plan.KernelImpl{Library: "flashinfer", Variant: "BatchMLAPagedAttentionWrapper[fa3]", CodeHash: "abc"}
	p := plan.PoolPlan{Layers: []plan.LoweredLayer{
		{Layer: 0, Primitive: plan.PrimAttentionPrefill, KernelImpl: theirs},
		{Layer: 0, Primitive: plan.PrimAttentionDecode, KernelImpl: theirs},
	}}
	given := map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionPrefill: {ours}}
	got := PlanImplementations(given, p)
	if len(got[plan.PrimAttentionPrefill]) != 1 || got[plan.PrimAttentionPrefill][0] != ours {
		t.Errorf("the caller's prefill kernel does not win: %v", got[plan.PrimAttentionPrefill])
	}
	if len(got[plan.PrimAttentionDecode]) != 1 || got[plan.PrimAttentionDecode][0] != theirs {
		t.Errorf("the plan's decode kernel is not taken: %v", got[plan.PrimAttentionDecode])
	}
	if _, added := given[plan.PrimAttentionDecode]; added || len(given) != 1 {
		t.Errorf("the caller's map was changed: %v", given)
	}
	if out := PlanImplementations(given, plan.PoolPlan{}); len(out) != len(given) {
		t.Errorf("a plan that states no layer changed the implementations: %v", out)
	}
}

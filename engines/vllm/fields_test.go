// Original to blis-vllm.

package blisvllm

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
)

// fields.json is the field list VLM-3 asks for: every configuration field and environment
// variable vLLM v0.32.0 reads, each with a status, its evidence and an estimation rule. The
// names come from vLLM's source (fields/gen_fields.py, checked against the pinned commit in
// CI); this test holds the file to the document type blis-substrate defines for it.
//
//   - It loads through the substrate's strict loader (fields.Load), which refuses unknown
//     keys, a status without its evidence, an issue that is not <owner>/<repo>#<n>, an
//     estimation rule without its setting, and an unsorted or repeated entry.
//   - It is the engine and release this build models (Upstream).
//   - It is in canonical form: the substrate's Marshal writes the same bytes.
//   - Every trace a modelled field cites is a file in this repository.
func TestTheFieldListLoadsAndNamesThisRelease(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	if l.Engine.Kind != Kind || l.Engine.Version != Upstream.Version {
		t.Errorf("fields.json is for %s %s; this build models %s %s", l.Engine.Kind, l.Engine.Version, Kind, Upstream.Version)
	}
	raw, err := os.ReadFile("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := l.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, canonical) {
		t.Error("fields.json is not in the substrate's canonical form; regenerate it with fields/gen_fields.py")
	}
	// A floor, so a generator that emits almost nothing cannot pass (X22).
	const floor = 970
	if len(l.Fields) < floor {
		t.Errorf("fields.json lists %d fields, want at least %d: vLLM v0.32.0 reads more than that", len(l.Fields), floor)
	}
	for _, f := range l.Fields {
		for _, tr := range f.Evidence.Traces {
			if strings.HasPrefix(tr, "/") || strings.Contains(tr, "..") {
				t.Errorf("%s: trace %q is not a path inside this repository", f.Name, tr)
				continue
			}
			if _, err := os.Stat(tr); err != nil {
				t.Errorf("%s: trace %q: %v", f.Name, tr, err)
			}
		}
	}
}

// FieldListJSON is the committed file, byte for byte: what a simulator prints is what CI checks.
func TestFieldListJSONIsTheCommittedFile(t *testing.T) {
	raw, err := os.ReadFile("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(FieldListJSON, raw) {
		t.Error("FieldListJSON differs from fields.json")
	}
}

// README.md names every not_modelled field whose estimation rule is leave_unset, and says what a
// deployment that states it gets (design 11.2: a stated value is never replaced; the rule fills only
// a value the run did not record). The list drifts the moment a field gains the rule, so this test
// fails when fields.json has such a field the README does not name, or the README names one as
// leave_unset that fields.json no longer lists.
func TestReadmeNamesEveryLeaveUnsetField(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatal(err)
	}
	readme := string(raw)
	start := strings.Index(readme, "Every `not_modelled` field with the rule `leave_unset` is named below")
	if start < 0 {
		t.Fatal("README.md has no list of the not_modelled leave_unset fields")
	}
	end := strings.Index(readme[start:], "The two CUDA-graph switches")
	if end < 0 {
		t.Fatal("README.md: the leave_unset list has no end marker")
	}
	section := readme[start : start+end]
	want := map[string]bool{}
	for _, f := range l.Fields {
		if f.Status == fields.StatusNotModelled && f.Estimation.Rule == fields.RuleLeaveUnset {
			want[f.Name] = true
			if !strings.Contains(section, "`"+f.Name+"`") {
				t.Errorf("fields.json lists %s as not_modelled with leave_unset, and the README's list does not name it", f.Name)
			}
		}
	}
	if len(want) != 12 {
		t.Errorf("fields.json has %d not_modelled leave_unset fields, want 12; update the README list and this count", len(want))
	}
}

// gpu_memory_utilization sizes the KV pool the engine admits against (compat/kernelmodel
// KVMemory, after vLLM's request_memory), and a stated value reaches it, so the list classes it
// modelled with the tests that show that as its evidence. A not_modelled field with the rule
// leave_unset is dropped to the default by a scorer, which priced the pool at the wrong size for
// every run that stated another value.
func TestMemoryUtilizationIsClassedModelledWithItsEvidence(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range l.Fields {
		if f.Name != "gpu_memory_utilization" || f.Source != fields.SourceConfig {
			continue
		}
		if f.Status != fields.StatusModelled {
			t.Fatalf("gpu_memory_utilization is %s; the engine sizes the KV pool from it", f.Status)
		}
		if len(f.Evidence.Traces) == 0 {
			t.Error("gpu_memory_utilization is modelled and cites no evidence")
		}
		return
	}
	t.Fatal("fields.json has no gpu_memory_utilization")
}

// A stated moe_backend is not refused where the engine can price it: the field list covers, with a
// named execution-mode substitution, the backends a recorded run states on the parts it states
// them on (blis-substrate#150). On an SM90 part Triton and FlashInfer CUTLASS are the backends the
// engine's own expert-kernel selection orders for block-FP8 experts; CuteDSL is a Blackwell
// backend. A backend or part no clause covers stays unapproximated, so the point is refused by name.
func TestAStatedMoEBackendIsApproximatedWhereTheEnginePricesIt(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	var e fields.Estimation
	for _, f := range l.Fields {
		if f.Name == "moe_backend" {
			e = f.Estimation
		}
	}
	for _, c := range []struct {
		value, cc string
		covered   bool
	}{
		{"triton", "9.0", true},
		{"Triton", "9.0", true},
		{"flashinfer_cutlass", "9.0", true},
		{"flashinfer_cutedsl", "10.0", true},
		{"flashinfer_trtllm", "10.0", true},
		{"deep_gemm_mega_moe", "10.0", true},
		{"triton", "10.0", false},
		{"flashinfer_cutedsl", "9.0", false},
		{"marlin", "9.0", false},
		{"triton", "", false},
	} {
		_, got := fields.Approximate(e, fields.Facts{Value: c.value, ComputeCapability: c.cc})
		if got != c.covered {
			t.Errorf("moe_backend %q on compute capability %q: approximated %v, want %v", c.value, c.cc, got, c.covered)
		}
		if fields.Exact(e, fields.Facts{Value: c.value, ComputeCapability: c.cc}) {
			t.Errorf("moe_backend %q is exact, but it has no stand-in", c.value)
		}
	}
	// The SM90 clause is valid for block-FP8 experts only and the clause grammar cannot test the
	// weight scheme, so the approximation must say so where a reader of a scored point sees it.
	a, _ := fields.Approximate(e, fields.Facts{Value: "triton", ComputeCapability: "9.0"})
	if !strings.Contains(a.Differs, "block-FP8 experts only") {
		t.Errorf("the approximation covering moe_backend on SM90 does not say it is valid for block-FP8 only: %q", a.Differs)
	}
}

// Original to blis-vllm.

package blisvllm

import (
	"bytes"
	"os"
	"reflect"
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
		{"marlin", "9.0", true},
		{"marlin", "8.0", false},
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
	// The SM90 clause is valid for block-FP8 experts (Triton, FlashInfer CUTLASS) and MXFP4 experts
	// (Marlin) only, and the clause grammar cannot test the weight scheme, so the approximation must
	// say so, and name the ramp a stated Marlin is priced on, where a reader of a scored point sees it.
	for _, backend := range []string{"triton", "marlin"} {
		a, _ := fields.Approximate(e, fields.Facts{Value: backend, ComputeCapability: "9.0"})
		for _, want := range []string{"block-FP8 experts (Triton, FlashInfer CUTLASS) and for MXFP4 experts (Marlin) only", "w4a16_mxfp4_grouped"} {
			if !strings.Contains(a.Differs, want) {
				t.Errorf("the approximation covering moe_backend %q on SM90 does not say %q: %q", backend, want, a.Differs)
			}
		}
	}
}

// A run recorded on an earlier release states an environment variable that v0.33.0 no longer reads.
// One upstream per release: the field list translates the variable to the v0.33.0 flags that make the
// same selection, as a named substitution (env_var_not_in_release) that cites the vLLM commit that
// removed it, and does not stand in where v0.33.0 has no such kernel, so the point is refused by that
// name. gpt-oss-120b on v0.22.0 states VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8 on B200 and
// VLLM_MXFP4_USE_MARLIN on H200. The translated setting then meets moe_backend's own rule: FlashInfer
// TRT-LLM on capability 10.0 is approximated, Marlin on 9.0 is not, so the H200 points pass the
// variable and stop at moe_backend.
func TestAnEnvVarOfAnEarlierReleaseIsTranslatedToThisReleasesFlags(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]fields.Field{}
	for _, f := range l.Fields {
		byName[f.Name] = f
	}
	for _, c := range []struct {
		name    string
		setting map[string]any
		cases   []struct {
			value, cc string
			exact     bool
		}
	}{
		{"VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8", map[string]any{"moe_backend": "flashinfer_trtllm", "quantization_config.moe.activation": "mxfp8"},
			[]struct {
				value, cc string
				exact     bool
			}{{"true", "10.0", true}, {"false", "10.0", false}, {"1", "10.0", true}, {"1", "10.3", true}, {"1", "9.0", false}, {"1", "12.0", false}, {"0", "10.0", false}, {"", "10.0", false}}},
		{"VLLM_MXFP4_USE_MARLIN", map[string]any{"moe_backend": "marlin"},
			[]struct {
				value, cc string
				exact     bool
			}{{"true", "9.0", true}, {"false", "9.0", false}, {"1", "9.0", true}, {"1", "10.0", true}, {"1", "7.5", true}, {"1", "7.0", false}, {"0", "9.0", false}}},
	} {
		f, ok := byName[c.name]
		if !ok {
			t.Fatalf("%s is not in the field list", c.name)
		}
		e := f.Estimation
		if e.Rule != fields.RuleSubstitute || !reflect.DeepEqual(e.Setting, c.setting) {
			t.Errorf("%s: rule %q setting %v, want a substitution by %v", c.name, e.Rule, e.Setting, c.setting)
		}
		for _, frag := range []string{"env_var_not_in_release", c.name, "v0.22.0", "v0.24.0", "03878d1c2", "vllm-project/vllm#44992", "--moe-backend"} {
			if !strings.Contains(e.Reason, frag) {
				t.Errorf("%s: the substitution's reason does not name %q: %s", c.name, frag, e.Reason)
			}
		}
		for _, k := range c.cases {
			if got := fields.Exact(e, fields.Facts{Value: k.value, ComputeCapability: k.cc}); got != k.exact {
				t.Errorf("%s=%q on compute capability %q: stood in for = %v, want %v", c.name, k.value, k.cc, got, k.exact)
			}
		}
	}
	// The translated setting meets moe_backend's own rule.
	var mb fields.Estimation
	for _, f := range l.Fields {
		if f.Name == "moe_backend" {
			mb = f.Estimation
		}
	}
	if _, ok := fields.Approximate(mb, fields.Facts{Value: "flashinfer_trtllm", ComputeCapability: "10.0"}); !ok {
		t.Error("the B200 translation (flashinfer_trtllm on 10.0) is not approximated by moe_backend's rule")
	}
	if _, ok := fields.Approximate(mb, fields.Facts{Value: "marlin", ComputeCapability: "9.0"}); !ok {
		t.Error("the H200 translation (marlin on 9.0) is not approximated by moe_backend's rule")
	}
}

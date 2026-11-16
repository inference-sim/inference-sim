package dataset

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/kernelimpls"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

func tableListing(impls ...kernelimpls.Implementation) *kernelimpls.Table {
	return &kernelimpls.Table{Kind: kernelimpls.Kind, Primitives: []string{"attention.prefill"}, Implementations: impls}
}

func implRow(variant, arch string) kernelimpls.Implementation {
	in := []string{"fmha/checksums.txt sha256:" + strings.Repeat("1", 64)}
	return kernelimpls.Implementation{Primitive: "attention.prefill", Library: "attnlib", LibraryVersion: "0.7.0",
		Variant: variant, TargetArch: arch, HashBasis: kernelimpls.HashBinary, HashInputs: in,
		CodeHash: kernelimpls.CodeHash(kernelimpls.HashBinary, "attnlib", variant, arch, in)}
}

func keyedEntry(name string, method vocab.Method, impls ...coefficient.KernelImpl) coefficient.Entry {
	return coefficient.Entry{Name: name, Method: method, Scope: coefficient.Scope{Hardware: []string{"b200"}}, KernelImpls: impls}
}

func names(us []Unselected) []string {
	var out []string
	for _, u := range us {
		out = append(out, u.Set+"/"+u.Entry.Name)
	}
	return out
}

// VAL-8's second check. An entry is accounted for when a conformance run selected it or a
// published table lists a kernel implementation it is valid for; every other entry is listed.
func TestAnEntryNeitherSelectedNorPublishedIsListed(t *testing.T) {
	pub := implRow("prefill_kernel", "sm_100a")
	unpub := implRow("prefill_kernel", "sm_90a").Impl() // a build no table lists
	sets := map[string]*coefficient.Set{
		"attention": {Name: "attention", Coefficients: []coefficient.Entry{
			keyedEntry("attention_prefill_floor", vocab.MethodMeasured),                                   // selected below
			keyedEntry("attention_decode_floor", vocab.MethodMeasured),                                    // neither
			keyedEntry("attention_prefill_floor_trtllm_gen_kv_bf16", vocab.MethodTransferred, pub.Impl()), // published
			keyedEntry("attention_prefill_work_scale_trtllm_gen_kv_bf16", vocab.MethodTransferred, unpub), // neither
		}},
	}
	selected := []result.SelectedEntries{{ClosenessVersion: result.ClosenessVersion, Coefficients: []result.SelectedCoefficient{
		{Set: "attention", Name: "attention_prefill_floor", Method: vocab.MethodMeasured, Scope: coefficient.Scope{Hardware: []string{"b200"}}},
	}}}
	got, _ := FindUnselected(sets, selected, []*kernelimpls.Table{tableListing(pub)})
	want := []string{"attention/attention_decode_floor", "attention/attention_prefill_work_scale_trtllm_gen_kv_bf16"}
	if strings.Join(names(got), " ") != strings.Join(want, " ") {
		t.Fatalf("unselected = %v, want %v", names(got), want)
	}
	if len(got[1].Unpublished) != 1 || got[1].Unpublished[0] != unpub {
		t.Errorf("the keyed entry does not name its unpublished implementation: %+v", got[1].Unpublished)
	}
	if !strings.Contains(got[1].String(), "kernel_impls.json") || !strings.Contains(got[0].String(), "states no kernel implementation") {
		t.Errorf("messages do not say which condition failed:\n%s\n%s", got[0], got[1])
	}
}

// A selection names an entry by its whole identity: the same name selected in another scope,
// by another method, from another set or for another build does not account for it.
func TestASelectionOfAnotherEntryOfTheNameDoesNotCount(t *testing.T) {
	k := implRow("prefill_kernel", "sm_100a").Impl()
	e := keyedEntry("attention_prefill_floor", vocab.MethodMeasured, k)
	sets := map[string]*coefficient.Set{"attention": {Name: "attention", Coefficients: []coefficient.Entry{e}}}
	sel := func(c result.SelectedCoefficient) []result.SelectedEntries {
		return []result.SelectedEntries{{ClosenessVersion: result.ClosenessVersion, Coefficients: []result.SelectedCoefficient{c}}}
	}
	base := result.SelectedCoefficient{Set: "attention", Name: e.Name, Method: e.Method, Scope: e.Scope, KernelImpls: e.KernelImpls}
	if got, _ := FindUnselected(sets, sel(base), nil); len(got) != 0 {
		t.Fatalf("the entry's own selection did not account for it: %v", names(got))
	}
	other := k
	other.CodeHash = strings.Repeat("0", 64)
	for name, c := range map[string]result.SelectedCoefficient{
		"another scope":  {Set: base.Set, Name: base.Name, Method: base.Method, Scope: coefficient.Scope{Hardware: []string{"h200"}}, KernelImpls: base.KernelImpls},
		"another method": {Set: base.Set, Name: base.Name, Method: vocab.MethodCalibrated, Scope: base.Scope, KernelImpls: base.KernelImpls},
		"another set":    {Set: "other", Name: base.Name, Method: base.Method, Scope: base.Scope, KernelImpls: base.KernelImpls},
		"another build":  {Set: base.Set, Name: base.Name, Method: base.Method, Scope: base.Scope, KernelImpls: []coefficient.KernelImpl{other}},
	} {
		if got, _ := FindUnselected(sets, sel(c), nil); len(got) != 1 {
			t.Errorf("%s: a selection of a different entry accounted for this one", name)
		}
	}
}

// A table that lists the same kernel at another code hash does not publish the entry's build:
// the hash is the key (VAL-8, no sibling).
func TestATableListingASiblingBuildDoesNotPublishTheEntry(t *testing.T) {
	row := implRow("prefill_kernel", "sm_100a")
	sibling := row.Impl()
	sibling.CodeHash = strings.Repeat("0", 64)
	sets := map[string]*coefficient.Set{"attention": {Name: "attention", Coefficients: []coefficient.Entry{
		keyedEntry("attention_prefill_floor", vocab.MethodMeasured, sibling)}}}
	if got, _ := FindUnselected(sets, nil, []*kernelimpls.Table{tableListing(row)}); len(got) != 1 {
		t.Errorf("a sibling build accounted for the entry: %v", names(got))
	}
}

// A report without closeness data names a keyed entry without its kernel implementations, so
// it cannot show the entry selected; it is refused rather than letting the entry read as
// unselected (X16).
func TestAReportWithoutClosenessDataIsRefused(t *testing.T) {
	k := implRow("prefill_kernel", "sm_100a").Impl()
	e := keyedEntry("attention_prefill_floor", vocab.MethodMeasured, k)
	sets := map[string]*coefficient.Set{"attention": {Name: "attention", Coefficients: []coefficient.Entry{e}}}
	bare := []result.SelectedEntries{{Coefficients: []result.SelectedCoefficient{
		{Set: "attention", Name: e.Name, Method: e.Method, Scope: e.Scope}}}}
	if got, err := FindUnselected(sets, bare, nil); err == nil {
		t.Fatalf("a report without closeness data was read; it listed %v", names(got))
	}
}

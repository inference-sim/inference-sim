package kernelimpls

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
)

// row builds an implementation whose code hash is the rule's, so a test changes one field and
// sees what the validator says about it.
func row(prim, lib, ver, variant, arch string, basis HashBasis, inputs ...string) Implementation {
	return Implementation{Primitive: prim, Library: lib, LibraryVersion: ver, Variant: variant, TargetArch: arch,
		HashBasis: basis, HashInputs: inputs, CodeHash: CodeHash(basis, lib, variant, arch, inputs),
		Toolkit: "12.9", SelectedBy: []string{"upstream/selector.py:10@0123abcd"}}
}

func valid() *Table {
	return &Table{
		Kind:       Kind,
		Engine:     Engine{Kind: "an-engine", Version: "1.2.3", Commit: strings.Repeat("ab", 20)},
		Primitives: []string{"attention.decode", "attention.prefill"},
		Libraries: []Library{
			{Name: "attnlib", Version: "0.6.0", Pin: "seen in runs r1", Origin: "pypi attnlib==0.6.0"},
			{Name: "attnlib", Version: "0.7.0", Pinned: true, Pin: "requirements.txt:12@0123abcd", Origin: "pypi attnlib==0.7.0"},
			{Name: "engine-kernels", Version: strings.Repeat("ab", 20), Pinned: true, Pin: "CMakeLists.txt:40@0123abcd", Origin: "git https://example.org/e " + strings.Repeat("ab", 20)},
		},
		Implementations: []Implementation{
			row("attention.decode", "attnlib", "0.6.0", "decode_kernel", "sm_100a", HashBinary, "fmha/checksums.txt sha256:"+strings.Repeat("1", 64)),
			row("attention.decode", "attnlib", "0.7.0", "decode_kernel", "sm_100a", HashBinary, "fmha/checksums.txt sha256:"+strings.Repeat("1", 64)),
			row("attention.decode", "engine-kernels", strings.Repeat("ab", 20), "paged_attention", "sm_90a", HashSource, "csrc/attention tree "+strings.Repeat("c", 40)),
			row("attention.prefill", "attnlib", "0.7.0", "prefill_kernel", "sm_100a", HashBinary, "fmha/checksums.txt sha256:"+strings.Repeat("2", 64)),
		},
	}
}

func TestAValidTableValidatesAndRoundTrips(t *testing.T) {
	tab := valid()
	if err := tab.Validate(); err != nil {
		t.Fatalf("valid table refused: %v", err)
	}
	b, err := tab.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	back, err := Parse("t.json", b)
	if err != nil {
		t.Fatalf("Parse of Marshal refused: %v", err)
	}
	again, _ := back.Marshal()
	if string(again) != string(b) {
		t.Errorf("Marshal(Parse(Marshal(t))) differs from Marshal(t)")
	}
}

// SUB-2: a failing example for each way a table can be wrong. Each names the field the
// validator must point at.
func TestEachWayATableCanBeWrongIsRefused(t *testing.T) {
	cases := []struct {
		name, want string
		edit       func(*Table)
	}{
		{"wrong kind", "kind", func(t *Table) { t.Kind = "FieldList" }},
		{"no engine kind", "engine.kind", func(t *Table) { t.Engine.Kind = "" }},
		{"no engine version", "engine.version", func(t *Table) { t.Engine.Version = "" }},
		{"engine commit not 40 hex", "engine.commit", func(t *Table) { t.Engine.Commit = "v1.2.3" }},
		{"no primitives", "primitives", func(t *Table) { t.Primitives = nil }},
		{"a primitive that is not lowered", "primitives[1]", func(t *Table) { t.Primitives[1] = "attention" }},
		{"primitives out of order", "primitives[1]", func(t *Table) { t.Primitives = []string{"attention.prefill", "attention.decode"} }},
		{"a primitive twice", "primitives[1]", func(t *Table) { t.Primitives = []string{"attention.decode", "attention.decode"} }},
		{"library with no name", "libraries[0].name", func(t *Table) { t.Libraries[0].Name = "" }},
		{"library with no version", "libraries[0].version", func(t *Table) { t.Libraries[0].Version = "" }},
		{"library with no pin citation", "libraries[1].pin", func(t *Table) { t.Libraries[1].Pin = "" }},
		{"library with no origin", "libraries[1].origin", func(t *Table) { t.Libraries[1].Origin = " " }},
		{"library listed twice", "libraries[1]", func(t *Table) { t.Libraries[1].Version = "0.6.0" }},
		{"libraries out of order", "libraries[1]", func(t *Table) { t.Libraries[0], t.Libraries[1] = t.Libraries[1], t.Libraries[0] }},
		{"a library with no pinned version", "libraries", func(t *Table) { t.Libraries[1].Pinned = false }},
		{"a library with two pinned versions", "libraries", func(t *Table) { t.Libraries[0].Pinned = true }},
		{"row of an uncovered primitive", "implementations[3].primitive", func(t *Table) {
			t.Primitives = []string{"attention.decode"}
		}},
		{"row with no primitive", "implementations[0].primitive", func(t *Table) { t.Implementations[0].Primitive = "" }},
		{"row of a library version not listed", "implementations[0].library", func(t *Table) {
			t.Implementations[0] = row("attention.decode", "attnlib", "0.5.0", "decode_kernel", "sm_100a", HashBinary, "x sha256:"+strings.Repeat("1", 64))
		}},
		{"row with no variant", "implementations[2].variant", func(t *Table) {
			t.Implementations[2] = row("attention.decode", "engine-kernels", strings.Repeat("ab", 20), "", "sm_90a", HashSource, "csrc tree "+strings.Repeat("c", 40))
		}},
		{"architecture not as the compiler names it", "implementations[2].target_arch", func(t *Table) {
			t.Implementations[2] = row("attention.decode", "engine-kernels", strings.Repeat("ab", 20), "paged_attention", "hopper", HashSource, "csrc tree "+strings.Repeat("c", 40))
		}},
		{"unknown hash basis", "implementations[2].hash_basis", func(t *Table) {
			t.Implementations[2] = row("attention.decode", "engine-kernels", strings.Repeat("ab", 20), "paged_attention", "sm_90a", "version", "csrc tree "+strings.Repeat("c", 40))
		}},
		{"no hash inputs", "implementations[2].hash_inputs", func(t *Table) {
			t.Implementations[2] = row("attention.decode", "engine-kernels", strings.Repeat("ab", 20), "paged_attention", "sm_90a", HashSource)
		}},
		{"hash inputs out of order", "implementations[2].hash_inputs[1]", func(t *Table) {
			t.Implementations[2] = row("attention.decode", "engine-kernels", strings.Repeat("ab", 20), "paged_attention", "sm_90a", HashSource, "b tree 1", "a tree 2")
		}},
		{"a hash input with a line break", "implementations[2].hash_inputs[0]", func(t *Table) {
			t.Implementations[2] = row("attention.decode", "engine-kernels", strings.Repeat("ab", 20), "paged_attention", "sm_90a", HashSource, "a\nb")
		}},
		{"no toolkit", "implementations[0].toolkit", func(t *Table) { t.Implementations[0].Toolkit = "" }},
		{"no selection citation", "implementations[0].selected_by", func(t *Table) { t.Implementations[0].SelectedBy = nil }},
		{"code hash not a SHA-256", "implementations[0].code_hash", func(t *Table) { t.Implementations[0].CodeHash = "abc123abc123" }},
		{"code hash not the rule's", "implementations[0].code_hash", func(t *Table) {
			t.Implementations[0].CodeHash = strings.Repeat("e", 64)
		}},
		{"inputs changed under a kept hash", "implementations[0].code_hash", func(t *Table) {
			t.Implementations[0].HashInputs = []string{"other sha256:" + strings.Repeat("9", 64)}
		}},
		{"row listed twice", "implementations[1]", func(t *Table) { t.Implementations[1] = t.Implementations[0] }},
		{"rows out of order", "implementations[1]", func(t *Table) {
			t.Implementations[0], t.Implementations[3] = t.Implementations[3], t.Implementations[0]
		}},
		{"one build with two hashes across primitives", "implementations[3].code_hash", func(t *Table) {
			t.Implementations[3] = row("attention.prefill", "attnlib", "0.7.0", "decode_kernel", "sm_100a", HashBinary, "fmha/checksums.txt sha256:"+strings.Repeat("3", 64))
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tab := valid()
			c.edit(tab)
			err := tab.Validate()
			if err == nil {
				t.Fatalf("validated; want a problem at %s", c.want)
			}
			if !strings.Contains(err.Error(), c.want+":") {
				t.Errorf("problem not reported at %s:\n%v", c.want, err)
			}
		})
	}
}

// The hash does not depend on the library's version: a release that bumps a library and
// leaves a kernel's inputs alone keeps the kernel's hash, so an entry keyed by it stays valid
// (design 6.2). It does depend on every other part of the rule.
func TestTheCodeHashIgnoresTheVersionAndNothingElse(t *testing.T) {
	tab := valid()
	if tab.Implementations[0].CodeHash != tab.Implementations[1].CodeHash {
		t.Errorf("the same kernel inputs at versions 0.6.0 and 0.7.0 hash differently")
	}
	base := CodeHash(HashSource, "lib", "k", "sm_90a", []string{"a tree 1", "b tree 2"})
	if CodeHash(HashSource, "lib", "k", "sm_90a", []string{"b tree 2", "a tree 1"}) != base {
		t.Errorf("the order of the inputs changes the hash")
	}
	for name, h := range map[string]string{
		"basis":   CodeHash(HashBinary, "lib", "k", "sm_90a", []string{"a tree 1", "b tree 2"}),
		"library": CodeHash(HashSource, "lib2", "k", "sm_90a", []string{"a tree 1", "b tree 2"}),
		"variant": CodeHash(HashSource, "lib", "k2", "sm_90a", []string{"a tree 1", "b tree 2"}),
		"arch":    CodeHash(HashSource, "lib", "k", "sm_100a", []string{"a tree 1", "b tree 2"}),
		"inputs":  CodeHash(HashSource, "lib", "k", "sm_90a", []string{"a tree 1", "b tree 3"}),
	} {
		if h == base {
			t.Errorf("changing the %s leaves the hash unchanged", name)
		}
	}
}

func TestParseRefusesWhatTheStrictDecoderRefuses(t *testing.T) {
	b, _ := valid().Marshal()
	for name, doc := range map[string]string{
		"unknown key":                 strings.Replace(string(b), `"kind": "KernelImplTable"`, `"kind": "KernelImplTable", "notes": "x"`, 1),
		"repeated key":                strings.Replace(string(b), `"kind": "KernelImplTable"`, `"kind": "KernelImplTable", "kind": "KernelImplTable"`, 1),
		"key spelled in another case": strings.Replace(string(b), `"primitives"`, `"Primitives"`, 1),
	} {
		if _, err := Parse("t.json", []byte(doc)); err == nil {
			t.Errorf("%s: parsed", name)
		}
	}
}

func TestTheLookupsFindTheRowsOfABuild(t *testing.T) {
	tab := valid()
	v, ok := tab.PinnedVersion("attnlib")
	if !ok || v != "0.7.0" {
		t.Fatalf("PinnedVersion(attnlib) = %q, %v; want 0.7.0", v, ok)
	}
	r, ok := tab.Find("attention.prefill", "attnlib", v, "prefill_kernel", "sm_100a")
	if !ok || r.CodeHash != tab.Implementations[3].CodeHash {
		t.Fatalf("Find of the pinned prefill build = %+v, %v", r, ok)
	}
	if _, ok := tab.Find("attention.prefill", "attnlib", v, "prefill_kernel", "sm_90a"); ok {
		t.Errorf("Find returned a build for an architecture the table has no row for")
	}
	if !tab.Lists(r.Impl()) {
		t.Errorf("Lists does not find a row's own implementation")
	}
	sibling := r.Impl()
	sibling.CodeHash = strings.Repeat("0", 64)
	if tab.Lists(sibling) {
		t.Errorf("Lists matched a different build of the same kernel (VAL-8)")
	}
	if !tab.Covers("attention.decode") || tab.Covers("gemm") {
		t.Errorf("Covers reports the wrong primitives")
	}
	if got := (coefficient.KernelImpl{Library: "attnlib", Variant: "prefill_kernel", CodeHash: r.CodeHash}); got != r.Impl() {
		t.Errorf("Impl() = %+v, want %+v", r.Impl(), got)
	}
}

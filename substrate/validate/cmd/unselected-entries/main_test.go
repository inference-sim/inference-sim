package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/kernelimpls"
)

// prefillRow is one published trtllm-gen-style prefill build.
func prefillRow() kernelimpls.Implementation {
	in := []string{"fmha/checksums.txt sha256:" + strings.Repeat("1", 64)}
	return kernelimpls.Implementation{Primitive: "attention.prefill", Library: "attnlib", LibraryVersion: "0.7.0",
		Variant: "prefill_kernel", TargetArch: "sm_100a", HashBasis: kernelimpls.HashBinary, HashInputs: in,
		CodeHash: kernelimpls.CodeHash(kernelimpls.HashBinary, "attnlib", "prefill_kernel", "sm_100a", in),
		Toolkit:  "12.9", SelectedBy: []string{"upstream/attn.py:1@0123abcd"}}
}

func writeTable(t *testing.T, dir string) string {
	tab := &kernelimpls.Table{Kind: kernelimpls.Kind,
		Engine:          kernelimpls.Engine{Kind: "an-engine", Version: "1.0.0", Commit: strings.Repeat("ab", 20)},
		Primitives:      []string{"attention.prefill"},
		Libraries:       []kernelimpls.Library{{Name: "attnlib", Version: "0.7.0", Pinned: true, Pin: "req.txt:1@0123abcd", Origin: "pypi attnlib==0.7.0"}},
		Implementations: []kernelimpls.Implementation{prefillRow()}}
	b, err := tab.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, "kernel_impls.json")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func writeRegistry(t *testing.T, codeHash string) string {
	root := t.TempDir()
	set := `kind: CoefficientSet
name: lane
coefficients:
  - attention_prefill_floor_trtllm_gen_kv_bf16:
      value: 18.5
      units: us_per_transfer
      method: transferred
      fitted: false
      scope: {hardware: [b200]}
      transferred_from: "hardware=b200"
      transfer_rule: "same kernel, older release"
      rationale: "example"
      kernel_impls: [{library: attnlib, variant: prefill_kernel, code_hash: ` + codeHash + `}]
      measured_on: [{library: attnlib, variant: prefill_kernel, code_hash: ` + strings.Repeat("9", 64) + `}]
`
	if err := os.MkdirAll(filepath.Join(root, "coefficients"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "coefficients", "lane.yaml"), []byte(set), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestAnEntryKeyedToAPublishedBuildIsNotListed(t *testing.T) {
	table := writeTable(t, t.TempDir())
	var out, errb bytes.Buffer
	code := run([]string{"-kernel-impls", table, writeRegistry(t, prefillRow().CodeHash)}, &out, &errb)
	if code != 0 || out.Len() != 0 {
		t.Fatalf("exit %d, listed:\n%s%s", code, out.String(), errb.String())
	}
}

func TestAnEntryKeyedToAnUnpublishedBuildIsListed(t *testing.T) {
	table := writeTable(t, t.TempDir())
	var out, errb bytes.Buffer
	code := run([]string{"-kernel-impls", table, writeRegistry(t, strings.Repeat("0", 64))}, &out, &errb)
	if code != 1 || !strings.Contains(out.String(), "attention_prefill_floor_trtllm_gen_kv_bf16") {
		t.Fatalf("exit %d, want 1 listing the entry; stdout:\n%s", code, out.String())
	}
	// Without the table nothing publishes the build either.
	out.Reset()
	if code := run([]string{writeRegistry(t, prefillRow().CodeHash)}, &out, &errb); code != 1 {
		t.Errorf("with no table, exit %d; want 1", code)
	}
}

func TestATableThatDoesNotLoadIsMisuse(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "kernel_impls.json")
	if err := os.WriteFile(bad, []byte(`{"kind": "KernelImplTable"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := run([]string{"-kernel-impls", bad, writeRegistry(t, prefillRow().CodeHash)}, &out, &errb); code != 2 {
		t.Errorf("exit %d for a table that does not validate; want 2", code)
	}
	if code := run(nil, &out, &errb); code != 2 {
		t.Errorf("exit %d with no registry root; want 2", code)
	}
}

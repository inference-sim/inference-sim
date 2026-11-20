// Original to blis-vllm.

package rulepack

import (
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/engines/vllm/core/rulepack/pack"
	"github.com/inference-sim/inference-sim/engines/vllm/core/rulepack/v0_33"
	"github.com/inference-sim/inference-sim/engines/vllm/internal/upstream"
)

func backport(commit string) upstream.Backport {
	return upstream.Backport{
		Commit:        commit,
		Subject:       "synthetic",
		Architectures: []string{"SomeForCausalLM"},
		Paths:         []string{"vllm/model_executor/models/some.py"},
	}
}

const (
	commitA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	commitB = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)

// The record and the rules agree or the build is refused, and the patches are applied in the
// order the record lists the backports.
func TestWithBackports(t *testing.T) {
	var applied []string
	mark := func(name string) Patch { return func(*pack.Pack) { applied = append(applied, name) } }
	tests := []struct {
		name    string
		bs      []upstream.Backport
		patches map[string]Patch
		want    []string
		wantErr string
	}{
		{"none", nil, nil, nil, ""},
		{"one", []upstream.Backport{backport(commitA)}, map[string]Patch{commitA: mark("a")}, []string{"a"}, ""},
		{"in the record's order", []upstream.Backport{backport(commitB), backport(commitA)},
			map[string]Patch{commitA: mark("a"), commitB: mark("b")}, []string{"b", "a"}, ""},
		{"a backport without a patch", []upstream.Backport{backport(commitA)}, nil, nil, "has no patch"},
		{"a patch for an unlisted commit", nil, map[string]Patch{commitA: mark("a")}, nil, "does not list"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			applied = nil
			got, err := WithBackports(v0_33.Pack(), tc.bs, tc.patches)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want one mentioning %q", err, tc.wantErr)
				}
				return
			}
			if err != nil || got == nil {
				t.Fatalf("WithBackports = %v, %v", got, err)
			}
			if !reflect.DeepEqual(applied, tc.want) {
				t.Errorf("patches applied %v, want %v", applied, tc.want)
			}
		})
	}
}

// A patch changes the pack it is given and no other: Packs hands out fresh copies.
func TestAPatchDoesNotReachTheNextPack(t *testing.T) {
	arch := "SomeForCausalLM"
	patch := func(p *pack.Pack) {
		p.CacheDTypeResolutions[arch] = pack.CacheDTypeResolutions{{CapabilityFamily: 10, Requested: []string{"auto"}, ResolvesTo: "fp8_e4m3"}}
	}
	patched, err := WithBackports(v0_33.Pack(), []upstream.Backport{backport(commitA)}, map[string]Patch{commitA: patch})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := patched.CacheDTypeResolutions[arch]; !ok {
		t.Fatal("the patch did not apply")
	}
	if _, ok := v0_33.Pack().CacheDTypeResolutions[arch]; ok {
		t.Error("a patch reached a pack it was not given")
	}
}

// The patches the rules carry are exactly those of the backports the record lists.
func TestPatchesAreThoseOfTheRecordedBackports(t *testing.T) {
	var listed, patched []string
	for _, b := range upstream.Backports() {
		listed = append(listed, b.Commit)
	}
	for c := range Patches() {
		patched = append(patched, c)
	}
	sort.Strings(listed)
	sort.Strings(patched)
	if !reflect.DeepEqual(listed, patched) {
		t.Errorf("recorded backports %v, patched commits %v", listed, patched)
	}
	if Packs()[v0_33.Version] == nil {
		t.Error("Packs() has no pack for the release")
	}
}

func TestPatchesReturnsACopy(t *testing.T) {
	saved := patches
	defer func() { patches = saved }()
	patches = map[string]Patch{commitA: func(*pack.Pack) {}}
	got := Patches()
	delete(got, commitA)
	got[commitB] = func(*pack.Pack) {}
	if len(patches) != 1 || patches[commitA] == nil {
		t.Errorf("the rules' patches moved: %v", patches)
	}
}

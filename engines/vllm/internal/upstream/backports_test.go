// Original to blis-vllm.

package upstream

import (
	"reflect"
	"strings"
	"testing"
)

const goodCommit = "0123456789abcdef0123456789abcdef01234567"

func goodBackport() Backport {
	return Backport{
		Commit:        goodCommit,
		Subject:       "synthetic: a well-formed entry for the validator's tests",
		Architectures: []string{"SomeForCausalLM"},
		Paths:         []string{"vllm/model_executor/models/some.py"},
	}
}

// The validator accepts a well-formed record and names each way one is not.
func TestBackportProblems(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(b *Backport)
		extra  []Backport
		want   string // a substring of a problem; empty means none
	}{
		{"well formed", func(b *Backport) {}, nil, ""},
		{"short commit", func(b *Backport) { b.Commit = "0123456789ab" }, nil, "40-hex"},
		{"upper-case commit", func(b *Backport) { b.Commit = strings.ToUpper(goodCommit) }, nil, "40-hex"},
		{"empty commit", func(b *Backport) { b.Commit = "" }, nil, "40-hex"},
		{"the pin itself", func(b *Backport) { b.Commit = Commit }, nil, "is the pin"},
		{"listed twice", func(b *Backport) {}, []Backport{goodBackport()}, "listed twice"},
		{"blank subject", func(b *Backport) { b.Subject = "  " }, nil, "subject is empty"},
		{"no architecture", func(b *Backport) { b.Architectures = nil }, nil, "no architecture"},
		{"padded architecture", func(b *Backport) { b.Architectures = []string{" X"} }, nil, "empty or padded"},
		{"empty architecture", func(b *Backport) { b.Architectures = []string{""} }, nil, "empty or padded"},
		{"duplicate architecture", func(b *Backport) { b.Architectures = []string{"X", "X"} }, nil, "listed twice"},
		{"no path", func(b *Backport) { b.Paths = nil }, nil, "no path"},
		{"absolute path", func(b *Backport) { b.Paths = []string{"/vllm/x.py"} }, nil, "clean path under vllm/"},
		{"unclean path", func(b *Backport) { b.Paths = []string{"vllm/../x.py"} }, nil, "clean path under vllm/"},
		{"path outside vllm", func(b *Backport) { b.Paths = []string{"tests/x.py"} }, nil, "clean path under vllm/"},
		{"empty path", func(b *Backport) { b.Paths = []string{""} }, nil, "clean path under vllm/"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b := goodBackport()
			tc.mutate(&b)
			bad := BackportProblems(append([]Backport{b}, tc.extra...), Commit)
			if tc.want == "" {
				if len(bad) != 0 {
					t.Fatalf("a well-formed record was refused: %v", bad)
				}
				return
			}
			for _, p := range bad {
				if strings.Contains(p, tc.want) {
					return
				}
			}
			t.Fatalf("no problem mentions %q: %v", tc.want, bad)
		})
	}
}

// The build's own record is well formed, and states what the pin carries: no backport.
func TestRecordedBackportsAreWellFormed(t *testing.T) {
	bs := Backports()
	if bad := BackportProblems(bs, Commit); len(bad) != 0 {
		t.Fatalf("the Upstream record's backports are malformed: %v", bad)
	}
	if len(bs) != 0 {
		t.Errorf("the pin carries %d backports; each needs a patch in core/rulepack and the equivalence check", len(bs))
	}
}

func TestBackportUses(t *testing.T) {
	b := Backport{Architectures: []string{"A", "B"}}
	for arch, want := range map[string]bool{"A": true, "B": true, "C": false, "": false, "a": false} {
		if got := b.Uses(arch); got != want {
			t.Errorf("Uses(%q) = %v, want %v", arch, got, want)
		}
	}
}

func TestBackportURLCitesTheCommit(t *testing.T) {
	if got, want := goodBackport().URL(), "https://github.com/vllm-project/vllm/commit/"+goodCommit; got != want {
		t.Errorf("URL() = %q, want %q", got, want)
	}
}

// Backports hands out a copy: a caller that edits it does not edit the record.
func TestBackportsReturnsACopy(t *testing.T) {
	saved := backports
	defer func() { backports = saved }()
	backports = []Backport{goodBackport()}
	got := Backports()
	got[0].Architectures[0] = "Changed"
	got[0].Paths[0] = "Changed"
	if want := []Backport{goodBackport()}; !reflect.DeepEqual(backports, want) || len(Backports()) != 1 {
		t.Errorf("the record moved: %v", backports)
	}
}

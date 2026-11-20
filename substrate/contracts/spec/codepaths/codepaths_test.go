package codepaths

import (
	"bytes"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/coefficient"
)

const (
	commit = "4ba787ec9959e566d3b587a796dff07673324da5"
	h1     = "1111111111111111111111111111111111111111111111111111111111111111"
	h2     = "2222222222222222222222222222222222222222222222222222222222222222"
	h3     = "3333333333333333333333333333333333333333333333333333333333333333"
)

func fn(id, hash string) Function {
	return Function{ID: id, Source: "vllm/x.py:10@4ba787ec", Hash: hash}
}

// valid builds a table whose code ids are the ones the rule computes.
func valid() *Table {
	host := []Function{fn("vllm/a.py::A.f", h1), fn("vllm/b.py::g", h2)}
	graphs := []Function{fn("vllm/c.py::C.h", h3)}
	return &Table{
		Kind:   Kind,
		Engine: Engine{Kind: "vllm", Version: "0.33.0", Commit: commit},
		Paths: []Path{
			{Path: coefficient.CodePathHost, CodeID: CodeID(coefficient.CodePathHost, host), Functions: host},
			{Path: coefficient.CodePathGraphs, CodeID: CodeID(coefficient.CodePathGraphs, graphs), Functions: graphs},
		},
	}
}

func TestValidTableValidatesAndRoundTrips(t *testing.T) {
	tab := valid()
	if err := tab.Validate(); err != nil {
		t.Fatal(err)
	}
	raw, err := tab.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	back, err := Parse("t.json", raw)
	if err != nil {
		t.Fatal(err)
	}
	again, _ := back.Marshal()
	if !bytes.Equal(raw, again) {
		t.Error("Marshal(Parse(Marshal(t))) is not stable")
	}
	for _, p := range []string{coefficient.CodePathHost, coefficient.CodePathGraphs} {
		id, ok := back.CodeIDOf(p)
		if !ok || len(id) != 64 {
			t.Errorf("CodeIDOf(%s) = %q, %v", p, id, ok)
		}
	}
	if _, ok := back.CodeIDOf("elsewhere"); ok {
		t.Error("CodeIDOf answered for a path the table does not list")
	}
}

// The CodeID is a property of the code: a function's hash changing changes it, so does adding or
// removing a function, and the path is part of it; the order functions were listed in is not.
func TestCodeIDFollowsTheCode(t *testing.T) {
	a := []Function{fn("a", h1), fn("b", h2)}
	base := CodeID("host", a)
	for name, other := range map[string]string{
		"a function's hash changed": CodeID("host", []Function{fn("a", h1), fn("b", h3)}),
		"a function removed":        CodeID("host", a[:1]),
		"a function added":          CodeID("host", append(append([]Function(nil), a...), fn("c", h3))),
		"a function renamed":        CodeID("host", []Function{fn("a", h1), fn("z", h2)}),
		"another path":              CodeID("graphs", a),
	} {
		if other == base {
			t.Errorf("%s, and the CodeID did not change", name)
		}
	}
	if CodeID("host", []Function{a[1], a[0]}) != base {
		t.Error("the CodeID depends on the order the functions are listed in")
	}
	if CodeID("host", []Function{{ID: "a", Source: "other.py:1@4ba787ec", Hash: h1}, a[1]}) != base {
		t.Error("the CodeID depends on a function's citation, which is not the code")
	}
	if len(base) != 64 || strings.Trim(base, "0123456789abcdef") != "" {
		t.Errorf("CodeID = %q, want 64 lower-case hexadecimal digits", base)
	}
}

// A CodeID the table states that is not the one its functions give is refused: the generator's
// restatement of the rule cannot drift from this one.
func TestValidateRefuses(t *testing.T) {
	for _, c := range []struct {
		name   string
		edit   func(*Table)
		reason string
	}{
		{"wrong kind", func(t *Table) { t.Kind = "Other" }, "kind"},
		{"no engine kind", func(t *Table) { t.Engine.Kind = "" }, "engine.kind"},
		{"bad commit", func(t *Table) { t.Engine.Commit = "abc" }, "engine.commit"},
		{"a CodeID the functions do not give", func(t *Table) { t.Paths[0].CodeID = strings.Repeat("a", 64) }, "code_id"},
		{"a function hash edited without the CodeID", func(t *Table) { t.Paths[0].Functions[0].Hash = h3 }, "code_id"},
		{"a path missing", func(t *Table) { t.Paths = t.Paths[:1] }, "graphs"},
		{"an unknown path", func(t *Table) { t.Paths[1].Path = "kernels" }, "path"},
		{"paths out of order", func(t *Table) { t.Paths[0], t.Paths[1] = t.Paths[1], t.Paths[0] }, "path"},
		{"a path with no function", func(t *Table) { t.Paths[1].Functions = nil; t.Paths[1].CodeID = CodeID("graphs", nil) }, "functions"},
		{"functions unsorted", func(t *Table) {
			f := t.Paths[0].Functions
			f[0], f[1] = f[1], f[0]
		}, "sorted"},
		{"a function twice", func(t *Table) {
			t.Paths[0].Functions[1].ID = t.Paths[0].Functions[0].ID
		}, "twice"},
		{"a short function hash", func(t *Table) { t.Paths[0].Functions[0].Hash = "abc" }, "hash"},
		{"a citation at another commit", func(t *Table) { t.Paths[0].Functions[0].Source = "vllm/x.py:10@deadbeef" }, "commit"},
		{"a line break in an id", func(t *Table) { t.Paths[0].Functions[0].ID = "a\nb" }, "line break"},
	} {
		t.Run(c.name, func(t *testing.T) {
			tab := valid()
			c.edit(tab)
			err := tab.Validate()
			if err == nil || !strings.Contains(err.Error(), c.reason) {
				t.Errorf("Validate = %v, want an error naming %q", err, c.reason)
			}
		})
	}
}

func TestParseIsStrict(t *testing.T) {
	raw, _ := valid().Marshal()
	if _, err := Parse("t.json", bytes.Replace(raw, []byte(`"kind"`), []byte(`"surprise": 1, "kind"`), 1)); err == nil {
		t.Error("an unknown key was accepted")
	}
}

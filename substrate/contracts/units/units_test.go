package units

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestConversions(t *testing.T) {
	s, err := Bytes(2_000_000_000).Over(BytesPerSecond(1e12))
	if err != nil || s != 0.002 {
		t.Fatalf("Over = %v, %v", s, err)
	}
	if got := s.Duration(); got != 2*time.Millisecond {
		t.Errorf("Duration = %v", got)
	}
	if got := FromDuration(2 * time.Millisecond); got != 0.002 {
		t.Errorf("FromDuration = %v", got)
	}
	if got := BytesPerSecond(5e9).In(0.5); got != 2_500_000_000 {
		t.Errorf("In = %d", got)
	}
	f, err := FLOPs(1e12).Over(FLOPsPerSecond(5e11))
	if err != nil || f != 2 {
		t.Errorf("FLOPs.Over = %v, %v", f, err)
	}
	if got := Bytes(1000).Scale(0.333); got != 333 {
		t.Errorf("Scale = %d", got)
	}
}

// A rate that is not positive and finite is refused, never an infinite time a sum hides.
func TestARateThatCannotDivideIsRefused(t *testing.T) {
	for _, r := range []float64{0, -1, math.Inf(1), math.NaN()} {
		if _, err := Bytes(1).Over(BytesPerSecond(r)); err == nil {
			t.Errorf("a rate of %v was accepted", r)
		}
		if _, err := FLOPs(1).Over(FLOPsPerSecond(r)); err == nil {
			t.Errorf("a FLOP rate of %v was accepted", r)
		}
	}
}

func TestEfficiencyLiesInZeroExcludedToOneIncluded(t *testing.T) {
	for _, v := range []float64{0.001, 0.5, 1} {
		e, err := NewEfficiency(v)
		if err != nil {
			t.Errorf("%v: %v", v, err)
		}
		if got := e.OfBytes(1000); float64(got) != v*1000 {
			t.Errorf("OfBytes(%v) = %v", v, got)
		}
		if got := e.OfFLOPs(1000); float64(got) != v*1000 {
			t.Errorf("OfFLOPs(%v) = %v", v, got)
		}
	}
	for _, v := range []float64{0, -0.1, 1.0000001, 2, math.NaN(), math.Inf(1)} {
		if _, err := NewEfficiency(v); err == nil {
			t.Errorf("%v was accepted as an efficiency", v)
		}
	}
}

func TestBasisArithmetic(t *testing.T) {
	r := Rank(Bytes(10)).Add(Rank(Bytes(5)))
	if r.Value() != 15 {
		t.Errorf("per-rank sum = %d", r.Value())
	}
	g, err := r.Across(8)
	if err != nil || g.Value() != 120 {
		t.Errorf("Across = %d, %v", g.Value(), err)
	}
	n, err := Node(Bytes(7)).OnNodes(3)
	if err != nil || n.Value() != 21 {
		t.Errorf("OnNodes = %d, %v", n.Value(), err)
	}
	if got := Group(Tokens(4)).Add(Group(Tokens(6))); got.Value() != 10 {
		t.Errorf("group sum = %d", got.Value())
	}
	if _, err := Rank(Bytes(1)).Across(0); err == nil {
		t.Error("a group of zero ranks was accepted")
	}
	if _, err := Node(Bytes(1)).OnNodes(-1); err == nil {
		t.Error("a group on -1 nodes was accepted")
	}
}

// The point of the types (SUB-16): these mixtures do not compile. Each program is type-checked
// against this package and must fail with the named complaint; the last must compile, so the
// test would also catch a harness that rejects everything.
func TestMixedUnitsDoNotCompile(t *testing.T) {
	dir, err := filepath.Abs(".")
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name, body, want string
	}{
		{"bytes plus tokens", "var _ = units.Bytes(1) + units.Tokens(1)", "mismatched types"},
		{"bytes plus seconds", "var _ = units.Bytes(1) + units.Seconds(1)", "mismatched types"},
		{"bytes per second plus flops per second", "var _ = units.BytesPerSecond(1) + units.FLOPsPerSecond(1)", "mismatched types"},
		{"per rank plus per group", "var _ = units.Rank(units.Bytes(1)).Add(units.Group(units.Bytes(1)))", "cannot use"},
		{"per rank plus per node", "var _ = units.Rank(units.Bytes(1)).Add(units.Node(units.Bytes(1)))", "cannot use"},
		{"per rank bytes plus per rank tokens", "var _ = units.Rank(units.Bytes(1)).Add(units.Rank(units.Tokens(1)))", "cannot use"},
		{"a bare number as bytes per second", "var x float64 = 1; var _ units.BytesPerSecond = x", "cannot use"},
		{"flops over a bytes rate", "var _, _ = units.FLOPs(1).Over(units.BytesPerSecond(1))", "cannot use"},
		{"same units compile", "var _ = units.Bytes(1) + units.Bytes(2); var _ = units.Rank(units.Bytes(1)).Add(units.Rank(units.Bytes(2)))", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			src := "package p\nimport \"github.com/inference-sim/blis-substrate/contracts/units\"\n" + c.body + "\n"
			err := typeCheck(t, dir, src)
			switch {
			case c.want == "" && err != nil:
				t.Errorf("should compile: %v", err)
			case c.want != "" && err == nil:
				t.Errorf("compiled; want an error containing %q", c.want)
			case c.want != "" && !strings.Contains(err.Error(), c.want):
				t.Errorf("error %q does not contain %q", err, c.want)
			}
		})
	}
}

// typeCheck compiles one source file that imports this package, with go vet in a scratch module
// that replaces the substrate with the working tree.
func typeCheck(t *testing.T, pkgDir, src string) error {
	t.Helper()
	root := filepath.Join(pkgDir, "..", "..")
	tmp := t.TempDir()
	write := func(name, content string) {
		if err := os.WriteFile(filepath.Join(tmp, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module scratch\ngo 1.24\nrequire github.com/inference-sim/blis-substrate v0.0.0\nreplace github.com/inference-sim/blis-substrate => "+root+"\n")
	write("p.go", src)
	cmd := exec.Command("go", "vet", "./...")
	cmd.Dir = tmp
	cmd.Env = append(os.Environ(), "GOFLAGS=-mod=mod", "GOWORK=off", "GOTOOLCHAIN=local")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return &compileError{string(out)}
	}
	return nil
}

type compileError struct{ out string }

func (e *compileError) Error() string { return e.out }

func TestBasisTypesSerialiseAsTheirAmount(t *testing.T) {
	b, err := json.Marshal(struct {
		A PerRank[Bytes]
		B []PerNode[Tokens]
	}{Rank(Bytes(7)), []PerNode[Tokens]{Node(Tokens(3))}})
	if err != nil || string(b) != `{"A":7,"B":[3]}` {
		t.Fatalf("%s, %v", b, err)
	}
	var back struct {
		A PerRank[Bytes]
		B []PerNode[Tokens]
	}
	if err := json.Unmarshal(b, &back); err != nil || back.A.Value() != 7 || back.B[0].Value() != 3 {
		t.Fatalf("%+v, %v", back, err)
	}
	if err := json.Unmarshal([]byte(`{"A":"x"}`), &back); err == nil {
		t.Error("a string was read as a byte count")
	}
}

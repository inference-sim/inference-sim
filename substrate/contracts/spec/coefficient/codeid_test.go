package coefficient

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/substrate/contracts/check"
	"github.com/inference-sim/inference-sim/substrate/contracts/vocab"
)

const (
	idA = "aaaaaaaaaaaaaaaa"
	idB = "bbbbbbbbbbbbbbbb"
)

func hostEntry(ids ...string) Entry {
	return Entry{Name: "host_output_token", Method: vocab.MethodMeasured, CodeIDs: ids}
}

func TestCodePathOf(t *testing.T) {
	for name, want := range map[string]string{
		"host_admission_per_request":             CodePathHost,
		"host_admission_per_token":               CodePathHost,
		"host_admission_per_token_rust_frontend": CodePathHost,
		"host_output_token":                      CodePathHost,
		"host_completion":                        CodePathHost,
		"host_step_per_request":                  CodePathHost,
		"host_launch_eager_per_layer":            CodePathGraphs,
		"host_launch_eager_per_kernel":           CodePathGraphs,
		"host_launch_per_kernel":                 CodePathGraphs,
		"host_replay_graph_per_step":             CodePathGraphs,
		"host_link_bandwidth":                    "",
		"host_gather_rate_uva":                   "",
		"gemm_eps_max_bf16":                      "",
		"hbm_derate":                             "",
	} {
		if got := CodePathOf(name); got != want {
			t.Errorf("CodePathOf(%q) = %q, want %q", name, got, want)
		}
	}
}

// A lookup matches an engine-path entry only for the code that runs: a different CodeID is
// refused, never matched as a sibling (VAL-8 applied to engine-path code).
func TestMatchCode(t *testing.T) {
	for _, c := range []struct {
		name    string
		e       Entry
		running map[string]string
		want    CodeMatch
	}{
		{"unkeyed entry applies to any code", hostEntry(), map[string]string{CodePathHost: idA}, CodeUnkeyed},
		{"unkeyed entry applies when no code is stated", hostEntry(), nil, CodeUnkeyed},
		{"the code that runs is listed", hostEntry(idA), map[string]string{CodePathHost: idA}, CodeExact},
		{"one of several listed", hostEntry(idB, idA), map[string]string{CodePathHost: idA}, CodeExact},
		{"a different CodeID is refused", hostEntry(idA), map[string]string{CodePathHost: idB}, CodeRefused},
		{"no CodeID stated by the engine is refused", hostEntry(idA), nil, CodeRefused},
		{"an empty CodeID is refused", hostEntry(idA), map[string]string{CodePathHost: ""}, CodeRefused},
		{"another path's CodeID does not match", hostEntry(idA), map[string]string{CodePathGraphs: idA}, CodeRefused},
		{"a keyed entry of a name with no code path is refused",
			Entry{Name: "gemm_eps_max_bf16", CodeIDs: []string{idA}}, map[string]string{CodePathHost: idA}, CodeRefused},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := c.e.MatchCode(c.running); got != c.want {
				t.Errorf("MatchCode = %q, want %q", got, c.want)
			}
		})
	}
}

func TestCodesCompatible(t *testing.T) {
	for _, c := range []struct {
		name string
		a, b Entry
		want bool
	}{
		{"either unkeyed", hostEntry(idA), hostEntry(), true},
		{"a CodeID in common", hostEntry(idA, idB), hostEntry(idB), true},
		{"disjoint CodeIDs never both match", hostEntry(idA), hostEntry(idB), false},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := CodesCompatible(c.a, c.b); got != c.want {
				t.Errorf("CodesCompatible = %v, want %v", got, c.want)
			}
		})
	}
}

// Two entries of one name, scope and method that differ only in the code they are valid for are
// different coefficients; listing the same ids in another order is not a difference, and an
// entry with no code ids keeps the identity it had before the field existed.
func TestIdentityCodeIDs(t *testing.T) {
	if hostEntry(idA).Identity() == hostEntry(idB).Identity() {
		t.Error("entries valid for different code share an identity")
	}
	if hostEntry(idA, idB).Identity() != hostEntry(idB, idA).Identity() {
		t.Error("the order of code ids changed the identity")
	}
	e := hostEntry()
	want := e.Name + "\x00" + e.Scope.Key() + "\x00" + string(e.Method) + "\x00" + DescribeImpls(nil)
	if got := e.Identity(); got != want {
		t.Errorf("an unkeyed entry's identity changed: %q, want %q", got, want)
	}
}

func TestValidateCodeIDs(t *testing.T) {
	for _, c := range []struct {
		name string
		e    Entry
		want string
	}{
		{"valid", hostEntry(idA, idB), ""},
		{"not hexadecimal", hostEntry("Not-A-Hash-At-All"), "lower-case hexadecimal"},
		{"too short", hostEntry("abc"), "lower-case hexadecimal"},
		{"duplicate", hostEntry(idA, idA), "duplicate"},
		{"on a name with no code path", Entry{Name: "gemm_eps_max_bf16", Method: vocab.MethodMeasured,
			CodeIDs: []string{idA}}, "prices no engine-path code"},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := &validate.Problems{}
			c.e.validateCodeIDs(p, "e")
			got := ""
			if !p.OK() {
				got = p.Error()
			}
			if c.want == "" && got != "" || !strings.Contains(got, c.want) {
				t.Errorf("problems = %q, want to contain %q", got, c.want)
			}
		})
	}
}

func TestScopeHostCPU(t *testing.T) {
	var s Scope
	if err := yaml.Unmarshal([]byte("{hardware: [h200], host_cpu: [x86]}"), &s); err != nil {
		t.Fatalf("host_cpu is a scope dimension: %v", err)
	}
	if s.Empty() {
		t.Error("a scope stating only host_cpu is not empty")
	}
	if !strings.Contains(s.Key(), "host_cpu=x86") {
		t.Errorf("Key() = %q omits host_cpu", s.Key())
	}
	if (Scope{HostCPU: []string{"x86"}}).Key() == (Scope{HostCPU: []string{"grace"}}).Key() {
		t.Error("scopes on different host CPUs share a key")
	}
	if !s.Admits(Point{Hardware: "h200", HostCPU: "x86"}) {
		t.Error("the host CPU the scope lists is admitted")
	}
	if s.Admits(Point{Hardware: "h200", HostCPU: "grace"}) {
		t.Error("a host CPU the scope does not list is admitted")
	}
	if s.Admits(Point{Hardware: "h200"}) {
		t.Error("a point that states no host CPU is admitted by a scope that lists one")
	}
	if !(Scope{Hardware: []string{"h200"}}).Admits(Point{Hardware: "h200"}) {
		t.Error("a scope that states no host CPU admits any")
	}
}

// host_cpu is the last dimension of the precedence: an entry that states it is narrower than
// one that does not, and two that list different CPUs are not ordered.
func TestPrecedenceHostCPU(t *testing.T) {
	open := Scope{Hardware: []string{"h200"}}
	x86 := Scope{Hardware: []string{"h200"}, HostCPU: []string{"x86"}}
	grace := Scope{Hardware: []string{"h200"}, HostCPU: []string{"grace"}}
	if cmp, ok := ComparePrecedence(x86, open); !ok || cmp != -1 {
		t.Errorf("stating host_cpu should win: cmp=%d ordered=%v", cmp, ok)
	}
	if _, ok := ComparePrecedence(x86, grace); ok {
		t.Error("entries on different host CPUs were ordered")
	}
	if PrecedenceOrder[len(PrecedenceOrder)-1] != "host_cpu" {
		t.Errorf("host_cpu should be the last precedence dimension: %v", PrecedenceOrder)
	}
}

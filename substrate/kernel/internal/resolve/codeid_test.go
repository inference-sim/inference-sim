package resolve

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/coefficient"
	"github.com/inference-sim/inference-sim/substrate/contracts/vocab"
)

const (
	hostA   = "aaaaaaaaaaaaaaaa"
	hostB   = "bbbbbbbbbbbbbbbb"
	graphsA = "cccccccccccccccc"
)

func codeSets(ids ...string) []*coefficient.Set {
	return []*coefficient.Set{{Kind: "CoefficientSet", Name: "host", Coefficients: []coefficient.Entry{
		{Name: "host_output_token", Value: 45.9, Method: vocab.MethodMeasured, CodeIDs: ids,
			Scope: coefficient.Scope{Hardware: []string{"h200"}}},
		{Name: "host_launch_per_kernel", Value: 3.2, Method: vocab.MethodMeasured, CodeIDs: []string{graphsA},
			Scope: coefficient.Scope{Hardware: []string{"h200"}}},
	}}}
}

// A lookup of an engine-path coefficient fails on a different CodeID: the entry is not returned,
// it is listed as refused, and the error that follows names the code that ran.
func TestEnginePathLookupFailsOnADifferentCodeID(t *testing.T) {
	for _, c := range []struct {
		name    string
		code    map[string]string
		listed  []string
		wantHas bool
	}{
		{"the code that runs", map[string]string{coefficient.CodePathHost: hostA, coefficient.CodePathGraphs: graphsA}, []string{hostA}, true},
		{"a different CodeID", map[string]string{coefficient.CodePathHost: hostB, coefficient.CodePathGraphs: graphsA}, []string{hostA}, false},
		{"no CodeID stated", map[string]string{coefficient.CodePathGraphs: graphsA}, []string{hostA}, false},
		{"an unkeyed entry", map[string]string{coefficient.CodePathHost: hostB, coefficient.CodePathGraphs: graphsA}, nil, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			co, err := Load(codeSets(c.listed...), Scope{Hardware: "h200", CodeIDs: c.code})
			if err != nil {
				t.Fatal(err)
			}
			if got := co.Has("host_output_token"); got != c.wantHas {
				t.Fatalf("Has(host_output_token) = %v, want %v", got, c.wantHas)
			}
			if !c.wantHas {
				if _, err := co.Value("host_output_token"); err == nil || !strings.Contains(err.Error(), "CodeID") {
					t.Errorf("the lookup's error should name the CodeID refusal, got %v", err)
				}
				if len(co.CodeRefused()) != 1 || co.CodeRefused()[0].Entry.Name != "host_output_token" {
					t.Errorf("CodeRefused() = %+v, want the host_output_token entry", co.CodeRefused())
				}
			}
			if !co.Has("host_launch_per_kernel") {
				t.Error("the graph-path entry is keyed by the graphs CodeID, which ran")
			}
		})
	}
}

// The path decides which CodeID an entry is compared with: a host entry is not matched by the
// graphs path's CodeID, even when it is the same digest.
func TestEnginePathEntryIsComparedWithItsOwnPath(t *testing.T) {
	co, err := Load(codeSets(hostA), Scope{Hardware: "h200",
		CodeIDs: map[string]string{coefficient.CodePathHost: graphsA, coefficient.CodePathGraphs: hostA}})
	if err == nil {
		t.Fatalf("an entry was matched by another path's CodeID: %v", co.Names())
	}
	if !strings.Contains(err.Error(), "CodeID") {
		t.Errorf("the error should name the CodeID refusal, got %v", err)
	}
}

// host_cpu scopes an engine-path entry: the chip's host CPU must be one the entry lists.
func TestHostCPUScopesAnEntry(t *testing.T) {
	sets := []*coefficient.Set{{Kind: "CoefficientSet", Name: "host", Coefficients: []coefficient.Entry{
		{Name: "host_output_token", Value: 1, Method: vocab.MethodAssumed,
			Scope: coefficient.Scope{Hardware: []string{"h200"}, HostCPU: []string{"x86"}}}}}}
	for cpu, want := range map[string]bool{"x86": true, "grace": false, "": false} {
		co, err := Load(sets, Scope{Hardware: "h200", HostCPU: cpu})
		if want != (err == nil && co.Has("host_output_token")) {
			t.Errorf("host_cpu %q: admitted = %v, want %v (err %v)", cpu, !want, want, err)
		}
	}
}

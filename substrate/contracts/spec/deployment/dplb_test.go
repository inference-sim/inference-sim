package deployment

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestDataParallelLoadBalancingValidation(t *testing.T) {
	for _, c := range []struct {
		mode DPLoadBalancing
		dp   int
		ok   bool
	}{
		{"", 1, true},
		{DPLBInternal, 1, true},
		{DPLBHybrid, 2, true},
		{DPLBExternal, 8, true},
		{DPLBHybrid, 1, false},
		{DPLBExternal, 1, false},
		{"round_robin", 8, false},
	} {
		p := (&Deployment{Kind: "Deployment", Name: "d", Pools: []Pool{{
			Role: RoleColocated, Nodes: 1,
			Parallel: Parallelism{TP: 1, PP: 1, DP: c.dp, DataParallelLoadBalancing: c.mode}}}}).Validate()
		if p.OK() != c.ok {
			t.Errorf("%q at dp %d: ok=%v, want %v\n%s", c.mode, c.dp, p.OK(), c.ok, p.Error())
		}
	}
	if DPLoadBalancing("").Mode() != DPLBInternal {
		t.Error("an unstated mode is not internal")
	}
}

// Load, marshal, load: every mode comes back as itself, and the default is omitted.
func TestDataParallelLoadBalancingRoundTrips(t *testing.T) {
	for _, m := range []DPLoadBalancing{"", DPLBInternal, DPLBHybrid, DPLBExternal} {
		var in Parallelism
		if err := yaml.Unmarshal([]byte("{tp: 1, pp: 1, dp: 4, data_parallel_load_balancing: "+string(m)+"}"), &in); err != nil {
			t.Fatal(err)
		}
		out, err := yaml.Marshal(in)
		if err != nil {
			t.Fatal(err)
		}
		var back Parallelism
		if err := yaml.Unmarshal(out, &back); err != nil || back != in || back.DataParallelLoadBalancing != m {
			t.Fatalf("%q round-tripped through %q to %+v (%v)", m, out, back, err)
		}
		if m == "" && containsKey(string(out), "data_parallel_load_balancing") {
			t.Fatalf("the default was written: %s", out)
		}
	}
}

// containsKey reports whether a YAML document has a top-level key.
func containsKey(doc, key string) bool {
	for _, line := range strings.Split(doc, "\n") {
		if strings.HasPrefix(line, key+":") {
			return true
		}
	}
	return false
}

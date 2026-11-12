package fields

import (
	"testing"
)

func nodes(n int) *int { return &n }

func TestARuleReadBackFromItsCanonicalBytesStillDecides(t *testing.T) {
	l, err := Parse(listWith(allReduce))
	if err != nil {
		t.Fatal(err)
	}
	out, err := l.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	again, err := Parse(out)
	if err != nil {
		t.Fatal(err)
	}
	if !Exact(again.Fields[0].Estimation, Facts{Value: "mnnvl", TPGroupNodes: nodes(2), ComputeCapability: "10.3"}) {
		t.Error("the rule read back from the canonical bytes lost a clause")
	}
}

func TestASubstituteRuleMayStateWhenItIsExact(t *testing.T) {
	e := parseEstimation(t, `{"rule": "substitute", "setting": {"kv_cache_dtype": "auto"}, "reason": "r",
	  "preserves": "p", "valid_when": [{"value_in": ["fp8"]}]}`)
	if !Exact(e, Facts{Value: "fp8"}) || Exact(e, Facts{Value: "int4"}) {
		t.Error("a substitute rule's valid_when was not applied")
	}
}

// The worked example holds only on the clauses it states.
func TestTheAllReduceStandInIsExactOnlyWhereItsClausesHold(t *testing.T) {
	e := parseEstimation(t, allReduce)
	for _, c := range []struct {
		name  string
		facts Facts
		want  bool
	}{
		{"auto needs no fact", Facts{Value: "auto"}, true},
		{"auto in other case and padding", Facts{Value: "  Auto "}, true},
		{"mnnvl on one node", Facts{Value: "mnnvl", TPGroupNodes: nodes(1)}, true},
		{"mnnvl on one node of an older part", Facts{Value: "mnnvl", TPGroupNodes: nodes(1), ComputeCapability: "9.0"}, true},
		{"mnnvl on two nodes at 10.0", Facts{Value: "mnnvl", TPGroupNodes: nodes(2), ComputeCapability: "10.0"}, true},
		{"mnnvl on two nodes at 10.3", Facts{Value: "mnnvl", TPGroupNodes: nodes(2), ComputeCapability: "10.3"}, true},
		{"mnnvl on two nodes at 12.0", Facts{Value: "mnnvl", TPGroupNodes: nodes(2), ComputeCapability: "12.0"}, true},
		{"mnnvl on two nodes below 10.0", Facts{Value: "mnnvl", TPGroupNodes: nodes(2), ComputeCapability: "9.0"}, false},
		{"capability orders by major, then minor", Facts{Value: "mnnvl", TPGroupNodes: nodes(2), ComputeCapability: "9.10"}, false},
		{"mnnvl on two nodes, capability unknown", Facts{Value: "mnnvl", TPGroupNodes: nodes(2)}, false},
		{"mnnvl on two nodes, capability malformed", Facts{Value: "mnnvl", TPGroupNodes: nodes(2), ComputeCapability: "ten"}, false},
		{"mnnvl, node count unknown", Facts{Value: "mnnvl", ComputeCapability: "10.0"}, false},
		{"mnnvl with no facts", Facts{Value: "mnnvl"}, false},
		{"a value no clause names", Facts{Value: "trtllm", TPGroupNodes: nodes(1), ComputeCapability: "10.0"}, false},
		{"an empty value", Facts{TPGroupNodes: nodes(1)}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := Exact(e, c.facts); got != c.want {
				t.Errorf("Exact(%+v) = %v, want %v", c.facts, got, c.want)
			}
		})
	}
}

// Within a clause every key must hold; across clauses one is enough.
func TestAClauseIsAConjunctionAndTheListIsADisjunction(t *testing.T) {
	e := parseEstimation(t, `{"rule": "leave_unset", "preserves": "p", "valid_when": [
	  {"value_in": ["a"], "tp_group_nodes": {"gte": 2, "lt": 4}},
	  {"value_in": ["b"]}]}`)
	for _, c := range []struct {
		facts Facts
		want  bool
	}{
		{Facts{Value: "a", TPGroupNodes: nodes(2)}, true},
		{Facts{Value: "a", TPGroupNodes: nodes(3)}, true},
		{Facts{Value: "a", TPGroupNodes: nodes(4)}, false}, // lt fails though gte holds
		{Facts{Value: "a", TPGroupNodes: nodes(1)}, false}, // gte fails though lt holds
		{Facts{Value: "b"}, true},
		{Facts{Value: "c", TPGroupNodes: nodes(2)}, false},
	} {
		if got := Exact(e, c.facts); got != c.want {
			t.Errorf("Exact(%+v) = %v, want %v", c.facts, got, c.want)
		}
	}
}

// The UCX_TLS clause: a list that names a RDMA transport, and is not a deny list.
func TestListClausesSplitTheValueOnCommas(t *testing.T) {
	e := parseEstimation(t, `{"rule": "leave_unset", "preserves": "p", "valid_when": [
	  {"value_list_has_any": ["rc", "dc_x", "all"], "value_not_prefixed": ["^"]},
	  {"value_list_has_none": ["tcp", "cuda_ipc"], "value_in": ["sm,self"]}]}`)
	for _, c := range []struct {
		value string
		want  bool
	}{
		{"rc", true},
		{"cuda_copy, RC_X ,rc", true}, // padding and case are ignored; one match is enough
		{"dc_x,tcp", true},
		{"all", true},
		{"^rc", false},    // a deny list names rc but excludes it
		{"tcp,sm", false}, // no RDMA transport
		{"rcx", false},    // an item matches whole, not by prefix
		{"sm,self", true}, // second clause: value_in and has_none both hold
		{"", false},
	} {
		if got := Exact(e, Facts{Value: c.value}); got != c.want {
			t.Errorf("Exact(%q) = %v, want %v", c.value, got, c.want)
		}
	}
	none := parseEstimation(t, `{"rule": "leave_unset", "preserves": "p", "valid_when": [{"value_list_has_none": ["tcp"]}]}`)
	if Exact(none, Facts{Value: "sm,tcp"}) || !Exact(none, Facts{Value: "sm,self"}) || !Exact(none, Facts{Value: ""}) {
		t.Error("value_list_has_none is wrong")
	}
}

// A rule without valid_when applies to every value, as before the key existed.
func TestWithoutValidWhenTheRuleAppliesToEveryValue(t *testing.T) {
	for _, est := range []string{`{"rule": "leave_unset"}`,
		`{"rule": "leave_unset", "preserves": "the default"}`,
		`{"rule": "substitute", "setting": {"a": 1}, "reason": "r"}`} {
		e := parseEstimation(t, est)
		for _, v := range []string{"", "anything", "mnnvl"} {
			if !Exact(e, Facts{Value: v}) {
				t.Errorf("Exact(%s, %q) = false, want true", est, v)
			}
		}
	}
}

// A rule with no stand-in is never exact, whatever the value.
func TestRuleNoneIsNeverExact(t *testing.T) {
	if Exact(parseEstimation(t, `{"rule": "none"}`), Facts{Value: "auto"}) {
		t.Error("rule none reported a stand-in as exact")
	}
}

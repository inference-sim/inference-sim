package fields

import (
	"strings"
	"testing"
)

// listWith is a one-field list whose estimation block is est, a JSON object body.
func listWith(est string) []byte {
	return []byte(`{
  "kind": "FieldList",
  "engine": {"kind": "vllm", "version": "0.32.0"},
  "fields": [
    {"name": "VLLM_FLASHINFER_ALLREDUCE_BACKEND", "source": "env", "status": "not_modelled",
     "evidence": {"issue": "inference-sim/blis-vllm#5"},
     "estimation": ` + est + `}
  ]
}`)
}

// threeClauses is a rule that uses every kind of clause: a value alone, a value with a node
// comparison, and a value with a node comparison and a capability comparison. It checks the shape
// the loader reads and writes back; it does not say what any engine's setting does.
const threeClauses = `{
  "rule": "leave_unset",
  "preserves": "what the upstream resolves identically",
  "valid_when": [
    {"value_in": ["auto"]},
    {"value_in": ["mnnvl"], "tp_group_nodes": {"eq": 1}},
    {"value_in": ["mnnvl"], "tp_group_nodes": {"gt": 1}, "compute_capability": {"gte": "10.0"}}
  ]
}`

func parseEstimation(t *testing.T, est string) Estimation {
	t.Helper()
	l, err := Parse(listWith(est))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	return l.Fields[0].Estimation
}

func TestARuleWithValidWhenLoadsAndWritesBackItsOwnBytes(t *testing.T) {
	l, err := Parse(listWith(threeClauses))
	if err != nil {
		t.Fatal(err)
	}
	e := l.Fields[0].Estimation
	if e.Preserves == "" || len(e.ValidWhen) != 3 {
		t.Fatalf("estimation read as %+v", e)
	}
	if got := e.ValidWhen[2]; got.TPGroupNodes == nil || *got.TPGroupNodes.Gt != 1 || *got.ComputeCapability.Gte != "10.0" {
		t.Errorf("third clause read as %+v", got)
	}
	out, err := l.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	again, err := Parse(out)
	if err != nil {
		t.Fatalf("the canonical bytes do not parse: %v\n%s", err, out)
	}
	if got, _ := again.Marshal(); string(got) != string(out) {
		t.Errorf("Marshal is not stable:\n%s\n%s", out, got)
	}
	if len(again.Fields[0].Estimation.ValidWhen) != 3 || again.Fields[0].Estimation.Preserves != e.Preserves {
		t.Error("the canonical bytes lost a clause or the preserves sentence")
	}
}

// Without valid_when the rule applies to every value, and a list that states neither member
// still loads, so no engine list written before the members existed is refused.
func TestAnEstimationWithoutTheNewMembersStillLoads(t *testing.T) {
	e := parseEstimation(t, `{"rule": "leave_unset"}`)
	if e.ValidWhen != nil || e.Preserves != "" {
		t.Errorf("read as %+v", e)
	}
	e = parseEstimation(t, `{"rule": "leave_unset", "preserves": "p"}`)
	if e.ValidWhen != nil || e.Preserves != "p" {
		t.Errorf("preserves alone read as %+v", e)
	}
}

// A comparison may set several operators, and a capability orders by major and then minor.
func TestACapabilityComparisonIsReadAsWritten(t *testing.T) {
	e := parseEstimation(t, `{"rule": "leave_unset", "preserves": "p",
	  "valid_when": [{"compute_capability": {"gte": "9.0", "lt": "10.0"}}]}`)
	c := e.ValidWhen[0].ComputeCapability
	if c == nil || *c.Gte != "9.0" || *c.Lt != "10.0" || c.Eq != nil {
		t.Errorf("read as %+v", c)
	}
	if a, _ := parseCapability("9.10"); a.cmp(capability{10, 0}) >= 0 {
		t.Error("9.10 must order below 10.0")
	}
}

func TestEveryBadExactnessStatementIsRefused(t *testing.T) {
	const pre = `"rule": "leave_unset", "preserves": "p", `
	for _, c := range []struct {
		name, est string
		want      []string
	}{
		{"unknown clause key", `{` + pre + `"valid_when": [{"value_is": ["auto"]}]}`,
			[]string{"valid_when[0]", `unknown key "value_is"`}},
		{"unknown comparison operator", `{` + pre + `"valid_when": [{"tp_group_nodes": {"ge": 1}}]}`,
			[]string{"valid_when[0].tp_group_nodes", `unknown key "ge"`}},
		{"unknown key on the estimation", `{` + pre + `"valid_if": []}`, []string{`unknown key "valid_if"`}},
		{"capability that is not major.minor", `{` + pre + `"valid_when": [{"compute_capability": {"gte": "10"}}]}`,
			[]string{"valid_when[0].compute_capability.gte", "<major>.<minor>"}},
		{"capability that is not a number", `{` + pre + `"valid_when": [{"compute_capability": {"lt": "hopper"}}]}`,
			[]string{"compute_capability.lt"}},
		{"capability given as a number", `{` + pre + `"valid_when": [{"compute_capability": {"gte": 10.0}}]}`,
			[]string{"gte"}},
		{"node count that is not a whole number", `{` + pre + `"valid_when": [{"tp_group_nodes": {"eq": 1.5}}]}`,
			[]string{"eq"}},
		{"empty clause", `{` + pre + `"valid_when": [{}]}`, []string{"valid_when[0]", "at least one key"}},
		{"empty comparison", `{` + pre + `"valid_when": [{"tp_group_nodes": {}}]}`,
			[]string{"valid_when[0].tp_group_nodes", "at least one of eq"}},
		{"empty valid_when", `{` + pre + `"valid_when": []}`, []string{"valid_when", "at least one clause"}},
		{"empty string list", `{` + pre + `"valid_when": [{"value_in": []}]}`, []string{"value_in", "at least one string"}},
		{"blank string in a list", `{` + pre + `"valid_when": [{"value_in": ["auto", " "]}]}`, []string{"value_in[1]"}},
		{"valid_when without preserves", `{"rule": "leave_unset", "valid_when": [{"value_in": ["auto"]}]}`,
			[]string{"estimation.preserves", "required with valid_when"}},
		{"valid_when on rule none", `{"rule": "none", "valid_when": [{"value_in": ["auto"]}]}`,
			[]string{"estimation.valid_when", "only with rule"}},
		{"preserves on rule none", `{"rule": "none", "preserves": "p"}`, []string{"estimation.preserves", "only with rule"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := Parse(listWith(c.est))
			if err == nil {
				t.Fatal("Parse accepted it")
			}
			for _, w := range c.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error does not mention %q:\n%v", w, err)
				}
			}
		})
	}
}

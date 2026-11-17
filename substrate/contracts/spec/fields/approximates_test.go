package fields

import (
	"strings"
	"testing"
)

// approximation is a rule-none field whose value on one part is priced as another execution mode.
// It checks the shape the loader reads; it does not say what any engine's setting does.
const approximation = `{
  "rule": "none",
  "approximates": {
    "mode": {"kernel_config.linear_backend": "auto"},
    "evidence": "the auto kernel's measured ramp, the nearest kernel any table measures",
    "differs": "the selected kernel's own speed, which no table measures",
    "when": [{"value_in": ["fast"], "compute_capability": {"gte": "10.0", "lt": "11.0"}}]
  }
}`

// An approximation covers exactly the values and parts its clauses name, and is never exact:
// a point it covers is priced as the substituted mode and says so, it is not stood in for.
func TestAnApproximationCoversItsClausesAndIsNeverExact(t *testing.T) {
	e := parseEstimation(t, approximation)
	for _, c := range []struct {
		value, cc string
		covers    bool
	}{
		{"fast", "10.0", true}, {"FAST", "10.3", true}, {"fast", "9.0", false}, {"slow", "10.0", false}, {"fast", "", false},
	} {
		f := Facts{Value: c.value, ComputeCapability: c.cc}
		a, ok := Approximate(e, f)
		if ok != c.covers {
			t.Errorf("%q on %q: covered %v, want %v", c.value, c.cc, ok, c.covers)
		}
		if ok && (a.Mode["kernel_config.linear_backend"] != "auto" || a.Differs == "") {
			t.Errorf("%q on %q: approximation %+v names no mode or difference", c.value, c.cc, a)
		}
		if Exact(e, f) {
			t.Errorf("%q on %q: an approximation is reported exact", c.value, c.cc)
		}
	}
	if ExecutionModeSubstituted != "execution_mode" {
		t.Errorf("a covered point carries %q, want the point record's execution_mode", ExecutionModeSubstituted)
	}
}

// An approximation states its mode, its evidence, what differs and the cases it covers, and only
// on a field with no exact stand-in.
func TestAnApproximationMissingAPartIsRefused(t *testing.T) {
	for _, c := range []struct{ edit, want string }{
		{`"mode": {"kernel_config.linear_backend": "auto"},`, "approximates.mode"},
		{`"evidence": "the auto kernel's measured ramp, the nearest kernel any table measures",`, "approximates.evidence"},
		{`"differs": "the selected kernel's own speed, which no table measures",`, "approximates.differs"},
	} {
		_, err := Parse(listWith(strings.Replace(approximation, c.edit, "", 1)))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("without %s: %v", c.want, err)
		}
	}
	noWhen := strings.Replace(approximation, `"when": [{"value_in": ["fast"], "compute_capability": {"gte": "10.0", "lt": "11.0"}}]`, `"when": []`, 1)
	noWhen = strings.Replace(noWhen, `which no table measures",`, `which no table measures",`, 1)
	if _, err := Parse(listWith(noWhen)); err == nil || !strings.Contains(err.Error(), "approximates.when") {
		t.Errorf("an approximation covering every value: %v", err)
	}
	withStandIn := strings.Replace(approximation, `"rule": "none"`, `"rule": "leave_unset"`, 1)
	if _, err := Parse(listWith(withStandIn)); err == nil || !strings.Contains(err.Error(), "approximates") {
		t.Errorf("an approximation beside an exact stand-in: %v", err)
	}
}

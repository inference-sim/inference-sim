package vocab

import "testing"

// TestVocabulariesAreClosed pins each vocabulary's membership. A term added
// upstream should require a deliberate change here, not appear by accident.
func TestVocabulariesAreClosed(t *testing.T) {
	cases := []struct {
		name string
		got  []string
		want []string
	}{
		{"units", AllUnits(), []string{
			"bytes_per_rank", "bytes_per_us", "dimensionless", "sm_count",
			"tokens", "us_per_hop", "us_per_layer", "us_per_load",
			"us_per_request", "us_per_step", "us_per_token", "us_per_transfer"}},
		{"methods", AllMethods(), []string{
			"assumed", "calibrated", "copied", "literature", "measured",
			"not_charged", "vendor_spec"}},
		{"scope keys", AllScopeKeys(), []string{
			"ep", "fabric", "hardware", "model", "nodes_spanned", "tp"}},
		{"provenances", AllProvenances(), []string{"derived", "vendor_spec"}},
		{"source kinds", AllSourceKinds(), []string{
			"datasheet", "discussion", "measured_case", "model", "publication",
			"vendor_doc"}},
		{"source roles", AllSourceRoles(), []string{
			"primary", "supporting", "upper_bound"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if len(c.got) != len(c.want) {
				t.Fatalf("size = %d, want %d: %v", len(c.got), len(c.want), c.got)
			}
			for i := range c.want {
				if c.got[i] != c.want[i] {
					t.Errorf("[%d] = %q, want %q", i, c.got[i], c.want[i])
				}
			}
		})
	}
}

func TestUnknownValuesRejected(t *testing.T) {
	if Unit("us_per_femtosecond").Valid() {
		t.Error("an invented unit was accepted")
	}
	if Method("guessed").Valid() {
		t.Error("an invented method was accepted")
	}
	if ScopeKey("dtype").Valid() {
		t.Error("dtype is deliberately not a scope key; it rides in the entry name")
	}
	if ScopeKey("backend").Valid() {
		t.Error("backend is deliberately not a scope key; it rides in the entry name")
	}
	if Provenance("vibes").Valid() {
		t.Error("an invented provenance was accepted")
	}
}

// TestEvidenced separates methods resting on an observation from those resting on a
// judgement. A vendor spec counts as evidenced although it is nominal, which is why
// the method is not a quality ranking.
func TestEvidenced(t *testing.T) {
	for _, m := range []Method{MethodMeasured, MethodLiterature, MethodVendorSpec, MethodCalibrated} {
		if !m.Evidenced() {
			t.Errorf("%s should be evidenced", m)
		}
	}
	for _, m := range []Method{MethodAssumed, MethodCopied, MethodNotCharged} {
		if m.Evidenced() {
			t.Errorf("%s should not be evidenced", m)
		}
	}
}

func TestUnknownValueErrorNamesTheAllowedSet(t *testing.T) {
	err := &UnknownValueError{Field: "units", Value: "furlongs",
		Allowed: []string{"tokens", "us_per_step"}}
	msg := err.Error()
	for _, want := range []string{"units", "furlongs", "tokens", "us_per_step"} {
		if !contains(msg, want) {
			t.Errorf("message %q omits %q", msg, want)
		}
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

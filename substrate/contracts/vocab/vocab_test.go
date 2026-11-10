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
			"bytes_per_rank", "bytes_per_token", "bytes_per_us", "dimensionless", "sm_count",
			"tokens", "us_per_hop", "us_per_layer", "us_per_load",
			"us_per_request", "us_per_step", "us_per_token", "us_per_transfer"}},
		{"methods", AllMethods(), []string{
			"assumed", "calibrated", "literature", "measured",
			"not_charged", "roofline", "transferred", "vendor_spec"}},
		{"scope keys", AllScopeKeys(), []string{
			"ep", "fabric", "hardware", "model", "nodes_spanned", "tp"}},
		{"provenances", AllProvenances(), []string{"derived", "vendor_spec"}},
		{"source kinds", AllSourceKinds(), []string{
			"datasheet", "discussion", "measured_case", "model", "publication",
			"vendor_doc"}},
		{"source roles", AllSourceRoles(), []string{
			"primary", "supporting", "upper_bound"}},
		{"rate bases", AllRateBases(), []string{"memory", "work"}},
		{"fit dimensions", AllFitDims(), []string{"context_tokens", "requests", "tokens"}},
		{"measurement times", AllMeasurementTimes(), []string{"device", "device_and_host"}},
		{"measurement executions", AllMeasurementExecutions(), []string{"captured", "eager"}},
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
	for _, m := range []Method{MethodAssumed, MethodTransferred, MethodRoofline, MethodNotCharged} {
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

// The licence allow-list of X12 and D6 is closed, and its identifiers are exact: a
// lower-case or spelled-out form is refused, and so is a LicenseRef that names no text.
func TestLicenceAllowList(t *testing.T) {
	want := []string{"Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC-BY-4.0", "CC0-1.0",
		"LicenseRef-deepseek", "LicenseRef-glm-5.3", "LicenseRef-kimi-k3", "LicenseRef-llama2",
		"LicenseRef-llama3.1", "LicenseRef-llama4", "LicenseRef-minimax-community",
		"LicenseRef-modified-mit", "LicenseRef-muse-model-license",
		"LicenseRef-muse-spark-open", "LicenseRef-openmdw-1.1", "MIT"}
	got := RedistributableLicences()
	if len(got) != len(want) {
		t.Fatalf("allow-list %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] = %q, want %q", i, got[i], want[i])
		}
	}
	for _, bad := range []string{"apache-2.0", "Apache 2.0", "GPL-3.0", "NOASSERTION", "",
		"LicenseRef-other", "LicenseRef-none-stated", "llama2"} {
		if LicenceRedistributable(bad) {
			t.Errorf("%q is on the allow-list", bad)
		}
	}
}

// TestRankFollowsDesignSection92 pins the order of design §9.2: the smallest number is the
// most trusted, every method but not_charged has one, and no two share a number.
func TestRankFollowsDesignSection92(t *testing.T) {
	order := []Method{MethodMeasured, MethodCalibrated, MethodTransferred, MethodLiterature,
		MethodVendorSpec, MethodRoofline, MethodAssumed}
	for i, m := range order {
		if r, ok := m.Rank(); !ok || r != i+1 {
			t.Errorf("%s: Rank() = %d, %v; want %d, true", m, r, ok, i+1)
		}
	}
	if r, ok := MethodNotCharged.Rank(); ok || r != 0 {
		t.Errorf("not_charged is outside the ranking, got %d, %v", r, ok)
	}
	if _, ok := Method("copied").Rank(); ok {
		t.Error("copied was retired in favour of transferred and must not rank")
	}
	for _, m := range AllMethods() {
		if _, ranked := Method(m).Rank(); !ranked && Method(m) != MethodNotCharged {
			t.Errorf("method %q is neither ranked nor not_charged", m)
		}
	}
}

// A file's Provenance is what an omitted field inherits, so it must itself be a field source
// (SUB-18): vendor_spec is the datasheet's figures, derived is not a source at all.
func TestProvenanceAsAFieldSource(t *testing.T) {
	got, err := ProvenanceVendorSpec.FieldSource()
	if err != nil || got.Kind != FieldDatasheet || got.Device != "" {
		t.Errorf("vendor_spec -> %+v, %v; want datasheet", got, err)
	}
	if _, err := ProvenanceDerived.FieldSource(); err == nil {
		t.Error("derived was accepted as a source of a catalog fact")
	}
	if _, err := Provenance("rumour").FieldSource(); err == nil {
		t.Error("an unrecognized provenance was accepted as a source")
	}
}

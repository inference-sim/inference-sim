// Original to blis-vllm.

package blisvllm

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
)

// A scoring harness reads fields.json and, for each setting a recorded run made, either states it to the
// engine (modelled), records it and states nothing (proven inert), takes the field's estimation rule
// (not modelled, with a rule) or refuses the point (the release has no such field, or it is not modelled
// and has no rule). The sixteen InferenceX DeepSeek-V4.1-Flash H200 rows (vLLM v0.33.0, agentic, DSpark)
// were refused for the three settings below; each is now one the harness states or records, at the value
// the rows used.
func TestTheH200RowsSettingsAreNotRefusedByTheFieldList(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name   string
		source fields.Source
		used   any // the value the rows used, as the class check reads it
	}{
		{"PYTHONUNBUFFERED", fields.SourceEnv, 1.0},
		{"VLLM_USE_RUST_FRONTEND", fields.SourceEnv, true},
		{"speculative_config.rejection_sample_method", fields.SourceConfig, "synthetic"},
	} {
		var f *fields.Field
		for i := range l.Fields {
			if l.Fields[i].Name == c.name && l.Fields[i].Source == c.source {
				f = &l.Fields[i]
			}
		}
		if f == nil {
			t.Errorf("%s: not in the list, so the harness refuses it as a field the release does not have", c.name)
			continue
		}
		if f.Status != fields.StatusModelled && f.Status != fields.StatusProvenInert {
			t.Errorf("%s: status %s with rule %s: the harness refuses or estimates it", c.name, f.Status, f.Estimation.Rule)
			continue
		}
		if ok, err := f.ValueClasses.Covers(c.used); err != nil || !ok {
			t.Errorf("%s: the value the rows used (%v) is outside the classes the evidence covers (%v)", c.name, c.used, err)
		}
	}
}

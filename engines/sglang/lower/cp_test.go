// Original to blis-sglang.

package lower

import (
	"errors"
	"testing"

	"github.com/inference-sim/inference-sim/engines/sglang/config"
)

// TestContextParallelismIsRefusedByName: each context-parallel setting is refused by its SGLang
// name, citing the issue that models it, by config.Resolve when a user states it and by lowering
// when a ServerArgs holds it; the same input without it resolves and lowers (SIM-9).
func TestContextParallelismIsRefusedByName(t *testing.T) {
	c := readLower(t).Cases[0]
	base := map[string]any{"model_path": "<model>"}
	r, err := config.Resolve(config.Input{Args: base, Chip: c.Chip, Model: c.ModelFacts})
	if err != nil {
		t.Fatalf("the accepted partner does not resolve: %v", err)
	}
	if _, err := Alignment(&r.Args); err != nil {
		t.Fatalf("the accepted partner does not lower: %v", err)
	}
	strategy := "zigzag"
	for _, tc := range []struct {
		field string
		value any
		set   func(*config.ServerArgs)
	}{
		{"attn_cp_size", 2, func(a *config.ServerArgs) { a.AttnCpSize = 2 }},
		{"dcp_size", 2, func(a *config.ServerArgs) { a.DcpSize = 2 }},
		{"enable_prefill_cp", true, func(a *config.ServerArgs) { a.EnablePrefillCp = true }},
		{"cp_strategy", strategy, func(a *config.ServerArgs) { a.CpStrategy = &strategy }},
		{"dcp_comm_backend", "a2a", nil},
	} {
		args := map[string]any{"model_path": "<model>", tc.field: tc.value}
		_, err := config.Resolve(config.Input{Args: args, Chip: c.Chip, Model: c.ModelFacts})
		var nm *config.NotModelled
		if !errors.As(err, &nm) || nm.Name != tc.field || nm.Issue != issueCP {
			t.Errorf("config.Resolve with %s=%v: got %v, want a refusal naming it and citing %s", tc.field, tc.value, err, issueCP)
		}
		if tc.set == nil {
			continue
		}
		a := r.Args
		tc.set(&a)
		_, err = CaptureBatchSizes(&a, 256, 1)
		var nl *NotLowered
		if !errors.As(err, &nl) || nl.Name != tc.field || nl.Issue != issueCP {
			t.Errorf("lowering with %s=%v: got %v, want a refusal naming it and citing %s", tc.field, tc.value, err, issueCP)
		}
	}
}

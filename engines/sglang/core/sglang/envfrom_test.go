// Original to blis-sglang.

package sglang

import (
	"strings"
	"testing"
)

// The variables a pool's env map sets reach the scheduler's environment as environ.py parses them.
func TestEnvFromSetsTheVariablesTheSchedulerReads(t *testing.T) {
	got, err := EnvFrom(DefaultEnv(), map[string]any{
		"SGLANG_INIT_NEW_TOKEN_RATIO":                "0.5",
		"SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR":          0.2,
		"SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS":         "300",
		"SGLANG_RETRACT_DECODE_STEPS":                7,
		"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION":      "100",
		"SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP": "Yes",
		"SGLANG_ENABLE_JIT_DEEPGEMM":                 true,
		"SGLANG_RADIX_FORCE_MISS":                    "no",
	})
	if err != nil {
		t.Fatal(err)
	}
	want := Env{InitNewTokenRatio: 0.5, MinNewTokenRatioFactor: 0.2, NewTokenRatioDecaySteps: 300, RetractDecodeSteps: 7,
		ClipMaxNewTokens: 100, DisableConsecutivePrefillOverlap: true}
	if got != want {
		t.Errorf("got %+v, want %+v", got, want)
	}
	if def, _ := EnvFrom(DefaultEnv(), nil); def != DefaultEnv() {
		t.Errorf("an empty map changes the default environment: %+v", def)
	}
}

func TestEnvFromRefusesWhatItCannotHonourByName(t *testing.T) {
	for name, tc := range map[string]struct {
		v     any
		issue string
	}{
		"SGLANG_ENABLE_JIT_DEEPGEMM":                 {"false", "#9"},
		"SGLANG_RADIX_FORCE_MISS":                    {"true", "#3"},
		"SGLANG_SOMETHING_ELSE":                      {"1", "#9"},
		"SGLANG_RETRACT_DECODE_STEPS":                {"1.5", "integer"},
		"SGLANG_INIT_NEW_TOKEN_RATIO":                {"0x1p-1", "float"},
		"SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR":          {"0X.8", "float"},
		"SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP": {"maybe", "boolean"},
	} {
		_, err := EnvFrom(DefaultEnv(), map[string]any{name: tc.v})
		if err == nil || !strings.Contains(err.Error(), name) || !strings.Contains(err.Error(), tc.issue) {
			t.Errorf("%s=%v: got %v, want a refusal naming it and %q", name, tc.v, err, tc.issue)
		}
	}
}

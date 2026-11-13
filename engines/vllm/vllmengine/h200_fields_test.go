// Original to blis-vllm.

package vllmengine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// The DeepSeek-V4.1-Flash H200 rows of InferenceX state the Rust front end, synthetic rejection sampling
// at a supplied acceptance length and PYTHONUNBUFFERED. testdata/deepseek-v41-flash-h200-tp8-dspark5-
// synthetic-rust.yaml is that recipe at vLLM 0.32.0; these tests build it through the engine factory,
// as a host does, with one setting changed.
func buildRecipe(t *testing.T, swap map[string]string) error {
	t.Helper()
	Register()
	raw, err := os.ReadFile(filepath.Join("testdata", "deepseek-v41-flash-h200-tp8-dspark5-synthetic-rust.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	for old, repl := range swap {
		if !strings.Contains(body, old) {
			t.Fatalf("the recipe has no %q", old)
		}
		body = strings.Replace(body, old, repl, 1)
	}
	path := filepath.Join(t.TempDir(), "v.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: map[string]string{RootScenario: path}})
	return err
}

const (
	envBoth  = `env: {VLLM_USE_RUST_FRONTEND: "1", PYTHONUNBUFFERED: "1"}`
	specFull = `rejection_sample_method: synthetic, synthetic_acceptance_length: 3.51`
)

// The recipe as the run stated it builds; so does each of its three settings alone, and stated the other
// way where it has one.
func TestTheH200RecipeBuildsWithEachOfItsThreeSettings(t *testing.T) {
	for name, swap := range map[string]map[string]string{
		"as recorded":               nil,
		"python front end stated":   {envBoth: `env: {VLLM_USE_RUST_FRONTEND: "0", PYTHONUNBUFFERED: "1"}`},
		"no front end stated":       {envBoth: `env: {PYTHONUNBUFFERED: "1"}`},
		"no container variable":     {envBoth: `env: {VLLM_USE_RUST_FRONTEND: "1"}`},
		"no env block":              {"      " + envBoth + "\n": ""},
		"a standard sampler":        {specFull: `rejection_sample_method: standard`},
		"the length at its limit":   {"synthetic_acceptance_length: 3.51": "synthetic_acceptance_length: 6"},
		"the sampler left unstated": {", " + specFull: ""},
	} {
		if err := buildRecipe(t, swap); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}

// The refusals that remain are by name: synthetic without the length vLLM requires (and a length beyond
// its limit, 1 + num_speculative_tokens), a front end value vLLM does not read as 0 or 1, and a variable
// the engine neither reads nor knows to be inert.
func TestTheH200RecipeIsRefusedWhereVLLMRefusesIt(t *testing.T) {
	for _, c := range []struct {
		name string
		swap map[string]string
		want string
	}{
		{"synthetic without a length", map[string]string{", synthetic_acceptance_length: 3.51": ""}, "rejection_sample_method"},
		{"a length past 1 + drafts", map[string]string{"synthetic_acceptance_length: 3.51": "synthetic_acceptance_length: 6.5"}, "synthetic_acceptance_length"},
		{"a length with another sampler", map[string]string{"rejection_sample_method: synthetic": "rejection_sample_method: block"}, "synthetic"},
		{"a front end vLLM cannot read", map[string]string{envBoth: `env: {VLLM_USE_RUST_FRONTEND: "maybe"}`}, "VLLM_USE_RUST_FRONTEND"},
		{"an unread variable", map[string]string{envBoth: `env: {VLLM_USE_RUST_FRONTEND: "1", VLLM_LOGGING_LEVEL: DEBUG}`}, "VLLM_LOGGING_LEVEL"},
	} {
		err := buildRecipe(t, c.swap)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err %v, want a refusal naming %q", c.name, err, c.want)
		}
	}
}

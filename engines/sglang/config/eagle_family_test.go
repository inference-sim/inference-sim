// Original to blis-sglang.

package config

import (
	"errors"
	"strings"
	"testing"
)

// EAGLE3 and the multi-layer worker are the same resolution as EAGLE (S30, inference-sim/inference-sim#2355): EAGLE3
// is_eagle() and runs _handle_eagle_family (speculative_spec_info.py:104-118, 255-266@297e54fd),
// and enable_multi_layer_eagle selects only the worker. The reference is an EAGLE case SGLang's own
// pipeline resolved (testdata/serverargs.json): the others resolve to what it resolved, field by
// field, except the field they set.
func TestEagleFamilyResolvesAsEagleDoes(t *testing.T) {
	in := eagleReference(t)
	resolve := func(set map[string]any) *Resolved {
		t.Helper()
		got, err := Resolve(in.with(set))
		if err != nil {
			t.Fatalf("%v: %v", set, err)
		}
		return got
	}
	ref := resolve(nil)
	cases := []struct {
		name   string
		set    map[string]any
		differ map[string]any
	}{
		{"EAGLE3", map[string]any{"speculative_algorithm": "EAGLE3"}, map[string]any{"speculative_algorithm": "EAGLE3"}},
		{"NEXTN is EAGLE", map[string]any{"speculative_algorithm": "NEXTN"}, nil},
		{"multi-layer EAGLE", map[string]any{"enable_multi_layer_eagle": true}, map[string]any{"enable_multi_layer_eagle": true}},
		{"multi-layer EAGLE3", map[string]any{"speculative_algorithm": "EAGLE3", "enable_multi_layer_eagle": true},
			map[string]any{"speculative_algorithm": "EAGLE3", "enable_multi_layer_eagle": true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := resolve(tc.set)
			for _, name := range FieldNames() {
				want, _ := ref.Value(name)
				if w, ok := tc.differ[name]; ok {
					want = w
				}
				if v, _ := got.Value(name); canon(v) != canon(want) {
					t.Errorf("%s resolves to %v, want %v", name, v, want)
				}
			}
			if v, _ := got.Value("max_running_requests"); canon(v) != "48" {
				t.Errorf("max_running_requests %v: a speculative run defaults to 48 (_handle_eagle_family)", v)
			}
		})
	}
}

// What the engine does not model is refused by name citing #68: another algorithm, and an EAGLE3
// with a separate draft checkpoint (it resolves speculative_draft_model_revision, which needs
// goldens recorded from SGLang).
func TestEagleFamilyRefusalsCiteTheIssue(t *testing.T) {
	in := eagleReference(t)
	for _, set := range []map[string]any{
		{"speculative_algorithm": "STANDALONE"},
		{"speculative_algorithm": "DFLASH"},
		{"speculative_algorithm": "NGRAM"},
		{"speculative_algorithm": "EAGLE3", "speculative_draft_model_path": "org/eagle3-draft"},
	} {
		_, err := Resolve(in.with(set))
		var nm *NotModelled
		if !errors.As(err, &nm) || !strings.HasSuffix(nm.Issue, "inference-sim/inference-sim#2355") {
			t.Errorf("%v: got %v, want NotModelled citing inference-sim/inference-sim#2355", set, err)
		}
	}
	// EAGLE3 on a model whose checkpoint ships no draft layers needs a separate checkpoint.
	noDraft := in
	noDraft.Model.Architecture = "Qwen2ForCausalLM"
	_, err := Resolve(noDraft.with(map[string]any{"speculative_algorithm": "EAGLE3"}))
	var nm *NotModelled
	if !errors.As(err, &nm) || !strings.HasSuffix(nm.Issue, "inference-sim/inference-sim#2355") {
		t.Errorf("EAGLE3 without a draft in the checkpoint: got %v, want NotModelled citing inference-sim/inference-sim#2355", err)
	}
}

type eagleInput Input

// eagleReference is the input of an oracle case SGLang resolved with EAGLE (glm53-h100-tp8-mtp).
func eagleReference(t *testing.T) eagleInput {
	t.Helper()
	for _, c := range readAnswers(t).Cases {
		if c.Name == "glm53-h100-tp8-mtp" && c.Refused == nil {
			return eagleInput{Args: c.Args, Chip: c.Chip, Model: c.ModelFacts}
		}
	}
	t.Fatal("the EAGLE reference case is missing from testdata/serverargs.json")
	return eagleInput{}
}

// with is the input with set applied over its arguments, leaving the reference untouched.
func (e eagleInput) with(set map[string]any) Input {
	args := map[string]any{"model_path": "<model>"}
	for k, v := range e.Args {
		args[k] = v
	}
	for k, v := range set {
		args[k] = v
	}
	return Input{Args: args, Chip: e.Chip, Model: e.Model}
}

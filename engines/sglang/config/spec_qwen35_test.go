package config

import "testing"

// Qwen3.5's NextN draft layer ships in the target checkpoint but the architecture is not
// in _handle_eagle_family's list that defaults the draft path (speculative_hook.py:1103-1136@297e54fd), so
// the path stays None; a path-less NEXTN/MTP draft is the target checkpoint read in draft mode
// (model_executor/model_runner_components/spec_aux_hidden_state.py:146-176). InferenceX launches the
// Qwen3.5 recipes that way (--speculative-algorithm NEXTN with no draft path). Resolve resolves it.
func TestQwen35NextnWithoutADraftPathResolves(t *testing.T) {
	facts := qwen35Facts()
	vh := 1152
	facts.VisionHiddenSize = &vh
	in := Input{
		Args: map[string]any{"model_path": "<model>", "speculative_algorithm": "NEXTN", "speculative_num_steps": 3,
			"speculative_eagle_topk": 1, "speculative_num_draft_tokens": 4, "disable_radix_cache": true, "mem_fraction_static": 0.8},
		Chip:  Chip{Capability: [2]int{10, 0}, MemoryMiB: 183359},
		Model: facts,
	}
	res, err := Resolve(in)
	if err != nil {
		t.Fatal(err)
	}
	a := res.Args
	if a.SpeculativeAlgorithm == nil || *a.SpeculativeAlgorithm != "EAGLE" {
		t.Errorf("speculative_algorithm %v, want EAGLE (NEXTN's alias)", a.SpeculativeAlgorithm)
	}
	if a.MaxRunningRequests == nil || *a.MaxRunningRequests != 48 {
		t.Errorf("max_running_requests %v, want 48 (_handle_eagle_family's default when none is stated)", a.MaxRunningRequests)
	}
	if a.SpeculativeDraftModelPath != nil {
		t.Errorf("the draft path is %q, want none: Qwen3.5 is not in the list that defaults it", *a.SpeculativeDraftModelPath)
	}
}

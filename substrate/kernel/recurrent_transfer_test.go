package latencykernel

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/load"
)

// A P/D handoff carries each recurrent layer's state once, whatever the prompt's length: NIXL
// clips a Mamba group to its one state block (clip_ssm_state_blocks) and moves that block's
// conv and temporal regions. The payload is the attention caches' bytes, which grow with the
// prompt, plus the model's recurrent state, which does not; a model with no recurrent layer
// carries none.
func TestHandoffCarriesTheRecurrentStateOnceWhateverThePromptLength(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		model, chip, parallel string
		recurrent             bool
	}{
		{"nemotron-3.5-lightning-30b-a3b-bf16", "h100", "tp: 2, pp: 1, dp: 1", true},
		{"nemotron-3.5-lightning-30b-a3b-bf16", "h100", "tp: 1, pp: 1, dp: 1", true},
		{"llama-3.1-70b-instruct", "h100", "tp: 4, pp: 1, dp: 1", false},
	} {
		in := planCase(t, d, c.model, c.chip, 1, c.parallel, nil)
		k, err := New(in)
		if err != nil {
			t.Fatal(err)
		}
		var want float64
		for _, l := range k.plan.Layers {
			if l.RecurrentStateBytes == 0 {
				continue
			}
			shards := k.tp
			if h := float64(l.RecurrentHeads); h > 0 && h < shards {
				shards = h
			}
			want += l.RecurrentStateBytes / shards * float64(l.Count)
		}
		if c.recurrent != (want > 0) {
			t.Fatalf("%s: the fixture's recurrent state is %v, want recurrent=%v", c.model, want, c.recurrent)
		}
		for _, tokens := range []int{1, 100, 5000} {
			attention := float64(tokens)*k.kvBytesPerToken + k.ringTransferBytes(tokens) + k.cacheTransferBytes(tokens)
			if got := k.pdTransferBytes(tokens) - attention; math.Abs(got-want) > 1e-6*math.Max(want, 1) {
				t.Errorf("%s %s: a %d-token handoff carries %.0f bytes beside the attention caches, want the recurrent state %.0f",
					c.model, c.parallel, tokens, got, want)
			}
		}
	}
}

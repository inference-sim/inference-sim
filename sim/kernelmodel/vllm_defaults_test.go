package kernelmodel

import "testing"

// Every chip the accuracy corpus scores must resolve to the values vLLM resolves. These are
// not round numbers chosen here; they are what vllm/engine/arg_utils.py returns, and a
// scenario that states nothing is simulated with them.
func TestEveryCorpusChipResolvesAsVLLMDoes(t *testing.T) {
	for _, c := range []struct {
		chip      string
		memoryGiB float64
		wantSeqs  int
		wantToks  int
	}{
		// Blackwell: at or above the 160 GiB threshold.
		{"b200", 180, 1024, 16384},
		{"b300", 288, 1024, 16384},
		// Hopper: above 70 GiB, not an A100. The served context takes 8192 tokens.
		{"h100", 80, 1024, 8192},
		{"h200", 141, 1024, 8192},
		// Excluded by NAME rather than by memory: an 80 GiB A100 clears the threshold and
		// still takes the small defaults, because large token counts measured worse on it.
		{"a100-sxm", 80, 256, 2048},
		{"a100-80", 80, 256, 2048},
		// Below both thresholds.
		{"l40s", 48, 256, 2048},
	} {
		got := ResolveVLLMBatchDefaults(c.memoryGiB, c.chip)
		if got.MaxNumSeqs != c.wantSeqs || got.MaxNumBatchedTokens != c.wantToks {
			t.Errorf("%s (%.0f GiB): got seqs=%d tokens=%d, want seqs=%d tokens=%d",
				c.chip, c.memoryGiB, got.MaxNumSeqs, got.MaxNumBatchedTokens,
				c.wantSeqs, c.wantToks)
		}
	}
}

// The scenario files this project ships had assumed a fixed 256 sequences everywhere. On
// every chip in the corpus that is four times too small, which is what made a simulated
// request wait where the engine admitted it immediately.
func TestTheAssumedDefaultWasWrongOnEveryCorpusChip(t *testing.T) {
	const assumed = 256
	for _, c := range []struct {
		chip      string
		memoryGiB float64
	}{{"h100", 80}, {"h200", 141}, {"b200", 180}, {"b300", 288}} {
		if got := ResolveVLLMBatchDefaults(c.memoryGiB, c.chip); got.MaxNumSeqs == assumed {
			t.Errorf("%s resolves to %d sequences, the same as the value this project had "+
				"assumed; the test fixture no longer demonstrates anything",
				c.chip, got.MaxNumSeqs)
		}
	}
}

// Parallel width must not move the answer. world_size is a parameter of vLLM's
// get_batch_defaults and is never read in its body, so a resolver that scaled by tp or pp
// would diverge from the engine on every multi-GPU deployment.
func TestResolutionIgnoresParallelWidth(t *testing.T) {
	// The function takes no width, so this asserts the API shape rather than a value: if a
	// future change adds one, this test is where the question gets asked.
	base := ResolveVLLMBatchDefaults(141, "h200")
	if base.MaxNumSeqs != 1024 {
		t.Fatalf("h200 should resolve to 1024 sequences, got %d", base.MaxNumSeqs)
	}
	// Same device, repeated: the resolution is a pure function of (memory, name).
	for i := 0; i < 4; i++ {
		if got := ResolveVLLMBatchDefaults(141, "H200"); got != base {
			t.Errorf("resolution is not case-insensitive or not pure: %+v vs %+v", got, base)
		}
	}
}

// The threshold is a boundary, and both sides of it must be reachable. A resolver that used
// > where vLLM uses >= would send a 160 GiB part to the Hopper branch.
func TestThresholdsAreInclusive(t *testing.T) {
	if got := ResolveVLLMBatchDefaults(160, "b200"); got.MaxNumBatchedTokens != 16384 {
		t.Errorf("exactly 160 GiB must take the large branch, got %d tokens",
			got.MaxNumBatchedTokens)
	}
	if got := ResolveVLLMBatchDefaults(159.9, "someblackwell"); got.MaxNumBatchedTokens != 8192 {
		t.Errorf("just below 160 GiB must fall to the medium branch, got %d tokens",
			got.MaxNumBatchedTokens)
	}
	if got := ResolveVLLMBatchDefaults(70, "h100"); got.MaxNumSeqs != 1024 {
		t.Errorf("exactly 70 GiB must take the medium branch, got %d sequences",
			got.MaxNumSeqs)
	}
	if got := ResolveVLLMBatchDefaults(69.9, "h100"); got.MaxNumSeqs != 256 {
		t.Errorf("just below 70 GiB must fall to the small branch, got %d sequences",
			got.MaxNumSeqs)
	}
}

// An unnamed device cannot be an A100, so it must take the memory branch rather than the
// conservative fallback.
func TestUnnamedDeviceTakesTheMemoryBranch(t *testing.T) {
	if got := ResolveVLLMBatchDefaults(141, ""); got.MaxNumSeqs != 1024 {
		t.Errorf("an unnamed 141 GiB device should resolve to 1024 sequences, got %d",
			got.MaxNumSeqs)
	}
}

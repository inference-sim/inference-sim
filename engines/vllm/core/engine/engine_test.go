package engine

import "testing"

// max_concurrent_batches for every configuration vLLM v0.32.0 starts with
// (vllm/config/vllm.py:577-588@5df63a72).
func TestMaxConcurrentBatches(t *testing.T) {
	for _, c := range []struct {
		pp        int
		async, v2 bool
		want      int
	}{
		{1, false, false, 1}, {1, false, true, 1}, {2, false, false, 2}, {4, false, true, 4},
		{1, true, false, 2}, {1, true, true, 2}, {2, true, true, 3}, {4, true, true, 5},
	} {
		if got := MaxConcurrentBatches(c.pp, c.async, c.v2); got != c.want {
			t.Errorf("pp %d async %v v2 %v: %d, want %d", c.pp, c.async, c.v2, got, c.want)
		}
	}
}

// The V1 model runner refuses async scheduling with PP; everything else starts.
func TestCheckConfigRefusesWhatVLLMRefuses(t *testing.T) {
	if err := CheckConfig(2, true, false); err == nil {
		t.Error("PP 2 with async scheduling on the V1 runner was accepted")
	}
	for _, c := range []struct {
		pp        int
		async, v2 bool
	}{{1, true, false}, {2, true, true}, {2, false, false}, {1, false, false}} {
		if err := CheckConfig(c.pp, c.async, c.v2); err != nil {
			t.Errorf("pp %d async %v v2 %v refused: %v", c.pp, c.async, c.v2, err)
		}
	}
}

// Original to blis-vllm.

package vllm

import (
	"strings"
	"testing"
)

// vLLM v0.33.0 applies prefill_schedule_interval on an engine without data parallelism, as the
// scheduler's local prefill cadence (vllm/v1/core/sched/scheduler.py:395-425, 770-790@4ba787ec), so
// the setting is refused there as not modelled, citing its issue; a value below 1 is refused as
// vLLM refuses it (vllm/config/scheduler.py:203@4ba787ec); and in a data-parallel group, where the
// DP engine core throttles, it runs.
func TestPrefillScheduleIntervalWithoutDataParallelismIsTheUnportedLocalCadence(t *testing.T) {
	for _, c := range []struct {
		name, block string
		want        []string
	}{
		{"local cadence", "prefill_schedule_interval: 2\n", []string{"local prefill cadence", "not modelled", "issues/172"}},
		{"below vLLM's minimum", "prefill_schedule_interval: -1\n", []string{"below 1", "vllm/config/scheduler.py:203"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := build(t, c.block)()
			if err == nil {
				t.Fatal("built, want refused")
			}
			for _, w := range c.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("refusal %q does not say %q", err, w)
				}
			}
		})
	}
	if _, err := build(t, "data_parallel_size: 2\nprefill_schedule_interval: 2\n")(); err != nil {
		t.Errorf("a data-parallel group with prefill_schedule_interval 2 was refused: %v", err)
	}
}

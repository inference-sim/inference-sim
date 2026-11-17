// Original to blis-vllm.

package vllm

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
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

// vLLM v0.33.0 checks a drafter against async scheduling only on the V1 model runner
// (vllm/config/vllm.py:1651-1719@4ba787ec). On V1, a stated async_scheduling with ngram is refused
// with vLLM's reason and an unstated one runs synchronously; on V2, where vLLM runs ngram with async
// scheduling (resolved on when unstated), that combination is refused as not modelled, citing its
// issue, and stating async_scheduling: false runs.
func TestAsyncSchedulingWithNgramFollowsTheModelRunner(t *testing.T) {
	const ngram = "speculative_config: {method: ngram, num_speculative_tokens: 3}\n"
	for _, c := range []struct {
		name, block string
		refuse      []string // empty: builds
	}{
		{"V1 stated", "use_v2_model_runner: false\nasync_scheduling: true\n" + ngram, []string{"not supported by the V1 model runner", "ngram"}},
		{"V1 unstated", "use_v2_model_runner: false\n" + ngram, nil},
		{"V2 stated", "use_v2_model_runner: true\nasync_scheduling: true\n" + ngram, []string{"V2 model runner is not modelled", "issues/186"}},
		{"V2 unstated", "use_v2_model_runner: true\n" + ngram, []string{"V2 model runner is not modelled", "issues/186"}},
		{"V2 stated off", "use_v2_model_runner: true\nasync_scheduling: false\n" + ngram, nil},
	} {
		t.Run(c.name, func(t *testing.T) {
			// The whole block, so that async_scheduling can be left unstated (base states it).
			src := "block_size: 16\nmax_num_seqs: 4\nmax_num_batched_tokens: 64\nmax_model_len: 512\n" +
				"num_gpu_blocks: 64\npipeline_parallel_size: 1\n" + c.block
			var n yaml.Node
			if err := yaml.Unmarshal([]byte(src), &n); err != nil {
				t.Fatal(err)
			}
			cfg, err := decodeConfig(*n.Content[0])
			if err == nil {
				_, err = resolve(cfg)
			}
			if len(c.refuse) == 0 {
				if err != nil {
					t.Fatalf("refused: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatal("built, want refused")
			}
			for _, w := range c.refuse {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("refusal %q does not say %q", err, w)
				}
			}
		})
	}
}

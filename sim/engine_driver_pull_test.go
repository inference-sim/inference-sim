package sim

import (
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-vllm/core/vllm"
)

// decoder is blis-vllm's v0.32.0 engine as a NIXL kv_consumer under policy, at a fixed step price.
func decoder(t *testing.T, policy string) *vllm.Engine {
	t.Helper()
	src := "block_size: 16\nmax_num_seqs: 4\nmax_num_batched_tokens: 64\nmax_model_len: 512\nnum_gpu_blocks: 64\n" +
		"use_v2_model_runner: false\nasync_scheduling: false\npipeline_parallel_size: 1\n" +
		"kv_transfer: {connector: NixlConnector, kv_role: kv_consumer, kv_load_failure_policy: " + policy + "}\n"
	var n yaml.Node
	if err := yaml.Unmarshal([]byte(src), &n); err != nil {
		t.Fatal(err)
	}
	f := vllm.Factory{Pricer: func(engine.Spec) (vllm.Pricer, error) { return vllm.FixedPricer(100), nil }}
	eng, err := f.New(engine.Spec{Kind: vllm.Kind, UpstreamVersion: vllm.Version, Config: *n.Content[0]})
	if err != nil {
		t.Fatal(err)
	}
	return eng.(*vllm.Engine)
}

// A handoff the host ends before its pull lands leaves the decode engine holding nothing, under
// either kv_load_failure_policy, and runs nothing for it:
//   - evicted mid-pull (AbortRemote): vLLM's finish_requests delays the free of a request whose
//     load is in flight until the worker reports the load over in finished_recving
//     (vllm/v1/core/sched/scheduler.py:2690-2700, 3226-3231@5df63a72), so the host must end the
//     pull it gives up on, or the blocks stay held for the run;
//   - failed by the host (FailRemote, the decode instance gone unroutable): the cluster drops the
//     request, so the engine must not keep it, which under "recompute" it would run as a local
//     prefill (scheduler.py:3132-3153, 3341-3354@5df63a72).
func TestEngineDriver_AHandoffTheHostEndsMidPullHoldsNothing(t *testing.T) {
	for _, policy := range []string{"fail", "recompute"} {
		for _, how := range []string{"abort", "fail"} {
			eng := decoder(t, policy)
			d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40})
			started := 0
			d.OnPullStarted = func(*Request, int64, int64) { started++ }
			req := &Request{ID: "a", InputTokens: tokens(80), OutputTokens: tokens(6), State: StateWaitingForRemoteKVs}
			d.AddRemote(req, 0)
			if started != 1 {
				t.Fatalf("%s/%s: %d pulls started, want 1", policy, how, started)
			}
			if how == "abort" {
				d.AbortRemote(req, 50)
			} else {
				d.FailRemote(req, 50)
			}
			d.Run()
			s := eng.Snapshot()
			if s.KVBlocksUsed != 0 || s.Waiting != 0 || s.Running != 0 || d.stepIdx != 0 {
				t.Errorf("%s/%s: engine holds %d blocks, %d waiting, %d running after %d steps; want none",
					policy, how, s.KVBlocksUsed, s.Waiting, s.Running, d.stepIdx)
			}
		}
	}
}

// Original to this repository: H1 of the DP external load-balancing RFC (inference-sim#2699).

package dprankview_test

import (
	"testing"

	vllm "github.com/inference-sim/inference-sim/engines/vllm/core/vllm"
	"github.com/inference-sim/inference-sim/engines/vllm/core/vllm/dprankview"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"gopkg.in/yaml.v3"
)

// externalGroup builds a 4-rank vLLM data-parallel group under external load balancing through the
// public Factory, so this external test does not reach the vllm package's internal helpers.
func externalGroup(t *testing.T) *vllm.DPGroup {
	t.Helper()
	cfg := "block_size: 4\nmax_num_seqs: 4\nmax_num_batched_tokens: 32\nmax_model_len: 256\nnum_gpu_blocks: 80\n" +
		"use_v2_model_runner: false\nasync_scheduling: false\npipeline_parallel_size: 1\ndata_parallel_size: 4\n" +
		"enable_prefix_caching: true\ndata_parallel_external_lb: true\n"
	var n yaml.Node
	if err := yaml.Unmarshal([]byte(cfg), &n); err != nil {
		t.Fatal(err)
	}
	s := engine.Spec{Kind: vllm.Kind, UpstreamVersion: vllm.Version, Config: *n.Content[0]}
	e, err := vllm.Factory{Pricer: func(engine.Spec) (vllm.Pricer, error) { return vllm.FixedPricer(100), nil }}.New(s)
	if err != nil {
		t.Fatal(err)
	}
	g, ok := e.(*vllm.DPGroup)
	if !ok {
		t.Fatalf("engine is %T, want *vllm.DPGroup", e)
	}
	return g
}

func mkreq(id string, prompt int) engine.Request {
	r := engine.Request{ID: id, Prompt: make([]int32, prompt), Output: make([]int32, 2)}
	for i := range r.Prompt {
		r.Prompt[i] = int32(1000 + i)
	}
	for i := range r.Output {
		r.Output[i] = int32(9000 + i)
	}
	return r
}

// NewEndpointViews gives one engine per endpoint; Add routes to that endpoint's rank; only the
// leader (rank 0) steps the group, a member view schedules an empty batch; and each view's Snapshot
// reports only its own rank. This is the per-endpoint engine the host registers for external
// balancing.
func TestEndpointViewsArePerRankAndOnlyTheLeaderSteps(t *testing.T) {
	g := externalGroup(t)
	views := dprankview.NewEndpointViews(g)
	if len(views) != 4 {
		t.Fatalf("got %d views, want 4 (one per endpoint)", len(views))
	}

	// Work on endpoint 2 only.
	const held = 2
	if err := views[held].Add(mkreq("only-on-2", 40), 0); err != nil {
		t.Fatal(err)
	}

	// A member view schedules nothing of its own: the leader steps the group.
	if b, err := views[1].Schedule(0); err != nil || !b.Empty() {
		t.Errorf("member view (endpoint 1) scheduled %d reqs (err %v); a member must return an empty batch", len(b.Reqs), err)
	}

	// The leader view (endpoint 0) steps the whole group, so the request on rank 2 runs.
	b, err := views[0].Schedule(0)
	if err != nil {
		t.Fatal(err)
	}
	if b.Empty() {
		t.Fatal("the leader view scheduled nothing though rank 2 has work; the leader must step the group")
	}

	// Each view reports only its own rank: rank 2 has the request, the others are idle.
	for e, v := range views {
		st := v.(interface{ Snapshot() engine.State }).Snapshot()
		busy := st.Running + st.Waiting
		if e == held && busy == 0 {
			t.Errorf("endpoint %d holds the request but its snapshot is idle (running %d waiting %d)", e, st.Running, st.Waiting)
		}
		if e != held && busy != 0 {
			t.Errorf("endpoint %d served no request but its snapshot shows running %d waiting %d; a view must report only its own rank", e, st.Running, st.Waiting)
		}
	}
}

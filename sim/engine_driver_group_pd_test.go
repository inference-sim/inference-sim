// Original to this repository: H2/H3 of the DP external load-balancing RFC (inference-sim#2699).
// The P/D decode-completion path for a shared-group view-driver: a remote-prefilled decode
// sub-request routed to one rank of an external data-parallel group must run and complete through
// the leader/member step split, as it does on a single collapsed decode instance. This is the path
// mtoslalibu's review requires for the P/D proof; it is exercised here at the driver level with a
// light engine, standing in for the cluster's pull-completion (LandRemote) and event loop.

package sim

import (
	"fmt"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/engines/vllm/core/vllm"
	"github.com/inference-sim/inference-sim/engines/vllm/core/vllm/dprankview"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"

	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// externalDecodeDPGroup builds a dp-rank external-balancing vLLM group as a NIXL kv_consumer (the
// decode side of a P/D pair), at a fixed step price.
func externalDecodeDPGroup(t *testing.T, dp int) *vllm.DPGroup {
	t.Helper()
	src := fmt.Sprintf("block_size: 16\nmax_num_seqs: 8\nmax_num_batched_tokens: 256\nmax_model_len: 512\n"+
		"num_gpu_blocks: 400\nuse_v2_model_runner: false\nasync_scheduling: false\npipeline_parallel_size: 1\n"+
		"enable_prefix_caching: true\ndata_parallel_size: %d\ndata_parallel_external_lb: true\n"+
		"placement: {hardware: H100, tensor: 1, ranks_per_node: 8, nodes: 1}\n"+
		"kv_transfer: {connector: NixlConnector, kv_role: kv_consumer, kv_load_failure_policy: fail}\n", dp)
	var n yaml.Node
	if err := yaml.Unmarshal([]byte(src), &n); err != nil {
		t.Fatal(err)
	}
	f := vllm.Factory{Pricer: func(engine.Spec) (vllm.Pricer, error) { return vllm.FixedPricer(100), nil }}
	eng, err := f.New(engine.Spec{Kind: vllm.Kind, UpstreamVersion: vllm.Version, Config: *n.Content[0]})
	if err != nil {
		t.Fatalf("building the external decode dp=%d group: %v", dp, err)
	}
	g, ok := eng.(*vllm.DPGroup)
	if !ok {
		t.Fatalf("built %T, want a *DPGroup", eng)
	}
	return g
}

// TestExternalDP_RemotePrefilledDecodeCompletes pins the P/D decode path (inference-sim#2699): a
// decode sub-request whose prompt a peer prefilled, routed to one rank of an external group, runs
// and completes through the leader/member step split. It fails if the request runs on its rank but
// never finishes (the stall the review found on the shared-group view driver).
func TestExternalDP_RemotePrefilledDecodeCompletes(t *testing.T) {
	const dp = 4
	const rank = 2
	g := externalDecodeDPGroup(t, dp)
	views := dprankview.NewEndpointViews(g)
	env := testutil.FixedPull(16, func(tokens int64, _, _ plan.Endpoint) int64 { return 100 })
	drivers := make([]*EngineDriver, dp)
	for k, v := range views {
		drivers[k] = NewEngineDriver(v, EngineDriverConfig{Horizon: 1 << 40, Environment: env})
	}
	for k := range drivers {
		drivers[k].JoinGroup(drivers[0], k)
	}

	var pulled *Request
	drivers[rank].OnPullPriced = func(r *Request, _, _ int64, _ kernel.TransferDemand, _ int64) { pulled = r }

	pump := func(stop func() bool) {
		for range 2_000_000 {
			if stop != nil && stop() {
				return
			}
			best := -1
			var bt int64
			for j, d := range drivers {
				if d.HasPendingEvents() {
					if tt := d.PeekNextEventTime(); best == -1 || tt < bt {
						best, bt = j, tt
					}
				}
			}
			if best == -1 {
				return
			}
			drivers[best].ProcessNextEvent()
		}
		t.Fatal("pump did not terminate")
	}

	req := &Request{ID: "d0", InputTokens: tokens(80), OutputTokens: tokens(5), State: StateWaitingForRemoteKVs, IsDecodeSubRequest: true}
	drivers[rank].AddRemote(req, plan.Endpoint{Node: 0}, 0)

	// Let the group step so rank 2's schedule starts the pull, then land it.
	pump(func() bool { return pulled != nil })
	if pulled == nil {
		t.Fatal("the decode rank never started its remote-KV pull")
	}
	drivers[rank].LandRemote(pulled, 1000)

	// Drive to completion.
	pump(nil)

	if req.State != StateCompleted {
		t.Fatalf("remote-prefilled decode on rank %d ended in state %q with %d generated tokens; it must complete (it runs on its rank but never finishes through the leader/member split)",
			rank, req.State, req.GeneratedTokens)
	}
	if req.GeneratedTokens != len(req.OutputTokens) {
		t.Errorf("decode generated %d tokens, want %d", req.GeneratedTokens, len(req.OutputTokens))
	}
}

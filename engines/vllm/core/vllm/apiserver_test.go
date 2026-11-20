// Original to blis-vllm.

package vllm

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

const dp4 = "block_size: 4\nmax_num_seqs: 4\nmax_num_batched_tokens: 32\nmax_model_len: 256\nnum_gpu_blocks: 80\n" +
	"use_v2_model_runner: false\nasync_scheduling: false\npipeline_parallel_size: 1\ndata_parallel_size: 4\n"

func ranksOf(t *testing.T, extra string, setFrontend *Frontend) []int {
	t.Helper()
	e, err := build(t, dp4+extra)()
	if err != nil {
		t.Fatal(err)
	}
	g := e.(*DPGroup)
	if setFrontend != nil {
		if err := g.SetFrontend(*setFrontend); err != nil {
			t.Fatal(err)
		}
	}
	var out []int
	for _, id := range []string{"a", "b", "c", "d"} {
		if err := g.Add(req(id, 12, 4), engine.Time(0)); err != nil {
			t.Fatal(err)
		}
		out = append(out, g.rankOf[id])
	}
	return out
}

// api_server_count in the engine block is the number of balancing clients: two clients start
// their scans at ranks 0 and 2, so four idle-rank arrivals go 0, 2, 1, 3 where one client sends
// them 0, 1, 2, 3.
func TestAPIServerCountSetsTheBalancingClients(t *testing.T) {
	if got, want := ranksOf(t, "api_server_count: 2\n", nil), []int{0, 2, 1, 3}; !reflect.DeepEqual(got, want) {
		t.Errorf("api_server_count 2 routed %v, want %v", got, want)
	}
	if got, want := ranksOf(t, "", nil), []int{0, 1, 2, 3}; !reflect.DeepEqual(got, want) {
		t.Errorf("an unstated api_server_count routed %v, want %v", got, want)
	}
}

// The host that states the front end's settings through SetFrontend sets the same clients.
func TestTheHostsFrontendStatesTheBalancingClients(t *testing.T) {
	f := DefaultFrontend()
	f.APIServerCount = 2
	if got, want := ranksOf(t, "", &f), []int{0, 2, 1, 3}; !reflect.DeepEqual(got, want) {
		t.Errorf("a host stating 2 API servers routed %v, want %v", got, want)
	}
}

func TestANegativeAPIServerCountIsRefused(t *testing.T) {
	_, err := build(t, dp4+"api_server_count: -1\n")()
	if err == nil || !strings.Contains(err.Error(), "api_server_count") {
		t.Errorf("api_server_count -1 gave %v, want a refusal naming it", err)
	}
}

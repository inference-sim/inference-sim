package cmd

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
	"github.com/inference-sim/inference-sim/substrate/contracts/units"
)

// A p2p WRITE is priced from the bytes the engine states, on the links between the two nodes it
// names, and not as the decode pull of the GPU handoff (blis-vllm#204).

func writeOf(tokens int64, from, to plan.Endpoint) engine.Transfer {
	return engine.Transfer{ID: 1, Request: "r", Kind: engine.TransferStore, Tokens: int(tokens),
		Move: plan.Transfer{Tokens: units.Tokens(tokens), From: from, To: to, Kind: plan.TransferPeer}}
}

func wideNodes() plan.PoolPlan {
	return plan.PoolPlan{Hardware: "chip", Layout: plan.Layout{Tensor: 1, RanksPerNode: 1, Nodes: 1 << 10}}
}

// offloadDPPools opens the data-parallel P/D fixture with the offload-tier rates listed, as a run
// that offloads lists them (a tier's rates are registry entries).
func offloadDPPools(t *testing.T) (*kernelPools, string) {
	t.Helper()
	dir, name := dpPoolScenarios(t, 4, 4)
	path := filepath.Join(dir, name)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	re := regexp.MustCompile(`(?m)^(coefficients: \[[^\]]*)\]`)
	if !re.Match(raw) {
		t.Fatal("the fixture states no coefficients list")
	}
	if err := os.WriteFile(path, re.ReplaceAll(raw, []byte("${1}, cost-model-offload-tiers]")), 0o644); err != nil {
		t.Fatal(err)
	}
	pools, _ := openDPPools(t, dir, 2, 2)
	return pools, dir
}

func linksOf(d kernel.TransferDemand) string {
	var s []string
	for _, l := range d.Links {
		s = append(s, string(l.Link))
	}
	return strings.Join(s, " ")
}

func TestPeerWriteDemand_AcrossNodesNamesTheNICsAndCarriesTheStatedBytes(t *testing.T) {
	pools, _ := offloadDPPools(t)
	e, err := newTransferEnvironment(pools.decode, pools.prefill) // a decode engine writing to the prefill pool
	if err != nil {
		t.Fatal(err)
	}
	const bytes = 3*4096 + 17
	d, err := e.PeerWriteDemand(wideNodes(), writeOf(96, plan.Endpoint{Node: 0}, plan.Endpoint{Node: 1}), bytes)
	if err != nil {
		t.Fatal(err)
	}
	if d.Bytes != bytes {
		t.Errorf("the WRITE carries %d bytes, the engine stated %d", d.Bytes, bytes)
	}
	if len(d.Links) == 0 || strings.Contains(linksOf(d), "host/") {
		t.Errorf("a cross-node WRITE names links %q, want the nodes' NICs", linksOf(d))
	}
	from, to := placement(e.own, plan.Endpoint{Node: 0}), placement(e.peer, plan.Endpoint{Node: 1})
	want := e.own.Kernel().PDTransferDemand(int(e.blocksOf(96)*e.blockSize), from, to)
	if linksOf(d) != linksOf(want) || d.Floor != want.Floor {
		t.Errorf("links %q floor %v, the kernel's pair demand is %q floor %v", linksOf(d), d.Floor, linksOf(want), want.Floor)
	}
}

func TestPeerWriteDemand_WithinANodeIsOneHostLinkAndNoNIC(t *testing.T) {
	pools, _ := offloadDPPools(t)
	e, err := newTransferEnvironment(pools.prefill, pools.prefill)
	if err != nil {
		t.Fatal(err)
	}
	e.across = pools.decode
	// Decode-pool node 0 is the cluster's node 4, where the prefill engine's node 4 does not exist as an
	// endpoint: pick the prefill endpoint whose node equals decode endpoint 0's global node.
	dnode := poolOrigin(pools.decode).Node
	from := plan.Endpoint{Node: dnode - poolOrigin(pools.prefill).Node}
	d, err := e.PeerWriteDemand(wideNodes(), writeOf(96, from, plan.Endpoint{Node: 0}), 4096)
	if err != nil {
		t.Fatal(err)
	}
	if want := "host/node/" + itoa(dnode); linksOf(d) != want {
		t.Errorf("a same-node WRITE names links %q, want %q", linksOf(d), want)
	}
	if d.Bytes != 4096 || d.Links[0].BytesPerSecond <= 0 || d.AloneTime() <= 0 {
		t.Errorf("same-node demand %+v", d)
	}
}

func TestPeerWriteDemand_AColocatedWriteIsPricedNotRefused(t *testing.T) {
	m := openScenarioModel(t, "../testdata/kernel/aisimulate", "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml")
	e, err := newTransferEnvironment(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	same, err := e.PeerWriteDemand(wideNodes(), writeOf(64, plan.Endpoint{Node: 0}, plan.Endpoint{Node: 0}), 8192)
	if err != nil {
		t.Fatalf("a colocated same-node WRITE: %v", err)
	}
	if !strings.HasPrefix(linksOf(same), "host/node/") {
		t.Errorf("same-node links %q", linksOf(same))
	}
	cross, err := e.PeerWriteDemand(wideNodes(), writeOf(64, plan.Endpoint{Node: 0}, plan.Endpoint{Node: 1}), 8192)
	if err != nil {
		t.Fatalf("a colocated cross-node WRITE: %v", err)
	}
	if cross.Bytes != 8192 || strings.Contains(linksOf(cross), "host/") || len(cross.Links) == 0 {
		t.Errorf("cross-node demand %+v", cross)
	}
}

// A disaggregated WRITE's From is read in the writer's pool and To in the other pool: the prefill
// pool's endpoint 0 and the decode pool's endpoint 0 are different global nodes, so reading To in
// the writer's pool would name one node for both ends and price a host copy.
func TestPeerWriteDemand_ADisaggregatedWriteReadsToInTheOtherPool(t *testing.T) {
	pools, _ := offloadDPPools(t)
	e, err := newTransferEnvironment(pools.prefill, pools.prefill)
	if err != nil {
		t.Fatal(err)
	}
	e.across = pools.decode
	if poolOrigin(pools.prefill).Node == poolOrigin(pools.decode).Node {
		t.Fatal("fixture: the two pools start at one node")
	}
	d, err := e.PeerWriteDemand(wideNodes(), writeOf(96, plan.Endpoint{Node: 0}, plan.Endpoint{Node: 0}), 4096)
	if err != nil {
		t.Fatal(err)
	}
	from, to := placement(pools.prefill, plan.Endpoint{}), placement(pools.decode, plan.Endpoint{})
	if from.Node == to.Node {
		t.Fatal("fixture: the ends are one node")
	}
	want := pools.prefill.Kernel().PDTransferDemand(int(e.blocksOf(96)*e.blockSize), from, to)
	if strings.Contains(linksOf(d), "host/") || linksOf(d) != linksOf(want) {
		t.Errorf("links %q, want the pair's %q", linksOf(d), linksOf(want))
	}
}

func TestPeerWriteDemand_RefusesWhatItCannotPrice(t *testing.T) {
	m := openScenarioModel(t, "../testdata/kernel/aisimulate", "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml")
	e, err := newTransferEnvironment(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := e.PeerWriteDemand(wideNodes(), writeOf(64, plan.Endpoint{}, plan.Endpoint{Node: 1}), 0); err == nil {
		t.Error("a WRITE of zero bytes was priced")
	}
	if _, err := e.PeerWriteDemand(wideNodes(), writeOf(64, plan.Endpoint{Node: -1}, plan.Endpoint{Node: 1}), 8); err == nil {
		t.Error("a WRITE from an endpoint that does not resolve was priced")
	}
}

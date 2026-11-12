package cmd

import (
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/env"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/units"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"pgregory.net/rapid"
)

// P/D disaggregation with data parallelism. The fixture
// (testdata/scenarios/minimax-m2.5-gb200-1p1d-dp2.yaml) states a routed MoE at dp 2 in both
// pools. The v0.32.0 engine does not build a KV transfer from a data-parallel group yet
// (blis-vllm#44), so a run of it is refused; the handoff pricer between the pools' placements
// is checked without a run.

const pdDPScenario = "minimax-m2.5-gb200-1p1d-dp2.yaml"

// After DP expansion, a pull between replica i of the prefill pool and replica j of the decode
// pool, under the cluster's own instance numbering, is priced by the host's Environment as the
// prefill kernel's PDTransferTicks between rank i's and rank j's placements: the Environment is
// handed the endpoints the cluster states for the prefill instance (instanceEndpoint) and the
// decode engine states for itself, and nothing else. The ranks are derived independently of
// rankInPool: a replica's rank is its position among its pool's instances in numeric order.
// Non-vacuity: the fixture's rack boundary prices the two decode ranks differently, so an
// Environment that ignored or confused the decode rank would break the law.
func TestTransferEnvironment_APullIsThePlacedPairsPrice(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	saved := []any{kernelScenario, kernelScenarioDir, registryPath, kernelOpened,
		prefillInstances, decodeInstances, numInstances, blockSizeTokens}
	defer func() {
		kernelScenario, kernelScenarioDir, registryPath = saved[0].(string), saved[1].(string), saved[2].(string)
		kernelOpened, _ = saved[3].(*kernelmodel.Model)
		prefillInstances, decodeInstances, numInstances = saved[4].(int), saved[5].(int), saved[6].(int)
		blockSizeTokens = saved[7].(int64)
	}()
	// NIXL moves KV over multi-node NVLink only with UCX_CUDA_IPC_ENABLE_MNNVL=y; without it every
	// cross-node handoff takes the NIC and the rack boundary prices nothing. The fixture states it.
	dir := t.TempDir()
	body := strings.ReplaceAll(readFile(t, filepath.Join(pdScenarios, pdDPScenario)), "      kv_cache_dtype: fp8\n",
		"      kv_cache_dtype: fp8\n      env: {UCX_CUDA_IPC_ENABLE_MNNVL: \"y\"}\n")
	if err := os.WriteFile(filepath.Join(dir, pdDPScenario), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.CopyFS(filepath.Join(dir, "router"), os.DirFS(filepath.Join(pdScenarios, "router"))); err != nil {
		t.Fatal(err)
	}
	repos := reposOf(dir, catalog, registry)
	m, err := kernelmodel.Open(pdDPScenario, repos)
	if err != nil {
		t.Fatal(err)
	}
	st, err := m.Settings()
	if err != nil {
		t.Fatal(err)
	}
	if st.DataParallel != 2 {
		t.Fatalf("fixture lost its point: dp %d", st.DataParallel)
	}
	kernelScenario, kernelScenarioDir, registryPath, kernelOpened = pdDPScenario, dir, registry, m
	blockSizeTokens = int64(st.BlockSize)
	prefillInstances, decodeInstances, numInstances = 1, 1, 2 // logical, as the CLI states them
	pools := openKernelPools()
	if pools == nil {
		t.Fatal("no pools opened for a disaggregated run")
	}
	// Every pool at dp instances, one per rank, as the pricer places them.
	dp := st.DataParallel
	prefillInstances, decodeInstances, numInstances = dp, dp, 2*dp
	endpointOf := pools.instanceEndpoint()
	decodeEnv, err := newTransferEnvironment(pools.decode, pools.prefill)
	if err != nil {
		t.Fatal(err)
	}
	decodePlan := plan.PoolPlan{Hardware: "decode", Layout: plan.Layout{Tensor: 1, RanksPerNode: 1, Nodes: 8}}
	price := func(tokens int64, from, to cluster.InstanceID) int64 {
		d, err := env.DemandFor(decodeEnv, decodePlan, engine.Transfer{ID: 1, Request: "r", Kind: engine.TransferLoad,
			Tokens: int(tokens), Blocks: int((tokens + blockSizeTokens - 1) / blockSizeTokens),
			Move: plan.Transfer{Tokens: units.Tokens(tokens), From: endpointOf(from), To: endpointOf(to), Kind: plan.TransferPeer}})
		if err != nil {
			t.Fatal(err)
		}
		return cluster.AloneTicks(cluster.FlowOf(d))
	}

	members := cluster.BuildPoolMembershipFromIndices(numInstances, prefillInstances, decodeInstances, 0, 0)
	byRole := map[cluster.PoolRole][]string{}
	for id, role := range members {
		byRole[role] = append(byRole[role], id)
	}
	num := func(id string) int {
		v, err := strconv.Atoi(strings.TrimPrefix(id, "instance_"))
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	for _, ids := range byRole {
		slices.SortFunc(ids, func(a, b string) int { return num(a) - num(b) })
	}
	pre, dec := byRole[cluster.PoolRolePrefill], byRole[cluster.PoolRoleDecode]
	if len(pre) != dp || len(dec) != dp {
		t.Fatalf("membership has %d prefill and %d decode instances, want %d each", len(pre), len(dec), dp)
	}
	if pools.prefill.Role() != deployment.RolePrefill || pools.decode.Role() != deployment.RoleDecode {
		t.Fatalf("pools opened as %s/%s", pools.prefill.Role(), pools.decode.Role())
	}

	rapid.Check(t, func(rt *rapid.T) {
		tokens := rapid.Int64Range(1, int64(st.MaxModelLen)).Draw(rt, "tokens")
		// The KV moves in whole blocks: the retired path priced the blocks' token capacity.
		capacity := (tokens + blockSizeTokens - 1) / blockSizeTokens * blockSizeTokens
		for i, from := range pre {
			for j, to := range dec {
				want := pools.prefill.PDTransferTicks(capacity, pools.prefill.PlacementOf(i), pools.decode.PlacementOf(j))
				if got := price(tokens, cluster.InstanceID(from), cluster.InstanceID(to)); got != want {
					rt.Fatalf("%d tokens %s(prefill rank %d) -> %s(decode rank %d): priced %d, the kernel %d",
						tokens, from, i, to, j, got, want)
				}
			}
		}
	})

	const big = 32768 // a multiple of every block size the fixtures use
	if a, b := price(big, cluster.InstanceID(pre[0]), cluster.InstanceID(dec[0])),
		price(big, cluster.InstanceID(pre[0]), cluster.InstanceID(dec[1])); a == b {
		t.Errorf("fixture lost its point: both decode ranks are priced %d ticks for %d tokens", a, big)
	}
}

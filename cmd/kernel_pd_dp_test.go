package cmd

import (
	"fmt"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"pgregory.net/rapid"
)

// P/D disaggregation with data parallelism on the kernel backend. The fixture
// (testdata/scenarios/minimax-m2.5-gb200-1p1d-dp2.yaml) states a routed MoE at dp 2 in both
// pools, so DP-as-placement expands every logical instance into two per-rank replicas:
//
//   - 1P1D logical runs as 2 prefill + 2 decode replicas, conserving every request;
//   - its exported trace replays byte-identically with the same flags and horizon (INV-13);
//   - after the expansion, the handoff pricer between any prefill and decode replica is the
//     prefill kernel's price between those replicas' placements in their pools.

const pdDPScenario = "minimax-m2.5-gb200-1p1d-dp2.yaml"

func TestRunCmd_KernelBackend_DisaggregatedDataParallel(t *testing.T) {
	const n = 20
	prefix := filepath.Join(t.TempDir(), "trace")
	common := []string{"--scenarios", pdScenarios, "--scenario", pdDPScenario, "--pd-decider", "always",
		"--num-instances", "2", "--prefill-instances", "1", "--decode-instances", "1", "--seed", "11",
		"--horizon", "600000000"}
	runOut, stderr, err := runKernelCLI(t, append([]string{"run", "--num-requests", strconv.Itoa(n), "--rate", "8",
		"--trace-output", prefix}, common...)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	if ids := instanceIDs(t, runOut); len(ids) != 4 {
		t.Fatalf("1P1D logical at dp 2 must run 2 prefill + 2 decode replicas, got %v", ids)
	}
	clusterConservationHolds(t, runOut, n)
	if got := clusterMetricInt(t, runOut, "completed_requests"); got != n {
		t.Errorf("completed %d of %d requests", got, n)
	}
	if !strings.Contains(runOut, fmt.Sprintf("Disaggregated Requests: %d", n)) {
		t.Errorf("not every request was disaggregated:\n%s", runOut)
	}

	repOut, stderr, err := runKernelCLI(t, append([]string{"replay", "--trace-header", prefix + ".yaml",
		"--trace-data", prefix + ".csv"}, common...)...)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 3))
	}
	if repOut != runOut {
		t.Errorf("the dp=2 disaggregated run's replay differs from the run\n--- run\n%s\n--- replay\n%s", runOut, repOut)
	}
}

// After DP expansion, transferTime() between prefill replica i and decode replica j, under the
// cluster's own instance numbering, is the prefill kernel's PDTransferTicks between rank i's and
// rank j's placements. The ranks are derived independently of rankInPool: a replica's rank is
// its position among its pool's instances in numeric order. Non-vacuity: the fixture's rack
// boundary prices the two decode ranks differently, so a pricer that ignored or confused the
// decode rank would break the law.
func TestKernelPools_TransferTime_IsThePlacedPairsPrice(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	saved := []any{kernelScenario, kernelScenarioDir, kernelRegistry, kernelOpened,
		prefillInstances, decodeInstances, numInstances, blockSizeTokens}
	defer func() {
		kernelScenario, kernelScenarioDir, kernelRegistry = saved[0].(string), saved[1].(string), saved[2].(string)
		kernelOpened, _ = saved[3].(*kernelmodel.Model)
		prefillInstances, decodeInstances, numInstances = saved[4].(int), saved[5].(int), saved[6].(int)
		blockSizeTokens = saved[7].(int64)
	}()
	repos := kernelmodel.Repos{Scenarios: pdScenarios, Catalog: catalog, Registry: registry}
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
	kernelScenario, kernelScenarioDir, kernelRegistry, kernelOpened = pdDPScenario, pdScenarios, registry, m
	blockSizeTokens = int64(st.BlockSize)
	prefillInstances, decodeInstances, numInstances = 1, 1, 2 // logical, as the CLI states them
	pools := openKernelPools(catalog)
	if pools == nil {
		t.Fatal("no pools opened for a disaggregated run")
	}
	// DP-as-placement multiplies every pool by the replica factor.
	dp := st.DataParallel
	prefillInstances, decodeInstances, numInstances = dp, dp, 2*dp
	price := pools.transferTime()

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
		for i, from := range pre {
			for j, to := range dec {
				want := pools.prefill.PDTransferTicks(tokens, pools.prefill.PlacementOf(i), pools.decode.PlacementOf(j))
				if got := price(tokens, cluster.InstanceID(from), cluster.InstanceID(to)); got != want {
					rt.Fatalf("%d tokens %s(prefill rank %d) -> %s(decode rank %d): priced %d, the kernel %d",
						tokens, from, i, to, j, got, want)
				}
			}
		}
	})

	const big = 32768
	if a, b := price(big, cluster.InstanceID(pre[0]), cluster.InstanceID(dec[0])),
		price(big, cluster.InstanceID(pre[0]), cluster.InstanceID(dec[1])); a == b {
		t.Errorf("fixture lost its point: both decode ranks are priced %d ticks for %d tokens", a, big)
	}
}

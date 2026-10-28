package cluster

import (
	"math"
	"testing"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"
)

// The cluster's per-role configuration (DeploymentConfig.Roles): a P/D pool's instances are
// built with their pool's own configuration, a shared prefill-decode pod with the decode pool's
// ("decode wins", issue #1276 D-2), and every other instance with the deployment's. The build site
// resolves each pool's configuration from its engine (cmd/engine_build.go resolvePoolConfig and
// rolesOf, tested in cmd/pool_overrides_test.go); the cluster only applies them.

// pdSimConfig is a P/D deployment's global engine configuration, TP 4.
func pdSimConfig() simConfig {
	return simConfig{
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       cengine.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         cengine.NewBatchConfig(256, 2048, 0),
		LatencyModel:        testFakeLatency(),
		ModelHardwareConfig: cengine.NewModelHardwareConfig(testModelConfig(), "test-model", "H100", 4, 1, false, 0),
	}
}

// newHeterogeneousDeploymentConfig is a 2-prefill, 2-decode deployment whose pools hold
// prefillKV and decodeKV KV blocks.
func newHeterogeneousDeploymentConfig(prefillKV, decodeKV int64) DeploymentConfig {
	prefill, decode := pdSimConfig(), pdSimConfig()
	prefill.TotalKVBlocks, decode.TotalKVBlocks = prefillKV, decodeKV
	d := withRoles(compatDeployment(pdSimConfig(), 4), &prefill, &decode)
	d.PrefillInstances, d.DecodeInstances, d.PDTransferTime = 2, 2, testPDTransferTime
	return d
}

// INV-P2-1's construction half: a prefill instance gets the prefill pool's configuration, a decode
// or shared one the decode pool's, any other the deployment's, and neither pool's leaks into the
// other's.
func TestResolveConfigForRole_EachRoleTakesItsPoolsConfiguration(t *testing.T) {
	prefill, decode := pdSimConfig(), pdSimConfig()
	prefill.TP, decode.TP = 8, 2
	dc := withRoles(compatDeployment(pdSimConfig(), 4), &prefill, &decode)
	for _, c := range []struct {
		name string
		role PoolRole
		tp   int
	}{
		{"prefill", PoolRolePrefill, 8},
		{"decode", PoolRoleDecode, 2},
		{"shared takes the decode pool's", PoolRolePrefillDecode, 2},
		{"no role", PoolRole(0), 4},
		{"encode", PoolRoleEncode, 4},
	} {
		got := dc.resolveConfigForRole(c.role)
		if got.TP != c.tp || engineConfigIn(got).TP != c.tp {
			t.Errorf("%s: TP = %d (engine %d), want %d", c.name, got.TP, engineConfigIn(got).TP, c.tp)
		}
	}
	if got := compatDeployment(pdSimConfig(), 4).resolveConfigForRole(PoolRolePrefill); got.TP != 4 {
		t.Errorf("a prefill role with no pool configuration: TP = %d, want the deployment's 4", got.TP)
	}
}

// INV-P2-1: each instance's engine holds its pool's KV capacity before the run (FreeKVBlocks),
// and a run of the heterogeneous deployment disaggregates and completes.
func TestINV_P2_1_PoolConfigConsistency(t *testing.T) {
	const prefillKV, decodeKV = 20000, 5000
	cs := newTestCluster(newHeterogeneousDeploymentConfig(prefillKV, decodeKV), NewSliceRequestSource(newTestRequests(5)), nil)
	membership := cs.PoolMembership()
	for _, inst := range cs.Instances() {
		role := membership[string(inst.ID())]
		want := map[PoolRole]int64{PoolRolePrefill: prefillKV, PoolRoleDecode: decodeKV}[role]
		if got := inst.FreeKVBlocks(); got != want {
			t.Errorf("instance %s (%v): FreeKVBlocks = %d, want %d", inst.ID(), role, got, want)
		}
	}
	mustRun(t, cs)
	if cs.AggregatedMetrics().CompletedRequests == 0 {
		t.Fatal("no requests completed with heterogeneous pool configurations")
	}
	if len(cs.ParentRequests()) == 0 {
		t.Fatal("no parent requests: disaggregation should be active")
	}
}

// INV-1 holds over a heterogeneous P/D deployment.
func TestINV_P2_1_RequestConservation(t *testing.T) {
	requests := newTestRequests(10)
	cs := newTestCluster(newHeterogeneousDeploymentConfig(20000, 5000), NewSliceRequestSource(requests), nil)
	mustRun(t, cs)
	assertClusterINV1Conservation(t, cs, len(requests), noRejections, "heterogeneous node pools")
}

// Issue #1276: a cluster of shared prefill-decode pods only is a pool configuration, every
// instance carrying both role bits.
func TestNewClusterSimulator_PureSharedCluster(t *testing.T) {
	d := compatDeployment(pdSimConfig(), 3)
	d.SharedInstances, d.PDTransferTime = 3, testPDTransferTime
	cs := newTestCluster(d, NewSliceRequestSource(nil), nil)
	if !cs.poolsConfigured() {
		t.Error("pure-shared cluster should have poolsConfigured() == true")
	}
	for id, role := range cs.PoolMembership() {
		if !role.Has(PoolRolePrefill) || !role.Has(PoolRoleDecode) {
			t.Errorf("instance %s: role %v missing prefill and/or decode bit", id, role)
		}
	}
}

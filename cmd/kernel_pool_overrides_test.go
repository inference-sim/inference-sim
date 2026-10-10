package cmd

import (
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"
)

// Each role's overrides are its own pool's engine, answered by its own pool's kernel: the
// latency model is that pool's model, the KV budget is that pool's SettingsReserving answer at
// the run's LoRA reservation, and TP, max_num_seqs and max_num_batched_tokens are that pool's.
// The variant's pools differ in every one of the sized fields (prefill admits 128 sequences
// and 16384 batched tokens, decode 256 and 8192, and decode caches in bf16 where prefill
// caches in fp8), so overrides that swapped the pools, or sized both from one, would fail.
func TestKernelPools_Overrides_AreEachPoolsOwnEngine(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	dir := writeScenarioVariant(t,
		"      max_num_batched_tokens: 8192\n      max_num_seqs: 256\n",
		"      max_num_batched_tokens: 16384\n      max_num_seqs: 128\n",
		"      cache_dtype: fp8\n      block_size: 64\n      max_num_batched_tokens: 8192\n      max_num_seqs: 256\n      max_model_len: 32768\n      cudagraph_mode: PIECEWISE\n      gpu_memory_utilization: 0.9\n\npd_transfer",
		"      cache_dtype: auto\n      block_size: 64\n      max_num_batched_tokens: 8192\n      max_num_seqs: 256\n      max_model_len: 32768\n      cudagraph_mode: PIECEWISE\n      gpu_memory_utilization: 0.9\n\npd_transfer")
	saved := []any{kernelScenario, kernelScenarioDir, kernelRegistry, kernelOpened,
		prefillInstances, decodeInstances, loraReservedBytesForKV, numInstances}
	defer func() {
		kernelScenario, kernelScenarioDir, kernelRegistry = saved[0].(string), saved[1].(string), saved[2].(string)
		kernelOpened, _ = saved[3].(*kernelmodel.Model)
		prefillInstances, decodeInstances, loraReservedBytesForKV = saved[4].(int), saved[5].(int), saved[6].(int64)
		numInstances = saved[7].(int)
	}()
	repos := kernelmodel.Repos{Scenarios: dir, Catalog: catalog, Registry: registry}
	m, err := kernelmodel.Open("variant.yaml", repos)
	if err != nil {
		t.Fatal(err)
	}
	kernelScenario, kernelScenarioDir, kernelRegistry, kernelOpened = "variant.yaml", dir, registry, m
	prefillInstances, decodeInstances, numInstances = 3, 1, 4

	for _, reserve := range []int64{0, 2 << 30} {
		loraReservedBytesForKV = reserve
		pools := openKernelPools(catalog)
		pre, dec := pools.overrides()
		want := map[deployment.Role]kernelmodel.Settings{}
		for _, c := range []struct {
			role deployment.Role
			m    *kernelmodel.Model
			o    cluster.PoolOverrides
		}{{deployment.RolePrefill, pools.prefill, pre}, {deployment.RoleDecode, pools.decode, dec}} {
			// The oracle opens the role's pool afresh rather than trusting the model the pools hold.
			own, err := kernelmodel.OpenRole("variant.yaml", repos, c.role)
			if err != nil {
				t.Fatal(err)
			}
			s, err := own.SettingsReserving(reserve)
			if err != nil {
				t.Fatal(err)
			}
			want[c.role] = s
			if c.m.Role() != c.role {
				t.Errorf("the %s pool's model prices the %s pool", c.role, c.m.Role())
			}
			if c.o.LatencyModel != sim.LatencyModel(c.m) {
				t.Errorf("reserve %d: the %s overrides price with another pool's model", reserve, c.role)
			}
			if c.o.TP == nil || c.o.TotalKVBlocks == nil || c.o.MaxNumSeqs == nil || c.o.MaxNumBatchedTokens == nil {
				t.Fatalf("%s overrides leave a sized field unset: %+v", c.role, c.o)
			}
			for _, f := range []struct {
				name      string
				got, want int64
			}{
				{"TotalKVBlocks", *c.o.TotalKVBlocks, s.KVBlocks},
				{"TP", int64(*c.o.TP), int64(own.Deployment().TP)},
				{"MaxNumSeqs", *c.o.MaxNumSeqs, int64(s.MaxNumSeqs)},
				{"MaxNumBatchedTokens", *c.o.MaxNumBatchedTokens, int64(s.MaxNumBatchedTokens)},
			} {
				if f.got != f.want {
					t.Errorf("reserve %d: %s pool's %s is %d, its own kernel answers %d",
						reserve, c.role, f.name, f.got, f.want)
				}
			}
		}
		p, d := want[deployment.RolePrefill], want[deployment.RoleDecode]
		if p.KVBlocks == d.KVBlocks || p.MaxNumSeqs == d.MaxNumSeqs || p.MaxNumBatchedTokens == d.MaxNumBatchedTokens {
			t.Fatalf("variant lost its point: prefill %+v and decode %+v agree on a sized field", p, d)
		}
	}
}

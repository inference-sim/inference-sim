// Moved from blis-latency-kernel/refusals_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"testing"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// WHAT THE KERNEL DOES NOT PRICE, IT REFUSES. Each case is a deployment that either does not
// start in vLLM v0.31.0 or that this kernel cannot price truthfully, and each sits beside a
// control that differs in the one setting and does start, so a refusal for some other reason
// cannot pass for the right one.
func TestWhatTheKernelCannotPriceItRefuses(t *testing.T) {
	starts := func(fixtureName string, edit func(*Inputs)) bool {
		t.Helper()
		in := fixtureInputs(t, fixtureName)
		edit(&in)
		_, err := newKernel(in)
		return err == nil
	}
	cache := func(c string) func(*Inputs) {
		return func(in *Inputs) { editInputsEngine(in, func(e *engineconfig.Engine) { e.CacheDType = c }) }
	}
	mamba := func(c string) func(*Inputs) {
		return func(in *Inputs) { editInputsEngine(in, func(e *engineconfig.Engine) { e.MambaCacheDType = c }) }
	}
	// twoNodes lays the pool out over two 8-GPU nodes at pipeline width pp and tensor width
	// tp, with the fabric a two-node cluster needs.
	twoNodes := func(pp, tp int) func(*Inputs) {
		return func(in *Inputs) {
			fabric := pcpInputs(t, dcpMLAFixture, 1)
			in.Scenario.Cluster.Fabric, in.Fabric = fabric.Scenario.Cluster.Fabric, fabric.Fabric
			pool := &in.Deployment.Pools[0]
			pool.Parallel.PP, pool.Parallel.TP = pp, tp
			pool.Nodes, in.Scenario.Cluster.Nodes = 2, 2
		}
	}
	ppWithDP := func(pp, dp int) func(*Inputs) {
		return func(in *Inputs) {
			twoNodes(pp, 4)(in)
			in.Deployment.Pools[0].Parallel.DP = dp
		}
	}
	renamed := func(name string) func(*Inputs) {
		return func(in *Inputs) {
			chip := *in.Chip
			chip.Name = name
			in.Chip = &chip
		}
	}
	for _, c := range []struct {
		name    string
		fixture string
		edit    func(*Inputs)
		want    bool
	}{
		// Pipeline parallelism is priced per stage (blis-latency-kernel#90), on two nodes and
		// one alike; beside data parallelism it is still refused.
		{"pp 1 on two nodes", dcpMLAFixture, twoNodes(1, 16), true},
		{"pp 2 on two nodes", dcpMLAFixture, twoNodes(2, 8), true},
		{"pp 2 with dp 2", dcpMLAFixture, ppWithDP(2, 2), false},
		// Cache dtypes: the engine's closed list (vllm/config/cache.py:39-58), less the
		// formats whose page this kernel does not size.
		{"cache fp8_e5m2", dcpMLAFixture, cache("fp8_e5m2"), true},
		{"cache bogus", dcpMLAFixture, cache("bogus"), false},
		{"cache int8_per_token_head", dcpMLAFixture, cache("int8_per_token_head"), false},
		{"cache turboquant_k8v4", dcpMLAFixture, cache("turboquant_k8v4"), false},
		{"mamba float32", dcpMLAFixture, mamba("float32"), true},
		{"mamba fp8", dcpMLAFixture, mamba("fp8"), false},
		// Dtypes the engine accepts but no backend serves for this attention kind: plain
		// nvfp4 with MLA (vllm/config/vllm.py:3414-3430), the ds_mla layouts outside sparse
		// MLA, and fp8_inc anywhere on CUDA.
		{"nvfp4 on gqa", "minimax-m25-h200-tp8.yaml", cache("nvfp4"), true},
		{"nvfp4 on mla", dcpMLAFixture, cache("nvfp4"), false},
		{"fp8_ds_mla on sparse mla", dcpSparseFixture, cache("fp8_ds_mla"), true},
		{"fp8_ds_mla on plain mla", dcpMLAFixture, cache("fp8_ds_mla"), false},
		{"fp8_inc", dcpMLAFixture, cache("fp8_inc"), false},
		// A DSA model below SM90 has no sparse-MLA backend; a plain-MLA model on the same
		// renamed part is the control.
		{"plain MLA on l40s", dcpMLAFixture, renamed("l40s"), true},
		{"DSA on l40s", dcpSparseFixture, renamed("l40s"), false},
		{"DSA on h200", dcpSparseFixture, renamed("h200"), true},
	} {
		if got := starts(c.fixture, c.edit); got != c.want {
			t.Errorf("%s: starts %v, want %v", c.name, got, c.want)
		}
	}
}

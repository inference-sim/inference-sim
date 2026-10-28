// Moved from blis-latency-kernel/resolution_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// A consumer reaches what this kernel resolved through kernel.Kernel alone: Deployment for
// the pool as stated, Resolved for what resolution settled. Nothing a simulator needs is on
// the concrete type, so a second cost model implementing the interface can serve the same
// consumer. (The two context-parallel widths are the interim exception; see
// DecodeContextParallelWidth.)
//
// Each test asserts what THIS pool states or resolved, not merely something non-zero. A
// kernel answering for pool 0 whatever index it was opened at would satisfy a non-zero
// check and still be the bug -- a decode pool priced at a prefill pool's parallelism --
// that holding a pool index alongside the kernel used to cause.

// TestDeploymentIsThePoolThisKernelPrices: one kernel per pool, and each reports its own.
func TestDeploymentIsThePoolThisKernelPrices(t *testing.T) {
	r := writeDisaggregated(t)
	for _, c := range []struct {
		pool    int
		role    deployment.Role
		batched int
	}{
		// The fixture gives prefill a large token budget and decode a small one, which is
		// the realistic shape and makes the two engines distinguishable.
		{0, deployment.RolePrefill, 8192},
		{1, deployment.RoleDecode, 2048},
	} {
		k, err := OpenPool("disagg.yaml", r, c.pool)
		if err != nil {
			t.Fatal(err)
		}
		got := k.Deployment()
		if got.Role != c.role {
			t.Errorf("pool %d reports role %q, want %q", c.pool, got.Role, c.role)
		}
		if b := mustDecode(t, got).MaxNumBatchedTokens; b != c.batched {
			t.Errorf("pool %d reports max_num_batched_tokens %d, want %d", c.pool, b, c.batched)
		}
	}
}

// TestDeploymentIsTheRequestAndResolvedIsTheResolution is the distinction the two methods
// exist to keep apart.
//
// A tensor-parallel group of 16 spans two 8-GPU H200 nodes, which have no multi-node
// NVLink, so the custom all-reduce the engine would otherwise use cannot run and NCCL
// does. The deployment does not opt out of the custom kernel, so it REQUESTS it. Deployment
// must report that request unchanged, and Resolved must report what will run and why --
// a reader shown one while believing it was the other would cost the reduction on the
// wrong resource.
func TestDeploymentIsTheRequestAndResolvedIsTheResolution(t *testing.T) {
	in := fixtureInputs(t, "minimax-m25-h200-ep16.yaml")
	pool := &in.Deployment.Pools[0]
	pool.Parallel.TP, pool.Parallel.DP = 16, 1
	k, err := newKernel(in)
	if err != nil {
		t.Fatal(err)
	}

	stated := mustDecode(t, k.Deployment())
	if stated.DisableCustomAllReduce != nil || stated.AllReduceBackend != "" {
		t.Fatalf("the fixture must leave the custom all-reduce requested by default; it "+
			"states disable=%v backend=%q", stated.DisableCustomAllReduce,
			stated.AllReduceBackend)
	}
	res := k.Resolved()
	if res.AllReduceBackend != "nccl" {
		t.Errorf("Resolved reports the all-reduce backend %q; a 16-rank group across "+
			"nodes without multi-node NVLink runs NCCL", res.AllReduceBackend)
	}
	declined := false
	for _, o := range res.Overrides {
		if o.Field == "allreduce_backend" && o.Requested == "custom" && o.Resolved == "nccl" {
			declined = o.Reason != ""
		}
	}
	if !declined {
		t.Errorf("Resolved does not record the declined request with a reason: %+v",
			res.Overrides)
	}
	if tp := k.Deployment().Parallel.TP; res.TensorParallelWidth != tp {
		t.Errorf("Resolved reports tp %d against the %d priced", res.TensorParallelWidth, tp)
	}
}

// TestResolvedFollowsTheLayoutWhenItChanges is the metamorphic half: change one width in
// the deployment and Resolved must move with it, and only where the engine's own
// definitions say it does.
//
// The expert group is tp x pcp x dp with expert parallelism on and one rank with it off,
// whatever tp and dp are (vllm/distributed/parallel_state.py:2212-2220 at v0.31.0; the
// width-of-one case is ExpertParallelWidth's own definition). So doubling dp doubles the
// data-parallel width in both cases, and the expert width only when the group exists.
func TestResolvedFollowsTheLayoutWhenItChanges(t *testing.T) {
	for _, ep := range []bool{true, false} {
		build := func(dp int) kernel.Resolution {
			t.Helper()
			in := fixtureInputs(t, "minimax-m25-h200-ep16.yaml")
			pool := &in.Deployment.Pools[0]
			pool.Parallel.EnableExpertParallel = ep
			pool.Parallel.DP = dp
			pool.Nodes, in.Scenario.Cluster.Nodes = dp, dp
			k, err := newKernel(in)
			if err != nil {
				t.Fatalf("ep=%v dp=%d: %v", ep, dp, err)
			}
			return k.Resolved()
		}
		one, two := build(1), build(2)
		if two.DataParallelWidth != 2*one.DataParallelWidth {
			t.Errorf("ep=%v: doubling dp took the data-parallel width from %d to %d",
				ep, one.DataParallelWidth, two.DataParallelWidth)
		}
		if two.TensorParallelWidth != one.TensorParallelWidth {
			t.Errorf("ep=%v: changing dp moved the tensor-parallel width from %d to %d",
				ep, one.TensorParallelWidth, two.TensorParallelWidth)
		}
		wantExpert := one.ExpertParallelWidth
		if ep {
			wantExpert *= 2
		}
		if two.ExpertParallelWidth != wantExpert {
			t.Errorf("ep=%v: doubling dp took the expert width from %d to %d, want %d",
				ep, one.ExpertParallelWidth, two.ExpertParallelWidth, wantExpert)
		}
		if !ep && two.ExpertParallelWidth != 1 {
			t.Errorf("expert parallelism off reports an expert group of %d, want 1",
				two.ExpertParallelWidth)
		}
	}
}

// TestOpenInputsAreTheDocumentsOpenPoolPrices: a harness that takes the documents from
// OpenInputs and builds the kernel itself must get the kernel OpenPool would have built,
// and must be holding the same chip and graph that kernel priced.
//
// That is what lets a scoring harness read deployment identity -- model, chip memory,
// expert geometry -- without the kernel re-exporting it and without loading a file a
// second time.
func TestOpenInputsAreTheDocumentsOpenPoolPrices(t *testing.T) {
	r := writeDisaggregated(t)
	for _, pool := range []int{0, 1} {
		in, err := OpenInputs("disagg.yaml", r, pool)
		if err != nil {
			t.Fatal(err)
		}
		mine, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		theirs, err := OpenPool("disagg.yaml", r, pool)
		if err != nil {
			t.Fatal(err)
		}
		for _, b := range []kernel.Batch{decodeBatch(32, 1, 4096), decodeBatch(1, 2048, 2048)} {
			if a, c := mine.StepTime(b), theirs.StepTime(b); a.NoOverlap != c.NoOverlap ||
				a.Overlap != c.Overlap {
				t.Errorf("pool %d: New(OpenInputs) priced %v/%v, OpenPool %v/%v", pool,
					a.Overlap, a.NoOverlap, c.Overlap, c.NoOverlap)
			}
		}
		if mine.FixedBytes() != theirs.FixedBytes() {
			t.Errorf("pool %d: the two kernels hold different fixed occupancy", pool)
		}
		if in.PoolIndex != pool || in.Deployment.Pools[pool].Role != mine.Deployment().Role {
			t.Errorf("pool %d: OpenInputs selected pool %d (%s), the kernel prices %s",
				pool, in.PoolIndex, in.Deployment.Pools[in.PoolIndex].Role,
				mine.Deployment().Role)
		}
		if in.Chip.Name != in.Scenario.Cluster.Hardware || in.Model.Name != in.Scenario.Model {
			t.Errorf("pool %d: OpenInputs holds chip %q and model %q for a scenario naming "+
				"%q and %q", pool, in.Chip.Name, in.Model.Name,
				in.Scenario.Cluster.Hardware, in.Scenario.Model)
		}
	}
}

// THE KERNEL OWNS ITS STATE. The engine block's optional settings are pointers, so a kernel
// that shared them would reprice when its caller edited a deployment it had already priced
// -- which a configuration search sweeping variants does -- or wrote through what
// Deployment returned. Neither may move any answer.
func TestEditingTheDocumentsAfterNewChangesNothing(t *testing.T) {
	in := fixtureInputs(t, "nemotron3-ultra-h100-agg.yaml") // states speculative decoding
	k, err := newKernel(in)
	if err != nil {
		t.Fatal(err)
	}
	batch := decodeBatch(32, 2, 8192)
	step, fixed, perSeq := k.StepTime(batch).NoOverlap, k.SequenceFixedBytes(),
		k.SequenceVariableBytes(8192)

	if poolEngine(t, in).Speculative == nil {
		t.Fatal("the fixture states no speculative block to edit")
	}
	// The caller edits its own documents, the block the kernel's Deployment returned, and the
	// settings Engine returned; none of the three reaches the kernel.
	editEngine(t, &in, func(e *engineconfig.Engine) { e.Speculative.NumSpecTokens = 7 })
	d := k.Deployment()
	editPoolEngine(t, &d, func(e *engineconfig.Engine) { e.Speculative.NumSpecTokens = 9 })
	if s := k.Engine(); s.Speculative != nil {
		s.Speculative.NumSpecTokens = 11
	}
	if got := mustDecode(t, k.Deployment()).Speculative.NumSpecTokens; got == 7 || got == 9 {
		t.Errorf("Deployment reports %d speculative tokens after the caller's edits", got)
	}
	if got := k.Engine().Speculative.NumSpecTokens; got == 7 || got == 9 || got == 11 {
		t.Errorf("Engine reports %d speculative tokens after the caller's edits", got)
	}
	if k.StepTime(batch).NoOverlap != step || k.SequenceFixedBytes() != fixed ||
		k.SequenceVariableBytes(8192) != perSeq {
		t.Error("editing the documents after New changed the kernel's answers")
	}
}

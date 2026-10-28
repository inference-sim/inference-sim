// Moved from blis-latency-kernel/pcp_shard_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"math"
	"path/filepath"
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
)

// Prefill-context parallelism splits a prefill sequence across ranks, and these pin that
// behaviourally.
//
// THE DEFECT. PCP was half-modelled where DCP was not modelled at all. It reached a price,
// but only as a side effect of the expert group -- ExpertParallelWidth widens with pcp --
// so a wider PCP sharded more experts and paid a wider all-to-all. The sharding PCP is
// NAMED for -- "Number of ranks that split prefill sequence computation"
// (vllm/config/parallel.py:131-133 at v0.31.0) -- was not credited at all, so a PCP
// deployment's prefill was priced as whole work on every rank. The error is one-sided and
// the opposite of DCP's: prefill was OVERSTATED.
//
// WHAT THE ENGINE DOES, which these tests encode:
//
//	a rank runs a forward over its own share   worker/gpu/pcp_manager.py:485
//	the split is a zigzag over 2*pcp chunks    worker/gpu/pcp_manager.py:235-273
//	decodes are replicated, not split          worker/gpu/pcp_manager.py:260-263
//	the KV cache stays replicated              config/parallel.py:131-133
//	so the ranks all-gather what they wrote     attention/ops/pcp.py:31-35
//
// The asymmetry against DCP is the thing most easily got backwards, and two tests here
// assert the non-effects rather than leaving them to inference.
const pcpFixture = dcpMLAFixture

// pcpKernel builds a kernel from a committed fixture with the prefill-context-parallel
// width changed, laid out the way the engine actually runs it.
func pcpKernel(t testing.TB, fixture string, pcp int) *Kernel {
	t.Helper()
	in := pcpInputs(t, fixture, pcp)
	k, err := newKernel(in)
	if err != nil {
		t.Fatalf("%s at pcp=%d: %v", fixture, pcp, err)
	}
	return k
}

// pcpInputs is a committed fixture's Inputs with the prefill-context-parallel width set to
// pcp AND the nodes to hold it.
//
// PCP EXPANDS THE WORLD SIZE. vLLM's ParallelConfig sets world_size = pp * tp * pcp
// (vllm/config/parallel.py:899-903 at v0.31.0) -- DCP reuses ranks, PCP adds them -- so a
// tp=8 fixture on one 8-GPU node at pcp=2 needs sixteen GPUs. An earlier form of this
// helper set the width alone and priced exactly that sixteen-rank layout on eight GPUs,
// which the engine cannot start; blis-schemas v0.2.2 refuses it ("needs 16 GPUs (pp 1 x
// tp 8 x pcp 2 x dp 1, one per rank) but the pool's 1 node(s) of 8 GPUs provide 8").
//
// So the pool and the cluster grow by the split, which is the shape PCP is deployed in:
// the tensor-parallel group stays inside a node and the PCP ranks are added across nodes.
// The ranks are laid out DP x PP x PCP x TP with TP innermost (parallel_state.py:2054-2060),
// which is what places a PCP group's members one tensor-parallel width apart and therefore
// on different nodes here.
//
// A cross-node cluster must name its fabric, a rule blis-schemas held before PCP existed.
// The fabric is set at EVERY width, pcp=1 included, so a comparison across widths differs
// in the split and nothing else; nothing crosses a node at pcp=1, so it prices no term there.
func pcpInputs(t testing.TB, fixture string, pcp int) Inputs {
	t.Helper()
	in := fixtureInputs(t, fixture)
	if len(in.Deployment.Pools) != 1 {
		t.Fatalf("%s has %d pools; pcpInputs grows one pool and its cluster together",
			fixture, len(in.Deployment.Pools))
	}
	pool := &in.Deployment.Pools[0]
	pool.Parallel.PCP = pcp
	if pcp > 1 {
		pool.Nodes *= pcp
		in.Scenario.Cluster.Nodes *= pcp
	}
	if in.Fabric == nil {
		const fabric = "ib-400g"
		f, err := schemas.LoadFabric(filepath.Join(catalogRoot, "networks", fabric+".yaml"))
		if err != nil {
			t.Fatalf("fabric %s: %v", fabric, err)
		}
		in.Scenario.Cluster.Fabric, in.Fabric = fabric, f
	}
	return in
}

// The causal attention term must divide by exactly the split. Isolated the same way the
// DCP read is: with work(p) = W/p plus terms independent of p, the difference of
// differences cancels everything else and must be exactly 2.
//
// Charged to SM, because prefill attention is compute-bound -- which is itself the reason
// the regime carries its own measured floor and scale.
func TestPCPDividesTheCausalTermByExactlyTheSplit(t *testing.T) {
	batch := decodeBatch(1, 4096, 4096)
	sm := func(pcp int) float64 {
		return pcpKernel(t, pcpFixture, pcp).StepTime(batch).
			PerResource[kernel.ResourceSM].Seconds()
	}
	s1, s2, s4 := sm(1), sm(2), sm(4)
	if s2 >= s1 || s4 >= s2 {
		t.Fatalf("the SM term is not falling with the split: %.6f, %.6f, %.6f ms",
			s1*1e3, s2*1e3, s4*1e3)
	}
	if ratio := (s1 - s2) / (s2 - s4); math.Abs(ratio-2) > 1e-6 {
		t.Errorf("(t1-t2)/(t2-t4) is %.9f, want 2: the prefill work is not dividing by "+
			"exactly the split width", ratio)
	}

	// AND THE DIVISOR MUST BE THE ENGINE'S PARTITION, NOT A NOMINAL 1/pcp. The two agree
	// whenever 2*pcp divides the chunk -- 4,096 is divisible by 16, which is why the ratio
	// above cannot tell them apart -- and diverge on a ragged one: at 100 tokens over 8
	// ranks the busiest rank carries 25.4% more causal work than an even share, because a
	// ceil-based chunking hands it two chunks from the expensive end.
	//
	// COVERAGE LIMIT, STATED RATHER THAN PAPERED OVER. An exact call-site ratio is NOT
	// recoverable, and the reason is itself a modelled effect rather than a weakness of
	// the test. Prefill SM time is floors plus causal count over (peak * efficiency), and
	// the efficiency ramp is evaluated at the RANK-LOCAL token count -- correctly, since a
	// prefill-context rank runs its forward over its own share -- so the rate differs
	// between two widths and does not cancel under differencing. Two drafts of this test
	// tried: differencing across widths read 3.09 where the partition predicts 9.00, and
	// dividing out the rate implied a pair count 1.69x the true one, because the ramp had
	// moved. On top of that, the chunk sizes where the two divisors differ most are the
	// ones where the per-call floors are the largest share of the step: 1.4% excess at
	// 1,001 tokens where floors are 12.6% of SM, against 0.024% excess at 8,191 where
	// they are 2.1%.
	//
	// So the exact arithmetic is pinned on the FUNCTION, by
	// TestPCPCausalCountKeepsTheUnshardedContinuumConvention, which checks the busiest
	// rank's count against the engine's partition including the ragged cases. What is
	// pinned here is that the call site consumes it: the excess exists and the step
	// carries a causal term above its floors.
	for _, ragged := range []int{100, 1001} {
		s := float64(ragged)
		nominal := 2 * 2 * (s * s * 0.5) / 8
		exact := pcpCausalPairs(ragged, 0, 8, 1)
		if exact <= nominal {
			t.Errorf("at %d tokens over 8 ranks the engine's partition gives %v pairs "+
				"against an even share of %v; a ceil-based chunking cannot give less",
				ragged, exact, nominal)
		}
	}
}

// The per-call prefill floor must NOT divide with the split, and the split must NOT make a
// rank's work exactly linear either.
//
// Two claims, isolated by holding the RANK-LOCAL token count fixed: at pcp 1, 2 and 4 with
// chunks of 1024, 2048 and 4096, every rank computes 1,024 token-rows, so the GEMM shapes
// and the efficiency ramp's argument are identical across the three. What differs is the
// GLOBAL sequence each of those tokens attends, and the floors.
//
// The SM term must then RISE with the split, not fall and not hold: a rank doing the same
// number of rows against a longer global sequence does strictly more attention work, while
// its per-call floors stay put. A floor scaled by the split would pull this the other way.
//
// Measured: 38.39, 40.63 and 45.13 ms at pcp 1, 2 and 4.
func TestPCPDoesNotDivideThePerCallFloorOrMakeARankLinear(t *testing.T) {
	const perRank = 1024
	sm := func(pcp, sched int) float64 {
		return pcpKernel(t, pcpFixture, pcp).StepTime(decodeBatch(1, sched, sched)).
			PerResource[kernel.ResourceSM].Seconds()
	}
	var prev float64
	for i, pcp := range []int{1, 2, 4} {
		sched := perRank * pcp
		if local := pcpLocalTokens(sched, pcp, 1); pcp > 1 && local != perRank {
			t.Fatalf("pcp=%d: a %d-token chunk leaves %d tokens on a rank, want %d; the "+
				"comparison needs the rank-local count held fixed", pcp, sched, local, perRank)
		}
		got := sm(pcp, sched)
		if i > 0 && got <= prev {
			t.Errorf("pcp=%d priced %.6f ms against %.6f ms at the previous width; a rank "+
				"computing the same %d rows against a longer global sequence does more "+
				"attention work, and its floors do not shrink", pcp, got*1e3, prev*1e3, perRank)
		}
		prev = got
	}

	// And the floors are really there: a chunk short enough that the causal work vanishes
	// still costs at least one prefill floor per layer, at every width.
	for _, pcp := range []int{1, 2, 4} {
		k := pcpKernel(t, pcpFixture, pcp)
		tiny := k.StepTime(decodeBatch(1, 2*2*pcp, 2*2*pcp)).
			PerResource[kernel.ResourceSM].Seconds()
		floors := k.attentionPrefillFloor.Seconds() * float64(k.plan.TotalLayers)
		if tiny < floors {
			t.Errorf("pcp=%d: a near-empty prefill priced %.6f ms, below the %.6f ms its "+
				"per-layer floors alone cost", pcp, tiny*1e3, floors*1e3)
		}
	}
}

// The structural check, as for DCP: poke the resolved layout on a built kernel and require
// the prefill price to move. This catches a field that is plumbed but never read, which is
// the state PCP's prefill split was in.
func TestPCPGeometryReachesThePricer(t *testing.T) {
	k := fixture(t, pcpFixture)
	batch := decodeBatch(1, 4096, 4096)
	before := k.StepTime(batch).NoOverlap
	k.layout.PCP = 4
	after := k.StepTime(batch).NoOverlap
	if after >= before {
		t.Errorf("setting the resolved PCP width to 4 left the prefill step at %v "+
			"against %v; the layout field is not reaching the pricer", after, before)
	}
}

// collectiveSeconds is a step's whole collective time, on-node and cross-node together.
// The PCP gather lands on either depending on where its group's members sit, so a test
// about whether the gather is charged reads both.
func collectiveSeconds(e kernel.StepEstimate) float64 {
	return (e.PerResource[kernel.ResourceNVLink] + e.PerResource[kernel.ResourceNIC]).Seconds()
}

// A SLIDING-WINDOW layer's prefill must be split by WHERE this rank's tokens sit, not
// merely by how many it has.
//
// THE DEFECT THIS WAS WRITTEN AGAINST. prefillChunks carries `chunk{sched, prefix}`, which
// windowedCausalFLOPs reads POSITIONALLY: a query at chunk-local offset j attends
// min(prefix+j+1, window) keys. Crediting the split by replacing `sched` with the rank's
// token COUNT while leaving `prefix` at the request's own offset described the rank as one
// short run at the cheap start of the sequence. The zigzag gives it two runs, one near the
// start and one near the END, where every query reads a full window. Measured on the
// function: a 4,096-token chunk on no prefix at pcp 8 and window 2,176 was charged 524,288
// pair-units against a true 2,359,296 -- understated 4.5x.
//
// WHY IT NEEDS A WIDE WINDOW TO SHOW, which is why a first draft of this test missed it:
// the error needs the window to exceed the rank's token count. At window 128 a rank
// holding 512 tokens saturates almost everywhere regardless of position, so both readings
// agree to the digit -- and gpt-oss-120b's window IS 128, so the hybrid fixture the DCP
// tests use cannot see this at all.
func TestPCPSplitsAWindowedPrefillByPositionNotByCount(t *testing.T) {
	// THROUGH THE PRICER'S OWN PATH. appendPrefillRuns is what stepTime calls, and it
	// reads the split widths from the resolved layout -- so this exercises the same code
	// the step does rather than a free function beside it.
	k8 := pcpKernel(t, pcpFixture, 8)
	runs := k8.appendPrefillRuns(nil, 4096, 0)
	if len(runs) != 2 {
		t.Fatalf("a split prefill over 8 ranks should yield two runs, got %d: %+v",
			len(runs), runs)
	}
	lo, hi := runs[0], runs[1]
	if hi.prefix < 4096/2 {
		t.Errorf("the high run starts at %d, below the sequence midpoint; a zigzag rank "+
			"owns a chunk from the expensive end and that is what makes its queries "+
			"saturate a wide window", hi.prefix)
	}
	if lo.prefix+lo.sched > hi.prefix {
		t.Errorf("the runs overlap (%d+%d > %d)", lo.prefix, lo.sched, hi.prefix)
	}
	if held, want := lo.sched+hi.sched, pcpLocalTokens(4096, 8, 1); held != want {
		t.Errorf("the runs hold %d tokens, want the busiest rank's %d", held, want)
	}
	// Without the split, the runs are the request itself, so no existing deployment moves.
	if off := fixture(t, pcpFixture).appendPrefillRuns(nil, 4096, 13); len(off) != 1 ||
		off[0].sched != 4096 || off[0].prefix != 13 {
		t.Errorf("with pcp off the runs should be the request itself, got %+v", off)
	}

	// THE WINDOWED TERM, as two metamorphic relations rather than pinned figures. A window
	// WIDER than this rank's token count must cost more positionally than collapsed to the
	// start, because the high run then reads a full window; a window much NARROWER must
	// cost exactly the same, because it saturates wherever the tokens sit. Both sides are
	// computed from the kernel's own function, so a registry refit cannot break them.
	for _, pcp := range []int{2, 4, 8} {
		positional := pcpKernel(t, pcpFixture, pcp).appendPrefillRuns(nil, 4096, 0)
		held := 0
		for _, c := range positional {
			held += c.sched
		}
		flat := []chunk{{sched: held, prefix: 0}}
		if wide, wideFlat := windowedCausalFLOPs(positional, 2176),
			windowedCausalFLOPs(flat, 2176); wide <= wideFlat {
			t.Errorf("pcp=%d: a 2,176-token window costs %v positionally against %v by "+
				"token count; the high run must be charged for a full window",
				pcp, wide, wideFlat)
		}
		if narrow, narrowFlat := windowedCausalFLOPs(positional, 8),
			windowedCausalFLOPs(flat, 8); narrow != narrowFlat {
			t.Errorf("pcp=%d: an 8-token window costs %v positionally against %v by "+
				"token count; a window this narrow saturates wherever the tokens sit",
				pcp, narrow, narrowFlat)
		}
	}

	// COVERAGE LIMIT, STATED RATHER THAN PAPERED OVER. No sound relation on the STEP
	// separates the two readings. On the most windowed fixture committed -- minimax-m3,
	// 57 of 60 layers at a 2,176-token window -- the defect moves a 2,048-token prefill by
	// 5.1% of the SM term, because a dense GEMM dominates a windowed attention term
	// roughly five to one. Three step-level relations were tried and all three hold under
	// the defect as well as without it: a pinned ratio (also fitted to the current
	// registry, so unacceptable regardless), a sandwich between the unsplit figure and a
	// same-length contiguous chunk, and a sandwich between a saturating and an unbounded
	// window. A bound loose enough to survive a refit is wider than 5.1%.
	//
	// So the guard is the chunk geometry above, taken through the pricer's own method.
	// Closing the step-level gap needs a fixture whose windowed layers dominate its GEMMs,
	// which the catalog does not carry.
}

// A LATENT LAYER'S PCP GATHER IS TWO LAUNCHES. _gather_prefill_cache_inputs launches one
// all-gather per tensor (vllm/v1/attention/ops/pcp.py:31-35 at v0.31.0), and the MLA cache
// write hands it kv_c_normed and k_pe separately (pcp.py:56-74). Each pays its own floor.
//
// Isolated by inflating the 2-rank all-gather floor: at tp=8 and pcp=2 on deepseek-v3 the
// only 2-rank all-gather is the PCP group's, and a collective's floor adds to its byte
// term, so inflating it tenfold moves the prefill's collective time by exactly nine floors
// per launch. Two launches per layer gives a ratio of exactly 2.
func TestALatentLayersPCPGatherIsTwoLaunches(t *testing.T) {
	prefill := decodeBatch(1, 4096, 4096)
	base := pcpKernel(t, pcpFixture, 2)
	inflatedIn := pcpInputs(t, pcpFixture, 2)
	inflatedIn.Coefficients = scaleFloors(inflatedIn.Coefficients, "all_gather_fp16_2rank", 10)
	inflated, err := newKernel(inflatedIn)
	if err != nil {
		t.Fatal(err)
	}
	floor := base.collectiveFloors[collKey{Op: model.OpAllGather, Group: price.GroupPCP}]
	moved := collectiveSeconds(inflated.StepTime(prefill)) - collectiveSeconds(base.StepTime(prefill))
	launches := moved / (9 * floor.Seconds() * float64(base.plan.TotalLayers))
	if math.Abs(launches-2) > 1e-6 {
		t.Errorf("inflating the PCP all-gather floor moved the prefill by %.6f launches' worth "+
			"per layer, want 2: kv_c_normed and k_pe are gathered separately", launches)
	}
}

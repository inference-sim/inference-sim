// Moved from blis-latency-kernel/pcp_shard_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"math"
	"path/filepath"
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/kernel"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
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

// THE ACCEPTANCE CRITERION for the PCP half: a wider split must make a prefill cheaper,
// monotonically. Before this the prefill term did not move at all.
//
// Four widths, not two, so a constant divisor cannot pass.
func TestPCPLowersAPrefillStepAndWideningItLowersFurther(t *testing.T) {
	batch := decodeBatch(1, 4096, 4096)
	var prev float64
	for i, pcp := range []int{1, 2, 4, 8} {
		got := pcpKernel(t, pcpFixture, pcp).StepTime(batch).NoOverlap.Seconds()
		if i > 0 && got >= prev {
			t.Errorf("pcp=%d priced %.6f ms, not below the %.6f ms of the previous "+
				"width: splitting a prefill further must compute less per rank",
				pcp, got*1e3, prev*1e3)
		}
		prev = got
	}
}

// PCP MUST NOT TOUCH THE DECODE READ. It splits prefill computation and leaves the KV
// cache replicated -- "PCP expands the process world size but does not increase the
// KV-cache shard count" (config/parallel.py:131-133) -- which is exactly the opposite of
// DCP. Conflating the two is the single easiest thing to get backwards, so the non-effect
// is asserted rather than inferred.
func TestPCPDoesNotShardTheDecodeRead(t *testing.T) {
	decode := decodeBatch(32, 1, 32768)
	want := pcpKernel(t, pcpFixture, 1).StepTime(decode).NoOverlap
	for _, pcp := range []int{2, 4, 8} {
		if got := pcpKernel(t, pcpFixture, pcp).StepTime(decode).NoOverlap; got != want {
			t.Errorf("pcp=%d priced a decode-only step at %v against %v at pcp=1; a "+
				"decode row is replicated across prefill-context ranks, and the cache "+
				"it reads is replicated too", pcp, got, want)
		}
	}
}

// And the converse, so neither axis can quietly acquire the other's divisor: PCP must not
// shard the capacity a rank holds, where DCP does.
func TestPCPDoesNotShardCapacityWhereDCPDoes(t *testing.T) {
	want := pcpKernel(t, pcpFixture, 1).SequenceVariableBytes(32768)
	for _, pcp := range []int{2, 8} {
		if got := pcpKernel(t, pcpFixture, pcp).SequenceVariableBytes(32768); got != want {
			t.Errorf("pcp=%d holds %d bytes against %d at pcp=1; the KV cache is "+
				"replicated across prefill-context ranks", pcp, got, want)
		}
	}
	// The same quantity DOES move with dcp, which is what makes the contrast meaningful
	// rather than a tautology about an unread field.
	if sharded := dcpKernel(t, pcpFixture, 8).SequenceVariableBytes(32768); sharded >= want {
		t.Errorf("dcp=8 holds %d bytes against %d unsharded; the decode axis must shard "+
			"the cache even though the prefill axis does not", sharded, want)
	}
}

// collectiveSeconds is a step's whole collective time, on-node and cross-node together.
// The PCP gather lands on either depending on where its group's members sit, so a test
// about whether the gather is charged reads both.
func collectiveSeconds(e kernel.StepEstimate) float64 {
	return (e.PerResource[kernel.ResourceNVLink] + e.PerResource[kernel.ResourceNIC]).Seconds()
}

// A PCP rank writes only its share of the new KV, but every rank must hold the whole
// cache, so the ranks all-gather what they wrote. Omitting it would make PCP look like a
// pure win, which it is not.
//
// Two properties: the gather exists on a prefill step, and it does NOT exist on a
// decode-only one -- the engine keeps replicated decode writes local and gathers
// partitioned prefills only (vllm/v1/attention/ops/pcp.py:16, :31-35 at v0.31.0).
func TestPCPAddsAKVGatherOnPrefillOnly(t *testing.T) {
	coll := func(pcp int, b kernel.Batch) float64 {
		return collectiveSeconds(pcpKernel(t, pcpFixture, pcp).StepTime(b))
	}
	// A decode-only step carries no PCP gather at all.
	decode := decodeBatch(32, 1, 32768)
	if on, off := coll(2, decode), coll(1, decode); on != off {
		t.Errorf("a decode-only step carried %.6f ms of collective at pcp=2 against "+
			"%.6f ms at pcp=1; a replicated decode write is not gathered", on*1e3, off*1e3)
	}

	// ON A SHORT PREFILL THE GATHER CHANGES THE SIGN, which is the shape that makes its
	// omission visible. Splitting halves the tensor-parallel payload, so the collective
	// total would fall if nothing were added -- but the gather pays a floor on every
	// KV-holding layer, and at a small chunk those floors outweigh what the halving saves.
	// So the total must RISE.
	//
	// At a long chunk it falls again, because the bytes saved dominate the floors added.
	// Both directions are asserted: a test that only looked at the long chunk would pass
	// with the gather deleted.
	short := decodeBatch(1, 64, 64)
	if whole, split := coll(1, short), coll(2, short); split <= whole {
		t.Errorf("a 64-token prefill priced %.6f ms of collective at pcp=2 against "+
			"%.6f ms at pcp=1; the KV gather's per-layer floors must outweigh the halved "+
			"tensor-parallel payload at this size", split*1e3, whole*1e3)
	}
	long := decodeBatch(1, 4096, 4096)
	if whole, split := coll(1, long), coll(2, long); split >= whole {
		t.Errorf("a 4,096-token prefill priced %.6f ms of collective at pcp=2 against "+
			"%.6f ms at pcp=1; at this size the bytes saved must dominate the floors added",
			split*1e3, whole*1e3)
	}
}

// The PCP gather is charged to the fabric exactly when the engine's rank layout puts its
// group across nodes, and to NVLink otherwise.
//
// vLLM lays ranks out with the tensor-parallel axis innermost, so a PCP group's members are
// tp ranks apart (vllm/distributed/parallel_state.py:2054-2060, :2155-2160 at v0.31.0). On
// 8-GPU nodes: tp=8, pcp=2 puts one member on each of two nodes, every hop crossing; tp=4,
// pcp=2 and tp=2, pcp=4 fit one node. Only the gather moves between the two -- a decode
// step has none, and the tensor-parallel group stays on a node in all three layouts -- so
// a prefill's NIC term is the gather or nothing.
func TestThePCPGatherCrossesTheFabricExactlyWhenItsGroupDoes(t *testing.T) {
	prefill := decodeBatch(1, 4096, 4096)
	for _, c := range []struct {
		tp, pcp, nodes int
		crosses        bool
	}{
		{8, 2, 2, true},
		{8, 4, 4, true},
		{4, 2, 1, false},
		{2, 4, 1, false},
	} {
		in := pcpInputs(t, pcpFixture, 1)
		pool := &in.Deployment.Pools[0]
		pool.Parallel.TP, pool.Parallel.PCP = c.tp, c.pcp
		pool.Nodes, in.Scenario.Cluster.Nodes = c.nodes, c.nodes
		k, err := newKernel(in)
		if err != nil {
			t.Fatalf("tp=%d pcp=%d: %v", c.tp, c.pcp, err)
		}
		per := k.StepTime(prefill).PerResource
		if nic := per[kernel.ResourceNIC]; (nic > 0) != c.crosses {
			t.Errorf("tp=%d pcp=%d on %d node(s): the prefill charged %v to the fabric; "+
				"the PCP group crosses a node: %v", c.tp, c.pcp, c.nodes, nic, c.crosses)
		}
	}
}

// The gather is charged on PREFILL tokens only. The engine gathers "partitioned prefills"
// while keeping "replicated decode writes local" (vllm/v1/attention/ops/pcp.py:16 at
// v0.31.0), so a 64-token prefill alone and the same prefill beside 64 decode rows must add
// exactly the same gather.
//
// The gather is also scoped to KV-holding layers in the code (it divides the KV figure by
// the KV-holding layer count and skips a layer with no attention). That half is not
// exercised here because no admissible deployment can see it: at v0.31.0 every stack with a
// cache-free layer has a recurrent mixer, and no recurrent backend supports PCP, so the
// engine refuses those layouts and New refuses them too (see TestPCPRunsOnlyOnLatentStacks).
// An earlier form of this test exercised it on Nemotron-3-Ultra, a Mamba hybrid, which
// priced a layout the engine does not start.
func TestThePCPGatherIsScopedToPrefillTokens(t *testing.T) {
	coll := func(pcp int, b kernel.Batch) float64 {
		return collectiveSeconds(pcpKernel(t, pcpFixture, pcp).StepTime(b))
	}
	alone := kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: 64, Computed: 0, PromptLen: 64},
	}}
	withDecodes := kernel.Batch{DecodeThreshold: 8, Reqs: append(
		[]kernel.ReqShape{{Scheduled: 64, Computed: 0, PromptLen: 64}},
		make([]kernel.ReqShape, 64)...)}
	for i := 1; i < len(withDecodes.Reqs); i++ {
		withDecodes.Reqs[i] = kernel.ReqShape{
			Scheduled: 1, Computed: 32767, PromptLen: 32768,
		}
	}
	addedAlone := coll(2, alone) - coll(1, alone)
	addedMixed := coll(2, withDecodes) - coll(1, withDecodes)
	if addedAlone <= 0 {
		t.Fatalf("the split added %.6f ms of collective on a 64-token prefill; the KV "+
			"gather must appear", addedAlone*1e3)
	}
	if math.Abs(addedMixed-addedAlone) > 2e-9 {
		t.Errorf("the gather added %.6f ms beside 64 decode rows against %.6f ms without "+
			"them, on the same 64-token prefill; a replicated decode write is not "+
			"gathered", addedMixed*1e3, addedAlone*1e3)
	}
}

// PCP runs only on a stack whose every layer's backend supports it, which at v0.31.0 is a
// stack of latent attention alone (vllm/v1/worker/cp_utils.py:35-38; backend.py:843, :1048,
// :230-235). New refuses the rest rather than pricing a layout the engine does not start,
// and admits the latent stacks. PCP with DCP narrows it further, to DSA sparse-MLA layers;
// TestNewRefusesTheContextParallelLayoutsTheEngineRefuses covers that.
func TestPCPRunsOnlyOnLatentStacks(t *testing.T) {
	for _, c := range []struct {
		fixture string
		runs    bool
		why     string
	}{
		{dcpMLAFixture, true, "deepseek-v3: every layer is mla"},
		{dcpSparseFixture, true, "glm5: every layer is sparse_mla"},
		{dcpHybridFixture, false, "gpt-oss-120b: gqa and swa layers"},
		{"nemotron3-ultra-h100-agg.yaml", false, "nemotron-3-ultra: mamba2 layers"},
		{"kimi-k3-h100-nospec.yaml", false, "kimi-k3: kda layers beside its mla"},
	} {
		in := pcpInputs(t, c.fixture, 2)
		in.Deployment.Pools[0].Parallel.DP = 1
		_, err := newKernel(in)
		if c.runs && err != nil {
			t.Errorf("%s: refused, but the engine runs it: %v", c.why, err)
		}
		if !c.runs && err == nil {
			t.Errorf("%s: priced at pcp=2, but the engine refuses it at startup", c.why)
		}
	}
}

// The expert group spans the prefill-context ranks as well as the tensor- and data-parallel
// ones: vLLM builds it over all three axes together, `data_parallel_size *
// prefill_context_model_parallel_size * tensor_model_parallel_size`
// (vllm/distributed/parallel_state.py:2212-2220 at v0.31.0).
//
// THE CASE THAT SEPARATES THE TWO FORMULAS is dp > 1 AND pcp > 1. blis-schemas v0.2.0
// computed tp * max(dp, pcp), which agrees with the product whenever one of dp and pcp is
// one -- every layout that version admitted -- and is a factor of min(dp, pcp) narrow once
// both exceed one, which v0.31.0 runs ("DP 4 x PCP 8 is an expert group of 32, not 8", in
// v0.2.2's own changelog). So the grid below includes that corner, and a regression to the
// max would fail exactly there.
//
// Asserted on the RESOLVED width, which is what selects the routed-expert shard and the
// all-to-all's group, and which a consumer reads.
func TestTheExpertGroupSpansTensorPrefillContextAndDataParallelRanks(t *testing.T) {
	for _, c := range []struct{ pcp, dp int }{{1, 1}, {2, 1}, {1, 2}, {2, 2}, {4, 2}} {
		in := pcpInputs(t, pcpFixture, c.pcp)
		pool := &in.Deployment.Pools[0]
		pool.Parallel.EnableExpertParallel = true
		pool.Parallel.DP = c.dp
		pool.Nodes *= c.dp
		in.Scenario.Cluster.Nodes *= c.dp
		k, err := newKernel(in)
		if err != nil {
			t.Fatalf("pcp=%d dp=%d: %v", c.pcp, c.dp, err)
		}
		want := pool.Parallel.TP * c.pcp * c.dp
		if got := k.Resolved().ExpertParallelWidth; got != want {
			t.Errorf("tp=%d pcp=%d dp=%d: the expert group is %d ranks, want %d "+
				"(tp x pcp x dp)", pool.Parallel.TP, c.pcp, c.dp, got, want)
		}
	}
}

// The width a consumer reads must be the resolved one and floor at one, so an absent field
// reads as "not split" rather than zero.
func TestPrefillContextParallelWidthComesFromTheResolvedLayout(t *testing.T) {
	if got := pcpKernel(t, pcpFixture, 4).PrefillContextParallelWidth(); got != 4 {
		t.Errorf("got %d, want 4", got)
	}
	if got := fixture(t, pcpFixture).PrefillContextParallelWidth(); got != 1 {
		t.Errorf("an unstated width read as %d, want 1", got)
	}
	// The two context axes are independent and must not read each other's field.
	k := pcpKernel(t, pcpFixture, 2)
	if got := k.DecodeContextParallelWidth(); got != 1 {
		t.Errorf("setting pcp moved the decode-context width to %d", got)
	}
}

// Changing how the step's token count is derived must not reprice a batch PCP does not
// touch, and the request that exposes this is the one the accumulation loop SKIPS.
//
// THE DEFECT THIS WAS WRITTEN AGAINST. Crediting the prefill split meant the step's token
// count could no longer be the batch's own, so a draft re-summed it from the per-request
// loop. That loop skips a request whose context is non-positive -- `Computed + Scheduled
// <= 0`, reachable with a negative Computed, which the schema documents as a real state
// ("an engine advances it optimistically at dispatch and rolls it back on speculative
// rejection") -- while Batch.Tokens() still counts its scheduled tokens. The re-sum
// dropped them from every term derived from the token count, repricing such a batch by
// 10% with PCP off entirely: 11.64 ms against 12.87 ms.
//
// The fix subtracts what the split WITHHOLDS rather than rebuilding the total, so a
// skipped request contributes exactly what it always did.
func TestTheStepTokenCountStillCountsARequestTheLoopSkips(t *testing.T) {
	k := fixture(t, pcpFixture)

	withSkipped := kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: 2, Computed: -2},
		{Scheduled: 4, Computed: 0, PromptLen: 4},
	}}
	withoutSkipped := kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: 4, Computed: 0, PromptLen: 4},
	}}
	if withSkipped.Tokens() != 6 || withoutSkipped.Tokens() != 4 {
		t.Fatalf("the batches must differ by the skipped request's tokens: %d and %d",
			withSkipped.Tokens(), withoutSkipped.Tokens())
	}

	with := k.StepTime(withSkipped).NoOverlap
	without := k.StepTime(withoutSkipped).NoOverlap
	if with <= without {
		t.Errorf("a batch of %d tokens priced %v, at or below the %v of a batch of %d; "+
			"the tokens of a request the accumulation loop skips are being dropped from "+
			"the step's token count", withSkipped.Tokens(), with, without,
			withoutSkipped.Tokens())
	}
	// And PCP must not change that: it withholds only what it actually splits, and it
	// splits nothing here, since neither request is in the prefill regime.
	for _, pcp := range []int{2, 8} {
		if got := pcpKernel(t, pcpFixture, pcp).StepTime(withSkipped).NoOverlap; got != with {
			t.Errorf("pcp=%d repriced a batch it splits nothing in: %v against %v",
				pcp, got, with)
		}
	}
}

// THE PCP GATHER CROSSES AT THE MODEL DTYPE. It runs before the cache write quantizes
// (vllm/model_executor/layers/attention/mla_attention.py:781-800 at v0.31.0; on a DSA
// layer, vllm/models/deepseek_v32/attention.py:511-529), so an fp8
// cache must not change what a prefill step's gather costs.
func TestThePCPGatherIgnoresTheCacheDType(t *testing.T) {
	prefill := decodeBatch(1, 4096, 4096)
	coll := func(cache string) float64 {
		in := pcpInputs(t, pcpFixture, 2)
		editEngine(t, &in, func(e *engineconfig.Engine) { e.CacheDType = cache })
		k, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		return collectiveSeconds(k.StepTime(prefill))
	}
	if a, b := coll("fp8"), coll("auto"); a != b {
		t.Errorf("an fp8 cache priced the prefill's collectives at %.6f ms against %.6f ms "+
			"for a bf16 one; the gather moves unquantized activations", a*1e3, b*1e3)
	}
}

// Moved from blis-latency-kernel/dcp_shard_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"math"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// Decode-context parallelism shards the KV cache by token, and these pin that
// behaviourally.
//
// THE DEFECT. Layout.DCP was assigned once and never read: over the whole repository
// `grep -rnE '\.DCP' --include='*.go'` returned the one assignment in the resolver. So two
// deployments differing only in dcp priced byte-identically -- on h200, 4 nodes, tp8/dp4
// with expert parallelism, dcp 1 and dcp 8 both gave 1104.4547515127522 tokens/sec,
// 186.05310625 ms TTFT and 6.880028693656094 ms ITL.
//
// WHY IT MATTERS MOST FOR LATENT ATTENTION. price.KVBytesPerToken divides KV heads by tp
// and floors at one head, so a model the catalog declares `n_kv: 1, d_h: 576` gets ZERO KV
// reduction from tensor parallelism: per-rank heads are 1 at tp=1 and still 1 at tp=8. DCP
// is the only axis that can shard such a cache, because it shards by token rather than by
// head. Nine catalog models declare an mla or sparse_mla layer.
//
// WHAT THE ENGINE DOES, which these tests encode (vllm at v0.31.0):
//
//	tokens divide, bytes per token do not   kv_cache_interface.py:545-550, :578-583
//	a sliding window is refused outright    kv_cache_interface.py:868-872
//	recurrent state is never sharded        kv_cache_interface.py:1097-1098
//	the per-rank length, with striping      attention/backends/utils.py:1143-1156
//	the ag_rs combine, three collectives    attention/ops/dcp.py:458, :493, :1593
//
// The fixtures are chosen for their layer kinds, which a probe confirmed: glm5-h200-tp8 is
// 78 sparse_mla layers, deepseek-v3 is 61 mla, and gpt-oss-120b is the hybrid -- 18 gqa
// against 18 swa -- which is the case a deployment-wide divisor would get wrong.
const (
	dcpMLAFixture    = "aisimulate/deepseek-v3-h200-fp8-sglang-tp8.yaml"
	dcpSparseFixture = "glm5-h200-tp8.yaml"
	dcpHybridFixture = "aisimulate/gpt-oss-120b-h200-fp4-vllm-tp8.yaml"
)

// dcpKernel builds a kernel from a committed fixture with one parallelism field changed.
// Mutating the documents before New is what lets a test vary a layout the fixtures do not
// carry: no committed scenario sets dcp, which is also why every existing test stayed
// green while the field was write-only.
func dcpKernel(t testing.TB, fixture string, dcp int) *Kernel {
	t.Helper()
	in := fixtureInputs(t, fixture)
	in.Deployment.Pools[0].Parallel.DCP = dcp
	k, err := newKernel(in)
	if err != nil {
		t.Fatalf("%s at dcp=%d: %v", fixture, dcp, err)
	}
	return k
}

// The shard is applied PER REQUEST at the call site, not as one ratio over the batch.
//
// The engine keeps a per-request vector of local lengths and sums it in the kernel
// (dcp_local_seq_lens, vllm/v1/attention/backend.py:427-428, built per request at
// flash_attn.py:884-893 of v0.31.0). Each sequence's shard rounds on its own, so a batch
// of many short contexts reads more than one long context of the same total -- every
// request pays its own ceiling.
//
// Measured here: eight requests of 1,001 tokens at dcp=4 read 8*ceil(1001/4) = 2,008
// positions, where one request of 8,008 reads ceil(8008/4) = 2,002. Same total context,
// six positions apart, and a batch-wide ratio would report 2,002 for both.
//
// AT THE CALL SITE RATHER THAN ON THE PURE FUNCTION, which is the point: an earlier draft
// of this change applied the shard as a batch-wide fraction multiplied into the already
// bounded token count. That is exact for a non-sparse layer and not in general -- searched
// over 200,000 random (width, top-k, compression, context) draws the two disagree by up to
// 4.3% on a heavily compressed sparse layer -- and no test on the pure function could see
// it, because the pure function was right either way. This asserts the composition.
func TestDCPShardsEachRequestSeparatelyAtTheCallSite(t *testing.T) {
	// A full-attention cache, so the only thing narrowing the read is the shard. A sparse
	// layer selects a different number of positions in the two batches below -- that is
	// its own bound doing its job -- which would mask the rounding this isolates. The
	// sparse composition is asserted in the subtest that follows.
	read := func(k *Kernel, b kernel.Batch) float64 {
		return k.StepTime(b).PerResource[kernel.ResourceHBM].Seconds()
	}
	sharded := dcpKernel(t, dcpMLAFixture, 4)
	whole := dcpKernel(t, dcpMLAFixture, 1)

	many := decodeBatch(8, 1, 1001)
	one := decodeBatch(1, 1, 8008)

	// Sharding lowers both reads. The SAVING is what differs: one long request rounds
	// once, eight short ones round eight times and so keep more positions.
	savedMany := read(whole, many) - read(sharded, many)
	savedOne := read(whole, one) - read(sharded, one)
	if savedMany <= 0 || savedOne <= 0 {
		t.Fatalf("sharding should lower both reads: savings of %.9f and %.9f s",
			savedMany, savedOne)
	}
	if savedOne <= savedMany {
		t.Errorf("one 8,008-token request saved %.9f s and eight 1,001-token requests "+
			"saved %.9f s; eight separate ceilings retain more positions than one, so "+
			"the single request must save more", savedOne, savedMany)
	}

	// The sparse composition, stated directly on the quantity the pricer uses: selection
	// first, then the shard, each per request. A batch-wide ratio over the selected total
	// would report the same number for both batches.
	var sparse *price.PlannedLayer
	ks := dcpKernel(t, dcpSparseFixture, 4)
	for i := range ks.plan.Layers {
		if ks.plan.Layers[i].AttnIndexTopK > 0 {
			sparse = &ks.plan.Layers[i]
			break
		}
	}
	if sparse == nil {
		t.Fatal("the sparse fixture carries no selecting layer")
	}
	eight := make([]int, 8)
	for i := range eight {
		eight[i] = 1001
	}
	perRequest := ks.dcpDecodeTokens(sparse, sparse.AttnKind, eight)
	batchWide := ks.dcpDecodeTokens(sparse, sparse.AttnKind, []int{8008})
	if perRequest <= batchWide {
		t.Errorf("eight selecting requests of 1,001 read %v positions and one of 8,008 "+
			"reads %v; each request selects and then shards on its own", perRequest, batchWide)
	}

	// THE CASE THAT SEPARATES THE TWO COMPOSITIONS, and the reason this is not a matter of
	// taste. Where the selection SATURATES -- a context past the top-k, so the layer reads
	// exactly top-k positions whatever the context -- the shard of that selection is
	// exact: ceil(2048/4) = 512. A ratio taken from the RAW contexts instead carries their
	// remainder into a quantity the remainder has nothing to do with: ceil(3298/4)/3298 is
	// 0.25015, and 2048 times that is 512.3105.
	//
	// The error is small and it is in the wrong direction for the wrong reason, which is
	// the kind that survives review. Asserted on an exact integer so there is nothing to
	// tune: a saturated selection sharded four ways is a whole number of positions.
	topk := sparse.AttnIndexTopK
	if sparse.AttnCompressRatio > 0 {
		t.Skipf("this fixture's sparse layer compresses (ratio %d), so its selection does "+
			"not saturate at the top-k and the exact figure below does not apply",
			sparse.AttnCompressRatio)
	}
	saturated := ks.dcpDecodeTokens(sparse, sparse.AttnKind, []int{topk + 1250})
	if want := float64((topk + 3) / 4); saturated != want {
		t.Errorf("a context past the top-k read %v positions at dcp=4, want exactly %v "+
			"(the top-k of %d sharded four ways); a shard taken from the raw context "+
			"rather than from the selection leaks the context's remainder",
			saturated, want, topk)
	}
}

// With the interleave unstated and no NIXL connector, the pricer must shard at the engine's
// DEFAULT of one token.
//
// cp_kv_cache_interleave_size defaults to 1 (vllm/config/parallel.py:393 at v0.31.0):
// "Interleave_size=1: token-level alignment, where token `i` is stored on dcp_rank
// `i % dcp_world_size`." At that value the slowest rank holds exactly ceil(context/dcp).
//
// Pinned as an exact integer per request, because a larger run shifts the figure by up to
// one run per request and a step time would not show it: at dcp=4 a 1,001-token context
// reads 251 positions at the default and 256 at a block-aligned 16, and both look equally
// plausible. The stated and NIXL-pinned cases are in dcp_knobs_test.go.
func TestDCPShardsAtTheEngineDefaultTokenInterleave(t *testing.T) {
	k := dcpKernel(t, dcpMLAFixture, 4)
	var full *price.PlannedLayer
	for i := range k.plan.Layers {
		if k.plan.Layers[i].AttnQHeads > 0 && !readBounded(&k.plan.Layers[i]) {
			full = &k.plan.Layers[i]
			break
		}
	}
	if full == nil {
		t.Fatal("this fixture carries no non-selecting attention layer")
	}
	for _, c := range []struct {
		contexts []int
		want     float64
	}{
		{[]int{1001}, 251},       // ceil(1001/4); at interleave 16 it would be 256
		{[]int{8008}, 2002},      // ceil(8008/4)
		{[]int{1001, 1001}, 502}, // each request rounds on its own
		{[]int{4096}, 1024},      // an exact multiple leaves no remainder
		{[]int{1, 1, 1, 1}, 4},   // one token each, one position each
		{[]int{32768, 1001}, 8192 + 251},
	} {
		if got := k.dcpDecodeTokens(full, full.AttnKind, c.contexts); got != c.want {
			t.Errorf("contexts %v at dcp=4: read %v positions, want %v",
				c.contexts, got, c.want)
		}
	}
}

// The capacity shard must agree with the engine's LOGICAL BLOCK, which DCP widens.
//
// vLLM scales the scheduler's block to its token span under DCP:
// resolve_dcp_kv_block_size returns `spec.block_size * dcp_world_size` for a sharded spec
// (vllm/v1/core/kv_cache_utils.py:717-719 at v0.31.0). So a sequence is allocated in units
// of block_size*dcp tokens, of which each rank stores block_size -- which is why a real
// deployment keeps the product in sync with its router's block size, the GLM-5.3-Flash
// canary running KV_BLOCK_SIZE 64 at DCP_SIZE 8 for a 512-token logical block.
//
// Sharding the token count and THEN paging at block_size is the same arithmetic -- the
// integer identity cdiv(cdiv(L, dcp), bs) == cdiv(L, bs*dcp) -- and this pins that the two
// readings agree, so a later change cannot drift from the allocator's granularity. The
// lengths below straddle a logical block boundary at 512, where a form that paged before
// sharding would disagree.
func TestDCPCapacityAgreesWithTheEnginesLogicalBlock(t *testing.T) {
	in := fixtureInputs(t, dcpMLAFixture)
	in.Deployment.Pools[0].Parallel.DCP = 8
	editEngine(t, &in, func(e *engineconfig.Engine) { e.BlockSize = 64 })
	k, err := newKernel(in)
	if err != nil {
		t.Fatalf("dcp=8 at block size 64: %v", err)
	}
	const blockSize, dcp = 64, 8
	for _, tokens := range []int{1, 64, 100, 511, 512, 513, 1000, 4096, 32768} {
		// The engine's reading: whole logical blocks of blockSize*dcp tokens, each
		// leaving this rank blockSize tokens to store.
		logical := (tokens + blockSize*dcp - 1) / (blockSize * dcp)
		want := price.PagedBytes(logical*blockSize, blockSize, k.kvBytesPerToken)
		if got := k.SequenceVariableBytes(tokens); got != want {
			t.Errorf("%d tokens at dcp=8, block 64: holds %d bytes, want %d (%d logical "+
				"block(s) of %d tokens, %d of them on this rank)",
				tokens, got, want, logical, blockSize*dcp, blockSize)
		}
	}
}

// The structural check: a shard the pricer ignores is the defect this change fixes, so
// poke the resolved layout on a built kernel and require the price to move. This is what
// catches a field that is plumbed but never read -- which is exactly the state DCP was in.
func TestDCPGeometryReachesThePricer(t *testing.T) {
	k := fixture(t, dcpMLAFixture)
	batch := decodeBatch(32, 1, 32768)
	before := k.StepTime(batch).NoOverlap
	k.layout.DCP = 8
	after := k.StepTime(batch).NoOverlap
	if after >= before {
		t.Errorf("setting the resolved DCP width to 8 left the step at %v against %v; "+
			"the layout field is not reaching the pricer", after, before)
	}
}

// Sharding a sequence means no rank holds the whole context, so each computes a partial
// attention output and the group must combine them. That combine is what DCP trades for
// the capacity it buys, and omitting it would make DCP look free in both directions.
//
// THE PAYLOAD IS PER DECODE TOKEN, one query row per token the decode regime runs (the MQA
// rows, num_mqa_tokens). A prefill token takes no part in a decode combine, so the charge
// must be independent of how many prefill tokens share the step; and a speculative decode
// verifying two tokens moves two rows.
//
// A pure decode batch of one-token requests cannot show either, because there the request
// count, the decode-token count and the step's token count are one number -- which is why
// the discriminating cases are a MIXED batch (two decode rows beside a 2,048-token prefill)
// and a SPECULATIVE one (two requests of two tokens against four requests of one).
func TestDCPAddsACombineCollectiveSizedByDecodeTokens(t *testing.T) {
	link := func(dcp int, b kernel.Batch) float64 {
		return dcpKernel(t, dcpMLAFixture, dcp).StepTime(b).
			PerResource[kernel.ResourceNVLink].Seconds()
	}
	added := func(b kernel.Batch) float64 { return link(2, b) - link(1, b) }
	// Two decode rows, and the same two beside a prefill chunk.
	pure := decodeBatch(2, 1, 32768)
	mixed := kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: 1, Computed: 32767, PromptLen: 32768},
		{Scheduled: 1, Computed: 32767, PromptLen: 32768},
		{Scheduled: 2048, Computed: 0, PromptLen: 2048},
	}}
	if pure.Tokens() != 2 || mixed.Tokens() != 2050 {
		t.Fatalf("the batches must differ sharply in token count to discriminate: "+
			"got %d and %d", pure.Tokens(), mixed.Tokens())
	}

	// A combine must appear at all.
	if on, off := link(2, pure), link(1, pure); on <= off {
		t.Fatalf("the on-node collective term is %.6f ms at dcp=2 against %.6f ms at "+
			"dcp=1; a DCP combine must add communication", on*1e3, off*1e3)
	}

	// And it must be the SAME charge in both batches, since both hold two decode rows.
	// Charging the step's token count would make the mixed figure 1,025 times larger.
	// Compared to 2ns: each resource total is reported as a time.Duration, so a difference
	// of two totals carries up to a nanosecond of truncation from each.
	addedPure, addedMixed := added(pure), added(mixed)
	if math.Abs(addedMixed-addedPure) > 2e-9 {
		t.Errorf("the combine cost %.6f ms beside a prefill chunk against %.6f ms "+
			"without one, on the same two decode rows; a prefill token does not join a "+
			"decode combine", addedMixed*1e3, addedPure*1e3)
	}

	// Rows are TOKENS. Two speculative requests verifying two tokens each run four query
	// rows, exactly as four one-token decodes do, and the step's token count is four in
	// both, so every other collective is the same too.
	spec := decodeBatch(2, 2, 32768)
	four := decodeBatch(4, 1, 32768)
	if a, b := added(spec), added(four); math.Abs(a-b) > 2e-9 {
		t.Errorf("two requests of two decode tokens added %.6f ms of combine against "+
			"%.6f ms for four requests of one; a combine moves one row per decode token",
			a*1e3, b*1e3)
	}

	// Doubling the decode rows must double the BYTES, which is a weaker statement than
	// doubling the charge: a collective is floor + bytes/rate, and each of the three
	// launches pays its floor however small the payload. Subtracting the floors isolates
	// the half that scales. ag_rs launches a query all-gather, a log-sum-exp all-gather and
	// an output reduce-scatter, so the floors are two all-gathers and one reduce-scatter.
	//
	// Asserting the total would have been wrong, and was: an earlier draft of this test
	// required the total to double and failed against correct code.
	at2 := dcpKernel(t, dcpMLAFixture, 2)
	floors := float64(at2.plan.TotalLayers) *
		(2*at2.collectiveFloors[collKey{Op: model.OpAllGather, Group: price.GroupDCP}].Seconds() +
			at2.collectiveFloors[collKey{Op: model.OpReduceScatter, Group: price.GroupDCP}].Seconds())
	bytesAtTwo := addedPure - floors
	bytesAtFour := added(four) - floors
	if bytesAtTwo <= 0 {
		t.Fatalf("the combine's byte term is %.6f ms, not positive above its floors",
			bytesAtTwo*1e3)
	}
	// Exactly 2 up to the reporting resolution: each step's resource totals are whole
	// nanoseconds, and each byte term is a difference of two of them less a floor sum, so
	// it carries a few nanoseconds of truncation.
	if diff := bytesAtFour - 2*bytesAtTwo; math.Abs(diff) > 6e-9 {
		t.Errorf("doubling the decode rows scaled the combine's byte term by %.6f, want "+
			"exactly 2: one query row crosses per decode token", bytesAtFour/bytesAtTwo)
	}

	// A prefill-only step must carry no DCP combine at all.
	prefill := decodeBatch(1, 4096, 4096)
	if sharded, bare := link(8, prefill), link(1, prefill); sharded != bare {
		t.Errorf("a prefill-only step carried %.6f ms of collective at dcp=8 against "+
			"%.6f ms at dcp=1", sharded*1e3, bare*1e3)
	}
}

// The DCP collectives must resolve coefficients at THEIR OWN width, not the
// tensor-parallel one.
//
// This is what the (op, group) coefficient key exists for. A DCP all-gather and a
// tensor-parallel all-gather are the same primitive at two widths, and a single entry per
// op would have silently priced the narrow group at the wide group's floor. The gap is not
// a rounding error: at tp=8 and dcp=2 on h200 the two floors are 10.3us and 5.08us.
//
// Read from the resolved maps rather than from Provenance, which reports every coefficient
// the set carries rather than the ones this layout selected.
func TestTheDCPCollectivesArePricedAtTheirOwnWidthNotTheTensorParallelOne(t *testing.T) {
	k := dcpKernel(t, dcpMLAFixture, 2)
	if k.layout.TP != 8 {
		t.Fatalf("this fixture is expected to be tp=8, got %d: the test needs the two "+
			"widths to differ", k.layout.TP)
	}
	tp := k.collectiveFloors[collKey{Op: model.OpAllGather, Group: price.GroupTP}]
	dcp, ok := k.collectiveFloors[collKey{Op: model.OpAllGather, Group: price.GroupDCP}]
	if !ok || dcp <= 0 {
		t.Fatal("no all-gather floor resolved for the decode-context group")
	}
	if _, ok := k.collectiveFloors[collKey{
		Op: model.OpReduceScatter, Group: price.GroupDCP}]; !ok {
		t.Error("no reduce-scatter floor resolved for the decode-context group; the " +
			"ag_rs combine is a gather AND a scatter")
	}
	if dcp == tp {
		t.Errorf("the 2-rank DCP all-gather resolved the same floor as the 8-rank "+
			"tensor-parallel one (%v); the group is not selecting the width", dcp)
	}
	if dcp >= tp {
		t.Errorf("a 2-rank all-gather floor of %v is not below the 8-rank %v", dcp, tp)
	}
	if got, ok := k.groupSize(price.GroupDCP); !ok || got != 2 {
		t.Errorf("the decode-context group spans %d ranks, want 2", got)
	}

	// RESOLVING THE RIGHT TRIPLE IS NOT ENOUGH: the pricing path has to USE it. A
	// mutation keying the combine to the tensor-parallel axis leaves every assertion
	// above green while charging the narrow group the wide group's floor, which is the
	// precise failure the composite key exists to prevent.
	//
	// So price the combine and compare it against the floors it should have been built
	// from. Three collectives per layer under ag_rs -- two all-gathers and a
	// reduce-scatter -- each paying its own floor, which is the dominant term at this
	// payload. The 2-rank and 8-rank floor triples differ by about 2x on this part, the
	// same ratio that makes borrowing a width wrong in the first place.
	batch := decodeBatch(2, 1, 32768)
	link := func(width int) float64 {
		return dcpKernel(t, dcpMLAFixture, width).StepTime(batch).
			PerResource[kernel.ResourceNVLink].Seconds()
	}
	perLayer := (link(2) - link(1)) / float64(k.plan.TotalLayers)
	floorTriple := 2*dcp.Seconds() + k.collectiveFloors[collKey{
		Op: model.OpReduceScatter, Group: price.GroupDCP}].Seconds()
	tpTriple := 2*tp.Seconds() + k.collectiveFloors[collKey{
		Op: model.OpReduceScatter, Group: price.GroupTP}].Seconds()
	if perLayer < floorTriple {
		t.Errorf("the combine costs %.4fus per layer, below the %.4fus of floor its three "+
			"collectives must each pay", perLayer*1e6, floorTriple*1e6)
	}
	// The decisive bound: it must sit nearer the narrow triple than the wide one.
	if math.Abs(perLayer-floorTriple) >= math.Abs(perLayer-tpTriple) {
		t.Errorf("the combine costs %.4fus per layer, nearer the 8-rank floor triple "+
			"%.4fus than the 2-rank %.4fus; the decode-context collectives are being "+
			"priced at the tensor-parallel width",
			perLayer*1e6, tpTriple*1e6, floorTriple*1e6)
	}
}

// A deployment without DCP must not be REFUSED for want of a decode-context coefficient
// it has no use for.
//
// This is the one way the (op, group) key could regress a deployment that has nothing to
// do with DCP. An unmeasured width inside the search space is an error rather than a
// clamp, by design -- borrowing a narrower width's floor understates a collective by
// 1.53x to 1.91x across the parts measured at both -- so asking for a width a part lacks
// turns a working deployment into a construction failure.
//
// Stated as the OBSERVABLE consequence rather than as the absence of a map entry, because
// the absence has no consequence a caller can see on a registry where every part carries
// every width, which every part in this one does. So the test builds a coefficient set
// that is deliberately missing the narrow width and requires construction to succeed
// anyway while dcp is off, and to refuse once dcp asks for the width that is missing.
func TestADeploymentWithoutDCPIsNotRefusedForADCPCoefficient(t *testing.T) {
	// A part swept at 8 ranks only: a tensor-parallel group of 8 resolves, a 2-rank
	// decode-context group has nothing to resolve against.
	var entries []coefficient.Entry
	for _, op := range []string{"all_reduce", "all_gather", "reduce_scatter", "alltoall"} {
		entries = append(entries, tripleFor(op, "fp16", "h200", 8, 9.37)...)
	}
	c := coeffs(t, entries)
	k := &Kernel{chip: hardware.Chip{Name: "h200"}}
	k.layout.TP = 8

	// Tensor-parallel: resolves, because 8 is measured.
	if _, err := k.groupWidth(c, collKey{Op: model.OpAllGather, Group: price.GroupTP},
		"all_gather", "fp16", "h200"); err != nil {
		t.Errorf("a tensor-parallel group of 8 should resolve against an 8-rank sweep: %v",
			err)
	}
	// A decode-context group NARROWER than anything measured clamps to the narrowest
	// figure, which is the existing policy for a group below the sweep range and is the
	// conservative direction: the narrowest measured floor is the only one available, and
	// a narrower group cannot cost more than a wider one. What the resolver refuses is a
	// width INSIDE the search space that a part happens to lack, which is the case where
	// a fitted figure could exist and does not.
	k.layout.DCP = 2
	got, err := k.groupWidth(c, collKey{Op: model.OpAllGather, Group: price.GroupDCP},
		"all_gather", "fp16", "h200")
	if err != nil {
		t.Errorf("a 2-rank decode-context group below the sweep range should clamp to "+
			"the narrowest measured width, not refuse: %v", err)
	} else if got != 8 {
		t.Errorf("clamped to %d ranks; the only measured width is 8", got)
	}

	// And the refusal that does matter: a width inside the search space the part lacks.
	// Swept at 2 and 16 only, a 4-rank decode-context group must be refused rather than
	// priced at the 2-rank floor.
	var gapped []coefficient.Entry
	for _, op := range []string{"all_reduce", "all_gather", "reduce_scatter", "alltoall"} {
		for _, w := range []int{2, 16} {
			gapped = append(gapped, tripleFor(op, "fp16", "h200", w, 9.37)...)
		}
	}
	k.layout.DCP = 4
	if _, err := k.groupWidth(coeffs(t, gapped),
		collKey{Op: model.OpAllGather, Group: price.GroupDCP},
		"all_gather", "fp16", "h200"); err == nil {
		t.Error("a 4-rank decode-context group resolved against a part swept at 2 and " +
			"16 only; borrowing a narrower width's floor understates a collective by " +
			"1.53x to 1.91x and is the substitution liftCollectiveFloors refuses")
	}

	// And the behaviour that matters for every deployment already in the corpus: with DCP
	// off, a real kernel constructs and prices. If the DCP triples were resolved
	// unconditionally, a part lacking one of those widths would fail here instead.
	for _, scenario := range []string{
		dcpMLAFixture, dcpSparseFixture, dcpHybridFixture,
		"minimax-m25-h200-ep8.yaml", "minimax-m25-h200-ep72.yaml",
		"minimax-m25-gb300-tp4.yaml", "nemotron3-ultra-h100-agg.yaml",
		"kimi-k3-h100-nospec.yaml",
	} {
		if got := fixture(t, scenario).StepTime(decodeBatch(8, 1, 2048)).NoOverlap; got <= 0 {
			t.Errorf("%s priced a step at %v", scenario, got)
		}
	}
}

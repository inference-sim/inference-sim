// Moved from blis-latency-kernel/methods_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/vocab"
)

// These tests assert BEHAVIOUR rather than arithmetic: what must hold of a method's
// answers across inputs, which is what a caller depends on and what survives a change to
// any single constant. A test that pinned a millisecond figure would fail on every
// coefficient update while proving nothing about whether the method is right.

// --- Memory occupancy ---------------------------------------------------------

func TestFixedBytesAccountsForTheWholeModelAcrossRanks(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	fixed := k.FixedBytes()
	if fixed.Weights <= 0 {
		t.Fatalf("no weight bytes reported")
	}
	// The fixture is a ~230B-parameter MoE served fp8, so roughly 230 GB across the whole
	// deployment. One rank of 16 holds its expert shard plus a TP slice of the dense
	// layers, which must land within an order of magnitude of 230/16 GB. A rank holding
	// the whole model, or a thousandth of it, means a sharding term is wrong.
	perRankGB := float64(fixed.Weights) / 1e9
	if perRankGB < 5 || perRankGB > 40 {
		t.Errorf("one rank of a 16-way deployment holds %.1f GB of a ~230 GB model; "+
			"that is outside any plausible sharding", perRankGB)
	}
	if fixed.ActivationPeak <= 0 {
		t.Error("no activation scratch reported; a batch needs somewhere to land")
	}
	if k.PreSnapshotBytes() <= 0 {
		t.Error("a multi-rank deployment reserves communicator buffers")
	}
}

// WIDER EXPERT PARALLELISM HOLDS FEWER WEIGHTS PER RANK, by exactly the busiest rank's
// share. The three minimax-m2.5 layouts all run tp=8, so a rank's dense weights are the same
// on each and its weights are D + n*E, with n the experts its BUSIEST rank holds: vLLM gives
// the first experts % width ranks one more (vllm/model_executor/layers/fused_moe/
// expert_map_manager.py:67-69 at v0.31.0), and that rank's memory fills first. 256 experts
// over 8, 16 and 72 ranks is 32, 16 and ceil(256/72) = 4. So with D and E unknown,
//
//	(W8 - W72) / (W8 - W16) = (32 - 4) / (32 - 16)
//
// exactly. Pricing the lighter rank (3 experts) gives 29/16 instead, and an expert term not
// sharded by EP gives 0/0.
func TestWiderExpertParallelismHoldsFewerWeightsPerRank(t *testing.T) {
	weights := func(f string) float64 {
		t.Helper()
		return float64(fixture(t, f).FixedBytes().Weights)
	}
	w8, w16, w72 := weights("minimax-m25-h200-ep8.yaml"),
		weights("minimax-m25-h200-ep16.yaml"), weights("minimax-m25-h200-ep72.yaml")
	in := fixtureInputs(t, "minimax-m25-h200-ep8.yaml")
	experts := 0
	for _, lk := range in.Model.LayerKinds {
		for _, n := range lk.Nodes {
			if n.Op == model.OpGroupedGEMM {
				experts = max(experts, n.Experts)
			}
		}
	}
	busiest := func(width int) float64 { return float64((experts + width - 1) / width) }
	want := (busiest(8) - busiest(72)) / (busiest(8) - busiest(16))
	if got := (w8 - w72) / (w8 - w16); math.Abs(got-want) > 1e-6 {
		t.Errorf("(W8 - W72)/(W8 - W16) = %.6f, want %.6f: per-rank weights must fall by "+
			"the busiest rank's expert count (%v, %v, %v experts at ep 8, 16, 72)",
			got, want, busiest(8), busiest(16), busiest(72))
	}
}

func TestSequenceVariableBytesGrowsWithContextAndQuantizesToPages(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	if k.SequenceVariableBytes(0) != 0 {
		t.Error("a sequence with no tokens occupies no KV")
	}
	prev := int64(0)
	for _, tokens := range []int{1, 16, 17, 100, 1000, 8192} {
		got := k.SequenceVariableBytes(tokens)
		if got < prev {
			t.Errorf("KV at %d tokens (%d) is below the previous step (%d)",
				tokens, got, prev)
		}
		prev = got
	}
	// A page is the allocation unit, so one token and a full page cost the same.
	if k.SequenceVariableBytes(1) != k.SequenceVariableBytes(16) {
		t.Errorf("1 token costs %d and 16 cost %d; with a 16-token page they are the "+
			"same allocation", k.SequenceVariableBytes(1), k.SequenceVariableBytes(16))
	}
	// And crossing into a second page must cost strictly more.
	if k.SequenceVariableBytes(17) <= k.SequenceVariableBytes(16) {
		t.Error("crossing a page boundary did not allocate another page")
	}
}

func TestSequenceFixedBytesIsZeroForAPureAttentionModel(t *testing.T) {
	// The fixture is MoE but not hybrid: it carries no recurrent state, so there is no
	// per-sequence occupancy independent of context length.
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	if got := k.SequenceFixedBytes(); got != 0 {
		t.Errorf("a pure-attention model reported %d bytes of recurrent state", got)
	}
}

func TestStepTimeIsZeroWorkForAnEmptyBatchButNotFree(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	e := k.StepTime(kernel.Batch{})
	if e.Overlap <= 0 {
		t.Error("an empty step still costs the host its per-step work")
	}
	if e.Bottleneck != kernel.ResourceHost {
		t.Errorf("an empty step is host-bound, not %s", e.Bottleneck)
	}
}

func TestStepTimeGrowsWithTokensAndWithContext(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	// Compared on the HBM term rather than on Overlap. At small batch every collective
	// is floor-bound and the floors dominate the step, so Overlap is flat in context
	// there — which is correct behaviour, not a defect, and asserting on Overlap would
	// make this test demand the wrong thing.
	small := k.StepTime(decodeBatch(8, 1, 1024))
	moreTokens := k.StepTime(decodeBatch(64, 1, 1024))
	moreContext := k.StepTime(decodeBatch(8, 1, 16384))
	if moreTokens.PerResource[kernel.ResourceHBM] <=
		small.PerResource[kernel.ResourceHBM] {
		t.Errorf("8x the requests did not read more memory: %v against %v",
			moreTokens.PerResource[kernel.ResourceHBM],
			small.PerResource[kernel.ResourceHBM])
	}
	if moreContext.PerResource[kernel.ResourceHBM] <=
		small.PerResource[kernel.ResourceHBM] {
		t.Errorf("16x the context did not read more memory: %v against %v",
			moreContext.PerResource[kernel.ResourceHBM],
			small.PerResource[kernel.ResourceHBM])
	}
	// And a batch large enough to leave the floor-dominated regime must move the step.
	big := k.StepTime(decodeBatch(512, 1, 16384))
	if big.Overlap <= moreTokens.Overlap {
		t.Errorf("a far larger batch did not cost more: %v against %v",
			big.Overlap, moreTokens.Overlap)
	}
}

func TestOverlapNeverExceedsNoOverlap(t *testing.T) {
	// The two bracket the estimate: the max over resources cannot exceed their sum.
	// Property-checked over a spread of shapes rather than asserted on one.
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	for _, b := range []kernel.Batch{
		{}, decodeBatch(1, 1, 1), decodeBatch(1, 2048, 2048),
		decodeBatch(256, 2, 8192), decodeBatch(4, 2048, 32768),
	} {
		e := k.StepTime(b)
		if e.Overlap > e.NoOverlap {
			t.Errorf("overlap %v exceeds no-overlap %v at %d tokens",
				e.Overlap, e.NoOverlap, b.Tokens())
		}
		// Bottleneck names the resource that did the most work, which with per-stage
		// composition is not the resource whose total equals Overlap — no resource's
		// total does, since Overlap is a sum of per-stage maxima. What must hold is that
		// no other resource did more.
		for r, d := range e.PerResource {
			if d > e.PerResource[e.Bottleneck] {
				t.Errorf("%s did %v of work but %s is named the bottleneck at %v",
					r, d, e.Bottleneck, e.PerResource[e.Bottleneck])
			}
		}
		// And Overlap must sit between the largest single resource and the sum: it
		// cannot be less than the busiest resource, nor more than everything serialized.
		if e.Overlap < e.PerResource[e.Bottleneck] {
			t.Errorf("overlap %v is below the busiest resource's %v",
				e.Overlap, e.PerResource[e.Bottleneck])
		}
	}
}

func TestStepTimeIsPureAndSafeUnderConcurrency(t *testing.T) {
	// Purity is the interface's central claim: it is what lets a simulator call this at
	// any simulated instant from any goroutine. Repeated calls must agree exactly, and
	// concurrent calls on one kernel -- every method, on batches that reach every pricing
	// path -- must agree with the serial ones. CI runs this test under -race, which is the
	// only place this repository starts goroutines, so it is also the race check.
	kernels := []*Kernel{
		fixture(t, "minimax-m25-h200-ep8.yaml"),
		mustDCPVariant(t, dcpMLAFixture, 8, 1, 2, nil),
		mustDCPVariant(t, dcpSparseFixture, 2, 4, 4, agRS),
	}
	decode := decodeBatch(37, 2, 4096)
	batches := []kernel.Batch{
		decode,
		decodeBatch(1, 3000, 3000),
		{DecodeThreshold: 8, Reqs: append(append([]kernel.ReqShape{}, decode.Reqs...),
			kernel.ReqShape{Scheduled: 512, Computed: 2048, PromptLen: 2560})},
		{},
	}
	type answer struct {
		step     kernel.StepEstimate
		fixed    kernel.MemoryBreakdown
		variable int64
		origins  int
		// Compared by value with reflect.DeepEqual, which follows pointers: Deployment
		// returns a fresh deep copy on every call, by design, so its pointers differ.
		resolved, deployment any
	}
	ask := func(k *Kernel, b kernel.Batch) answer {
		return answer{
			step: k.StepTime(b), fixed: k.FixedBytes(),
			variable: k.SequenceVariableBytes(4096), origins: len(k.Provenance()),
			resolved: k.Resolved(), deployment: k.Deployment(),
		}
	}
	for ki, k := range kernels {
		for bi, b := range batches {
			want := ask(k, b)
			results := make(chan answer, 16)
			for i := 0; i < 16; i++ {
				go func() { results <- ask(k, b) }()
			}
			for i := 0; i < 16; i++ {
				if got := <-results; !reflect.DeepEqual(got, want) {
					t.Fatalf("kernel %d, batch %d: a concurrent call gave %+v, the serial "+
						"call %+v", ki, bi, got, want)
				}
			}
		}
	}
}

func TestTierTimeOrdersTiersBySpeedAndDirection(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	const bytes = 64 << 20
	dram := k.TierTime("cpu_dram", kernel.DirectionFromTier, bytes, 1)
	nvme := k.TierTime("nvme_gen4", kernel.DirectionFromTier, bytes, 1)
	object := k.TierTime("s3", kernel.DirectionFromTier, bytes, 1)
	if !(dram < nvme && nvme < object) { //nolint:staticcheck // moved verbatim from blis-latency-kernel
		t.Errorf("tiers out of order: dram %v, nvme %v, s3 %v", dram, nvme, object)
	}
	// Writes are slower than reads on the NVMe classes, which is what the catalog says.
	read := k.TierTime("nvme_gen4", kernel.DirectionFromTier, bytes, 1)
	write := k.TierTime("nvme_gen4", kernel.DirectionToTier, bytes, 1)
	if write <= read {
		t.Errorf("an NVMe write (%v) should cost more than a read (%v)", write, read)
	}
}

func TestTierTimeChargesForConcurrencyAndForSize(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	one := k.TierTime("nvme_gen4", kernel.DirectionFromTier, 64<<20, 1)
	eight := k.TierTime("nvme_gen4", kernel.DirectionFromTier, 64<<20, 8)
	if eight <= one {
		t.Errorf("eight concurrent readers (%v) did not cost more than one (%v)",
			eight, one)
	}
	small := k.TierTime("nvme_gen4", kernel.DirectionFromTier, 4<<10, 1)
	large := k.TierTime("nvme_gen4", kernel.DirectionFromTier, 64<<20, 1)
	if large <= small {
		t.Error("a 64 MiB read did not cost more than a 4 KiB one")
	}
}

func TestAnUnknownTierIsNotAFreeTransfer(t *testing.T) {
	// Returning zero would make an unmodelled tier look costless, which is the opposite
	// of what not knowing means.
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	got := k.TierTime("tier-that-does-not-exist", kernel.DirectionFromTier, 1<<20, 1)
	known := k.TierTime("s3", kernel.DirectionFromTier, 1<<20, 1)
	if got <= known {
		t.Errorf("an unknown tier priced %v, no worse than the slowest known tier's "+
			"%v", got, known)
	}
}

func TestPDTransferGrowsWithTokensAndCostsMoreAcrossNodes(t *testing.T) {
	// ep72 rather than the EP=8 fixture: this compares an on-node transfer against a
	// cross-node one, so it needs a deployment that declares a fabric. The single-node
	// EP=8 fixture has none, and with no inter-node rate the two placements price alike.
	k := fixture(t, "minimax-m25-h200-ep72.yaml")
	onNode := kernel.Placement{Node: 0, Rack: 0}
	sameNode := kernel.Placement{Node: 0, Rack: 0}
	otherNode := kernel.Placement{Node: 1, Rack: 0}

	if k.PDTransferTime(0, onNode, otherNode) != 0 {
		t.Error("a transfer of no tokens costs nothing")
	}
	short := k.PDTransferTime(1024, onNode, otherNode)
	long := k.PDTransferTime(32768, onNode, otherNode)
	if long <= short {
		t.Errorf("32x the tokens did not cost more: %v against %v", long, short)
	}
	local := k.PDTransferTime(32768, onNode, sameNode)
	if local >= long {
		t.Errorf("an on-node transfer (%v) should beat a cross-node one (%v)",
			local, long)
	}
}

func TestHostOverheadsScaleWithWhatTheyProcess(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	if k.AdmissionOverhead(0) != 0 {
		t.Error("admitting an empty prompt costs nothing")
	}
	short, long := k.AdmissionOverhead(100), k.AdmissionOverhead(10000)
	if long <= short {
		t.Errorf("a 100x longer prompt did not cost more to admit: %v against %v",
			long, short)
	}
	// Admission is an intercept plus a per-token slope, so the DIFFERENCE over a token
	// range is the slope alone and the ratio is not the token ratio. This test asserted
	// a 90-110x ratio while the registry shipped no intercept; it now pins the two
	// components separately, which is what distinguishes them.
	perToken := float64(long-short) / float64(10000-100)
	if perToken <= 0 {
		t.Errorf("per-token admission slope is not positive: %v", perToken)
	}
	if intercept := float64(short) - perToken*100; intercept < 0 {
		t.Errorf("implied admission intercept is negative: %v", intercept)
	}
	if k.OutputTokenOverhead() <= 0 {
		t.Error("emitting a token costs host time")
	}
	if k.CompletionOverhead() <= 0 {
		t.Error("finishing a request costs host time")
	}
}

func TestProvenanceNamesEveryCoefficientAndItsEvidence(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	origins := k.Provenance()
	if len(origins) == 0 {
		t.Fatal("no coefficients reported")
	}
	for _, o := range origins {
		if o.Name == "" || o.Method == "" || o.Set == "" {
			t.Errorf("incomplete origin: %+v", o)
		}
	}
	measured, total, assumed := k.Evidence()
	if total != len(origins) {
		t.Errorf("Evidence counts %d coefficients, Provenance lists %d",
			total, len(origins))
	}
	if measured == 0 {
		t.Error("no coefficient rests on a measurement")
	}

	// What Evidence() owes a caller is that the assumed set is REPORTED and that it is a
	// subset of what Provenance lists, so a prediction can state its own footing. It does
	// not owe any particular term being measured: `assumed` is a first-class method in the
	// schema's vocabulary, and which terms carry it is the registry's call, not this
	// kernel's.
	//
	// This used to require every assumed coefficient to be named host_*, on the reasoning
	// that host overheads are the only terms no public dataset measures. That was a true
	// observation about one registry snapshot written as a permanent rule, and it broke as
	// soon as the registry gained assumed collective entries for a part whose sweeps do not
	// exist yet (gb300 carries eighteen). The kernel prices them without complaint -- it
	// reads Method only to COUNT evidence, never to accept or reject a coefficient -- so a
	// test that failed on them was asserting something the kernel does not do, against a
	// registry that is expected to grow.
	byName := make(map[string]string, len(origins))
	for _, o := range origins {
		byName[o.Name] = o.Method
	}
	for _, name := range assumed {
		method, listed := byName[name]
		if !listed {
			t.Errorf("Evidence reports %q as assumed but Provenance does not list it; "+
				"the two views must describe one resolved set", name)
			continue
		}
		if method != string(vocab.MethodAssumed) {
			t.Errorf("Evidence reports %q as assumed where Provenance calls it %q",
				name, method)
		}
	}
	// Measured and assumed cannot overlap, and neither can exceed the total.
	if measured+len(assumed) > total {
		t.Errorf("%d measured plus %d assumed exceeds %d coefficients; a coefficient "+
			"cannot be counted in both", measured, len(assumed), total)
	}
}

// TestAssumedCoefficientsArePricedAndReported is the property the registry's design
// depends on: an assumed coefficient is usable.
//
// `assumed` is one of six methods in the schema's vocabulary and the registry uses it for
// terms no sweep covers yet -- host overheads throughout, and the collective entries for a
// part whose comm sweeps have not landed. A consumer that refused them would make those
// entries unusable and force the registry to misreport a reasoned estimate as a
// measurement to get a prediction at all, which is the exact confusion the method axis
// exists to prevent.
//
// So this pins the two halves of the contract together: the kernel PRICES an assumed
// coefficient, and it REPORTS it as assumed. Either alone would be wrong -- silently
// pricing one hides the footing, refusing one discards a deliberate registry decision.
func TestAssumedCoefficientsArePricedAndReported(t *testing.T) {
	// Host overheads are assumed in every registry snapshot, by the nature of the term:
	// no public dataset measures an engine's per-request Python work. Using them rather
	// than a synthetic set keeps this honest about the committed registry.
	k := fixture(t, "minimax-m25-h200-ep8.yaml")

	measured, total, assumed := k.Evidence()
	if len(assumed) == 0 {
		t.Skip("no assumed coefficient in the resolved set; nothing to check here")
	}
	// Priced: the kernel built, and a step costs something. A kernel that refused assumed
	// coefficients would have failed in New() before reaching this line.
	if e := k.StepTime(decodeBatch(32, 1, 4096)); e.NoOverlap <= 0 {
		t.Error("a deployment resting partly on assumed coefficients priced to nothing")
	}
	// Reported: the count is visible and the set is not silently folded into measured.
	if measured >= total {
		t.Errorf("%d of %d coefficients counted as measured while %d are assumed; an "+
			"assumed term must not be reported as evidenced", measured, total, len(assumed))
	}
	t.Logf("priced with %d measured and %d assumed of %d coefficients",
		measured, len(assumed), total)
}

// TestAssumedCOLLECTIVESArePriced is the case the old host_* rule got wrong.
//
// A deployment whose HOST terms are assumed has always worked, because host overheads are
// assumed everywhere and every fixture exercises them. A deployment whose COLLECTIVES are
// assumed is the new shape: gb300 carries eighteen such entries, for a part whose comm
// sweeps do not exist yet. Collectives sit on the step-time path where host overheads do
// not, so this reaches further into the kernel than the test above.
//
// Separate from the general case, and named for the term, because the failure it guards is
// specific: a consumer that refused assumed collectives would make gb300 unpriceable and
// would push the registry toward labelling a reasoned estimate `measured` to get any
// prediction at all -- defeating the method axis the schema added to tell those apart.
func TestAssumedCollectivesArePriced(t *testing.T) {
	k := fixture(t, "minimax-m25-gb300-tp4.yaml")

	_, _, assumed := k.Evidence()
	var collectives int
	for _, name := range assumed {
		if strings.HasPrefix(name, "collective_") {
			collectives++
		}
	}
	if collectives == 0 {
		t.Skip("gb300's collectives are no longer assumed; this fixture has served its " +
			"purpose and the general case above still covers the evidence axis")
	}

	// The whole point: a step prices, and the collective terms are in it.
	e := k.StepTime(decodeBatch(64, 1, 4096))
	if e.NoOverlap <= 0 {
		t.Fatal("a deployment whose collectives are assumed priced to nothing")
	}
	// TP=4 on one tray, so the reductions are on-node: NVLink must carry them. This is
	// what proves the assumed coefficients were actually USED rather than skipped over.
	if e.PerResource[kernel.ResourceNVLink] <= 0 {
		t.Errorf("no NVLink time on a TP=4 step; the assumed collective coefficients "+
			"were resolved (%d of them) but apparently not charged", collectives)
	}
	t.Logf("priced a step with %d assumed collective coefficients", collectives)
}

func TestTwoKernelsFromOneScenarioAgreeEverywhere(t *testing.T) {
	// §7's first sense of idempotence: equal inputs yield equivalent kernels. A kernel
	// that resolved differently on a second construction would make a simulation depend
	// on construction order.
	a := fixture(t, "minimax-m25-h200-ep8.yaml")
	b := fixture(t, "minimax-m25-h200-ep8.yaml")
	for _, batch := range []kernel.Batch{
		decodeBatch(1, 1, 128), decodeBatch(256, 2, 8192), decodeBatch(4, 2048, 2048),
	} {
		if a.StepTime(batch).Overlap != b.StepTime(batch).Overlap {
			t.Errorf("two kernels from one scenario disagree at %d tokens",
				batch.Tokens())
		}
	}
	if a.FixedBytes() != b.FixedBytes() {
		t.Error("two kernels from one scenario report different fixed occupancy")
	}
	if a.SequenceVariableBytes(4096) != b.SequenceVariableBytes(4096) {
		t.Error("two kernels from one scenario report different KV occupancy")
	}
}

func TestAttentionTermScalesWithContextLength(t *testing.T) {
	// Attention work is quadratic in context at fixed query length: doubling the context
	// doubles the per-token key and value reads. A term that ignored context would leave
	// the SM figure flat, which no monotonicity check on total step time would catch
	// because HBM dominates these batches.
	// Attention is an HBM term now, not an SM one: it is priced by a measured floor and
	// bandwidth rather than from FLOPs, so a context increase lands there.
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	short := k.StepTime(decodeBatch(64, 1, 2048)).PerResource[kernel.ResourceHBM]
	long := k.StepTime(decodeBatch(64, 1, 32768)).PerResource[kernel.ResourceHBM]
	if long <= short {
		t.Fatalf("16x the context left the HBM term at %v against %v", long, short)
	}
	// The attention term grows 16x while the weight reads do not move at all, so the total
	// must rise by strictly less than 16x and strictly more than nothing.
	if ratio := float64(long) / float64(short); ratio < 1.05 || ratio > 16 {
		t.Errorf("HBM rose %.2fx over a 16x context increase; attention grows 16x and the "+
			"weight terms are flat, so the total must sit between", ratio)
	}
	// And the SM term must NOT move with context, since no SM term depends on it.
	sShort := k.StepTime(decodeBatch(64, 1, 2048)).PerResource[kernel.ResourceSM]
	sLong := k.StepTime(decodeBatch(64, 1, 32768)).PerResource[kernel.ResourceSM]
	if sShort != sLong {
		t.Errorf("SM moved from %v to %v with context; with attention priced by a "+
			"measured bandwidth form, no SM term depends on context", sShort, sLong)
	}
}

func TestHostTermIsChargedOnEveryStep(t *testing.T) {
	// The host term is small against a decode step and it is not zero. A step that
	// dropped it would report a NoOverlap below the sum of its parts.
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	e := k.StepTime(decodeBatch(32, 2, 8192))
	host := e.PerResource[kernel.ResourceHost]
	if host <= 0 {
		t.Error("no host time charged for a step that launches kernels")
	}
	var sum time.Duration
	for _, d := range e.PerResource {
		sum += d
	}
	if sum != e.NoOverlap {
		t.Errorf("per-resource terms sum to %v but NoOverlap reports %v", sum, e.NoOverlap)
	}
}

// The allocation-free form must be the same function, not a faster approximation of it.
// A simulator choosing StepTimeInto for speed would otherwise be choosing different
// numbers, and the choice would be invisible.
func TestStepTimeIntoAgreesWithStepTime(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	per := make(map[kernel.Resource]time.Duration, 8)
	for _, b := range []kernel.Batch{
		{}, decodeBatch(1, 1, 1), decodeBatch(1, 2048, 2048),
		decodeBatch(256, 2, 8192), decodeBatch(1024, 1, 32768),
	} {
		want := k.StepTime(b)
		got := k.StepTimeInto(b, per)
		if got.Overlap != want.Overlap || got.NoOverlap != want.NoOverlap ||
			got.Bottleneck != want.Bottleneck {
			t.Errorf("at %d tokens: into gave %v/%v/%s, allocating gave %v/%v/%s",
				b.Tokens(), got.Overlap, got.NoOverlap, got.Bottleneck,
				want.Overlap, want.NoOverlap, want.Bottleneck)
		}
		if len(got.PerResource) != len(want.PerResource) {
			t.Errorf("at %d tokens: breakdowns differ in size, %d against %d",
				b.Tokens(), len(got.PerResource), len(want.PerResource))
		}
		for r, d := range want.PerResource {
			if got.PerResource[r] != d {
				t.Errorf("at %d tokens: %s is %v into, %v allocating",
					b.Tokens(), r, got.PerResource[r], d)
			}
		}
	}
}

// And the reused map must be cleared, so a resource that fell out between steps does not
// linger from the previous call.
func TestStepTimeIntoClearsStaleResources(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	per := make(map[kernel.Resource]time.Duration, 8)
	k.StepTimeInto(decodeBatch(256, 2, 8192), per)
	busy := len(per)
	k.StepTimeInto(kernel.Batch{}, per)
	if len(per) >= busy {
		t.Errorf("an empty step left %d resources in a map that held %d; stale entries "+
			"would report work no step did", len(per), busy)
	}
	if _, ok := per[kernel.ResourceHost]; !ok {
		t.Error("an empty step still costs host time, which should be in the breakdown")
	}
}

// A mixed batch — some requests prefilling, some decoding — must pay both regimes, because
// the engine launches a kernel for each. Charging only the larger would understate the step.
func TestMixedBatchPaysBothAttentionRegimes(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-tp8.yaml")
	decodeOnly := k.StepTime(decodeBatch(32, 1, 4096))
	mixed := decodeBatch(32, 1, 4096)
	mixed.Reqs = append(mixed.Reqs, kernel.ReqShape{
		Scheduled: 2048, Computed: 0, PromptLen: 2048,
	})
	both := k.StepTime(mixed)
	if both.PerResource[kernel.ResourceSM] <= decodeOnly.PerResource[kernel.ResourceSM] {
		t.Error("adding a prefill request did not add compute")
	}
	if both.PerResource[kernel.ResourceHBM] < decodeOnly.PerResource[kernel.ResourceHBM] {
		t.Error("adding a prefill request reduced the decode requests' bandwidth cost")
	}
}

// The routed-expert term composes its compute and memory as a SUM, where every other term in a
// layer composes as a max. This is the measured asymmetry recorded in kernel.go: NVIDIA's MoE
// tables fit the sum better on all four parts tested, while the same test on dense GEMM prefers
// the max, which is why the sum is scoped to the routed term rather than applied to the layer.
//
// Behavioural, and it must DISCRIMINATE the two rules. The discriminating property is that the
// step time exceeds the largest single resource: under a max the stage equals its binding
// resource, and under the routed sum it exceeds it by the smaller routed half. A first version
// of this test perturbed the SM budget instead and did not discriminate -- on the reference
// fixture SM is 157us against 5.1ms of HBM, so the routed compute half is about 1% of the step
// and halving the SM budget moved it 1.2%, which a max-composed kernel also does through its
// own SM term. The mutation reverting the sum to a max survived it.
func TestTheRoutedExpertTermComposesAsASumNotAMax(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	b := decodeBatch(32, 2, 8192)
	e := k.StepTime(b)

	// The largest single resource. Under a pure max over resources the stage would equal this,
	// summed over layers, and the step could not exceed it.
	var largest time.Duration
	for _, r := range []kernel.Resource{kernel.ResourceSM, kernel.ResourceHBM,
		kernel.ResourceNVLink, kernel.ResourceNIC} {
		if d := e.PerResource[r]; d > largest {
			largest = d
		}
	}
	if largest <= 0 {
		t.Fatal("no resource carried any time")
	}
	// The host term is additive by design and is not part of the resource max, so it is
	// excluded from the comparison.
	gpu := e.Overlap - e.PerResource[kernel.ResourceHost]
	if gpu <= largest {
		t.Errorf("the GPU part of the step (%v) does not exceed its largest resource (%v); "+
			"under the routed sum it must, because the smaller routed half is added to the "+
			"stage rather than absorbed by the max", gpu, largest)
	}
	// And the excess must be bounded by the routed term, not by the whole layer: a kernel that
	// summed EVERY resource would exceed the largest by far more.
	var total time.Duration
	for _, r := range []kernel.Resource{kernel.ResourceSM, kernel.ResourceHBM,
		kernel.ResourceNVLink, kernel.ResourceNIC} {
		total += e.PerResource[r]
	}
	if gpu >= total {
		t.Errorf("the GPU part of the step (%v) is at or above the sum of every resource "+
			"(%v); only the routed halves compose additively, so the rest must still be "+
			"max-composed", gpu, total)
	}
}

// The per-resource breakdown must remain a true account of where time went: the routed compute
// charged to SM and the routed memory to HBM. An earlier version of the sum charged the WHOLE
// routed cost to whichever half was larger, which doubled one resource and zeroed the other,
// and three existing tests caught it -- including one asserting routed compute falls as the
// expert group widens.
func TestTheRoutedHalvesAreChargedToTheirOwnResources(t *testing.T) {
	narrow := fixture(t, "minimax-m25-h200-ep8.yaml")
	wide := fixture(t, "minimax-m25-h200-ep72.yaml")

	// Widening the expert group gives each rank fewer experts, so at the SAME global load
	// both routed halves fall. The load is held: ep8 is one replica and ep72 nine, and a
	// batch is one replica's, so 72 decodes on the first against 8 on each of the second.
	// An earlier form gave both 32 a replica, which is nine times the load at ep72; per rank
	// that is 9 x 4 of 256 experts' work against 1 x 32, and it fell only while the lighter
	// rank (3 experts) was the one priced.
	nb, wb := decodeBatch(72, 2, 8192), decodeBatch(8, 2, 8192)
	nSM := narrow.StepTime(nb).PerResource[kernel.ResourceSM]
	wSM := wide.StepTime(wb).PerResource[kernel.ResourceSM]
	nHBM := narrow.StepTime(nb).PerResource[kernel.ResourceHBM]
	wHBM := wide.StepTime(wb).PerResource[kernel.ResourceHBM]
	if wSM >= nSM {
		t.Errorf("SM at EP=72 (%v) did not fall below EP=8 (%v); the routed compute half is "+
			"not being charged to SM", wSM, nSM)
	}
	if wHBM >= nHBM {
		t.Errorf("HBM at EP=72 (%v) did not fall below EP=8 (%v); the routed memory half is "+
			"not being charged to HBM", wHBM, nHBM)
	}
}

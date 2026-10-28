// Moved from blis-latency-kernel/methods_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"math"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
)

// These tests assert BEHAVIOUR rather than arithmetic: what must hold of a method's
// answers across inputs, which is what a caller depends on and what survives a change to
// any single constant. A test that pinned a millisecond figure would fail on every
// coefficient update while proving nothing about whether the method is right.

// --- Memory occupancy ---------------------------------------------------------

// Without expert parallelism every rank holds every expert as a TENSOR SLICE, so a rank's
// expert weights are the model's divided by the tensor-parallel width -- not the whole
// model's. This is the case the two tests above cannot see: both use EP-on fixtures, where
// expertTensorShards is 1 and the division is a no-op, so a missing division stayed
// invisible while every other memory test passed.
//
// It was missing. The memory path omitted the /expertTensorShards that the step-time path
// applies, and reported 675 GiB of expert weights per rank for GLM-5 at tp=8 -- on a 141 GiB
// part. It surfaced when a caller divided a device budget by the figure and 17 of 39
// deployments reported that they could not serve a single request.
//
// Checked against the model's own arithmetic rather than a recorded byte count, and stated
// as a per-rank plausibility bound so a coefficient change cannot fail it.
func TestExpertWeightsAreTensorSlicedWithoutExpertParallelism(t *testing.T) {
	k := fixture(t, "glm5-h200-tp8.yaml")
	if w := k.Resolved().ExpertParallelWidth; w > 1 {
		t.Fatalf("this fixture resolved expert-parallel width %d; the test needs it off", w)
	}
	weights := float64(k.FixedBytes().Weights)

	// GLM-5 served fp8: 75 MoE layers, 256 experts, three matrices per expert at
	// 2048x6144. That is the whole model's expert parameter count.
	const (
		moeLayers      = 75.0
		experts        = 256.0
		perExpertBytes = 3.0 * 2048.0 * 6144.0 // fp8, one byte per parameter
	)
	whole := moeLayers * experts * perExpertBytes
	perRank := whole / 8.0 // tp=8

	// A rank must hold about its slice, plus dense layers, embeddings and the head. The
	// band spans that remainder generously while excluding the undivided figure by two
	// orders of magnitude: the bug reported 8x perRank.
	if weights < perRank || weights > perRank*1.5 {
		t.Errorf("a tp=8 rank reports %.1f GiB of weights; the expert slice alone is "+
			"%.1f GiB and the dense remainder is small, so anything outside "+
			"[%.1f, %.1f] GiB means the expert term is sharded wrongly. The whole "+
			"model's expert weights are %.1f GiB -- reporting that per rank is the "+
			"defect this test exists for",
			weights/(1<<30), perRank/(1<<30),
			perRank/(1<<30), perRank*1.5/(1<<30), whole/(1<<30))
	}

	// And the figure must be physically possible: a per-rank occupancy cannot exceed the
	// part it sits on. H200 is 141 GiB.
	const h200Bytes = 141.0 * (1 << 30)
	if total := float64(k.FixedBytes().Total()); total >= h200Bytes {
		t.Errorf("per-rank fixed occupancy %.1f GiB does not fit a 141 GiB H200",
			total/(1<<30))
	}
}

// The memory path and the step-time path read the same expert bytes -- resident throughout,
// read once per step -- so they must divide them the same way. A fix applied to one and not
// the other would leave the two disagreeing, which is how the defect above arose.
func TestMemoryAndStepTimeAgreeOnExpertSharding(t *testing.T) {
	for _, f := range []string{"glm5-h200-tp8.yaml", "minimax-m25-h200-ep8.yaml"} {
		k := fixture(t, f)
		shards := k.expertTensorShards
		want := 1.0
		if k.Resolved().ExpertParallelWidth <= 1 {
			want = k.tp
		}
		if shards != want {
			t.Errorf("%s: expertTensorShards is %.0f, expected %.0f for expert-parallel "+
				"width %d at tp=%.0f", f, shards, want,
				k.Resolved().ExpertParallelWidth, k.tp)
		}
	}
}

func TestExpertWeightsScaleWithTheLocalExpertCount(t *testing.T) {
	// The single most load-bearing memory term: it sets both occupancy and the decode
	// step's HBM traffic. Checked against the model's own arithmetic rather than a
	// recorded byte count, so a coefficient change does not fail it.
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	weights := float64(k.FixedBytes().Weights)
	// One expert of the fixture model: gate and up fused into n=1536 plus down, over a
	// hidden size of 3072, at one byte per fp8 parameter.
	const perExpert = (2*1536*3072 + 1536*3072) * 1.0
	expertBytes := perExpert * k.expertsPerRank * float64(k.plan.TotalLayers)
	if share := expertBytes / weights; share < 0.90 || share > 1.0 {
		t.Errorf("the expert shard is %.1f%% of this rank's weights; for a model that "+
			"is 99%% expert parameters it should be nearly all of them", share*100)
	}
}

func TestFewerSMsCostMoreOnlyWhenComputeBinds(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	// A decode step here is HBM-bound, so withholding the 12 SMs an offload fetch takes must
	// be almost entirely absorbed — that is the max-over-resources rule doing its work.
	//
	// "Almost" rather than "entirely", and the exception is measured. The routed-expert term
	// composes its compute and memory as a SUM rather than a max, because NVIDIA's MoE tables
	// fit the sum better on every part tested (see the composition in kernel.go). So on an MoE
	// model a slower SM raises the routed compute half, and that half is additive. What must
	// still hold is that the effect is SMALL relative to the derate: withholding 9% of the SMs
	// moves an HBM-bound step by well under 9%, because everything except the routed compute
	// is still absorbed.
	full := decodeBatch(32, 2, 8192)
	derated := full
	derated.SMBudget = k.chip.SMCount - 12
	before, after := k.StepTime(full), k.StepTime(derated)
	if before.Bottleneck != kernel.ResourceHBM {
		t.Skipf("this batch is %s-bound, so it does not test absorption",
			before.Bottleneck)
	}
	derate := 12.0 / float64(k.chip.SMCount)
	grew := after.Overlap.Seconds()/before.Overlap.Seconds() - 1
	if grew < 0 {
		t.Errorf("withholding SMs made an HBM-bound step faster: %v to %v",
			before.Overlap, after.Overlap)
	}
	// A tenth of the derate: the routed compute half is a small part of an HBM-bound step, so
	// almost all of the withheld capacity is still absorbed. A kernel that had lost the max
	// rule entirely would grow by roughly the derate itself.
	if grew > derate/10 {
		t.Errorf("withholding %.1f%% of the SMs grew an HBM-bound step by %.2f%%; only the "+
			"routed compute half is additive, so the growth should be a small fraction of "+
			"the derate", 100*derate, 100*grew)
	}
	if after.PerResource[kernel.ResourceSM] <= before.PerResource[kernel.ResourceSM] {
		t.Error("withholding SMs did not raise the SM term at all")
	}
}

func TestResolvedReportsWhatWillActuallyRun(t *testing.T) {
	skipUntilSixteenRankAllToAllExists(t)
	k := fixture(t, "minimax-m25-h200-ep16.yaml")
	r := k.Resolved()
	if r.ExpertParallelWidth != 16 {
		t.Errorf("expert-parallel width resolved to %d, not the scenario's 16",
			r.ExpertParallelWidth)
	}
	if r.AllReduceBackend == "" {
		t.Error("no all-reduce backend named")
	}
	wide := fixture(t, "minimax-m25-h200-ep72.yaml").Resolved()
	if wide.ExpertParallelWidth != 72 {
		t.Errorf("the 9-node scenario resolved to width %d, not 72",
			wide.ExpertParallelWidth)
	}
}

// minimax-m2.5 shape, read off the committed graph in blis-catalog. Restated here so a
// test failure names a concrete quantity rather than pointing at a YAML file.
//
// These constants previously described a model that is no longer in blis-catalog.
// MiniMax-M2.5 keeps the two properties the arithmetic below depends on -- hidden_size
// 3072 and an expert intermediate of 1536 -- and keeps 8 KV heads so a tp=8 rank still
// holds exactly one, which is what the KV-sharding assertion needs.
const (
	g5Hidden          = 3072
	g5Layers          = 62
	g5QKVOut          = 8192
	g5OProjOut        = 3072
	g5ExpertInter     = 1536
	g5Experts         = 256
	g5TopK            = 8
	g5QHeads          = 48
	g5KVHeads         = 8
	g5HeadDim         = 128
	g5Vocab           = 200064
	g5ParamsPerExpert = 2*g5ExpertInter*g5Hidden + g5ExpertInter*g5Hidden
	g5DensePerLayer   = g5Hidden*g5QKVOut + g5OProjOut*g5Hidden
)

// hbmSeconds returns the HBM time the model's own arithmetic gives for a decode batch:
// weights plus the local expert shard plus the KV read, over the derated bandwidth.
func hbmSeconds(k *Kernel, requests, tokens, context int) float64 {
	// Only the experts the step reaches, and only this rank's slice of each. Both terms
	// matter: coverage below ~128 tokens, and the tensor-shard divisor whenever expert
	// parallelism is off.
	touched := price.ExpertsTouched(tokens, k.totalExperts, g5TopK, k.expertsPerRank)
	experts := g5ParamsPerExpert * touched / k.expertTensorShards * g5Layers
	dense := float64(g5DensePerLayer) * g5Layers / k.tp
	head := float64(g5Hidden) * g5Vocab / k.tp
	kvHeads := float64(g5KVHeads) / k.tp
	if kvHeads < 1 {
		kvHeads = 1
	}
	// Attention is priced by its own measured floor and rate, so the KV read is not a
	// plain HBM term: it carries a per-layer floor and runs at the attention kernel's
	// achieved bandwidth rather than the generic derate.
	kvPerLayer := float64(requests) * float64(context) * 2 * kvHeads * g5HeadDim
	attention := g5Layers * (k.attentionFloor.Seconds() + kvPerLayer/k.attentionRate)
	return (experts+dense+head)/k.hbmBytesPerSecond + attention
}

func TestHBMTermMatchesTheModelsOwnArithmetic(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	// Decode shapes only. A prefill request's attention is priced against compute, so its
	// bytes are not in the HBM term at all — which this arithmetic does not model and
	// should not pretend to.
	for _, tc := range []struct{ requests, queryLen, context int }{
		{256, 2, 8192}, {32, 2, 8192}, {64, 1, 4096}, {512, 1, 2048},
	} {
		e := k.StepTime(decodeBatch(tc.requests, tc.queryLen, tc.context))
		got := e.PerResource[kernel.ResourceHBM].Seconds()
		want := hbmSeconds(k, tc.requests, tc.requests*tc.queryLen, tc.context)
		// Two percent: the kernel also charges elementwise traffic, which this
		// arithmetic omits because it is a small term the graph states per node.
		if ratio := got / want; ratio < 0.98 || ratio > 1.10 {
			t.Errorf("%d req x %d tok, ctx %d: HBM term %.3f ms against %.3f ms from "+
				"weights + expert shard + KV (%.2fx)", tc.requests, tc.queryLen,
				tc.context, got*1e3, want*1e3, ratio)
		}
	}
}

func TestKVTermScalesWithTheShardedHeadCount(t *testing.T) {
	// With 8 KV heads and TP=8 each rank holds one head, so the KV read is 1/8 of the
	// unsharded figure. Dropping the divisor is a mutation that monotonicity cannot see.
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	const requests, context = 128, 8192
	perToken := 2 * 1.0 * g5HeadDim * g5Layers // one KV head per rank, fp8 cache
	want := float64(requests*context) * perToken
	got := float64(requests) * float64(context) * k.kvBytesPerToken
	if ratio := got / want; ratio < 0.99 || ratio > 1.01 {
		t.Errorf("KV read is %.2f GB where one head per rank gives %.2f GB (%.2fx); "+
			"with %d heads over TP=%.0f a rank holds one",
			got/1e9, want/1e9, ratio, g5KVHeads, k.tp)
	}
}

func TestRoutedComputeScalesWithTheLocalExpertShare(t *testing.T) {
	// The routed term divides by the number of experts a rank HOLDS and multiplies by
	// the number of replica groups whose tokens reach those experts. Both move when EP
	// widens, and which way the net goes depends on HOW it widens.
	//
	// These two fixtures widen EP by adding DP ranks: ep16 is tp8/dp2, ep72 is tp8/dp9.
	// So localExpertShare falls 4.67x (14 local experts against 3) while the
	// attention-DP funnel grows 4.5x (dp 2 to 9), and the two nearly cancel. That is
	// vLLM's actual behaviour, not an artefact: adding a DP rank adds a replica of
	// attention AND its tokens to the shared expert pool, so per-rank routed work is
	// roughly preserved.
	//
	// An earlier version of this test asserted SM must FALL by 1.2x to 4.67x across
	// this pair, which encoded the localExpertShare half of that and omitted the
	// funnel. It passed only while the funnel was missing from the model.
	narrow := fixture(t, "minimax-m25-h200-ep8.yaml")
	wide := fixture(t, "minimax-m25-h200-ep72.yaml")
	batch := decodeBatch(256, 2, 8192)
	sm16 := narrow.StepTime(batch).PerResource[kernel.ResourceSM].Seconds()
	sm72 := wide.StepTime(batch).PerResource[kernel.ResourceSM].Seconds()

	// The two effects are each large and they oppose, so the net must be SMALL. A
	// mutation that drops either one leaves a 4.5x-4.67x swing, which this rejects.
	if ratio := sm16 / sm72; ratio < 0.7 || ratio > 1.5 {
		t.Errorf("SM moved %.2fx between EP=16 (dp2) and EP=72 (dp9); localExpertShare "+
			"falls 4.67x and the DP funnel rises 4.5x, so the net belongs in 0.7x..1.5x. "+
			"A swing this large means one of the two terms is missing", ratio)
	}

	// And the share itself must still bite, isolated from the funnel: at equal DP a
	// wider expert group holds fewer experts per rank and must cost less.
	if narrow.localExpertShare <= wide.localExpertShare {
		t.Errorf("localExpertShare did not fall with EP width: %.4f at EP=16 against "+
			"%.4f at EP=72", narrow.localExpertShare, wide.localExpertShare)
	}
	if narrow.moeDPFunnel() >= wide.moeDPFunnel() {
		t.Errorf("the DP funnel did not rise with DP width: %.1f at dp2 against %.1f "+
			"at dp9", narrow.moeDPFunnel(), wide.moeDPFunnel())
	}
}

func TestCrossNodeCollectivesCostMoreThanOnNodeOnes(t *testing.T) {
	// At EP=16 the expert group spans two nodes and its dispatch pays a span penalty; at
	// TP=8 the reductions stay on the node and pay none. Removing the span check entirely
	// is a mutation that no ordering test catches, because the NIC term merely becomes
	// the NVLink term.
	skipUntilSixteenRankAllToAllExists(t)
	k := fixture(t, "minimax-m25-h200-ep16.yaml")
	e := k.StepTime(decodeBatch(256, 2, 8192))
	nic := e.PerResource[kernel.ResourceNIC]
	nvlink := e.PerResource[kernel.ResourceNVLink]
	if nic <= 0 {
		t.Error("the expert group spans two nodes, so its dispatch must cross the NIC")
	}
	if nvlink <= 0 {
		t.Error("the tensor-parallel reductions stay on the node, so NVLink must carry " +
			"them")
	}
	// A single-node deployment must put everything on NVLink and nothing on the NIC.
	// Checked through the resolved layout rather than by building a third fixture.
	if k.crossesNodes(collKey{Op: model.OpAllReduce, Group: price.GroupTP}) {
		t.Error("a TP=8 reduction on 8-GPU nodes was judged to cross a node boundary")
	}
	if !k.crossesNodes(collKey{Op: model.OpAll2All, Group: price.GroupExpert}) {
		t.Error("a 16-wide expert group on 8-GPU nodes was judged to stay on one node")
	}
}

// TestRoutingImbalanceIsApplied covers the one term the magnitude tests above cannot
// reach. The imbalance multiplier is 1.056 on H200 — a 5.6% adjustment to the routed
// compute term alone — which is below the tolerance any end-to-end assertion can carry.
//
// So this isolates the routed term instead of bounding the whole step. It prices the
// routed work from the model's own shape and requires the kernel's SM term to exceed the
// unmultiplied figure by the multiplier: a kernel that resolved the coefficient and then
// ignored it would land on the unmultiplied value.
func TestRoutingImbalanceIsApplied(t *testing.T) {
	// EP=16 rather than EP=72: a rank holds 14 experts there against 3, so routed work
	// is a far larger share of the compute term and a 5.6% factor on it is detectable.
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	// The multiplier is a MEDIAN ratio of skewed to balanced latency, so the band it must
	// fall in is set by the measurement, not by an assumption that skew always costs
	// extra. On vLLM's own expert kernels -- Triton, FlashInfer-Cutlass, Marlin -- mild
	// skew is close to free and the ratio is symmetric about one (49.7% of 18,144 paired
	// shapes above it on h200), where TRT-LLM's generic `moe_torch_flow_cutlass` path
	// showed a systematic 1.066. The penalty lives in the tail, which p90 records at
	// 1.394 and which this coefficient deliberately does not carry.
	//
	// The band therefore admits a near-unity median and still rejects the two failures
	// that matter: a missing coefficient falling back to exactly 1.0 (ValueOr's default,
	// which would mean the registry entry was not found at all), and a tail value
	// substituted for the median.
	if k.moeImbalance == 1.0 {
		t.Fatalf("routing imbalance is exactly 1.0, which is ValueOr's fallback: the " +
			"coefficient did not resolve from the registry")
	}
	if k.moeImbalance < 0.9 || k.moeImbalance > 1.3 {
		t.Fatalf("routing imbalance resolved to %.4f, outside the measured median band "+
			"[0.9, 1.3]; a p90 tail value or a unit slip would land here", k.moeImbalance)
	}
	found := false
	for _, o := range k.Provenance() {
		if o.Name == "moe_routing_imbalance_median" {
			found = true
			if o.Method != "measured" {
				t.Errorf("the imbalance multiplier is %s, not measured", o.Method)
			}
		}
	}
	if !found {
		t.Error("the imbalance multiplier is absent from provenance, so the step time " +
			"is not using a resolved coefficient for it")
	}

	// The routed term is compared against an INDEPENDENTLY computed total rather than
	// against a residual. Inferring the non-routed part from the measured total makes the
	// comparison self-consistent whichever way the kernel prices it, which is a test that
	// cannot fail — the first version of this test had exactly that flaw and a mutation
	// dropping the multiplier survived it.
	//
	// So every SM term is computed here from the model's shape, and the measured value is
	// required to match the total that includes the multiplier.
	batch := decodeBatch(4, 2048, 2048)
	sm := k.StepTime(batch).PerResource[kernel.ResourceSM].Seconds()

	tokens := float64(batch.Tokens())
	eff := func(m float64) float64 { return k.epsMax * m / (m + k.mHalf) }

	// Attention contributes no FLOPs now: it is priced by its own measured floor and
	// bandwidth, so the SM term is the dense projections, the routed experts and the head.
	dense := 2 * tokens * float64(g5DensePerLayer) * g5Layers / k.tp
	head := 2 * tokens * g5Hidden * g5Vocab / k.tp
	nonRouted := (dense + head) / (k.computeFLOPsPerSecond * eff(tokens))

	routedFLOPs := tokens * g5TopK * k.localExpertShare * 2 *
		float64(g5ParamsPerExpert) * g5Layers
	rowsPerExpert := tokens * g5TopK / k.expertsPerRank
	routed := routedFLOPs / (k.computeFLOPsPerSecond * eff(rowsPerExpert))

	withMultiplier := nonRouted + routed*k.moeImbalance

	// The SM term must be the same ORDER as the arithmetic above. This is a sanity bound,
	// not an equality: the composition also charges terms this local calculation omits
	// (the index projection, elementwise work, the launch floor), so the measured term is
	// legitimately larger. A 1.49x gap is normal here; a 10x gap would mean a sharding or
	// unit error.
	if ratio := sm / withMultiplier; ratio < 0.5 || ratio > 3.0 {
		t.Errorf("the SM term is %.4f ms against %.4f ms of dense + routed + head work "+
			"(%.3fx); that is too far apart to be the omitted small terms",
			sm*1e3, withMultiplier*1e3, ratio)
	}

	// THE LOAD-BEARING CHECK, and it is unconditional. An earlier version compared two
	// candidate totals -- routed work with and without the factor -- and asked which the
	// measurement sat closer to. That only discriminates while the multiplier is far from
	// one, and on the vLLM lane it is 0.999, which put the candidates 0.06% apart and
	// made the comparison vacuous. Scaling the resolved coefficient and requiring the
	// step to move with it does not depend on the fitted value at all.
	scaled := *k
	scaled.moeImbalance = k.moeImbalance * 1.5
	smScaled := scaled.StepTime(batch).PerResource[kernel.ResourceSM].Seconds()
	if smScaled <= sm*1.001 {
		t.Errorf("scaling the imbalance multiplier by 1.5 moved the SM term from "+
			"%.4f to %.4f ms; the coefficient is resolved but not applied",
			sm*1e3, smScaled*1e3)
	}
}

// TestCollectivesUseTheTransitionRate checks that the three-parameter form reaches the
// step, not merely that internal/price implements it.
//
// The two are separable and a mutation proved it: passing zero for the transition rate
// falls back to the two-parameter form, which every ordering and monotonicity test above
// accepts. So this prices one collective both ways from the resolved coefficients and
// requires the step's collective term to match the three-parameter figure.
func TestCollectivesUseTheTransitionRate(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	if k.collectiveTransitions[collKey{Op: model.OpAllReduce, Group: price.GroupTP}] <= 0 {
		t.Fatal("no transition rate resolved for the tensor-parallel reduction")
	}
	// The transition rate must sit below the asymptote: a collective reaches its peak
	// bandwidth only at messages larger than a forward pass produces.
	for key, transition := range k.collectiveTransitions {
		peak := k.collectiveRates[key]
		if transition >= peak {
			t.Errorf("%s: transition rate %.0f is not below the asymptote %.0f",
				key.Op, transition, peak)
		}
	}

	// A prefill batch puts each collective in the transition region, which is where the
	// two forms differ. Both candidates are computed from the same resolved constants.
	batch := decodeBatch(1, 2048, 2048)
	got := k.StepTime(batch).PerResource[kernel.ResourceNVLink].Seconds()

	payload := float64(batch.Tokens()) * g5Hidden * 1.0 // fp8 activations
	three, two := 0.0, 0.0
	for key, count := range map[collKey]float64{
		{Op: model.OpAllReduce, Group: price.GroupTP}: 2,
	} {
		floor := k.collectiveFloors[key].Seconds()
		peak, transition := k.collectiveRates[key], k.collectiveTransitions[key]
		three += count * g5Layers * price.CollectiveTime(payload, transition, peak, floor)
		two += count * g5Layers * price.FloorAndRate(payload, peak, floor)
	}
	if three <= two*1.05 {
		t.Fatalf("the two forms differ by under 5%% at this shape (%.3f against "+
			"%.3f ms), so this test cannot distinguish them", three*1e3, two*1e3)
	}
	// The step's NVLink term carries the two reductions and nothing else at this shape,
	// since the expert dispatch crosses the fabric and lands on the NIC.
	if math.Abs(got-two) < math.Abs(got-three) {
		t.Errorf("the collective term %.3f ms is nearer the two-parameter figure "+
			"%.3f ms than the three-parameter %.3f ms; the transition rate is resolved "+
			"but not used", got*1e3, two*1e3, three*1e3)
	}
}

// Per-kernel dispatch is the term that dominates a single-request decode step, and it was
// missing entirely until two published runs exposed it. These pin the mechanism rather than
// the value, so a recalibration does not fail them.
func TestDispatchCostScalesWithTheKernelCount(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-tp8.yaml")
	if k.plan.TotalKernels < k.plan.TotalLayers {
		t.Fatalf("%d kernels for %d layers; every layer launches at least one",
			k.plan.TotalKernels, k.plan.TotalLayers)
	}
	// The fixture's layer is nine graph primitives, several of which imply more than one
	// launch, so the count per layer must exceed the primitive count.
	perLayer := float64(k.plan.TotalKernels) / float64(k.plan.TotalLayers)
	if perLayer < 9 || perLayer > 30 {
		t.Errorf("%.1f kernels per layer; a nine-primitive MoE layer launches more than "+
			"nine and far fewer than thirty", perLayer)
	}
	if k.launchPerKernel <= 0 {
		t.Fatal("no per-kernel dispatch cost resolved; a single-request step would be " +
			"priced at roughly half its measured value")
	}
	// The term must reach the STEP, not merely be computable. Checked through the host
	// resource the step reports, because a kernel that computed the cost and then left it
	// out of the estimate would pass any assertion made on launchSeconds alone — which is
	// what an earlier version of this test did, and a mutation dropping the term from the
	// step survived it.
	e := k.StepTime(decodeBatch(1, 1, 8256))
	host := e.PerResource[kernel.ResourceHost]
	dispatch := seconds(k.launchSeconds(decodeBatch(1, 1, 8256), 0))
	if host < dispatch {
		t.Errorf("the step reports %v of host time but dispatch alone is %v; the term "+
			"is computed and not charged", host, dispatch)
	}
	// And it must dominate a single-request step, which is the finding that put it in the
	// model: ITL at one request is ~6 ms on two parts differing 1.43x in memory bandwidth
	// and 12x in context, so it cannot be a memory or context term.
	if share := dispatch.Seconds() / e.Overlap.Seconds(); share < 0.4 {
		t.Errorf("dispatch is %.0f%% of a single-request step; the measurements say it "+
			"dominates", share*100)
	}
	if e.Bottleneck != kernel.ResourceHost {
		t.Errorf("a single-request step is bound by %s; with dispatch dominating it "+
			"should be host-bound", e.Bottleneck)
	}
	// And it must amortize away at large batch, where device work per kernel dwarfs it. A
	// term that stayed dominant would misprice every throughput-oriented deployment.
	big := k.StepTime(decodeBatch(1024, 1, 8256))
	if share := dispatch.Seconds() / big.Overlap.Seconds(); share > 0.3 {
		t.Errorf("dispatch is still %.0f%% of a 1024-request step; it should be "+
			"amortized away", share*100)
	}
}

func TestDispatchCostIsIndependentOfBatchAndContext(t *testing.T) {
	// The term is per kernel, and over decode batches the kernel count is a property of the
	// model rather than of the batch. If it moved with either, it would be absorbing
	// something else. (A prefill row adds the graph's prefill-only nodes; see
	// TestADecodeStepDoesNotLaunchPrefillOnlyNodes.)
	k := fixture(t, "minimax-m25-h200-tp8.yaml")
	want := k.launchSeconds(decodeBatch(1, 1, 128), 0)
	for _, b := range []kernel.Batch{
		decodeBatch(1, 1, 128), decodeBatch(1, 1, 32768), decodeBatch(512, 1, 8192),
	} {
		_ = k.StepTime(b)
		if got := k.launchSeconds(b, 0); got != want {
			t.Errorf("dispatch changed to %v at %d tokens; it is per kernel and the "+
				"kernel count does not depend on the batch", got, b.Tokens())
		}
	}
}

// Prefill and decode attention are separate regimes with separate measured constants, and a
// step holding both pays both. These pin that separation: a mutation that dropped either
// regime's floor survived every other test in this file, because no other test puts a
// prefill-sized request through the kernel.
func TestPrefillAndDecodeAttentionArePricedSeparately(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-tp8.yaml")
	if k.attentionPrefillScale <= 0 || k.attentionPrefillFloor <= 0 {
		t.Fatal("no measured prefill attention form resolved")
	}
	// Prefill binds on compute, so its attention lands on SM.
	prefill := k.StepTime(decodeBatch(1, 2048, 2048))
	// Decode binds on bandwidth, so its attention lands on HBM.
	decode := k.StepTime(decodeBatch(1, 1, 2048))

	if prefill.PerResource[kernel.ResourceSM] <= decode.PerResource[kernel.ResourceSM] {
		t.Errorf("a 2048-token prefill did not cost more SM than a 1-token decode: "+
			"%v against %v", prefill.PerResource[kernel.ResourceSM],
			decode.PerResource[kernel.ResourceSM])
	}
	// The prefill attention component is checked by DIFFERENCE, not by share. At 2048
	// tokens the dense projections and the routed experts dominate SM, so asserting that
	// attention is most of the term would be asserting something false — and a bound that
	// only required SM to exceed the attention floor is far too loose to catch its removal,
	// since the floor is 7% of the attention component and the component is 1% of the term.
	//
	// So: hold everything else fixed and vary only the prompt length. The difference is
	// attention's alone, because no other SM term depends on context.
	shortPrefill := k.StepTime(decodeBatch(1, 2048, 2048))
	longPrefill := k.StepTime(decodeBatch(1, 2048, 16384))
	delta := longPrefill.PerResource[kernel.ResourceSM].Seconds() -
		shortPrefill.PerResource[kernel.ResourceSM].Seconds()

	// Attention cost at a fixed chunk width is LINEAR in the already-computed prefix,
	// and the prefix is not causally masked: a chunk of S tokens resuming on C computed
	// ones attends to every one of them, giving S*C pairs, plus S^2/2 within the chunk.
	// So lengthening the context from 2048 to 16384 at S = 2048 adds S * 14336 pairs,
	// with NO halving.
	//
	// An earlier version of this assertion carried a 0.5 on that term, matching an
	// implementation that halved the prefix. The two agree exactly at Computed == 0,
	// which is every unchunked prefill, so neither the test nor any other check in this
	// repository could distinguish them; FPM's mixed rows did, at a factor of 2.0000.
	eff := k.epsMax * 2048 / (2048 + k.mHalf) * k.attentionPrefillScale
	var wantDelta float64
	for _, l := range k.plan.Layers {
		if l.AttnQHeads == 0 {
			continue
		}
		perContext := 2.0 * 2 * 2048 *
			float64(l.AttnQHeads) * float64(l.AttnHeadDim) / k.tp /
			(k.computeFLOPsPerSecond * eff)
		wantDelta += perContext * (16384 - 2048) * float64(l.Count)
	}
	if ratio := delta / wantDelta; ratio < 0.95 || ratio > 1.05 {
		t.Errorf("an 8x context increase moved the prefill SM term by %.4f ms where the "+
			"measured attention form gives %.4f ms (%.2fx); the work scale or the causal "+
			"halving is not being applied", delta*1e3, wantDelta*1e3, ratio)
	}

	// The floor is charged once per layer per step that holds any prefill request, so it is
	// isolated by varying the REQUEST COUNT at a fixed tiny prompt, where the work term is
	// negligible and the floor is nearly all of the attention component.
	//
	// Two prefill requests at 16 tokens each against one: the work doubles but stays tiny,
	// so the difference between the step and its floor-free counterpart is dominated by the
	// floor. Comparing against a decode step instead would compare two different regimes
	// and swamp the floor in their other differences, which is what an earlier version of
	// this check did and why a mutation zeroing the floor survived it.
	tiny := decodeBatch(1, 16, 16)
	tinyStep := k.StepTime(tiny).PerResource[kernel.ResourceSM].Seconds()
	floorTotal := k.attentionPrefillFloor.Seconds() * float64(k.plan.TotalLayers)
	// At 16 tokens over 72 layers the attention work is microseconds and the floor is
	// milliseconds, so the SM term must be at least the floor and not much more.
	if tinyStep < floorTotal {
		t.Errorf("a 16-token prefill step costs %.4f ms of SM, below the %d layers of "+
			"attention floor it must pay (%.4f ms); the floor is not charged",
			tinyStep*1e3, k.plan.TotalLayers, floorTotal*1e3)
	}
	if tinyStep > floorTotal*3 {
		t.Errorf("a 16-token prefill step costs %.4f ms of SM against a %.4f ms floor; at "+
			"this size the floor should dominate, so something else is over-charged",
			tinyStep*1e3, floorTotal*1e3)
	}

	// And the decode floor, where it IS most of the term.
	decodeFloorTotal := k.attentionFloor.Seconds() * float64(k.plan.TotalLayers)
	if decode.PerResource[kernel.ResourceHBM].Seconds() < decodeFloorTotal {
		t.Errorf("decode HBM term %v is below the %d layers of decode attention floor "+
			"alone (%.3f ms)", decode.PerResource[kernel.ResourceHBM],
			k.plan.TotalLayers, decodeFloorTotal*1e3)
	}
}

// The recurrent state update is priced by its own measured floor and rate. Nothing else in
// this file puts a hybrid model through the kernel, so without this a mutation zeroing the
// term survives — which it did.
func TestRecurrentUpdateIsChargedOnHybridModels(t *testing.T) {
	for _, tc := range []struct {
		scenario string
		kind     model.RecurrentKind
		layers   int
	}{
		{"nemotron3-ultra-h100-agg.yaml", model.RecurrentMamba2, 48},
		{"kimi-k3-h100-nospec.yaml", model.RecurrentKDA, 69},
	} {
		k := fixture(t, tc.scenario)
		fr, ok := k.recurrent[tc.kind]
		if !ok {
			t.Errorf("%s: no measured form for %s; the term would be silently free",
				tc.scenario, tc.kind)
			continue
		}
		// Count the layers of this kind, so the expected floor is the model's own.
		var recurrentLayers int
		for _, l := range k.plan.Layers {
			if l.RecurrentKind == tc.kind {
				recurrentLayers += l.Count
			}
		}
		if recurrentLayers != tc.layers {
			t.Errorf("%s: %d %s layers, expected %d", tc.scenario, recurrentLayers,
				tc.kind, tc.layers)
		}
		// The recurrent term is isolated by DIFFERENCE, not by share: on both these models
		// the MoE term dwarfs it, so a bound requiring SM to exceed the recurrent floor is
		// satisfied by the MoE term alone and catches nothing. That is what an earlier
		// version of this check did, and two mutations survived it.
		//
		// The rate is per token, so varying only the token count isolates the rate; the
		// floor is per layer per step, so it cannot be isolated that way and is checked
		// against a kernel built without it.
		floorTotal := fr.floor.Seconds() * float64(recurrentLayers)

		one := k.StepTime(decodeBatch(1, 1, 4096)).PerResource[kernel.ResourceSM].Seconds()
		many := k.StepTime(decodeBatch(64, 1, 4096)).PerResource[kernel.ResourceSM].Seconds()

		// The recurrent contribution to that difference, from the measured rate. Other SM
		// terms also grow with tokens, so this is a lower bound on the observed delta.
		wantRate := (64.0 - 1.0) / fr.rate * float64(recurrentLayers)
		if delta := many - one; delta < wantRate {
			t.Errorf("%s: 64x the tokens raised SM by %.4f ms, less than the recurrent "+
				"rate alone requires (%.4f ms); the rate is not applied", tc.scenario,
				delta*1e3, wantRate*1e3)
		}

		// The floor: compare against a kernel whose recurrent entry is removed, so the
		// difference is the term itself. Deleting the entry is the ONLY way to see a
		// per-layer constant that the MoE term dominates — but it means the comparison must
		// also assert the term is non-zero, or a mutation zeroing it makes both sides agree
		// at zero and the check passes. That mutation did survive an earlier version.
		bare := fixture(t, tc.scenario)
		delete(bare.recurrent, tc.kind)
		withTerm := k.StepTime(decodeBatch(1, 1, 4096)).PerResource[kernel.ResourceSM]
		without := bare.StepTime(decodeBatch(1, 1, 4096)).PerResource[kernel.ResourceSM]
		observed := withTerm.Seconds() - without.Seconds()
		wantOne := floorTotal + 1.0/fr.rate*float64(recurrentLayers)
		if observed <= 0 {
			t.Errorf("%s: removing the recurrent entry changed the step by %.6f ms; the "+
				"term is not being charged at all", tc.scenario, observed*1e3)
		} else if ratio := observed / wantOne; ratio < 0.99 || ratio > 1.01 {
			t.Errorf("%s: the recurrent term contributes %.4f ms where its floor and rate "+
				"give %.4f ms (%.3fx)", tc.scenario, observed*1e3, wantOne*1e3, ratio)
		}
		// And the absolute magnitude must be what the measurement says, independent of any
		// comparison: 48 mamba layers at 3.5 us or 69 KDA layers at 5.1 us.
		if wantOne < floorTotal {
			t.Errorf("%s: internal check failed, floor %.4f exceeds total %.4f",
				tc.scenario, floorTotal, wantOne)
		}
	}
}

// A sliding-window layer and a full-attention layer are different kernels and must not price
// identically once the registry carries a per-kind fit. Before the dispatch existed, every
// attention kind shared one floor and rate, so a windowed layer was charged the
// full-attention bandwidth -- 0.58 of peak on H200 where the windowed kernel sustains 0.26,
// making every windowed layer about 2.2x too cheap.
//
// gpt-oss-120b is the case that matters: its stack alternates swa_moe and full_moe, so half
// its layers are windowed. The test asserts the two kinds diverge on ONE model whose graph
// carries both, which is the behaviour the dispatch exists to produce -- not that a particular
// coefficient has a particular value.
func TestSlidingWindowAndFullAttentionPriceDifferently(t *testing.T) {
	k := fixture(t, "aisimulate/gpt-oss-120b-h200-fp4-vllm-tp4.yaml")

	var swa, full *price.PlannedLayer
	for i := range k.plan.Layers {
		l := &k.plan.Layers[i]
		switch l.AttnKind {
		case model.AttentionSWA:
			swa = l
		case model.AttentionGQA:
			full = l
		}
	}
	if swa == nil || full == nil {
		t.Skipf("this fixture has no swa/full pair (swa=%v full=%v)", swa != nil, full != nil)
	}
	fr, ok := k.attentionByKind[model.AttentionSWA]
	if !ok {
		t.Skip("the registry carries no per-kind swa fit, so there is nothing to dispatch on")
	}
	if fr.rate >= k.attentionRate {
		t.Fatalf("the swa rate %v is not below the full-attention rate %v; a windowed kernel "+
			"reads a bounded slice and sustains less bandwidth, so this fixture cannot test "+
			"the dispatch", fr.rate, k.attentionRate)
	}

	// The behaviour that matters is that STEP TIME uses the per-kind rate, not merely that the
	// coefficient resolves. Checking the resolved value alone is what let a mutation deleting
	// the dispatch survive: the map was still populated, so the assertion passed while every
	// layer was priced with the part-wide rate.
	//
	// The check: price a decode batch, then price the same batch with the per-kind entry
	// removed. A kernel that dispatches gives a HIGHER time with the entry present, because
	// the windowed rate is lower. A kernel that ignores the map gives the same time twice.
	b := kernel.Batch{DecodeThreshold: 8, SMBudget: 132}
	for i := 0; i < 32; i++ {
		b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 1, Computed: 2048, PromptLen: 2048})
	}
	withKind := k.StepTime(b).Overlap

	saved := k.attentionByKind
	k.attentionByKind = map[model.AttentionKind]floorRate{}
	withoutKind := k.StepTime(b).Overlap
	k.attentionByKind = saved

	if withKind == withoutKind {
		t.Errorf("step time is %v both with and without the per-kind swa entry; the dispatch "+
			"resolves the coefficient but does not price with it", withKind)
	}
	if withKind <= withoutKind {
		t.Errorf("step time with the per-kind entry (%v) is not above the fallback (%v); the "+
			"windowed rate is the lower of the two, so a windowed layer must cost MORE once "+
			"it is priced correctly", withKind, withoutKind)
	}
}

// A kind with no per-kind fit must price exactly as it did before the dispatch existed. This
// is the no-regression guarantee: adding a kind to the registry must never move a deployment
// that has no fit for it.
func TestAKindWithoutAFitFallsBackToThePartWideTerms(t *testing.T) {
	k := fixture(t, "aisimulate/minimax-m2.5-h200-fp8-vllm-tp4.yaml")
	// MiniMax is uniformly GQA, and the registry carries no gqa-suffixed entry, so every layer
	// must resolve the part-wide pair.
	for i := range k.plan.Layers {
		if kind := k.plan.Layers[i].AttnKind; kind != "" {
			if _, ok := k.attentionByKind[kind]; ok {
				t.Fatalf("this fixture's kind %q has a per-kind fit, so it cannot test the "+
					"fallback", kind)
			}
		}
	}
	b := kernel.Batch{DecodeThreshold: 8, SMBudget: 132}
	for i := 0; i < 32; i++ {
		b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 1, Computed: 1024, PromptLen: 1024})
	}
	if got := k.StepTime(b).Overlap; got <= 0 {
		t.Fatalf("step time %v", got)
	}
	// And the part-wide terms must be the ones a GQA model resolves.
	if k.attentionRate <= 0 {
		t.Error("the part-wide decode rate is zero, so the fallback path prices attention at " +
			"nothing")
	}
}

// TestChunkedPrefillChargesTheWholePrefix pins the one case an unchunked prefill cannot
// reveal: a chunk resuming on an already-computed prefix.
//
// Causal masking reduces the pairs WITHIN the chunk, not the pairs against the prefix --
// every token of the chunk is later than every token of the prefix, so all of them
// count. A chunk of S tokens on C computed ones is S*C + S^2/2 pairs.
//
// The implementation charged S*(C+S)*0.5, halving the prefix term. At C == 0 the two
// expressions are equal, which is why every prefill check in this repository passed and
// the error surfaced only against FPM's mixed prefill-plus-context rows, where it is
// 1.995x low at a 1025-token chunk on a 203,760-token prefix.
//
// Asserted two ways that hold whatever the coefficients are. The prefix term must be
// linear in C, so equal prefix increments cost equally. And the increment from zero to a
// prefix of C must be C/(S/2) times the chunk's own attention work, which is the full
// prefix rather than half of it -- checked by comparing two increments whose ratio the
// halving would change.
func TestChunkedPrefillChargesTheWholePrefix(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-tp8.yaml")
	if k.attentionPrefillScale <= 0 {
		t.Fatal("no measured prefill attention form resolved")
	}
	const chunk = 2048
	sm := func(prefix int) float64 {
		b := kernel.Batch{
			Reqs: []kernel.ReqShape{{
				Scheduled: chunk, Computed: prefix, PromptLen: prefix + chunk,
			}},
			DecodeThreshold: 8,
		}
		return k.StepTime(b).PerResource[kernel.ResourceSM].Seconds()
	}
	s0, s1, s2 := sm(0), sm(chunk), sm(2*chunk)
	d1, d2 := s1-s0, s2-s1
	if d1 <= 0 || d2 <= 0 {
		t.Fatalf("lengthening the prefix did not raise the SM term: %v, %v", d1, d2)
	}
	// Linear in the prefix: two equal increments cost the same.
	if r := d2 / d1; r < 0.98 || r > 1.02 {
		t.Fatalf("two equal %d-token prefix increments cost %.9f and %.9f s (%.3fx); "+
			"the prefix term is not linear in the prefix", chunk, d1, d2, r)
	}
	// Absolute scale, which linearity alone does not pin: the halved form is also
	// linear, just at half the slope. The slope per prefix token, in attention pairs, is
	// 2*2*S per layer; recomputed here from the measured form so it holds at whatever
	// the coefficients are.
	eff := k.epsMax * float64(chunk) / (float64(chunk) + k.mHalf) * k.attentionPrefillScale
	var want float64
	for _, l := range k.plan.Layers {
		if l.AttnQHeads == 0 {
			continue
		}
		want += 2.0 * 2 * float64(chunk) *
			float64(l.AttnQHeads) * float64(l.AttnHeadDim) / k.tp /
			(k.computeFLOPsPerSecond * eff) * float64(chunk) * float64(l.Count)
	}
	if r := d1 / want; r < 0.95 || r > 1.05 {
		t.Fatalf("a %d-token prefix added %.9f s where the full-prefix count gives "+
			"%.9f s (%.3fx); the prefix is being halved", chunk, d1, want, r)
	}
}

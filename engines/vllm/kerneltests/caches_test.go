// Moved from blis-latency-kernel/caches_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// Per-layer caches, as a graph states them (blis-schemas model.Cache), checked through
// what a simulator reads: SequenceVariableBytes, SequenceFixedBytes, PDTransferDemand and
// StepTime.

const v41 = "deepseek-v41-flash-h200-tp8.yaml"

// growth is the bytes a sequence's cache adds per context token far from any page edge.
func growth(k *Kernel) float64 {
	const lo, hi = 1 << 18, 1 << 20
	return float64(k.SequenceVariableBytes(hi)-k.SequenceVariableBytes(lo)) / (hi - lo)
}

// ownedRecordRate is what a fixture's owned growing caches hold per context token, before
// paging: each record over the tokens it covers, once per layer that owns it, on the graph
// as served on the fixture's chip and under the cache dtype its release resolves.
func ownedRecordRate(t *testing.T, in Inputs) float64 {
	t.Helper()
	family, _ := in.Chip.CapabilityMajor()
	g, err := in.Model.ForCapability(family)
	if err != nil {
		t.Fatal(err)
	}
	pool := in.Deployment.Pools[0]
	dtype := rulesFor(in.Scenario.EngineVersion).ResolveCacheDType(g.Global.Architecture,
		family, pool.Parallel.TP, mustDecode(t, pool).CacheDType)
	byID := map[string]model.LayerKind{}
	for _, lk := range g.LayerKinds {
		byID[lk.ID] = lk
	}
	var rate float64
	for _, id := range g.Stack.Expand() {
		for _, n := range byID[id].Nodes {
			for _, c := range n.Caches {
				rate += c.BytesPerContextToken(dtype)
			}
		}
	}
	return rate
}

// The capacity a sequence takes grows at the rate of the records its layers OWN, not at one
// per-model width times every attention layer: DeepSeek-V4.1-Flash holds about 1,790 B a
// context token on SM90, where the inferred figure was 20,480. Paging adds at most the page
// alignment's rounding.
func TestStatedCachesGrowAtTheOwnedRecordRate(t *testing.T) {
	in := fixtureInputs(t, v41)
	want := ownedRecordRate(t, in)
	got := growth(build(t, in))
	if got < want || got > want*1.1 {
		t.Errorf("a sequence grows %.1f B a token; its owned records are %.1f B a token, "+
			"and paging adds at most a tenth", got, want)
	}

	// Inferring the caches instead -- the same graph with every cache statement removed --
	// charges every attention layer a full-length cache of its read width, which can only
	// be more.
	stripped := fixtureInputs(t, v41)
	stripCaches(stripped.Model)
	if inferred := growth(build(t, stripped)); inferred <= got {
		t.Errorf("inferred caches grow %.1f B a token, no more than the %.1f the owned "+
			"records hold", inferred, got)
	}
}

// stripCaches removes every cache statement and every reference to one, which is a graph
// that leaves its caches to inference.
func stripCaches(g *model.Graph) {
	for i := range g.LayerKinds {
		for j := range g.LayerKinds[i].Nodes {
			n := &g.LayerKinds[i].Nodes[j]
			n.Caches, n.WritesCache = nil, ""
			if n.Context != nil {
				n.Context.ReadsCache = ""
				n.Context.TokensPerEntry = max(n.Context.TokensPerEntry, 1)
			}
			if n.WritesCache == "" && n.PerTokens > 0 {
				n.PerTokens = 0
			}
		}
	}
	for i := range g.ArchOverrides {
		g.ArchOverrides[i].Caches = nil
	}
}

// Stating caches is a per-NODE opt-in. A layer whose scorer states its index cache while its
// primary attention states none keeps the primary's inferred cache: it is resident, and its
// decode read grows with context.
func TestCacheStatementsArePerNode(t *testing.T) {
	full := build(t, fixtureInputs(t, v41))
	in := fixtureInputs(t, v41)
	stripped := 0
	for i := range in.Model.LayerKinds {
		for j := range in.Model.LayerKinds[i].Nodes {
			n := &in.Model.LayerKinds[i].Nodes[j]
			switch {
			case n.Op == model.OpAttention && n.Role == "":
				stripped += len(n.Caches)
				n.Caches = nil
			case n.WritesCache != "" && n.WritesCache != "index":
				n.WritesCache, n.PerTokens = "", 0
			case n.Context != nil && n.Context.ReadsCache != "":
				n.Context.ReadsCache = ""
			}
		}
	}
	for i := range in.Model.ArchOverrides {
		var kept []model.CacheOverride
		for _, c := range in.Model.ArchOverrides[i].Caches {
			if c.Name == "index" {
				kept = append(kept, c)
			}
		}
		in.Model.ArchOverrides[i].Caches = kept
	}
	if stripped == 0 {
		t.Fatal("no primary attention stated a cache to strip")
	}
	editEngine(t, &in, func(e *engineconfig.Engine) { e.Speculative = nil })
	partial := build(t, in)
	if p, f := growth(partial), growth(full); p <= f {
		t.Errorf("with its primaries' caches inferred a sequence grows %.1f B a token, no "+
			"more than the %.1f the stated records hold", p, f)
	}
	// Against the graph that states nothing: the primaries are inferred alike in both, and
	// the scorers' stated index caches only add to them. A layer whose scorer's statement
	// silenced its primary's inferred cache would grow less.
	none := fixtureInputs(t, v41)
	stripCaches(none.Model)
	editPoolEngine(t, &none.Deployment.Pools[0], func(e *engineconfig.Engine) { e.Speculative = nil })
	if p, n := growth(partial), growth(build(t, none)); p <= n {
		t.Errorf("stating only the scorers' caches grows %.1f B a token, no more than "+
			"stating none (%.1f): a scorer's statement dropped its primary's cache", p, n)
	}
	short := partial.StepTime(decodeBatch(64, 1, 8192)).PerResource[kernel.ResourceHBM]
	long := partial.StepTime(decodeBatch(64, 1, 131072)).PerResource[kernel.ResourceHBM]
	if long <= short {
		t.Errorf("an inferred primary's decode read is flat in context: %v at 8K, %v at 128K",
			short, long)
	}
}

// A window ring is bounded per sequence: it is in SequenceFixedBytes, and a longer window
// grows that term and leaves the per-token growth alone.
func TestWindowRingsAreFixedPerSequence(t *testing.T) {
	base := build(t, fixtureInputs(t, v41))
	if base.SequenceFixedBytes() <= 0 {
		t.Fatal("a graph with window caches reports no fixed per-sequence bytes")
	}
	in := fixtureInputs(t, v41)
	for i := range in.Model.LayerKinds {
		for j := range in.Model.LayerKinds[i].Nodes {
			n := &in.Model.LayerKinds[i].Nodes[j]
			for c := range n.Caches {
				if w := n.Caches[c].Window; n.Caches[c].Kind == model.CacheWindow {
					// The read keeps whatever it reads beyond the ring (a parallel-block
					// draft also reads its block positions).
					n.Window += 3 * w
					n.Caches[c].Window = 4 * w
				}
			}
		}
	}
	wide := build(t, in)
	if wide.SequenceFixedBytes() <= base.SequenceFixedBytes() {
		t.Errorf("a 4x window holds %d fixed bytes a sequence, not more than %d",
			wide.SequenceFixedBytes(), base.SequenceFixedBytes())
	}
	if g, w := growth(base), growth(wide); g != w {
		t.Errorf("widening the window moved the per-token growth from %.1f to %.1f", g, w)
	}
}

// The drafter's rings are resident only while the deployment speculates.
func TestDraftCachesAreHeldOnlyWhileSpeculating(t *testing.T) {
	spec := build(t, fixtureInputs(t, v41))
	in := fixtureInputs(t, v41)
	editEngine(t, &in, func(e *engineconfig.Engine) { e.Speculative = nil })
	plain := build(t, in)
	if spec.SequenceFixedBytes() <= plain.SequenceFixedBytes() {
		t.Errorf("speculating holds %d fixed bytes a sequence, not more than %d without",
			spec.SequenceFixedBytes(), plain.SequenceFixedBytes())
	}
}

// Which layer kind a graph declares last is not a property of the model. Reordering the
// declarations changes no price, and DeepSeek-V4.1-Flash, whose last declaration is a draft
// layer outside its target stack, is served the sparse-MLA cache dtype its target stack
// holds.
func TestLayerKindDeclarationOrderDoesNotPrice(t *testing.T) {
	a := build(t, fixtureInputs(t, v41))
	in := fixtureInputs(t, v41)
	lk := in.Model.LayerKinds
	for i, j := 0, len(lk)-1; i < j; i, j = i+1, j-1 {
		lk[i], lk[j] = lk[j], lk[i]
	}
	b := build(t, in)
	if a.FixedBytes() != b.FixedBytes() {
		t.Errorf("reversing the declarations moved FixedBytes from %+v to %+v",
			a.FixedBytes(), b.FixedBytes())
	}
	for _, n := range []int{1, 4096, 1 << 20} {
		if a.SequenceVariableBytes(n) != b.SequenceVariableBytes(n) {
			t.Errorf("reversing the declarations moved SequenceVariableBytes(%d)", n)
		}
	}
	for name, bt := range batchShapes() {
		if a.StepTime(bt).NoOverlap != b.StepTime(bt).NoOverlap {
			t.Errorf("%s: reversing the declarations moved the step time", name)
		}
	}

	ds := fixtureInputs(t, v41)
	editPoolEngine(t, &ds.Deployment.Pools[0], func(e *engineconfig.Engine) { e.CacheDType = "fp8_ds_mla" })
	if _, err := newKernel(ds); err != nil {
		t.Errorf("fp8_ds_mla refused for a sparse-MLA target stack: %v", err)
	}
}

// The graph is resolved for the chip it runs on: SM100's records are smaller than SM90's
// (and its compressed cache NVFP4 where mega attention serves it), so the same deployment on
// a B200 holds less a token. A chip that does not say which generation it is cannot run a
// graph that distinguishes generations.
func TestCachesFollowTheChipsArchitecture(t *testing.T) {
	h200 := build(t, fixtureInputs(t, v41))
	in := fixtureInputs(t, v41)
	b200 := chipInputs(t, "b200")
	in.Chip, in.Scenario.Cluster.Hardware = b200, "b200"
	if g, h := growth(build(t, in)), growth(h200); g >= h {
		t.Errorf("on B200 a sequence grows %.1f B a token, not less than H200's %.1f", g, h)
	}

	anon := fixtureInputs(t, v41)
	anon.Chip.ComputeCapability = ""
	if _, err := newKernel(anon); err == nil {
		t.Error("a chip with no compute capability ran a graph that distinguishes generations")
	}
}

// auto is the engine fitting max_model_len to memory, capped at the model's window, and the
// kernel takes it at that cap: auto prices as stating the window does.
func TestAutoMaxModelLenPricesAsTheModelsWindow(t *testing.T) {
	in := fixtureInputs(t, v41)
	editEngine(t, &in, func(e *engineconfig.Engine) { e.MaxModelLen = engineconfig.MaxModelLenAuto })
	auto := build(t, in)
	in = fixtureInputs(t, v41)
	editEngine(t, &in, func(e *engineconfig.Engine) {
		e.MaxModelLen = engineconfig.MaxModelLen(in.Model.Global.DefaultWindow())
	})
	stated := build(t, in)
	if auto.SequenceFixedBytes() != stated.SequenceFixedBytes() {
		t.Errorf("auto holds %d fixed bytes, the stated window %d",
			auto.SequenceFixedBytes(), stated.SequenceFixedBytes())
	}
	for name, bt := range batchShapes() {
		if auto.StepTime(bt).NoOverlap != stated.StepTime(bt).NoOverlap {
			t.Errorf("%s: auto and the stated window price differently", name)
		}
	}
}

// A handoff moves the owned caches the way NIXL does: never more than the sequence holds,
// a window ring clipped to the pages its window spans, and the drafter's caches not at all.
func TestHandoffMovesOnlyWhatTheTargetOwns(t *testing.T) {
	k := build(t, fixtureInputs(t, v41))
	from, to := kernel.Placement{Node: 0}, kernel.Placement{Node: 1}
	for _, n := range []int{1, 1000, 8192, 131072} {
		moved := k.PDTransferDemand(n, from, to).Bytes
		held := k.SequenceVariableBytes(n) + k.SequenceFixedBytes()
		if moved > held {
			t.Errorf("a %d-token handoff moves %d bytes, more than the %d a sequence holds",
				n, moved, held)
		}
	}
	// Far from the start, a handoff grows at the owned records' rate: the rings stop
	// growing once they span their window.
	lo, hi := 1<<18, 1<<20
	rate := float64(k.PDTransferDemand(hi, from, to).Bytes-
		k.PDTransferDemand(lo, from, to).Bytes) / float64(hi-lo)
	if want := ownedRecordRate(t, fixtureInputs(t, v41)); rate < want || rate > 1.1*want {
		t.Errorf("a handoff grows %.1f B a token; the owned records are %.1f", rate, want)
	}
}

// A window bounds the decode read of every attention kind, not only sparse MLA: past its
// window, a windowed layer's HBM time does not grow with context.
func TestAWindowBoundsTheDecodeReadOfEveryKind(t *testing.T) {
	for _, c := range []struct {
		scenario string
		kind     model.AttentionKind
	}{
		{"minimax-m25-h200-tp8.yaml", model.AttentionSWA},
		{"aisimulate/deepseek-v3-h200-fp8-sglang-tp8.yaml", model.AttentionMLA},
	} {
		t.Run(string(c.kind), func(t *testing.T) {
			in := fixtureInputs(t, c.scenario)
			for i := range in.Model.LayerKinds {
				for j := range in.Model.LayerKinds[i].Nodes {
					n := &in.Model.LayerKinds[i].Nodes[j]
					if n.Op == model.OpAttention && n.Role == "" {
						n.AttentionKind, n.Window = c.kind, 128
					}
				}
			}
			k := build(t, in)
			short := k.StepTime(decodeBatch(64, 1, 8192)).PerResource[kernel.ResourceHBM]
			long := k.StepTime(decodeBatch(64, 1, 131072)).PerResource[kernel.ResourceHBM]
			if long != short {
				t.Errorf("a 128-token window read %v at 8K and %v at 128K", short, long)
			}
		})
	}
}

// A cache sized at the record the graph states for the engine's own default is not an
// assumption: only a deployment running another dtype, which the graph lists no record for,
// is recorded.
func TestTheDefaultCacheRecordIsNotAnAssumption(t *testing.T) {
	for _, chip := range []string{"h200", "b200"} {
		in := fixtureInputs(t, v41)
		if chip != "h200" {
			in.Chip, in.Scenario.Cluster.Hardware = chipInputs(t, chip), chip
		}
		if recorded(build(t, in), "cache_record") {
			t.Errorf("%s: the default cache dtype's record was recorded as an assumption", chip)
		}
	}
}

// vLLM's block for DeepSeek-V4.1-Flash on SM90 at block 64 under fp8_ds_mla: the full-attention
// group's pages, 3 x 19,008 + 37,440 + 3 x 4,608 + 8,640 B, which every group's block costs
// (_get_kv_cache_bytes_per_block, vllm/v1/core/kv_cache_utils.py:1606-1632). It is the figure
// vLLM's own functions return for the same layers in blis-schemas' golden vectors
// (rules/testdata/kvgroups), where the 43 window layers fall into 8 groups of at most six.
const v41PoolBlock = 116928

// The stated caches are sized in whole blocks of the engine's shared pool, group by group: a
// sequence grows one block per 64 tokens of context, and holds in steady decode six pages in
// each of the eight window groups and one in the compressor-ring group, 49 blocks
// (inference-sim/blis-latency-kernel#46). At admission each window group holds its cap,
// cdiv(127 + 2 x 8,192, 32) + 1 = 517 pages, the two batches asynchronous scheduling keeps in
// flight (vllm/config/vllm.py:592-613; vllm/v1/kv_cache_interface.py:840-866).
func TestStatedCachesAreSizedInWholePoolBlocks(t *testing.T) {
	k := build(t, fixtureInputs(t, v41))
	if got := k.SequenceVariableBytes(64); got != v41PoolBlock {
		t.Errorf("64 tokens hold %d B, want one pool block of %d", got, v41PoolBlock)
	}
	if got := k.SequenceVariableBytes(131072); got != 2048*v41PoolBlock {
		t.Errorf("131,072 tokens hold %d B, want 2,048 pool blocks", got)
	}
	if got := k.SequenceFixedBytes(); got != (8*6+1)*v41PoolBlock {
		t.Errorf("steady decode holds %d B a sequence, want 49 pool blocks, %d", got,
			(8*6+1)*v41PoolBlock)
	}
	if got := k.SequenceAdmissionBytes(); got != (8*517+1)*v41PoolBlock {
		t.Errorf("admission reserves %d B a sequence, want 4,137 pool blocks, %d", got,
			(8*517+1)*v41PoolBlock)
	}
}

// PROPERTIES of the admission reservation: never below the steady decode hold, growing
// with the token budget, and the same as it for a model with no window cache.
func TestAdmissionReservationProperties(t *testing.T) {
	prev := int64(0)
	for _, mnbt := range []int{2048, 8192, 16384, 32768} {
		in := fixtureInputs(t, v41)
		editEngine(t, &in, func(e *engineconfig.Engine) { e.MaxNumBatchedTokens = mnbt })
		k := build(t, in)
		adm := k.SequenceAdmissionBytes()
		if adm < k.SequenceFixedBytes() {
			t.Errorf("mnbt %d: admission %d B below the steady hold %d", mnbt, adm,
				k.SequenceFixedBytes())
		}
		if adm <= prev {
			t.Errorf("mnbt %d: admission %d B, not above a smaller budget's %d", mnbt, adm, prev)
		}
		prev = adm
	}
	v3 := fixture(t, memoryFixture)
	if v3.SequenceAdmissionBytes() != v3.SequenceFixedBytes() {
		t.Errorf("deepseek-v3 states no window cache, yet admits at %d B against %d",
			v3.SequenceAdmissionBytes(), v3.SequenceFixedBytes())
	}
}

// THE BOUNDED HOLD AS A FUNCTION OF WHAT IS IN FLIGHT. SequenceBoundedBytes(1 + K) is the
// steady decode hold and SequenceBoundedBytes(max_in_flight_tokens) the admission cap; the
// hold never falls as tokens in flight or tokens computed grow; and once a request has
// computed a window's worth, what it has computed no longer matters, as vLLM frees every
// block wholly behind the window (SlidingWindowManager.get_num_skipped_tokens and
// remove_skipped_blocks, vllm/v1/core/single_type_kv_cache_manager.py:687-725, 1153-1175 at
// v0.31.0). A request that has seen nothing holds only its compressor rings.
func TestSequenceBoundedBytesProperties(t *testing.T) {
	k := build(t, fixtureInputs(t, v41))
	if got, want := k.SequenceBoundedBytes(6), k.SequenceFixedBytes(); got != want {
		t.Errorf("SequenceBoundedBytes(1 + K) = %d, want SequenceFixedBytes %d", got, want)
	}
	if got, want := k.SequenceBoundedBytes(2*8192), k.SequenceAdmissionBytes(); got != want {
		t.Errorf("SequenceBoundedBytes(max_in_flight_tokens) = %d, want "+
			"SequenceAdmissionBytes %d", got, want)
	}
	// A ring crosses a page edge where window - 1 + in flight passes a multiple of 32:
	// 33 in flight page 160 tokens in 5 pages + 1, 34 page 161 in 6 + 1, one more block in
	// each of the eight window groups.
	if d := k.SequenceBoundedBytes(34) - k.SequenceBoundedBytes(33); d != 8*v41PoolBlock {
		t.Errorf("the 34th token in flight adds %d B, want one block in each of 8 window "+
			"groups, %d", d, 8*v41PoolBlock)
	}
	const window = 128
	prevFlight := int64(-1)
	for _, f := range []int{0, 1, 6, 31, 32, 33, 512, 8192, 16384, 65536} {
		b := k.SequenceBoundedBytes(f)
		if b < prevFlight {
			t.Errorf("%d tokens in flight hold %d B, below fewer tokens' %d", f, b, prevFlight)
		}
		prevFlight = b
		prevComputed := int64(-1)
		for _, c := range []int{0, 1, 31, 64, 127, 128, 129, 1000, 100000} {
			at := k.SequenceBoundedBytesAt(c, f)
			if at < prevComputed {
				t.Errorf("(%d computed, %d in flight) holds %d B, below fewer computed's %d",
					c, f, at, prevComputed)
			}
			prevComputed = at
			if c >= window && at != b {
				t.Errorf("(%d computed, %d in flight) holds %d B; past the window it must be "+
					"SequenceBoundedBytes(%d) = %d", c, f, at, f, b)
			}
			if at > b {
				t.Errorf("(%d computed, %d in flight) holds %d B, above the window's cap %d",
					c, f, at, b)
			}
		}
	}
	if got, want := k.SequenceBoundedBytesAt(0, 0), int64(v41PoolBlock); got != want {
		t.Errorf("a request that has seen nothing holds %d B, want its ring group's one "+
			"block, %d", got, want)
	}
}

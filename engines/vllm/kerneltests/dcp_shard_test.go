// Moved from blis-latency-kernel/dcp_shard_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

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

// New refuses exactly the context-parallel layouts the engine refuses, and admits their
// neighbours.
//
// vLLM v0.31.0 checks two things at startup (vllm/config/parallel.py:563-578):
//
//	pcp == 1:  tp % dcp == 0                  "DCP reuses the TP ranks"
//	pcp  > 1:  dcp in {1, pcp, tp*pcp}         "disabled, span the PCP axis, or span the
//	                                            full TP x PCP axis"
//
// and its world size is pp * tp * pcp (:899-903), so PCP adds ranks where DCP adds none.
// blis-schemas v0.2.2 enforces all three, and New runs its field validation, so a kernel is
// never built for a layout that does not start. Asserted here because it is the contract a
// caller sweeping layouts relies on: a refused layout is an error, never a price.
//
// glm5, a DSA sparse-MLA stack, because PCP with DCP runs only on DSA layers; the plain-MLA
// refusal is the last row.
func TestNewRefusesTheContextParallelLayoutsTheEngineRefuses(t *testing.T) {
	build := func(fixture string, tp, pcp, dcp, nodes int) error {
		t.Helper()
		in := fixtureInputs(t, fixture)
		pool := &in.Deployment.Pools[0]
		pool.Parallel.TP, pool.Parallel.PCP, pool.Parallel.DCP = tp, pcp, dcp
		pool.Nodes, in.Scenario.Cluster.Nodes = nodes, nodes
		if nodes > 1 {
			in.Scenario.Cluster.Fabric = "ib-400g"
			f, err := schemas.LoadFabric(filepath.Join(catalogRoot, "networks", "ib-400g.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			in.Fabric = f
		}
		_, err := newKernel(in)
		return err
	}
	for _, c := range []struct {
		name                string
		fixture             string
		tp, pcp, dcp, nodes int
		admitted            bool
	}{
		{"dcp divides tp", dcpSparseFixture, 8, 1, 4, 1, true},
		{"dcp does not divide tp", dcpSparseFixture, 8, 1, 3, 1, false},
		{"dcp wider than tp", dcpSparseFixture, 4, 1, 8, 1, false},
		{"dcp spans the pcp axis", dcpSparseFixture, 8, 2, 2, 2, true},
		{"dcp spans tp x pcp", dcpSparseFixture, 4, 2, 8, 1, true},
		{"dcp is neither", dcpSparseFixture, 8, 2, 4, 2, false},
		{"pcp fits the GPUs", dcpSparseFixture, 4, 2, 1, 1, true},
		{"pcp needs more GPUs than the pool has", dcpSparseFixture, 8, 2, 1, 1, false},
		// MLAAttention.supports_pcp_dcp is False (mla_attention.py:440-442, raised at
		// :684-687); only DeepseekV32Attention opts in.
		{"plain mla runs pcp alone", dcpMLAFixture, 4, 2, 1, 1, true},
		{"plain mla refuses pcp with dcp", dcpMLAFixture, 4, 2, 8, 1, false},
	} {
		err := build(c.fixture, c.tp, c.pcp, c.dcp, c.nodes)
		if c.admitted && err != nil {
			t.Errorf("%s (tp=%d pcp=%d dcp=%d on %d node(s)): refused, but the engine "+
				"starts it: %v", c.name, c.tp, c.pcp, c.dcp, c.nodes, err)
		}
		if !c.admitted && err == nil {
			t.Errorf("%s (tp=%d pcp=%d dcp=%d on %d node(s)): priced, but the engine "+
				"refuses it at startup", c.name, c.tp, c.pcp, c.dcp, c.nodes)
		}
	}
}

// THE ACCEPTANCE CRITERION. Two deployments differing only in dcp must no longer price
// identically, and widening the shard must keep lowering what a rank READS.
//
// The read, not the whole step. DCP trades decode time for capacity: each width shards
// the read further and adds a combine whose payload and floors grow with the group, so the
// step need not fall monotonically, and at 32 requests on a 32k context it does not -- on
// deepseek-v3 the step rises from dcp=4 to dcp=8 once the log-sum-exp all-gather, the
// third collective vLLM launches per layer (vllm/v1/attention/ops/dcp.py:458 at v0.31.0),
// is priced. An earlier form of this test required the step to fall at every width, which
// held only because that collective was left out.
//
// Four widths rather than two, deliberately: a mutation replacing the divisor with any
// constant passes a two-point check. The pooled-scan commit found the same thing.
func TestDCPLowersTheDecodeReadAndWideningItLowersFurther(t *testing.T) {
	for _, fixture := range []string{dcpMLAFixture, dcpSparseFixture} {
		t.Run(fixture, func(t *testing.T) {
			batch := decodeBatch(32, 1, 32768)
			var prevRead, prevStep float64
			for i, dcp := range []int{1, 2, 4, 8} {
				est := dcpKernel(t, fixture, dcp).StepTime(batch)
				read := est.PerResource[kernel.ResourceHBM].Seconds()
				step := est.NoOverlap.Seconds()
				if i > 0 && read >= prevRead {
					t.Errorf("dcp=%d read %.6f ms of HBM, not below the %.6f ms of the "+
						"previous width: a wider shard must read less",
						dcp, read*1e3, prevRead*1e3)
				}
				if i > 0 && step == prevStep {
					t.Errorf("dcp=%d priced the step identically to the previous width",
						dcp)
				}
				prevRead, prevStep = read, step
			}
		})
	}
}

// The decode READ must divide by the shard exactly, which a difference of differences
// isolates: with read(d) = R/d plus terms that do not depend on d,
//
//	(t(1) - t(2)) / (t(2) - t(4)) = (R - R/2) / (R/2 - R/4) = 2
//
// Every unsharded term -- the floor, the GEMMs, the MoE, the host -- cancels. So this
// kills a divisor that merely decreases in dcp: an off-by-one, a square root, dcp/2.
//
// Charged on the HBM resource because that is what a decode read binds on.
func TestDCPDividesTheDecodeReadByExactlyTheShard(t *testing.T) {
	for _, fixture := range []string{dcpMLAFixture, dcpSparseFixture} {
		t.Run(fixture, func(t *testing.T) {
			batch := decodeBatch(32, 1, 32768)
			read := func(dcp int) float64 {
				return dcpKernel(t, fixture, dcp).StepTime(batch).
					PerResource[kernel.ResourceHBM].Seconds()
			}
			h1, h2, h4 := read(1), read(2), read(4)
			if h2 >= h1 || h4 >= h2 {
				t.Fatalf("the HBM term is not falling with the shard: %.6f, %.6f, %.6f ms",
					h1*1e3, h2*1e3, h4*1e3)
			}
			ratio := (h1 - h2) / (h2 - h4)
			if math.Abs(ratio-2) > 1e-6 {
				t.Errorf("(t1-t2)/(t2-t4) is %.9f, want 2: the read is not dividing by "+
					"exactly the shard width", ratio)
			}
		})
	}
}

// A hybrid stack must shard only the half the engine shards. gpt-oss-120b alternates 18
// gqa layers with 18 sliding-window ones, and a sliding window is a configuration vLLM
// refuses to start under DCP at all -- so a deployment-wide divisor would shard layers
// whose cache the engine never shards.
//
// Asserted as a comparison rather than a value: the hybrid's decode read must fall by
// strictly less than an all-sharding model's does. Measured here: about 15% against about
// 55%.
func TestDCPDoesNotShardASlidingWindowLayer(t *testing.T) {
	fall := func(fixture string) float64 {
		batch := decodeBatch(32, 1, 32768)
		read := func(dcp int) float64 {
			return dcpKernel(t, fixture, dcp).StepTime(batch).
				PerResource[kernel.ResourceHBM].Seconds()
		}
		at1 := read(1)
		return (at1 - read(8)) / at1
	}
	hybrid, allSharding := fall(dcpHybridFixture), fall(dcpMLAFixture)
	if hybrid <= 0 {
		t.Errorf("the hybrid's gqa half should still shard, but its read did not fall")
	}
	if hybrid >= allSharding {
		t.Errorf("the hybrid read fell %.1f%% and the all-sharding model %.1f%%; a stack "+
			"half of whose layers are sliding-window must shard strictly less",
			hybrid*100, allSharding*100)
	}
}

// Capacity: the KV a rank holds must fall with the shard, and the paging must happen AFTER
// the division rather than before it -- which is the engine's order
// (max_memory_usage_bytes divides max_model_len by dcp_world_size, then rounds to blocks;
// page_size_bytes never sees dcp).
//
// Dividing a per-token byte figure instead would make a page fractional and understate
// occupancy by up to dcp below block_size*dcp tokens. This asserts the whole-page
// behaviour that distinguishes the two.
func TestDCPShardsCapacityByTokensBeforePaging(t *testing.T) {
	k1 := dcpKernel(t, dcpMLAFixture, 1)
	k8 := dcpKernel(t, dcpMLAFixture, 8)

	// Well above block_size*dcp, where the two orderings agree and the division is exact.
	long1, long8 := k1.SequenceVariableBytes(32768), k8.SequenceVariableBytes(32768)
	if long8*8 != long1 {
		t.Errorf("at 32768 tokens: dcp=1 holds %d and dcp=8 holds %d; the shard should be "+
			"exact at a context this long", long1, long8)
	}

	// Below one page per rank. Sharding the TOKEN count and then paging gives one whole
	// page; dividing bytes per token would give an eighth of one.
	perRankPage := k8.SequenceVariableBytes(1)
	if perRankPage != k1.SequenceVariableBytes(1) {
		t.Errorf("a single token occupies %d bytes at dcp=8 against %d at dcp=1; one "+
			"token is one page on one rank either way, because a page is not divisible",
			perRankPage, k1.SequenceVariableBytes(1))
	}

	// The hybrid is left unsharded, which is the stated coverage limit rather than an
	// oversight: kvBytesPerToken is one whole-model scalar, so a stack mixing sharding
	// and non-sharding kinds cannot be expressed exactly here. Over-stating is the safe
	// direction, and the engine refuses the configuration anyway.
	h1 := dcpKernel(t, dcpHybridFixture, 1).SequenceVariableBytes(32768)
	h8 := dcpKernel(t, dcpHybridFixture, 8).SequenceVariableBytes(32768)
	if h1 != h8 {
		t.Errorf("the hybrid's capacity moved from %d to %d; a stack holding a "+
			"sliding-window cache is deliberately left unsharded here", h1, h8)
	}
}

// DCP must not shard a prefill-to-decode transfer. A prefill pool holds the WHOLE cache
// for a request -- DCP shards the decode cache -- so the bytes crossing between pools are
// the whole request's whatever the decode pool's width is.
//
// THIS IS THE TEST THAT RULES OUT THE OBVIOUS IMPLEMENTATION. Dividing kvBytesPerToken
// once at construction would reach all three decode read sites in one edit, and it would
// also divide this, making every PD transfer exactly dcp times too cheap at every token
// count. That is why the shard is applied to token counts instead.
func TestDCPDoesNotShardAPrefillToDecodeTransfer(t *testing.T) {
	from := kernel.Placement{Node: 0, Rack: 0}
	to := kernel.Placement{Node: 1, Rack: 0}
	want := dcpKernel(t, dcpMLAFixture, 1).PDTransferTime(4096, from, to)
	for _, dcp := range []int{2, 4, 8} {
		if got := dcpKernel(t, dcpMLAFixture, dcp).PDTransferTime(4096, from, to); got != want {
			t.Errorf("dcp=%d priced a PD transfer at %v against %v at dcp=1; a prefill "+
				"pool sends the whole request's KV", dcp, got, want)
		}
	}
}

// DCP must not touch prefill. It shards the DECODE cache; a prefill chunk computes every
// position it was scheduled. Prefill-context parallelism is the axis that splits a prefill,
// and conflating the two is the easiest thing to get backwards.
func TestDCPDoesNotShardPrefill(t *testing.T) {
	prefill := decodeBatch(1, 4096, 4096)
	want := dcpKernel(t, dcpMLAFixture, 1).StepTime(prefill).NoOverlap
	for _, dcp := range []int{2, 8} {
		if got := dcpKernel(t, dcpMLAFixture, dcp).StepTime(prefill).NoOverlap; got != want {
			t.Errorf("dcp=%d priced a prefill-only step at %v against %v at dcp=1",
				dcp, got, want)
		}
	}
}

// The per-call attention floor must NOT divide with the shard, and the bytes MUST.
//
// An MLA decode's floor is 51.5-89.5us against full attention's 9.5-19.5us precisely
// because its per-call setup reads a latent cache. That is a launch-and-descriptor cost,
// paid once per rank per invocation however many tokens the rank holds; sharding a
// sequence does not make a kernel launch cheaper. Dividing it by 8 would put an MLA floor
// at 6.4-11.2us -- BELOW full attention's -- for the kernel the registry measures as
// several times costlier to set up. This is the modelling judgement the issue asked to be
// made on the evidence rather than inherited from whichever behaviour falls out of the
// byte arithmetic.
//
// SEPARATING THE TWO TERMS. The decode read is affine in context: slope*context +
// intercept, where the slope is bytes-per-token over the rate and the intercept is the
// per-layer floors. Two contexts at one width recover both. The shard must then divide the
// SLOPE exactly and leave the INTERCEPT untouched.
//
// Asserting it this way rather than as a lower bound on the step is what gives the test
// teeth: a mutation scaling the floor by the shard survives any assertion that the step
// exceeds its floor total, because on this fixture the GEMM and MoE terms are five times
// the floors and swamp them. It also has to be measured at a context long enough to
// actually shard -- at a one-token context the shard is 1 by construction, so the
// mutation is inert there and the test would pass for the wrong reason.
func TestDCPDividesTheReadRateButNotThePerCallFloor(t *testing.T) {
	// The affine fit, from two contexts well above block_size*dcp.
	fit := func(dcp int) (slope, intercept float64) {
		k := dcpKernel(t, dcpMLAFixture, dcp)
		read := func(ctx int) float64 {
			return k.StepTime(decodeBatch(1, 1, ctx)).
				PerResource[kernel.ResourceHBM].Seconds()
		}
		lo, hi := 8192, 16384
		atLo, atHi := read(lo), read(hi)
		slope = (atHi - atLo) / float64(hi-lo)
		return slope, atLo - slope*float64(lo)
	}

	slope1, intercept1 := fit(1)
	if slope1 <= 0 || intercept1 <= 0 {
		t.Fatalf("the decode read is not affine and positive in context: slope %.6e, "+
			"intercept %.6f ms", slope1, intercept1*1e3)
	}
	for _, dcp := range []int{2, 8} {
		slope, intercept := fit(dcp)
		// The rate term divides by exactly the shard.
		if want := slope1 / float64(dcp); math.Abs(slope-want)/want > 1e-9 {
			t.Errorf("dcp=%d: the per-token read rate is %.6e, want %.6e (the dcp=1 "+
				"figure over %d)", dcp, slope, want, dcp)
		}
		// The floor does not move at all.
		if math.Abs(intercept-intercept1) > 1e-12 {
			t.Errorf("dcp=%d: the per-call floor moved to %.9f ms from %.9f ms; a "+
				"launch cost is paid once per rank whatever the shard",
				dcp, intercept*1e3, intercept1*1e3)
		}
	}
}

// The width a consumer reads must be the resolved one, and must floor at one so an absent
// field reads as "not sharded" rather than zero.
func TestDecodeContextParallelWidthComesFromTheResolvedLayout(t *testing.T) {
	if got := dcpKernel(t, dcpMLAFixture, 4).DecodeContextParallelWidth(); got != 4 {
		t.Errorf("got %d, want 4", got)
	}
	// The fixtures state no dcp at all, which is the absent case.
	if got := fixture(t, dcpMLAFixture).DecodeContextParallelWidth(); got != 1 {
		t.Errorf("an unstated width read as %d, want 1", got)
	}
}

// THE DEPLOYMENT THIS WAS WRITTEN FOR, priced arm by arm.
//
// The GLM-5.3-Flash canary runs tp=1 with --prefill-context-parallel-size 8,
// --decode-context-parallel-size 8, --dcp-comm-backend ag_rs and a 64-token KV block. That
// shape is why the two axes had to be separated rather than collapsed: at tp=1 there is no
// tensor-parallel width to divide anything, so before this change NOTHING in the layout
// moved the price of that deployment at all. glm5 stands in for it: the pinned catalog
// predates GLM-5.3-Flash, and both are sparse-MLA stacks.
//
// THREE ARMS, NOT FOUR. A DCP-only arm at tp=1 is a layout the engine refuses: with PCP off
// "DCP reuses the TP ranks", so tp must be divisible by dcp (vllm/config/parallel.py:566-569
// at v0.31.0), and blis-schemas v0.2.2 now enforces that too. An earlier form of this test
// priced tp=1, dcp=8 anyway, so its DCP-only assertions described a deployment that does
// not start. DCP alone is exercised at tp=8 by the tests above, where it is admissible.
//
// The backend is STATED, as the canary states it. GLM's model hook would otherwise choose
// a2a with a replicated query projection (models/config.py:43-50), which is a different
// set of collectives; stating ag_rs is what the measured deployment did.
//
// Asserted as a matrix of non-interference rather than as figures: PCP must move prefill
// and only prefill, and adding DCP to it must move decode and capacity and not prefill.
func TestTheContextParallelArmsOfTheCanaryDeploymentPriceIndependently(t *testing.T) {
	build := func(pcp, dcp int) *Kernel {
		t.Helper()
		in := fixtureInputs(t, dcpSparseFixture)
		pool := &in.Deployment.Pools[0]
		pool.Parallel.TP = 1
		pool.Parallel.DP = 1
		pool.Parallel.PCP = pcp
		pool.Parallel.DCP = dcp
		editPoolEngine(t, pool, func(e *engineconfig.Engine) { e.BlockSize = 64 })
		editPoolEngine(t, pool, func(e *engineconfig.Engine) { e.DCPCommBackend = "ag_rs" })
		k, err := newKernel(in)
		if err != nil {
			t.Fatalf("tp=1 pcp=%d dcp=%d: %v", pcp, dcp, err)
		}
		return k
	}
	decode := decodeBatch(4, 1, 32768)
	prefill := decodeBatch(1, 2048, 2048)

	base := build(1, 1)
	pcpOnly := build(8, 1)
	both := build(8, 8)

	// PCP moves prefill and nothing else.
	if pcpOnly.StepTime(prefill).NoOverlap >= base.StepTime(prefill).NoOverlap {
		t.Error("pcp=8 did not lower the prefill step at tp=1, where no tensor-parallel " +
			"width can be doing the work instead")
	}
	if got, want := pcpOnly.StepTime(decode).NoOverlap,
		base.StepTime(decode).NoOverlap; got != want {
		t.Errorf("pcp=8 moved the decode step to %v from %v", got, want)
	}
	if got, want := pcpOnly.SequenceVariableBytes(32768),
		base.SequenceVariableBytes(32768); got != want {
		t.Errorf("pcp=8 moved the cache a rank holds to %d from %d", got, want)
	}

	// Adding DCP moves decode and capacity, and not an unchunked prefill: a prefill with no
	// computed prefix has no sharded context to read.
	if both.StepTime(decode).NoOverlap == pcpOnly.StepTime(decode).NoOverlap {
		t.Error("dcp=8 left the decode step where pcp alone put it; the shard and the " +
			"combine it adds both change a decode step")
	}
	if got, want := both.SequenceVariableBytes(32768),
		pcpOnly.SequenceVariableBytes(32768)/8; got != want {
		t.Errorf("dcp=8 holds %d bytes, want %d (an eighth of the unsharded %d)",
			got, want, pcpOnly.SequenceVariableBytes(32768))
	}
	if got, want := both.StepTime(prefill).NoOverlap,
		pcpOnly.StepTime(prefill).NoOverlap; got != want {
		t.Errorf("dcp=8 moved an unchunked prefill step to %v from %v", got, want)
	}
}

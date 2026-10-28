// Moved from blis-latency-kernel/dcp_knobs_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"math"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/vocab"
	"github.com/inference-sim/blis-vllm/rules/v0_29"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// The three decode-context-parallel engine knobs blis-schemas v0.2.2 made expressible --
// dcp_comm_backend, dcp_q_replicate, cp_kv_cache_interleave_size -- priced the way vLLM
// v0.31.0 runs them. resolve.ResolveDecodeContext and dcpDecodeCollectives carry the
// citations; these pin the behaviour.

// dcpVariant is a committed fixture with one pool's DCP, PCP, TP and node count set, and
// any engine edits applied, laid out as the engine needs it (pcpInputs adds the nodes and
// fabric a PCP layout requires).
func dcpVariant(t *testing.T, fixture string, tp, pcp, dcp int,
	edit func(*Inputs)) (*Kernel, error) {
	t.Helper()
	in := pcpInputs(t, fixture, 1)
	pool := &in.Deployment.Pools[0]
	pool.Parallel.TP, pool.Parallel.PCP, pool.Parallel.DCP = tp, pcp, dcp
	nodes := max((tp*max(pcp, 1)+7)/8, 1)
	pool.Nodes, in.Scenario.Cluster.Nodes = nodes, nodes
	if edit != nil {
		edit(&in)
	}
	return newKernel(in)
}

func mustDCPVariant(t *testing.T, fixture string, tp, pcp, dcp int, edit func(*Inputs)) *Kernel {
	t.Helper()
	k, err := dcpVariant(t, fixture, tp, pcp, dcp, edit)
	if err != nil {
		t.Fatalf("%s tp=%d pcp=%d dcp=%d: %v", fixture, tp, pcp, dcp, err)
	}
	return k
}

func withBackend(b string) func(*Inputs) {
	return func(in *Inputs) { editInputsEngine(in, func(e *engineconfig.Engine) { e.DCPCommBackend = b }) }
}

// scaleFloors returns a copy of the coefficient sets with every collective floor of one
// primitive multiplied by factor. It is how a test asks which collectives a layout
// launches without reading the kernel's internals: inflate one primitive's floors and see
// which step times move.
func scaleFloors(sets []*coefficient.Set, primitive string, factor float64) []*coefficient.Set {
	prefix := "collective_floor_" + primitive + "_"
	out := make([]*coefficient.Set, len(sets))
	for i, s := range sets {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		for j := range c.Coefficients {
			if strings.HasPrefix(c.Coefficients[j].Name, prefix) {
				c.Coefficients[j].Value *= factor
			}
		}
		out[i] = &c
	}
	return out
}

// WHICH COLLECTIVES EACH LAYOUT LAUNCHES, asked of the step rather than of the code.
//
// Inflating one primitive's floors tenfold moves a decode step exactly when that layout
// launches the primitive. On deepseek-v3 and glm5, whose graphs emit only all-reduces (and
// an all-to-all only with expert parallelism, which these layouts leave off), the
// decode-context combine is the only source of a reduce-scatter or an all-to-all. The
// PCP row is glm5 because PCP with DCP runs only on DSA sparse-MLA layers
// (resolveContextParallel); deepseek-v3's plain MLA refuses it.
//
//	ag_rs, pcp off   reduce-scatter                  (cp_lse_ag_out_rs, dcp.py:493)
//	a2a,   pcp off   all-to-all, and no reduce-scatter (dcp_a2a_lse_reduce, :939-1010)
//	ag_rs, pcp on    neither: it all-reduces instead  (cp_lse_ag_out_ar, :526; :1525-1531)
//
// A step whose combine launches the primitive must rise; one that does not must not move
// at all. Decode only -- the combine is a decode-row collective -- and a prefill step is
// checked not to move under either, since no combine runs on it.
func TestEachDCPLayoutLaunchesTheCombineTheEngineRuns(t *testing.T) {
	decode := decodeBatch(8, 1, 8192)
	prefill := decodeBatch(1, 2048, 2048)
	for _, c := range []struct {
		name               string
		fixture            string
		tp, pcp, dcp       int
		backend            string
		reduceScatter, a2a bool
	}{
		{"ag_rs without pcp", dcpMLAFixture, 8, 1, 8, "ag_rs", true, false},
		{"a2a without pcp", dcpMLAFixture, 8, 1, 8, "a2a", false, true},
		{"ag_rs with pcp", dcpSparseFixture, 2, 4, 4, "ag_rs", false, false},
	} {
		for _, prim := range []struct {
			name     string
			launched bool
		}{
			{"reduce_scatter", c.reduceScatter},
			{"alltoall", c.a2a},
		} {
			base := mustDCPVariant(t, c.fixture, c.tp, c.pcp, c.dcp, withBackend(c.backend))
			inflated := mustDCPVariant(t, c.fixture, c.tp, c.pcp, c.dcp, func(in *Inputs) {
				withBackend(c.backend)(in)
				in.Coefficients = scaleFloors(in.Coefficients, prim.name, 10)
			})
			before, after := base.StepTime(decode).NoOverlap, inflated.StepTime(decode).NoOverlap
			switch {
			case prim.launched && after <= before:
				t.Errorf("%s: inflating %s floors left the decode step at %v (was %v); "+
					"this combine launches a %s", c.name, prim.name, after, before, prim.name)
			case !prim.launched && after != before:
				t.Errorf("%s: inflating %s floors moved the decode step from %v to %v; "+
					"this combine launches no %s", c.name, prim.name, before, after, prim.name)
			}
			if a, b := base.StepTime(prefill).NoOverlap,
				inflated.StepTime(prefill).NoOverlap; a != b {
				t.Errorf("%s: inflating %s floors moved a prefill step from %v to %v; no "+
					"decode-context combine runs on a step without decode rows",
					c.name, prim.name, a, b)
			}
		}
	}
}

// AND WHAT IT DOES LAUNCH UNDER PCP: an LSE all-gather and an all-reduce over the
// decode-context group (cp_lse_ag_out_ar, vllm/v1/attention/ops/dcp.py:503-526, chosen at
// :1525-1531). Asserted positively, since the row above only shows what it does not launch.
// At tp=2, pcp=4, dcp=4 the decode-context group is the only 4-rank group a decode step
// touches -- the graph's own all-reduce is 2-rank, and the PCP gather runs on prefill rows
// only -- so inflating either 4-rank primitive must move a decode step.
func TestUnderPCPTheCombineIsAnLSEGatherAndAnAllReduce(t *testing.T) {
	decode := decodeBatch(8, 1, 8192)
	base := mustDCPVariant(t, dcpSparseFixture, 2, 4, 4, nil).StepTime(decode).NoOverlap
	for _, prim := range []string{"all_reduce_fp16_4rank", "all_gather_fp16_4rank"} {
		inflated := mustDCPVariant(t, dcpSparseFixture, 2, 4, 4, func(in *Inputs) {
			in.Coefficients = scaleFloors(in.Coefficients, prim, 10)
		}).StepTime(decode).NoOverlap
		if inflated <= base {
			t.Errorf("inflating %s floors left the decode step at %v (was %v); under PCP the "+
				"combine launches it over the decode-context group", prim, inflated, base)
		}
	}
}

// THE QUERY GATHER UNDER PCP. With prefill-context parallelism on, a DSA layer -- the only
// kind that runs PCP with DCP -- gathers its query over the tensor-parallel group only when
// the decode-context group spans the whole tp x pcp block, and not at all when it spans the
// PCP axis alone (vllm/models/deepseek_v32/attention.py:560-563 at v0.31.0).
//
// Asked the same way: at tp=2 the tensor-parallel all-gather is the 2-rank one, and the
// decode-context group is 4 or 8 ranks wide, so inflating the 2-rank all-gather floor moves
// the decode step by more exactly where that gather runs. The step's other 2-rank all-gather,
// the LM head's logits gather, is the same in both layouts, so it moves both alike and the
// difference between them is the query gather's.
func TestUnderPCPTheQueryIsGatheredOverTPOnlyWhenDCPSpansTheBlock(t *testing.T) {
	decode := decodeBatch(8, 1, 8192)
	moved := func(dcp int) time.Duration {
		base := mustDCPVariant(t, dcpSparseFixture, 2, 4, dcp, nil)
		inflated := mustDCPVariant(t, dcpSparseFixture, 2, 4, dcp, func(in *Inputs) {
			in.Coefficients = scaleFloors(in.Coefficients, "all_gather_fp16_2rank", 10)
		})
		return inflated.StepTime(decode).NoOverlap - base.StepTime(decode).NoOverlap
	}
	// dcp == pcp: the PCP axis alone; dcp == tp*pcp: the whole block, after a
	// tensor-parallel query gather.
	if alone, block := moved(4), moved(8); block <= alone {
		t.Errorf("inflating the 2-rank all-gather moved the decode step %v at dcp=8 and %v at "+
			"dcp=4; only dcp == tp*pcp gathers the query over the tensor-parallel group",
			block, alone)
	}
}

// The backend is a decode-only choice: switching it must leave a prefill step exactly where
// it was, and must change a decode step.
func TestTheDCPBackendChangesDecodeAndNothingElse(t *testing.T) {
	agrs := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, withBackend("ag_rs"))
	a2a := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, withBackend("a2a"))
	prefill := decodeBatch(1, 2048, 2048)
	if a, b := agrs.StepTime(prefill).NoOverlap, a2a.StepTime(prefill).NoOverlap; a != b {
		t.Errorf("the backend moved a prefill step: ag_rs %v, a2a %v", a, b)
	}
	if a, b := agrs.StepTime(decodeBatch(8, 1, 8192)).NoOverlap,
		a2a.StepTime(decodeBatch(8, 1, 8192)).NoOverlap; a == b {
		t.Errorf("the backend did not change a decode step (%v under both)", a)
	}
	if a, b := agrs.SequenceVariableBytes(8192), a2a.SequenceVariableBytes(8192); a != b {
		t.Errorf("the backend moved capacity: %d against %d", a, b)
	}
}

// An unstated backend is the stock default, ag_rs, and that is an ASSUMPTION the kernel
// discloses: a model's configuration hook may choose otherwise, and the kernel cannot see
// the model class. A stated one is the deployment's own fact and is not reported as an
// assumption.
func TestAnUnstatedDCPBackendIsTheStockDefaultAndIsDisclosed(t *testing.T) {
	unstated := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, nil)
	stated := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, withBackend("ag_rs"))
	decode := decodeBatch(8, 1, 8192)
	if a, b := unstated.StepTime(decode).NoOverlap, stated.StepTime(decode).NoOverlap; a != b {
		t.Errorf("an unstated backend priced %v against %v for ag_rs stated", a, b)
	}
	disclosed := func(k *Kernel) bool {
		for _, o := range k.Provenance() {
			if o.Name == "dcp_comm_backend" && o.Set == KernelAssumptionSet &&
				o.Method == string(vocab.MethodAssumed) && strings.HasPrefix(o.Scope, "ag_rs") {
				return true
			}
		}
		return false
	}
	if !disclosed(unstated) {
		t.Error("an unstated backend resolved to ag_rs without a kernel assumption in " +
			"Provenance")
	}
	if disclosed(stated) {
		t.Error("a stated backend was reported as an assumption")
	}
	_, _, assumed := unstated.Evidence()
	found := false
	for _, a := range assumed {
		found = found || a == "dcp_comm_backend"
	}
	if !found {
		t.Errorf("Evidence does not list the backend default among the assumptions: %v",
			assumed)
	}
	// And with dcp at one the knob does nothing, so nothing is assumed about it.
	if disclosed(mustDCPVariant(t, dcpMLAFixture, 8, 1, 1, nil)) {
		t.Error("a deployment without DCP reported an assumption about its DCP backend")
	}
}

// The backend names a release accepts are the release's rules pack's to say. A pack that
// does not list a2a refuses it; the v0.29 pack lists both; a name no release has is refused
// even where DCP is off, because the engine types the setting as a closed literal and
// validates it at startup whatever the width (vllm/config/parallel.py:40, :371).
func TestTheDCPBackendIsGatedByTheReleaseAndByWhatTheKernelPrices(t *testing.T) {
	if _, err := dcpVariant(t, dcpMLAFixture, 8, 1, 8, withBackend("a2a")); err != nil {
		t.Errorf("a2a, which the v0.29 pack lists, was refused: %v", err)
	}
	narrow := *v0_29.Pack()
	narrow.DCPCommBackends = map[string]bool{"ag_rs": true}
	_, err := dcpVariant(t, dcpMLAFixture, 8, 1, 8, func(in *Inputs) {
		withBackend("a2a")(in)
		in.Rules = &narrow
	})
	if err == nil {
		t.Error("a2a was priced under a rules pack that does not accept it")
	}
	for _, dcp := range []int{1, 8} {
		if _, err := dcpVariant(t, dcpMLAFixture, 8, 1, dcp,
			withBackend("all_gather_only")); err == nil {
			t.Errorf("dcp=%d: an unknown backend name was priced", dcp)
		}
	}
}

// A rules value with no list of the release's backend names cannot gate a stated backend,
// and the kernel says so rather than letting the check vanish: the stated name is still
// held to what the kernel prices, and Provenance records that the release was not consulted.
func TestAStatedBackendTheReleaseCouldNotCheckIsDisclosed(t *testing.T) {
	noList := *v0_29.Pack()
	noList.DCPCommBackends = nil
	k, err := dcpVariant(t, dcpMLAFixture, 8, 1, 8, func(in *Inputs) {
		withBackend("a2a")(in)
		in.Rules = &noList
	})
	if err != nil {
		t.Fatalf("a2a under a pack with no backend list was refused: %v", err)
	}
	if !hasAssumption(k, "dcp_comm_backend") {
		t.Error("a stated backend the release could not check left no trace in Provenance")
	}
	checked := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, withBackend("a2a"))
	if hasAssumption(checked, "dcp_comm_backend") {
		t.Error("a stated backend the release did check was reported as an assumption")
	}
}

// a2a is refused alongside PCP: "MRV2 PCP + DCP requires dcp_comm_backend='ag_rs'"
// (vllm/v1/worker/gpu/pcp_manager.py:188-194 at v0.31.0).
func TestA2AIsRefusedAlongsidePCP(t *testing.T) {
	if _, err := dcpVariant(t, dcpSparseFixture, 2, 4, 4, withBackend("a2a")); err == nil {
		t.Error("a2a with pcp=4, dcp=4 was priced; the engine refuses it")
	}
	if _, err := dcpVariant(t, dcpSparseFixture, 2, 4, 4, withBackend("ag_rs")); err != nil {
		t.Errorf("ag_rs with pcp=4, dcp=4 was refused: %v", err)
	}
}

// A replicated query projection is refused where it would take effect, and accepted where
// the engine ignores it.
//
// It takes effect only on a latent layer with dcp > 1 and pcp <= 1
// (vllm/model_executor/models/deepseek_v2.py:1072-1076 at v0.31.0); there it costs a
// projection built over tp/dcp ranks instead of tp (linear.py:632-659), which this kernel
// does not price, so it refuses rather than crediting the skipped gather alone. Everywhere
// else the request changes nothing and must change nothing here.
func TestAReplicatedQueryIsRefusedOnlyWhereItWouldTakeEffect(t *testing.T) {
	yes, no := true, false
	qrep := func(v *bool) func(*Inputs) {
		return func(in *Inputs) { editInputsEngine(in, func(e *engineconfig.Engine) { e.DCPQReplicate = v }) }
	}
	if _, err := dcpVariant(t, dcpMLAFixture, 8, 1, 8, qrep(&yes)); err == nil {
		t.Error("dcp_q_replicate on a latent model with dcp=8 was priced; its replicated " +
			"projection is not")
	}
	plain := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, nil)
	decode := decodeBatch(8, 1, 8192)
	for _, c := range []struct {
		name         string
		fixture      string
		tp, pcp, dcp int
		v            *bool
	}{
		{"stated false", dcpMLAFixture, 8, 1, 8, &no},
		{"dcp off", dcpMLAFixture, 8, 1, 1, &yes},
		{"pcp on", dcpSparseFixture, 2, 4, 4, &yes},
		{"full attention", "minimax-m25-h200-tp8.yaml", 8, 1, 8, &yes},
	} {
		k, err := dcpVariant(t, c.fixture, c.tp, c.pcp, c.dcp, qrep(c.v))
		if err != nil {
			t.Errorf("%s: refused, but the engine ignores the request here: %v", c.name, err)
			continue
		}
		if c.name == "stated false" && k.StepTime(decode).NoOverlap != plain.StepTime(decode).NoOverlap {
			t.Error("stating dcp_q_replicate false priced differently from leaving it unset")
		}
	}
}

// THE STRIPE. cp_kv_cache_interleave_size sets how many consecutive tokens a rank holds,
// and the slowest rank binds, so a coarse stripe makes a short context's read land on one
// rank (vllm/v1/attention/backends/utils.py:1143-1156 at v0.31.0).
//
// At a 100-token context over 8 ranks the slowest rank holds 13 tokens at a stripe of 1 and
// 64 at a stripe of 64, so a decode read must cost more; at a context that is a multiple of
// stripe x dcp every rank holds the same share under both, so the two must agree exactly.
func TestTheInterleaveSetsTheSlowestRanksShareOfTheRead(t *testing.T) {
	interleave := func(n int) func(*Inputs) {
		return func(in *Inputs) {
			editEngine(t, in, func(e *engineconfig.Engine) { e.CPKVCacheInterleaveSize = n })
			editEngine(t, in, func(e *engineconfig.Engine) { e.BlockSize = 64 })
		}
	}
	fine := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, interleave(1))
	coarse := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, interleave(64))
	hbm := func(k *Kernel, ctx int) float64 {
		return k.StepTime(decodeBatch(16, 1, ctx)).PerResource[kernel.ResourceHBM].Seconds()
	}
	if a, b := hbm(coarse, 100), hbm(fine, 100); a <= b {
		t.Errorf("a 100-token context read %.6f ms of HBM at a stripe of 64 against %.6f at "+
			"1; the slowest rank holds 64 tokens against 13", a*1e3, b*1e3)
	}
	if a, b := hbm(coarse, 4096), hbm(fine, 4096); a != b {
		t.Errorf("a 4,096-token context read %.6f ms at a stripe of 64 against %.6f at 1; "+
			"it is a multiple of stripe x dcp, so every rank holds 512 under both", a*1e3, b*1e3)
	}
	// Capacity does not see the stripe: the engine allocates block_size x dcp tokens per
	// logical block whatever the interleave (kv_cache_interface.py:545-550), so a rank
	// holds the same pages either way.
	if a, b := coarse.SequenceVariableBytes(100), fine.SequenceVariableBytes(100); a != b {
		t.Errorf("the stripe moved capacity: %d against %d", a, b)
	}
}

// UNDER NIXL AN UNSTATED STRIPE IS THE BLOCK SIZE. v0.31.0 pins an unstated
// cp_kv_cache_interleave_size to the local block size when NixlConnector is configured, and
// honours a stated one; any other connector leaves the default of 1. vLLM looks inside a
// MultiConnector for a NIXL child (vllm/config/kv_transfer.py:156-163), but a deployment does
// not name its children, so the answer is unknown and the kernel refuses rather than guess
// (vllm/config/vllm.py:3333-3378).
//
// Priced on the decode pool of a disaggregated deployment, which is the case the pin exists
// for: a decode pool sharding its cache while NIXL moves blocks in from prefill.
func TestUnderNIXLAnUnstatedInterleaveIsTheBlockSize(t *testing.T) {
	r := writeDisaggregated(t)
	build := func(connector string, interleave int) (*Kernel, error) {
		t.Helper()
		in, err := OpenInputs("disagg.yaml", r, 1)
		if err != nil {
			t.Fatal(err)
		}
		in.Deployment.PDTransfer.Connector = connector
		pool := &in.Deployment.Pools[1]
		pool.Parallel.DCP = 4
		editPoolEngine(t, pool, func(e *engineconfig.Engine) { e.BlockSize = 64 })
		editPoolEngine(t, pool, func(e *engineconfig.Engine) { e.CPKVCacheInterleaveSize = interleave })
		return newKernel(in)
	}
	must := func(connector string, interleave int) *Kernel {
		t.Helper()
		k, err := build(connector, interleave)
		if err != nil {
			t.Fatalf("%s, interleave %d: %v", connector, interleave, err)
		}
		return k
	}
	decode := decodeBatch(16, 1, 100)
	price := func(k *Kernel) float64 { return k.StepTime(decode).NoOverlap.Seconds() }

	pinned := must("NixlConnector", 0)
	if a, b := price(pinned), price(must("NixlConnector", 64)); a != b {
		t.Errorf("an unstated stripe under NIXL priced %.6f ms against %.6f with the block "+
			"size stated; the engine pins it to the block size", a*1e3, b*1e3)
	}
	if a, b := price(must("NixlConnector", 1)), price(must("MooncakeConnector", 0)); a != b {
		t.Errorf("a stated stripe of 1 under NIXL priced %.6f ms against %.6f for the "+
			"default under another connector; a stated size is honoured", a*1e3, b*1e3)
	}
	if price(pinned) == price(must("MooncakeConnector", 0)) {
		t.Error("the NIXL pin did not change the price of a short-context decode")
	}
	recorded := false
	for _, o := range pinned.Resolved().Overrides {
		recorded = recorded || (o.Field == "cp_kv_cache_interleave_size" && o.Resolved == "64")
	}
	if !recorded {
		t.Errorf("the pin is not recorded in Resolved: %+v", pinned.Resolved().Overrides)
	}
	if _, err := build("MultiConnector", 0); err == nil {
		t.Error("an unstated stripe under a MultiConnector was priced; whether it is " +
			"pinned depends on children a deployment cannot name")
	}
	if _, err := build("MultiConnector", 16); err != nil {
		t.Errorf("a stated stripe under a MultiConnector was refused: %v", err)
	}
}

// A STATED STRIPE MUST FIT THE BLOCK. Under DCP v0.31.0 asserts the interleave is no larger
// than the block and divides it, unless NIXL P/D is configured, where each worker pins it
// (VllmConfig.validate_block_size, vllm/config/vllm.py:3388-3402). On the disaggregated decode
// pool at block 64: 32 starts, 48 and 128 do not, 128 does under NIXL, and under a
// MultiConnector -- whose NIXL child a deployment cannot name -- a misfit is refused rather
// than guessed. Against a block the kernel only assumed, a misfit is refused too: the
// platform usually keeps the default 16 and the kernel cannot tell when it would not, so the
// deployment must state the block. A DSA model's block, 64, is the engine's own and checked
// as such: 64 fits it, 128 does not.
func TestAStatedInterleaveMustFitTheBlock(t *testing.T) {
	r := writeDisaggregated(t)
	starts := func(connector string, block, interleave int) bool {
		t.Helper()
		in, err := OpenInputs("disagg.yaml", r, 1)
		if err != nil {
			t.Fatal(err)
		}
		in.Deployment.PDTransfer.Connector = connector
		pool := &in.Deployment.Pools[1]
		pool.Parallel.DCP = 4
		editPoolEngine(t, pool, func(e *engineconfig.Engine) { e.BlockSize = block })
		editPoolEngine(t, pool, func(e *engineconfig.Engine) { e.CPKVCacheInterleaveSize = interleave })
		_, err = newKernel(in)
		return err == nil
	}
	for _, c := range []struct {
		connector         string
		block, interleave int
		want              bool
	}{
		{"MooncakeConnector", 64, 32, true},
		{"MooncakeConnector", 64, 64, true},
		{"MooncakeConnector", 64, 48, false},
		{"MooncakeConnector", 64, 128, false},
		{"NixlConnector", 64, 128, true},
		{"MultiConnector", 64, 128, false},
		{"MooncakeConnector", 0, 128, false},
		{"MooncakeConnector", 0, 16, true},
	} {
		if got := starts(c.connector, c.block, c.interleave); got != c.want {
			t.Errorf("%s, block %d, interleave %d: starts %v, want %v",
				c.connector, c.block, c.interleave, got, c.want)
		}
	}
	for _, c := range []struct {
		interleave int
		want       bool
	}{{64, true}, {128, false}} {
		_, err := dcpVariant(t, dcpSparseFixture, 8, 1, 2, func(in *Inputs) {
			e := mustDecode(t, in.Deployment.Pools[0])
			off := false
			e.BlockSize, e.CPKVCacheInterleaveSize = 0, c.interleave
			e.DCPCommBackend, e.DCPQReplicate = "ag_rs", &off
			setEngine(t, &in.Deployment.Pools[0], e)
		})
		if got := err == nil; got != c.want {
			t.Errorf("glm5 (DSA, block 64), interleave %d: starts %v, want %v (%v)",
				c.interleave, got, c.want, err)
		}
	}
}

// A FULL-ATTENTION DCP LAYER COMBINES EVERY QUERY ROW WHEN ANY REQUEST READS CONTEXT.
// FlashAttention runs one context pass over the whole step's queries -- gathering them all
// and combining every row -- and skips it when no request has cached context
// (vllm/v1/attention/backends/flash_attn.py:1591-1686 at v0.31.0). So beside decode rows a
// prefill's tokens join the combine; on a step of fresh prefills nothing is combined at all.
func TestAFullAttentionDCPLayerCombinesEveryRowWhenContextIsRead(t *testing.T) {
	const fixture = "minimax-m25-h200-tp8.yaml" // gqa
	added := func(b kernel.Batch) float64 {
		t.Helper()
		on := mustDCPVariant(t, fixture, 8, 1, 2, nil).StepTime(b).PerResource
		off := mustDCPVariant(t, fixture, 8, 1, 1, nil).StepTime(b).PerResource
		return (on[kernel.ResourceNVLink] + on[kernel.ResourceNIC] -
			off[kernel.ResourceNVLink] - off[kernel.ResourceNIC]).Seconds()
	}
	decodes := kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: 1, Computed: 4095, PromptLen: 4096},
		{Scheduled: 1, Computed: 4095, PromptLen: 4096},
	}}
	mixed := kernel.Batch{DecodeThreshold: 8, Reqs: append(append([]kernel.ReqShape{},
		decodes.Reqs...), kernel.ReqShape{Scheduled: 512, Computed: 0, PromptLen: 512})}
	fresh := kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: 512, Computed: 0, PromptLen: 512},
	}}
	if a, b := added(mixed), added(decodes); a <= b {
		t.Errorf("a 512-token prefill beside two decodes added %.6f ms of DCP collective "+
			"against %.6f ms for the decodes alone; its rows join the context pass",
			a*1e3, b*1e3)
	}
	if a := added(fresh); math.Abs(a) > 2e-9 {
		t.Errorf("a step of fresh prefills added %.6f ms of DCP collective; with no context "+
			"read there is no context pass", a*1e3)
	}
}

// A LATENT PREFILL READS ITS SHARDED CONTEXT BACK IN CHUNKS. The MHA prefill path attends
// the whole context, so the group all-gathers each rank's shard of it, one all-gather per
// workspace-sized chunk (vllm/model_executor/layers/attention/mla_attention.py:3090-3104,
// workspace :2297-2320). deepseek-v3's fixture states max_model_len 32768, so a chunk is
// min(max(8 x 32768, ...), 65536) = 65,536 context tokens.
//
// Chunks pack consecutive context-reading prefills (plan_mla_context_chunks, :1922-1990),
// so the count follows each run's summed context, and a fresh prefill ends a run. Runs are
// consecutive in the ENGINE's order, which Model Runner V2 sorts by scheduled tokens
// (sort_batch_req_ids, vllm/v1/worker/gpu/model_runner.py:2398-2410): a fresh prefill listed
// between two context-reading ones splits them only if it also sorts between them.
//
// Counted by inflating the 2-rank all-gather floor at tp=8, dcp=2, where the only 2-rank
// all-gather a prefill-only step can run is this one: launches per layer are 0 with no
// prefix, 1 for a 1,000-token prefix and 2 for a 70,000-token one; 1 for two adjacent
// 1,000-token prefixes, which share a chunk, and 2 when a fresh prefill separates them; and
// 2 for two adjacent 40,000-token prefixes, whose 80,000 tokens overflow one chunk. Listed
// as 512, fresh 256, 1,024 scheduled tokens, the fresh prefill sorts first and the two
// context reads share a chunk; listed as 256, fresh 512, 1,024, it sorts between them.
func TestALatentPrefillGathersItsContextOnceAChunk(t *testing.T) {
	type req struct{ scheduled, prefix int }
	// moved is how much a step's NVLink time rises when the 2-rank all-gather floor is
	// inflated tenfold, at the fixture's max_model_len or a stated one. Only the floor moves,
	// so the rise is proportional to the gathers launched, and a count is the rise over a
	// one-gather reference's -- no private state read.
	moved := func(maxModelLen int, reqs ...req) float64 {
		t.Helper()
		b := kernel.Batch{DecodeThreshold: 8}
		for _, r := range reqs {
			b.Reqs = append(b.Reqs, kernel.ReqShape{
				Scheduled: r.scheduled, Computed: r.prefix, PromptLen: r.prefix + r.scheduled})
		}
		setLen := func(in *Inputs) {
			if maxModelLen > 0 {
				editEngine(t, in, func(e *engineconfig.Engine) { e.MaxModelLen = engineconfig.MaxModelLen(maxModelLen) })
			}
		}
		base := mustDCPVariant(t, dcpMLAFixture, 8, 1, 2, setLen)
		inflated := mustDCPVariant(t, dcpMLAFixture, 8, 1, 2, func(in *Inputs) {
			setLen(in)
			in.Coefficients = scaleFloors(in.Coefficients, "all_gather_fp16_2rank", 10)
		})
		return (inflated.StepTime(b).PerResource[kernel.ResourceNVLink] -
			base.StepTime(b).PerResource[kernel.ResourceNVLink]).Seconds()
	}
	launches := func(maxModelLen int, reqs ...req) float64 {
		t.Helper()
		return moved(maxModelLen, reqs...) / moved(maxModelLen, req{256, 1000})
	}
	for _, c := range []struct {
		name        string
		maxModelLen int
		reqs        []req
		want        float64
	}{
		{"a fresh prefill", 0, []req{{256, 0}}, 0},
		{"two adjacent context reads share a chunk", 0,
			[]req{{256, 1000}, {256, 1000}}, 1},
		{"a fresh prefill between them splits the run", 0,
			[]req{{256, 1000}, {256, 0}, {256, 1000}}, 2},
		{"a fresh prefill first does not", 0,
			[]req{{256, 0}, {256, 1000}, {256, 1000}}, 1},
		{"three 30k contexts overflow one 65,536-token chunk", 0,
			[]req{{256, 30000}, {256, 30000}, {256, 30000}}, 2},
		// 130,000 tokens fit two chunks only if the overflowing request is split across
		// them, as plan_mla_context_chunks does (mla_attention.py:1966-1975); starting a new
		// chunk for each whole request that overflows would take three.
		{"the overflowing request is split", 0, []req{{256, 30000}, {256, 30000},
			{256, 30000}, {256, 30000}, {256, 10000}}, 2},
		// The sort puts the 256-token fresh prefill first, so the two context reads share a
		// chunk; listed as 256, fresh 512, 1,024 it sorts between them.
		{"the engine sorts before packing", 0,
			[]req{{512, 1000}, {256, 0}, {1024, 1000}}, 1},
		{"a fresh prefill that sorts between them splits", 0,
			[]req{{256, 1000}, {512, 0}, {1024, 1000}}, 2},
		// The chunk follows max_model_len: at 2,048 the workspace is
		// min(max(8 x 2048, 4 x 256 x 16), 65536) = 16,384 rows, so 18,000 tokens of context
		// take two gathers where the default chunk takes one.
		{"a short max_model_len shrinks the chunk", 2048, func() []req {
			r := make([]req, 12)
			for i := range r {
				r[i] = req{256, 1500}
			}
			return r
		}(), 2},
	} {
		if got := launches(c.maxModelLen, c.reqs...); math.Abs(got-c.want) > 1e-6 {
			t.Errorf("%s: %v (scheduled, prefix) ran %.6f context gathers per layer, "+
				"want %v", c.name, c.reqs, got, c.want)
		}
	}
}

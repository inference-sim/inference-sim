// Moved from blis-latency-kernel/ceiling_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// THE CAPTURE CEILING FOLLOWS THE ENGINE'S DEFAULT, on every input that sets it. vLLM v0.31.0
// captures token counts up to min(max_num_seqs x decode_query_len x 2, 512), or 1024 on
// data-center Blackwell (VllmConfig._set_cudagraph_sizes, vllm/config/vllm.py:2442-2460), and
// a step above that runs with no graph, a uniform decode included
// (vllm/v1/worker/gpu/cudagraph_utils.py:499-529). The fixture's own max_num_seqs, 256, puts
// it at 512 on h200, which fixes neither the Blackwell figure nor the seqs term; these cases
// move each:
//
//   - h200 marked Blackwell, seqs 1024: ceiling 1024.
//   - h200, seqs 64: ceiling 128 (the seqs term binds).
//   - h200, seqs 1024: ceiling 512, and a uniform decode of 513 requests -- admissible at
//     that cap -- runs with no graph under FULL_AND_PIECEWISE.
//
// At the ceiling a step prices as the mode's own graph; one token above, as NONE.
func TestTheCaptureCeilingFollowsTheEngineDefault(t *testing.T) {
	host := func(mode string, seqs int, blackwell bool, b kernel.Batch) time.Duration {
		t.Helper()
		in := fixtureInputs(t, memoryFixture)
		e := mustDecode(t, in.Deployment.Pools[0])
		e.CUDAGraphMode, e.MaxNumSeqs = mode, seqs
		if blackwell {
			chip := *in.Chip
			chip.NVFP4Peak = 1
			in.Chip = &chip
		}
		setEngine(t, &in.Deployment.Pools[0], e)
		k, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		return k.StepTime(b).PerResource[kernel.ResourceHost]
	}
	prefill := func(tokens int) kernel.Batch { return decodeBatch(1, tokens, tokens) }
	for _, c := range []struct {
		name      string
		seqs      int
		blackwell bool
		ceiling   int
	}{
		{"blackwell, seqs 1024", 1024, true, 1024},
		{"h200, seqs 64", 64, false, 128},
		{"h200, seqs 1024", 1024, false, 512},
	} {
		at, above := prefill(c.ceiling), prefill(c.ceiling+1)
		if a, b := host("PIECEWISE", c.seqs, c.blackwell, at),
			host("NONE", c.seqs, c.blackwell, at); a == b {
			t.Errorf("%s: a %d-token prefill, at the ceiling, priced like NONE", c.name,
				c.ceiling)
		}
		if a, b := host("PIECEWISE", c.seqs, c.blackwell, above),
			host("NONE", c.seqs, c.blackwell, above); a != b {
			t.Errorf("%s: a %d-token prefill, past the ceiling, priced %v against NONE's %v",
				c.name, c.ceiling+1, a, b)
		}
	}
	at, above := decodeBatch(512, 1, 2048), decodeBatch(513, 1, 2048)
	if a, b := host("FULL_AND_PIECEWISE", 1024, false, at),
		host("NONE", 1024, false, at); a == b {
		t.Error("a 512-request uniform decode, at the ceiling, priced like NONE")
	}
	if a, b := host("FULL_AND_PIECEWISE", 1024, false, above),
		host("NONE", 1024, false, above); a != b {
		t.Errorf("a 513-request uniform decode, past the ceiling, priced %v against NONE's %v",
			a, b)
	}
}

// A DCP GROUP SITS WHERE THE ENGINE'S RANK LAYOUT PUTS IT, seen in the step rather than the
// layout. vLLM numbers ranks DP x PP x PCP x TP with TP innermost
// (vllm/distributed/parallel_state.py:2054-2060), so a DCP group that spans the PCP axis has
// members tp ranks apart. On glm5 at tp=8, pcp=2, dcp=2 that is two nodes, so a decode step
// must charge the NIC: a decode step's only cross-node collective is that group's combine.
// Without the stride it would be priced as two adjacent ranks on one node's NVLink.
//
// glm5 states the ag_rs combine here and in the next test: GLM-5's model hook would
// otherwise choose a2a (vllm/model_executor/models/config.py:43-50), which PCP with DCP
// refuses (vllm/v1/worker/gpu/pcp_manager.py:187-194).
func TestADCPGroupSitsWhereTheRankLayoutPutsIt(t *testing.T) {
	k := mustDCPVariant(t, dcpSparseFixture, 8, 2, 2, agRS)
	if got := k.StepTime(decodeBatch(8, 1, 4096)).PerResource[kernel.ResourceNIC]; got <= 0 {
		t.Errorf("tp=8, pcp=2, dcp=2: a decode charged %v of NIC; the DCP group spans two "+
			"nodes", got)
	}
}

// UNDER PCP WITH DCP A SHORT PREFILL IS REPLICATED, and the step pays for it. With DCP on,
// MRV2 gives every PCP rank the whole query of a prefill too short to fill all 2 x pcp
// chunks: (2pcp - 1) x ceil(q / 2pcp) >= q (replicated_requests,
// vllm/v1/worker/gpu/pcp_manager.py:222-232 at v0.31.0). At pcp=4 a 17-token prefill drops a
// chunk (7 x 3 >= 17) and a 16-token one does not (7 x 2 < 16). So on glm5 at tp=2, pcp=4,
// the 17-token prefill must cost more SM at dcp=4 than at dcp=1, where nothing is
// replicated, and the 16-token one the same at both.
func TestUnderPCPWithDCPAShortPrefillIsReplicated(t *testing.T) {
	sm := func(dcp, tokens int) time.Duration {
		t.Helper()
		k := mustDCPVariant(t, dcpSparseFixture, 2, 4, dcp, agRS)
		return k.StepTime(decodeBatch(1, tokens, tokens)).PerResource[kernel.ResourceSM]
	}
	if a, b := sm(4, 17), sm(1, 17); a <= b {
		t.Errorf("a 17-token prefill at pcp=4 cost %v of SM with dcp=4 and %v without; "+
			"with DCP it is replicated to every rank", a, b)
	}
	if a, b := sm(4, 16), sm(1, 16); a != b {
		t.Errorf("a 16-token prefill at pcp=4 cost %v of SM with dcp=4 and %v without; it "+
			"fills every chunk, so DCP must not change its share", a, b)
	}
}

// agRS states the ag_rs DCP combine without a replicated query, which a GLM-5 deployment
// under DCP must state to start (see TestADCPGroupSitsWhereTheRankLayoutPutsIt).
func agRS(in *Inputs) {
	if in.Deployment.Pools[0].Parallel.DCP > 1 {
		off := false
		editInputsEngine(in, func(e *engineconfig.Engine) { e.DCPCommBackend, e.DCPQReplicate = "ag_rs", &off })
	}
}

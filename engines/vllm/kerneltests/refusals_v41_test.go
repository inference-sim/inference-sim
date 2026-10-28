// Moved from blis-latency-kernel/refusals_v41_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// A term the kernel cannot price from its inputs is refused or recorded, never charged at
// nothing in silence.

// dropEntries returns sets without the entries whose names start with any prefix.
func dropEntries(sets []*coefficient.Set, prefixes ...string) []*coefficient.Set {
	out := make([]*coefficient.Set, len(sets))
	for i, s := range sets {
		c := *s
		c.Coefficients = nil
		for _, e := range s.Coefficients {
			keep := true
			for _, p := range prefixes {
				keep = keep && !strings.HasPrefix(e.Name, p)
			}
			if keep {
				c.Coefficients = append(c.Coefficients, e)
			}
		}
		out[i] = &c
	}
	return out
}

// recorded reports whether a kernel's provenance names an entry.
func recorded(k *Kernel, name string) bool {
	for _, o := range k.Provenance() {
		if o.Name == name {
			return true
		}
	}
	return false
}

// With no measured attention form, a scorer's scan is still a read the step pays for. The
// scan grows with context where the index store, which writes per scheduled token, does
// not: so a larger index record widens the gap between a long-context and a short-context
// decode's HBM time.
func TestAScorerWithNoMeasuredFormIsStillCharged(t *testing.T) {
	gap := func(scale int) float64 {
		in := fixtureInputs(t, v41)
		in.Coefficients = dropEntries(in.Coefficients, "attention_decode_")
		// Owners and readers alike, as a reader's record must match its owner's.
		editNodes(in.Model, role("block_index_scores"), func(n *model.Node) {
			for c := range n.Caches {
				if n.Caches[c].Kind == model.CacheIndex {
					n.Caches[c].BytesPerEntry *= scale
				}
			}
		})
		k := build(t, in)
		hbm := func(ctx int) float64 {
			return k.StepTime(decodeBatch(64, 1, ctx)).PerResource[kernel.ResourceHBM].Seconds()
		}
		return hbm(65536) - hbm(8192)
	}
	if small, large := gap(1), gap(4); large <= small {
		t.Errorf("a 4x index record widens the long-context HBM gap to %.3g s, no more "+
			"than %.3g: the scan is not charged", large, small)
	}
}

// A host-resident table with no rate to cross the host link at is refused.
func TestAHostTableWithNoLinkRateIsRefused(t *testing.T) {
	in := fixtureInputs(t, v41)
	in.Coefficients = dropEntries(in.Coefficients, "host_gather_rate_uva", "host_link_bandwidth")
	if _, err := newKernel(in); err == nil {
		t.Error("an Engram table in host memory was priced with no host link rate")
	}
}

// A scorer the registry has no form for on this part is priced with its kind's pair, and
// says so.
func TestAScorerWithoutItsOwnRateIsRecorded(t *testing.T) {
	in := fixtureInputs(t, v41)
	in.Chip, in.Scenario.Cluster.Hardware = chipInputs(t, "b200"), "b200"
	if k := build(t, in); !recorded(k, "attention_decode_rate_block_index_scores") {
		t.Error("the indexer on B200 has no rate of its own and Provenance does not say so")
	}
}

// The DeepSeek-V4.1 layouts vLLM v0.31.0 refuses at startup are refused, not priced.
func TestV41LayoutsTheEngineRefuses(t *testing.T) {
	for name, edit := range map[string]func(*Inputs){
		"a bf16 cache": func(in *Inputs) { editInputsEngine(in, func(e *engineconfig.Engine) { e.CacheDType = "bfloat16" }) },
		"mega-MoE without expert parallelism": func(in *Inputs) {
			editEngine(t, in, func(e *engineconfig.Engine) { e.MoEBackend = "deep_gemm_mega_moe" })
		},
		"dual-batch overlap with Engram": func(in *Inputs) {
			editEngine(t, in, func(e *engineconfig.Engine) { e.DBO = &engineconfig.DBO{Enabled: true} })
		},
		"MTP drafting": func(in *Inputs) {
			editEngine(t, in, func(e *engineconfig.Engine) { e.Speculative.Method = "mtp" })
		},
	} {
		in := fixtureInputs(t, v41)
		edit(&in)
		if _, err := newKernel(in); err == nil {
			t.Errorf("%s: priced a layout vLLM v0.31.0 refuses at startup", name)
		}
	}
}

// A sparse layer gathers its window and top-k for every query token, so the context it reads
// grows once per verify token, where a dense layer reads its shared prefix once. Measured as
// how much more HBM time a longer context costs -- below the top-k, where a sparse read
// still grows -- which per-token terms that do not read context leave untouched.
func TestSparseReadsArePerQueryToken(t *testing.T) {
	k := fixture(t, "glm5-h200-tp8.yaml") // DSA: top-k 2048, read per query
	gap := func(q int) float64 {
		hbm := func(ctx int) float64 {
			return k.StepTime(decodeBatch(32, q, ctx)).PerResource[kernel.ResourceHBM].Seconds()
		}
		return hbm(2000) - hbm(1000)
	}
	if one, six := gap(1), gap(6); six < 3*one {
		t.Errorf("a 6-token verify's context costs %.3g s of HBM against a 1-token decode's "+
			"%.3g; six queries each gather their own selection", six, one)
	}
}

// nvfp4_ds_mla is the SM100 record: vLLM v0.31.0 refuses it on SM90.
func TestNVFP4CacheIsRefusedOnSM90(t *testing.T) {
	in := fixtureInputs(t, v41)
	editEngine(t, &in, func(e *engineconfig.Engine) { e.CacheDType = "nvfp4_ds_mla" })
	if _, err := newKernel(in); err == nil {
		t.Error("nvfp4_ds_mla on H200 was priced; FlashMLA decodes it only on SM100")
	}
	in = fixtureInputs(t, v41)
	in.Chip, in.Scenario.Cluster.Hardware = chipInputs(t, "b200"), "b200"
	editEngine(t, &in, func(e *engineconfig.Engine) { e.CacheDType = "nvfp4_ds_mla" })
	if _, err := newKernel(in); err != nil {
		t.Errorf("nvfp4_ds_mla on B200 was refused: %v", err)
	}
}

// DeepEP high-throughput with dp above one cannot be captured, so the engine runs it with
// no graphs whatever mode was stated: it prices as the same layout stated NONE, and the
// override is reported.
func TestUncapturableDispatchRunsEager(t *testing.T) {
	layout := func(mode string) *Kernel {
		in := fixtureInputs(t, v41)
		pl := &in.Deployment.Pools[0].Parallel
		pl.TP, pl.DP, pl.EnableExpertParallel = 4, 2, true
		editEngine(t, &in, func(e *engineconfig.Engine) { e.All2AllBackend = "deepep_high_throughput" })
		editEngine(t, &in, func(e *engineconfig.Engine) { e.CUDAGraphMode = mode })
		return build(t, in)
	}
	full, none := layout("FULL_AND_PIECEWISE"), layout("NONE")
	b := decodeBatch(16, 6, 8192)
	if f, n := full.StepTime(b).PerResource[kernel.ResourceHost],
		none.StepTime(b).PerResource[kernel.ResourceHost]; f != n {
		t.Errorf("a captured mode prices %v of host time and NONE %v; DeepEP "+
			"high-throughput at dp 2 runs eager either way", f, n)
	}
	overridden := false
	for _, o := range full.Resolved().Overrides {
		overridden = overridden || (o.Field == "cudagraph_mode" && o.Resolved == "NONE")
	}
	if !overridden {
		t.Error("the engine's switch to eager is not reported as an override")
	}
}

// Expert parallelism at dp 1 with no sequence-parallel MoE input runs no dispatch
// (use_all2all_kernels, vllm/model_executor/layers/fused_moe/config.py:1024-1027), so every
// all2all backend prices the same step, and a stated one is reported as not taking effect.
// At tp 4 dp 2 the dispatch runs and the backend still decides its volume.
func TestAnAll2AllBackendTakesEffectOnlyWhereTheEngineDispatches(t *testing.T) {
	layout := func(tp, dp int, backend string) *Kernel {
		in := fixtureInputs(t, v41)
		pl := &in.Deployment.Pools[0].Parallel
		pl.TP, pl.DP, pl.EnableExpertParallel = tp, dp, true
		editEngine(t, &in, func(e *engineconfig.Engine) { e.All2AllBackend = backend })
		if dp > 1 {
			// LL's buffer is sized by the token budget; at the fixture's 8,192 it cannot be
			// allocated (TestALowLatencyBufferThatCannotBeAllocatedIsRefused).
			editEngine(t, &in, func(e *engineconfig.Engine) { e.MaxNumBatchedTokens = 256 })
		}
		return build(t, in)
	}
	batches := []kernel.Batch{decodeBatch(1, 6, 100000), decodeBatch(128, 6, 100000),
		prefillOf(8192, 90000)}
	backends := []string{"allgather_reducescatter", "deepep_low_latency",
		"deepep_high_throughput"}
	base := layout(8, 1, "")
	for _, b := range backends {
		k := layout(8, 1, b)
		for i, batch := range batches {
			if got, want := k.StepTime(batch), base.StepTime(batch); !reflect.DeepEqual(got, want) {
				t.Errorf("tp8 dp1, %s, batch %d: %v against %v unstated; no dispatch runs",
					b, i, got.NoOverlap, want.NoOverlap)
			}
		}
		reported := false
		for _, o := range k.Resolved().Overrides {
			reported = reported || (o.Field == "all2all_backend" && o.Requested == b &&
				o.Resolved == "none")
		}
		if !reported {
			t.Errorf("tp8 dp1: the stated %s cannot take effect and no override says so", b)
		}
	}
	ll, ag := layout(4, 2, "deepep_low_latency"), layout(4, 2, "allgather_reducescatter")
	b := prefillOf(256, 90000)
	if ll.StepTime(b).NoOverlap == ag.StepTime(b).NoOverlap {
		t.Error("tp4 dp2: DeepEP and all-gather/reduce-scatter price the same dispatch")
	}
	for _, o := range ll.Resolved().Overrides {
		if o.Field == "all2all_backend" {
			t.Errorf("tp4 dp2: the backend takes effect, yet an override says %q", o.Reason)
		}
	}
}

// Layouts vLLM v0.31.0 refuses for the DeepSeek-V4 family because its own attention and MoE
// backends cannot run them.
func TestV4FamilyLayoutsItsBackendsCannotRun(t *testing.T) {
	for name, edit := range map[string]func(*Inputs){
		"prefill-context parallelism": func(in *Inputs) { in.Deployment.Pools[0].Parallel.PCP = 2 },
		"decode-context parallelism":  func(in *Inputs) { in.Deployment.Pools[0].Parallel.DCP = 2 },
		"mega-MoE on SM90": func(in *Inputs) {
			pl := &in.Deployment.Pools[0].Parallel
			pl.TP, pl.DP, pl.EnableExpertParallel = 4, 2, true
			editEngine(t, in, func(e *engineconfig.Engine) { e.MoEBackend = "deep_gemm_mega_moe" })
		},
		"a drafter the graph does not describe": func(in *Inputs) {
			editEngine(t, in, func(e *engineconfig.Engine) { e.Speculative.Method = "eagle3" })
		},
	} {
		in := fixtureInputs(t, v41)
		edit(&in)
		if _, err := newKernel(in); err == nil {
			t.Errorf("%s: priced a DeepSeek-V4.1 layout vLLM v0.31.0 cannot run", name)
		}
	}
}

// An FP8 DeepSeek-V4-family checkpoint is served by its own quantization method whatever
// --quantization names, so naming one changes no price and is reported as overridden.
func TestV4CheckpointIgnoresTheQuantizationFlag(t *testing.T) {
	base := build(t, fixtureInputs(t, v41))
	for _, q := range []string{"deepseek_v4_fp8", "fp8", "mxfp4", "modelopt_fp4"} {
		in := fixtureInputs(t, v41)
		editEngine(t, &in, func(e *engineconfig.Engine) { e.Quantization = q })
		k := build(t, in)
		if k.FixedBytes() != base.FixedBytes() {
			t.Errorf("--quantization %s moved the resident bytes", q)
		}
		for name, b := range batchShapes() {
			if k.StepTime(b).NoOverlap != base.StepTime(b).NoOverlap {
				t.Errorf("--quantization %s moved the %s step", q, name)
			}
		}
	}
}

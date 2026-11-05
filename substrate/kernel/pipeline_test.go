package latencykernel

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/vocab"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// The stage split is vLLM's own: these are get_pp_indices's outputs at vLLM v0.32.0
// (vllm/distributed/utils.py:126-169@5df63a72), printed from the pinned source with
// VLLM_PP_LAYER_PARTITION unset, for layer counts with and without a remainder.
func TestPipelineStagesSplitLayersAsVLLMDoes(t *testing.T) {
	for _, c := range []struct {
		layers, pp int
		want       [][2]int
	}{
		{5, 2, [][2]int{{0, 3}, {3, 5}}},
		{5, 3, [][2]int{{0, 2}, {2, 4}, {4, 5}}},
		{5, 4, [][2]int{{0, 1}, {1, 2}, {2, 4}, {4, 5}}},
		{7, 2, [][2]int{{0, 4}, {4, 7}}},
		{7, 3, [][2]int{{0, 2}, {2, 5}, {5, 7}}},
		{7, 4, [][2]int{{0, 2}, {2, 4}, {4, 6}, {6, 7}}},
		{30, 2, [][2]int{{0, 15}, {15, 30}}},
		{30, 3, [][2]int{{0, 10}, {10, 20}, {20, 30}}},
		{30, 4, [][2]int{{0, 7}, {7, 15}, {15, 23}, {23, 30}}},
		{30, 8, [][2]int{{0, 3}, {3, 7}, {7, 11}, {11, 15}, {15, 19}, {19, 23}, {23, 27}, {27, 30}}},
		{61, 2, [][2]int{{0, 31}, {31, 61}}},
		{61, 3, [][2]int{{0, 20}, {20, 41}, {41, 61}}},
		{61, 4, [][2]int{{0, 15}, {15, 30}, {30, 46}, {46, 61}}},
		{61, 8, [][2]int{{0, 7}, {7, 14}, {14, 22}, {22, 30}, {30, 38}, {38, 46}, {46, 54}, {54, 61}}},
		{62, 2, [][2]int{{0, 31}, {31, 62}}},
		{62, 3, [][2]int{{0, 21}, {21, 42}, {42, 62}}},
		{62, 4, [][2]int{{0, 15}, {15, 31}, {31, 47}, {47, 62}}},
		{62, 8, [][2]int{{0, 7}, {7, 15}, {15, 23}, {23, 31}, {31, 39}, {39, 47}, {47, 55}, {55, 62}}},
		{78, 2, [][2]int{{0, 39}, {39, 78}}},
		{78, 3, [][2]int{{0, 26}, {26, 52}, {52, 78}}},
		{78, 4, [][2]int{{0, 19}, {19, 39}, {39, 59}, {59, 78}}},
		{78, 8, [][2]int{{0, 9}, {9, 19}, {19, 29}, {29, 39}, {39, 49}, {49, 59}, {59, 69}, {69, 78}}},
		{94, 2, [][2]int{{0, 47}, {47, 94}}},
		{94, 3, [][2]int{{0, 31}, {31, 63}, {63, 94}}},
		{94, 4, [][2]int{{0, 23}, {23, 47}, {47, 71}, {71, 94}}},
		{94, 8, [][2]int{{0, 11}, {11, 23}, {23, 35}, {35, 47}, {47, 59}, {59, 71}, {71, 83}, {83, 94}}},
	} {
		if got := ppIndices(c.layers, c.pp); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%d layers over pp %d: %v, vLLM gives %v", c.layers, c.pp, got, c.want)
		}
	}
}

func fabricSet(entries ...coefficient.Entry) []*coefficient.Set {
	return []*coefficient.Set{{Kind: "CoefficientSet", Name: "t", Coefficients: entries}}
}

func efficiency(v float64, scope coefficient.Scope) coefficient.Entry {
	return coefficient.Entry{Name: fabricEfficiencyName, Value: v, Units: vocab.UnitDimensionless,
		Method: vocab.MethodMeasured, Scope: scope}
}

// The fabric correction is read only from an entry scoped to the deployment's fabric, and
// refused outside (0, 1].
func TestTheFabricEfficiencyIsTheFabricsOwn(t *testing.T) {
	roce := &hardware.Fabric{Name: "roce-200g", InterNodeBwGBps: 25}
	load := func(sets []*coefficient.Set, f *hardware.Fabric) (float64, bool, error) {
		name := ""
		if f != nil {
			name = f.Name
		}
		c, err := resolve.Load(append(sets, fabricSet(coefficient.Entry{Name: "x", Value: 1,
			Units: vocab.UnitDimensionless, Method: vocab.MethodMeasured,
			Scope: coefficient.Scope{Hardware: []string{"h100"}}})...),
			resolve.Scope{Hardware: "h100", Fabric: name})
		if err != nil {
			t.Fatal(err)
		}
		return resolveFabricEfficiency(c, f)
	}
	scoped := fabricSet(efficiency(0.8, coefficient.Scope{Fabric: []string{"roce-200g"}}))
	if e, ok, err := load(scoped, roce); err != nil || !ok || e != 0.8 {
		t.Errorf("over roce-200g: %v %v %v, want 0.8 stated", e, ok, err)
	}
	ib := &hardware.Fabric{Name: "ib-400g", InterNodeBwGBps: 50}
	if e, ok, _ := load(scoped, ib); ok || e != 1 {
		t.Errorf("over ib-400g a roce-scoped entry applied: %v %v", e, ok)
	}
	partWide := fabricSet(efficiency(0.5, coefficient.Scope{Hardware: []string{"h100"}}))
	if e, ok, _ := load(partWide, roce); ok || e != 1 {
		t.Errorf("an entry scoped by hardware alone corrected the fabric: %v %v", e, ok)
	}
	if e, ok, _ := load(nil, nil); !ok || e != 1 {
		t.Errorf("no fabric: %v %v, want 1 stated", e, ok)
	}
	for _, bad := range []float64{0, -0.1, 1.2} {
		if _, _, err := load(fabricSet(efficiency(bad, coefficient.Scope{Fabric: []string{"roce-200g"}})), roce); err == nil {
			t.Errorf("efficiency %v was accepted", bad)
		}
	}
}

// A fabric that achieves less of its line rate prices a crossing byte higher: the effective
// rate falls and the intra-to-inter ratio a collective is scaled by rises.
func TestALowerFabricEfficiencyRaisesTheCrossNodeRatio(t *testing.T) {
	nominal := resolve.Fabric{IntraNodeBwGBps: 450, InterNodeBwGBps: 50}
	derated := nominal
	derated.InterNodeEfficiency = 0.8
	if got := derated.EffectiveInterNodeBwGBps(); got != 40 {
		t.Errorf("effective rate %v, want 40", got)
	}
	if !(derated.Ratio() > nominal.Ratio()) {
		t.Errorf("ratio %v at 0.8 is not above %v at nominal", derated.Ratio(), nominal.Ratio())
	}
	if nominal.EffectiveInterNodeBwGBps() != 50 {
		t.Error("an unset efficiency changed the nominal rate")
	}
}

// The grouped NVFP4 MoE on a native-FP4 part reads its own ramp, falling back to the dense
// NVFP4 one; on Hopper NVFP4 experts are Marlin W4A16, reading their own grouped ramp and
// falling back to the MXFP4 Marlin one. A dense NVFP4 GEMM keeps the dense ramp.
func TestGroupedNVFP4ReadsItsOwnRamp(t *testing.T) {
	blackwell := hardware.Chip{Name: "b200", BF16Peak: 2250, FP8Peak: 4500, NVFP4Peak: 9000}
	hopper := hardware.Chip{Name: "h100", BF16Peak: 989.5, FP8Peak: 1979}
	nv := price.Format{DType: model.DTypeNVFP4}
	for _, c := range []struct {
		chip     hardware.Chip
		grouped  bool
		suffix   string
		fallback string
	}{
		{blackwell, true, "nvfp4_grouped", "nvfp4"},
		{blackwell, false, "nvfp4", ""},
		{hopper, true, "w4a16_nvfp4_grouped", "w4a16_mxfp4_grouped"},
		{hopper, false, "bf16", ""},
	} {
		got := formatFit(nv, c.grouped, c.chip, true)
		if got.suffix != c.suffix || got.fallback != c.fallback {
			t.Errorf("%s grouped=%v: %s (fallback %q), want %s (fallback %q)", c.chip.Name,
				c.grouped, got.suffix, got.fallback, c.suffix, c.fallback)
		}
	}
	if got := formatFit(price.Format{DType: model.DTypeMXFP4}, true, hopper, true); got.suffix != "w4a16_mxfp4_grouped" {
		t.Errorf("MXFP4 grouped on Hopper moved to %s", got.suffix)
	}
}

// An entry of the efficiency's name scoped by hardware or model alone does not hide the
// fabric-scoped one, whichever comes later in the sets and whatever the model precedence.
func TestANonFabricEntryDoesNotHideTheFabricsEfficiency(t *testing.T) {
	roce := &hardware.Fabric{Name: "roce-200g", InterNodeBwGBps: 25}
	for _, c := range []struct {
		what  string
		other coefficient.Scope
	}{
		{"hardware-only after it", coefficient.Scope{Hardware: []string{"h100"}}},
		{"model-only after it", coefficient.Scope{Model: []string{"m"}}},
		{"model and hardware after it", coefficient.Scope{Hardware: []string{"h100"}, Model: []string{"m"}}},
	} {
		sets := fabricSet(efficiency(0.6, coefficient.Scope{Fabric: []string{"roce-200g"}}),
			efficiency(0.3, c.other))
		co, err := resolve.Load(sets, resolve.Scope{Hardware: "h100", Model: "m", Fabric: "roce-200g"})
		if err != nil {
			t.Fatal(err)
		}
		if e, ok, err := resolveFabricEfficiency(co, roce); err != nil || !ok || e != 0.6 {
			t.Errorf("%s: %v stated=%v err=%v, want the fabric's 0.6", c.what, e, ok, err)
		}
	}
}

// Original to blis-vllm.
// Moved from blis-substrate's kernel tests (kernel/sparse_mla_test.go, int4_test.go and
// refusals_v41_test.go at 00817b10798cbc00f560e51921c30c7084fa5e0c) with the resolution they test
// (blis-substrate#58).

package lowering

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
)

// A model-specific MTP name is the method vLLM runs as mtp, so a deployment naming mtp
// drafts with a graph's deepseek_mtp speculator.
func TestModelSpecificMTPNamesAreMTP(t *testing.T) {
	if got := specMethod("deepseek_mtp"); got != "mtp" {
		t.Errorf("deepseek_mtp normalises to %q", got)
	}
	if got := specMethod("dspark"); got != "dspark" {
		t.Errorf("dspark normalises to %q", got)
	}
}

// Layouts no catalog fixture reaches, asserted on the selection directly. A 128-head DSA
// model (DeepSeek-V3.2) at tp=8 holds 16 heads a rank: at dcp=8 they gather to 128, which pads
// to a different fp8 decode size from the local 16, and FlashMLA-sparse refuses it; at dcp=4
// they gather to 64 and it starts (flashmla_sparse.py:436-460, :684-687). With PCP the gathered
// count is heads x tp when dcp spans tp x pcp: at tp4/pcp2/dcp8, 32 local heads gather to
// 128 and are refused, where pcp2/dcp2 leaves them at 32 (vLLM admits dcp only as 1, pcp or
// tp x pcp under PCP, vllm/config/parallel.py:571-577). The a2a combine is refused on
// FlashMLA-sparse (:416-422), and a dtype no backend lists has none.
func TestSparseMLABackendRefusesWhatNoBackendServes(t *testing.T) {
	starts := func(r sparseMLABackendRequest) bool {
		if r.dcpComm == "" {
			r.dcpComm = DCPAllGatherReduceScatter
		}
		_, _, err := sparseMLABackend(r)
		return err == nil
	}
	for _, c := range []struct {
		name string
		r    sparseMLABackendRequest
		want bool
	}{
		{"128 heads, tp8 dcp8", sparseMLABackendRequest{cache: "fp8", headsPerRank: 16,
			tp: 8, dcp: 8, pcp: 1}, false},
		{"128 heads, tp8 dcp4", sparseMLABackendRequest{cache: "fp8", headsPerRank: 16,
			tp: 8, dcp: 4, pcp: 1}, true},
		{"128 heads, tp4 pcp2 dcp8", sparseMLABackendRequest{cache: "fp8", headsPerRank: 32,
			tp: 4, dcp: 8, pcp: 2}, false},
		{"128 heads, tp8 pcp2 dcp2", sparseMLABackendRequest{cache: "fp8", headsPerRank: 16,
			tp: 8, dcp: 2, pcp: 2}, true},
		{"a2a combine", sparseMLABackendRequest{cache: "fp8", headsPerRank: 8, tp: 8, dcp: 2,
			pcp: 1, dcpComm: DCPAllToAll}, false},
		{"fp8_e5m2 on hopper", sparseMLABackendRequest{cache: "fp8_e5m2", headsPerRank: 8,
			tp: 8, dcp: 1, pcp: 1}, false},
	} {
		if got := starts(c.r); got != c.want {
			t.Errorf("%s: starts %v, want %v", c.name, got, c.want)
		}
	}
}

// The generation of a part comes from its stated compute capability, never from its name:
// a part of any name with SM8.x is pre-Hopper, SM9 and SM10 are not, and a part that states
// none, or states it malformed, is refused with its name.
func TestPreHopperComesFromTheComputeCapability(t *testing.T) {
	for _, c := range []struct {
		name, cc string
		want     bool
	}{
		{"rtx-6000-ada", "8.9", true},
		{"a100-80", "8.0", true},
		{"my-sm80-part", "8.6", true},
		{"my-sm90-part", "9.0", false},
		{"a100-lookalike", "9.0", false},
		{"my-sm100-part", "10.0", false},
		{"my-sm103-part", "10.3", false},
	} {
		chip := hardware.Chip{Name: c.name, ComputeCapability: c.cc}
		family, _ := chip.CapabilityMajor()
		got, err := (&lowerContext{chip: chip, family: family}).preHopper()
		if err != nil || got != c.want {
			t.Errorf("%s (SM%s): preHopper = %v, %v; want %v", c.name, c.cc, got, err, c.want)
		}
	}
	for _, cc := range []string{"", "eight", "-8.0", "08.0", "9"} {
		chip := hardware.Chip{Name: "mystery-part", ComputeCapability: cc}
		family, _ := chip.CapabilityMajor()
		_, err := (&lowerContext{chip: chip, family: family}).preHopper()
		if err == nil || !strings.Contains(err.Error(), "mystery-part") {
			t.Errorf("capability %q: err = %v, want a refusal naming the chip", cc, err)
		}
	}
}

// The catalog's nine parts keep the answers the name lists gave: A100 and L40S are
// pre-Hopper, the rest are not, and the A100 defaults hold for the two A100 parts alone.
func TestCatalogPartsKeepTheirGenerationAndBatchDefaults(t *testing.T) {
	d := data
	names := d.Chips()
	if len(names) != 9 {
		t.Fatalf("catalog has %d chips, want 9: %v", len(names), names)
	}
	for _, n := range names {
		chip, _ := d.Chip(n)
		family, _ := chip.CapabilityMajor()
		pre, err := (&lowerContext{chip: *chip, family: family}).preHopper()
		if err != nil {
			t.Fatalf("%s: %v", n, err)
		}
		wantPre := n == "a100-80" || n == "a100-sxm" || n == "l40s"
		if pre != wantPre {
			t.Errorf("%s: preHopper = %v, want %v", n, pre, wantPre)
		}
		if isA100(*chip) != strings.Contains(n, "a100") {
			t.Errorf("%s: isA100 = %v", n, isA100(*chip))
		}
	}
}

// A DSA model on a part stating SM8.x is refused whatever its name (here the catalog name stays
// h100), where it used to price on the SM90 backend.
func TestADSAModelOnAnArbitrarilyNamedPreHopperPartIsRefused(t *testing.T) {
	for _, cc := range []string{"8.9", "9.0", "10.0", ""} {
		chip, err := data.Chip("h100")
		if err != nil {
			t.Fatal(err)
		}
		c := *chip
		c.ComputeCapability = cc
		_, err = tryLowerOn(t, where{"glm-5", "h100", "tp: 8, pp: 1, dp: 1"}, &c,
			vllmsettings.Settings{BlockSize: 64})
		switch cc {
		case "8.9":
			if err == nil || !strings.Contains(err.Error(), "below compute capability 9.0") {
				t.Errorf("SM89: err = %v, want the pre-Hopper refusal", err)
			}
		case "":
			if err == nil || !strings.Contains(err.Error(), "h100") {
				t.Errorf("no capability: err = %v, want a refusal naming the chip", err)
			}
		default:
			if err != nil && strings.Contains(err.Error(), "below compute capability 9.0") {
				t.Errorf("SM%s: refused as pre-Hopper: %v", cc, err)
			}
		}
	}
}

// An "auto" KV cache follows the model's COMPUTE dtype, not its weight storage width.
//
// A W4A16 checkpoint stores weights at four bits and computes in bf16. Following the
// storage width gave a half-byte KV element, which rounded to a per-block figure of
// zero once paged, and the kernel refused twelve Kimi-K2.5 sweeps rather than divide a
// budget by it. vLLM sizes an auto cache at the model dtype (vllm/platforms/interface.py:
// 861-862 at v0.31.0) unless the checkpoint's quantization config declares a KV algorithm,
// which the graph does not carry; see cacheDTypeBytes.
func TestAnAutoCacheFollowsTheComputeWidthNotTheStorageWidth(t *testing.T) {
	for _, tc := range []struct {
		served model.DType
		want   float64
		why    string
	}{
		{model.DTypeINT4, 2, "W4A16 computes in bf16"},
		{model.DTypeNVFP4, 2, "a 4-bit float is a storage format too"},
		{model.DTypeMXFP4, 2, "likewise"},
		{model.DTypeFP8, 2, "fp8 linears return bf16 (out_dtype=x.dtype), so the model dtype is bf16"},
		{model.DTypeBF16, 2, "unquantized passes through"},
		{model.DTypeFP32, 4, "and so does fp32"},
	} {
		if got := cacheDTypeBytes("auto", tc.served); got != tc.want {
			t.Errorf("auto cache on a %s model = %.1f bytes, want %.1f (%s)",
				tc.served, got, tc.want, tc.why)
		}
		// An empty setting means the engine's default, which is auto.
		if got := cacheDTypeBytes("", tc.served); got != tc.want {
			t.Errorf("empty cache dtype on a %s model = %.1f, want %.1f",
				tc.served, got, tc.want)
		}
	}
	// An explicitly named narrow cache is still honoured: the rule above is about what
	// "auto" resolves to, not a floor on every cache.
	if got := cacheDTypeBytes("fp8", model.DTypeINT4); got != 1 {
		t.Errorf("an explicit fp8 cache = %.1f bytes, want 1", got)
	}
	// nvfp4 carries an fp8 scale per 16 elements beside its four-bit data:
	// head_size/2 + head_size/16 bytes a head (vllm/utils/torch_utils.py:543-545).
	if got := cacheDTypeBytes("nvfp4", model.DTypeBF16); got != 0.5625 {
		t.Errorf("an explicit nvfp4 cache = %.4f bytes, want 0.5625", got)
	}
}

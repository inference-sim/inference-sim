package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
	"github.com/inference-sim/blis-substrate/load"
)

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
			r.dcpComm = resolve.DCPAllGatherReduceScatter
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
			pcp: 1, dcpComm: resolve.DCPAllToAll}, false},
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
		got, err := (&Kernel{chip: chip, family: family}).preHopper()
		if err != nil || got != c.want {
			t.Errorf("%s (SM%s): preHopper = %v, %v; want %v", c.name, c.cc, got, err, c.want)
		}
	}
	for _, cc := range []string{"", "eight", "-8.0", "08.0", "9"} {
		chip := hardware.Chip{Name: "mystery-part", ComputeCapability: cc}
		family, _ := chip.CapabilityMajor()
		_, err := (&Kernel{chip: chip, family: family}).preHopper()
		if err == nil || !strings.Contains(err.Error(), "mystery-part") {
			t.Errorf("capability %q: err = %v, want a refusal naming the chip", cc, err)
		}
	}
}

// The catalog's nine parts keep the answers the name lists gave: A100 and L40S are
// pre-Hopper, the rest are not, and the A100 defaults hold for the two A100 parts alone.
func TestCatalogPartsKeepTheirGenerationAndBatchDefaults(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	names := d.Chips()
	if len(names) != 9 {
		t.Fatalf("catalog has %d chips, want 9: %v", len(names), names)
	}
	for _, n := range names {
		chip, _ := d.Chip(n)
		family, _ := chip.CapabilityMajor()
		pre, err := (&Kernel{chip: *chip, family: family}).preHopper()
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

// A DSA model on a part stating SM8.x is refused at construction whatever its name (here the
// catalog name stays h100, which the scenario checks), where it used
// to price on the SM90 backend.
func TestADSAModelOnAnArbitrarilyNamedPreHopperPartIsRefused(t *testing.T) {
	for _, cc := range []string{"8.9", "9.0", "10.0", ""} {
		d, err := load.Embedded()
		if err != nil {
			t.Fatal(err)
		}
		body := `kind: Scenario
name: dsa-chip
model: glm-5
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric]
engine_version: "0.32.0"
cluster:
  hardware: h100
  nodes: 1
  gpus_per_node: 8
---
kind: Deployment
name: tp8
pools:
  - role: colocated
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
`
		r, name := openRepos(t, d, body)
		r.Rules = func(string) rules.EngineRules { return inertRules{} }
		r.EngineSettings = func(*deployment.Deployment, int) (enginesettings.Settings, error) {
			return enginesettings.Settings{BlockSize: 64, MaxNumSeqs: 64, MaxNumBatchedTokens: 8192,
				MaxModelLen: 8192}, nil
		}
		in, err := OpenInputs(name, r, 0)
		if err != nil {
			t.Fatal(err)
		}
		chip := *in.Chip
		chip.ComputeCapability = cc
		in.Chip = &chip
		_, err = New(in)
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

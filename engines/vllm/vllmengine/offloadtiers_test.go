package vllmengine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// tieredScenario is Muse Spark's arm D PP1 sync scenario with TieringOffloadingSpec and tier, a
// tier below the CPU tier, written to a temporary file; its roots.
func tieredScenario(t *testing.T, tier string) map[string]string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "muse-h100-roce-armD-pp1-sync.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const cpu = "    - {tier: cpu_dram, bytes: 549755813888}\n"
	s := string(raw)
	if !strings.Contains(s, cpu) || !strings.Contains(s, "  connector: OffloadingConnector\n") {
		t.Fatal("the fixture's offload block changed")
	}
	s = strings.Replace(s, cpu, cpu+tier, 1)
	s = strings.Replace(s, "  connector: OffloadingConnector\n", "  connector: OffloadingConnector\n  spec: TieringOffloadingSpec\n", 1)
	path := filepath.Join(t.TempDir(), "s.yaml")
	if err := os.WriteFile(path, []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
	r := roots(t, "muse-h100-roce-armD-pp1-sync.yaml")
	r[RootScenario] = path
	return r
}

// A deployment's tier below the CPU tier, under TieringOffloadingSpec, is the fs tier on that
// catalog device, and the engine prices its jobs on the device: a store of n bytes is the kernel's
// TierTime to nvme_gen4 for n bytes.
func TestATierBelowTheCPUTierIsAnFSTierPricedOnItsDevice(t *testing.T) {
	Register()
	spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: tieredScenario(t, "    - {tier: nvme_gen4, bytes: 1099511627776}\n")}
	done, err := prepare(spec, nil)
	if err != nil {
		t.Fatal(err)
	}
	var b struct {
		O struct {
			Spec  string           `yaml:"spec"`
			Tiers []map[string]any `yaml:"secondary_tiers"`
		} `yaml:"kv_offload"`
	}
	if err := done.Config.Decode(&b); err != nil {
		t.Fatal(err)
	}
	if b.O.Spec != "TieringOffloadingSpec" || len(b.O.Tiers) != 1 || b.O.Tiers[0]["type"] != "fs" ||
		b.O.Tiers[0]["device_class"] != "nvme_gen4" {
		t.Fatalf("kv_offload spec %q, secondary_tiers %v: want one fs tier on nvme_gen4", b.O.Spec, b.O.Tiers)
	}
	p, err := pricerFor(spec, nil)
	if err != nil {
		t.Fatal(err)
	}
	k := p.(kernelPricer)
	for _, store := range []bool{true, false} {
		kind := engine.TransferLoad
		if store {
			kind = engine.TransferStore
		}
		tr := engine.Transfer{Kind: kind, Tier: "1:fs", Blocks: 4}
		tr.Move.Tier = "nvme_gen4"
		got := k.TierTransferTime(tr, 1<<24)
		want := engine.Duration(k.m.TierTransferTicksOf("nvme_gen4", store, 1<<24, 4))
		cpu := engine.Duration(k.m.TierTransferTicksOf("cpu_dram", store, 1<<24, 4))
		if got != want || got < 1 || got == cpu {
			t.Fatalf("a 16 MiB tier job (store %v) priced %d, want the kernel's nvme_gen4 price %d (cpu_dram's is %d)", store, got, want, cpu)
		}
	}
}

// A tier on a device the kernel has no coefficients for is refused, naming the tier, the
// coefficients and the issue.
func TestATierBelowTheCPUTierItCannotRunIsRefused(t *testing.T) {
	Register()
	for name, c := range map[string]struct{ tier, want string }{
		"unpriced device": {"    - {tier: tape_drive_x, bytes: 1099511627776}\n", "offload_tier_tape_drive_x_"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := prepare(engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: tieredScenario(t, c.tier)}, nil)
			if err == nil || !strings.Contains(err.Error(), c.want) || !strings.Contains(err.Error(), "offload.tiers[1]") ||
				!strings.Contains(err.Error(), "issues/63") {
				t.Fatalf("err %v, want a refusal of offload.tiers[1] naming %q and issues/63", err, c.want)
			}
		})
	}
}

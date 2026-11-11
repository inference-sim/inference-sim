package latencykernel

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
	"github.com/inference-sim/blis-substrate/load"
)

// tierBefore is what the catalog's devices/storage.yaml stated for each tier before the terms
// moved to the registry (SUB-18): read MB/s, write MB/s, base latency us. The registry entries
// must carry exactly these on every chip, so no offload price moves; a deliberate change to one
// edits this table and says why.
var tierBefore = map[string][3]float64{
	"nvme_gen4":          {7000, 5000, 80},
	"nvme_gen3":          {3500, 3000, 100},
	"sata_ssd":           {550, 500, 150},
	"cpu_dram":           {27058, 46089, 19.3},
	"s3":                 {1000, 1000, 30000},
	"nvme_gen4_raid0_8x": {50000, 30000, 100},
}

// tierCases builds a small model on every catalog chip with the deployment offloading to every
// tier, so each kernel reads (and reports) the tier entries.
func tierCases(t *testing.T, d *load.Data, offload bool) []perturbationCase {
	t.Helper()
	cases := perturbationScenarios(t, d, "llama-3.1-8b-instruct")
	if len(cases) < len(d.Chips()) {
		t.Fatalf("%d cases for %d chips: the model must build on every chip", len(cases), len(d.Chips()))
	}
	seen := map[string]bool{}
	var out []perturbationCase
	for _, c := range cases {
		chip := c.in.Chip.Name
		if seen[chip] {
			continue
		}
		seen[chip] = true
		if !offload {
			dep := *c.in.Deployment
			dep.Offload = nil
			c.in.Deployment = &dep
		}
		out = append(out, c)
	}
	return out
}

// hostCap is the chip's registry host_link_bandwidth, bytes per microsecond, from the sets the
// case lists.
func hostCap(t *testing.T, c perturbationCase) float64 {
	t.Helper()
	for _, s := range c.in.Coefficients {
		for _, e := range s.Coefficients {
			if e.Name == "host_link_bandwidth" && e.Scope.Admits(coefficient.Point{Hardware: c.in.Chip.Name}) {
				return e.Value
			}
		}
	}
	t.Fatalf("no host_link_bandwidth for %s", c.in.Chip.Name)
	return 0
}

// SUB-18: a tier's terms are the registry's, and TierTime prices a transfer from them as it
// priced it from the catalog's figures: max(floor, bytes / rate) per direction, the rate capped
// at the chip's host link and divided by the queue depth. Every tier on every chip is priced at
// the figures the catalog held (tierBefore), so no offload price moved when they moved.
func TestTierTimeReadsTheRegistryAndPricesAsTheCatalogDid(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, c := range tierCases(t, d, true) {
		k, err := New(c.in)
		if err != nil {
			t.Fatalf("%s: %v", c.label, err)
		}
		cap := hostCap(t, c)
		for tier, f := range tierBefore {
			for _, dir := range []kernel.Direction{kernel.DirectionFromTier, kernel.DirectionToTier} {
				rate := f[0]
				if dir == kernel.DirectionToTier {
					rate = f[1]
				}
				for _, bytes := range []int64{0, 4096, 1 << 20, 1 << 30} {
					for _, qd := range []int{0, 1, 3} {
						r := math.Min(rate, cap) * 1e6
						if qd > 1 {
							r /= float64(qd)
						}
						want := seconds(math.Max(f[2]*1e-6, float64(bytes)/r))
						if bytes <= 0 {
							want = seconds(f[2] * 1e-6)
						}
						if got := k.TierTime(tier, dir, bytes, qd); got != want {
							t.Errorf("%s %s %s %d bytes qd %d: %v, want %v", c.in.Chip.Name, tier, dir, bytes, qd, got, want)
						}
						checked++
					}
				}
			}
		}
	}
	if checked < 9*6*2*4*3 {
		t.Errorf("%d prices checked; every tier, direction, size and depth on every chip is expected", checked)
	}
}

// A tier the registry has no terms for is not a free transfer, and a deployment that offloads to
// it is refused with the entries and the sets named. Removing the terms is what the test does;
// the same kernel with them prices the tier.
func TestATierWithNoTermsIsRefusedWhereTheDeploymentUsesIt(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	c := tierCases(t, d, true)[0]
	without := func(drop string) []*coefficient.Set {
		var sets []*coefficient.Set
		for _, s := range c.in.Coefficients {
			cp := *s
			cp.Coefficients = nil
			for _, e := range s.Coefficients {
				if e.Name != drop {
					cp.Coefficients = append(cp.Coefficients, e)
				}
			}
			sets = append(sets, &cp)
		}
		return sets
	}
	name := coefficient.OffloadTierWriteRate("s3")
	in := c.in
	in.Coefficients = without(name)
	if _, err := New(in); err == nil {
		t.Fatal("a deployment offloading to a tier with no write rate was built")
	} else if !strings.Contains(err.Error(), name) || !strings.Contains(err.Error(), "s3") {
		t.Errorf("the refusal does not name the tier and the entry: %v", err)
	}
	// Not named by the deployment: the kernel builds, and the tier is unknown to TierTime.
	dep := *c.in.Deployment
	dep.Offload = &deployment.Offload{Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1 << 30}}}
	in.Deployment = &dep
	k, err := New(in)
	if err != nil {
		t.Fatalf("a tier the deployment does not use was refused: %v", err)
	}
	if got := k.TierTime("s3", kernel.DirectionToTier, 4096, 1); got != time.Duration(math.MaxInt64) {
		t.Errorf("a tier with no terms priced at %v; an unpriced tier is not a free transfer", got)
	}
	if got := k.TierTime("cpu_dram", kernel.DirectionToTier, 4096, 1); got <= 0 || got == time.Duration(math.MaxInt64) {
		t.Errorf("the tier that kept its terms priced at %v", got)
	}
}

// The value TierTime prices is the registry's: scaling the entry scales the price. This fails
// if the kernel kept reading the catalog's figure, which no longer exists, or a constant.
func TestTierTimeFollowsTheRegistryEntry(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	c := tierCases(t, d, true)[0]
	base, err := New(c.in)
	if err != nil {
		t.Fatal(err)
	}
	in := c.in
	in.Coefficients = perturbed(c.in.Coefficients, coefficient.OffloadTierReadRate("sata_ssd"))
	scaled, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	const big = 1 << 30
	a := base.TierTime("sata_ssd", kernel.DirectionFromTier, big, 1)
	b := scaled.TierTime("sata_ssd", kernel.DirectionFromTier, big, 1)
	if r := float64(b) / float64(a); math.Abs(r-1/0.8) > 1e-6 {
		t.Errorf("a read rate scaled by 0.8 changed the price by %v, want %v", r, 1/0.8)
	}
	if base.TierTime("sata_ssd", kernel.DirectionToTier, big, 1) != scaled.TierTime("sata_ssd", kernel.DirectionToTier, big, 1) {
		t.Error("the read rate moved the write price")
	}
}

// A kernel reports the tier entries its deployment offloads to, and no others: a scenario that
// offloads nowhere lists none of them in Provenance, so they do not dilute its evidence summary,
// and no tier entry enters the trust of a step price, which no tier term prices.
func TestOnlyTheTiersADeploymentUsesAreReported(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	count := func(k *Kernel) (origins, selected int) {
		for _, o := range k.Provenance() {
			if strings.HasPrefix(o.Name, coefficient.OffloadTierPrefix) {
				origins++
			}
		}
		for _, r := range k.selected {
			if strings.HasPrefix(r.Entry.Name, coefficient.OffloadTierPrefix) {
				selected++
			}
		}
		return
	}
	for _, c := range tierCases(t, d, false) {
		k, err := New(c.in)
		if err != nil {
			t.Fatal(err)
		}
		if o, s := count(k); o != 0 || s != 0 {
			t.Errorf("%s: a deployment that offloads nowhere reports %d tier origins and %d selected", c.in.Chip.Name, o, s)
		}
	}
	for _, c := range tierCases(t, d, true) {
		k, err := New(c.in)
		if err != nil {
			t.Fatal(err)
		}
		if o, s := count(k); o != 18 || s != 18 {
			t.Errorf("%s: six tiers named, %d tier origins and %d selected, want 18 of each", c.in.Chip.Name, o, s)
		}
		step := k.StepTime(kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 1, Computed: 100, PromptLen: 101}}})
		if strings.HasPrefix(step.Trust.Entry, coefficient.OffloadTierPrefix) {
			t.Errorf("%s: a step price's trust is set by a tier entry %s", c.in.Chip.Name, step.Trust.Entry)
		}
	}
}

// VAL-2 for the tiers: every catalog tier has all three terms on every catalog chip, in the
// offload-tiers set, as a method that ranks and a unit the formula needs. A chip added to the
// catalog, or a tier, fails here until the writer (registry/scripts/emit_offload_tiers.py) has
// run for it.
func TestEveryTierHasItsTermsOnEveryChip(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	set, err := d.CoefficientSet("cost-model-offload-tiers")
	if err != nil {
		t.Fatal(err)
	}
	devices := d.StorageDevices()
	if len(devices) != len(tierBefore) {
		t.Fatalf("the catalog has %d tiers; this test's table has %d", len(devices), len(tierBefore))
	}
	for _, chip := range d.Chips() {
		for _, dev := range devices {
			for i, name := range []string{coefficient.OffloadTierReadRate(dev.Name),
				coefficient.OffloadTierWriteRate(dev.Name), coefficient.OffloadTierBaseLatency(dev.Name)} {
				var hits []coefficient.Entry
				for _, e := range set.Coefficients {
					if e.Name == name && e.Scope.Admits(coefficient.Point{Hardware: chip}) {
						hits = append(hits, e)
					}
				}
				if len(hits) != 1 {
					t.Errorf("%s on %s: %d entries admit it, want exactly one", name, chip, len(hits))
					continue
				}
				e := hits[0]
				if want := tierBefore[dev.Name][i]; e.Value != want {
					t.Errorf("%s on %s is %v; the catalog held %v", name, chip, e.Value, want)
				}
				if _, ranked := e.Method.Rank(); !ranked {
					t.Errorf("%s on %s has method %q, which ranks nowhere", name, chip, e.Method)
				}
				// A fitted or assumed value says so, and a tier's figures are neither datasheet
				// facts nor measured on the engine they price (SUB-18, D5).
				if e.Method == vocab.MethodMeasured || e.Method == vocab.MethodVendorSpec {
					t.Errorf("%s on %s is %s; no tier figure is measured on the path priced or a datasheet's", name, chip, e.Method)
				}
			}
		}
	}
}

// A tier's terms price a transfer, not a step, so they are not among the entries a step price's
// trust is the least of (VAL-3). An assumed tier rate beside measured step terms leaves the step
// price measured; with the tier term counted, a step would read as assumed because a drive was.
func TestATierTermDoesNotSetAStepsTrust(t *testing.T) {
	terms := buildTrust([]resolve.Resolved{
		res("gemm_eps_max_bf16", vocab.MethodMeasured, 0.9, nil),
		res(coefficient.OffloadTierReadRate("s3"), vocab.MethodAssumed, 1000, nil),
		res(coefficient.OffloadTierBaseLatency("cpu_dram"), vocab.MethodAssumed, 19.3, nil),
	})
	got := trustOf(terms, map[string]float64{"tokens": 1, "requests": 1, "context_tokens": 1})
	if got.Rank != 1 || got.Entry != "gemm_eps_max_bf16" {
		t.Errorf("a step's trust is %+v; the tier terms are assumed and must not set it", got)
	}
}

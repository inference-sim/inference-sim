package latencykernel

import (
	"fmt"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// liftTiers reads what a transfer to or from each catalog storage tier costs, from the
// registry (design SUB-18). The catalog names a tier and states no number for it: its read rate,
// write rate and fixed cost per transfer are coefficients, offload_tier_<tier>_read_rate and its
// two companions, scoped to the hardware they describe and chosen by the same lookup as every
// other coefficient.
//
// A tier whose three terms the scenario's coefficient sets do not all carry for this chip is
// not priced: TierTime refuses it as an unknown tier, because a transfer is not free for want of
// a figure. If the deployment's offload block names it, the kernel is not built, and the error
// names the entries and the sets, since a deployment that offloads to a tier it cannot price
// would otherwise price every hit at an infinite cost without a word.
//
// A tier the deployment names is read with a lookup, so the selected-entry report lists its
// three entries and Provenance describes them. A tier it does not name is priced too, for a
// caller that asks TierTime about a tier of its own choosing, but is not reported: a scenario that
// offloads nowhere should not list eighteen entries it never read. Neither kind enters the trust
// of a step price (buildTrust): a tier's terms price a transfer, not a step.
func (k *Kernel) liftTiers(c *resolve.Coefficients, in Inputs) error {
	used := map[string]bool{}
	if in.Deployment != nil && in.Deployment.Offload != nil {
		for _, t := range in.Deployment.Offload.Tiers {
			if t.Device != "" {
				used[t.Device] = true
			}
		}
	}
	k.tiers = map[string]offloadTier{}
	k.tierEntries = map[string]bool{}
	for _, d := range in.Devices {
		if d == nil {
			continue
		}
		names := [3]string{
			coefficient.OffloadTierReadRate(d.Name),
			coefficient.OffloadTierWriteRate(d.Name),
			coefficient.OffloadTierBaseLatency(d.Name),
		}
		var missing []string
		for _, n := range names {
			if !c.Has(n) {
				missing = append(missing, n)
			}
		}
		if len(missing) > 0 {
			if used[d.Name] {
				return fmt.Errorf("the deployment offloads to tier %q, but the coefficient sets %v carry no %s "+
					"for hardware %s: a tier's read rate, write rate and base latency are registry entries "+
					"(the set cost-model-offload-tiers), and without them every transfer would price as infinitely slow",
					d.Name, c.Sets(), strings.Join(missing, ", "), in.Chip.Name)
			}
			continue
		}
		get := func(name string) float64 {
			if used[d.Name] {
				v, _ := c.Value(name) // recorded as selected; Has said it is there
				return v
			}
			r, _ := c.Admitted(name)
			return r.Entry.Value
		}
		k.tiers[d.Name] = offloadTier{readRate: get(names[0]), writeRate: get(names[1]), baseLatency: get(names[2])}
		if used[d.Name] {
			for _, n := range names {
				k.tierEntries[n] = true
			}
		}
	}
	return nil
}

// isOffloadTierEntry reports whether a coefficient name is one of a tier's three terms.
func isOffloadTierEntry(name string) bool {
	return strings.HasPrefix(name, coefficient.OffloadTierPrefix)
}

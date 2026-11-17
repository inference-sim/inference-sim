package cmd

import "testing"

// A run whose engines copied KV between tiers states, in the Result's transfer figures, the most
// transfers that shared each link (#2108), colocated or not.
func TestTheResultStatesTheLinksTierCopiesShared(t *testing.T) {
	got := pdBreakdown(nil, map[string]int{"tier/cpu_dram/store/node0": 2})
	if got["link_peak_concurrent_transfers_tier-cpu_dram-store-node0"] != 2 || len(got) != 1 {
		t.Errorf("colocated run's transfer figures %v, want the link's peak of 2", got)
	}
	if pdBreakdown(nil, nil) != nil {
		t.Error("a run with no disaggregation and no tier copy states transfer figures")
	}
}

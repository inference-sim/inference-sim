package cmd

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// A run whose engines copied KV between tiers states, in the Result's transfer figures, the most
// transfers that shared each link (#2108), colocated or not.
func TestTheResultStatesTheLinksTierCopiesShared(t *testing.T) {
	got := pdBreakdown(nil, map[string]int{"tier/cpu_dram/store/node0": 2}, cluster.PeerStats{})
	if got["link_peak_concurrent_transfers_tier-cpu_dram-store-node0"] != 2 || len(got) != 1 {
		t.Errorf("colocated run's transfer figures %v, want the link's peak of 2", got)
	}
	if pdBreakdown(nil, nil, cluster.PeerStats{}) != nil {
		t.Error("a run with no disaggregation and no tier copy states transfer figures")
	}
}

// A run whose engines exchanged peer messages states the delivery latency it assumed (the named
// substitution peer_message_delivery_latency) and how many messages it carried and dropped; a run
// with none states nothing.
func TestTheResultStatesThePeerNetworkOfARunWhoseEnginesTalked(t *testing.T) {
	got := pdBreakdown(nil, nil, cluster.PeerStats{Sent: 5, Delivered: 3, Dropped: 2, LatencyTicks: 250})
	want := map[string]float64{"peer_message_delivery_latency_us": 250, "peer_messages_sent": 5,
		"peer_messages_delivered": 3, "peer_messages_dropped": 2}
	if len(got) != len(want) {
		t.Fatalf("figures %v, want %v", got, want)
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s = %v, want %v", k, got[k], v)
		}
	}
	if pdBreakdown(nil, nil, cluster.PeerStats{}) != nil {
		t.Error("a run with no peer messages stated peer figures")
	}
}

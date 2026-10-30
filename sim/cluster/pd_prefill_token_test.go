package cluster

import "testing"

// A P/D prefill leg that samples the request's first token (PDPrefillSamplesFirstToken, the
// proxy's max_tokens 1) does not add that token to the cluster's output: the client receives the
// decode leg's tokens only, so total_output_tokens and tokens_per_sec count those (INV-9).
func TestPDPrefillSampledToken_NotCountedInClusterOutput(t *testing.T) {
	config := newTestDisaggDeploymentConfig(2, 1, 1)
	config.PDPrefillSamplesFirstToken = true
	requests := newTestRequests(5)
	cs := newTestCluster(config, NewSliceRequestSource(requests), nil)
	mustRun(t, cs)

	var prefillTokens, clientTokens int
	for _, parent := range cs.parentRequests {
		prefillTokens += parent.PrefillSubReq.GeneratedTokens
		clientTokens += parent.OriginalRequest.GeneratedTokens
		if want := len(parent.OriginalRequest.OutputTokens); parent.OriginalRequest.GeneratedTokens != want {
			t.Errorf("%s: client received %d tokens, want the full %d", parent.ID, parent.OriginalRequest.GeneratedTokens, want)
		}
	}
	if prefillTokens != len(requests) {
		t.Fatalf("prefill legs sampled %d tokens, want one per request (%d); the test would be vacuous", prefillTokens, len(requests))
	}
	if got := cs.AggregatedMetrics().TotalOutputTokens; got != clientTokens {
		t.Errorf("cluster TotalOutputTokens = %d, want the %d tokens the clients received (prefill legs sampled %d)",
			got, clientTokens, prefillTokens)
	}
}

package cmd

import (
	"math"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/metrics"
)

// Under speculative decoding one verification step streams 1 + accepted tokens to the client in
// one chunk, and the benchmark clients' "ITL" figures part ways (#2128):
//
//   - vllm bench serve's ITL and aiperf's inter_chunk_latency pool one gap per chunk after the
//     first: the registry's icl. One request's mean gap is its (E2E - TTFT) / (chunks - 1).
//   - aiperf's inter_token_latency is per request, (E2E - TTFT) / (output tokens - 1): tpot.
//   - inference-perf with a tokenizer gives every token of a chunk the chunk's arrival, so most
//     of its pooled gaps are zero: itl, whose median is therefore 0 here.
//
// The run is one DSpark K=5 request accepting every draft, so each verification step streams six
// tokens and nothing else shares the engine. Before #2128 the Result stated no icl_*, and the
// only pooled client gap it stated, itl_p50_ms, was 0.
func TestResult_SpeculativeRunStatesEachClientITLByItsDefinition(t *testing.T) {
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", pdScenarios, "--scenario", dsparkScenario,
		"--num-requests", "1", "--rate", "1", "--seed", "5", "--speculative-acceptance-rate", "1")
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	res := parseResult(t, stdout)
	if res.Requests.Completed != 1 || len(res.RequestRows) != 1 {
		t.Fatalf("%d requests completed, %d rows; the test needs the one request's row", res.Requests.Completed, len(res.RequestRows))
	}
	row := res.RequestRows[0]
	if row.E2EUs == nil || row.TTFTUs == nil || row.TPOTUs == nil || row.OutputTokens < 13 {
		t.Fatalf("the row states no E2E, TTFT or TPOT, or too few tokens (%d) for two full chunks: %+v", row.OutputTokens, row)
	}
	decodeMs := float64(*row.E2EUs-*row.TTFTUs) / 1e3

	icl, ok := res.Metrics["icl_mean_ms"]
	if !ok {
		t.Fatal("the Result states no icl_mean_ms: vllm bench serve's and aiperf's per-chunk ITL is missing")
	}
	chunks := icl.Samples + 1 // one gap before each chunk after the first
	if icl.Value <= 0 {
		t.Errorf("icl_mean_ms = %v, want a positive per-chunk gap", icl.Value)
	}
	if want := decodeMs / float64(chunks-1); math.Abs(icl.Value-want) > 1e-6*want {
		t.Errorf("icl_mean_ms = %v, want (E2E - TTFT) / (chunks - 1) = %v / %d = %v", icl.Value, decodeMs, chunks-1, want)
	}
	// Six tokens a chunk: the chunks are about a sixth of the tokens (the first and the last
	// chunk may carry fewer).
	if perChunk := float64(row.OutputTokens) / float64(chunks); perChunk < 5 || perChunk > 6 {
		t.Errorf("%d tokens in %d chunks, %.2f a chunk; want close to K+1 = 6 under full acceptance", row.OutputTokens, chunks, perChunk)
	}
	for _, id := range []string{"icl_p50_ms", "icl_p90_ms", "icl_p99_ms"} {
		if f := res.Metrics[id]; f.Value <= 0 || f.Samples != icl.Samples {
			t.Errorf("%s = %+v, want a positive gap over the %d chunk gaps", id, f, icl.Samples)
		}
	}

	tpot := metric(t, stdout, "tpot_mean_ms")
	if want := decodeMs / float64(row.OutputTokens-1); math.Abs(tpot-want) > 1e-6*want {
		t.Errorf("tpot_mean_ms = %v, want aiperf's ITL (E2E - TTFT) / (tokens - 1) = %v", tpot, want)
	}
	if got := metric(t, stdout, "itl_p50_ms"); got != 0 {
		t.Errorf("itl_p50_ms = %v, want 0: five of every six gaps between tokens are inside a chunk", got)
	}

	// The three figures are three metrics: a comparison of any two is refused (DEV-8).
	for _, pair := range [][2]string{{"icl_p50_ms", "itl_p50_ms"}, {"icl_mean_ms", "tpot_mean_ms"}, {"itl_mean_ms", "tpot_mean_ms"}} {
		a, okA := metrics.Lookup(pair[0])
		b, okB := metrics.Lookup(pair[1])
		if !okA || !okB {
			t.Fatalf("%s or %s is not in the registry", pair[0], pair[1])
		}
		fa := metrics.Figure{Definition: a, Value: 1, Scope: metrics.ScopeCluster}
		fb := metrics.Figure{Definition: b, Value: 1, Scope: metrics.ScopeCluster}
		if _, err := metrics.Compare(fa, fb); err == nil {
			t.Errorf("Compare(%s, %s) succeeded; the definitions differ", pair[0], pair[1])
		}
	}
}

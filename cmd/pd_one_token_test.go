package cmd

import (
	"slices"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
)

// A disaggregated request that asks for one output token completes with that token and its
// latencies, as a colocated one does: the single token leaves no inter-token gap, and the
// absence of a gap used to read as "no token served", so the parent lost its token count, its
// TTFT/E2E/scheduling delay and every sample of the throughput figures.
func TestRunCmd_KernelBackend_PDOneTokenCompletionKeepsTokenAndLatencies(t *testing.T) {
	const n = 2
	args := append(append([]string(nil), pdTopology...), "--num-requests", "2", "--rate", "5",
		"--prompt-tokens", "16", "--prompt-tokens-min", "16", "--prompt-tokens-max", "16", "--prompt-tokens-stdev", "0",
		"--output-tokens", "1", "--output-tokens-min", "1", "--output-tokens-max", "1", "--output-tokens-stdev", "0")
	args[slices.Index(args, "--seed")+1] = "3"
	args[slices.Index(args, "--horizon")+1] = "900000000"
	out, stderr, err := runKernelCLI(t, append([]string{"run"}, args...)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	r := parseResult(t, out)
	if r.Requests.Completed != n {
		t.Fatalf("completed %d requests, want %d; the test would be vacuous", r.Requests.Completed, n)
	}
	if len(r.RequestRows) != n {
		t.Fatalf("%d request rows, want %d", len(r.RequestRows), n)
	}
	for _, row := range r.RequestRows {
		if row.Fate != result.FateCompleted {
			t.Errorf("%s: fate %s, want completed", row.ID, row.Fate)
		}
		if row.OutputTokens != 1 {
			t.Errorf("%s: output_tokens %d, want 1", row.ID, row.OutputTokens)
		}
		if row.TTFTUs == nil || row.E2EUs == nil || row.SchedulingDelayUs == nil {
			t.Errorf("%s: TTFT, E2E and scheduling delay must be reported (ttft %v, e2e %v, delay %v)",
				row.ID, row.TTFTUs, row.E2EUs, row.SchedulingDelayUs)
			continue
		}
		if *row.TTFTUs <= 0 || *row.E2EUs < *row.TTFTUs {
			t.Errorf("%s: TTFT %d us, E2E %d us; the one token is the last token, so E2E >= TTFT > 0", row.ID, *row.TTFTUs, *row.E2EUs)
		}
	}
	for _, name := range []string{"request_throughput", "output_tok_s", "total_tokens_per_second_per_gpu"} {
		if f := r.Metrics[name]; f.Samples != n || f.Value <= 0 {
			t.Errorf("%s: value %v over %d samples, want a positive value over %d", name, f.Value, f.Samples, n)
		}
	}
}

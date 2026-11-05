package vllm

import (
	"strings"
	"testing"
)

// ResolveBatch is vLLM v0.32.0's resolution of an unstated max_num_seqs and
// max_num_batched_tokens (vllm/engine/arg_utils.py:2893-2985, 3097-3186@5df63a72), one case
// per branch of the rule; a stated value is kept whatever the branch.
func TestResolveBatchFollowsVLLMsRule(t *testing.T) {
	const g = int64(1) << 30
	base := BatchInput{ChunkedPrefill: true, MaxModelLen: 131072}
	with := func(f func(*BatchInput)) BatchInput { in := base; f(&in); return in }
	for _, c := range []struct {
		name         string
		in           BatchInput
		seqs, tokens int
		seqsFrom     string // a fragment of each source
		tokensFrom   string
	}{
		{"at least 160 GiB (b200)", with(func(i *BatchInput) { i.DeviceMemoryBytes, i.DeviceName = 180*g, "b200" }),
			1024, 16384, "get_batch_defaults", "get_batch_defaults"},
		{"exactly 160 GiB", with(func(i *BatchInput) { i.DeviceMemoryBytes = 160 * g }), 1024, 16384, "", ""},
		{"at least 70 GiB, not an A100 (h200)", with(func(i *BatchInput) { i.DeviceMemoryBytes, i.DeviceName = 141*g, "h200" }),
			1024, 8192, "", ""},
		{"exactly 70 GiB", with(func(i *BatchInput) { i.DeviceMemoryBytes = 70 * g }), 1024, 8192, "", ""},
		{"an 80 GiB A100, by name", with(func(i *BatchInput) { i.DeviceMemoryBytes, i.DeviceName = 80*g, "NVIDIA A100-SXM4-80GB" }),
			256, 2048, "", ""},
		{"below 70 GiB (l40s)", with(func(i *BatchInput) { i.DeviceMemoryBytes, i.DeviceName = 48*g, "l40s" }), 256, 2048, "", ""},
		{"batched data-parallel MoE", with(func(i *BatchInput) { i.DeviceMemoryBytes, i.BatchedDPMoE = 141*g, true }),
			// max_num_seqs then caps at the 256-token budget: 1024 -> 256.
			256, 256, "capped at max_num_batched_tokens", "DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP"},
		{"throughput mode doubles both", with(func(i *BatchInput) { i.DeviceMemoryBytes, i.ThroughputMode = 141*g, true }),
			2048, 16384, "doubled", "doubled"},
		{"throughput mode keeps a stated value", with(func(i *BatchInput) {
			i.DeviceMemoryBytes, i.ThroughputMode, i.MaxNumSeqs, i.MaxNumBatchedTokens = 141*g, true, 512, 4096
		}), 512, 4096, SourceStated, SourceStated},
		{"capped at max_num_seqs x max_model_len", with(func(i *BatchInput) { i.DeviceMemoryBytes, i.MaxModelLen = 141*g, 4 }),
			1024, 4096, "", "capped at max_num_seqs x max_model_len"},
		{"the cap uses a stated max_num_seqs", with(func(i *BatchInput) { i.DeviceMemoryBytes, i.MaxModelLen, i.MaxNumSeqs = 141*g, 64, 8 }),
			8, 512, SourceStated, "capped at max_num_seqs x max_model_len"},
		{"seqs capped at the stated token budget", with(func(i *BatchInput) { i.DeviceMemoryBytes, i.MaxNumBatchedTokens = 141*g, 512 }),
			512, 512, "capped at max_num_batched_tokens", SourceStated},
		{"a stated max_num_seqs above the budget is kept", with(func(i *BatchInput) {
			i.DeviceMemoryBytes, i.MaxNumSeqs, i.MaxNumBatchedTokens = 141*g, 2048, 512
		}), 2048, 512, SourceStated, SourceStated},
		{"without chunked prefill the budget rises to max_model_len", with(func(i *BatchInput) {
			i.DeviceMemoryBytes, i.ChunkedPrefill, i.MaxModelLen = 141*g, false, 32768
		}), 1024, 32768, "", "raised to max_model_len"},
	} {
		t.Run(c.name, func(t *testing.T) {
			r := ResolveBatch(c.in)
			if r.MaxNumSeqs != c.seqs || r.MaxNumBatchedTokens != c.tokens {
				t.Errorf("max_num_seqs %d, max_num_batched_tokens %d; want %d and %d", r.MaxNumSeqs, r.MaxNumBatchedTokens, c.seqs, c.tokens)
			}
			if !strings.Contains(r.MaxNumSeqsSource, c.seqsFrom) || !strings.Contains(r.MaxNumBatchedTokensSource, c.tokensFrom) {
				t.Errorf("sources %q and %q; want them to name %q and %q", r.MaxNumSeqsSource, r.MaxNumBatchedTokensSource, c.seqsFrom, c.tokensFrom)
			}
			if (c.in.MaxNumSeqs > 0) != (r.MaxNumSeqsSource == SourceStated) ||
				(c.in.MaxNumBatchedTokens > 0) != (r.MaxNumBatchedTokensSource == SourceStated) {
				t.Errorf("sources %q and %q: a value is %q exactly when it was stated", r.MaxNumSeqsSource, r.MaxNumBatchedTokensSource, SourceStated)
			}
		})
	}
}

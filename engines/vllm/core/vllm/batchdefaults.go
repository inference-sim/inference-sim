// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/engine/arg_utils.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// The translation covers get_batch_defaults and _set_default_max_num_seqs_and_batched_tokens_args.

package vllm

import "strings"

// BatchInput is what vLLM v0.32.0 resolves an unstated max_num_seqs and max_num_batched_tokens
// from (EngineArgs._set_default_max_num_seqs_and_batched_tokens_args,
// vllm/engine/arg_utils.py:3097-3186@5df63a72, with get_batch_defaults, :2893-2985). Zero in
// MaxNumSeqs or MaxNumBatchedTokens is unstated.
type BatchInput struct {
	MaxNumSeqs          int
	MaxNumBatchedTokens int

	// DeviceMemoryBytes and DeviceName are the GPU's total memory and name, which vLLM reads
	// from the platform (current_platform.get_device_total_memory and get_device_name,
	// :2912-2914).
	DeviceMemoryBytes int64
	DeviceName        string
	// BatchedDPMoE is ParallelConfig.use_batched_dp_moe: all2all_backend deepep_low_latency or
	// nixl_ep, with expert parallelism, at data_parallel_size above 1
	// (vllm/config/parallel.py:792-801@5df63a72).
	BatchedDPMoE bool
	// ThroughputMode is performance_mode "throughput" (vllm/config/vllm.py:433@5df63a72,
	// default "balanced"). The deployment schema has no field for it, so a deployment always
	// resolves as balanced.
	ThroughputMode bool
	// ChunkedPrefill is enable_chunked_prefill as resolved (ResolveChunkedPrefill): the engine
	// block's stated value, else true. Off, an unstated token budget is raised to max_model_len.
	ChunkedPrefill bool
	// MaxModelLen is the resolved context bound, which caps an unstated token budget.
	MaxModelLen int
}

// BatchResolution is the two settings as vLLM v0.32.0 resolves them, and for each, where its
// value came from: "stated", or the rule that set it.
type BatchResolution struct {
	MaxNumSeqs, MaxNumBatchedTokens             int
	MaxNumSeqsSource, MaxNumBatchedTokensSource string
}

// The settings' sources when the deployment does not state them.
const (
	SourceStated = "stated"
	// sourceBatchDefaults names the rule an unstated value came from, for the source report.
	sourceBatchDefaults = "vLLM v0.32.0 OpenAI-API-server default for the GPU " +
		"(EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:2893-2985@5df63a72; " +
		"_set_default_max_num_seqs_and_batched_tokens_args, :3097-3186)"
	sourceBatchedDP = "vLLM v0.32.0 default under a batched data-parallel MoE " +
		"(SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP, " +
		"vllm/config/scheduler.py:42@5df63a72; vllm/engine/arg_utils.py:3112-3116)"
)

const gib = int64(1) << 30 // GiB_bytes (vllm/utils/mem_constants.py:18@5df63a72)

// ResolveBatch resolves max_num_seqs and max_num_batched_tokens as vLLM v0.32.0's vllm serve
// does, in the OpenAI-API-server usage context (vllm/entrypoints/cli/serve.py:191@5df63a72).
// A stated value is kept, and checked by the engine as it is today (config.go resolve).
func ResolveBatch(in BatchInput) BatchResolution {
	// get_batch_defaults, OPENAI_API_SERVER column (vllm/engine/arg_utils.py:2923-2952):
	// at least 160 GiB, 16384 tokens and 1024 sequences (:2923-2932); at least 70 GiB and not
	// an A100 by name, 8192 and 1024 (:2933-2942); otherwise 2048 and 256 (:2943-2952). The
	// A100 exclusion is by device name (:2920-2922, 2933). The TPU and CPU branches
	// (:2954-2983) are not this engine's platform.
	name := strings.ToLower(in.DeviceName)
	tokens, seqs := 2048, 256
	switch {
	case in.DeviceMemoryBytes >= 160*gib:
		tokens, seqs = 16384, 1024
	case in.DeviceMemoryBytes >= 70*gib && !strings.Contains(name, "a100"):
		tokens, seqs = 8192, 1024
	}
	r := BatchResolution{MaxNumSeqs: in.MaxNumSeqs, MaxNumBatchedTokens: in.MaxNumBatchedTokens,
		MaxNumSeqsSource: SourceStated, MaxNumBatchedTokensSource: SourceStated}
	if in.MaxNumBatchedTokens <= 0 {
		r.MaxNumBatchedTokensSource = sourceBatchDefaults
		r.MaxNumBatchedTokens = tokens
		if in.BatchedDPMoE {
			// use_batched_dp_moe takes 256 before the usage context's default (:3112-3116).
			r.MaxNumBatchedTokens = 256
			r.MaxNumBatchedTokensSource = sourceBatchedDP
		}
	}
	if in.MaxNumSeqs <= 0 {
		r.MaxNumSeqs = seqs
		r.MaxNumSeqsSource = sourceBatchDefaults
	}
	// Throughput mode doubles each value it defaulted (:3129-3134).
	if in.ThroughputMode {
		if in.MaxNumBatchedTokens <= 0 {
			r.MaxNumBatchedTokens *= 2
			r.MaxNumBatchedTokensSource += "; doubled by performance_mode throughput (:3129-3134)"
		}
		if in.MaxNumSeqs <= 0 {
			r.MaxNumSeqs *= 2
			r.MaxNumSeqsSource += "; doubled by performance_mode throughput (:3129-3134)"
		}
	}
	if in.MaxNumBatchedTokens <= 0 {
		if !in.ChunkedPrefill && in.MaxModelLen > r.MaxNumBatchedTokens {
			// Without chunked prefill a prompt must fit in one step: max(max_model_len, budget)
			// (:3136-3146).
			r.MaxNumBatchedTokens = in.MaxModelLen
			r.MaxNumBatchedTokensSource += "; raised to max_model_len without chunked prefill (:3140-3146)"
		}
		// The multimodal prefix-LM floor (:3147-3161) applies to multimodal models only, which
		// this engine does not run.
		// Capped at max_num_seqs x max_model_len (:3164-3170).
		if in.MaxModelLen > 0 && r.MaxNumSeqs*in.MaxModelLen < r.MaxNumBatchedTokens {
			r.MaxNumBatchedTokens = r.MaxNumSeqs * in.MaxModelLen
			r.MaxNumBatchedTokensSource += "; capped at max_num_seqs x max_model_len (:3164-3170)"
		}
	}
	if in.MaxNumSeqs <= 0 && r.MaxNumBatchedTokens < r.MaxNumSeqs {
		// An unstated max_num_seqs is capped at the token budget (:3178-3180).
		r.MaxNumSeqs = r.MaxNumBatchedTokens
		r.MaxNumSeqsSource += "; capped at max_num_batched_tokens (:3178-3180)"
	}
	return r
}

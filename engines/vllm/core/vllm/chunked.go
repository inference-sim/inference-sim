// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/config/scheduler.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/model_executor/models/config.py (the Mamba hook's chunked-prefill assertion) and
// vllm/v1/core/sched/scheduler.py (the waiting-queue check that chunked prefill off changes).

package vllm

import (
	"fmt"

	"github.com/inference-sim/blis-vllm/core/kv"
)

// chunkedInput is what checkChunkedPrefill reads besides the engine block: the values the rest of
// resolve has already settled.
type chunkedInput struct {
	prefixCaching bool
	maxModelLen   int
	budget        int
	groups        []kv.Group
	method        string // the drafter's method, "" for none
	pp            int
}

// chunkedOffKnown names what the committed scenarios run with chunked prefill off, so each
// refusal below can say what is covered.
const chunkedOffKnown = "oracle/scenarios/chunked-prefill-off*.json and oracle/scenarios-t3/t3-chunked-prefill-off*.json " +
	"run vLLM's Scheduler with it off on one full-attention KV group, with and without prefix caching, " +
	"preemption, KV events, a long-prefill cap, priority scheduling and async scheduling at one pipeline stage"

// ResolveChunkedPrefill is the enable_chunked_prefill that runs: the stated value, else True, the
// default of a causal decoder (SchedulerConfig, vllm/config/scheduler.py:134@5df63a72;
// EngineArgs sets it to ModelConfig.is_chunked_prefill_supported, true for a generative decoder,
// vllm/engine/arg_utils.py:2987-2994, vllm/config/model.py:2174-2220@5df63a72).
//
// The other places that change it do not apply to a causal decoder with a KV cache, which is
// all this engine runs: an encoder-decoder model turns it off (scheduler.py:290-296), a model
// with no KV cache groups turns it off (vllm/v1/engine/core.py:164-168), a model with non-causal
// attention layers turns it off (core.py:276-285), the CPU platform on RISC-V turns it off
// (arg_utils.py:3040-3049; vllm/platforms/cpu.py:525), and a multimodal budget shrinks when it is
// off (vllm/multimodal/encoder_budget.py:169). MiniMax-M3 and DeepSeek-V4 are none of these.
// vLLM only warns when a generative model's chunked prefill is disabled by hand
// (arg_utils.py:3000-3009), and when a Turing GPU runs float32 with it on (vllm/config/vllm.py:1681);
// TurboQuant reserves a workspace for it (vllm/v1/attention/backends/turboquant_attn.py:250), which
// is memory, the cost model's.
func ResolveChunkedPrefill(stated *bool) bool {
	return stated == nil || *stated
}

// checkChunkedPrefill refuses enable_chunked_prefill false where vLLM v0.32.0 refuses it, and
// where this engine has no oracle evidence that its scheduler matches vLLM's. The scheduler's
// code for it is one line in the waiting loop (core/sched/scheduler.go, vllm/v1/core/sched/
// scheduler.py:1161@5df63a72); a configuration is allowed only if a committed scenario runs it
// with the setting off (chunkedOffKnown). On is vLLM's default and is checked by every other
// scenario, so it passes here.
func checkChunkedPrefill(c Config, chunked bool, in chunkedInput) error {
	if chunked {
		return nil
	}
	e := c.Engine
	// vLLM's own refusals first, with its messages.
	if in.budget < in.maxModelLen {
		return fmt.Errorf("vllm: enable_chunked_prefill false: max_num_batched_tokens (%d) is smaller than max_model_len (%d). "+
			"This effectively limits the maximum sequence length to max_num_batched_tokens and makes vLLM reject longer "+
			"sequences. Please increase max_num_batched_tokens or decrease max_model_len "+
			"(vllm/config/scheduler.py:314-327@5df63a72)", in.budget, in.maxModelLen)
	}
	if c.MambaModel && in.prefixCaching {
		// The Mamba hook sets mamba_cache_mode "align" under prefix caching and asserts on it
		// (effectiveMambaCacheMode).
		return fmt.Errorf("vllm: enable_chunked_prefill false: Chunked prefill is required for mamba cache mode 'align', " +
			"which a Mamba model runs with enable_prefix_caching (vllm/model_executor/models/config.py:694-696@5df63a72)")
	}
	// What this engine does not model with it off.
	var with string
	switch {
	case in.method != "":
		with = "speculative decoding"
	case e.MaxLoRAs > 0:
		with = "max_loras"
	case c.KVTransfer != nil:
		with = "a KV transfer connector"
	case c.KVOffload != nil:
		with = "a KV offload tier"
	case c.DataParallelSize > 1:
		with = "data parallelism"
	case in.pp > 1:
		with = "pipeline parallelism"
	case len(in.groups) > 1:
		with = "hybrid KV cache groups"
	case e.LongPrefillTokenThresholdAdaptive != nil && *e.LongPrefillTokenThresholdAdaptive:
		with = "long_prefill_token_threshold_adaptive"
	case e.MaxNumScheduledTokens != 0:
		with = "max_num_scheduled_tokens"
	case e.Watermark != nil && *e.Watermark != 0:
		with = "watermark"
	case e.MaxNumActiveSeqs != 0:
		with = "max_num_active_seqs"
	}
	if with != "" {
		return fmt.Errorf("vllm: enable_chunked_prefill false with %s is not modelled: no oracle scenario runs vLLM's "+
			"scheduler with both (%s)", with, chunkedOffKnown)
	}
	return nil
}

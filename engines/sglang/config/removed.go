// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/server_args.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Derived from python/sglang/srt/arg_groups/validation_hook.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Derived from python/sglang/srt/arg_groups/choices.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).

package config

import (
	"fmt"
	"sort"
	"strings"
)

// removedFlags is REMOVED_FLAGS (server_args.py:227-241@c4af57ff): the command-line flags SGLang 1.0
// removed, each with what to pass instead, in SGLang's order.
var removedFlags = []struct{ flag, use string }{
	{"--disable-cuda-graph", "use --disable-decode-cuda-graph --disable-prefill-cuda-graph"},
	{"--enable-dp-attention", "use --attn-dp-size <N> instead of --dp-size <N> --enable-dp-attention"},
	{"--monokernel-backend", "use --enable-amd-monokernel instead of --monokernel-backend amd, and --enable-nvidia-megakernel instead of --monokernel-backend msa"},
	{"--enable-flashinfer-allreduce-fusion", "use --flashinfer-allreduce-fusion-backend auto"},
	{"--disable-fast-image-processor", "use --image-processor-backend pil"},
	{"--disable-outlines-disk-cache", "it had no effect, drop it"},
	{"--show-time-cost", "it had no effect, drop it"},
	{"--triton-attention-reduce-in-fp32", "it had no effect, drop it"},
}

// checkRemovedKwargs is reject_removed_kwargs (server_args.py:244-250@c4af57ff), which Engine(**kwargs)
// runs before it builds ServerArgs (entrypoints/engine.py:315@c4af57ff): a keyword for a removed flag
// that is no longer a field stops with a TypeError that names what to use. SGLang walks the keywords
// in the caller's order; a Go map has none, so the names are walked sorted.
func checkRemovedKwargs(args map[string]any) error {
	names := make([]string, 0, len(args))
	for n := range args {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, key := range names {
		flag := "--" + strings.ReplaceAll(key, "_", "-")
		for _, rf := range removedFlags {
			if rf.flag == flag && !IsField(key) {
				return &UpstreamRefusal{Site: "sglang/srt/server_args.py:reject_removed_kwargs",
					Message: fmt.Sprintf("%s was removed in SGLang 1.0; %s.", key, rf.use)}
			}
		}
	}
	return nil
}

var removedAttentionBackends = map[string]string{
	"compressed": "dsv4",
	"nsa":        "dsa",
	"tokenspeed_mla": "cutedsl_mla with --kv-cache-dtype fp8_e4m3 " +
		"(trtllm_mla for prefill), and --speculative-attention-mode decode " +
		"with speculative decoding",
}

var removedMoeA2ABackends = map[string]string{"flashinfer": "flashinfer_nvlink_one_sided"}

// removedChoices is REMOVED_CHOICES (choices.py:130-138@c4af57ff), in SGLang's order.
var removedChoices = []struct {
	field string
	by    map[string]string
}{
	{"attention_backend", removedAttentionBackends},
	{"decode_attention_backend", removedAttentionBackends},
	{"prefill_attention_backend", removedAttentionBackends},
	{"speculative_draft_attention_backend", removedAttentionBackends},
	{"moe_a2a_backend", removedMoeA2ABackends},
	{"speculative_moe_a2a_backend", removedMoeA2ABackends},
	{"tool_call_parser", map[string]string{"qwen25": "qwen", "glm45": "glm"}},
}

// checkRemovedOptions is validate_removed_options (validation_hook.py:991-1016@c4af57ff), the first step
// of SGLang's pipeline: a removed choice value stops resolution and names its replacement. It reads the
// input as given, since a removed value is no longer one of the field's choices. The step's other
// two duties do not reach Resolve: num_continuous_decode_steps is refused as a field not modelled,
// and Resolve takes no environment.
func checkRemovedOptions(args map[string]any) error {
	for _, rc := range removedChoices {
		v, ok := args[rc.field].(string)
		if !ok {
			continue
		}
		if use, removed := rc.by[v]; removed {
			return &UpstreamRefusal{Site: "sglang/srt/arg_groups/validation_hook.py:validate_removed_options",
				Message: fmt.Sprintf("--%s: %s was removed in SGLang 1.0; use %s.", strings.ReplaceAll(rc.field, "_", "-"), v, use)}
		}
	}
	return nil
}

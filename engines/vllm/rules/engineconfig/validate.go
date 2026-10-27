// Moved from inference-sim/blis-schemas/spec/deployment/validate.go at 70246460a379f8b1898fbf6c36a3091364118950: the field
// checks of vLLM's engine block (validateEngine, validateSchedulerKnobs), which blis-schemas
// ran on its Engine struct before it held the block unopened. Messages and paths are
// unchanged, so a refusal reads as it did.

package engineconfig

import (
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/blis-schemas/rules"
)

// Validate is the release-independent field checks of one pool's engine block: the bounds
// the block's own types declare, as blis-schemas ran them before it held the block unopened.
// at is the pool's path, such as "pools[0]"; each finding is at at + ".engine.<field>". The
// version-scoped checks are the rules packs'.
func Validate(p *rules.Problems, at string, e Engine) {
	validateEngine(p, at, e)
}

func validateEngine(p *rules.Problems, at string, e Engine) {
	if !e.Frontend.Valid() {
		p.Field(at+".engine.frontend", "%q is not one of %v", e.Frontend,
			[]Frontend{FrontendPython, FrontendRust})
	}
	for field, v := range map[string]int{
		"block_size": e.BlockSize, "max_num_batched_tokens": e.MaxNumBatchedTokens,
		"max_num_seqs":                 e.MaxNumSeqs,
		"long_prefill_token_threshold": e.LongPrefillTokenThreshold,
		"cp_kv_cache_interleave_size":  e.CPKVCacheInterleaveSize,
	} {
		if v < 0 {
			p.Field(at+".engine."+field, "must not be negative")
		}
	}
	// -1 is auto, the one negative the engine gives a meaning; anything below it is a
	// mistake.
	if e.MaxModelLen < MaxModelLenAuto {
		p.Field(at+".engine.max_model_len", "must be a positive count, or auto (-1), got %d", e.MaxModelLen)
	}
	// Stating both the boolean and the resolved name invites them to disagree, and
	// which one an implementation honours would be arbitrary.
	if e.DisableCustomAllReduce != nil && e.AllReduceBackend != "" {
		disabled := *e.DisableCustomAllReduce
		if disabled && e.AllReduceBackend == "custom" {
			p.Field(at+".engine.allreduce_backend",
				"requests the custom kernel while disable_custom_all_reduce is true")
		}
		if !disabled && e.AllReduceBackend == "nccl" {
			p.Warnf("%s.engine: allreduce_backend names nccl while disable_custom_all_reduce is false; the engine would use the custom kernel where reachable", at)
		}
	}
	// gpu_memory_utilization is a non-pointer float, so an omitted one is zero. Zero is
	// the unset sentinel — the engine picks its default — and a stated fraction lies in
	// (0, 1], so the accepted range is [0, 1]. The message states that range rather than
	// (0, 1] so it does not read as rejecting the zero the check deliberately allows. The
	// finite check runs first: a NaN would pass the range comparison (NaN < 0 and NaN > 1
	// are both false) and reach the engine as a garbage fraction.
	if p.FiniteField(at+".engine.gpu_memory_utilization", e.GPUMemoryUtilization) &&
		(e.GPUMemoryUtilization < 0 || e.GPUMemoryUtilization > 1) {
		p.Field(at+".engine.gpu_memory_utilization",
			"must lie in [0, 1] (0 means unset), got %v", e.GPUMemoryUtilization)
	}
	// The engine asserts the capture maximum is at least 1 once it is capped at
	// max_num_batched_tokens, and that a stated size list is non-empty (v0.31.0
	// vllm/config/vllm.py _set_cudagraph_sizes). A size below 1 captures no step.
	if m := e.MaxCUDAGraphCaptureSize; m != nil && *m < 1 {
		p.Field(at+".engine.max_cudagraph_capture_size", "must be at least 1, got %d; omit it for the release default", *m)
	}
	if e.CUDAGraphCaptureSizes != nil && len(e.CUDAGraphCaptureSizes) == 0 {
		p.Field(at+".engine.cudagraph_capture_sizes", "must list at least one size; omit it to generate them")
	}
	for i, s := range e.CUDAGraphCaptureSizes {
		if s < 1 {
			p.Field(fmt.Sprintf("%s.engine.cudagraph_capture_sizes[%d]", at, i), "must be at least 1, got %d", s)
		}
	}
	// The engine types both thresholds Field(ge=0) whether or not overlap is enabled.
	if e.DBO != nil {
		for field, v := range map[string]*int{
			"dbo_decode_token_threshold":  e.DBO.DecodeTokenThreshold,
			"dbo_prefill_token_threshold": e.DBO.PrefillTokenThreshold,
		} {
			if v != nil && *v < 0 {
				p.Field(at+".engine.dbo."+field, "must not be negative")
			}
		}
	}
	if e.EPLB != nil && e.EPLB.Enabled && e.EPLB.NumRedundantExperts < 0 {
		p.Field(at+".engine.eplb.num_redundant_experts", "must not be negative")
	}
	if e.Speculative != nil {
		if e.Speculative.Method == "" {
			p.Field(at+".engine.speculative.method", "required")
		}
		if e.Speculative.NumSpecTokens < 1 {
			p.Field(at+".engine.speculative.num_spec_tokens",
				"must be at least 1; omit the block instead of declaring zero drafts")
		}
		sp := e.Speculative
		if sp.DraftModel != "" && strings.TrimSpace(sp.DraftModel) == "" {
			p.Field(at+".engine.speculative.draft_model", "must name a catalog model, not be blank")
		}
		if m := sp.DraftSampleMethod; m != "" && !draftSampleMethods[m] {
			p.Field(at+".engine.speculative.draft_sample_method", "%q is not greedy or probabilistic", m)
		}
		if m := sp.RejectionSampleMethod; m != "" && !rejectionSampleMethods[m] {
			p.Field(at+".engine.speculative.rejection_sample_method", "%q is not standard, synthetic or block", m)
		}
	}
	validateSchedulerKnobs(p, at, e)
}

// validateSchedulerKnobs checks the scheduler, cache, LoRA and pass-through knobs against the
// bounds vLLM v0.32.0's own config types declare, so a value the engine would refuse at
// launch is refused here. A zero integer means unset and takes the engine's default.
func validateSchedulerKnobs(p *rules.Problems, at string, e Engine) {
	for field, v := range map[string]int{
		// Field(ge=1) in vllm/config/scheduler.py:55, 69, 203, 213 and vllm/config/lora.py:35;
		// gt=0 in vllm/config/cache.py:90. Zero is unset here.
		"max_num_scheduled_tokens":  e.MaxNumScheduledTokens,
		"max_num_active_seqs":       e.MaxNumActiveSeqs,
		"prefill_schedule_interval": e.PrefillScheduleInterval,
		"stream_interval":           e.StreamInterval,
		"prefix_match_unit":         e.PrefixMatchUnit,
		"max_loras":                 e.MaxLoRAs,
		"max_cpu_loras":             e.MaxCPULoRAs,
	} {
		if v < 0 {
			p.Field(at+".engine."+field, "must not be negative; omit it, or 0, for the engine default")
		}
	}
	// vllm/config/scheduler.py:336-343@5df63a72 refuses max_num_active_seqs above max_num_seqs.
	if e.MaxNumActiveSeqs > 0 && e.MaxNumSeqs > 0 && e.MaxNumActiveSeqs > e.MaxNumSeqs {
		p.Field(at+".engine.max_num_active_seqs", "%d exceeds max_num_seqs %d, which the engine refuses",
			e.MaxNumActiveSeqs, e.MaxNumSeqs)
	}
	// vllm/config/lora.py:120-126@5df63a72 refuses max_cpu_loras below max_loras; unset
	// max_loras is the default, 1.
	if e.MaxCPULoRAs > 0 && e.MaxCPULoRAs < max(e.MaxLoRAs, 1) {
		p.Field(at+".engine.max_cpu_loras", "%d is below max_loras %d, which the engine refuses",
			e.MaxCPULoRAs, max(e.MaxLoRAs, 1))
	}
	// Field(ge=0.0, lt=1.0), vllm/config/scheduler.py:196@5df63a72. FiniteField first, as for
	// gpu_memory_utilization: a NaN passes both comparisons.
	if w := e.Watermark; w != nil && p.FiniteField(at+".engine.watermark", *w) && (*w < 0 || *w >= 1) {
		p.Field(at+".engine.watermark", "must lie in [0, 1), got %v", *w)
	}
	if b := e.KVCacheMemoryBytes; b != nil && *b < 1 {
		p.Field(at+".engine.kv_cache_memory_bytes", "must be positive, got %d; omit it to size the cache from gpu_memory_utilization", *b)
	}
	// Field(ge=0), vllm/config/cache.py:159@5df63a72.
	if r := e.PrefixCacheRetentionInterval; r != nil && *r < 0 {
		p.Field(at+".engine.prefix_cache_retention_interval", "must not be negative, got %d", *r)
	}
	if k := e.KVEvents; k != nil && k.Publisher != "" && !kvEventPublishers[k.Publisher] {
		p.Field(at+".engine.kv_events.publisher", "%q is not a publisher vLLM accepts (null, zmq)", k.Publisher)
	}
	for i, op := range e.CustomOps {
		if strings.TrimSpace(op) == "" {
			p.Field(fmt.Sprintf("%s.engine.custom_ops[%d]", at, i), "must not be empty")
		}
	}
	for i, s := range e.DecoderReplayCUDAGraphCaptureSizes {
		if s < 1 {
			p.Field(fmt.Sprintf("%s.engine.decoder_replay_cudagraph_capture_sizes[%d]", at, i), "must be at least 1, got %d", s)
		}
	}
	for _, name := range sortedEnvNames(e.Env) {
		if !envName(name) {
			p.Field(at+".engine.env", "%q is not an environment-variable name ([A-Za-z_][A-Za-z0-9_]*)", name)
		}
	}
}

func sortedEnvNames(env map[string]string) []string {
	out := make([]string, 0, len(env))
	for k := range env {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// envName reports whether s is a POSIX environment-variable name.
func envName(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '_' || c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || i > 0 && c >= '0' && c <= '9' {
			continue
		}
		return false
	}
	return true
}

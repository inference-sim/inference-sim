// vllm_defaults.go reproduces vLLM's own resolution of the two batch-size settings a
// deployment most often leaves unstated.
//
// A benchmark that passes neither --max-num-seqs nor --max-num-batched-tokens does not get
// the values in vLLM's SchedulerConfig struct. Those are documented in the source as
// "mainly for convenience when testing"; before any engine runs, EngineArgs overrides them
// from the device's total memory. Simulating with the struct defaults, or with a number
// chosen here, describes an engine nobody ran.
//
// The resolution matters most for time to first token. The sequence cap decides whether a
// request waits at all, so a cap set too low makes the simulator queue requests the real
// engine admitted immediately -- an error no step-time calibration can absorb, and one that
// a shape comparison across concurrency does not cancel.
//
// Source: vllm/engine/arg_utils.py, EngineArgs.get_batch_defaults and
// _set_default_max_num_seqs_and_batched_tokens_args, read at vLLM 0.22. Three properties of
// that code are reproduced deliberately:
//
//   - world_size is a parameter of get_batch_defaults and is never used in its body, so
//     tensor and pipeline width do NOT change the defaults.
//   - the usage context selects between an offline LLM-class value and a served one. A
//     benchmark driving an OpenAI-compatible endpoint is the served case, which is the only
//     one this reproduces; the offline values differ on Hopper (16384 against 8192).
//   - the A100 exclusion is by device NAME, not by memory. An 80 GiB A100 clears the 70 GiB
//     threshold and is still sent to the small defaults, because large batched-token counts
//     measured worse on it.
package latency

import "strings"

// Thresholds and values from vllm/engine/arg_utils.py. Named rather than inlined so a
// reader can check them against that file without re-deriving the arithmetic.
const (
	vllmLargeMemoryGiB  = 160.0 // B200/B300 class
	vllmMediumMemoryGiB = 70.0  // H100/H200 class, A100 excluded by name
)

// VLLMBatchDefaults is what an engine resolves for a served deployment that states neither
// setting.
type VLLMBatchDefaults struct {
	MaxNumSeqs          int
	MaxNumBatchedTokens int
}

// ResolveVLLMBatchDefaults returns the values vLLM would resolve for a device of this total
// memory and name, for a served (OpenAI-API) deployment.
//
// deviceName is matched case-insensitively and may be empty; an empty name cannot be an
// A100, so it takes the memory-based branch. This mirrors vLLM, which falls back to the
// small defaults only when it cannot query the device at all -- a case that cannot arise
// here, because a scenario always names its hardware.
func ResolveVLLMBatchDefaults(memoryGiB float64, deviceName string) VLLMBatchDefaults {
	isA100 := strings.Contains(strings.ToLower(deviceName), "a100")
	switch {
	case memoryGiB >= vllmLargeMemoryGiB:
		return VLLMBatchDefaults{MaxNumSeqs: 1024, MaxNumBatchedTokens: 16384}
	case memoryGiB >= vllmMediumMemoryGiB && !isA100:
		// Hopper. The served context takes 8192; the offline LLM class would take 16384.
		return VLLMBatchDefaults{MaxNumSeqs: 1024, MaxNumBatchedTokens: 8192}
	default:
		return VLLMBatchDefaults{MaxNumSeqs: 256, MaxNumBatchedTokens: 2048}
	}
}

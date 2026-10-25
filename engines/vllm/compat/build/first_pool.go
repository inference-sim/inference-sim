// Moved from inference-sim/cmd/root.go at b631140d48ba37fedf834874d7c083bab3e8b116.

package build

import (
	"fmt"

	"github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// FirstPoolSettings is the run's first (colocated) pool's engine, as the simulator sizes every
// instance it does not override per pool: the caps, pages and pool the kernel priced, read from
// the pool's own kernel. The values and the rules that choose them are the engine's (#2021 B2,
// verification pass 4, gap 1); the host copies them into its run configuration.
type FirstPoolSettings struct {
	TotalKVBlocks         int64
	SequenceState         engine.KVSequenceState
	BlockSize             int64
	MaxNumSeqs            int64
	MaxNumBatchedTokens   int64
	MaxModelLen           int64
	MaxModelLenFrom       string
	PrefixCachingDisabled bool
	SpeculativeTokens     int
	SpeculativeMethod     string
	// Notes are the startup notes the pool's settings produce, in order, for the caller to log:
	// its sequence-state charge (SequenceFixedCharge).
	Notes []string
}

// FirstPoolSettingsOf is the first pool's engine from its resolved settings s (Adopted.Settings).
// It refuses an engine with no window at all.
func FirstPoolSettingsOf(scenario string, s kernelmodel.Settings) (FirstPoolSettings, error) {
	const what = "first pool"
	if err := RequireWindowFits(scenario, what, s); err != nil {
		return FirstPoolSettings{}, err
	}
	fp := FirstPoolSettings{
		TotalKVBlocks: s.KVBlocks, SequenceState: s.SequenceState, BlockSize: int64(s.BlockSize),
		MaxNumSeqs: int64(s.MaxNumSeqs), MaxNumBatchedTokens: int64(s.MaxNumBatchedTokens),
		MaxModelLen: int64(s.MaxModelLen), MaxModelLenFrom: s.MaxModelLenFrom,
		PrefixCachingDisabled: s.PrefixCachingDisabled,
		SpeculativeTokens:     s.SpeculativeTokens, SpeculativeMethod: s.SpeculativeMethod,
	}
	if note := SequenceFixedCharge(what, s); note != "" {
		fp.Notes = append(fp.Notes, note)
	}
	return fp, nil
}

// KVBlocksReserving is the first pool's KV blocks per rank with reservedBytes of each rank's HBM
// set aside (the LoRA adapter reservation, which vLLM takes beside the weights before it profiles
// the KV pool). It refuses a pool whose budget cannot be sized, or whose window no longer fits.
func KVBlocksReserving(scenario string, m *kernelmodel.Model, reservedBytes int64) (int64, error) {
	s, err := m.SettingsReserving(reservedBytes)
	if err != nil {
		return 0, fmt.Errorf("scenario %q with a %d-byte LoRA adapter reservation: %v", scenario, reservedBytes, err)
	}
	if err := RequireWindowFits(scenario, "pool with the LoRA adapter reservation set aside", s); err != nil {
		return 0, err
	}
	return s.KVBlocks, nil
}

// PrefixHitOf is how a pool's KV cache groups cut a prefix hit, with the run's SWA bounded-replay
// override (nil when unset), and the warning the caller logs when the model's layout is not one
// the engine maps ("" otherwise).
func PrefixHitOf(what string, m *kernelmodel.Model, boundedReplay *bool) (kernelmodel.PrefixHitResolution, string) {
	r := m.PrefixHit(boundedReplay)
	return r, prefixHitWarning(what, r.Note)
}

// prefixHitWarning is the startup warning for a pool whose layout the engine does not map: the
// hit falls back to the full-attention prefix, and note says why. "" when there is no note.
func prefixHitWarning(what, note string) string {
	if note == "" {
		return ""
	}
	return fmt.Sprintf("prefix cache (%s): the hit is the full-attention prefix alone: %s", what, note)
}

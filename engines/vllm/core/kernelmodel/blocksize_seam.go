// Original to blis-vllm.

package kernelmodel

import "github.com/inference-sim/inference-sim/engines/vllm/core/vllm"

// ReadBlockSizeTableAsRecorded makes an unstated block_size resolve from the table the oracle recorded
// at vLLM v0.33.0 (core/vllm.ResolveBlockSizeAsRecorded, default block size 16) instead of the table at
// the modelled commit (core/vllm.ResolveBlockSize, default 64), and returns the function that undoes it.
// It is for tests that compare an engine with figures recorded on a vLLM release that predates the
// default's change; a run never calls it.
func ReadBlockSizeTableAsRecorded() (restore func()) {
	prev := blockSizeResolver
	blockSizeResolver = vllm.ResolveBlockSizeAsRecorded
	return func() { blockSizeResolver = prev }
}

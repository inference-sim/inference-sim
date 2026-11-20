// Original to blis-vllm.

package kernelmodel

import "github.com/inference-sim/inference-sim/engines/vllm/core/vllm"

// ReadBlockSizeTableAsRecorded makes an unstated block_size resolve from the block-size oracle's table
// as recorded (core/vllm.ResolveBlockSizeFromTable), whatever commit the build models, and returns the
// function that undoes it. It is for tests of other packages that compare an engine with figures
// recorded at the oracle's commit; a run never calls it, and the engine otherwise refuses the unstated
// size while the table is from another commit (core/vllm.ResolveBlockSize, BlockSizeIssue).
func ReadBlockSizeTableAsRecorded() (restore func()) {
	prev := blockSizeResolver
	blockSizeResolver = vllm.ResolveBlockSizeFromTable
	return func() { blockSizeResolver = prev }
}

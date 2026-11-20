package harness

import (
	"testing"

	"github.com/inference-sim/inference-sim/engines/vllm/core/kernelmodel"
	"github.com/inference-sim/inference-sim/tools/harness/internal/artifacts"
)

// TestMain reads the block-size oracle's table as recorded: the build models a vLLM commit the
// oracle has not been re-run at, so the engine refuses an unstated block_size (core/vllm.BlockSizeIssue)
// and these tests, which run the shipped scenarios that leave it unstated, would test that refusal
// instead of the harness. engines/vllm/vllmengine holds the refusal.
func TestMain(m *testing.M) {
	kernelmodel.ReadBlockSizeTableAsRecorded()
	artifacts.Main(m)
}

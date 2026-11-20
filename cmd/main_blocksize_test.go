package cmd

import (
	"os"
	"testing"

	"github.com/inference-sim/inference-sim/engines/vllm/core/kernelmodel"
)

// TestMain reads the block-size oracle's table as recorded. The build models a vLLM commit the
// oracle has not been re-run at, so the engine refuses an unstated block_size (core/vllm.BlockSizeIssue,
// engines/vllm/vllmengine's TestAnUnstatedBlockSizeIsRefusedWhileTheOracleTableIsFromAnotherCommit
// holds that refusal); the tests here exercise the CLI on the shipped scenarios, which leave it
// unstated, and not the block size, until a maintainer re-records the oracle at the pin.
func TestMain(m *testing.M) {
	kernelmodel.ReadBlockSizeTableAsRecorded()
	os.Exit(m.Run())
}

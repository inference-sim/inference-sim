// Moved from blis-latency-kernel/deepep_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	"strings"
	"testing"
)

// vLLM sizes DeepEP's low-latency RDMA buffer by max_num_batched_tokens and nothing else
// (FusedMoEConfig.max_num_tokens, fused_moe/layer.py:272,359; all2all_utils.py:244-246), so
// a DEP8 DeepSeek-V4.1 engine at 8,192 tokens asks for ~122 GiB per rank and fails at weight
// loading. BLIS refuses it, names the cause, and accepts the same engine at 256.
func TestALowLatencyBufferThatCannotBeAllocatedIsRefused(t *testing.T) {
	in := fixtureInputs(t, v41)
	pl := &in.Deployment.Pools[0].Parallel
	pl.TP, pl.DP, pl.EnableExpertParallel = 1, 8, true
	e := mustDecode(t, in.Deployment.Pools[0])
	e.All2AllBackend, e.MaxNumBatchedTokens = "deepep_low_latency", 8192
	setEngine(t, &in.Deployment.Pools[0], e)
	_, err := newKernel(in)
	if err == nil || !strings.Contains(err.Error(), "DeepEP buffers") {
		t.Fatalf("LL at max_num_batched_tokens 8192 built (err %v); vLLM fails allocating "+
			"its buffer", err)
	}
	e.MaxNumBatchedTokens = 256
	setEngine(t, &in.Deployment.Pools[0], e)
	build(t, in)
}

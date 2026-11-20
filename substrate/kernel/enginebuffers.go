package latencykernel

import "github.com/inference-sim/blis-substrate/contracts/engine/plan"

// statedBufferBytes is the per-rank GPU memory of the buffers the engine states in its plan
// (plan.PoolPlan.Buffers, SUB-14). The engine sizes each from its own function and states it
// per rank, so the kernel adds the figures and resolves nothing (SUB-6). An example is vLLM's
// draft-logits buffer under probabilistic draft sampling, max_num_seqs x K x vocab x the head's
// dtype bytes (vllm/v1/worker/gpu/spec_decode/speculator.py:177-189@v0.31.0): the engine
// allocates it at construction, before memory profiling, so it is charged inside the KV budget.
// Host buffers and buffers not stated per rank are not GPU memory of a rank and are not
// charged here (plan.MemoryOf refuses a GPU buffer on another basis).
func (k *Kernel) statedBufferBytes() int64 {
	var total int64
	for _, b := range k.low.Buffers {
		if b.Host || b.Basis != plan.BasisRank {
			continue
		}
		total += int64(b.Bytes)
	}
	return total
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/models/glm5next/common/attention.py at vLLM v0.34.0+0fda86a35419 (0fda86a35419c9c70dbfada224c838c10fbb62ad).

package kernelmodel

import (
	"fmt"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"

	"github.com/inference-sim/inference-sim/engines/vllm/core/engineconfig"
	v0_33 "github.com/inference-sim/inference-sim/engines/vllm/core/rulepack/v0_33"
)

func isGlm5Next(g *model.Graph) bool {
	a := g.Global.Architecture
	return a == v0_33.ArchGlm5NextCausal || a == v0_33.ArchGlm5NextConditional
}

// glm5NextIndexerQuantBlock is the width one fp32 scale covers in the indexer's key cache
// (Indexer.quant_block_size, attention.py:289-291).
const glm5NextIndexerQuantBlock = 128

// glm5NextLayerRuns is the KV cache specs the sparse-MLA layer whose main node is n in layer kind
// lk registers, in construction order: the indexer's key cache (Glm5NextIndexerCache,
// attention.py:93-140, built at :298-305), the kpool tail ring (Glm5NextTailCache, :142-200, built
// at :308-316) and the main latent (MLAAttentionSpec, mla_attention.py:1636-1672). Nothing for the
// indexer's scorer node, which registers its caches with its main node. The drafter's layer has the
// same caches, marked as the drafter's. Every cache states its element size, the main latent's the
// cache dtype's, so no page is taken from the memory model's remainder (blis-vllm#308).
func glm5NextLayerRuns(g *model.Graph, id string, lk model.LayerKind, n model.Node, e engineconfig.Engine, draft bool) ([]KVLayerRun, error) {
	if !n.ReadsEngineKV() {
		return nil, nil
	}
	var scorer *model.Node
	for i := range lk.Nodes {
		if c := lk.Nodes[i]; c.Op == model.OpAttention && c.AttentionKind == model.AttentionSparseMLA && !c.ReadsEngineKV() {
			scorer = &lk.Nodes[i]
		}
	}
	if scorer == nil || scorer.CompressRatio < 2 || scorer.HeadDim <= 0 {
		return nil, fmt.Errorf("layer kind %s: a sparse_mla attention node with no indexer scorer of compress_ratio above 1 beside it", id)
	}
	if n.HeadDim <= 0 {
		return nil, fmt.Errorf("layer kind %s: the sparse_mla main node states no d_h", id)
	}
	elem, err := attentionElemBytes(e.CacheDType, g)
	if err != nil {
		return nil, fmt.Errorf("layer kind %s: %w", id, err)
	}
	numSpec := 0
	if e.Speculative != nil {
		numSpec = e.Speculative.NumSpecTokens
	}
	kpool := scorer.CompressRatio
	ring := kpool * nextPowerOfTwo((kpool+numSpec+kpool-1)/kpool)
	if e.BlockSize%ring != 0 {
		return nil, fmt.Errorf("layer kind %s: block_size %d is not a multiple of the kpool tail ring %d (Glm5NextTailCache, "+
			"attention.py:174-181@0fda86a3)", id, e.BlockSize, ring)
	}
	runs := []KVLayerRun{
		{Count: 1, Spec: "mla", NumKVHeads: 1, HeadSize: scorer.HeadDim + scorer.HeadDim/glm5NextIndexerQuantBlock*4,
			BlockSize: e.BlockSize, TokensPerState: kpool, ElemBytes: 1},
		{Count: 1, Spec: "circular_buffer", NumKVHeads: 2, HeadSize: scorer.HeadDim, BlockSize: ring, ElemBytes: 2},
		{Count: 1, Spec: "mla", NumKVHeads: 1, HeadSize: n.HeadDim, BlockSize: e.BlockSize, TokensPerState: 1, ElemBytes: int(elem)},
	}
	for i := range runs {
		runs[i].Draft = draft
	}
	return runs, nil
}

func nextPowerOfTwo(n int) int {
	p := 1
	for p < n {
		p *= 2
	}
	return p
}

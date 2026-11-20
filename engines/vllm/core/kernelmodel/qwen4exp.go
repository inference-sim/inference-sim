// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/models/qwen4_exp/nvidia/qsa.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).
// Also from vllm/models/qwen4_exp/nvidia/indexer_qsa.py and vllm/models/qwen4_exp/common/qsa_cache.py at the same commit.

package kernelmodel

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-vllm/core/engineconfig"
)

// sparseGQA is model.AttentionSparseGQA (blis-substrate#253). It is spelled here until this module
// pins a substrate that carries the constant; the schema's value is the string.
const sparseGQA model.AttentionKind = "sparse_gqa"

// Qwen4Exp's full-attention layers run Qwen4ExpQSAAttention on the QWEN4_EXP_QSA_TRITON backend
// (nvidia/qsa.py:146-206). This file holds what the engine decides about them that the graph
// does not: which settings the backend refuses, and the KV cache specs the layer registers.
// The wiring that calls these from kvLayersRead, planMamba's geometry check and the rules pack
// belongs to the files that hold those decisions (blis-vllm#202).

// Qwen4ExpArchitectures are the architecture strings whose layers are QSA.
var qwen4ExpArchitectures = map[string]bool{
	"Qwen4ExpForCausalLM": true, "Qwen4ExpForConditionalGeneration": true, "Qwen4ExpMTP": true,
}

func isQwen4Exp(arch string) bool { return qwen4ExpArchitectures[arch] }

// qsaSettings are the settings the QSA backend and layer read.
type qsaSettings struct {
	Arch       string
	TP, PP     int
	DCP, PCP   int
	BlockSize  int // 0: not stated, the engine's default applies
	CacheDType string
	// KVConnector is whether the deployment sets kv_transfer_config.
	KVConnector bool
	// EnableHCSP is --enable-hc-sp.
	EnableHCSP bool
}

// qsaKVDTypes are the main-cache dtypes the layer accepts (qsa.py:361-368).
var qsaKVDTypes = map[string]bool{"": true, "auto": true, "bfloat16": true, "fp8": true, "fp8_e4m3": true}

// checkQwen4Exp refuses what vLLM v0.33.0 refuses for a Qwen4Exp deployment, naming the
// source. It returns nil for any other architecture.
func checkQwen4Exp(s qsaSettings) error {
	if !isQwen4Exp(s.Arch) {
		return nil
	}
	const issue = "inference-sim/blis-vllm#202"
	switch {
	case s.DCP > 1:
		return fmt.Errorf("%s: decode context parallelism %d: the QSA backend refuses it (Qwen4ExpQSAFlashAttentionImpl "+
			"supports_dcp False, qsa.py:212-213, :254-256; the layer, qsa.py:370-377) (%s)", s.Arch, s.DCP, issue)
	case s.PCP > 1:
		return fmt.Errorf("%s: prefill context parallelism %d: the QSA layer refuses it (qsa.py:370-377) (%s)", s.Arch, s.PCP, issue)
	case s.KVConnector:
		return fmt.Errorf("%s: a KV connector: the QWEN4_EXP_QSA_TRITON backend reports supports_kv_connector False "+
			"(qsa.py:205-206) (%s)", s.Arch, issue)
	case !qsaKVDTypes[s.CacheDType]:
		return fmt.Errorf("%s: cache_dtype %q: the QSA layer requires a BF16 or FP8-e4m3 main KV cache (qsa.py:361-368) (%s)",
			s.Arch, s.CacheDType, issue)
	case s.BlockSize != 0 && s.BlockSize%16 != 0:
		return fmt.Errorf("%s: block_size %d: the backend's kernel block sizes are MultipleOf(16) (qsa.py:188-190) (%s)",
			s.Arch, s.BlockSize, issue)
	case s.EnableHCSP && s.PP > 1:
		return fmt.Errorf("%s: --enable-hc-sp requires pipeline_parallel_size 1, got %d (config/model.py:1549-1556) (%s)",
			s.Arch, s.PP, issue)
	}
	return nil
}

// hcSequenceParallel is whether Qwen4Exp's hyper-connection sequence parallelism is on:
// tensor parallelism above 1 and either --enable-hc-sp or sequence-parallel MoE
// (is_hc_sequence_parallel_enabled, common/hyperconnection.py:41-46).
func hcSequenceParallel(tp int, enableHCSP, sequenceParallelMoE bool) bool {
	return tp > 1 && (enableHCSP || sequenceParallelMoE)
}

// qsaRingRows is the indexer's key-state ring in rows: the open group's keys plus every row a
// speculative step stores before acceptance is known, in whole groups
// (QSAKeyStateCache.get_kv_cache_spec, common/qsa_cache.py:842-860).
func qsaRingRows(ratio, numSpec int) int {
	span := ratio + numSpec
	return (span + ratio - 1) / ratio * ratio
}

// qwen4ExpAttentionRuns is the KV cache specs one QSA attention layer registers, in order: the
// main K/V cache (FullAttentionSpec, qsa.py:552-561), the indexer's key-state ring
// (CircularBufferSpec) and its compressed key cache (MLAAttentionSpec at tokens_per_state =
// ratio, qsa_cache.py:866-875). main is the layer's sparse_gqa node that holds the engine's KV
// and scorer its indexer's. blockSize is the engine's block_size; indexerDType is
// attention_config.indexer_kv_dtype ("auto" and "" resolve to bf16, indexer_qsa.py:152-154);
// mrope says the model keeps exact int64 positions beside each ring row (qsa_cache.py:817-831).
func qwen4ExpAttentionRuns(main, scorer model.Node, tp, blockSize, numSpec int, indexerDType string, mrope bool) ([]KVLayerRun, error) {
	if main.AttentionKind != sparseGQA || !main.ReadsEngineKV() {
		return nil, fmt.Errorf("qwen4exp: the main node is not a sparse_gqa node that holds the engine's KV")
	}
	if scorer.AttentionKind != sparseGQA || scorer.ReadsEngineKV() || scorer.CompressRatio < 1 {
		return nil, fmt.Errorf("qwen4exp: the scorer is not a sparse_gqa node without engine KV and with a compress_ratio")
	}
	var elem int
	switch indexerDType {
	case "", "auto", "bf16":
		elem = 2
	case "fp8":
		elem = 1
	default:
		return nil, fmt.Errorf("qwen4exp: indexer_kv_dtype %q: the QSA indexer supports only bf16 or fp8 (indexer_qsa.py:155-162)", indexerDType)
	}
	ratio := scorer.CompressRatio
	if blockSize%ratio != 0 {
		return nil, fmt.Errorf("qwen4exp: block_size %d is not divisible by the compression ratio %d (qsa_cache.py:777-780)", blockSize, ratio)
	}
	keyWidth := scorer.HeadDim
	ringWidth := keyWidth
	if mrope {
		// round_up(key, 4) bf16 slots, then 3 axes x 4 bf16 slots of int64 positions.
		ringWidth = (keyWidth+3)/4*4 + 3*4
	}
	return []KVLayerRun{
		{Count: 1, Spec: "full", NumKVHeads: max(main.NumKVHeads/max(tp, 1), 1), HeadSize: main.HeadDim},
		{Count: 1, Spec: "circular_buffer", NumKVHeads: 1, HeadSize: ringWidth, BlockSize: qsaRingRows(ratio, numSpec), ElemBytes: 2},
		{Count: 1, Spec: "mla", NumKVHeads: 1, HeadSize: keyWidth, BlockSize: blockSize, TokensPerState: ratio, ElemBytes: elem},
	}, nil
}

// hasSparseGQA is whether g has a sparse_gqa layer, whose indexer reads indexer_kv_dtype
// (QSAIndexer, nvidia/indexer_qsa.py:152-154).
func hasSparseGQA(g *model.Graph) bool {
	for _, k := range g.LayerKinds {
		for _, n := range k.Nodes {
			if n.Op == model.OpAttention && n.AttentionKind == sparseGQA {
				return true
			}
		}
	}
	return false
}

// qsaLayerRuns is the runs of the QSA layer whose main node is n in layer kind lk: nothing for the
// indexer's scorer, which registers its caches with its main node. The MTP module's layer (draft)
// has the same caches, marked as the drafter's.
func qsaLayerRuns(id string, lk model.LayerKind, n model.Node, tp int, e engineconfig.Engine, draft bool) ([]KVLayerRun, error) {
	if !n.ReadsEngineKV() {
		return nil, nil
	}
	var scorer *model.Node
	for i := range lk.Nodes {
		if c := lk.Nodes[i]; c.Op == model.OpAttention && c.AttentionKind == sparseGQA && !c.ReadsEngineKV() {
			scorer = &lk.Nodes[i]
		}
	}
	if scorer == nil {
		return nil, fmt.Errorf("layer kind %s: a sparse_gqa attention node with no indexer scorer beside it", id)
	}
	numSpec := 0
	if e.Speculative != nil {
		numSpec = e.Speculative.NumSpecTokens
	}
	runs, err := qwen4ExpAttentionRuns(n, *scorer, tp, e.BlockSize, numSpec, e.IndexerKVDType, true)
	if err != nil {
		return nil, fmt.Errorf("layer kind %s: %w", id, err)
	}
	for i := range runs {
		runs[i].Draft = draft
	}
	return runs, nil
}

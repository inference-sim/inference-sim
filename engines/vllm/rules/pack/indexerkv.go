// Original to blis-vllm.

package pack

import (
	"fmt"
	"strings"
)

// DSAIndexerKVDTypes is how a release's DeepSeek sparse indexer (V4 and V4.1) resolves
// attention_config.indexer_kv_dtype. A release that does not carry the rule leaves
// Pack.DSAIndexer nil.
type DSAIndexerKVDTypes struct {
	// Architectures are those that build the indexer's key cache through dsa_indexer_uses_fp4.
	Architectures map[string]bool
	// Literal is AttentionConfig.indexer_kv_dtype's type, IndexerKVDType, in order
	// (vllm/config/attention.py:13@5df63a72): a value outside it is refused when the
	// configuration is built, for every model.
	Literal []string
	// Accepted is DSA_INDEXER_KV_DTYPES, in order (vllm/v1/attention/backends/mla/indexer.py:17@5df63a72).
	Accepted []string
	// Default is the dtype "auto" resolves to, the default dsa_indexer_uses_fp4 passes to
	// resolve_indexer_kv_dtype (:61).
	Default string
	// FP4 is the member that needs a Blackwell datacenter GPU, and FP4Family that GPU's
	// compute-capability major: is_device_capability_family(100) is major 10 (:80-84).
	FP4       string
	FP4Family int
}

// MXFP4BlockSize is the values one UE8M0 scale byte covers in MXFP4
// (vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:20@5df63a72).
const MXFP4BlockSize = 32

// IndexerQuantBlockSize is the values one fp32 scale covers in the FP8 indexer cache, 128 in
// DeepseekV4Indexer (vllm/models/deepseek_v4/attention.py:992@5df63a72) and in V4.1's
// _indexer_k_cache_head_dim (vllm/models/deepseek_v41/attention.py:100@5df63a72).
const IndexerQuantBlockSize = 128

// IndexerRecordBytes is the bytes of one key of the DeepSeek indexer's cache for the dtype that
// runs (a result of ResolveIndexerKVDType): head_dim fp8 values and head_dim/128 fp32 scales
// (132 for head_dim 128), or for mxfp4 two values to the byte and one scale byte per 32 values
// (68). V4 states it inline (deepseek_v4/attention.py:1005-1014@5df63a72) and V4.1 in
// _indexer_k_cache_head_dim (deepseek_v41/attention.py:92-100@5df63a72), by the same formulas.
func IndexerRecordBytes(resolved string, headDim int) int {
	if resolved == "mxfp4" {
		return headDim/2 + headDim/MXFP4BlockSize
	}
	return headDim + headDim/IndexerQuantBlockSize*4
}

// ResolveIndexerKVDType is the indexer cache dtype that runs for a requested
// attention_config.indexer_kv_dtype on a DeepSeek V4 or V4.1 deployment, or the error vLLM
// raises. family is the GPU's compute-capability major (9 for Hopper, 10 for Blackwell
// datacenter parts). Unstated is "auto".
//
// known is false when the pack has no rule for architecture, in which case resolved is requested
// and err is nil: the caller decides what that release does with the value. When known is true:
//
//   - a value outside IndexerKVDType is refused (pydantic, when the configuration is built);
//   - "auto" resolves to fp8, and fp8 and mxfp4 are the only values the indexer accepts
//     (dsa_indexer_uses_fp4, indexer.py:59-84@5df63a72), so bf16 and nvfp4 are refused although the
//     configuration accepts them;
//   - mxfp4 needs compute-capability major 10.
//
// The error texts are vLLM's own, so a deployment is refused with the reason vLLM gives. The oracle
// (oracle/indexer/oracle.py) runs vLLM's functions and the pack is compared with its table.
func (p *Pack) ResolveIndexerKVDType(architecture string, family int, requested string) (resolved string, known bool, err error) {
	r := p.DSAIndexer
	if r == nil || !r.Architectures[architecture] {
		return requested, false, nil
	}
	if requested == "" {
		requested = "auto"
	}
	member := func(list []string, s string) bool {
		for _, m := range list {
			if m == s {
				return true
			}
		}
		return false
	}
	if !member(r.Literal, requested) {
		return "", true, vllmMessage("Input should be " + pydanticChoices(r.Literal))
	}
	if requested == "auto" {
		requested = r.Default
	}
	if !member(r.Accepted, requested) {
		return "", true, vllmMessage(fmt.Sprintf("indexer_kv_dtype=%s is not supported by the DeepSeek sparse indexer (expected one of %s).",
			pyRepr(requested), pyTuple(r.Accepted)))
	}
	if requested == r.FP4 && family != r.FP4Family {
		return "", true, vllmMessage(fmt.Sprintf("indexer_kv_dtype=%s requires Blackwell datacenter GPUs (sm_10x, e.g. B200/GB200); "+
			"sm_120 (consumer Blackwell) and earlier architectures are not supported.", pyRepr(requested)))
	}
	return requested, true, nil
}

// vllmMessage is the text of an error vLLM raises, kept as it is written there: capitalised or
// ending in a full stop, which a Go error string is not.
type vllmMessage string

func (m vllmMessage) Error() string { return string(m) }

// pyRepr is Python's repr of a string with no quote or backslash in it, which every dtype name is.
func pyRepr(s string) string { return "'" + s + "'" }

// pyTuple is Python's repr of a tuple of such strings.
func pyTuple(l []string) string {
	q := make([]string, len(l))
	for i, s := range l {
		q[i] = pyRepr(s)
	}
	if len(q) == 1 {
		return "(" + q[0] + ",)"
	}
	return "(" + strings.Join(q, ", ") + ")"
}

// pydanticChoices is how pydantic lists the members of a Literal: 'a', 'b' or 'c'.
func pydanticChoices(l []string) string {
	q := make([]string, len(l))
	for i, s := range l {
		q[i] = pyRepr(s)
	}
	if len(q) < 2 {
		return strings.Join(q, "")
	}
	return strings.Join(q[:len(q)-1], ", ") + " or " + q[len(q)-1]
}

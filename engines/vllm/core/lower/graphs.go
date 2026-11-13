// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/worker/gpu/cudagraph_utils.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// _is_compatible (:106-133), CudaGraphManager._init_candidates (:251-389) and dispatch (:518-548),
// for a model without LoRA adapters, dynamic speculative decoding or microbatching. The recoverSSM
// kernel (use_kda_recoverssm) is modelled: it keeps mixed graphs to the decode token bound.

package lower

import (
	"fmt"
	"sort"
)

// Descriptor is BatchExecutionDescriptor: the graph a batch replays, or the shape it runs
// without one. Reqs, Uniform and MaxQueryLen are nil where vLLM's are None.
type Descriptor struct {
	Mode        GraphMode
	Tokens      int
	Reqs        *int
	Uniform     *int
	MaxQueryLen *int
}

func ptr(n int) *int { return &n }

func eq(a, b *int) bool { return a != nil && b != nil && *a == *b }

// GraphManager is the runner's CudaGraphManager for one set of captured graphs: it dispatches a
// batch to the graph that serves it and reports the padded shape.
type GraphManager struct {
	candidates map[int][]Descriptor
}

// ManagerInput is what the manager is built from.
type ManagerInput struct {
	// Mode is the cudagraph mode the runner got (resolve_cudagraph_mode_and_sizes).
	Mode GraphMode
	// Capture is the resolved capture of CaptureSizes.
	Capture Capture
	// MaxNumReqs is max_num_seqs, and DecodeQueryLen the runner's decode_query_len: draft tokens
	// plus the new tokens sampled per step.
	MaxNumReqs, DecodeQueryLen int
	// VarlenDecode is the runner's varlen_decode, on under adaptive verification.
	VarlenDecode bool
	// UseKDARecoverSSM is cache_config.use_kda_recoverssm: set for a Kimi linear-attention
	// model with speculation and replay state (vllm/config/vllm.py:3317-3334). Such a model cannot
	// capture a mixed graph wider than max_num_reqs x decode_query_len tokens
	// (cudagraph_utils.py:347-350).
	UseKDARecoverSSM bool
}

// NewGraphManager builds the manager (CudaGraphManager.__init__ and _init_candidates). A manager
// of mode NONE or of no sizes dispatches every batch eagerly.
func NewGraphManager(in ManagerInput) (*GraphManager, error) {
	if in.MaxNumReqs < 1 || in.DecodeQueryLen < 1 {
		return nil, fmt.Errorf("lower: a graph manager of %d requests and decode query length %d", in.MaxNumReqs, in.DecodeQueryLen)
	}
	g := &GraphManager{candidates: map[int][]Descriptor{}}
	if in.Mode == GraphNone || len(in.Capture.Sizes) == 0 {
		return g, nil
	}
	sizes := append([]int(nil), in.Capture.Sizes...)
	sort.Ints(sizes)
	maxDecodeTokens := in.MaxNumReqs * in.DecodeQueryLen
	decode, mixed, separate := in.Mode.decode(), in.Mode.mixed(), in.Mode.separate()
	byMode := map[GraphMode][]Descriptor{}
	varlen := separate && decode != GraphNone && in.VarlenDecode
	for _, n := range sizes {
		if varlen && n <= maxDecodeTokens {
			byMode[decode] = append(byMode[decode], Descriptor{Mode: decode, Tokens: n,
				Reqs: ptr(min(n, in.MaxNumReqs)), MaxQueryLen: ptr(in.DecodeQueryLen)})
		} else if separate && decode != GraphNone && !in.VarlenDecode {
			rounded := (n + in.DecodeQueryLen - 1) / in.DecodeQueryLen * in.DecodeQueryLen
			reqs := rounded / in.DecodeQueryLen
			if rounded <= maxDecodeTokens && rounded <= in.Capture.Max && reqs <= in.MaxNumReqs {
				d := Descriptor{Mode: decode, Tokens: rounded, Reqs: ptr(reqs), Uniform: ptr(in.DecodeQueryLen)}
				if !contains(byMode[decode], d) {
					byMode[decode] = append(byMode[decode], d)
				}
			}
		}
		if mixed != GraphNone && (!in.UseKDARecoverSSM || n <= maxDecodeTokens) {
			d := Descriptor{Mode: mixed, Tokens: n}
			if mixed == GraphFull {
				d.Reqs = ptr(min(n, in.MaxNumReqs))
			}
			byMode[mixed] = append(byMode[mixed], d)
		}
	}
	for _, mode := range []GraphMode{GraphFull, GraphPiecewise} {
		ds := byMode[mode]
		// descs.sort(key=num_tokens, reverse=True) is stable, and the candidates walk the list reversed.
		sort.SliceStable(ds, func(i, j int) bool { return ds[i].Tokens > ds[j].Tokens })
		for i, j := 0, len(ds)-1; i < j; i, j = i+1, j-1 {
			ds[i], ds[j] = ds[j], ds[i]
		}
		start := 0
		for i := 0; i < len(ds); {
			j := i
			for j < len(ds) && ds[j].Tokens == ds[i].Tokens {
				j++
			}
			for k := start; k <= ds[i].Tokens; k++ {
				g.candidates[k] = append(g.candidates[k], ds[i:j]...)
			}
			start = ds[i].Tokens + 1
			i = j
		}
	}
	return g, nil
}

func contains(ds []Descriptor, d Descriptor) bool {
	for _, x := range ds {
		if x.Mode == d.Mode && x.Tokens == d.Tokens && eq(x.Reqs, d.Reqs) && eq(x.Uniform, d.Uniform) &&
			(x.Uniform == nil) == (d.Uniform == nil) && (x.Reqs == nil) == (d.Reqs == nil) &&
			(x.MaxQueryLen == nil) == (d.MaxQueryLen == nil) {
			return true
		}
	}
	return false
}

// Dispatch is CudaGraphManager.dispatch (:518-548): the first candidate for the token count that
// is compatible with the batch, or an eager descriptor of the batch's own shape. uniform is the
// per-request token count of a uniform decode batch and maxQueryLen the bound a varlen graph
// checks; nil is None.
func (g *GraphManager) Dispatch(reqs, tokens int, uniform, maxQueryLen *int) Descriptor {
	if tokens > 0 {
		for _, d := range g.candidates[tokens] {
			if compatible(d, reqs, tokens, uniform, maxQueryLen) {
				return d
			}
		}
	}
	return Descriptor{Mode: GraphNone, Tokens: tokens, Reqs: ptr(reqs)}
}

// compatible is _is_compatible (:106-133): a graph with no uniform count serves any, one with no
// request count needs no request padding, and a graph that bounds the query length serves only a
// batch that states a bound within it.
func compatible(d Descriptor, reqs, tokens int, uniform, maxQueryLen *int) bool {
	return (d.Uniform == nil || eq(d.Uniform, uniform)) &&
		(d.MaxQueryLen == nil || (maxQueryLen != nil && *d.MaxQueryLen >= *maxQueryLen)) &&
		(d.Reqs == nil || *d.Reqs >= reqs) &&
		d.Tokens >= tokens
}

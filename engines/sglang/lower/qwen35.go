// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/configs/mamba_utils.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/mem_cache/allocation_sizing.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package lower

import (
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"

	"github.com/inference-sim/inference-sim/engines/sglang/config"
)

// issueHybrid is the issue that models the hybrid (Gated DeltaNet plus attention) model's cache.
const issueHybrid = "inference-sim/inference-sim#2366"

// lowerRecurrent states, for a model with recurrent layers, the width of the recurrent state's
// element and how many states a request keeps.
//
// The state's element is the temporal state's dtype: mamba2_state_dtype
// (configs/mamba_utils.py:47-110@297e54fd) takes SGLANG_MAMBA_SSM_DTYPE, which --mamba-ssm-dtype
// sets (arg_groups/serving_hook.py:439), then the model configuration's mamba_ssm_dtype, which
// the graph carries as the recurrent node's state_dtype, then float32. The convolution state is
// bfloat16 unless SGLANG_MAMBA_CONV_DTYPE names another width; that variable is not an input of
// Resolve (the oracle clears every SGLANG_ variable), so the width is bfloat16. The plan carries
// one state width, so the convolution state, a few percent of the state's elements, is priced at
// the temporal state's width; the plan notes it.
//
// With the radix cache off a request holds one slot of the mamba pool, the latest state
// (RecurrentLatest). With it on, SGLang keeps the states of cached prefixes in that pool by the
// mamba radix-cache strategy. For extra_buffer a request also holds tracking slots beside its
// working slot (get_mamba_tracking_slots, mem_cache/allocation_sizing.py:11-19): 2 with the
// overlap schedule, 1 without it or with extra_buffer_lazy, so 3 states a request with overlap and
// 2 otherwise. The plan's page-aligned word prices 2 (memory.go RecurrentStateBytes: two pages
// plus speculation and prefill checkpoints), so it is one state short with overlap, and it states
// neither the snapshot interval (mamba_track_interval) nor the pool's share of the cache budget.
// The pool is refused by name until the scheduler's side of it is lowered (#94).
func lowerRecurrent(a *config.ServerArgs, g *model.Graph, c *plan.CacheFormat, note func(name, value, reason string)) error {
	var graphDType model.DType
	mixed := false
	g.EachNode(func(_ string, n model.Node) {
		if n.Op != model.OpRecurrentUpdate {
			return
		}
		if graphDType != "" && n.StateDType != graphDType {
			mixed = true
		}
		graphDType = n.StateDType
	})
	if mixed {
		return &NotLowered{Name: "recurrent layers with different state dtypes", Issue: issueHybrid}
	}
	temporal := model.DTypeFP32 // mamba2_state_dtype's default
	if graphDType != "" {
		temporal = graphDType
	}
	if a.MambaSsmDtype != nil {
		switch *a.MambaSsmDtype {
		case "float32":
			temporal = model.DTypeFP32
		case "bfloat16":
			temporal = model.DTypeBF16
		case "float16":
			temporal = model.DTypeFP16
		default:
			return &NotLowered{Name: "mamba_ssm_dtype=" + *a.MambaSsmDtype, Issue: issueHybrid}
		}
	}
	c.StateElementBytes = temporal.Bytes()
	if !a.DisableRadixCache {
		return &NotLowered{Name: "the mamba radix cache (strategy " + a.MambaRadixCacheStrategy + ")", Issue: issueHybrid}
	}
	c.Recurrent = plan.RecurrentLatest
	note("recurrent_state_width", string(temporal),
		"the temporal state's dtype (mamba_ssm_dtype); the bfloat16 convolution state is priced at the same width")
	return nil
}

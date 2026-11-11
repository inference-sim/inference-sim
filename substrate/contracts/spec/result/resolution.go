package result

import (
	"sort"
	"strconv"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// ResolutionsOf converts the cost model's resolution of a deployment (kernel.Resolved) to the
// Result's resolutions, sorted by setting.
//
// Each resolved layout width and mode is a setting, named as the field of kernel.Resolution
// in snake case: tensor_parallel_width, data_parallel_width, expert_parallel_width,
// all_reduce_backend, async_scheduling, cascade_attention and sequence_parallel_moe. Each
// request the cost model declined (kernel.Resolution.Overrides) carries the value the host asked for
// and the reason onto the setting of the same name, or adds a setting when it names one the
// widths and modes do not. A setting the cost model declined to honour is always recorded,
// and one it honoured is recorded with the value it resolved, so a reader can compare a
// value a benchmark run logged with the cost model's own for every setting, not only for the
// ones that moved.
func ResolutionsOf(r kernel.Resolution) []Resolution {
	by := map[string]*Resolution{}
	set := func(name, value string) {
		if value == "" {
			return // an implementation that does not report it
		}
		by[name] = &Resolution{Setting: name, Engine: value}
	}
	set("tensor_parallel_width", strconv.Itoa(r.TensorParallel()))
	set("data_parallel_width", strconv.Itoa(r.DataParallel()))
	set("expert_parallel_width", strconv.Itoa(r.ExpertParallel()))
	set("all_reduce_backend", r.AllReduceBackend)
	set("async_scheduling", strconv.FormatBool(r.AsyncScheduling))
	set("cascade_attention", strconv.FormatBool(r.CascadeAttention))
	set("sequence_parallel_moe", strconv.FormatBool(r.SequenceParallelMoE))
	for _, o := range r.Overrides {
		x, ok := by[o.Field]
		if !ok {
			x = &Resolution{Setting: o.Field, Engine: o.Resolved}
			by[o.Field] = x
		}
		x.Host, x.Reason = o.Requested, o.Reason
		if x.Host == x.Engine {
			x.Reason = "" // nothing moved, so there is nothing to explain
		}
	}
	out := make([]Resolution, 0, len(by))
	for _, x := range by {
		out = append(out, *x)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Setting < out[j].Setting })
	return out
}

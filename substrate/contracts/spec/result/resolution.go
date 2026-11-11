package result

import (
	"fmt"
	"sort"
	"strconv"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
)

// settingOf names the Result's setting for a field the cost model names in a declined request.
// The cost model spells the all-reduce request allreduce_backend; the Result's setting is
// all_reduce_backend, so a declined request lands on the row of the setting it answers and not
// on a second one. A field with no entry keeps its own name.
func settingOf(field string) string {
	if s, ok := settingNames[field]; ok {
		return s
	}
	return field
}

var settingNames = map[string]string{"allreduce_backend": "all_reduce_backend"}

// ResolutionsOf converts the cost model's resolution of a deployment (kernel.Resolved) to the
// Result's resolutions, sorted by setting.
//
// Each resolved layout width and mode is a setting, named as the field of kernel.Resolution
// in snake case: tensor_parallel_width, data_parallel_width, expert_parallel_width,
// all_reduce_backend, async_scheduling, cascade_attention and sequence_parallel_moe. Each
// request the cost model declined (kernel.Resolution.Overrides) carries the value the host asked for
// and the reason onto the setting it answers (settingOf), or adds a setting when it names one the
// widths and modes do not. A setting the cost model declined to honour is always recorded,
// and one it honoured is recorded with the value it resolved, so a reader can compare a
// value a benchmark run logged with the cost model's own for every setting, not only for the
// ones that moved.
func ResolutionsOf(r kernel.Resolution) []Resolution { return ResolutionsOfPool(r, "") }

// ResolutionsOfPool is ResolutionsOf for the cost model of one pool of the deployment: each
// resolution names that pool, as PoolLabels does, so a disaggregated run's prefill and decode
// pools, which resolve differently, are told apart. An empty pool is the whole deployment.
// Join the lists of several pools with SortResolutions.
func ResolutionsOfPool(r kernel.Resolution, pool string) []Resolution {
	by := map[string]*Resolution{}
	set := func(name, value string) {
		if value == "" {
			return // an implementation that does not report it
		}
		by[name] = &Resolution{Setting: name, Pool: pool, Engine: value}
	}
	set("tensor_parallel_width", strconv.Itoa(r.TensorParallel()))
	set("data_parallel_width", strconv.Itoa(r.DataParallel()))
	set("expert_parallel_width", strconv.Itoa(r.ExpertParallel()))
	set("all_reduce_backend", r.AllReduceBackend)
	set("async_scheduling", strconv.FormatBool(r.AsyncScheduling))
	set("cascade_attention", strconv.FormatBool(r.CascadeAttention))
	set("sequence_parallel_moe", strconv.FormatBool(r.SequenceParallelMoE))
	for _, o := range r.Overrides {
		name := settingOf(o.Field)
		x, ok := by[name]
		if !ok {
			x = &Resolution{Setting: name, Pool: pool, Engine: o.Resolved}
			by[name] = x
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
	SortResolutions(out)
	return out
}

// SortResolutions puts resolutions in the order a Result lists them: by setting, then pool.
func SortResolutions(xs []Resolution) {
	sort.Slice(xs, func(i, j int) bool {
		if xs[i].Setting != xs[j].Setting {
			return xs[i].Setting < xs[j].Setting
		}
		return xs[i].Pool < xs[j].Pool
	})
}

// PoolLabels names the pools of a deployment, in order, as a Resolution's Pool does: the
// pool's role when no other pool has it, else the role and the pool's position among those
// of that role, counted from zero ("decode-0", "decode-1").
func PoolLabels(d deployment.Deployment) []string {
	count := map[deployment.Role]int{}
	for _, p := range d.Pools {
		count[p.Role]++
	}
	seen := map[deployment.Role]int{}
	out := make([]string, 0, len(d.Pools))
	for _, p := range d.Pools {
		if count[p.Role] == 1 {
			out = append(out, string(p.Role))
		} else {
			out = append(out, fmt.Sprintf("%s-%d", p.Role, seen[p.Role]))
		}
		seen[p.Role]++
	}
	return out
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

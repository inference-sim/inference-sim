package result

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
)

// withResolutions returns the valid Result with an override and two resolutions.
func withResolutions() *Result {
	r := valid()
	r.Overrides = []Override{{Setting: "async_scheduling", Input: "true", Run: "false", Source: "command line"}}
	r.Resolutions = []Resolution{
		{Setting: "async_scheduling", Host: "false", Engine: "false"},
		{Setting: "tensor_parallel_width", Engine: "2"},
	}
	return r
}

func TestAResultWithResolutionsValidatesAndRoundTrips(t *testing.T) {
	r := withResolutions()
	if p := r.Validate(); !p.OK() {
		t.Fatalf("%s", p.Error())
	}
	for _, f := range []Format{FormatJSON, FormatYAML} {
		b, err := Marshal(r, f)
		if err != nil {
			t.Fatal(err)
		}
		back := &Result{}
		if err := Unmarshal(b, back); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		if !reflect.DeepEqual(back.Resolutions, r.Resolutions) {
			t.Errorf("%s: the resolutions did not round-trip:\n got %+v\nwant %+v", f, back.Resolutions, r.Resolutions)
		}
	}
}

// Known-bad cases (SUB-2): each breaks one rule of the resolutions and expects a problem at
// the named path.
func TestResolutionsAreRefusedWhenMalformed(t *testing.T) {
	cases := []struct {
		name   string
		break_ func(r *Result)
		path   string
		msg    string
	}{
		{"resolution without a setting", func(r *Result) { r.Resolutions[0].Setting = "" },
			"resolutions[0].setting", "required"},
		{"resolutions out of order", func(r *Result) { r.Resolutions[0], r.Resolutions[1] = r.Resolutions[1], r.Resolutions[0] },
			"resolutions[1].setting", "sorted"},
		{"one setting resolved twice", func(r *Result) { r.Resolutions[1].Setting = "async_scheduling" },
			"resolutions[1].setting", "once each"},
		{"resolution without a value", func(r *Result) { r.Resolutions[1].Engine = "" },
			"resolutions[1].engine", "required"},
		{"a changed value with no reason", func(r *Result) {
			r.Resolutions[1].Host = "4"
			r.Overrides = nil
		}, "resolutions[1].reason", "required"},
		{"a reason for an unchanged value", func(r *Result) { r.Resolutions[1].Host, r.Resolutions[1].Reason = "2", "because" },
			"resolutions[1].reason", "only where"},
		{"a pool the deployment lacks", func(r *Result) { r.Resolutions[1].Pool = "prefill" },
			"resolutions[1].pool", "not a pool of the deployment"},
		{"one setting resolved twice in one pool", func(r *Result) {
			r.Resolutions = []Resolution{{Setting: "a", Pool: "colocated", Engine: "1"}, {Setting: "a", Pool: "colocated", Engine: "1"}}
		}, "resolutions[1].setting", "once each"},
		{"pools out of order within a setting", func(r *Result) {
			r.Input.Deployment.Pools = append(r.Input.Deployment.Pools, deployment.Pool{Role: deployment.RoleColocated})
			r.Resolutions = []Resolution{{Setting: "a", Pool: "colocated-1", Engine: "1"}, {Setting: "a", Pool: "colocated-0", Engine: "1"}}
		}, "resolutions[1].setting", "sorted by setting and then pool"},
		{"asked for the input's value after the host overrode it", func(r *Result) { r.Resolutions[0].Host = "true" },
			"resolutions[0].host", "overrode"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := withResolutions()
			c.break_(r)
			p := r.Validate()
			for _, it := range p.Errors() {
				if it.Path == c.path && strings.Contains(it.Message, c.msg) {
					return
				}
			}
			t.Errorf("expected a problem at %q containing %q, got:\n%s", c.path, c.msg, p.Error())
		})
	}
}

func TestPoolLabelsNameEachPoolByRoleAndPositionWhenSeveralShareOne(t *testing.T) {
	pool := func(role deployment.Role) deployment.Pool { return deployment.Pool{Role: role} }
	for name, c := range map[string]struct {
		pools []deployment.Pool
		want  []string
	}{
		"one colocated pool":             {[]deployment.Pool{pool("colocated")}, []string{"colocated"}},
		"prefill and decode":             {[]deployment.Pool{pool("prefill"), pool("decode")}, []string{"prefill", "decode"}},
		"two decode pools and a prefill": {[]deployment.Pool{pool("decode"), pool("prefill"), pool("decode")}, []string{"decode-0", "prefill", "decode-1"}},
	} {
		if got := PoolLabels(deployment.Deployment{Pools: c.pools}); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: %v, want %v", name, got, c.want)
		}
	}
}

// A disaggregated run's pools resolve the same setting differently; each resolution names its
// pool, and the joined list is in the order a Result requires.
func TestResolutionsOfPoolNamesThePool(t *testing.T) {
	for _, x := range ResolutionsOfPool(kernel.Resolution{Overrides: []kernel.Override{{Field: "quantization", Requested: "a", Resolved: "b", Reason: "r"}}}, "decode") {
		if x.Pool != "decode" {
			t.Errorf("%s names pool %q, want decode", x.Setting, x.Pool)
		}
	}
	prefill := ResolutionsOfPool(kernel.Resolution{TensorParallelWidth: 1, DataParallelWidth: 1}, "prefill")
	decode := ResolutionsOfPool(kernel.Resolution{TensorParallelWidth: 4, DataParallelWidth: 1}, "decode")
	all := append(append([]Resolution(nil), decode...), prefill...)
	SortResolutions(all)
	for i := 1; i < len(all); i++ {
		a, b := all[i-1], all[i]
		if a.Setting > b.Setting || (a.Setting == b.Setting && a.Pool >= b.Pool) {
			t.Fatalf("not sorted by setting and pool at %d: %+v then %+v", i, a, b)
		}
	}
	tp := map[string]string{}
	for _, x := range all {
		if x.Pool == "" {
			t.Errorf("%s has no pool", x.Setting)
		}
		if x.Setting == "tensor_parallel_width" {
			tp[x.Pool] = x.Engine
		}
	}
	if !reflect.DeepEqual(tp, map[string]string{"prefill": "1", "decode": "4"}) {
		t.Errorf("tensor_parallel_width by pool: %v", tp)
	}
	// A deployment with those two pools accepts them.
	r := valid()
	r.Input.Deployment.Pools = []deployment.Pool{
		{Role: deployment.RolePrefill, Nodes: 1, Parallel: deployment.Parallelism{TP: 1, PP: 1, DP: 1}},
		{Role: deployment.RoleDecode, Nodes: 1, Parallel: deployment.Parallelism{TP: 4, PP: 1, DP: 1}},
	}
	r.Resolutions = all
	for _, it := range r.Validate().Errors() {
		if strings.HasPrefix(it.Path, "resolutions") {
			t.Errorf("%s: %s", it.Path, it.Message)
		}
	}
}

func TestResolutionsOfRecordsEverySettingAndTheDeclinedRequests(t *testing.T) {
	got := ResolutionsOf(kernel.Resolution{
		TensorParallelWidth: 2, DataParallelWidth: 4, ExpertParallelWidth: 1,
		AllReduceBackend: "custom", AsyncScheduling: true,
		Overrides: []kernel.Override{
			{Field: "async_scheduling", Requested: "false", Resolved: "true", Reason: "the layout needs it"},
			{Field: "quantization", Requested: "fp4", Resolved: "fp8", Reason: "the checkpoint's own method runs"},
			{Field: "cascade_attention", Requested: "true", Resolved: "false", Reason: "off for this model"},
		},
	})
	want := []Resolution{
		{Setting: "all_reduce_backend", Engine: "custom"},
		{Setting: "async_scheduling", Host: "false", Engine: "true", Reason: "the layout needs it"},
		{Setting: "cascade_attention", Host: "true", Engine: "false", Reason: "off for this model"},
		{Setting: "data_parallel_width", Engine: "4"},
		{Setting: "expert_parallel_width", Engine: "1"},
		{Setting: "quantization", Host: "fp4", Engine: "fp8", Reason: "the checkpoint's own method runs"},
		{Setting: "sequence_parallel_moe", Engine: "false"},
		{Setting: "tensor_parallel_width", Engine: "2"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got\n%+v\nwant\n%+v", got, want)
	}
	r := valid()
	r.Resolutions = got
	if p := r.Validate(); !p.OK() {
		t.Errorf("the conversion's output does not validate:\n%s", p.Error())
	}
	// A reporter that fills no width still yields the floor of one, never an empty value, and
	// one that names no backend adds no backend setting.
	for _, x := range ResolutionsOf(kernel.Resolution{}) {
		if x.Setting == "all_reduce_backend" {
			t.Errorf("a backend that was not reported is recorded: %+v", x)
		}
		if x.Engine == "" {
			t.Errorf("%s has no value", x.Setting)
		}
	}
}

// With no declined request the conversion still records every layout width and mode, so a
// reader can compare a logged value with the cost model's for a setting that did not move.
func TestResolutionsOfRecordsEverySettingWhenNothingWasDeclined(t *testing.T) {
	var settings []string
	for _, x := range ResolutionsOf(kernel.Resolution{AllReduceBackend: "nccl"}) {
		settings = append(settings, x.Setting)
		if x.Host != "" || x.Reason != "" {
			t.Errorf("%s: nothing was declined, but the record names a host value or a reason: %+v", x.Setting, x)
		}
	}
	want := []string{"all_reduce_backend", "async_scheduling", "cascade_attention", "data_parallel_width",
		"expert_parallel_width", "sequence_parallel_moe", "tensor_parallel_width"}
	if !reflect.DeepEqual(settings, want) {
		t.Errorf("settings %v, want %v", settings, want)
	}
}

// The cost model spells the all-reduce request allreduce_backend. A declined request lands on
// the always-present all_reduce_backend row, not on a second row of its own name.
func TestADeclinedAllReduceRequestLandsOnTheSettingItAnswers(t *testing.T) {
	got := ResolutionsOf(kernel.Resolution{AllReduceBackend: "nccl", Overrides: []kernel.Override{
		{Field: "allreduce_backend", Requested: "custom", Resolved: "nccl", Reason: "3 is not a supported rank count"}}})
	var rows []Resolution
	for _, x := range got {
		if strings.Contains(x.Setting, "reduce") {
			rows = append(rows, x)
		}
	}
	want := []Resolution{{Setting: "all_reduce_backend", Host: "custom", Engine: "nccl", Reason: "3 is not a supported rank count"}}
	if !reflect.DeepEqual(rows, want) {
		t.Errorf("rows %+v, want %+v", rows, want)
	}
}

// A declined request whose requested value equals the resolved one has nothing to explain: the
// reason is dropped so the output validates.
func TestARequestTheEngineKeptCarriesNoReason(t *testing.T) {
	got := ResolutionsOf(kernel.Resolution{Overrides: []kernel.Override{
		{Field: "quantization", Requested: "fp8", Resolved: "fp8", Reason: "kept"}}})
	r := valid()
	r.Resolutions = got
	if p := r.Validate(); !p.OK() {
		t.Fatalf("%s", p.Error())
	}
	for _, x := range got {
		if x.Setting == "quantization" && (x.Reason != "" || x.Host != "fp8") {
			t.Errorf("%+v", x)
		}
	}
}

// Resolutions of one setting in several pools are ordered by pool, whatever order they arrive in.
func TestSortResolutionsOrdersPoolsWithinASetting(t *testing.T) {
	for _, in := range [][]Resolution{
		{{Setting: "s", Pool: "prefill", Engine: "1"}, {Setting: "s", Pool: "decode", Engine: "1"}, {Setting: "s", Engine: "1"}},
		{{Setting: "s", Engine: "1"}, {Setting: "s", Pool: "decode", Engine: "1"}, {Setting: "s", Pool: "prefill", Engine: "1"}},
	} {
		SortResolutions(in)
		if in[0].Pool != "" || in[1].Pool != "decode" || in[2].Pool != "prefill" {
			t.Errorf("order %+v", in)
		}
	}
}

// The cost model spells the cache dtype request cache_dtype; the engine's setting, and so the
// Result's, is kv_cache_dtype. A run that asked for fp8 and was served the attention backend's
// packed record names kv_cache_dtype, and no row is named cache_dtype.
func TestADeclinedCacheDTypeRequestIsNamedAsTheEngineNamesIt(t *testing.T) {
	got := ResolutionsOf(kernel.Resolution{Overrides: []kernel.Override{
		{Field: "cache_dtype", Requested: "fp8", Resolved: "fp8_ds_mla", Reason: "the attention backend serves this cache as fp8_ds_mla"}}})
	var rows []Resolution
	for _, x := range got {
		if strings.Contains(x.Setting, "cache_dtype") {
			rows = append(rows, x)
		}
	}
	want := []Resolution{{Setting: "kv_cache_dtype", Host: "fp8", Engine: "fp8_ds_mla", Reason: "the attention backend serves this cache as fp8_ds_mla"}}
	if !reflect.DeepEqual(rows, want) {
		t.Errorf("rows %+v, want %+v", rows, want)
	}
}

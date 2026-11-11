package result

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
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

// Moved from inference-sim/blis-schemas/spec/deployment/cudagraph_test.go at 70246460a379f8b1898fbf6c36a3091364118950.

package engineconfig

import (
	"math/rand"
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"
)

func capture(m int) *int { return &m }

// The capture maximum is tri-state like the other engine requests: unstated takes the
// release default, and a stated value survives a round trip as itself.
func TestCUDAGraphCaptureDecodes(t *testing.T) {
	for _, c := range []struct {
		in    string
		max   *int
		sizes []int
	}{
		{"{}", nil, nil},
		{"{max_cudagraph_capture_size: 64}", capture(64), nil},
		{"{cudagraph_capture_sizes: [1, 2, 4, 8]}", nil, []int{1, 2, 4, 8}},
		{"{max_cudagraph_capture_size: 8, cudagraph_capture_sizes: [8, 1]}", capture(8), []int{8, 1}},
	} {
		var e Engine
		if err := yaml.Unmarshal([]byte(c.in), &e); err != nil {
			t.Fatalf("%s: %v", c.in, err)
		}
		if !reflect.DeepEqual(e.MaxCUDAGraphCaptureSize, c.max) || !reflect.DeepEqual(e.CUDAGraphCaptureSizes, c.sizes) {
			t.Errorf("%s: got %v %v", c.in, e.MaxCUDAGraphCaptureSize, e.CUDAGraphCaptureSizes)
		}
	}
	// A stated empty list is not the field left out: it decodes non-nil, so validation
	// can refuse it as the engine does.
	var e Engine
	if err := yaml.Unmarshal([]byte("{cudagraph_capture_sizes: []}"), &e); err != nil || e.CUDAGraphCaptureSizes == nil {
		t.Errorf("an empty list decoded as %v (%v)", e.CUDAGraphCaptureSizes, err)
	}
}

// Load, marshal and load again over many stated values: both knobs come back unchanged,
// and an unstated one stays unstated.
func TestCUDAGraphCaptureRoundTrips(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	for i := 0; i < 1000; i++ {
		var e Engine
		if i%3 != 0 {
			e.MaxCUDAGraphCaptureSize = capture(1 + r.Intn(4096))
		}
		if i%2 == 0 {
			for n := 1 + r.Intn(8); n > 0; n-- {
				e.CUDAGraphCaptureSizes = append(e.CUDAGraphCaptureSizes, 1+r.Intn(4096))
			}
		}
		out, err := yaml.Marshal(e)
		if err != nil {
			t.Fatal(err)
		}
		var back Engine
		if err := yaml.Unmarshal(out, &back); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(back.MaxCUDAGraphCaptureSize, e.MaxCUDAGraphCaptureSize) ||
			!reflect.DeepEqual(back.CUDAGraphCaptureSizes, e.CUDAGraphCaptureSizes) {
			t.Fatalf("%+v round-tripped through %q to %+v", e, out, back)
		}
	}
}

func TestCUDAGraphCaptureValidation(t *testing.T) {
	for _, c := range []struct {
		name  string
		max   *int
		sizes []int
		ok    bool
	}{
		{"unstated", nil, nil, true},
		{"a maximum", capture(64), nil, true},
		{"sizes", nil, []int{1, 8, 64}, true},
		{"both", capture(64), []int{8, 64}, true},
		{"zero maximum", capture(0), nil, false},
		{"negative maximum", capture(-8), nil, false},
		{"empty list", nil, []int{}, false},
		{"zero size", nil, []int{0, 8}, false},
		{"negative size", nil, []int{8, -1}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := engineOnly(Engine{MaxCUDAGraphCaptureSize: c.max, CUDAGraphCaptureSizes: c.sizes}).Validate()
			if p.OK() != c.ok {
				t.Errorf("ok = %v, want %v\n%s", p.OK(), c.ok, p.Error())
			}
		})
	}
}

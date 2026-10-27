// Moved from inference-sim/blis-schemas/rules/cudagraph_test.go at 70246460a379f8b1898fbf6c36a3091364118950.

package pack_test

import (
	"errors"
	"reflect"
	"testing"

	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-vllm/rules/v0_31"
)

func ip(n int) *int { return &n }

// stride is the sizes the engine generates for a maximum m, with nothing else added:
// 1, 2, 4, multiples of 8 below 256, multiples of 16 from 256.
func stride(m int) []int {
	var out []int
	for _, s := range []int{1, 2, 4} {
		if s <= m {
			out = append(out, s)
		}
	}
	for s := 8; s <= m && s < 256; s += 8 {
		out = append(out, s)
	}
	for s := 256; s <= m; s += 16 {
		out = append(out, s)
	}
	return out
}

func TestCUDAGraphCaptureSizes(t *testing.T) {
	p := v0_31.Pack()
	for _, c := range []struct {
		name     string
		in       rules.CUDAGraphCaptureInput
		resolved bool
		want     rules.CUDAGraphCapture
		err      error
	}{
		{"unstated is the release default", rules.CUDAGraphCaptureInput{MaxNumBatchedTokens: 8192}, false, rules.CUDAGraphCapture{}, nil},
		{"mode NONE captures nothing", rules.CUDAGraphCaptureInput{Mode: "NONE", MaxCaptureSize: ip(64)}, true,
			rules.CUDAGraphCapture{Sizes: []int{}}, nil},
		{"an internal deployment's 64", rules.CUDAGraphCaptureInput{MaxNumBatchedTokens: 8192, MaxCaptureSize: ip(64)}, true,
			rules.CUDAGraphCapture{Sizes: stride(64), Max: 64}, nil},
		{"above 256 on the 16 stride", rules.CUDAGraphCaptureInput{MaxNumBatchedTokens: 8192, MaxCaptureSize: ip(512)}, true,
			rules.CUDAGraphCapture{Sizes: stride(512), Max: 512}, nil},
		{"off the stride truncates", rules.CUDAGraphCaptureInput{MaxNumBatchedTokens: 8192, MaxCaptureSize: ip(100)}, true,
			rules.CUDAGraphCapture{Sizes: stride(100), Max: 96, Truncated: true}, nil},
		{"capped at max_num_batched_tokens, which is added", rules.CUDAGraphCaptureInput{MaxNumBatchedTokens: 100, MaxCaptureSize: ip(512)}, true,
			rules.CUDAGraphCapture{Sizes: append(stride(100), 100), Max: 100, Capped: true}, nil},
		{"max_num_batched_tokens inside the range is added", rules.CUDAGraphCaptureInput{MaxNumBatchedTokens: 100, MaxCaptureSize: ip(100)}, true,
			rules.CUDAGraphCapture{Sizes: append(stride(100), 100), Max: 100}, nil},
		{"stated sizes de-duplicated, sorted, filtered", rules.CUDAGraphCaptureInput{MaxNumBatchedTokens: 64,
			CaptureSizes: []int{32, 8, 128, 8, 1}}, true,
			rules.CUDAGraphCapture{Sizes: []int{1, 8, 32}, Max: 32, Dropped: []int{128}}, nil},
		{"consistent pair", rules.CUDAGraphCaptureInput{MaxNumBatchedTokens: 64, MaxCaptureSize: ip(32),
			CaptureSizes: []int{8, 32}}, true, rules.CUDAGraphCapture{Sizes: []int{8, 32}, Max: 32}, nil},
		{"inconsistent pair", rules.CUDAGraphCaptureInput{MaxNumBatchedTokens: 64, MaxCaptureSize: ip(64),
			CaptureSizes: []int{8, 32}}, true, rules.CUDAGraphCapture{Sizes: []int{8, 32}, Max: 32},
			rules.ErrCUDAGraphCaptureInconsistent},
		{"pair made inconsistent by the filter", rules.CUDAGraphCaptureInput{MaxNumBatchedTokens: 16, MaxCaptureSize: ip(32),
			CaptureSizes: []int{8, 32}}, true, rules.CUDAGraphCapture{Sizes: []int{8}, Max: 8, Dropped: []int{32}},
			rules.ErrCUDAGraphCaptureInconsistent},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, resolved, err := p.CUDAGraphCaptureSizes(c.in)
			if resolved != c.resolved || !errors.Is(err, c.err) || (c.err == nil && err != nil) {
				t.Fatalf("resolved %v err %v, want %v %v", resolved, err, c.resolved, c.err)
			}
			if c.resolved && !reflect.DeepEqual(got, c.want) {
				t.Errorf("got %+v\nwant %+v", got, c.want)
			}
		})
	}
	if _, resolved, err := p.CUDAGraphCaptureSizes(rules.CUDAGraphCaptureInput{MaxCaptureSize: ip(64)}); !resolved || err == nil {
		t.Error("resolved a capture without max_num_batched_tokens")
	}
}

// Properties over many stated maxima: the sizes are ascending, at least 1, within both
// the stated maximum and max_num_batched_tokens, and the last is the maximum that runs.
func TestCUDAGraphCaptureSizesProperties(t *testing.T) {
	p := v0_31.Pack()
	for m := 1; m <= 1100; m++ {
		for _, mnbt := range []int{1, 7, 64, 100, 512, 8192} {
			c, _, err := p.CUDAGraphCaptureSizes(rules.CUDAGraphCaptureInput{MaxNumBatchedTokens: mnbt, MaxCaptureSize: ip(m)})
			if err != nil {
				t.Fatal(err)
			}
			limit := min(m, mnbt)
			for i, s := range c.Sizes {
				if s < 1 || s > limit || (i > 0 && s <= c.Sizes[i-1]) {
					t.Fatalf("m %d mnbt %d: sizes %v", m, mnbt, c.Sizes)
				}
			}
			if c.Max != c.Sizes[len(c.Sizes)-1] || c.Capped != (m > mnbt) || c.Truncated != (c.Max < limit) {
				t.Fatalf("m %d mnbt %d: %+v", m, mnbt, c)
			}
		}
	}
}

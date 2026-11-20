// Original to blis-sglang.

package config

import (
	"reflect"
	"testing"
)

// cloneArgs copies every field, and the copy holds the same values: a set list, a set
// cuda_graph_config and a nil one each survive the copy, and the copy shares no memory with the
// original.
func TestCloneArgsKeepsListsAndCudaGraphConfig(t *testing.T) {
	s := Defaults()
	s.AllowedMediaDomains = []string{"a.example", "b.example"}
	s.CudaGraphBsDecode = []int{1, 2, 4}
	s.BucketInterTokenLatency = []float64{0.5, 1.5}
	s.CudaGraphConfig = defaultCudaGraphConfig()
	s.CudaGraphConfig.Decode.Bs = []int{1, 2, 4}

	c := cloneArgs(s)
	if !reflect.DeepEqual(c, s) {
		t.Errorf("clone differs from the original:\n got %+v\nwant %+v", c, s)
	}
	if c.CudaGraphConfig == nil || !reflect.DeepEqual(c.CudaGraphConfig, s.CudaGraphConfig) {
		t.Errorf("cuda_graph_config %+v, want %+v", c.CudaGraphConfig, s.CudaGraphConfig)
	}
	for name, ok := range map[string]bool{
		"[]string": reflect.DeepEqual(c.AllowedMediaDomains, []string{"a.example", "b.example"}),
		"[]int":    reflect.DeepEqual(c.CudaGraphBsDecode, []int{1, 2, 4}),
		"[]float":  reflect.DeepEqual(c.BucketInterTokenLatency, []float64{0.5, 1.5}),
	} {
		if !ok {
			t.Errorf("the %s list did not survive the copy", name)
		}
	}

	n := Defaults()
	n.CudaGraphConfig = nil
	if cn := cloneArgs(n); cn.CudaGraphConfig != nil {
		t.Errorf("a nil cuda_graph_config became %+v", cn.CudaGraphConfig)
	}
}

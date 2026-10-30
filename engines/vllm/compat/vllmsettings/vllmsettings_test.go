// Moved from blis-latency-kernel/vllmsettings/vllmsettings_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package vllmsettings

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"

	"github.com/inference-sim/blis-schemas/kernel/enginesettings"
)

// populate sets every settable field of v, recursively, to a non-zero value, so a conversion that
// leaves a field at its zero value is caught. A pointer gets a fresh pointee; a slice gets one
// element; a string field gets its own name, so two fields swapped in the conversion read wrong.
func populate(t *testing.T, v reflect.Value, path string) {
	t.Helper()
	switch v.Kind() {
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if v.Type().Field(i).IsExported() {
				populate(t, v.Field(i), path+"."+v.Type().Field(i).Name)
			}
		}
	case reflect.Pointer:
		v.Set(reflect.New(v.Type().Elem()))
		populate(t, v.Elem(), path)
	case reflect.Slice:
		v.Set(reflect.MakeSlice(v.Type(), 1, 1))
		populate(t, v.Index(0), path+"[0]")
	case reflect.Map:
		v.Set(reflect.MakeMap(v.Type()))
		key, val := reflect.New(v.Type().Key()).Elem(), reflect.New(v.Type().Elem()).Elem()
		populate(t, key, path+"[k]")
		populate(t, val, path+"[v]")
		v.SetMapIndex(key, val)
	case reflect.String:
		name := path[strings.LastIndex(path, ".")+1:]
		v.SetString(strings.ToLower(name))
	case reflect.Bool:
		v.SetBool(true)
	case reflect.Int, reflect.Int64, reflect.Int32:
		v.SetInt(int64(7 + len(path)))
	case reflect.Float64, reflect.Float32:
		v.SetFloat(0.5 + float64(len(path)))
	default:
		t.Fatalf("%s: populate does not handle %s", path, v.Kind())
	}
}

// zeroes lists every leaf of v that is its zero value.
func zeroes(v reflect.Value, path string, out *[]string) {
	switch v.Kind() {
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			zeroes(v.Field(i), path+"."+v.Type().Field(i).Name, out)
		}
	case reflect.Pointer:
		if v.IsNil() {
			*out = append(*out, path)
			return
		}
		zeroes(v.Elem(), path, out)
	case reflect.Slice:
		if v.Len() == 0 {
			*out = append(*out, path)
		}
	default:
		if v.IsZero() {
			*out = append(*out, path)
		}
	}
}

// Every setting the kernel prices is filled from the vLLM engine block: Of over an
// engineconfig.Engine with every field stated leaves no field of enginesettings.Settings at its zero value. A field added to the
// kernel's settings and not to the conversion fails here rather than pricing as unstated.
func TestOfFillsEverySetting(t *testing.T) {
	var e engineconfig.Engine
	populate(t, reflect.ValueOf(&e).Elem(), "Engine")
	var missing []string
	zeroes(reflect.ValueOf(Of(e)), "Settings", &missing)
	if len(missing) > 0 {
		t.Errorf("Of leaves these settings unfilled from a fully stated engine block: %v", missing)
	}
}

// The conversion carries each value, not just some value: the kernel's settings name the same
// field in the block. A string carries its own field name (populate), so a swap of two string
// fields reads wrong here.
func TestOfCarriesEachValue(t *testing.T) {
	var e engineconfig.Engine
	populate(t, reflect.ValueOf(&e).Elem(), "Engine")
	s := Of(e)
	for name, got := range map[string]string{
		"Quantization": s.Quantization, "CacheDType": s.CacheDType, "MambaCacheDType": s.MambaCacheDType,
		"MambaCacheMode": s.MambaCacheMode, "MoEBackend": s.MoEBackend, "All2AllBackend": s.All2AllBackend,
		"AllReduceBackend": s.AllReduceBackend, "CUDAGraphMode": s.CUDAGraphMode,
		"DCPCommBackend": s.DCPCommBackend, "Frontend": s.Frontend, "Method": s.Speculative.Method,
	} {
		if got != strings.ToLower(name) {
			t.Errorf("%s = %q, want the block's %q", name, got, strings.ToLower(name))
		}
	}
	for name, p := range map[string][2]any{
		"BlockSize":               {s.BlockSize, e.BlockSize},
		"MaxNumSeqs":              {s.MaxNumSeqs, e.MaxNumSeqs},
		"MaxNumBatchedTokens":     {s.MaxNumBatchedTokens, e.MaxNumBatchedTokens},
		"GPUMemoryUtilization":    {s.GPUMemoryUtilization, e.GPUMemoryUtilization},
		"MaxModelLen":             {int(s.MaxModelLen), int(e.MaxModelLen)},
		"CPKVCacheInterleaveSize": {s.CPKVCacheInterleaveSize, e.CPKVCacheInterleaveSize},
		"NumSpecTokens":           {s.Speculative.NumSpecTokens, e.Speculative.NumSpecTokens},
		"NumRedundantExperts":     {s.EPLB.NumRedundantExperts, e.EPLB.NumRedundantExperts},
		"MaxCUDAGraphCaptureSize": {*s.MaxCUDAGraphCaptureSize, *e.MaxCUDAGraphCaptureSize},
		"CUDAGraphCaptureSizes":   {s.CUDAGraphCaptureSizes[0], e.CUDAGraphCaptureSizes[0]},
	} {
		if p[0] != p[1] {
			t.Errorf("%s = %v, want the block's %v", name, p[0], p[1])
		}
	}
}

// The settings are the kernel's own: changing the block after the conversion changes nothing.
func TestOfSharesNoMemoryWithTheBlock(t *testing.T) {
	var e engineconfig.Engine
	populate(t, reflect.ValueOf(&e).Elem(), "Engine")
	s := Of(e)
	*e.EnablePrefixCaching = false
	*e.MaxCUDAGraphCaptureSize = -1
	e.CUDAGraphCaptureSizes[0] = -1
	*e.EngramConfig.CPUOffload = false
	if !*s.EnablePrefixCaching || *s.MaxCUDAGraphCaptureSize == -1 || s.CUDAGraphCaptureSizes[0] == -1 ||
		!*s.EngramConfig.CPUOffload {
		t.Error("a write to the block after Of reached the kernel's settings")
	}
}

// OfPool decodes the pool's own block through blis-vllm, strictly, and prices exactly what Of
// gives for that decoding.
func TestOfPoolDecodesThePoolsBlock(t *testing.T) {
	var e engineconfig.Engine
	populate(t, reflect.ValueOf(&e).Elem(), "Engine")
	block, err := engineconfig.Encode(e)
	if err != nil {
		t.Fatal(err)
	}
	dep := &deployment.Deployment{Pools: []deployment.Pool{{}, {Engine: block}}}
	got, err := OfPool(dep, 1)
	if err != nil {
		t.Fatal(err)
	}
	want, err := Decode(dep.Pools[1])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, Of(want)) || !reflect.DeepEqual(want, e) {
		t.Errorf("OfPool = %+v, want Of of the pool's decoded block %+v", got, Of(e))
	}
	if empty, err := OfPool(dep, 0); err != nil || !reflect.DeepEqual(empty, enginesettings.Settings{}) {
		t.Errorf("an unstated block: OfPool = %+v, %v; want no settings and no error", empty, err)
	}
}

// speculative.enable_adaptive_verification reaches the kernel as stated, and an unstated flag
// is vLLM's default, off (vllm/config/speculative.py:551@5df63a72).
func TestOfCarriesAdaptiveVerification(t *testing.T) {
	on, off := true, false
	for _, c := range []struct {
		name string
		flag *bool
		want bool
	}{{"unstated", nil, false}, {"false", &off, false}, {"true", &on, true}} {
		e := engineconfig.Engine{Speculative: &engineconfig.Speculative{Method: "dspark", NumSpecTokens: 7,
			EnableAdaptiveVerification: c.flag}}
		if got := Of(e).Speculative.EnableAdaptiveVerification; got != c.want {
			t.Errorf("%s: EnableAdaptiveVerification = %v, want %v", c.name, got, c.want)
		}
	}
}

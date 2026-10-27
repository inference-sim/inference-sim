// Original to blis-vllm.

package build

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"

	latencykernel "github.com/inference-sim/blis-latency-kernel"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// The goldens under testdata/scenariovariant are the bytes inference-sim's harness wrote for these
// settings (writeScenarioVariant at inference-sim 01bbf6bf50c4b81bc91b0bfdb9b26f91e3155b60) before
// the variant moved here: ScenarioVariant writes them byte for byte.
func TestScenarioVariantIsTheHarnessesBytes(t *testing.T) {
	src := filepath.Join(kernelmodel.DefaultScenarios(), "gpt-oss-120b-h200-fp4-vllm-tp4.yaml")
	capture := 256
	for _, c := range []struct {
		golden string
		rs     RunSettings
		router *deployment.Router
	}{
		{"single.yaml", RunSettings{MaxNumSeqs: 37, MaxNumBatchedTokens: 4097, BlockSize: 32,
			PrefixCachingDisabled: true, MaxModelLen: 2304, GPUMemoryUtilization: 0.95, EnableExpertParallel: true}, nil},
		{"capture.yaml", RunSettings{MaxNumSeqs: 1024, MaxNumBatchedTokens: 16384, BlockSize: 16, MaxModelLen: 9472,
			GPUMemoryUtilization: 0.9, MaxCUDAGraphCaptureSize: &capture}, nil},
		{"router.yaml", RunSettings{MaxNumSeqs: 8, MaxNumBatchedTokens: 4096, BlockSize: 16, MaxModelLen: 2048,
			GPUMemoryUtilization: 0.9}, &deployment.Router{EndpointPickerConfig: "vllm-dp-lb.epp.yaml"}},
	} {
		got, err := ScenarioVariant(src, c.rs, c.router)
		if err != nil {
			t.Fatalf("%s: %v", c.golden, err)
		}
		want, err := os.ReadFile(filepath.Join("testdata", "scenariovariant", c.golden))
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != string(want) {
			t.Errorf("%s: ScenarioVariant wrote\n%s\nwant the harness's\n%s", c.golden, got, want)
		}
	}
}

// A scenario of more than one pool, and one that already states a router block, are refused with
// the messages the harness gave.
func TestScenarioVariantRefusesWhatTheHarnessRefused(t *testing.T) {
	rs := RunSettings{MaxNumSeqs: 8, MaxNumBatchedTokens: 4096, BlockSize: 16, MaxModelLen: 2048, GPUMemoryUtilization: 0.9}
	pd := filepath.Join("..", "..", "testdata", "scenarios", "minimax-m2.5-gb200-1p1d-dp2.yaml")
	if _, err := ScenarioVariant(pd, rs, nil); err == nil || err.Error() != pd+" states 2 pools; the harness scores one aggregated pool" {
		t.Errorf("a two-pool scenario: got %v, want the harness's refusal", err)
	}
	routed := filepath.Join("testdata", "scenariovariant", "router.yaml")
	router := &deployment.Router{EndpointPickerConfig: "vllm-dp-lb.epp.yaml"}
	if _, err := ScenarioVariant(routed, rs, router); err == nil ||
		err.Error() != routed+" states a router block; the harness routes every point with its own" {
		t.Errorf("a scenario stating a router block: got %v, want the harness's refusal", err)
	}
}

// harnessKeys are the YAML keys inference-sim's harness compares a run's passed settings by: the
// vLLM arguments its measured corpus records (the run's PassedRaw keys), whichever the schema has.
var harnessKeys = []string{
	"max_num_seqs", "max_num_batched_tokens", "block_size", "enable_prefix_caching", "max_model_len",
	"gpu_memory_utilization", "enable_expert_parallel", "max_cudagraph_capture_size", "cudagraph_mode",
	"quantization", "cache_dtype", "kv_cache_dtype", "tp", "pp", "dp", "async_scheduling",
	"cudagraph_capture_sizes", "long_prefill_token_threshold", "scheduling_policy", "speculative",
	"moe_backend", "all2all_backend", "data_parallel_size", "tensor_parallel_size", "model",
	"no_such_key",
}

// harnessSchemaValue is inference-sim's schemaValue as it stood at 01bbf6bf (gaps.go:406-430), the
// reflection PoolEngineValue replaced, kept here to compare against.
func harnessSchemaValue(pool deployment.Pool, key string) (any, bool) {
	for _, v := range []reflect.Value{reflect.ValueOf(pool.Engine), reflect.ValueOf(pool.Parallel)} {
		t := v.Type()
		for i := 0; i < t.NumField(); i++ {
			tag := strings.Split(t.Field(i).Tag.Get("yaml"), ",")[0]
			if tag != key {
				continue
			}
			f := v.Field(i)
			if f.Kind() == reflect.Pointer && f.IsNil() {
				return nil, true
			}
			if f.IsZero() {
				return nil, true
			}
			raw, err := json.Marshal(f.Interface())
			if err != nil {
				return nil, true
			}
			var out any
			if err := json.Unmarshal(raw, &out); err != nil {
				return string(raw), true
			}
			return out, true
		}
	}
	return nil, false
}

// PoolEngineValue answers every key the harness asks for exactly as its reflection did, on the
// variant a point runs (the pool with the run settings written on it), over every vendored
// scenario's pool.
func TestPoolEngineValueIsTheHarnessesLookup(t *testing.T) {
	capture := 512
	rs := RunSettings{MaxNumSeqs: 64, MaxNumBatchedTokens: 8192, BlockSize: 16, PrefixCachingDisabled: true,
		MaxModelLen: 4096, GPUMemoryUtilization: 0.92, EnableExpertParallel: true, MaxCUDAGraphCaptureSize: &capture}
	compared := 0
	for _, dir := range []string{kernelmodel.DefaultScenarios(), filepath.Join("..", "..", "testdata", "scenarios")} {
		files, _ := filepath.Glob(filepath.Join(dir, "*.yaml"))
		for _, path := range files {
			_, dep, err := latencykernel.LoadBundle(path)
			if err != nil {
				continue
			}
			for _, pool := range dep.Pools {
				variant := pool
				applyRunSettings(&variant, rs)
				for _, key := range harnessKeys {
					wantV, wantOK := harnessSchemaValue(variant, key)
					gotV, gotOK := PoolEngineValue(pool, &rs, key)
					if gotOK != wantOK || !reflect.DeepEqual(gotV, wantV) {
						t.Errorf("%s %s: PoolEngineValue = %v, %v; the harness's lookup = %v, %v",
							filepath.Base(path), key, gotV, gotOK, wantV, wantOK)
					}
					compared++
				}
			}
		}
	}
	if compared < 1000 {
		t.Fatalf("non-vacuity: %d lookups compared", compared)
	}
}

// StatedRunSettings is what each pool's scenario states, field for field, which is what
// inference-sim's harness read from Model.Engine and Model.Deployment (resolveAdmission at
// 01bbf6bf): the block's caps, block size, context bound and memory utilization as written, prefix
// caching off only where the block disables it, and the resolved expert parallelism.
func TestStatedRunSettingsAreTheScenarios(t *testing.T) {
	opened := 0
	for _, dir := range []string{kernelmodel.DefaultScenarios(), filepath.Join("..", "..", "testdata", "scenarios")} {
		files, _ := filepath.Glob(filepath.Join(dir, "*.yaml"))
		for _, path := range files {
			_, dep, err := latencykernel.LoadBundle(path)
			if err != nil {
				continue
			}
			r := kernelmodel.DefaultRepos()
			r.Scenarios = dir
			for pool := range dep.Pools {
				m, err := kernelmodel.OpenPool(filepath.Base(path), r, pool)
				if err != nil {
					continue
				}
				got, err := StatedRunSettings(m)
				e := dep.Pools[pool].Engine
				if e.BlockSize <= 0 || e.MaxNumSeqs <= 0 || e.MaxNumBatchedTokens <= 0 {
					if err == nil {
						t.Errorf("%s pool %d: no refusal for a block missing a required setting", path, pool)
					}
					continue
				}
				if err != nil {
					t.Fatalf("%s pool %d: %v", path, pool, err)
				}
				want := RunSettings{MaxNumSeqs: e.MaxNumSeqs, MaxNumBatchedTokens: e.MaxNumBatchedTokens,
					BlockSize: e.BlockSize, PrefixCachingDisabled: e.EnablePrefixCaching != nil && !*e.EnablePrefixCaching,
					MaxModelLen: int(e.MaxModelLen), GPUMemoryUtilization: e.GPUMemoryUtilization,
					EnableExpertParallel: m.Deployment().ExpertParallel}
				if !reflect.DeepEqual(got, want) {
					t.Errorf("%s pool %d: StatedRunSettings = %+v, want the scenario's %+v", path, pool, got, want)
				}
				opened++
			}
		}
	}
	if opened < 50 {
		t.Fatalf("non-vacuity: only %d pools compared", opened)
	}
}

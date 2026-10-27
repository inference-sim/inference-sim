// Moved from inference-sim/sim/kernelmodel/harness/blisrun.go at 01bbf6bf50c4b81bc91b0bfdb9b26f91e3155b60.
// Moved from inference-sim/sim/kernelmodel/harness/gaps.go at 01bbf6bf50c4b81bc91b0bfdb9b26f91e3155b60.
// (writeScenarioVariant, applyPointConfig and schemaValue; StatedRunSettings is original.)

package build

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/spec/deployment"

	latencykernel "github.com/inference-sim/blis-latency-kernel"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// RunSettings are the settings one measured point ran with that a scoring harness restates on a
// scenario before replaying it: the scheduler's caps, the block size, prefix caching, the context
// bound, the memory utilization, expert parallelism and, when the run passed one, the CUDA graph
// capture ceiling. The engine block is the engine model's, so the harness hands these over and
// this package writes them onto it (ScenarioVariant).
type RunSettings struct {
	MaxNumSeqs              int
	MaxNumBatchedTokens     int
	BlockSize               int
	PrefixCachingDisabled   bool
	MaxModelLen             int
	GPUMemoryUtilization    float64
	EnableExpertParallel    bool
	MaxCUDAGraphCaptureSize *int
}

// StatedRunSettings are the run settings m's pool states, which a harness restates a point with
// where the run recorded nothing better: the engine block's caps, block size, context bound and
// memory utilization as written, prefix caching off only where the block disables it, and the
// expert parallelism the kernel resolved. It errors as Model.Engine does, for a pool that states no
// block size, sequence cap or token budget. MaxCUDAGraphCaptureSize is nil: a variant keeps the
// scenario's own ceiling unless a run passed one.
func StatedRunSettings(m *kernelmodel.Model) (RunSettings, error) {
	e, err := m.Engine()
	if err != nil {
		return RunSettings{}, err
	}
	d := m.Deployment()
	return RunSettings{
		MaxNumSeqs:            e.MaxNumSeqs,
		MaxNumBatchedTokens:   e.MaxNumBatchedTokens,
		BlockSize:             e.BlockSize,
		PrefixCachingDisabled: d.PrefixCachingDisabled,
		MaxModelLen:           int(e.MaxModelLen),
		GPUMemoryUtilization:  e.GPUMemoryUtilization,
		EnableExpertParallel:  d.ExpertParallel,
	}, nil
}

// ScenarioVariant is the scenario file src with its single pool's run settings replaced by rs:
// max_num_seqs, max_num_batched_tokens, block_size, enable_prefix_caching, max_model_len,
// gpu_memory_utilization, parallel.enable_expert_parallel and, when rs states one,
// max_cudagraph_capture_size. Everything else is carried over as the committed file states it.
//
// The file is decoded and re-encoded through the blis-schemas types (the kernel's own loader)
// rather than edited as text, so the variant is a well-formed Scenario + Deployment pair by
// construction and a field the schema does not know is refused rather than copied. Prefix caching
// is written in both directions, so the variant states the point's resolved value rather than
// leaving it to the tri-state's default.
//
// A scenario stating more than one pool is refused: the resolved settings describe one engine. A
// non-nil router becomes the variant's router block; a scenario that already states one is
// refused rather than overwritten.
func ScenarioVariant(src string, rs RunSettings, router *deployment.Router) ([]byte, error) {
	sc, dep, err := latencykernel.LoadBundle(src)
	if err != nil {
		return nil, err
	}
	if len(dep.Pools) != 1 {
		return nil, fmt.Errorf("%s states %d pools; the harness scores one aggregated pool", src, len(dep.Pools))
	}
	if err := applyRunSettings(&dep.Pools[0], rs); err != nil {
		return nil, fmt.Errorf("%s: %w", src, err)
	}
	if router != nil {
		if dep.Router != nil {
			return nil, fmt.Errorf("%s states a router block; the harness routes every point with its own", src)
		}
		dep.Router = router
	}
	var buf bytes.Buffer
	enc := yaml.NewEncoder(&buf)
	enc.SetIndent(2)
	if err := enc.Encode(sc); err != nil {
		return nil, fmt.Errorf("encoding scenario: %w", err)
	}
	if err := enc.Encode(dep); err != nil {
		return nil, fmt.Errorf("encoding deployment: %w", err)
	}
	if err := enc.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// applyRunSettings writes a point's run settings onto a pool: its engine block decoded as vLLM's
// engine configuration, the settings written, and the block encoded back.
func applyRunSettings(pool *deployment.Pool, rs RunSettings) error {
	e, err := kernelmodel.DecodeEngine(*pool)
	if err != nil {
		return err
	}
	writeRunSettings(&e, rs)
	pool.Parallel.EnableExpertParallel = rs.EnableExpertParallel
	return kernelmodel.SetEngine(pool, e)
}

// writeRunSettings writes the run settings that live in the engine block.
func writeRunSettings(e *engineconfig.Engine, rs RunSettings) {
	e.MaxNumSeqs = rs.MaxNumSeqs
	e.MaxNumBatchedTokens = rs.MaxNumBatchedTokens
	e.BlockSize = rs.BlockSize
	enabled := !rs.PrefixCachingDisabled
	e.EnablePrefixCaching = &enabled
	e.MaxModelLen = engineconfig.MaxModelLen(rs.MaxModelLen)
	e.GPUMemoryUtilization = rs.GPUMemoryUtilization
	if rs.MaxCUDAGraphCaptureSize != nil {
		c := *rs.MaxCUDAGraphCaptureSize
		e.MaxCUDAGraphCaptureSize = &c
	}
}

// PoolEngineValue is the value pool states for a setting named by its engine-block or parallel
// YAML key, decoded the way a JSON measurement is (so the two compare), after rs is written onto
// it, and whether either block has such a field at all. An unstated field reads as nil.
//
// A harness compares the settings a run passed with what its variant states through this, so it
// names no engine field itself.
func PoolEngineValue(pool deployment.Pool, rs *RunSettings, key string) (any, bool) {
	e, err := kernelmodel.DecodeEngine(pool)
	if err != nil {
		return nil, false
	}
	if rs != nil {
		writeRunSettings(&e, *rs)
		pool.Parallel.EnableExpertParallel = rs.EnableExpertParallel
	}
	for _, v := range []reflect.Value{reflect.ValueOf(e), reflect.ValueOf(pool.Parallel)} {
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
				return nil, true // omitempty: unstated
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

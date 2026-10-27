// enginesettings.go loads the engine settings each measured run was launched with, so a
// simulated point is configured the way the point it is scored against was configured.
//
// The accuracy snapshot does not publish them. An earlier revision of this experiment wrote
// one assumed value per setting and reasoned that a constant cancels in a ratio across
// concurrency. That holds for step time and fails for time to first token, because the
// sequence cap decides whether a request waits at all.
//
// Nor is there a rule to apply instead. Across the 238 scored points the runs passed
// max_num_seqs equal to the client concurrency on 92, a fixed value on 69, and nothing at all
// on 77. A lookup is the only mechanism that is right everywhere, and
// blis-latency-kernel/scripts/extract_inferencex_engine_settings.py builds it from the same
// InferenceX run each sweep's absolute latencies came from.
//
// A setting the run did not pass is ABSENT here rather than filled in, which is the whole
// point of the distinction: the caller applies the engine's own resolution
// (kernelmodel.ResolveVLLMBatchDefaults) and can report which of the two it used.
package harness

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/inference-sim/blis-vllm/compat/build"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	_ "github.com/inference-sim/inference-sim/sim/vendored" // the adapter's vendoring root
)

// PassedSettings is what a run's command line carried. A nil field was not passed.
type PassedSettings struct {
	MaxNumSeqs           *int     `json:"max_num_seqs"`
	MaxNumBatchedTokens  *int     `json:"max_num_batched_tokens"`
	MaxModelLen          *int     `json:"max_model_len"`
	BlockSize            *int     `json:"block_size"`
	EnablePrefixCaching  *bool    `json:"enable_prefix_caching"`
	GPUMemoryUtilization *float64 `json:"gpu_memory_utilization"`
	EnableExpertParallel *bool    `json:"enable_expert_parallel"`
	// MaxCUDAGraphCaptureSize is compilation_config.max_cudagraph_capture_size (blis-schemas#72).
	MaxCUDAGraphCaptureSize *int `json:"max_cudagraph_capture_size"`
}

// ResolvedSettings is what the engine reported after resolving, which is a measurement of the
// running engine rather than a request.
type ResolvedSettings struct {
	// GPUKVTokens is the KV pool the engine actually had, in tokens. This project otherwise
	// derives that quantity from the kernel's memory methods; here it is measured.
	GPUKVTokens    int     `json:"resolved_gpu_kv_tokens"`
	MaxConcurrency float64 `json:"resolved_max_concurrency"`
	EngineVersion  string  `json:"engine_version"`
}

// PointSettings pairs the two for one concurrency.
type PointSettings struct {
	Passed   PassedSettings   `json:"passed"`
	Resolved ResolvedSettings `json:"resolved"`

	// PassedRaw is every setting the run's command line carried, keyed as the extraction
	// wrote it -- including the ones PassedSettings has no field for. The coverage report
	// reads it, so a setting the run used is reported as a gap from the data rather than
	// from a list of what this package happens to know.
	PassedRaw map[string]json.RawMessage `json:"-"`
}

// UnmarshalJSON decodes the typed fields and keeps the raw passed map beside them.
func (p *PointSettings) UnmarshalJSON(b []byte) error {
	type plain PointSettings // drops this method, so the decode below does not recurse
	var q plain
	if err := json.Unmarshal(b, &q); err != nil {
		return err
	}
	var raw struct {
		Passed map[string]json.RawMessage `json:"passed"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	*p = PointSettings(q)
	p.PassedRaw = raw.Passed
	return nil
}

// SweepSettings is one (scenario, label) with its per-concurrency record.
type SweepSettings struct {
	Scenario      string                    `json:"scenario"`
	Label         string                    `json:"label"`
	Framework     string                    `json:"framework"`
	GPU           string                    `json:"gpu"`
	RunDate       string                    `json:"source_run_date"`
	ConfigID      int                       `json:"source_config_id"`
	ByConcurrency map[string]*PointSettings `json:"by_concurrency"`
}

// EngineSettingSet indexes the measured settings by sweep.
type EngineSettingSet struct {
	Source       string          `json:"source"`
	SourceURL    string          `json:"source_url"`
	ReleaseTag   string          `json:"release_tag"`
	RunSelection string          `json:"run_selection"`
	Settings     []SweepSettings `json:"settings"`

	byKey map[string]*SweepSettings
}

// LoadEngineSettings reads the extracted settings. A missing file is an error rather than a
// silent fallback: a run configured from assumed values while reporting that it used measured
// ones would be the defect this file exists to remove.
func LoadEngineSettings(path string) (*EngineSettingSet, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("engine settings: %w", err)
	}
	var set EngineSettingSet
	if err := json.Unmarshal(data, &set); err != nil {
		return nil, fmt.Errorf("engine settings: %w", err)
	}
	if len(set.Settings) == 0 {
		return nil, fmt.Errorf("engine settings at %s carry no sweeps", path)
	}
	set.byKey = set.index()
	return &set, nil
}

func (s *EngineSettingSet) index() map[string]*SweepSettings {
	m := make(map[string]*SweepSettings, len(s.Settings))
	for i := range s.Settings {
		m[absKey(s.Settings[i].Scenario, s.Settings[i].Label)] = &s.Settings[i]
	}
	return m
}

// For returns the settings for a sweep, or nil when the extraction has none -- which is the
// case for every non-vLLM sweep, because no other engine prints vLLM's args line.
func (s *EngineSettingSet) For(sw Sweep) *SweepSettings {
	if s == nil {
		return nil
	}
	if s.byKey == nil {
		s.byKey = s.index()
	}
	return s.byKey[absKey(sw.Scenario, sw.Label)]
}

// At returns one concurrency's settings, or nil.
func (s *SweepSettings) At(concurrency int) *PointSettings {
	if s == nil {
		return nil
	}
	return s.ByConcurrency[strconv.Itoa(concurrency)]
}

// SettingSource names where a value came from, so a report can distinguish a measurement from
// a default rather than presenting both as the same kind of number.
type SettingSource string

const (
	// SourceMeasured is the run's own command line, read from its engine log.
	SourceMeasured SettingSource = "measured"
	// SourceResolved is vLLM's device-memory resolution, applied where the run passed
	// nothing and the engine therefore resolved it too.
	SourceResolved SettingSource = "resolved"
	// SourceScenario is the scenario file's value, used only where no measurement exists --
	// every non-vLLM sweep, because no other engine prints vLLM's args line.
	SourceScenario SettingSource = "scenario"
)

// PointConfig is the resolved configuration one point runs with, with the provenance of each
// field it decides.
//
// The first four decide admission directly. The last three decide it through the KV pool and
// the window (gpu_memory_utilization sizes the pool, max_model_len bounds a request and must
// fit the pool) or set step time (expert parallelism changes which GEMM a routed layer runs).
// Every one is written into the point's scenario variant, so a run setting the engine was
// launched with is the setting it is simulated with (#1920).
type PointConfig struct {
	MaxNumSeqs            int
	MaxNumBatchedTokens   int
	BlockSize             int
	PrefixCachingDisabled bool
	MaxModelLen           int
	GPUMemoryUtilization  float64
	EnableExpertParallel  bool
	// MaxCUDAGraphCaptureSize is the capture ceiling the run passed, or nil to keep the
	// scenario's (blis-schemas#72).
	MaxCUDAGraphCaptureSize *int

	SeqsFrom   SettingSource
	TokensFrom SettingSource
	BlockFrom  SettingSource
	PrefixFrom SettingSource
	WindowFrom SettingSource
	MemoryFrom SettingSource
	EPFrom     SettingSource

	// MeasuredGPUKVTokens is the KV pool the engine reported, in tokens, or zero when no
	// measurement exists. Recorded for comparison against the kernel's derived budget; it
	// does not size the pool, because changing two things at once would make the result
	// unattributable.
	MeasuredGPUKVTokens int
	EngineVersion       string
}

// runSettings is the point's settings as blis-vllm writes them onto a scenario.
func (pc PointConfig) runSettings() build.RunSettings {
	return build.RunSettings{
		MaxNumSeqs: pc.MaxNumSeqs, MaxNumBatchedTokens: pc.MaxNumBatchedTokens, BlockSize: pc.BlockSize,
		PrefixCachingDisabled: pc.PrefixCachingDisabled, MaxModelLen: pc.MaxModelLen,
		GPUMemoryUtilization: pc.GPUMemoryUtilization, EnableExpertParallel: pc.EnableExpertParallel,
		MaxCUDAGraphCaptureSize: pc.MaxCUDAGraphCaptureSize,
	}
}

// vLLM's CacheConfig.gpu_memory_utilization default, which changed in v0.20.0
// (vllm-project/vllm 96a85c5750, #38284, "Enable CUDAGraph memory profiling by default"):
// Field(default=0.9) in vllm/config/cache.py at v0.15.0 through v0.20.0rc1, Field(default=0.92)
// from v0.20.0. The measured runs report engine versions on both sides of it.
const (
	vllmGPUMemUtilBefore020 = 0.9
	vllmGPUMemUtilFrom020   = 0.92
)

// vllmDefaultGPUMemUtil is the default an engine of this version resolved, or false when the
// version is unknown or unparseable -- then no default is claimed.
func vllmDefaultGPUMemUtil(version string) (float64, bool) {
	parts := strings.SplitN(version, ".", 3)
	if len(parts) < 2 {
		return 0, false
	}
	major, err1 := strconv.Atoi(parts[0])
	minor, err2 := strconv.Atoi(parts[1])
	if err1 != nil || err2 != nil {
		return 0, false
	}
	if major > 0 || minor >= 20 {
		return vllmGPUMemUtilFrom020, true
	}
	return vllmGPUMemUtilBefore020, true
}

// resolveAdmission picks each setting from the strongest source available for this point.
//
// Measured first, because that is what the engine ran. vLLM's own resolution second, for a
// setting the run did not pass -- the engine resolved it the same way, so reproducing that is
// a measurement of the engine's behaviour rather than a guess. The scenario last, which in
// practice means the non-vLLM sweeps where no log exists.
//
// stated is what the scenario's pool states (build.StatedRunSettings): blis-vllm reads the
// engine block, so this names none of its fields.
func resolveAdmission(sw Sweep, concurrency int, stated build.RunSettings,
	dep kernelmodel.Deployment, set *EngineSettingSet) PointConfig {
	a := PointConfig{
		MaxNumSeqs:          stated.MaxNumSeqs,
		MaxNumBatchedTokens: stated.MaxNumBatchedTokens,
		BlockSize:           stated.BlockSize,
		// The scenario's own setting, so a sweep with no measurement and no vLLM resolution
		// (a non-vLLM engine) runs what its scenario states rather than caching by default.
		PrefixCachingDisabled: stated.PrefixCachingDisabled,
		MaxModelLen:           stated.MaxModelLen,
		GPUMemoryUtilization:  stated.GPUMemoryUtilization,
		EnableExpertParallel:  stated.EnableExpertParallel,
		SeqsFrom:              SourceScenario,
		TokensFrom:            SourceScenario,
		BlockFrom:             SourceScenario,
		PrefixFrom:            SourceScenario,
		WindowFrom:            SourceScenario,
		MemoryFrom:            SourceScenario,
		EPFrom:                SourceScenario,
	}
	// vLLM's resolution, which applies to any setting the run did not pass -- whether the
	// run passed SOME settings and not others, or carries no log at all. It is gated on the
	// framework: resolving an sglang or trtllm deployment with vLLM's device-memory defaults
	// would substitute one engine's behaviour for another's.
	vllm := sw.Framework == "vllm"
	fallback := kernelmodel.ResolveVLLMBatchDefaults(dep.DeviceMemoryGiB, dep.Hardware)
	p := set.For(sw).At(concurrency)
	version := ""
	if p != nil {
		version = p.Resolved.EngineVersion
	}

	// What a setting falls back to when the run did not pass it. Factored out because it
	// applies in two places that are easy to let drift: a point with a log that omits one
	// field, and a point with no log at all.
	applyVLLMDefaults := func() {
		if !vllm {
			return
		}
		if fallback.MaxNumSeqs > 0 {
			a.MaxNumSeqs, a.SeqsFrom = fallback.MaxNumSeqs, SourceResolved
		}
		if fallback.MaxNumBatchedTokens > 0 {
			a.MaxNumBatchedTokens, a.TokensFrom = fallback.MaxNumBatchedTokens, SourceResolved
		}
		// vLLM's CacheConfig.DEFAULT_BLOCK_SIZE is 16, which is what the scenarios already
		// carry, so an unpassed block size needs no substitution. Recorded as resolved
		// rather than scenario because the agreement is a fact about vLLM, not a coincidence.
		a.BlockFrom = SourceResolved
		// vLLM caches unless told not to.
		a.PrefixCachingDisabled, a.PrefixFrom = false, SourceResolved
		// ParallelConfig.enable_expert_parallel defaults to False (vllm/config/parallel.py,
		// unchanged from v0.11.0 through v0.31.0), so a run that did not pass it ran
		// without expert parallelism whatever the scenario file states.
		a.EnableExpertParallel, a.EPFrom = false, SourceResolved
		// gpu_memory_utilization's default depends on the engine version, which only a
		// log records; without one the scenario's value stands and says so.
		if v, ok := vllmDefaultGPUMemUtil(version); ok {
			a.GPUMemoryUtilization, a.MemoryFrom = v, SourceResolved
		}
		// max_model_len unpassed resolves from the model's max_position_embeddings, which
		// no document this harness reads carries, so the scenario's window stands.
	}

	if p == nil {
		// No log for this point at all. Llama-3.1-70B is the case: no row for it in the
		// InferenceX dump carries a server_log_id, under either framework, so its settings
		// were never captured.
		//
		// The engine still resolved them, and reproducing that resolution is closer to the
		// truth than the value this project assumed: on h200 vLLM resolves max_num_seqs to
		// 1024 where the scenario files carry 256, and the sequence cap decides whether a
		// request waits. Falling back to the scenario here -- which an earlier revision did
		// -- simulated a deployment nobody ran, and reported the provenance as "scenario"
		// while every other point in the table read "measured".
		//
		// This is still NOT a measurement, and the provenance says "resolved" so a reader
		// can separate the two. A comparison that mixes the two kinds in one average is the
		// thing that distinction exists to prevent.
		applyVLLMDefaults()
		return a
	}
	a.MeasuredGPUKVTokens = p.Resolved.GPUKVTokens
	a.EngineVersion = p.Resolved.EngineVersion

	// Start from what the engine would resolve, then let each measured value override it.
	// Ordering matters: a field the run passed must win, and a field it did not pass must
	// read "resolved" rather than inheriting the scenario's assumption.
	applyVLLMDefaults()
	if v := p.Passed.MaxNumSeqs; v != nil {
		a.MaxNumSeqs, a.SeqsFrom = *v, SourceMeasured
	}
	if v := p.Passed.MaxNumBatchedTokens; v != nil {
		a.MaxNumBatchedTokens, a.TokensFrom = *v, SourceMeasured
	}
	if v := p.Passed.BlockSize; v != nil {
		a.BlockSize, a.BlockFrom = *v, SourceMeasured
	}
	if v := p.Passed.EnablePrefixCaching; v != nil {
		a.PrefixCachingDisabled, a.PrefixFrom = !*v, SourceMeasured
	}
	if v := p.Passed.MaxModelLen; v != nil {
		a.MaxModelLen, a.WindowFrom = *v, SourceMeasured
	}
	if v := p.Passed.GPUMemoryUtilization; v != nil {
		a.GPUMemoryUtilization, a.MemoryFrom = *v, SourceMeasured
	}
	if v := p.Passed.EnableExpertParallel; v != nil {
		a.EnableExpertParallel, a.EPFrom = *v, SourceMeasured
	}
	if v := p.Passed.MaxCUDAGraphCaptureSize; v != nil {
		c := *v
		a.MaxCUDAGraphCaptureSize = &c
	}
	return a
}

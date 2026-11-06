// Package enginesettings is the engine block of one pool as a latency kernel prices it: the
// settings that change a step's time or the pool's occupancy, and nothing else.
//
// A kernel does not open the engine block itself. A scenario's engine: block is the engine
// model's to decode (deployment.EngineBlock holds it opaque), so the kernel's caller decodes it
// and hands the kernel these settings, as it hands the kernel the engine's rules
// (rules.EngineRules). An engine model fills every field here from its own engine block.
//
// The types are plain values: an unset pointer is "the engine decides", as in the block it was
// decoded from, and the kernel resolves and records what the engine decided.
//
// Moved verbatim, but for this comment, from blis-latency-kernel/enginesettings (blis-latency-
// kernel#82 at f556c19), so that the kernel and an engine model share it without importing each
// other.
package enginesettings

import "slices"

// Settings is one pool's engine settings, in the engine block's own terms.
type Settings struct {
	Quantization    string
	CacheDType      string
	MambaCacheDType string
	MambaCacheMode  string
	MoEBackend      string

	BlockSize           int
	MaxNumSeqs          int
	MaxNumBatchedTokens int
	// GPUMemoryUtilization is the fraction of each GPU's memory the engine may take, as the
	// block states it; zero when unstated (the engine's default). Under pipeline parallelism
	// every worker sizes its KV pool from its own share of it and the engine keeps the
	// smallest pool (vLLM's get_kv_cache_configs, vllm/v1/core/kv_cache_utils.py:2531-2680 at
	// v0.32.0), so which stage binds depends on it.
	GPUMemoryUtilization float64
	// MaxModelLen is the context bound as stated: a count, auto, or unstated (MaxModelLen).
	MaxModelLen MaxModelLen
	// EnablePrefixCaching is nil when unstated (the engine's default, on).
	EnablePrefixCaching *bool
	// AsyncScheduling is nil when unstated (the engine decides).
	AsyncScheduling *bool
	// DisableCascadeAttn is nil when unstated.
	DisableCascadeAttn *bool

	All2AllBackend         string
	AllReduceBackend       string
	DisableCustomAllReduce *bool

	CUDAGraphMode           string
	MaxCUDAGraphCaptureSize *int
	CUDAGraphCaptureSizes   []int

	DCPCommBackend          string
	DCPQReplicate           *bool
	CPKVCacheInterleaveSize int

	// Frontend is the API server: "" or "python" (the default), or "rust".
	Frontend string

	DBO          *DBO
	EPLB         *EPLB
	Speculative  *Speculative
	EngramConfig *EngramConfig
}

// FrontendRust is the Rust API-server frontend.
const FrontendRust = "rust"

// RustFrontend reports whether the pool serves through the Rust frontend.
func (s Settings) RustFrontend() bool { return s.Frontend == FrontendRust }

// Speculating reports whether the pool drafts tokens.
func (s Settings) Speculating() bool { return s.Speculative != nil && s.Speculative.NumSpecTokens > 0 }

// DBO is dual-batch overlap.
type DBO struct{ Enabled bool }

// EPLB is expert-parallel load balancing.
type EPLB struct {
	Enabled             bool
	NumRedundantExperts int
}

// Speculative is the drafter: its method and how many tokens it drafts a step.
type Speculative struct {
	Method        string
	NumSpecTokens int
	// DraftModel names the catalog model whose graph is the drafter, where the drafter is
	// a checkpoint of its own (the engine's speculative_config.model), as Muse Spark's
	// DSpark drafter and an EAGLE3 head are. Empty means the target graph's own
	// speculator.
	DraftModel string
	// EnableAdaptiveVerification is vLLM's speculative_config.enable_adaptive_verification: the
	// verifier runs a confidence-chosen draft budget rather than every scheduled draft
	// (vllm/config/speculative.py:551-553@5df63a72). DSpark only. A plain bool, so a copy of
	// the struct is a deep copy (Clone).
	EnableAdaptiveVerification bool
}

// EngramConfig is where an Engram table lives. Each nil field is the engine's default.
type EngramConfig struct {
	CPUOffload     *bool
	DPSharedMemory *bool
	UseTHP         *bool
}

// MaxModelLen is a context bound as an engine block states it: a positive count, auto
// (MaxModelLenAuto, fitted to the KV memory) or zero (unstated, derived from the model).
type MaxModelLen int

// MaxModelLenAuto is a bound fitted to the KV memory at startup.
const MaxModelLenAuto MaxModelLen = -1

// Auto reports whether the bound is auto.
func (m MaxModelLen) Auto() bool { return m == MaxModelLenAuto }

// Stated returns the bound and true when it is a count.
func (m MaxModelLen) Stated() (int, bool) {
	if m > 0 {
		return int(m), true
	}
	return 0, false
}

// Clone is a deep copy: a kernel keeps its own, so a caller editing the settings it passed
// does not change a kernel already built from them.
func (s Settings) Clone() Settings {
	for _, b := range []**bool{&s.EnablePrefixCaching, &s.AsyncScheduling, &s.DisableCascadeAttn,
		&s.DisableCustomAllReduce, &s.DCPQReplicate} {
		if *b != nil {
			v := **b
			*b = &v
		}
	}
	if s.MaxCUDAGraphCaptureSize != nil {
		v := *s.MaxCUDAGraphCaptureSize
		s.MaxCUDAGraphCaptureSize = &v
	}
	s.CUDAGraphCaptureSizes = slices.Clone(s.CUDAGraphCaptureSizes)
	if s.DBO != nil {
		v := *s.DBO
		s.DBO = &v
	}
	if s.EPLB != nil {
		v := *s.EPLB
		s.EPLB = &v
	}
	if s.Speculative != nil {
		v := *s.Speculative
		s.Speculative = &v
	}
	if s.EngramConfig != nil {
		v := *s.EngramConfig
		for _, b := range []**bool{&v.CPUOffload, &v.DPSharedMemory, &v.UseTHP} {
			if *b != nil {
				c := **b
				*b = &c
			}
		}
		s.EngramConfig = &v
	}
	return s
}

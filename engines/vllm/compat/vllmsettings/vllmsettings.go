// Moved from blis-latency-kernel/vllmsettings/vllmsettings.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

// Package vllmsettings converts a vLLM engine block into the settings the cost-model lowering
// resolves (package lowering): the settings that change a step's time or the pool's occupancy,
// and nothing else. It is the one copy of that conversion: the kernel tests and blis-vllm's
// kernel model use it.
//
// The settings were the substrate's typed engine-settings carrier until engine lowering took the
// resolution over (blis-substrate#58); they are vLLM's, so they live here, and the cost model
// receives only the plan the lowering resolves them to.
package vllmsettings

import (
	"slices"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// Settings is one pool's vLLM settings, in the engine block's own terms: an unset pointer is "the
// engine decides", as in the block it was decoded from, and the lowering resolves it.
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
	// LanguageModelOnly is nil when unstated (vLLM's default, false: a multimodal model keeps its
	// towers resident).
	LanguageModelOnly *bool

	All2AllBackend         string
	AllReduceBackend       string
	DisableCustomAllReduce *bool

	CUDAGraphMode string
	// EnforceEager is model_config.enforce_eager: no CUDA graph, whatever CUDAGraphMode states.
	EnforceEager            bool
	MaxCUDAGraphCaptureSize *int
	CUDAGraphCaptureSizes   []int

	DCPCommBackend          string
	DCPQReplicate           *bool
	CPKVCacheInterleaveSize int

	// Frontend is the API server: "" or "python" (the default), or "rust".
	Frontend string

	DBO         *DBO
	EPLB        *EPLB
	Speculative *Speculative
	// ServedWeights is the layout the engine serves the checkpoint in, nil for as stored.
	ServedWeights *plan.WeightLayout
	EngramConfig  *EngramConfig
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
	// DraftModel names the catalog model whose graph is the drafter, where the drafter is a
	// checkpoint of its own (the engine's speculative_config.model). Empty means the target
	// graph's own speculator.
	DraftModel string
	// EnableAdaptiveVerification is vLLM's speculative_config.enable_adaptive_verification: the
	// verifier runs a confidence-chosen draft budget rather than every scheduled draft
	// (vllm/config/speculative.py:551-553@5df63a72). A plain bool, so a copy of the struct is a
	// deep copy (Clone).
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

// Clone is a deep copy: a caller editing the settings it passed does not change a lowering
// already made from them.
func (s Settings) Clone() Settings {
	for _, b := range []**bool{&s.EnablePrefixCaching, &s.AsyncScheduling, &s.DisableCascadeAttn,
		&s.DisableCustomAllReduce, &s.DCPQReplicate, &s.LanguageModelOnly} {
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
	if s.ServedWeights != nil {
		v := *s.ServedWeights
		s.ServedWeights = &v
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

// Of is the settings of a vLLM engine block, as blis-vllm decodes it (engineconfig.Decode).
func Of(e engineconfig.Engine) Settings {
	s := Settings{
		Quantization:            e.Quantization,
		CacheDType:              e.CacheDType,
		MambaCacheDType:         e.MambaCacheDType,
		MambaCacheMode:          e.MambaCacheMode,
		MoEBackend:              e.MoEBackend,
		BlockSize:               e.BlockSize,
		MaxNumSeqs:              e.MaxNumSeqs,
		MaxNumBatchedTokens:     e.MaxNumBatchedTokens,
		GPUMemoryUtilization:    e.GPUMemoryUtilization,
		MaxModelLen:             MaxModelLen(e.MaxModelLen),
		EnablePrefixCaching:     e.EnablePrefixCaching,
		AsyncScheduling:         e.AsyncScheduling,
		DisableCascadeAttn:      e.DisableCascadeAttn,
		LanguageModelOnly:       e.LanguageModelOnly,
		All2AllBackend:          e.All2AllBackend,
		AllReduceBackend:        e.AllReduceBackend,
		DisableCustomAllReduce:  e.DisableCustomAllReduce,
		CUDAGraphMode:           e.CUDAGraphMode,
		EnforceEager:            e.EnforceEager,
		MaxCUDAGraphCaptureSize: e.MaxCUDAGraphCaptureSize,
		CUDAGraphCaptureSizes:   e.CUDAGraphCaptureSizes,
		DCPCommBackend:          e.DCPCommBackend,
		DCPQReplicate:           e.DCPQReplicate,
		CPKVCacheInterleaveSize: e.CPKVCacheInterleaveSize,
		Frontend:                string(e.Frontend),
	}
	if e.DBO != nil {
		s.DBO = &DBO{Enabled: e.DBO.Enabled}
	}
	if e.EPLB != nil {
		s.EPLB = &EPLB{Enabled: e.EPLB.Enabled, NumRedundantExperts: e.EPLB.NumRedundantExperts}
	}
	if e.Speculative != nil {
		s.Speculative = &Speculative{Method: e.Speculative.Method,
			NumSpecTokens: e.Speculative.NumSpecTokens, DraftModel: e.Speculative.DraftModel,
			// Tri-state in the engine block, nil being vLLM's default false
			// (vllm/config/speculative.py:551@5df63a72).
			EnableAdaptiveVerification: e.Speculative.EnableAdaptiveVerification != nil &&
				*e.Speculative.EnableAdaptiveVerification}
	}
	if e.EngramConfig != nil {
		s.EngramConfig = &EngramConfig{CPUOffload: e.EngramConfig.CPUOffload,
			DPSharedMemory: e.EngramConfig.DPSharedMemory, UseTHP: e.EngramConfig.UseTHP}
	}
	return s.Clone()
}

// OfPool is Of for one pool of a deployment: the pool's engine block decoded by blis-vllm,
// strictly.
func OfPool(dep *deployment.Deployment, pool int) (Settings, error) {
	e, err := Decode(dep.Pools[pool])
	if err != nil {
		return Settings{}, err
	}
	return Of(e), nil
}

// Decode is a pool's engine block as blis-vllm decodes it (engineconfig.Decode).
func Decode(pool deployment.Pool) (engineconfig.Engine, error) {
	return engineconfig.Decode(pool.Engine)
}

// DraftPasses states how vLLM runs a drafter's passes, in the plan's neutral words, for the kernel
// to price (latencykernel.Inputs.DraftFirstPassRows, DraftPassRequests).
// EAGLE, EAGLE3 and MTP share one speculator class
// (vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py): its first pass runs over every
// token the target scheduled this step, prefill tokens included (`_prefill` over `num_tokens`,
// :432-478), and the head and later passes run for every request of the step (`propose`, :165-364).
// DFlash, and DSpark which subclasses it, are parallel-block drafters with no first pass to state;
// their query pass, and DSpark's sequential head, run for every request of the input batch, a
// request mid-prompt included (num_query_tokens = num_reqs * num_query_per_req,
// vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:463-467 at v0.33.0; dspark/speculator.py:40).
// A method that does neither, or no speculation, is left unstated, so the kernel refuses a drafter
// this engine model has not read (draft_passes_unstated).
func DraftPasses(e engineconfig.Engine) (plan.FirstPassRows, plan.PassRequests) {
	if e.Speculative == nil {
		return "", ""
	}
	switch e.Speculative.Method {
	case "eagle", "eagle3", "mtp":
		return plan.FirstPassScheduled, plan.PassRequestsAll
	case "dflash", "dspark":
		return "", plan.PassRequestsAll
	}
	return "", ""
}

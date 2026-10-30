// Moved from blis-latency-kernel/vllmsettings/vllmsettings.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

// Package vllmsettings converts a vLLM engine block into the engine settings the kernel prices
// (blis-latency-kernel/enginesettings). It is the one copy of that conversion: the kernel's own
// commands, harness and tests use it, and so does blis-vllm when it opens a kernel.
//
// It sits outside the kernel library on purpose. The library takes its engine settings from its
// caller, as it takes the engine's rules, so it never imports an engine model; this package is
// where a vLLM caller does the decoding.
package vllmsettings

import (
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"

	"github.com/inference-sim/blis-schemas/kernel/enginesettings"
)

// Of is the kernel's engine settings for a vLLM engine block, as blis-vllm decodes it
// (engineconfig.Decode).
func Of(e engineconfig.Engine) enginesettings.Settings {
	s := enginesettings.Settings{
		Quantization:            e.Quantization,
		CacheDType:              e.CacheDType,
		MambaCacheDType:         e.MambaCacheDType,
		MambaCacheMode:          e.MambaCacheMode,
		MoEBackend:              e.MoEBackend,
		BlockSize:               e.BlockSize,
		MaxNumSeqs:              e.MaxNumSeqs,
		MaxNumBatchedTokens:     e.MaxNumBatchedTokens,
		GPUMemoryUtilization:    e.GPUMemoryUtilization,
		MaxModelLen:             enginesettings.MaxModelLen(e.MaxModelLen),
		EnablePrefixCaching:     e.EnablePrefixCaching,
		AsyncScheduling:         e.AsyncScheduling,
		DisableCascadeAttn:      e.DisableCascadeAttn,
		All2AllBackend:          e.All2AllBackend,
		AllReduceBackend:        e.AllReduceBackend,
		DisableCustomAllReduce:  e.DisableCustomAllReduce,
		CUDAGraphMode:           e.CUDAGraphMode,
		MaxCUDAGraphCaptureSize: e.MaxCUDAGraphCaptureSize,
		CUDAGraphCaptureSizes:   e.CUDAGraphCaptureSizes,
		DCPCommBackend:          e.DCPCommBackend,
		DCPQReplicate:           e.DCPQReplicate,
		CPKVCacheInterleaveSize: e.CPKVCacheInterleaveSize,
		Frontend:                string(e.Frontend),
	}
	if e.DBO != nil {
		s.DBO = &enginesettings.DBO{Enabled: e.DBO.Enabled}
	}
	if e.EPLB != nil {
		s.EPLB = &enginesettings.EPLB{Enabled: e.EPLB.Enabled, NumRedundantExperts: e.EPLB.NumRedundantExperts}
	}
	if e.Speculative != nil {
		s.Speculative = &enginesettings.Speculative{Method: e.Speculative.Method,
			NumSpecTokens: e.Speculative.NumSpecTokens, DraftModel: e.Speculative.DraftModel,
			// Tri-state in the engine block, nil being vLLM's default false
			// (vllm/config/speculative.py:551@5df63a72).
			EnableAdaptiveVerification: e.Speculative.EnableAdaptiveVerification != nil &&
				*e.Speculative.EnableAdaptiveVerification}
	}
	if e.EngramConfig != nil {
		s.EngramConfig = &enginesettings.EngramConfig{CPUOffload: e.EngramConfig.CPUOffload,
			DPSharedMemory: e.EngramConfig.DPSharedMemory, UseTHP: e.EngramConfig.UseTHP}
	}
	return s.Clone()
}

// OfPool is Of for one pool of a deployment, the shape the kernel's Repos.EngineSettings takes:
// the pool's engine block decoded by blis-vllm, strictly.
func OfPool(dep *deployment.Deployment, pool int) (enginesettings.Settings, error) {
	e, err := Decode(dep.Pools[pool])
	if err != nil {
		return enginesettings.Settings{}, err
	}
	return Of(e), nil
}

// Decode is a pool's engine block as blis-vllm decodes it (engineconfig.Decode).
func Decode(pool deployment.Pool) (engineconfig.Engine, error) {
	return engineconfig.Decode(pool.Engine)
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/utils/common.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/runtime_context.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/layers/moe/utils.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

// Package lower turns SGLang's resolved ServerArgs (package config) into what a decode step runs:
// the parallel widths SGLang derives at publish, whether the step gathers tokens across
// attention ranks, the CUDA-graph capture batch sizes and the bucket a batch pads to, the
// attention backends and the MoE backends. Each function is a port of the SGLang function its
// comment names, and the oracle (oracle/lower/extract.py) runs those functions on the same inputs.
//
// Lowering reads only settings config resolves. An input config does not model is refused there,
// before lowering, so the settings below that config holds at their defaults (elastic EP,
// scale-up joining, the MORI backends) are refused here by name rather than lowered by a guess.
package lower

import (
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/engines/sglang/config"
)

// issueLower is the issue that models a setting lowering refuses.
const issueLower = "inference-sim/inference-sim#2344"

// Parallel holds the widths SGLang derives from ServerArgs when it publishes the runtime
// context (runtime_context.py:142-215, derive_parallel_widths, attn_dp_enabled_of,
// num_dp_ranks_of).
type Parallel struct {
	TPSize        int
	AttnTPSize    int
	AttnDPSize    int
	AttnCPSize    int
	NumDPRanks    int
	AttnDPEnabled bool
}

// ParallelOf is the published parallel widths of a.
func ParallelOf(a *config.ServerArgs) Parallel {
	return Parallel{
		TPSize:     a.TpSize,
		AttnTPSize: a.TpSize / a.AttnDpSize / a.AttnCpSize,
		AttnDPSize: a.AttnDpSize,
		AttnCPSize: a.AttnCpSize,
		NumDPRanks: a.DpSize * a.AttnDpSize,
		// attn_dp_enabled_of: attn_dp_size > 1, or an EP scale-up joiner.
		AttnDPEnabled: a.AttnDpSize > 1 || (a.EpJoinMode != nil && *a.EpJoinMode == "scale"),
	}
}

// NotLowered refuses a setting lowering does not model, by its SGLang name.
type NotLowered struct {
	Name  string
	Issue string
}

func (e *NotLowered) Error() string {
	return fmt.Sprintf("%s is not lowered (%s)", e.Name, e.Issue)
}

// check refuses context parallelism (cp.go), and the settings whose lowering reads state a CPU run cannot know: elastic EP
// (elastic_expanded_world_enabled reads the live world, common.py:4083-4089), and the MORI
// all-to-all backend (its gather rule reads SGLANG_MORI_EP_V2 and SGLANG_MORI_RECV_BOUND,
// common.py:4100-4116).
func check(a *config.ServerArgs) error {
	if err := contextParallel(a); err != nil {
		return err
	}
	if a.ElasticEpBackend != nil {
		return &NotLowered{Name: "elastic_ep_backend", Issue: issueLower}
	}
	if a.MoeA2aBackend == "mori" {
		return &NotLowered{Name: "moe_a2a_backend=mori", Issue: issueLower}
	}
	if a.EnableTorchCompile {
		return &NotLowered{Name: "enable_torch_compile", Issue: issueLower}
	}
	return nil
}

// a2aBackend is MoeA2ABackend(value): None reads as "none" (layers/moe/utils.py:51-58).
func a2aBackend(a *config.ServerArgs) string {
	if a.MoeA2aBackend == "" {
		return "none"
	}
	return a.MoeA2aBackend
}

// RequireMLPTPGather is require_mlp_tp_gather (common.py:4067-4121): whether the MLP input is
// gathered across attention-DP groups rather than reduced.
func RequireMLPTPGather(a *config.ServerArgs) (bool, error) {
	if err := check(a); err != nil {
		return false, err
	}
	p := ParallelOf(a)
	if !p.AttnDPEnabled {
		return false, nil
	}
	if p.NumDPRanks <= 1 {
		return false, fmt.Errorf("attention DP needs more than one DP rank")
	}
	switch b := a2aBackend(a); {
	case a.MoeDenseTpSize == nil:
		return true, nil
	case !a.EnableDpLmHead:
		return true, nil
	case b == "none":
		return true, nil
	case b == "flashinfer", b == "flashinfer_megamoe":
		return true, nil
	default:
		return *a.MoeDenseTpSize > p.TPSize/p.NumDPRanks, nil
	}
}

// RequireAttnTPGather is require_attn_tp_gather (common.py:4124-4156): whether the attention
// input is scattered across attention-TP ranks.
func RequireAttnTPGather(a *config.ServerArgs) (bool, error) {
	if err := check(a); err != nil {
		return false, err
	}
	if a.DisableAttnTpGather {
		return false, nil
	}
	if a2aBackend(a) != "none" || a.MoeDenseTpSize != nil {
		p := ParallelOf(a)
		if p.AttnDPEnabled {
			return p.NumDPRanks < p.TPSize, nil
		}
		return true, nil
	}
	return false, nil
}

// RequireGatheredBuffer is require_gathered_buffer (common.py:4159-4160).
func RequireGatheredBuffer(a *config.ServerArgs) (bool, error) {
	m, err := RequireMLPTPGather(a)
	if err != nil {
		return false, err
	}
	t, err := RequireAttnTPGather(a)
	return m || t, err
}

// RequireMLPSync is require_mlp_sync (common.py:4163-4165): whether DP ranks synchronise each
// step's batch.
func RequireMLPSync(a *config.ServerArgs) (bool, error) {
	g, err := RequireGatheredBuffer(a)
	return ParallelOf(a).AttnDPEnabled || g, err
}

// Alignment is get_cuda_graph_batch_size_alignment (common.py:4168-4181) on CUDA: the multiple
// every captured batch's token count must be.
func Alignment(a *config.ServerArgs) (int, error) {
	g, err := RequireGatheredBuffer(a)
	if err != nil {
		return 0, err
	}
	p := ParallelOf(a)
	alignment := 1
	if a.EnableTwoBatchOverlap {
		alignment *= 2
	}
	if g {
		alignment *= p.AttnTPSize
	}
	if alignment%p.AttnCPSize != 0 {
		alignment *= p.AttnCPSize
	}
	return alignment, nil
}

func ceilAlign(x, y int) int { return (x + y - 1) / y * y }

// CaptureBatchSizes is get_batch_sizes_to_capture
// (model_executor/runner/base_cuda_graph_runner.py:65-110) for the decode runner: the batch
// sizes it captures, from the resolved decode capture list, the request pool's size and the
// captured width per request (1, or the draft tokens a speculative verify step carries).
func CaptureBatchSizes(a *config.ServerArgs, poolSize, reqWidth int) ([]int, error) {
	if a.CudaGraphConfig == nil || len(a.CudaGraphConfig.Decode.Bs) == 0 {
		return nil, fmt.Errorf("cuda_graph_config has no decode batch sizes: resolve the ServerArgs first")
	}
	mulBase, err := Alignment(a)
	if err != nil {
		return nil, err
	}
	capture := append([]int{}, a.CudaGraphConfig.Decode.Bs...)
	width := reqWidth
	if a.EnableTwoBatchOverlap {
		width = 1
	}
	numMax := ceilAlign(poolSize, mulBase)
	largest := 0
	for _, bs := range capture {
		if bs > largest {
			largest = bs
		}
	}
	if largest > numMax {
		capture = append(capture, numMax)
	}
	seen := map[int]bool{}
	var out []int
	for _, bs := range capture {
		if bs*width%mulBase == 0 && bs <= numMax && !seen[bs] {
			seen[bs] = true
			out = append(out, bs)
		}
	}
	sort.Ints(out)
	if len(out) == 0 || out[0] <= 0 {
		return nil, fmt.Errorf("no decode batch size to capture (capture_bs=%v)", out)
	}
	return out, nil
}

// PadToBucket is BaseCudaGraphRunner._pad_to_bucket (base_cuda_graph_runner.py:166-180): the
// smallest captured size that holds raw. ok is false when raw exceeds every bucket: the decode
// runner's can_run_graph then refuses the graph and the step runs eagerly at raw
// (decode_cuda_graph_runner.py:893-897).
func PadToBucket(raw int, buckets []int) (padded int, ok bool) {
	i := sort.SearchInts(buckets, raw)
	if i == len(buckets) {
		return raw, false
	}
	return buckets[i], true
}

// AttentionBackends is attention_backends() (runtime_context.py:2010-2022, by
// attention_backends_of): the (prefill, decode) backends, a split field falling back to
// attention_backend.
func AttentionBackends(a *config.ServerArgs) (prefill, decode string) {
	base := ""
	if a.AttentionBackend != nil {
		base = *a.AttentionBackend
	}
	prefill, decode = base, base
	if a.PrefillAttentionBackend != nil && *a.PrefillAttentionBackend != "" {
		prefill = *a.PrefillAttentionBackend
	}
	if a.DecodeAttentionBackend != nil && *a.DecodeAttentionBackend != "" {
		decode = *a.DecodeAttentionBackend
	}
	return prefill, decode
}

// MoE is what initialize_moe_config installs (layers/moe/utils.py:479-495): the all-to-all
// backend and the MoE runner backend, by their SGLang values.
type MoE struct {
	A2ABackend    string
	RunnerBackend string
}

// MoEOf is the MoE backends of a.
func MoEOf(a *config.ServerArgs) (MoE, error) {
	if err := check(a); err != nil {
		return MoE{}, err
	}
	return MoE{A2ABackend: a2aBackend(a), RunnerBackend: a.MoeRunnerBackend}, nil
}

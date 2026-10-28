// Moved from blis-latency-kernel/caches_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"errors"
	"math"
	"testing"

	"github.com/inference-sim/blis-schemas/rules"

	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// Per-layer caches, as a graph states them (blis-schemas model.Cache), checked through
// what a simulator reads: SequenceVariableBytes, SequenceFixedBytes, PDTransferDemand and
// StepTime.

const v41 = "deepseek-v41-flash-h200-tp8.yaml"

// engineCapacity is vLLM's "GPU KV cache size" for a kernel's figures, from a KV pool of
// gib GiB: the pool's blocks over a request's blocks at max_model_len, times max_model_len
// (get_kv_cache_capacity, vllm/v1/core/kv_cache_utils.py:2444-2471;
// get_max_concurrency_for_kv_cache_config, :1090-1122), with the slack a log's two-decimal
// GiB leaves in it.
func engineCapacity(k *Kernel, gib float64, maxModelLen int) (tokens, slack float64) {
	block := k.SequenceVariableBytes(k.blockSize)
	pool := int64(gib*(1<<30)) / block
	perRequest := (k.SequenceAdmissionBytes() + k.SequenceVariableBytes(maxModelLen)) / block
	tokens = float64(pool) / float64(perRequest) * float64(maxModelLen)
	return tokens, tokens * 0.005 * (1 << 30) / float64(block) / float64(pool)
}

// GOLDEN, from the engine's own log, H200. An internal DSV4.1 H200 deployment on
// 8x H200 under vLLM v0.31.0 (our calibration runs: TP 8 with --enable-expert-parallel,
// DSpark K=5, --max-model-len 1048576, --max-num-batched-tokens 8192, --block-size 64,
// max_cudagraph_capture_size 64) reported 61.18 GiB of KV cache and 28,706,997 tokens
// (vllm:cache_config_info of our calibration runs,
// and inference-sim/blis-latency-kernel#46). From the same 61.18 GiB the kernel's figures give
// the same count, to the rounding of the logged GiB; sized at the steady decode hold they
// would give 25% more.
func TestKVCapacityIsTheEnginesOwnFigureOnH200(t *testing.T) {
	in := fixtureInputs(t, v41)
	in.Deployment.Pools[0].Parallel.EnableExpertParallel = true
	e := mustDecode(t, in.Deployment.Pools[0])
	capture := 64
	e.MaxModelLen, e.BlockSize, e.MaxCUDAGraphCaptureSize = 1<<20, 64, &capture
	setEngine(t, &in.Deployment.Pools[0], e)
	tokens, slack := engineCapacity(build(t, in), 61.18, 1<<20)
	if d := tokens - 28706997; d > slack || d < -slack {
		t.Errorf("the kernel sizes %.0f tokens in vLLM's 61.18 GiB, against its logged "+
			"28,706,997 (slack %.0f)", tokens, slack)
	}
}

// GOLDEN, the five arms of our calibration runs' H200 sizing table under vLLM v0.31.0
// (from vLLM boot logs and vllm:cache_config_info; the baseline is the deployment
// above). Each arm's capacity follows from its own logged KV pool: without DSpark its 40 window rings fill seven groups,
// not eight; at TP 4 a request's caches are the same, since a latent cache is not divided
// across tensor parallelism. The weights move between TP 8 and TP 4 as the kernel's sharded
// weights do, to 0.3%: 81.49 - 45.81 = 35.68 GiB measured. The weights themselves are not
// asserted: vLLM holds about 9 GiB more than the kernel at both widths, a term that does
// not scale with TP and that the kernel does not yet account for (see computeFixedBytes).
func TestInternalH200ArmsAreTheEnginesFigures(t *testing.T) {
	type arm struct {
		name       string
		tp         int
		dspark     bool
		kvGiB      float64
		tokens     float64
		weightsGiB float64
	}
	arms := []arm{
		{"TP8-baseline", 8, true, 61.18, 28706997, 45.81},
		{"TP8-no-DSpark", 8, false, 65.17, 31367749, 42.64},
		{"TP8-no-KV-offload", 8, true, 61.26, 28743634, 45.73},
		{"TP8-no-Engram", 8, true, 61.18, 28706997, 45.81},
		{"TP4-baseline", 4, true, 25.35, 11893655, 81.49},
	}
	weights := map[string]float64{}
	for _, a := range arms {
		in := fixtureInputs(t, v41)
		pl := &in.Deployment.Pools[0].Parallel
		pl.TP, pl.EnableExpertParallel = a.tp, true
		e := mustDecode(t, in.Deployment.Pools[0])
		capture := 64
		e.MaxModelLen, e.BlockSize, e.MaxCUDAGraphCaptureSize = 1<<20, 64, &capture
		if !a.dspark {
			e.Speculative = nil
		}
		setEngine(t, &in.Deployment.Pools[0], e)
		k := build(t, in)
		tokens, slack := engineCapacity(k, a.kvGiB, 1<<20)
		if d := tokens - a.tokens; d > slack || d < -slack {
			t.Errorf("%s: the kernel sizes %.0f tokens in vLLM's %.2f GiB, against its "+
				"logged %.0f (slack %.0f)", a.name, tokens, a.kvGiB, a.tokens, slack)
		}
		weights[a.name] = float64(k.FixedBytes().Weights) / (1 << 30)
	}
	measured := arms[4].weightsGiB - arms[0].weightsGiB
	if got := weights["TP4-baseline"] - weights["TP8-baseline"]; math.Abs(got-measured) > 0.003*measured {
		t.Errorf("TP 8 to TP 4 adds %.2f GiB of weights a rank, against vLLM's %.2f", got,
			measured)
	}
}

// GOLDEN, from the engine's own log, H100. InferenceX's DeepSeek-V4.1-Flash AgentX recipe on
// 8x H100 (benchmarks/single_node/srt-slurm-recipes/dsv41flash/vllm/h100-fp4-mtp/agentic.yaml,
// override_tp8_c1: TP 8 without expert parallelism, DSpark K=5, --max-model-len 1048576,
// --max-num-batched-tokens 4096, --max-num-seqs 2, max-cudagraph-capture-size 16,
// --gpu-memory-utilization 0.92, the V2 model runner) logged at concurrency 1 "13.47 GiB KV
// per GPU, 7,022,899 tokens, 6.70x maximum concurrency at 1M tokens per request"
// (perf-changelog.yaml, InferenceX PR 2974). Its image is a vLLM nightly
// (nightly-cd10ed6f), whose pool sizing this checks against v0.31.0's: 8 window groups at
// cdiv(127 + 2 x 4,096, 32) + 1 = 261 pages a request.
func TestKVCapacityIsTheEnginesOwnFigureOnH100(t *testing.T) {
	in := fixtureInputs(t, v41)
	in.Chip, in.Scenario.Cluster.Hardware = chipInputs(t, "h100"), "h100"
	e := mustDecode(t, in.Deployment.Pools[0])
	capture := 16
	e.MaxModelLen, e.MaxNumBatchedTokens, e.MaxNumSeqs = 1<<20, 4096, 2
	e.GPUMemoryUtilization, e.MaxCUDAGraphCaptureSize = 0.92, &capture
	setEngine(t, &in.Deployment.Pools[0], e)
	tokens, slack := engineCapacity(build(t, in), 13.47, 1<<20)
	if d := tokens - 7022899; d > slack || d < -slack {
		t.Errorf("the kernel sizes %.0f tokens in vLLM's 13.47 GiB, against its logged "+
			"7,022,899 (slack %.0f)", tokens, slack)
	}
}

// ungroupedRules is a rules pack that cannot group stated caches.
type ungroupedRules struct{ resolve.EngineRules }

func (ungroupedRules) KVCacheGroups(rules.KVCacheGroupsInput) (rules.KVCacheGrouping, error) {
	return rules.KVCacheGrouping{}, errors.New("not transcribed")
}

// Where the rules pack cannot group the stated caches, each is sized at its own page, and
// that is recorded rather than silent.
func TestAnUngroupedPoolIsRecorded(t *testing.T) {
	in := fixtureInputs(t, v41)
	in.Rules = ungroupedRules{in.Rules}
	k := build(t, in)
	if !recorded(k, "kv_cache_grouping") {
		t.Error("the caches were sized per cache with nothing in Provenance saying so")
	}
	if pooled := build(t, fixtureInputs(t, v41)); k.SequenceFixedBytes() >= pooled.SequenceFixedBytes() {
		t.Errorf("per-cache pages hold %d B a sequence, not less than the pool's %d",
			k.SequenceFixedBytes(), pooled.SequenceFixedBytes())
	}
}

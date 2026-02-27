package sim

import (
	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// BatchFormation encapsulates the batch composition strategy for a simulation step.
// Implementations handle KV allocation and preemption decisions internally
// but do NOT schedule events or record metrics — those are kernel concerns
// handled by the Simulator after FormBatch returns.
type BatchFormation interface {
	FormBatch(ctx BatchContext) BatchResult
}

// BatchContext provides the inputs for batch formation.
// The BatchFormation implementation may mutate WaitQ (dequeue/prepend) and
// KVCache (allocate/release) during FormBatch. ComputedTokens must be updated
// by the implementation: for each request that receives new tokens, set or
// increment ComputedTokens[req.ID] to the total computed tokens (including
// cached). Phase 2 of Step() reads this map to advance ProgressIndex.
type BatchContext struct {
	RunningBatch          *Batch
	WaitQ                 *WaitQueue
	KVCache               KVStore
	MaxScheduledTokens    int64
	MaxRunningReqs        int64
	PrefillTokenThreshold int64
	Now                   int64
	StepCount             int
	ComputedTokens        map[string]int64
}

// ScheduledRequest carries metadata about a newly scheduled request.
type ScheduledRequest struct {
	Request        *Request
	ScheduledDelay int64
}

// PreemptedRequest carries metadata about a preempted request.
type PreemptedRequest struct {
	Request         *Request
	PreemptionDelay int64
}

// BatchResult describes the outcome of batch formation.
type BatchResult struct {
	RunningBatch       *Batch
	NewlyScheduled     []ScheduledRequest
	Preempted          []PreemptedRequest
	PreemptionHappened bool
}

// SLOPrefillConfig controls per-SLO-class chunked prefill thresholds.
// When Enabled, critical requests use CriticalThreshold and sheddable use SheddableThreshold.
// Other classes use the global PrefillTokenThreshold from BatchContext.
// Disabled by default (zero value = use global threshold for all).
var SLOPrefillConfig = struct {
	Enabled            bool
	CriticalThreshold  int64 // lower = more aggressive chunking for critical (better TTFT)
	SheddableThreshold int64 // higher = fewer chunks for sheddable (better throughput, 0=no chunking)
}{
	Enabled:            false,
	CriticalThreshold:  128,
	SheddableThreshold: 0, // 0 means no chunking (process entire prefill in one step)
}

// VLLMBatchFormation implements the vLLM FCFS + chunked-prefill + preemption strategy.
type VLLMBatchFormation struct {
	latencyModel LatencyModel
}

// effectivePrefillThreshold returns the chunked prefill threshold for a request.
// When SLOPrefillConfig is enabled, returns per-class thresholds.
// Otherwise returns the global ctx.PrefillTokenThreshold.
func effectivePrefillThreshold(req *Request, globalThreshold int64) int64 {
	if !SLOPrefillConfig.Enabled {
		return globalThreshold
	}
	switch req.SLOClass {
	case "critical":
		return SLOPrefillConfig.CriticalThreshold
	case "sheddable", "batch", "background":
		return SLOPrefillConfig.SheddableThreshold
	default: // "", "standard"
		return globalThreshold
	}
}

func (v *VLLMBatchFormation) FormBatch(ctx BatchContext) BatchResult {
	if ctx.RunningBatch == nil {
		ctx.RunningBatch = &Batch{}
	}

	result := BatchResult{
		RunningBatch: ctx.RunningBatch,
	}

	tokenBudget := ctx.MaxScheduledTokens

	// Phase 1: Process continuing requests (chunked prefill + decode).
	// NOTE: preemptForTokens may shorten result.RunningBatch.Requests during iteration
	// (tail eviction). Go's range captures the slice header at loop entry, so the loop
	// still visits evicted requests at their original indices. This matches the original
	// makeRunningBatch() behavior exactly — do NOT "fix" this.
	for _, req := range ctx.RunningBatch.Requests {
		if tokenBudget <= 0 {
			logrus.Warnf("[tick %07d] token budget exhausted, deferring remaining requests to next step", ctx.Now)
			break
		}
		numNewTokens := util.Len64(req.InputTokens) - req.ProgressIndex
		// Chunked prefill for running requests (SLO-aware threshold)
		if numNewTokens > 0 {
			threshold := effectivePrefillThreshold(req, ctx.PrefillTokenThreshold)
			if 0 < threshold && threshold < numNewTokens {
				numNewTokens = threshold
			}
			numNewTokens = min(numNewTokens, tokenBudget)

			if canSchedule := v.preemptForTokens(req, numNewTokens, &result, ctx); !canSchedule {
				break
			}

			tokenBudget -= numNewTokens
			req.NumNewTokens = int(numNewTokens)
			ctx.ComputedTokens[req.ID] += numNewTokens
		}
		// Decode phase: allocate 1 token
		if req.ProgressIndex >= util.Len64(req.InputTokens) && len(req.OutputTokens) > 0 {
			decodeTokens := int64(1)
			if canSchedule := v.preemptForTokens(req, decodeTokens, &result, ctx); !canSchedule {
				break
			}
			tokenBudget--
			req.NumNewTokens = 1
			ctx.ComputedTokens[req.ID] += 1
		}
	}

	// Phase 2: Dequeue new requests from wait queue
	for len(result.RunningBatch.Requests) < int(ctx.MaxRunningReqs) && ctx.WaitQ.Len() > 0 && tokenBudget > 0 && !result.PreemptionHappened {
		next := ctx.WaitQ.Peek()

		cachedBlocks := ctx.KVCache.GetCachedBlocks(next.InputTokens)
		numNewTokens := util.Len64(next.InputTokens) - util.Len64(cachedBlocks)*ctx.KVCache.BlockSize()

		threshold := effectivePrefillThreshold(next, ctx.PrefillTokenThreshold)
		if 0 < threshold && threshold < numNewTokens {
			numNewTokens = threshold
		}
		numNewTokens = min(numNewTokens, tokenBudget)
		startIndex := util.Len64(cachedBlocks) * ctx.KVCache.BlockSize()
		endIndex := startIndex + numNewTokens

		if ok := ctx.KVCache.AllocateKVBlocks(next, startIndex, endIndex, cachedBlocks); !ok {
			break
		}

		ctx.WaitQ.DequeueBatch()
		result.RunningBatch.Requests = append(result.RunningBatch.Requests, next)
		next.ScheduledStepIdx = ctx.StepCount

		scheduledDelay := v.latencyModel.SchedulingProcessingTime()
		result.NewlyScheduled = append(result.NewlyScheduled, ScheduledRequest{
			Request:        next,
			ScheduledDelay: scheduledDelay,
		})

		tokenBudget -= numNewTokens
		next.State = StateRunning
		next.NumNewTokens = int(numNewTokens)
		ctx.ComputedTokens[next.ID] = numNewTokens + util.Len64(cachedBlocks)*ctx.KVCache.BlockSize()
	}

	return result
}

// preemptForTokens tries to allocate numNewTokens of KV blocks for req,
// evicting from the batch tail if needed. Returns false if allocation is
// impossible (cache too small or request was itself evicted).
func (v *VLLMBatchFormation) preemptForTokens(req *Request, numNewTokens int64, result *BatchResult, ctx BatchContext) bool {
	for {
		if ok := ctx.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+numNewTokens, []int64{}); !ok {
			// Circuit breaker: empty batch means cache is too small (R19)
			if len(result.RunningBatch.Requests) == 0 {
				logrus.Warnf("[tick %07d] preemption: KV cache too small for request %s (need %d tokens, no running requests to evict)",
					ctx.Now, req.ID, numNewTokens)
				return false
			}

			result.PreemptionHappened = true
			preemptionDelay := v.latencyModel.PreemptionProcessingTime()
			preemptedRequest := result.RunningBatch.Requests[len(result.RunningBatch.Requests)-1]
			logrus.Warnf("[tick %07d] preemption: evicting %s to make room", ctx.Now, preemptedRequest.ID)
			result.RunningBatch.Requests = result.RunningBatch.Requests[:len(result.RunningBatch.Requests)-1]

			result.Preempted = append(result.Preempted, PreemptedRequest{
				Request:         preemptedRequest,
				PreemptionDelay: preemptionDelay,
			})

			preemptedRequest.State = StateQueued
			preemptedRequest.ProgressIndex = 0
			ctx.KVCache.ReleaseKVBlocks(preemptedRequest)
			delete(ctx.ComputedTokens, preemptedRequest.ID)
			ctx.WaitQ.PrependFront(preemptedRequest)

			if preemptedRequest == req {
				return false
			}
		} else {
			return true
		}
	}
}

// NewBatchFormation creates the default BatchFormation.
// Currently returns VLLMBatchFormation (the only implementation).
func NewBatchFormation(latencyModel LatencyModel) BatchFormation {
	return &VLLMBatchFormation{
		latencyModel: latencyModel,
	}
}

package sim

import "github.com/sirupsen/logrus"

// BatchFormation encapsulates the batch composition strategy for a simulation step.
// Implementations handle KV allocation and preemption decisions internally
// but do NOT schedule events or record metrics — those are kernel concerns
// handled by the Simulator after FormBatch returns.
type BatchFormation interface {
	FormBatch(ctx BatchContext) BatchResult
}

// BatchContext provides the inputs for batch formation.
// The BatchFormation implementation may mutate WaitQ (dequeue/prepend) and
// KVCache (allocate/release) during FormBatch.
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

// VLLMBatchFormation implements the vLLM FCFS + chunked-prefill + preemption strategy.
type VLLMBatchFormation struct {
	latencyModel LatencyModel
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
		numNewTokens := Len64(req.InputTokens) - req.ProgressIndex
		// Chunked prefill for running requests
		if numNewTokens > 0 {
			if 0 < ctx.PrefillTokenThreshold && ctx.PrefillTokenThreshold < numNewTokens {
				numNewTokens = ctx.PrefillTokenThreshold
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
		if req.ProgressIndex >= Len64(req.InputTokens) && len(req.OutputTokens) > 0 {
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
		numNewTokens := Len64(next.InputTokens) - Len64(cachedBlocks)*ctx.KVCache.BlockSize()

		if 0 < ctx.PrefillTokenThreshold && ctx.PrefillTokenThreshold < numNewTokens {
			numNewTokens = ctx.PrefillTokenThreshold
		}
		numNewTokens = min(numNewTokens, tokenBudget)
		startIndex := Len64(cachedBlocks) * ctx.KVCache.BlockSize()
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
		ctx.ComputedTokens[next.ID] = numNewTokens + Len64(cachedBlocks)*ctx.KVCache.BlockSize()
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
// SimConfig parameter is reserved for future strategy selection.
func NewBatchFormation(_ SimConfig, latencyModel LatencyModel) BatchFormation {
	return &VLLMBatchFormation{
		latencyModel: latencyModel,
	}
}

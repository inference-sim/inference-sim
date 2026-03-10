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
	RunningBatch             *Batch
	WaitQ                    *WaitQueue
	KVCache                  KVStore
	MaxScheduledTokens       int64
	MaxRunningReqs           int64
	PrefillTokenThreshold    int64
	PriorityPreemptionMargin float64 // from BatchConfig; if > 0, enables priority-based preemption in Phase 2
	Now                      int64
	StepCount                int
	ComputedTokens           map[string]int64
}

// ScheduledRequest carries metadata about a newly scheduled request.
type ScheduledRequest struct {
	Request *Request
}

// PreemptedRequest carries metadata about a preempted request.
type PreemptedRequest struct {
	Request *Request
}

// BatchResult describes the outcome of batch formation.
type BatchResult struct {
	RunningBatch       *Batch
	NewlyScheduled     []ScheduledRequest
	Preempted          []PreemptedRequest
	PreemptionHappened bool
}

// VLLMBatchFormation implements the vLLM FCFS + chunked-prefill + preemption strategy.
type VLLMBatchFormation struct{}

func (v *VLLMBatchFormation) FormBatch(ctx BatchContext) BatchResult {
	if ctx.RunningBatch == nil {
		ctx.RunningBatch = &Batch{}
	}

	result := BatchResult{
		RunningBatch: ctx.RunningBatch,
	}

	tokenBudget := ctx.MaxScheduledTokens

	// Zero NumNewTokens for all running requests at the start of each scheduling pass.
	// This prevents stale values from the prior step from causing phantom budget
	// restoration when a request is preempted before being visited in this pass.
	for _, req := range result.RunningBatch.Requests {
		req.NumNewTokens = 0
	}

	// Phase 1: Process continuing requests (chunked prefill + decode).
	// Index-based loop: re-evaluates len() each iteration so evicted requests
	// (removed by preemptForTokens tail eviction) are never visited.
	// This achieves the same behavioral property as vLLM v1 (evicted requests
	// are never revisited within a scheduling pass), though through a different
	// mechanism (vLLM uses deque popleft/pop; BLIS uses index bounds re-evaluation).
	reqIndex := 0
	for reqIndex < len(result.RunningBatch.Requests) {
		if tokenBudget <= 0 {
			logrus.Warnf("[tick %07d] token budget exhausted, deferring remaining requests to next step", ctx.Now)
			break
		}
		req := result.RunningBatch.Requests[reqIndex]

		numNewTokens := util.Len64(req.InputTokens) - req.ProgressIndex
		// Chunked prefill for running requests
		if numNewTokens > 0 {
			if 0 < ctx.PrefillTokenThreshold && ctx.PrefillTokenThreshold < numNewTokens {
				numNewTokens = ctx.PrefillTokenThreshold
			}
			numNewTokens = min(numNewTokens, tokenBudget)

			if canSchedule := v.preemptForTokens(req, numNewTokens, &result, ctx, &tokenBudget); !canSchedule {
				break
			}

			tokenBudget -= numNewTokens
			req.NumNewTokens = int(numNewTokens)
			ctx.ComputedTokens[req.ID] += numNewTokens
		}
		// Decode phase: allocate 1 token
		if req.ProgressIndex >= util.Len64(req.InputTokens) && len(req.OutputTokens) > 0 {
			decodeTokens := int64(1)
			if canSchedule := v.preemptForTokens(req, decodeTokens, &result, ctx, &tokenBudget); !canSchedule {
				break
			}
			tokenBudget--
			req.NumNewTokens = 1
			ctx.ComputedTokens[req.ID] += 1
		}
		reqIndex++
	}

	// Phase 2: Dequeue new requests from wait queue.
	// Priority preemption: when the batch is full and a high-priority request is waiting,
	// evict the lowest-priority running request to make room (R19: max 3 per step).
	priorityPreemptionsThisStep := 0
	for ctx.WaitQ.Len() > 0 && tokenBudget > 0 && !result.PreemptionHappened {
		batchFull := len(result.RunningBatch.Requests) >= int(ctx.MaxRunningReqs)

		if batchFull {
			// Priority preemption: evict lowest-priority running request if a much
			// higher-priority request is waiting.
			if ctx.PriorityPreemptionMargin <= 0 || priorityPreemptionsThisStep >= 3 {
				break // Disabled or circuit breaker (R19: max 3 priority preemptions per step)
			}
			next := ctx.WaitQ.Peek()
			lowestIdx := findLowestPriorityRunning(result.RunningBatch.Requests)
			if lowestIdx < 0 {
				break
			}
			lowest := result.RunningBatch.Requests[lowestIdx]
			if next.Priority-lowest.Priority < ctx.PriorityPreemptionMargin {
				break // Priority difference too small
			}

			// Evict the lowest-priority running request
			logrus.Warnf("[tick %07d] priority-preemption: evicting %s (pri=%.1f) for waiting %s (pri=%.1f)",
				ctx.Now, lowest.ID, lowest.Priority, next.ID, next.Priority)

			// Remove from batch (swap with last, then truncate — order doesn't matter
			// since Phase 1 already ran)
			lastIdx := len(result.RunningBatch.Requests) - 1
			result.RunningBatch.Requests[lowestIdx] = result.RunningBatch.Requests[lastIdx]
			result.RunningBatch.Requests = result.RunningBatch.Requests[:lastIdx]

			result.Preempted = append(result.Preempted, PreemptedRequest{Request: lowest})

			// Restore token budget if the preempted request was allocated tokens this step
			if lowest.NumNewTokens > 0 {
				tokenBudget += int64(lowest.NumNewTokens)
				lowest.NumNewTokens = 0
			}

			lowest.State = StateQueued
			lowest.ProgressIndex = 0
			ctx.KVCache.ReleaseKVBlocks(lowest)
			delete(ctx.ComputedTokens, lowest.ID)
			// Enqueue at back (not front) so the high-priority waiting request
			// that triggered this preemption gets scheduled first in the standard
			// Phase 2 path that follows.
			ctx.WaitQ.Enqueue(lowest)

			priorityPreemptionsThisStep++
			continue // Re-check loop: now batch has room
		}

		// Standard Phase 2: schedule from wait queue (existing logic)
		next := ctx.WaitQ.Peek()

		cachedBlocks := ctx.KVCache.GetCachedBlocks(next.InputTokens)
		numNewTokens := util.Len64(next.InputTokens) - util.Len64(cachedBlocks)*ctx.KVCache.BlockSize()

		if 0 < ctx.PrefillTokenThreshold && ctx.PrefillTokenThreshold < numNewTokens {
			numNewTokens = ctx.PrefillTokenThreshold
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

		result.NewlyScheduled = append(result.NewlyScheduled, ScheduledRequest{
			Request: next,
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
func (v *VLLMBatchFormation) preemptForTokens(req *Request, numNewTokens int64, result *BatchResult, ctx BatchContext, tokenBudget *int64) bool {
	for {
		if ok := ctx.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+numNewTokens, []int64{}); !ok {
			// Circuit breaker: empty batch means cache is too small (R19)
			if len(result.RunningBatch.Requests) == 0 {
				logrus.Warnf("[tick %07d] preemption: KV cache too small for request %s (need %d tokens, no running requests to evict)",
					ctx.Now, req.ID, numNewTokens)
				return false
			}

			result.PreemptionHappened = true
			preemptedRequest := result.RunningBatch.Requests[len(result.RunningBatch.Requests)-1]
			logrus.Warnf("[tick %07d] preemption: evicting %s to make room", ctx.Now, preemptedRequest.ID)
			result.RunningBatch.Requests = result.RunningBatch.Requests[:len(result.RunningBatch.Requests)-1]

			result.Preempted = append(result.Preempted, PreemptedRequest{
				Request: preemptedRequest,
			})

			// Defensive: restore token budget if preempted request was already
			// scheduled in this step. Currently unreachable with head-to-tail
			// iteration + tail-only eviction (evicted requests are always
			// unvisited, so NumNewTokens is 0 from the FormBatch entry zeroing).
			// Guards against future iteration order changes (e.g., priority-based
			// eviction that could evict an already-visited request).
			if preemptedRequest.NumNewTokens > 0 {
				*tokenBudget += int64(preemptedRequest.NumNewTokens)
				preemptedRequest.NumNewTokens = 0
			}

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

// findLowestPriorityRunning returns the index of the running request with the
// lowest Priority value. Returns -1 if the batch is empty.
func findLowestPriorityRunning(requests []*Request) int {
	if len(requests) == 0 {
		return -1
	}
	minIdx := 0
	for i := 1; i < len(requests); i++ {
		if requests[i].Priority < requests[minIdx].Priority {
			minIdx = i
		}
	}
	return minIdx
}

// NewBatchFormation creates the default BatchFormation.
// Currently returns VLLMBatchFormation (the only implementation).
func NewBatchFormation() BatchFormation {
	return &VLLMBatchFormation{}
}

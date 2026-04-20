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
	MaxModelLen           int64 // 0 = unlimited (proactive cap disabled)
	Now                   int64
	StepCount             int
	ComputedTokens        map[string]int64
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
			// Proactive MaxModelLen cap (BC-1): match vLLM scheduler.py:773-774.
			// Note: the enqueue guard (len(InputTokens) < maxModelLen) guarantees
			// maxAllowed >= 1 during prefill, so this clamp only reduces chunk size,
			// never eliminates it. Defense-in-depth for bypass scenarios.
			if ctx.MaxModelLen > 0 {
				maxAllowed := max(ctx.MaxModelLen-1-req.ProgressIndex, 0)
				numNewTokens = min(numNewTokens, maxAllowed)
			}

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
			// Proactive MaxModelLen cap (BC-1): skip decode at boundary.
			// Equivalent to max(0, maxModelLen-1-PI) < 1, specialized for single-token decode.
			// Note: when decodeTokens=0, the request stays in RunningBatch with its
			// previously-allocated KV blocks for one zero-work step until processCompletions
			// releases them via ReleaseKVBlocks. Under tight KV pressure this transiently
			// reduces available blocks by the request's allocation.
			if ctx.MaxModelLen > 0 && req.ProgressIndex+decodeTokens > ctx.MaxModelLen-1 {
				decodeTokens = 0
			}
			if decodeTokens > 0 {
				if canSchedule := v.preemptForTokens(req, decodeTokens, &result, ctx, &tokenBudget); !canSchedule {
					break
				}
				tokenBudget--
				req.NumNewTokens = 1
				ctx.ComputedTokens[req.ID] += 1
			}
		}
		reqIndex++
	}

	// Phase 1.5: Re-admit preempted requests from this step.
	// In real vLLM, a preempted request's freed blocks (with hashes preserved via
	// lazy deletion) are available within the same scheduling context — the next
	// schedule() call happens before any other allocation can consume them (one
	// forward pass between schedule calls). In BLIS's DES, Phase 1's preemption
	// guard (!result.PreemptionHappened) would otherwise create a 1-step delay
	// during which running requests consume freed prefix blocks via popFreeBlock,
	// destroying shared prefix hashes and causing infinite preemption cycles
	// (issue #1087). Phase 1.5 closes this timing window for preempted requests
	// only — Phase 2's guard remains intact for new requests.
	// Iterate in reverse: preemptForTokens prepends each victim to WaitQ front,
	// so the LAST evicted request is at WaitQ[0], second-to-last at WaitQ[1], etc.
	// result.Preempted is in eviction order (first evicted at index 0).
	for i := len(result.Preempted) - 1; i >= 0 && len(result.RunningBatch.Requests) < int(ctx.MaxRunningReqs) && tokenBudget > 0; i-- {
		preempted := result.Preempted[i].Request

		// The preempted request is at the front of WaitQ (placed there by preemptForTokens).
		// Verify it's actually at the front before dequeuing — defensive guard.
		if ctx.WaitQ.Len() == 0 || ctx.WaitQ.Peek() != preempted {
			break
		}

		cachedBlocks := ctx.KVCache.GetCachedBlocks(preempted.InputTokens)
		numNewTokens := util.Len64(preempted.InputTokens) - util.Len64(cachedBlocks)*ctx.KVCache.BlockSize()

		if 0 < ctx.PrefillTokenThreshold && ctx.PrefillTokenThreshold < numNewTokens {
			numNewTokens = ctx.PrefillTokenThreshold
		}
		numNewTokens = min(numNewTokens, tokenBudget)
		startIndex := util.Len64(cachedBlocks) * ctx.KVCache.BlockSize()
		if ctx.MaxModelLen > 0 {
			maxAllowed := max(ctx.MaxModelLen-1-startIndex, 0)
			numNewTokens = min(numNewTokens, maxAllowed)
		}
		endIndex := startIndex + numNewTokens

		if ok := ctx.KVCache.AllocateKVBlocks(preempted, startIndex, endIndex, cachedBlocks); !ok {
			// Cannot re-admit this preempted request — leave it and remaining
			// preempted requests in WaitQ for the next step.
			break
		}

		ctx.WaitQ.DequeueBatch()
		result.RunningBatch.Requests = append(result.RunningBatch.Requests, preempted)
		preempted.ScheduledStepIdx = ctx.StepCount
		result.NewlyScheduled = append(result.NewlyScheduled, ScheduledRequest{Request: preempted})
		tokenBudget -= numNewTokens
		preempted.State = StateRunning
		preempted.NumNewTokens = int(numNewTokens)
		ctx.ComputedTokens[preempted.ID] = numNewTokens + util.Len64(cachedBlocks)*ctx.KVCache.BlockSize()
	}

	// Phase 2: Dequeue new requests from wait queue
	for len(result.RunningBatch.Requests) < int(ctx.MaxRunningReqs) && ctx.WaitQ.Len() > 0 && tokenBudget > 0 && !result.PreemptionHappened {
		next := ctx.WaitQ.Peek()

		// Handle decode-only requests (PD disaggregation: KV pre-allocated by transfer).
		// IsDecodeSubRequest is set exclusively by KVTransferCompletedEvent, so this
		// path fires only for requests that genuinely arrived via PD KV transfer.
		// ProgressIndex has already been set to len(InputTokens) by AllocateTransferredKV.
		if next.IsDecodeSubRequest {
			decodeTokens := int64(1)
			if ok := ctx.KVCache.AllocateKVBlocks(next, next.ProgressIndex, next.ProgressIndex+decodeTokens, nil); !ok {
				break
			}
			ctx.WaitQ.DequeueBatch()
			result.RunningBatch.Requests = append(result.RunningBatch.Requests, next)
			next.ScheduledStepIdx = ctx.StepCount
			result.NewlyScheduled = append(result.NewlyScheduled, ScheduledRequest{Request: next})
			tokenBudget -= decodeTokens
			next.State = StateRunning
			next.NumNewTokens = 1
			ctx.ComputedTokens[next.ID] = next.ProgressIndex + decodeTokens
			continue
		}

		cachedBlocks := ctx.KVCache.GetCachedBlocks(next.InputTokens)
		numNewTokens := util.Len64(next.InputTokens) - util.Len64(cachedBlocks)*ctx.KVCache.BlockSize()

		if 0 < ctx.PrefillTokenThreshold && ctx.PrefillTokenThreshold < numNewTokens {
			numNewTokens = ctx.PrefillTokenThreshold
		}
		numNewTokens = min(numNewTokens, tokenBudget)
		startIndex := util.Len64(cachedBlocks) * ctx.KVCache.BlockSize()
		// Proactive MaxModelLen cap (BC-2): BLIS safety extension (vLLM only caps running requests).
		// For valid enqueued requests (input < maxModelLen), this is a no-op.
		if ctx.MaxModelLen > 0 {
			maxAllowed := max(ctx.MaxModelLen-1-startIndex, 0)
			numNewTokens = min(numNewTokens, maxAllowed)
		}
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
		if ok := ctx.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+numNewTokens, nil); !ok {
			// Circuit breaker: empty batch means cache is too small (R19)
			if len(result.RunningBatch.Requests) == 0 {
				logrus.Warnf("[tick %07d] preemption: KV cache too small for request %s (need %d tokens, no running requests to evict)",
					ctx.Now, req.ID, numNewTokens)
				return false
			}

			result.PreemptionHappened = true
			preemptedRequest := result.RunningBatch.Requests[len(result.RunningBatch.Requests)-1]
			preemptedRequest.PreemptionCount++
			logrus.Warnf("[tick %07d] preemption: evicting %s to make room (preemption #%d)",
				ctx.Now, preemptedRequest.ID, preemptedRequest.PreemptionCount)
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

// NewBatchFormation creates the default BatchFormation.
// Currently returns VLLMBatchFormation (the only implementation).
func NewBatchFormation() BatchFormation {
	return &VLLMBatchFormation{}
}

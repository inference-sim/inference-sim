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

// PreemptionPolicy controls how preemption selects a victim from the running batch.
type PreemptionPolicy string

const (
	// PreemptionFCFS evicts the last request in the running batch (tail).
	// Matches vLLM's FCFS scheduling mode (self.running.pop()).
	PreemptionFCFS PreemptionPolicy = "fcfs"

	// PreemptionPriority evicts the least-urgent request based on SLO tier priority.
	// Selects min(SLOPriority) with max(ArrivalTime) tiebreak.
	// Matches vLLM's PRIORITY scheduling mode (scheduler.py:827-829)
	// but using BLIS's inverted convention (BLIS: higher=more urgent; vLLM: lower=more urgent).
	PreemptionPriority PreemptionPolicy = "priority"
)

// VLLMBatchFormation implements the vLLM FCFS + chunked-prefill + preemption strategy.
type VLLMBatchFormation struct {
	preemptionPolicy PreemptionPolicy
	sloMap           *SLOPriorityMap
}

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
	// are never visited. In priority mode, non-tail eviction shifts elements
	// left; the reqAdjustment returned by preemptForTokens compensates by
	// decrementing reqIndex, preventing element skipping.
	// Analog of vLLM v1 req_index -= 1 (scheduler.py:853).
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

			canSchedule, adj := v.preemptForTokens(req, numNewTokens, &result, ctx, &tokenBudget, reqIndex)
			reqIndex -= adj
			if !canSchedule {
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
				canSchedule, adj := v.preemptForTokens(req, decodeTokens, &result, ctx, &tokenBudget, reqIndex)
				reqIndex -= adj
				if !canSchedule {
					break
				}
				tokenBudget--
				req.NumNewTokens = 1
				ctx.ComputedTokens[req.ID] += 1
			}
		}
		reqIndex++
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
// evicting victims if needed. Returns (canSchedule, reqAdjustment) where
// reqAdjustment counts evictions at indices below reqIndex.
// The caller must apply reqIndex -= reqAdjustment after each call to prevent
// element skipping when non-tail removal shifts elements left.
// Analog of vLLM scheduler.py:853 (req_index -= 1).
func (v *VLLMBatchFormation) preemptForTokens(req *Request, numNewTokens int64, result *BatchResult, ctx BatchContext, tokenBudget *int64, reqIndex int) (bool, int) {
	adjustment := 0
	for {
		if ok := ctx.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+numNewTokens, nil); !ok {
			// Circuit breaker: empty batch means cache is too small (R19)
			if len(result.RunningBatch.Requests) == 0 {
				logrus.Warnf("[tick %07d] preemption: KV cache too small for request %s (need %d tokens, no running requests to evict)",
					ctx.Now, req.ID, numNewTokens)
				return false, adjustment
			}

			result.PreemptionHappened = true

			var victimIdx int
			switch v.preemptionPolicy {
			case PreemptionPriority:
				victimIdx = v.selectPriorityVictim(result.RunningBatch.Requests)
			default:
				victimIdx = len(result.RunningBatch.Requests) - 1
			}

			preemptedRequest := result.RunningBatch.Requests[victimIdx]
			logrus.Warnf("[tick %07d] preemption: evicting %s to make room", ctx.Now, preemptedRequest.ID)

			// Remove by index (supports non-tail eviction in priority mode).
			result.RunningBatch.Requests = append(
				result.RunningBatch.Requests[:victimIdx],
				result.RunningBatch.Requests[victimIdx+1:]...,
			)

			// Track Phase 1 index adjustment: if the victim was before the
			// caller's current position, elements shifted left under the cursor.
			// The caller must decrement reqIndex by the returned adjustment.
			// Analog of vLLM scheduler.py:853 (req_index -= 1 when preempted
			// request was in scheduled_running_reqs).
			// For FCFS, victimIdx is always the tail (>= reqIndex), so adjustment
			// is always 0 — preserving current FCFS behavior exactly.
			if victimIdx < reqIndex-adjustment {
				adjustment++
			}

			result.Preempted = append(result.Preempted, PreemptedRequest{
				Request: preemptedRequest,
			})

			// Restore token budget if preempted request was already scheduled
			// in this step (visited earlier in Phase 1, NumNewTokens > 0).
			// Reachable in priority mode when victim was at index < reqIndex
			// (already visited and allocated tokens this step).
			// With FCFS (tail-only eviction), unreachable because evicted
			// requests are always unvisited (beyond reqIndex).
			if preemptedRequest.NumNewTokens > 0 {
				*tokenBudget += int64(preemptedRequest.NumNewTokens)
				preemptedRequest.NumNewTokens = 0
			}

			preemptedRequest.State = StateQueued
			preemptedRequest.ProgressIndex = 0
			preemptedRequest.ITL = nil
			preemptedRequest.TTFTSet = false // lets the !TTFTSet guard in executeBatchStep fire on re-prefill, updating FirstTokenTime (#1122)
			ctx.KVCache.ReleaseKVBlocks(preemptedRequest)
			delete(ctx.ComputedTokens, preemptedRequest.ID)
			ctx.WaitQ.PrependFront(preemptedRequest)

			if preemptedRequest == req {
				return false, adjustment
			}
		} else {
			return true, adjustment
		}
	}
}

// selectPriorityVictim returns the index of the least-urgent running request.
// Least urgent = lowest SLOPriorityMap value (BLIS convention: higher = more urgent).
// Ties broken by latest ArrivalTime (most recently arrived evicted first).
//
// This is the BLIS equivalent of vLLM's max(priority, arrival_time) (scheduler.py:827-829).
// The conventions are inverted: vLLM uses lower=more-urgent so max() selects least urgent;
// BLIS uses higher=more-urgent so min() selects least urgent. Both tiebreak by max(ArrivalTime).
func (v *VLLMBatchFormation) selectPriorityVictim(requests []*Request) int {
	victimIdx := len(requests) - 1
	victimPri := v.sloMap.Priority(requests[victimIdx].SLOClass)
	victimArrival := requests[victimIdx].ArrivalTime

	for i := len(requests) - 2; i >= 0; i-- {
		pri := v.sloMap.Priority(requests[i].SLOClass)
		if pri < victimPri || (pri == victimPri && requests[i].ArrivalTime > victimArrival) {
			victimIdx = i
			victimPri = pri
			victimArrival = requests[i].ArrivalTime
		}
	}
	return victimIdx
}

// NewBatchFormation creates the default BatchFormation.
// preemptionPolicy selects victim strategy: "fcfs" (tail-of-batch) or "priority" (least-urgent SLO tier).
// sloMap provides SLO class → priority mapping for "priority" mode; nil uses GAIE defaults.
func NewBatchFormation(preemptionPolicy string, sloMap *SLOPriorityMap) BatchFormation {
	policy := PreemptionPolicy(preemptionPolicy)
	if policy == "" {
		policy = PreemptionFCFS
	}
	if sloMap == nil {
		sloMap = DefaultSLOPriorityMap()
	}
	return &VLLMBatchFormation{
		preemptionPolicy: policy,
		sloMap:           sloMap,
	}
}

package sim

import (
	"fmt"

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

// VictimSelector chooses which index in the running batch to evict during KV pressure.
// Returns the index of the request to preempt.
// nil means LIFO (evict the last element — default vLLM behavior).
type VictimSelector func(requests []*Request) int

// SLOLowestPriorityVictim selects the running request with the lowest SLO tier priority.
// Ties (same SLO class) resolve to the last element (LIFO among equals).
// Uses SLOTierPriority which maps: background=0, batch=1, sheddable=2, standard=3, critical=4.
func SLOLowestPriorityVictim(requests []*Request) int {
	idx := 0
	minPriority := SLOTierPriority(requests[0].SLOClass)
	for i := 1; i < len(requests); i++ {
		p := SLOTierPriority(requests[i].SLOClass)
		if p <= minPriority {
			minPriority = p
			idx = i
		}
	}
	return idx
}

// VLLMBatchFormation implements the vLLM FCFS + chunked-prefill + preemption strategy.
type VLLMBatchFormation struct {
	selectVictim VictimSelector // nil = LIFO
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
		if ok := ctx.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+numNewTokens, []int64{}); !ok {
			// Circuit breaker: empty batch means cache is too small (R19)
			if len(result.RunningBatch.Requests) == 0 {
				logrus.Warnf("[tick %07d] preemption: KV cache too small for request %s (need %d tokens, no running requests to evict)",
					ctx.Now, req.ID, numNewTokens)
				return false
			}

			result.PreemptionHappened = true
			victimIdx := len(result.RunningBatch.Requests) - 1
			if v.selectVictim != nil {
				victimIdx = v.selectVictim(result.RunningBatch.Requests)
			}
			preemptedRequest := result.RunningBatch.Requests[victimIdx]
			logrus.Warnf("[tick %07d] preemption: evicting %s to make room", ctx.Now, preemptedRequest.ID)
			result.RunningBatch.Requests = append(
				result.RunningBatch.Requests[:victimIdx],
				result.RunningBatch.Requests[victimIdx+1:]...)

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

// NewBatchFormationFromPolicy creates a BatchFormation from a PolicyConfig.
// "" or "vllm" → VLLMBatchFormation (LIFO preemption, default)
// "slo-priority-preemption" → VLLMBatchFormation with SLO-priority victim selector
// "tier-budget" → TierBudgetBatchFormation (critFrac defaults to 0.50, stdFrac to 0.70)
func NewBatchFormationFromPolicy(cfg PolicyConfig) BatchFormation {
	switch cfg.BatchFormationPolicy {
	case "", "vllm":
		return &VLLMBatchFormation{}
	case "slo-priority-preemption":
		return NewSLOPriorityBatchFormation()
	case "tier-budget":
		cf := cfg.TierBudgetCritFrac
		if cf == 0 {
			cf = 0.50
		}
		sf := cfg.TierBudgetStdFrac
		if sf == 0 {
			sf = 0.70
		}
		return NewTierBudgetBatchFormation(cf, sf)
	default:
		panic(fmt.Sprintf("NewBatchFormationFromPolicy: unknown batch formation policy %q", cfg.BatchFormationPolicy))
	}
}

// NewSLOPriorityBatchFormation creates a BatchFormation that uses SLO-priority victim
// selection during KV preemption: the lowest-SLO running request is evicted first.
// Ties in SLO class resolve to LIFO ordering (last element evicted among equals).
// All other behavior (chunked prefill, decode, scheduling) is identical to VLLMBatchFormation.
func NewSLOPriorityBatchFormation() BatchFormation {
	return &VLLMBatchFormation{selectVictim: SLOLowestPriorityVictim}
}

// TierBudgetBatchFormation partitions the per-step token budget by SLO tier.
// Critical requests get first claim (CriticalFrac × MaxScheduledTokens),
// standard gets StandardFrac × (1-CriticalFrac) × MaxScheduledTokens,
// sheddable takes the remainder.
// All preemption and scheduling behavior is delegated to an inner VLLMBatchFormation.
//
// Use NewTierBudgetBatchFormation to construct with validated fractions.
type TierBudgetBatchFormation struct {
	CriticalFrac float64 // fraction of MaxScheduledTokens for critical tier; must be in (0,1)
	StandardFrac float64 // fraction of remaining budget for standard tier; must be in (0,1)
}

// NewTierBudgetBatchFormation creates a TierBudgetBatchFormation with validated fractions.
// criticalFrac: fraction of MaxScheduledTokens reserved for critical (must be in (0,1)).
// standardFrac: fraction of remaining budget for standard (must be in (0,1)).
// Sheddable fraction = (1-criticalFrac) * (1-standardFrac).
func NewTierBudgetBatchFormation(criticalFrac, standardFrac float64) *TierBudgetBatchFormation {
	if criticalFrac <= 0 || criticalFrac >= 1 {
		panic(fmt.Sprintf("NewTierBudgetBatchFormation: CriticalFrac must be in (0,1), got %v", criticalFrac))
	}
	if standardFrac <= 0 || standardFrac >= 1 {
		panic(fmt.Sprintf("NewTierBudgetBatchFormation: StandardFrac must be in (0,1), got %v", standardFrac))
	}
	return &TierBudgetBatchFormation{
		CriticalFrac: criticalFrac,
		StandardFrac: standardFrac,
	}
}

// TierBudgets computes per-tier token budgets from a total token count.
// Returns [criticalBudget, standardBudget, sheddableBudget].
// critical = int64(maxTokens * CriticalFrac)
// standard = int64(remaining * StandardFrac)
// sheddable = remaining - standard
func (t *TierBudgetBatchFormation) TierBudgets(maxTokens int64) [3]int64 {
	critBudget := int64(float64(maxTokens) * t.CriticalFrac)
	remaining := maxTokens - critBudget
	stdBudget := int64(float64(remaining) * t.StandardFrac)
	shedBudget := remaining - stdBudget
	return [3]int64{critBudget, stdBudget, shedBudget}
}

// tierBudgetIndex maps SLO class to a budget array index (0=critical,1=standard,2=sheddable/other).
func tierBudgetIndex(sloClass string) int {
	switch sloClass {
	case "critical":
		return 0
	case "standard":
		return 1
	default:
		return 2
	}
}

// FormBatch delegates to VLLMBatchFormation for core scheduling, then applies
// a post-pass that zeroes out token grants for requests that exceed their tier budget.
// This is a soft stall: over-budget requests remain in the running batch but receive
// 0 new tokens this step and are retried next step.
func (t *TierBudgetBatchFormation) FormBatch(ctx BatchContext) BatchResult {
	if ctx.RunningBatch == nil {
		ctx.RunningBatch = &Batch{}
	}
	budgets := t.TierBudgets(ctx.MaxScheduledTokens)

	// Snapshot ComputedTokens before inner FormBatch modifies it.
	// Used to restore stalled requests' token counts (not delete — deletion resets ProgressIndex).
	prevComputed := make(map[string]int64, len(ctx.ComputedTokens))
	for k, v := range ctx.ComputedTokens {
		prevComputed[k] = v
	}

	tierUsed := [3]int64{}

	inner := &VLLMBatchFormation{}
	result := inner.FormBatch(ctx)

	// Post-pass: enforce tier budgets by zeroing grants that exceed the cap.
	for _, req := range result.RunningBatch.Requests {
		if req.NumNewTokens <= 0 {
			continue
		}
		ti := tierBudgetIndex(req.SLOClass)
		if tierUsed[ti]+int64(req.NumNewTokens) > budgets[ti] {
			// Tier budget exhausted: soft-stall this request for this step.
			ctx.ComputedTokens[req.ID] = prevComputed[req.ID] // restore, not delete
			req.NumNewTokens = 0
			continue
		}
		tierUsed[ti] += int64(req.NumNewTokens)
	}

	return result
}

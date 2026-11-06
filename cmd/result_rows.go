package cmd

// The Result's per-request rows (DEV-9, #2071): one row per generated request, with its fate,
// the times the run measured for it, and the token record the engine emitted. They replaced the
// requests array of the --metrics-path file and the []SimResult of --results-path, both removed;
// the accuracy harness reads the rows from the Result.

import (
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/spec/result"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// requestRows builds one row per created request, in creation order, from the fate ledger
// (fate and drop reason), the facts recorded at creation (arrival, prompt, tags) and the
// engine's per-request record (latencies, tokens, cache hits, instance).
//
// A latency is stated only where the request produced it (the contract's rule): the time to
// first token and the request latency for a completed request that emitted a token, and the
// time per output token for one that emitted two or more. A request that did not complete
// states none, since the run does not keep the token count of an unfinished request.
func requestRows(cs *cluster.ClusterSimulator, m *sim.Metrics) []result.RequestRow {
	facts := cs.RequestFacts()
	byID := make(map[string]sim.RequestMetrics, len(m.Requests))
	for _, rm := range m.RequestRows() {
		byID[rm.ID] = rm
	}
	recs := cs.Fates().Records()
	rows := make([]result.RequestRow, 0, len(recs))
	ends := make([]int64, 0, len(recs))
	for _, rec := range recs {
		f := facts[rec.ID]
		row := result.RequestRow{
			ID: rec.ID, Fate: result.Fate(rec.Fate), ArrivalUs: f.ArrivalUs, PromptTokens: f.PromptTokens,
			Model: f.Model, Adapter: f.Adapter, SLOClass: f.SLOClass, TenantID: f.TenantID,
			SessionID: f.SessionID, RoundIndex: f.RoundIndex, Phase: result.Phase(f.Phase),
		}
		if rec.Fate == sim.FateDropped {
			row.DropReason = result.DropReason(rec.Reason)
		}
		if a := f.Agentic; a != nil {
			row.Agentic = &result.AgenticTag{Lane: a.Lane, Tree: a.Tree, TraceID: a.TraceID, Stream: a.Stream,
				Kind: result.AgenticKind(a.Kind), StreamInstance: a.StreamInstance, Turn: a.Turn, Outcome: a.Outcome}
		}
		if rm, ok := byID[rec.ID]; ok {
			row.Instance = rm.HandledBy
			row.CachedLocalTokens, row.CachedExternalTokens = rm.CachedLocalTokens, rm.CachedExternalTokens
			if d, ok := m.RequestSchedulingDelays[rec.ID]; ok {
				row.SchedulingDelayUs = &d
			}
			if rm.GatewayQueueDelay > 0 {
				us := int64(rm.GatewayQueueDelay*1e3 + 0.5)
				row.GatewayQueueDelayUs = &us
			}
			if rec.Fate == sim.FateCompleted {
				fillCompleted(&row, rm, m)
			}
		}
		rows = append(rows, row)
		ends = append(ends, rec.At)
	}
	numberCompletions(rows, ends)
	return rows
}

// numberCompletions gives each completed row its 1-based place in the run's completion order:
// by the clock the fate was written at, ties by the engine's own completion order, then by
// creation order. The numbers are 1..completed, each used once, as the contract requires.
func numberCompletions(rows []result.RequestRow, ends []int64) {
	var idx []int
	for i := range rows {
		if rows[i].Fate == result.FateCompleted {
			idx = append(idx, i)
		}
	}
	sort.SliceStable(idx, func(a, b int) bool {
		x, y := idx[a], idx[b]
		if ends[x] != ends[y] {
			return ends[x] < ends[y]
		}
		if rows[x].CompletionIndex != rows[y].CompletionIndex {
			return rows[x].CompletionIndex < rows[y].CompletionIndex
		}
		return x < y
	})
	for rank, i := range idx {
		rows[i].CompletionIndex = rank + 1
	}
}

// fillCompleted states a completed request's token record, latencies and completion index.
func fillCompleted(row *result.RequestRow, rm sim.RequestMetrics, m *sim.Metrics) {
	row.OutputTokens = rm.OutputTokens
	row.LengthCapped = rm.LengthCapped
	row.CompletionIndex = rm.CompletionIndex
	if rm.OutputTokens < 1 {
		return
	}
	ttft := int64(m.RequestTTFTs[rm.ID])
	e2e := int64(m.RequestE2Es[rm.ID])
	row.TTFTUs, row.E2EUs = &ttft, &e2e
	if rm.OutputTokens >= 2 {
		tpot := m.RequestITLs[rm.ID] // the per-request time per output token, in ticks
		row.TPOTUs = &tpot
	}
}

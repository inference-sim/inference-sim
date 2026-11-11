package latencykernel

import (
	"sync/atomic"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// trustTerm is one selected coefficient's contribution to the trust of a price (design VAL-3).
// The kernel reads every coefficient when it is built (k.selected is complete when New
// returns), so a price can only have used entries in that list. The rank each contributes is
// fixed except for extrapolation, which depends on the step; terms with a fit range are kept
// separately so a step is checked only against them.
type trustTerm struct {
	name   string
	method vocab.Method
	// base is the effective rank when the lookup is inside the fit range, and baseFlag says
	// why it is not the method's own rank ("unidentified"), or is empty.
	base     int
	baseFlag string
	// ranged is the entry, kept when it records a fit range a step can fall outside.
	ranged *resolve.Resolved
	// at is the entry's place in the kernel's selected list.
	at int
	// extrapolated is set once any step read the entry outside its fit range, so the
	// selected-entry report can say so. Only an entry with a fit range has one.
	extrapolated *atomic.Bool
}

// buildTrust turns the selected entries into terms. It runs once, after the lookups.
func buildTrust(selected []resolve.Resolved) []trustTerm {
	terms := make([]trustTerm, 0, len(selected))
	for i := range selected {
		e := selected[i].Entry
		if isOffloadTierEntry(e.Name) {
			continue // a tier's terms price a transfer (TierTime), not a step
		}
		rank, ranked := e.EffectiveRank(false)
		if !ranked {
			continue
		}
		t := trustTerm{name: e.Name, method: e.Method, base: rank, at: i}
		if own, _ := e.Method.Rank(); rank > own {
			t.baseFlag = "unidentified"
		}
		if len(e.FitRange) > 0 {
			t.ranged = &selected[i]
			t.extrapolated = new(atomic.Bool)
		}
		terms = append(terms, t)
	}
	return terms
}

// trustOf is the least trusted effective rank among the terms for a step whose inputs are in,
// the entry that sets it, and why. Ties go to the entry whose name sorts first, so the answer
// does not depend on lookup order.
func trustOf(terms []trustTerm, in map[string]float64) kernel.Trust {
	var worst kernel.Trust
	consider := func(rank int, name string, m vocab.Method, flag string) {
		if rank > worst.Rank || (rank == worst.Rank && rank > 0 && name < worst.Entry) {
			worst = kernel.Trust{Rank: rank, Method: string(m), Flag: flag, Entry: name}
		}
	}
	for _, t := range terms {
		rank, flag := t.base, t.baseFlag
		if t.ranged != nil && t.ranged.Entry.Extrapolated(in) {
			if !t.extrapolated.Load() {
				t.extrapolated.Store(true)
			}
			if r, _ := t.ranged.Entry.EffectiveRank(true); r > rank {
				rank, flag = r, "extrapolated"
			}
		}
		consider(rank, t.name, t.method, flag)
	}
	return worst
}

// extrapolatedEntries is the set of selected entries, by place in the selected list, that a
// step priced so far read outside their fit range.
func extrapolatedEntries(terms []trustTerm) map[int]bool {
	out := map[int]bool{}
	for _, t := range terms {
		if t.extrapolated != nil && t.extrapolated.Load() {
			out[t.at] = true
		}
	}
	return out
}

// inputs is the step's coordinates on the fit dimensions of design VAL-10: the tokens one
// rank runs, the requests in the step, and the longest context any request reads.
func (s *stepShape) inputs() map[string]float64 {
	longest := 0
	for _, c := range s.decodeContexts {
		if c > longest {
			longest = c
		}
	}
	for _, c := range s.prefillChunks {
		if end := c.prefix + c.sched; end > longest {
			longest = end
		}
	}
	return map[string]float64{
		string(vocab.FitTokens):        float64(s.tokens),
		string(vocab.FitRequests):      float64(s.prefillRequests + s.decodeRequests),
		string(vocab.FitContextTokens): float64(longest),
	}
}

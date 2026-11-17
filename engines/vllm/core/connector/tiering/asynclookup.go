// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/async_lookup.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).

package tiering

import "fmt"

type lookupPhase int

const (
	phasePending  lookupPhase = iota // in the step's batch, not yet submitted
	phaseInFlight                    // submitted to the worker
	phaseResolved                    // a verdict applied
)

// lookupState is LookupState (async_lookup.py:60-67).
type lookupState struct {
	generation int
	phase      lookupPhase
	result     *bool
	reqs       map[string]bool
}

type lookupItem struct {
	key        Key
	req        *ReqContext
	generation int
}

type lookupResult struct {
	key        Key
	generation int
	found      bool
}

// asyncLookup is AsyncLookupManager (async_lookup.py:70-290) with its worker thread replaced by
// Resolve: the host runs each submitted batch when the worker would have finished it. batch is
// batch_lookup, which answers a whole request's keys at once.
type asyncLookup struct {
	batch   func(keys []Key, c *ReqContext) []bool
	states  map[Key]*lookupState
	reqKeys map[string]map[Key]bool
	nextGen int
	step    []lookupItem
	queue   [][]lookupItem   // _lookup_queue: submitted, not yet run
	results [][]lookupResult // _pending_results: run, not yet drained
	drain   bool             // _need_to_drain
}

func newAsyncLookup(batch func(keys []Key, c *ReqContext) []bool) *asyncLookup {
	return &asyncLookup{batch: batch, states: map[Key]*lookupState{}, reqKeys: map[string]map[Key]bool{}}
}

// lookup is lookup (async_lookup.py:144-165): nil until the key's batch has resolved.
func (a *asyncLookup) lookup(k Key, c *ReqContext) *bool {
	if a.drain {
		a.drainResults()
		a.drain = false
	}
	st := a.states[k]
	if st == nil {
		st = &lookupState{generation: a.nextGen, reqs: map[string]bool{}}
		a.nextGen++
		a.states[k] = st
		a.step = append(a.step, lookupItem{key: k, req: c, generation: st.generation})
	}
	st.reqs[c.ReqID] = true
	if a.reqKeys[c.ReqID] == nil {
		a.reqKeys[c.ReqID] = map[Key]bool{}
	}
	a.reqKeys[c.ReqID][k] = true
	return st.result
}

// flush is flush (async_lookup.py:167-189), from the tier's on_schedule_end.
func (a *asyncLookup) flush() {
	a.drainResults()
	a.drain = true
	batch := a.step
	a.step = nil
	var inFlight []lookupItem
	for _, it := range batch {
		st := a.states[it.key]
		if st == nil || st.generation != it.generation {
			continue
		}
		if st.phase != phasePending {
			panic("tiering: a batched lookup is not pending")
		}
		st.phase = phaseInFlight
		inFlight = append(inFlight, it)
	}
	if len(inFlight) > 0 {
		a.queue = append(a.queue, inFlight)
	}
}

// Submitted is how many flushed batches the worker has not run yet.
func (a *asyncLookup) submitted() int { return len(a.queue) }

// resolve runs every submitted batch, as the worker thread does (_worker, async_lookup.py:254-290):
// within a batch, the keys of each request in first-seen order, one batch_lookup per request.
func (a *asyncLookup) resolve() {
	for _, pending := range a.queue {
		var order []string
		byReq := map[string][]lookupItem{}
		ctx := map[string]*ReqContext{}
		for _, it := range pending {
			id := it.req.ReqID
			if _, ok := byReq[id]; !ok {
				order = append(order, id)
				ctx[id] = it.req
			}
			byReq[id] = append(byReq[id], it)
		}
		var out []lookupResult
		for _, id := range order {
			items := byReq[id]
			keys := make([]Key, len(items))
			for i, it := range items {
				keys[i] = it.key
			}
			hits := a.batch(keys, ctx[id])
			for i, it := range items {
				out = append(out, lookupResult{key: it.key, generation: it.generation, found: hits[i]})
			}
		}
		if len(out) > 0 {
			a.results = append(a.results, out)
		}
	}
	a.queue = nil
}

// drainResults is drain_results (async_lookup.py:191-218).
func (a *asyncLookup) drainResults() {
	for _, batch := range a.results {
		for _, r := range batch {
			st := a.states[r.key]
			if st == nil || st.generation != r.generation {
				continue
			}
			if len(st.reqs) == 0 {
				delete(a.states, r.key)
				continue
			}
			if st.phase != phaseInFlight || st.result != nil {
				panic(fmt.Sprintf("tiering: lookup %x received a second result", []byte(r.key)))
			}
			found := r.found
			st.result = &found
			st.phase = phaseResolved
		}
	}
	a.results = nil
}

// markMiss is mark_miss (async_lookup.py:220-230): a failed load's keys become misses, unless a
// newer probe of the key is still pending or in flight.
func (a *asyncLookup) markMiss(keys []Key) {
	for _, k := range keys {
		if st := a.states[k]; st != nil && st.phase == phaseResolved {
			f := false
			st.result = &f
		}
	}
}

// cleanup is cleanup (async_lookup.py:232-242).
func (a *asyncLookup) cleanup(id string) {
	for k := range a.reqKeys[id] {
		st := a.states[k]
		delete(st.reqs, id)
		if len(st.reqs) == 0 && st.phase != phaseInFlight {
			delete(a.states, k)
		}
	}
	delete(a.reqKeys, id)
}

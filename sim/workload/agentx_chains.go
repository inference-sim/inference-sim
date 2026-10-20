package workload

import (
	"container/heap"
	"fmt"
	"math"
	"sort"
)

// This file ports aiperf 0.12.0's flattened-agent chain detection
// (dataset/loader/weka_agent_chains.py) and the preamble rule beside it
// (dataset/loader/weka_trace.py _split_off_preamble). aiperf runs it by default
// (WEKA_SPLIT_FLATTENED_AGENTS=True) on a trace's top-level requests and, nested, on
// each subagent's inner requests: it decides which requests form one sequential
// stream, so it decides the replay's streams and their concurrency. Comments cite the
// Python function each piece mirrors.

const agxEpsilonSeconds = 1e-6

// Defaults of aiperf's Environment.DATASET knobs (common/environment.py:383-537).
const (
	agxSeamMaxGapSeconds    = 3600.0
	agxSeamMinOverlapRatio  = 0.5
	agxAuxMaxRequests       = 1
	agxAuxISLRatio          = 0.10
	agxAuxISLFloor          = 16384
	agxAuxReductionOSLMax   = 4000
	agxAuxReductionRatio    = 20.0
	agxWorkerGroupMin       = 3
	agxTitleGenMaxOutTokens = 64
)

// agxReq is one recorded request as chain detection and plan building see it.
type agxReq struct {
	idx   int // outer index in trace.requests, or inner index within a subagent entry
	t     float64
	api   float64
	apiOK bool // api_time present and finite
	model string
	in    int
	out   int
	hash  []int64
}

// end is _req_end / _request_end_seconds: t + max(api, 0), missing api counting as 0.
func (r *agxReq) end() float64 {
	d := 0.0
	if r.apiOK {
		d = r.api
	}
	return r.t + math.Max(d, 0)
}

func agxLCP(a, b []int64) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

type agxFork struct {
	parentChain int // -1 = no shared context
	forkOuter   int // -1 = none
	depth       int
	forkTime    float64
}

type agxChain struct {
	reqs        []*agxReq
	fork        *agxFork
	splicedInto int // -1 = live
	tailOuter   int
	tailHash    []int64
	tailEnd     float64
	tailModel   string
}

type agxDetection struct {
	chains     []*agxChain
	mainIndex  int
	workers    []int
	seamsMerge int
}

type agxPhase1 struct {
	chains         []*agxChain
	chainOfRequest map[int]int
	forksByTail    map[int][]int
	reqByOuter     map[int]*agxReq
}

func newAgxChain(f *agxFork) *agxChain {
	return &agxChain{fork: f, splicedInto: -1, tailOuter: -1}
}

func (s *agxPhase1) appendTo(ci int, r *agxReq) {
	c := s.chains[ci]
	c.reqs = append(c.reqs, r)
	s.chainOfRequest[r.idx] = ci
	if len(r.hash) > 0 {
		c.tailOuter = r.idx
		c.tailHash = r.hash
		c.tailEnd = r.end()
		c.tailModel = r.model
	}
}

// classify is _Phase1State.classify.
func (s *agxPhase1) classify(r *agxReq) {
	s.reqByOuter[r.idx] = r
	if len(r.hash) == 0 {
		if len(s.chains) == 0 {
			s.chains = append(s.chains, newAgxChain(nil))
		}
		s.chains[0].reqs = append(s.chains[0].reqs, r)
		s.chainOfRequest[r.idx] = 0
		return
	}
	if len(s.chains) == 0 {
		s.chains = append(s.chains, newAgxChain(nil))
		s.appendTo(0, r)
		return
	}
	if target := agxExtensionTarget(s.chains, r.hash, r.t, r.model); target >= 0 {
		s.appendTo(target, r)
		return
	}
	parent, depth := agxMaxLCPChain(s.chains, r.hash)
	if parent < 0 {
		allEmpty := true
		for _, c := range s.chains {
			if len(c.tailHash) > 0 {
				allEmpty = false
				break
			}
		}
		if allEmpty {
			s.appendTo(0, r)
			return
		}
	}
	f := &agxFork{parentChain: parent, forkOuter: -1, depth: depth, forkTime: r.t}
	if parent >= 0 {
		f.forkOuter = s.chains[parent].tailOuter
	}
	ni := len(s.chains)
	s.chains = append(s.chains, newAgxChain(f))
	s.appendTo(ni, r)
	if f.forkOuter >= 0 && depth > 0 {
		s.forksByTail[f.forkOuter] = append(s.forksByTail[f.forkOuter], ni)
	}
}

// agxExtensionTarget is _find_extension_target: deepest same-model tail that is a full
// prefix of h and ended by t; ties to the lowest index.
func agxExtensionTarget(chains []*agxChain, h []int64, t float64, model string) int {
	best, bestLen := -1, -1
	for i, c := range chains {
		tl := len(c.tailHash)
		if tl == 0 || tl > len(h) || tl <= bestLen {
			continue
		}
		if c.tailModel != model || c.tailEnd > t+agxEpsilonSeconds {
			continue
		}
		if c.tailHash[tl-1] != h[tl-1] {
			continue
		}
		if agxLCP(c.tailHash, h[:tl]) == tl {
			best, bestLen = i, tl
		}
	}
	return best
}

// agxMaxLCPChain is _max_lcp_chain: deepest LCP, then deeper tail, then lower index.
func agxMaxLCPChain(chains []*agxChain, h []int64) (int, int) {
	best, bd, bl := -1, 0, 0
	for i, c := range chains {
		if len(c.tailHash) == 0 {
			continue
		}
		d := agxLCP(c.tailHash, h)
		if d == 0 {
			continue
		}
		if d > bd || (d == bd && len(c.tailHash) > bl) {
			best, bd, bl = i, d, len(c.tailHash)
		}
	}
	return best, bd
}

// agxSortByTimeIdx sorts requests by (t, idx), the order detection processes them in.
func agxSortByTimeIdx(rs []*agxReq) {
	sort.SliceStable(rs, func(i, j int) bool {
		if rs[i].t != rs[j].t {
			return rs[i].t < rs[j].t
		}
		return rs[i].idx < rs[j].idx
	})
}

// agxDetectChains is detect_agent_chains.
func agxDetectChains(normals []*agxReq) *agxDetection {
	if len(normals) == 0 {
		return &agxDetection{}
	}
	ordered := append([]*agxReq(nil), normals...)
	agxSortByTimeIdx(ordered)
	s := &agxPhase1{chainOfRequest: map[int]int{}, forksByTail: map[int][]int{}, reqByOuter: map[int]*agxReq{}}
	for _, r := range ordered {
		s.classify(r)
	}
	seams := agxResolveSeams(s)
	alias := map[int]int{}
	for i, c := range s.chains {
		if c.splicedInto >= 0 {
			alias[i] = c.splicedInto
		}
	}
	resolve := func(i int) int {
		for {
			j, ok := alias[i]
			if !ok {
				return i
			}
			i = j
		}
	}
	for _, c := range s.chains {
		if c.splicedInto < 0 && c.fork != nil && c.fork.parentChain >= 0 {
			c.fork.parentChain = resolve(c.fork.parentChain)
		}
	}
	d := &agxDetection{chains: s.chains, mainIndex: resolve(s.chainOfRequest[ordered[0].idx]), seamsMerge: seams}
	for i, c := range s.chains {
		if c.splicedInto < 0 && i != d.mainIndex {
			d.workers = append(d.workers, i)
		}
	}
	sort.SliceStable(d.workers, func(a, b int) bool {
		ra, rb := s.chains[d.workers[a]].reqs[0], s.chains[d.workers[b]].reqs[0]
		if ra.t != rb.t {
			return ra.t < rb.t
		}
		return ra.idx < rb.idx
	})
	return d
}

func agxLastHashOuter(c *agxChain) int {
	for i := len(c.reqs) - 1; i >= 0; i-- {
		if len(c.reqs[i].hash) > 0 {
			return c.reqs[i].idx
		}
	}
	return -1
}

// agxElectContinuation is _elect_continuation.
func agxElectContinuation(chains []*agxChain, registered []int, tail *agxReq) int {
	tEnd := tail.end()
	tailBlocks := len(tail.hash)
	best := -1
	for _, ci := range registered {
		c := chains[ci]
		if c.fork == nil || c.fork.depth <= 0 {
			continue
		}
		first := c.reqs[0]
		if tEnd > first.t+agxEpsilonSeconds || first.model != tail.model {
			continue
		}
		if tailBlocks != 0 {
			gap := first.t - tEnd
			overlap := float64(c.fork.depth) / float64(tailBlocks)
			if gap > agxSeamMaxGapSeconds && overlap < agxSeamMinOverlapRatio {
				continue
			}
		}
		if best < 0 {
			best = ci
			continue
		}
		b := chains[best]
		// max by (depth, -fork_time, -index)
		switch {
		case c.fork.depth != b.fork.depth:
			if c.fork.depth > b.fork.depth {
				best = ci
			}
		case c.fork.forkTime != b.fork.forkTime:
			if c.fork.forkTime < b.fork.forkTime {
				best = ci
			}
		case ci < best:
			best = ci
		}
	}
	return best
}

type agxIntHeap []int

func (h agxIntHeap) Len() int           { return len(h) }
func (h agxIntHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h agxIntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *agxIntHeap) Push(x any)        { *h = append(*h, x.(int)) }
func (h *agxIntHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

// agxResolveSeams is _resolve_seams (phase 2) with _rekey_leftover_forks.
func agxResolveSeams(s *agxPhase1) int {
	alias := map[int]int{}
	resolve := func(i int) int {
		for {
			j, ok := alias[i]
			if !ok {
				return i
			}
			i = j
		}
	}
	keys := make(agxIntHeap, 0, len(s.forksByTail))
	for k := range s.forksByTail {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	heap.Init(&keys)
	processed := map[int]bool{}
	seams := 0
	for keys.Len() > 0 {
		fo := heap.Pop(&keys).(int)
		if processed[fo] {
			continue
		}
		processed[fo] = true
		owner := resolve(s.chainOfRequest[fo])
		oc := s.chains[owner]
		if agxLastHashOuter(oc) != fo {
			continue
		}
		var registered []int
		for _, ci := range s.forksByTail[fo] {
			if s.chains[ci].splicedInto < 0 {
				registered = append(registered, ci)
			}
		}
		elected := agxElectContinuation(s.chains, registered, s.reqByOuter[fo])
		if elected < 0 {
			continue
		}
		target := s.chains[elected]
		oc.reqs = append(oc.reqs, target.reqs...)
		for _, r := range target.reqs {
			s.chainOfRequest[r.idx] = owner
		}
		target.splicedInto = owner
		alias[elected] = owner
		seams++
		newTail := agxLastHashOuter(oc)
		if newTail < 0 {
			continue
		}
		newHash := s.reqByOuter[newTail].hash
		rekeyed := false
		for _, ci := range registered {
			c := s.chains[ci]
			if ci == elected || c.splicedInto >= 0 || c.fork == nil {
				continue
			}
			d := agxLCP(newHash, c.reqs[0].hash)
			if d <= 0 {
				continue
			}
			c.fork.forkOuter = newTail
			c.fork.depth = d
			c.fork.parentChain = owner
			s.forksByTail[newTail] = append(s.forksByTail[newTail], ci)
			rekeyed = true
		}
		if rekeyed {
			delete(processed, newTail)
			heap.Push(&keys, newTail)
		}
	}
	return seams
}

// agxSplitOffPreamble is _split_off_preamble: the single earliest request leaves
// detection when it shares no prefix with any other request and is either small
// (out <= 64) or fully block-disjoint. It returns (preamble, rest) with rest in
// original index order, or (nil, normals) when nothing is split.
func agxSplitOffPreamble(normals []*agxReq) ([]*agxReq, []*agxReq) {
	if len(normals) < 2 {
		return nil, normals
	}
	ordered := append([]*agxReq(nil), normals...)
	agxSortByTimeIdx(ordered)
	first := ordered[0]
	if len(first.hash) == 0 {
		return nil, normals
	}
	rest := ordered[1:]
	for _, o := range rest {
		if len(o.hash) > 0 && agxLCP(first.hash, o.hash) > 0 {
			return nil, normals
		}
	}
	if first.out > agxTitleGenMaxOutTokens {
		other := map[int64]struct{}{}
		for _, o := range rest {
			for _, h := range o.hash {
				other[h] = struct{}{}
			}
		}
		for _, h := range first.hash {
			if _, ok := other[h]; ok {
				return nil, normals
			}
		}
	}
	out := append([]*agxReq(nil), rest...)
	sort.SliceStable(out, func(i, j int) bool { return out[i].idx < out[j].idx })
	return []*agxReq{first}, out
}

// agxIsAuxChain is is_aux_chain with the environment defaults.
func agxIsAuxChain(reqs []*agxReq, mainPeakISL int, mainModel string, hasMainModel bool) bool {
	if len(reqs) == 0 || len(reqs) > agxAuxMaxRequests {
		return false
	}
	first := reqs[0]
	if hasMainModel && first.model != mainModel {
		return true
	}
	threshold := max(agxAuxISLFloor, int(agxAuxISLRatio*float64(mainPeakISL)))
	return first.in < threshold
}

// agxIsReductionChain is is_reduction_chain with the environment defaults.
func agxIsReductionChain(reqs []*agxReq) bool {
	if len(reqs) != 1 {
		return false
	}
	f := reqs[0]
	if f.out <= 0 || f.out >= agxAuxReductionOSLMax || f.in < agxAuxISLFloor {
		return false
	}
	return float64(f.in) > agxAuxReductionRatio*float64(f.out)
}

// agxWorkerGroups is worker_group_assignment: chain index -> (group, member).
func agxWorkerGroups(d *agxDetection) map[int][2]int {
	type cand struct {
		t0, t1 float64
		first  int
		ci     int
	}
	type bkey struct{ parent, outer int }
	buckets := map[bkey][]cand{}
	var order []bkey
	for _, ci := range d.workers {
		c := d.chains[ci]
		f := c.fork
		if f == nil || f.depth <= 0 || f.parentChain < 0 || f.forkOuter < 0 || len(c.reqs) == 0 {
			continue
		}
		t1 := math.Inf(-1)
		for _, r := range c.reqs {
			t1 = math.Max(t1, r.end())
		}
		k := bkey{f.parentChain, f.forkOuter}
		if _, ok := buckets[k]; !ok {
			order = append(order, k)
		}
		buckets[k] = append(buckets[k], cand{c.reqs[0].t, t1, c.reqs[0].idx, ci})
	}
	var comps [][]cand
	for _, k := range order {
		cs := buckets[k]
		idx := make([]int, len(cs))
		for i := range idx {
			idx[i] = i
		}
		sort.SliceStable(idx, func(a, b int) bool {
			if cs[idx[a]].t0 != cs[idx[b]].t0 {
				return cs[idx[a]].t0 < cs[idx[b]].t0
			}
			return cs[idx[a]].first < cs[idx[b]].first
		})
		parent := make([]int, len(cs))
		for i := range parent {
			parent[i] = i
		}
		find := func(x int) int {
			for parent[x] != x {
				parent[x] = parent[parent[x]]
				x = parent[x]
			}
			return x
		}
		// active: min-heap of (t1, idx), as the Python sweep keeps it.
		var active [][2]float64
		less := func(a, b [2]float64) bool { return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]) }
		push := func(x [2]float64) {
			active = append(active, x)
			for i := len(active) - 1; i > 0; {
				p := (i - 1) / 2
				if !less(active[i], active[p]) {
					break
				}
				active[i], active[p] = active[p], active[i]
				i = p
			}
		}
		pop := func() {
			n := len(active) - 1
			active[0] = active[n]
			active = active[:n]
			for i := 0; ; {
				l, r, m := 2*i+1, 2*i+2, i
				if l < n && less(active[l], active[m]) {
					m = l
				}
				if r < n && less(active[r], active[m]) {
					m = r
				}
				if m == i {
					break
				}
				active[i], active[m] = active[m], active[i]
				i = m
			}
		}
		for _, i := range idx {
			t0 := cs[i].t0
			for len(active) > 0 && active[0][0] <= t0 {
				pop()
			}
			if len(active) > 0 {
				parent[find(i)] = find(int(active[0][1]))
			}
			push([2]float64{cs[i].t1, float64(i)})
		}
		groups := map[int][]cand{}
		var gorder []int
		for i := range cs {
			r := find(i)
			if _, ok := groups[r]; !ok {
				gorder = append(gorder, r)
			}
			groups[r] = append(groups[r], cs[i])
		}
		for _, r := range gorder {
			comps = append(comps, groups[r])
		}
	}
	var groups [][]cand
	for _, c := range comps {
		if len(c) >= agxWorkerGroupMin {
			groups = append(groups, c)
		}
	}
	minKey := func(c []cand) (float64, int) {
		bt, bf := math.Inf(1), math.MaxInt
		for _, x := range c {
			if x.t0 < bt || (x.t0 == bt && x.first < bf) {
				bt, bf = x.t0, x.first
			}
		}
		return bt, bf
	}
	sort.SliceStable(groups, func(a, b int) bool {
		ta, fa := minKey(groups[a])
		tb, fb := minKey(groups[b])
		if ta != tb {
			return ta < tb
		}
		return fa < fb
	})
	out := map[int][2]int{}
	for g, comp := range groups {
		sort.SliceStable(comp, func(a, b int) bool {
			if comp[a].t0 != comp[b].t0 {
				return comp[a].t0 < comp[b].t0
			}
			return comp[a].first < comp[b].first
		})
		for m, x := range comp {
			out[x.ci] = [2]int{g, m}
		}
	}
	return out
}

// agxWorkerSuffix is _worker_suffix.
func agxWorkerSuffix(n int, aux bool, wg *[2]int) string {
	if aux {
		return fmt.Sprintf("aux:%03d", n)
	}
	if wg != nil {
		return fmt.Sprintf("wg:%03d_%03d", wg[0], wg[1])
	}
	return fmt.Sprintf("fa:%03d", n)
}

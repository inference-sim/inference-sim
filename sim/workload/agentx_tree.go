package workload

import (
	"fmt"
	"math"
	"sort"
)

// This file rebuilds one Weka v7 trace as the session tree aiperf 0.12.0 replays under
// --scenario inferencex-agentx-mvp: the root conversation, its subagent and flattened
// worker chains as child conversations, the SPAWN branches and SPAWN_JOIN gates that link
// them, per-turn timestamps and end-to-start delays, and the recorded cross-stream
// replay barriers. It ports WekaTraceLoader._build_reconstruction_plans,
// _reconstruct_serial and _install_replay_dependencies (dataset/loader/weka_trace.py)
// and infer_cross_stream_predecessors (timing/replay_dependencies.py), with no per-trace
// idle-gap warp and no per-turn delay cap: the scenario forbids both.

// agxTurn is one replayed request of a conversation (aiperf's Turn).
type agxTurn struct {
	tsMs      float64 // recorded start, ms (t*1000)
	delayMs   float64 // end-to-start delay after the previous turn completes; turn 0: none
	hasDelay  bool
	apiMs     float64 // recorded server time, ms (max(0, api_time)*1000)
	in        int32   // recorded prompt tokens
	maxTokens int32   // max(out, 1): ignore_eos makes this the generated count
	hashNode  int32   // trie node of the request's full hash_ids list (-1: empty)
	nHash     int32
	spawns    []int32 // branches declared on this turn
	gates     []int32 // branches whose SPAWN_JOIN gates this turn
	preds     []agxRef
}

// agxRef names one turn of the same trace: conversation index, turn index.
type agxRef struct{ conv, turn int32 }

type agxConv struct {
	sessionID string
	scope     int32 // replay-barrier scope: 0 = root scope, s+1 = subagent s
	turns     []agxTurn
	// branch spawning metadata, resolved once (trajectory_source._branch_runtimes).
}

type agxBranch struct {
	id         string
	children   []int32
	background bool
	startMs    float64
	spawnTurn  int32 // first root turn declaring the branch
	joinTurn   int32 // first root turn gated on it; -1 when background or none
}

// AgentXTrace is one rebuilt trace: conversation 0 is the root.
type AgentXTrace struct {
	ID       string
	Peak     int // max over parent and subagent requests of in + max(out, 1)
	convs    []agxConv
	branches []agxBranch
	// hash-id trie: node i holds hash ids[i] under parent[i]; a request's prompt blocks
	// are the path from its node to the root, which keeps one copy of every shared prefix.
	trieParent []int32
	trieHash   []int64
	minMs      float64 // first and last recorded request start, ms, over every conversation
	maxMs      float64
}

// agxTrieBuilder interns hash-id prefixes for one trace.
type agxTrieBuilder struct {
	parent []int32
	hash   []int64
	child  map[agxTrieKey]int32
}

type agxTrieKey struct {
	parent int32
	hash   int64
}

func (b *agxTrieBuilder) intern(ids []int64) int32 {
	node := int32(-1)
	for _, h := range ids {
		k := agxTrieKey{node, h}
		c, ok := b.child[k]
		if !ok {
			c = int32(len(b.parent))
			b.parent = append(b.parent, node)
			b.hash = append(b.hash, h)
			b.child[k] = c
		}
		node = c
	}
	return node
}

// hashPrefix writes the first n hash ids of the request at node (whose list is nHash
// long) into dst and returns it.
func (t *AgentXTrace) hashPrefix(node, nHash int32, n int, dst []int64) []int64 {
	dst = dst[:0]
	if n <= 0 || node < 0 {
		return dst
	}
	for d := nHash; d > int32(n); d-- {
		node = t.trieParent[node]
	}
	if cap(dst) < n {
		dst = make([]int64, n)
	} else {
		dst = dst[:n]
	}
	for i := n - 1; i >= 0; i-- {
		dst[i] = t.trieHash[node]
		node = t.trieParent[node]
	}
	return dst
}

// wekaPeak is _trace_peak_context_length with no --synthesis-max-osl.
func wekaPeak(s *wekaSession) int {
	peak := 0
	add := func(r *wekaRequest) {
		out := 0
		if r.Out != nil {
			out = *r.Out
		}
		in := 0
		if r.In != nil {
			in = *r.In
		}
		peak = max(peak, in+max(out, 1))
	}
	for i := range s.Requests {
		r := &s.Requests[i]
		if r.Type == "subagent" {
			for j := range r.Requests {
				add(&r.Requests[j])
			}
			continue
		}
		add(r)
	}
	return peak
}

func toAgxReq(idx int, r *wekaRequest) (*agxReq, error) {
	if r.In == nil || r.Out == nil {
		return nil, fmt.Errorf("request %d has no in/out token counts", idx)
	}
	return &agxReq{idx: idx, t: r.T, api: r.API, apiOK: true, model: r.Model, in: *r.In, out: *r.Out, hash: r.HashIDs}, nil
}

type agxChildPlan struct {
	sessionID string
	saIndex   int
	reqs      []*agxReq
}

type agxFlatPlan struct {
	sessionID  string
	chainIndex int
	reqs       []*agxReq
}

// agxClassify names a worker chain: aux sidecar, worker-group member or solo agent.
func agxClassify(n int, chain []*agxReq, mainPeak int, mainModel string, hasModel bool, wg *[2]int) string {
	aux := agxIsAuxChain(chain, mainPeak, mainModel, hasModel)
	red := !aux && agxIsReductionChain(chain)
	if aux || red {
		wg = nil
	}
	return agxWorkerSuffix(n, aux || red, wg)
}

// agxExpandSubagent is _expand_subagent_to_child_plans.
func agxExpandSubagent(traceID string, saIndex int, e *wekaRequest, split bool) ([]agxChildPlan, error) {
	var norm []*agxReq
	for i := range e.Requests {
		r, err := toAgxReq(i, &e.Requests[i])
		if err != nil {
			return nil, fmt.Errorf("subagent %q: %w", e.AgentID, err)
		}
		if r.t+agxEpsilonSeconds < e.T {
			r.t = e.T + r.t
		}
		norm = append(norm, r)
	}
	var chains [][]*agxReq
	var classifyMain []*agxReq
	var wgOf []*[2]int
	if !split || len(norm) == 0 {
		o := append([]*agxReq(nil), norm...)
		agxSortByTimeIdx(o)
		chains = [][]*agxReq{o}
		classifyMain = o
		wgOf = []*[2]int{nil}
	} else {
		pre, rest := agxSplitOffPreamble(norm)
		d := agxDetectChains(rest)
		detectedMain := append([]*agxReq(nil), d.chains[d.mainIndex].reqs...)
		mainReqs := detectedMain
		if len(pre) > 0 {
			mainReqs = append(append([]*agxReq(nil), pre...), detectedMain...)
			agxSortByTimeIdx(mainReqs)
		}
		chains = [][]*agxReq{mainReqs}
		wgOf = []*[2]int{nil}
		wg := agxWorkerGroups(d)
		for _, ci := range d.workers {
			chains = append(chains, d.chains[ci].reqs)
			if c, ok := wg[ci]; ok {
				wgOf = append(wgOf, &c)
			} else {
				wgOf = append(wgOf, nil)
			}
		}
		classifyMain = detectedMain
	}
	mainPeak := 0
	for _, r := range classifyMain {
		mainPeak = max(mainPeak, r.in)
	}
	mainModel, hasModel := "", len(classifyMain) > 0
	if hasModel {
		mainModel = classifyMain[0].model
	}
	plans := make([]agxChildPlan, 0, len(chains))
	for ci, reqs := range chains {
		sid := fmt.Sprintf("%s::sa:%s", traceID, e.AgentID)
		if ci > 0 {
			sid += ":" + agxClassify(ci-1, reqs, mainPeak, mainModel, hasModel, wgOf[ci])
		}
		plans = append(plans, agxChildPlan{sessionID: sid, saIndex: saIndex, reqs: reqs})
	}
	return plans, nil
}

// agxSplitFlat is _detect_and_split_flat_chains: it returns the root's remaining requests
// and the detected flat worker chains.
func agxSplitFlat(traceID string, normals []*agxReq) ([]*agxReq, []agxFlatPlan) {
	pre, rest := agxSplitOffPreamble(normals)
	d := agxDetectChains(rest)
	if len(d.workers) == 0 {
		return normals, nil
	}
	main := d.chains[d.mainIndex]
	mainPeak := 0
	for _, r := range main.reqs {
		mainPeak = max(mainPeak, r.in)
	}
	mainModel, hasModel := "", len(main.reqs) > 0
	if hasModel {
		mainModel = main.reqs[0].model
	}
	wg := agxWorkerGroups(d)
	var flats []agxFlatPlan
	for n, ci := range d.workers {
		var w *[2]int
		if c, ok := wg[ci]; ok {
			w = &c
		}
		reqs := d.chains[ci].reqs
		flats = append(flats, agxFlatPlan{
			sessionID:  traceID + "::" + agxClassify(n, reqs, mainPeak, mainModel, hasModel, w),
			chainIndex: n,
			reqs:       append([]*agxReq(nil), reqs...),
		})
	}
	mainReqs := append([]*agxReq(nil), main.reqs...)
	if len(pre) > 0 {
		mainReqs = append(append([]*agxReq(nil), pre...), mainReqs...)
		agxSortByTimeIdx(mainReqs)
	}
	return mainReqs, flats
}

// agxSubagentEnd is _sa_end_seconds.
func agxSubagentEnd(e *wekaRequest) float64 {
	if e.DurationMs != nil {
		d := *e.DurationMs / 1000.0
		if math.IsNaN(d) || math.IsInf(d, 0) || d < 0 {
			d = 0
		}
		return e.T + d
	}
	if len(e.Requests) > 0 {
		end := math.Inf(-1)
		for i := range e.Requests {
			r := &e.Requests[i]
			t := r.T
			if t+agxEpsilonSeconds < e.T {
				t = e.T + t
			}
			end = math.Max(end, t+math.Max(r.API, 0))
		}
		return end
	}
	return e.T
}

// agxTurnsFor builds a conversation's turns: aiperf's per-turn timestamp, end-to-start
// delay (_end_to_start_delay_ms) and api_time_ms.
func agxTurnsFor(reqs []*agxReq, trie *agxTrieBuilder) []agxTurn {
	turns := make([]agxTurn, len(reqs))
	for k, r := range reqs {
		tsMs := float64(r.t * 1000.0) // explicit conversion: no fused multiply-add (Go spec), as CPython computes it
		tn := agxTurn{
			tsMs:      tsMs,
			apiMs:     math.Max(0, r.api) * 1000.0,
			in:        int32(r.in),
			maxTokens: int32(max(r.out, 1)),
			hashNode:  trie.intern(r.hash),
			nHash:     int32(len(r.hash)),
		}
		if k > 0 {
			p := reqs[k-1]
			tn.delayMs = math.Max(0, (tsMs-float64(p.t*1000.0))-float64(p.api*1000.0))
			tn.hasDelay = true
		}
		turns[k] = tn
	}
	return turns
}

// BuildAgentXTrace rebuilds one Weka v7 trace as aiperf's session tree. split selects
// flattened-agent chain splitting (aiperf's default, WEKA_SPLIT_FLATTENED_AGENTS).
func BuildAgentXTrace(s *wekaSession, split bool) (*AgentXTrace, error) {
	if s.ID == "" {
		return nil, fmt.Errorf("weka trace has no id")
	}
	if s.BlockSize != 0 && s.BlockSize != agxBlockSize {
		return nil, fmt.Errorf("trace %q: block_size %d, the replay models %d-token blocks", s.ID, s.BlockSize, agxBlockSize)
	}
	if s.ToolTokens != 0 || s.SystemTokens != 0 {
		return nil, fmt.Errorf("trace %q declares tool/system tokens (%d/%d), which the replay does not model", s.ID, s.ToolTokens, s.SystemTokens)
	}
	var normals []*agxReq
	type saEntry struct {
		outer int
		e     *wekaRequest
	}
	var subagents []saEntry
	var childPlans []agxChildPlan
	for i := range s.Requests {
		r := &s.Requests[i]
		if r.Type == "subagent" {
			plans, err := agxExpandSubagent(s.ID, len(subagents), r, split)
			if err != nil {
				return nil, fmt.Errorf("trace %q: %w", s.ID, err)
			}
			subagents = append(subagents, saEntry{i, r})
			childPlans = append(childPlans, plans...)
			continue
		}
		q, err := toAgxReq(i, r)
		if err != nil {
			return nil, fmt.Errorf("trace %q: %w", s.ID, err)
		}
		normals = append(normals, q)
	}
	var flats []agxFlatPlan
	if split && len(normals) > 1 {
		normals, flats = agxSplitFlat(s.ID, normals)
	}

	trie := &agxTrieBuilder{child: map[agxTrieKey]int32{}}
	tr := &AgentXTrace{ID: s.ID, Peak: wekaPeak(s)}
	tr.convs = append(tr.convs, agxConv{sessionID: s.ID, scope: 0, turns: agxTurnsFor(normals, trie)})
	root := &tr.convs[0]

	// outer index -> root turn position, and the outer indexes in ascending order.
	outerPos := make(map[int]int, len(normals))
	outers := make([]int, 0, len(normals))
	for pos, r := range normals {
		outerPos[r.idx] = pos
		outers = append(outers, r.idx)
	}
	sort.Ints(outers)
	preceding := func(outer, fallback int) int {
		best := -1
		for oi, pos := range outerPos {
			if oi < outer && pos > best {
				best = pos
			}
		}
		if best < 0 {
			return fallback
		}
		return best
	}
	joinTurn := func(outer int, end float64) int {
		for _, oi := range outers {
			if oi <= outer {
				continue
			}
			if normals[outerPos[oi]].t+agxEpsilonSeconds >= end {
				return outerPos[oi]
			}
		}
		return -1
	}

	// Subagent groups keyed by (preceding, join), in first-seen order.
	type gkey struct{ pre, join int }
	var gorder []gkey
	groups := map[gkey][]int{}
	dropped := map[int]bool{}
	for si, sa := range subagents {
		pre := preceding(sa.outer, -1)
		if pre < 0 {
			dropped[si] = true
			continue
		}
		k := gkey{pre, joinTurn(sa.outer, agxSubagentEnd(sa.e))}
		if _, ok := groups[k]; !ok {
			gorder = append(gorder, k)
		}
		groups[k] = append(groups[k], si)
	}
	seenAgent := map[string]bool{}
	for si, sa := range subagents {
		if dropped[si] {
			continue
		}
		if seenAgent[sa.e.AgentID] {
			return nil, fmt.Errorf("trace %q: duplicate subagent agent_id %q among retained subagents", s.ID, sa.e.AgentID)
		}
		seenAgent[sa.e.AgentID] = true
	}

	// Child conversations: active subagents' chains (in plan order), then flat chains.
	convOfChild := make([]int32, len(childPlans))
	for i, cp := range childPlans {
		convOfChild[i] = -1
		if dropped[cp.saIndex] || len(cp.reqs) == 0 {
			continue
		}
		convOfChild[i] = int32(len(tr.convs))
		tr.convs = append(tr.convs, agxConv{sessionID: cp.sessionID, scope: int32(cp.saIndex + 1), turns: agxTurnsFor(cp.reqs, trie)})
	}
	convOfFlat := make([]int32, len(flats))
	for i, fp := range flats {
		convOfFlat[i] = int32(len(tr.convs))
		tr.convs = append(tr.convs, agxConv{sessionID: fp.sessionID, scope: 0, turns: agxTurnsFor(fp.reqs, trie)})
	}
	root = &tr.convs[0]

	addBranch := func(b agxBranch, pre, join int) {
		bi := int32(len(tr.branches))
		b.spawnTurn = int32(pre)
		b.joinTurn = -1
		if !b.background {
			b.joinTurn = int32(join)
		}
		tr.branches = append(tr.branches, b)
		root.turns[pre].spawns = append(root.turns[pre].spawns, bi)
		if join >= 0 {
			root.turns[join].gates = append(root.turns[join].gates, bi)
		}
	}
	for _, k := range gorder {
		sis := groups[k]
		b := agxBranch{id: fmt.Sprintf("%s:spawn:%s", s.ID, subagents[sis[0]].e.AgentID), background: k.join < 0, startMs: math.Inf(1)}
		for _, si := range sis {
			b.startMs = math.Min(b.startMs, subagents[si].e.T)
			for ci, cp := range childPlans {
				if cp.saIndex == si && convOfChild[ci] >= 0 {
					b.children = append(b.children, convOfChild[ci])
				}
			}
		}
		b.startMs *= 1000.0
		addBranch(b, k.pre, k.join)
	}
	var forder []gkey
	fgroups := map[gkey][]int{}
	for fi, fp := range flats {
		first := fp.reqs[0].idx
		end := math.Inf(-1)
		for _, r := range fp.reqs {
			end = math.Max(end, r.end())
		}
		k := gkey{preceding(first, 0), joinTurn(first, end)}
		if _, ok := fgroups[k]; !ok {
			forder = append(forder, k)
		}
		fgroups[k] = append(fgroups[k], fi)
	}
	for _, k := range forder {
		fis := fgroups[k]
		b := agxBranch{id: fmt.Sprintf("%s:flatspawn:%d", s.ID, flats[fis[0]].chainIndex), background: k.join < 0, startMs: math.Inf(1)}
		for _, fi := range fis {
			b.startMs = math.Min(b.startMs, flats[fi].reqs[0].t)
			b.children = append(b.children, convOfFlat[fi])
		}
		b.startMs *= 1000.0
		addBranch(b, k.pre, k.join)
	}

	tr.installPredecessors()
	tr.trieParent, tr.trieHash = trie.parent, trie.hash
	tr.minMs, tr.maxMs = math.Inf(1), math.Inf(-1)
	for _, c := range tr.reachable() {
		for _, t := range tr.convs[c].turns {
			tr.minMs = math.Min(tr.minMs, t.tsMs)
			tr.maxMs = math.Max(tr.maxMs, t.tsMs)
		}
	}
	return tr, nil
}

// reachable lists the root and every conversation a branch reaches (all of them here:
// aiperf attaches every child to the root).
func (t *AgentXTrace) reachable() []int32 {
	seen := map[int32]bool{0: true}
	out := []int32{0}
	for _, b := range t.branches {
		for _, c := range b.children {
			if !seen[c] {
				seen[c] = true
				out = append(out, c)
			}
		}
	}
	return out
}

// installPredecessors is _install_replay_dependencies + infer_cross_stream_predecessors:
// within a replay scope, a turn waits for each other stream's latest turn whose recorded
// interval ended by this turn's recorded start, keeping the non-dominated frontier.
func (t *AgentXTrace) installPredecessors() {
	type iv struct {
		ref        agxRef
		start, end float64
	}
	byScope := map[int32][][]iv{} // scope -> streams -> intervals
	for ci := range t.convs {
		c := &t.convs[ci]
		var s []iv
		for k := range c.turns {
			tn := &c.turns[k]
			s = append(s, iv{agxRef{int32(ci), int32(k)}, tn.tsMs, tn.tsMs + tn.apiMs})
		}
		byScope[c.scope] = append(byScope[c.scope], s)
	}
	keyLess := func(a, b agxRef) bool {
		ca, cb := t.convs[a.conv].sessionID, t.convs[b.conv].sessionID
		if ca != cb {
			return ca < cb
		}
		return a.turn < b.turn
	}
	for _, streams := range byScope {
		if len(streams) < 2 {
			continue
		}
		for si, target := range streams {
			for _, tg := range target {
				var frontier []iv
				for sj, cands := range streams {
					if sj == si {
						continue
					}
					found := false
					var best iv
					for _, c := range cands {
						if !(c.start < tg.start && c.end <= tg.start) {
							continue
						}
						if !found || c.end > best.end || (c.end == best.end && (c.start > best.start ||
							(c.start == best.start && keyLess(best.ref, c.ref)))) {
							best, found = c, true
						}
					}
					if found {
						frontier = append(frontier, best)
					}
				}
				var preds []agxRef
				for i, c := range frontier {
					dominated := false
					for j, l := range frontier {
						if i != j && c.start < l.start && c.end <= l.start {
							dominated = true
							break
						}
					}
					if !dominated {
						preds = append(preds, c.ref)
					}
				}
				sort.Slice(preds, func(a, b int) bool { return keyLess(preds[a], preds[b]) })
				t.convs[tg.ref.conv].turns[tg.ref.turn].preds = preds
			}
		}
	}
}

// agxBlockSize is the Weka trace block size and the KV block the replay's prompts are
// laid out on.
const agxBlockSize = 64

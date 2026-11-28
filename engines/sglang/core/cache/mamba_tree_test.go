// Original to blis-sglang.

package cache

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/engines/sglang/internal/testgz"
)

// MambaTreeGoldenDir holds the traces oracle/run_mamba_tree.py records at SGLang's
// UnifiedRadixCache with the state component (inference-sim/inference-sim#2366, #2442).
const MambaTreeGoldenDir = "testdata/golden-mamba-tree"

type mtCall struct {
	m     string
	ints  []int
	nodes []NodeID
	slots [][]int64
	acts  []string
}

// mtCore records the tree-core calls the cache makes, with the figures of each answer that a
// replay compares.
type mtCore struct {
	TreeCore
	log []mtCall
}

func (l *mtCore) add(c mtCall) { l.log = append(l.log, c) }

func freeKV(actions []CacheAction) (slots [][]int64, others []string, tails []NodeID) {
	for _, a := range actions {
		switch a.Kind {
		case FreeDeviceKVAction:
			slots = append(slots, a.Indices...)
		case MambaEvictExcessPathStatesAction:
			others = append(others, string(a.Kind))
			tails = append(tails, a.TailNode)
		default:
			others = append(others, string(a.Kind))
		}
	}
	return
}

func stepCall(m string, s InsertStep) mtCall {
	slots, acts, tails := freeKV(s.Actions)
	c := mtCall{m: m, slots: slots, acts: acts, nodes: tails}
	if s.Result != nil {
		c.ints = []int{s.Result.PrefixLen, b2i(s.Result.MambaExist)}
		c.nodes = append(c.nodes, s.Result.LastDeviceNode)
	}
	return c
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

func (l *mtCore) Reset() { l.TreeCore.Reset(); l.add(mtCall{m: "reset"}) }
func (l *mtCore) MatchPrefix(p MatchPrefixParams) MatchResult {
	r := l.TreeCore.MatchPrefix(p)
	br := -1
	if r.MambaBranchingSeqlen != nil {
		br = *r.MambaBranchingSeqlen
	}
	l.add(mtCall{m: "match_prefix", ints: []int{r.DevicePrefixLen, r.FullKVHitLength, br}, nodes: []NodeID{r.LastDeviceNode, r.BestMatchNode}})
	return r
}
func (l *mtCore) RootNodeHandle(k *string) NodeID {
	n := l.TreeCore.RootNodeHandle(k)
	l.add(mtCall{m: "root_node_handle", nodes: []NodeID{n}})
	return n
}
func (l *mtCore) CollectFullDeviceIndices(from, until NodeID) []int64 {
	s := l.TreeCore.CollectFullDeviceIndices(from, until)
	l.add(mtCall{m: "collect_full_device_indices", slots: [][]int64{s}})
	return s
}
func (l *mtCore) HasOngoingInsert() bool {
	l.add(mtCall{m: "has_ongoing_insert"})
	return l.TreeCore.HasOngoingInsert()
}
func (l *mtCore) BeginInsert(p InsertParams) InsertStep {
	s := l.TreeCore.BeginInsert(p)
	l.add(stepCall("begin_insert", s))
	return s
}
func (l *mtCore) ResumeInsert() InsertStep {
	s := l.TreeCore.ResumeInsert()
	l.add(stepCall("resume_insert", s))
	return s
}
func (l *mtCore) EndInsert() []CacheAction {
	l.add(mtCall{m: "end_insert"})
	return l.TreeCore.EndInsert()
}
func (l *mtCore) IncLockRef(n NodeID) LockReceipt {
	r := l.TreeCore.IncLockRef(n)
	l.add(mtCall{m: "inc_lock_ref", ints: []int{*r.Delta}})
	return r
}
func (l *mtCore) DecLockRef(n NodeID, r LockReceipt) {
	l.TreeCore.DecLockRef(n, r)
	l.add(mtCall{m: "dec_lock_ref"})
}
func (l *mtCore) EvictDeviceStart(c ComponentType, n int) {
	l.TreeCore.EvictDeviceStart(c, n)
	l.add(mtCall{m: "evict_device_start", ints: []int{ctInt(c), n}})
}
func (l *mtCore) EvictDeviceNextNode(c ComponentType, t Tracker) EvictionStep {
	s := l.TreeCore.EvictDeviceNextNode(c, t)
	cc := mtCall{m: "evict_device_next_node", ints: []int{ctInt(c), s.Tracker[FullComponentType], s.Tracker[MambaComponentType]},
		slots: s.DeviceFrees[MambaComponentType]}
	if s.Node != nil {
		cc.nodes = []NodeID{*s.Node}
	}
	l.add(cc)
	return s
}
func (l *mtCore) EvictDeviceLeaf(n NodeID, wb bool) EvictionStep {
	s := l.TreeCore.EvictDeviceLeaf(n, wb)
	l.add(mtCall{m: "evict_device_leaf", ints: []int{s.Tracker[FullComponentType], s.Tracker[MambaComponentType], len(s.DeviceFrees[FullComponentType])},
		slots: append(slices.Clone(s.DeviceFrees[FullComponentType]), s.DeviceFrees[MambaComponentType]...)})
	return s
}
func (l *mtCore) EvictDeviceEnd(c ComponentType) {
	l.TreeCore.EvictDeviceEnd(c)
	l.add(mtCall{m: "evict_device_end", ints: []int{ctInt(c)}})
}
func (l *mtCore) EvictExcessPathStates(tail NodeID) EvictionStep {
	s := l.TreeCore.EvictExcessPathStates(tail)
	l.add(mtCall{m: "evict_excess_path_states", nodes: []NodeID{tail}, slots: s.DeviceFrees[MambaComponentType]})
	return s
}
func (l *mtCore) ComponentDeviceValue(c ComponentType, n NodeID) []int64 {
	v := l.TreeCore.ComponentDeviceValue(c, n)
	l.add(mtCall{m: "get_component_device_value", ints: []int{ctInt(c)}, slots: [][]int64{v}})
	return v
}
func (l *mtCore) EvictableSize() int {
	v := l.TreeCore.EvictableSize()
	l.add(mtCall{m: "evictable_size", ints: []int{v}})
	return v
}
func (l *mtCore) FullEvictableSize() int {
	v := l.TreeCore.FullEvictableSize()
	l.add(mtCall{m: "full_evictable_size", ints: []int{v}})
	return v
}
func (l *mtCore) FullProtectedSize() int {
	v := l.TreeCore.FullProtectedSize()
	l.add(mtCall{m: "full_protected_size", ints: []int{v}})
	return v
}
func (l *mtCore) MambaEvictableSize() int {
	v := l.TreeCore.MambaEvictableSize()
	l.add(mtCall{m: "mamba_evictable_size", ints: []int{v}})
	return v
}
func (l *mtCore) MambaProtectedSize() int {
	v := l.TreeCore.MambaProtectedSize()
	l.add(mtCall{m: "mamba_protected_size", ints: []int{v}})
	return v
}
func (l *mtCore) TotalSize() (int, int) {
	a, b := l.TreeCore.TotalSize()
	l.add(mtCall{m: "total_size", ints: []int{a, b}})
	return a, b
}

// ctInt is SGLang's ComponentType id of a component: FULL 0, MAMBA 2.
func ctInt(c ComponentType) int {
	if c == MambaComponentType {
		return 2
	}
	return 0
}

type mtReplay struct {
	tr     Trace
	name   string
	c      *RadixCache
	core   *mtCore
	alloc  *stubAllocator
	pool   *HybridRequestPool
	nodes  *Nodes
	reqs   map[string]*Req
	bySeq  map[int]Call
	desc   map[int][]Call
	driver int
}

func (r *mtReplay) within(seq int, iface Interface, methods ...string) []Call {
	if r.desc == nil {
		r.bySeq = map[int]Call{}
		for _, c := range r.tr.Calls {
			r.bySeq[c.Seq] = c
		}
		r.desc = map[int][]Call{}
		for _, c := range r.tr.Calls {
			for x := c; x.In != nil; x = r.bySeq[*x.In] {
				r.desc[*x.In] = append(r.desc[*x.In], c)
			}
		}
	}
	var out []Call
	for _, c := range r.desc[seq] {
		if c.Iface == iface && slices.Contains(methods, c.Method) {
			out = append(out, c)
		}
	}
	return out
}

// madeBy reports whether the call x was made on behalf of the call numbered seq: the chain of
// calls that made it reaches seq without passing through another tree-core call (a component's
// call inside a tree-core call is the tree core's own business).
func (r *mtReplay) madeBy(x Call, seq int) bool {
	for x.In != nil {
		if *x.In == seq {
			return true
		}
		x = r.bySeq[*x.In]
		if x.Iface == TreeCoreIface {
			return false
		}
	}
	return false
}

func (r *mtReplay) pair(s, p NodeID) error { return r.nodes.Pair(s, p) }
func (r *mtReplay) port(s NodeID) NodeID {
	p, err := r.nodes.Port(s)
	if err != nil {
		panic(err)
	}
	return p
}

type tracedMamba struct {
	ReqPoolIdx        *int    `json:"req_pool_idx"`
	PoolIdx           *int64  `json:"mamba_pool_idx"`
	CowSrc            []int64 `json:"mamba_cow_src_index"`
	NeedsClear        *bool   `json:"mamba_needs_clear"`
	PingPong          []int64 `json:"mamba_ping_pong_track_buffer"`
	PingPongEmpty     []bool  `json:"mamba_ping_pong_empty"`
	NextTrackIdx      *int    `json:"mamba_next_track_idx"`
	LastTrackIdx      *int    `json:"mamba_last_track_idx"`
	LastTrackSeqlen   *int    `json:"mamba_last_track_seqlen"`
	PrevTrackSeqlen   *int    `json:"mamba_prev_track_seqlen"`
	CacheProtectedLen int     `json:"cache_protected_len"`
	CacheInsertedLen  int     `json:"cache_inserted_len"`
	KVAllocatedLen    int     `json:"kv_allocated_len"`
}

type mtReq struct {
	ID       string      `json:"req"`
	KV       tracedMamba `json:"kv"`
	Origin   []int64     `json:"origin_input_ids"`
	Fill     []int64     `json:"full_untruncated_fill_ids"`
	Finished *bool       `json:"finished"`
	LastNode *NodeID     `json:"last_node"`
	Lock     *tracedLock `json:"lock"`
}

// syncReq makes the held request the one a call's argument names: what the driver set (the ids,
// whether it finished, the lengths of its KV, the tracked state lengths) is taken from the trace, and
// what the cache and the pool computed (the slots, the protected length, the lock) must equal it.
func (r *mtReplay) syncReq(raw json.RawMessage) (*Req, error) {
	have, err := r.reqOf(raw)
	if err != nil {
		return nil, err
	}
	var q mtReq
	if err := json.Unmarshal(raw, &q); err != nil {
		return nil, err
	}
	have.OriginInputIDs, have.FillIDs = q.Origin, q.Fill
	have.Finished = q.Finished != nil && *q.Finished
	have.KV.KVAllocatedLen = q.KV.KVAllocatedLen
	have.KV.Mamba.LastTrackSeqlen, have.KV.Mamba.PrevTrackSeqlen = q.KV.LastTrackSeqlen, q.KV.PrevTrackSeqlen
	if have.KV.CacheProtectedLen != q.KV.CacheProtectedLen || have.KV.CacheInsertedLen != q.KV.CacheInsertedLen {
		return nil, fmt.Errorf("request %s: protected %d, inserted %d; SGLang's %d, %d", q.ID, have.KV.CacheProtectedLen,
			have.KV.CacheInsertedLen, q.KV.CacheProtectedLen, q.KV.CacheInsertedLen)
	}
	return have, r.sameMambaKV(have, q.KV)
}

func (r *mtReplay) sameMambaKV(have *Req, q tracedMamba) error {
	m := have.KV.Mamba
	if !slices.Equal(m.PingPongBuffer, q.PingPong) || !slices.Equal(m.PingPongEmpty, q.PingPongEmpty) ||
		deref(m.NextTrackIdx) != deref(q.NextTrackIdx) || deref(m.LastTrackIdx) != deref(q.LastTrackIdx) {
		return fmt.Errorf("request %s: tracking slots %v %v next %v last %v; SGLang's %v %v next %v last %v", have.ID, m.PingPongBuffer,
			m.PingPongEmpty, deref(m.NextTrackIdx), deref(m.LastTrackIdx), q.PingPong, q.PingPongEmpty, deref(q.NextTrackIdx), deref(q.LastTrackIdx))
	}
	return nil
}

// reqOf is the request an argument names: the one the replay holds for its id, built from the
// trace the first time it is named. A later naming must agree with what the replay's own calls
// left on it.
func (r *mtReplay) reqOf(raw json.RawMessage) (*Req, error) {
	var q mtReq
	if err := json.Unmarshal(raw, &q); err != nil {
		return nil, err
	}
	have, ok := r.reqs[q.ID]
	if !ok {
		have = &Req{ID: q.ID, KV: ReqKV{Mamba: &MambaKV{}}}
		if q.KV.ReqPoolIdx != nil {
			have.KV.ReqPoolIdx = q.KV.ReqPoolIdx
		}
		if q.KV.PoolIdx != nil {
			have.KV.Mamba.PoolIdx = q.KV.PoolIdx
		}
		r.reqs[q.ID] = have
		return have, nil
	}
	if have.KV.Mamba == nil {
		have.KV.Mamba = &MambaKV{}
	}
	if (have.KV.Mamba.PoolIdx == nil) != (q.KV.PoolIdx == nil) || (q.KV.PoolIdx != nil && *have.KV.Mamba.PoolIdx != *q.KV.PoolIdx) {
		return nil, fmt.Errorf("request %s: the state slot is %v after the replayed calls; SGLang's is %v", q.ID, have.KV.Mamba.PoolIdx, q.KV.PoolIdx)
	}
	var cow []int64
	if have.KV.Mamba.CowSrcIndex != nil {
		cow = []int64{*have.KV.Mamba.CowSrcIndex}
	}
	if !slices.Equal(cow, q.KV.CowSrc) {
		return nil, fmt.Errorf("request %s: the copy source is %v after the replayed calls; SGLang's is %v", q.ID, cow, q.KV.CowSrc)
	}
	return have, nil
}

func (r *mtReplay) driverUpTo(seq int) error {
	for r.driver < len(r.tr.Driver) && r.tr.Driver[r.driver].After < seq {
		d := r.tr.Driver[r.driver]
		r.driver++
		if d.M != "alloc" {
			return fmt.Errorf("driver call %s is not replayed", d.M)
		}
		got, ok := r.pool.Mamba.Alloc(d.N)
		if !ok || !slices.Equal(got, d.Ret) {
			return fmt.Errorf("the state allocator gave %v for %d slots; SGLang's gave %v", got, d.N, d.Ret)
		}
	}
	return nil
}

func replayMambaTree(name string, tr Trace) (int, error) {
	r := &mtReplay{tr: tr, name: name, nodes: NewNodes(), reqs: map[string]*Req{}}
	raw, err := testgz.ReadFile(filepath.Join("..", "..", "oracle", "scenarios-mamba-tree", name+".json"))
	if err != nil {
		return 0, err
	}
	var sc struct {
		ServerArgs map[string]any `json:"server_args"`
		Cache      struct {
			MaxNumReqs int  `json:"max_num_reqs"`
			MaxContext int  `json:"max_context_len"`
			MambaSize  int  `json:"mamba_size"`
			ExtraBuf   bool `json:"extra_buffer"`
			Lazy       bool `json:"lazy"`
			Overlap    bool `json:"overlap"`
		} `json:"cache"`
	}
	if err := json.Unmarshal(raw, &sc); err != nil {
		return 0, err
	}
	n := 0
	for _, c := range tr.Calls {
		if err := r.driverUpTo(c.Seq); err != nil {
			return n, fmt.Errorf("seq %d: %w", c.Seq, err)
		}
		switch {
		case c.Iface == PrefixCacheIface && c.Method == "__init__":
			var p struct {
				PageSize int    `json:"page_size"`
				Policy   string `json:"eviction_policy"`
				Slots    []struct {
					Component string `json:"component"`
					Interval  *int   `json:"checkpoint_interval"`
				} `json:"tree_slots"`
			}
			if err := c.Arg("params", &p); err != nil {
				return n, err
			}
			interval := 0
			for _, s := range p.Slots {
				if s.Component == "MAMBA" && s.Interval != nil {
					interval = *s.Interval
				}
			}
			capStates, level := 1, 2
			if v, ok := sc.ServerArgs["hybrid_cache_checkpoints_level"].(float64); ok && v >= 3 {
				capStates, level = -1, 3
			}
			overlap := sc.Cache.Overlap
			r.pool = NewHybridRequestPool(sc.Cache.MaxNumReqs, sc.Cache.MambaSize, sc.Cache.MaxContext, sc.Cache.ExtraBuf, sc.Cache.Lazy, overlap)
			r.alloc = &stubAllocator{page: p.PageSize}
			r.c, err = NewRadixCache(RadixConfig{PageSize: p.PageSize, EvictionPolicy: p.Policy,
				Mamba: &MambaTreeConfig{CheckpointInterval: interval, PathStateCap: capStates, CheckpointsLevel: level, Pool: r.pool}}, r.alloc, r.pool.RequestPool)
			if err != nil {
				return n, err
			}
			r.core = &mtCore{TreeCore: r.c.core}
			r.c.core = r.core
		case c.Iface == PrefixCacheIface && c.In == nil:
			if err := r.call(c); err != nil {
				return n, fmt.Errorf("seq %d %s: %w", c.Seq, c.Method, err)
			}
			n++
		case c.Iface == TreeCoreIface && c.In == nil:
			if err := r.topCore(c); err != nil {
				return n, fmt.Errorf("seq %d tree_core.%s: %w", c.Seq, c.Method, err)
			}
			n++
		case c.Iface == ReqToTokenIface && c.In == nil && c.Method == "write":
			slots, err := c.RowWrite()
			if err != nil {
				return n, err
			}
			for _, s := range slots {
				r.pool.Write(s.Row, s.Pos, []int64{s.Slot})
			}
		case c.Iface == ReqToTokenIface && c.In == nil && c.Method == "alloc":
			if err := r.allocReqs(c); err != nil {
				return n, fmt.Errorf("seq %d req_to_token.alloc: %w", c.Seq, err)
			}
			n++
		}
	}
	return n, nil
}

func (r *mtReplay) topCore(c Call) error {
	var got, want []int
	switch c.Method {
	case "mamba_evictable_size":
		got = []int{r.c.core.MambaEvictableSize()}
	case "mamba_protected_size":
		got = []int{r.c.core.MambaProtectedSize()}
	default:
		return fmt.Errorf("not replayed")
	}
	var w int
	if err := c.Answer(&w); err != nil {
		return err
	}
	want = []int{w}
	r.core.log = nil
	if !slices.Equal(got, want) {
		return fmt.Errorf("%v, SGLang's %v", got, want)
	}
	return nil
}

func (r *mtReplay) allocReqs(c Call) error {
	var raws []json.RawMessage
	if err := c.Arg("reqs", &raws); err != nil {
		return err
	}
	var reqs []*Req
	for _, raw := range raws {
		q, err := r.reqOf(raw)
		if err != nil {
			return err
		}
		reqs = append(reqs, q)
	}
	rows, err := r.pool.AcquireRows(reqs)
	if err != nil {
		return err
	}
	var want []int
	if err := c.Answer(&want); err != nil {
		return err
	}
	if !slices.Equal(rows, want) {
		return fmt.Errorf("rows %v, SGLang's %v", rows, want)
	}
	return nil
}

type tracedParams struct {
	Key struct {
		Tokens    []int64 `json:"tokens"`
		ExtraKey  *string `json:"extra_key"`
		CacheSalt *string `json:"cache_salt"`
	} `json:"key"`
	Value      []int64         `json:"value"`
	PrevPrefix int             `json:"prev_prefix_len"`
	Priority   int             `json:"priority"`
	MambaValue []int64         `json:"mamba_value"`
	KeepIdx    *int            `json:"mamba_keep_idx"`
	AllowMiss  bool            `json:"allow_missing_mamba"`
	Cow        bool            `json:"cow_mamba"`
	DeferSlot  bool            `json:"defer_cow_slot"`
	Req        json.RawMessage `json:"req"`
	NumTokens  int             `json:"num_tokens"`
	MambaNum   int             `json:"mamba_num"`
}

func (r *mtReplay) call(c Call) error {
	r.alloc.avail = nil
	for _, x := range r.within(c.Seq, AllocatorIface, "available_size", "owner_available_size") {
		var v int
		if err := x.Answer(&v); err != nil {
			return err
		}
		r.alloc.avail = append(r.alloc.avail, v)
	}
	r.alloc.made, r.core.log = nil, nil
	if err := r.answer(c); err != nil {
		return err
	}
	want, err := r.wantCore(c.Seq)
	if err != nil {
		return err
	}
	if err := r.sameCore(r.core.log, want); err != nil {
		return err
	}
	return r.sameAllocatorCalls(c)
}

func (r *mtReplay) sameAllocatorCalls(c Call) error {
	var want []madeCall
	for _, x := range r.within(c.Seq, AllocatorIface, "available_size", "owner_available_size", "target_window_pool", "free_segment", "free_segments") {
		switch x.Method {
		case "free_segment", "free_segments":
			continue
		default:
			want = append(want, madeCall{x.Method, ""})
		}
	}
	var got []madeCall
	for _, m := range r.alloc.made {
		if m.m == "available_size" || m.m == "owner_available_size" || m.m == "target_window_pool" {
			got = append(got, m)
		}
	}
	if len(got) != len(want) {
		return fmt.Errorf("the cache read the allocator %v; SGLang's read it %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			return fmt.Errorf("the cache read the allocator %v; SGLang's read it %v", got, want)
		}
	}
	// The frees: the slots, in order.
	var wantFree, gotFree []string
	for _, x := range r.within(c.Seq, AllocatorIface, "free_segment", "free_segments") {
		var s any
		if x.Method == "free_segment" {
			var a []int64
			_ = x.Arg("free_index", &a)
			s = a
		} else {
			var a json.RawMessage
			_ = x.Arg("segments", &a)
			s = string(a)
		}
		wantFree = append(wantFree, fmt.Sprint(x.Method, s))
	}
	for _, m := range r.alloc.made {
		if m.m == "free_segment" || m.m == "free_segments" {
			gotFree = append(gotFree, m.m+m.args)
		}
	}
	if len(gotFree) != len(wantFree) {
		return fmt.Errorf("the cache freed %v; SGLang's freed %v", gotFree, wantFree)
	}
	return nil
}

func (r *mtReplay) answer(c Call) error {
	switch c.Method {
	case "reset":
		r.c.Reset()
	case "match_prefix":
		var p struct {
			Params tracedParams `json:"params"`
		}
		if err := c.Arg("params", &p.Params); err != nil {
			return err
		}
		mp := MatchPrefixParams{Key: Key{Tokens: p.Params.Key.Tokens, ExtraKey: p.Params.Key.ExtraKey, CacheSalt: p.Params.Key.CacheSalt},
			CowMamba: p.Params.Cow, DeferCowSlot: p.Params.DeferSlot}
		if len(p.Params.Req) > 0 && string(p.Params.Req) != "null" {
			q, err := r.reqOf(p.Params.Req)
			if err != nil {
				return err
			}
			mp.Req = q
		}
		got := r.c.MatchPrefix(mp)
		var want MatchResult
		if err := c.Answer(&want); err != nil {
			return err
		}
		return r.sameMatch(got, want)
	case "insert":
		var p struct {
			Params tracedParams `json:"params"`
		}
		if err := c.Arg("params", &p.Params); err != nil {
			return err
		}
		q := p.Params
		got := r.c.Insert(InsertParams{Key: Key{Tokens: q.Key.Tokens, ExtraKey: q.Key.ExtraKey, CacheSalt: q.Key.CacheSalt}, Value: q.Value,
			PrevPrefixLen: q.PrevPrefix, Priority: q.Priority, MambaValue: q.MambaValue, MambaKeepIdx: q.KeepIdx, AllowMissingMamba: q.AllowMiss})
		var want InsertResult
		if err := c.Answer(&want); err != nil {
			return err
		}
		if got.PrefixLen != want.PrefixLen || got.MambaExist != want.MambaExist {
			return fmt.Errorf("insert prefix %d (state existed %v); SGLang %d (%v)", got.PrefixLen, got.MambaExist, want.PrefixLen, want.MambaExist)
		}
		return r.pair(want.LastDeviceNode, got.LastDeviceNode)
	case "checkpoint", "on_release":
		var raw json.RawMessage
		if err := c.Arg("req", &raw); err != nil {
			return err
		}
		q, err := r.syncReq(raw)
		if err != nil {
			return err
		}
		if c.Method == "checkpoint" {
			var upTo int
			if err := c.Arg("up_to", &upTo); err != nil {
				return err
			}
			r.c.Checkpoint(q, upTo)
		} else {
			var cp bool
			if err := c.Arg("checkpointed", &cp); err != nil {
				return err
			}
			r.c.OnRelease(q, cp)
		}
		return r.sameAfter(c, q)
	case "evict":
		var p struct {
			Params tracedParams `json:"params"`
		}
		if err := c.Arg("params", &p.Params); err != nil {
			return err
		}
		got := r.c.Evict(EvictParams{NumTokens: p.Params.NumTokens, MambaNum: p.Params.MambaNum})
		return r.sameEvict(c, got)
	case "evict_for_alloc":
		var d struct {
			FullTokens int `json:"full_tokens"`
			StateSlots int `json:"state_slots"`
		}
		if err := c.Arg("demand", &d); err != nil {
			return err
		}
		return r.sameEvict(c, r.c.EvictForAlloc(KVDemand{FullTokens: d.FullTokens, StateSlots: d.StateSlots}))
	case "inc_lock_ref":
		var id NodeID
		if err := c.Arg("node_id", &id); err != nil {
			return err
		}
		got := r.c.IncLockRef(r.port(id))
		var want LockReceipt
		if err := c.Answer(&want); err != nil {
			return err
		}
		if (got.Delta == nil) != (want.Delta == nil) || (got.Delta != nil && *got.Delta != *want.Delta) {
			return fmt.Errorf("lock delta %v, SGLang's %v", deref(got.Delta), deref(want.Delta))
		}
	case "dec_lock_ref":
		var id NodeID
		if err := c.Arg("node_id", &id); err != nil {
			return err
		}
		r.c.DecLockRef(r.port(id), LockReceipt{Node: r.port(id)})
	case "evictable_size", "protected_size", "full_evictable_size", "full_protected_size":
		var want int
		if err := c.Answer(&want); err != nil {
			return err
		}
		got := map[string]func() int{"evictable_size": r.c.EvictableSize, "protected_size": r.c.ProtectedSize,
			"full_evictable_size": r.c.FullEvictableSize, "full_protected_size": r.c.FullProtectedSize}[c.Method]()
		if got != want {
			return fmt.Errorf("%d, SGLang's %d", got, want)
		}
	case "total_size":
		var want [2]int
		if err := c.Answer(&want); err != nil {
			return err
		}
		a, b := r.c.TotalSize()
		if [2]int{a, b} != want {
			return fmt.Errorf("%v, SGLang's %v", [2]int{a, b}, want)
		}
	default:
		return fmt.Errorf("the replay has no cache call %s", c.Method)
	}
	return nil
}

// sameAfter compares the request as the call left it with SGLang's record of it on return.
func (r *mtReplay) sameAfter(c Call, q *Req) error {
	if len(c.ReqAfter) == 0 {
		return nil
	}
	var a mtReq
	if err := json.Unmarshal(c.ReqAfter[0], &a); err != nil {
		return err
	}
	if q.KV.CacheProtectedLen != a.KV.CacheProtectedLen || q.KV.CacheInsertedLen != a.KV.CacheInsertedLen {
		return fmt.Errorf("after the call: protected %d, inserted %d; SGLang's %d, %d", q.KV.CacheProtectedLen, q.KV.CacheInsertedLen,
			a.KV.CacheProtectedLen, a.KV.CacheInsertedLen)
	}
	if (q.LastNode == nil) != (a.LastNode == nil) {
		return fmt.Errorf("after the call: last node %v; SGLang's %v", q.LastNode, a.LastNode)
	}
	if a.LastNode != nil {
		if err := r.pair(*a.LastNode, *q.LastNode); err != nil {
			return err
		}
	}
	m := q.KV.Mamba
	if (m.PoolIdx == nil) != (a.KV.PoolIdx == nil) || (m.PoolIdx != nil && *m.PoolIdx != *a.KV.PoolIdx) {
		return fmt.Errorf("after the call: state slot %v; SGLang's %v", m.PoolIdx, a.KV.PoolIdx)
	}
	if !slices.Equal(m.PingPongBuffer, a.KV.PingPong) {
		return fmt.Errorf("after the call: tracking slots %v; SGLang's %v", m.PingPongBuffer, a.KV.PingPong)
	}
	return nil
}

func (r *mtReplay) sameEvict(c Call, got EvictResult) error {
	var want EvictResult
	if err := c.Answer(&want); err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("evicted %d tokens and %d states; SGLang %d and %d", got.NumTokensEvicted, got.MambaNumEvicted,
			want.NumTokensEvicted, want.MambaNumEvicted)
	}
	return nil
}

func (r *mtReplay) sameMatch(got, want MatchResult) error {
	if got.DevicePrefixLen != want.DevicePrefixLen || got.FullKVHitLength != want.FullKVHitLength {
		return fmt.Errorf("matched %d (%d hit); SGLang %d (%d)", got.DevicePrefixLen, got.FullKVHitLength, want.DevicePrefixLen, want.FullKVHitLength)
	}
	if (got.MambaBranchingSeqlen == nil) != (want.MambaBranchingSeqlen == nil) ||
		(got.MambaBranchingSeqlen != nil && *got.MambaBranchingSeqlen != *want.MambaBranchingSeqlen) {
		return fmt.Errorf("branching %v; SGLang's %v", deref(got.MambaBranchingSeqlen), deref(want.MambaBranchingSeqlen))
	}
	for _, p := range [][2]NodeID{{want.LastDeviceNode, got.LastDeviceNode}, {want.BestMatchNode, got.BestMatchNode}, {want.LastHostNode, got.LastHostNode}} {
		if err := r.pair(p[0], p[1]); err != nil {
			return err
		}
	}
	if (got.LastVisitedNode == nil) != (want.LastVisitedNode == nil) {
		return fmt.Errorf("last visited node %v, SGLang's %v", got.LastVisitedNode, want.LastVisitedNode)
	}
	if want.LastVisitedNode != nil {
		return r.pair(*want.LastVisitedNode, *got.LastVisitedNode)
	}
	return nil
}

func ctOf(i int) ComponentType {
	if i == 2 {
		return MambaComponentType
	}
	return FullComponentType
}

// wantCore is the tree-core calls the call numbered seq made itself, in order.
func (r *mtReplay) wantCore(seq int) ([]mtCall, error) {
	var out []mtCall
	r.within(0, TreeCoreIface) // builds the call index
	for _, x := range r.tr.Calls {
		if x.Iface != TreeCoreIface || !r.madeBy(x, seq) {
			continue
		}
		cc := mtCall{m: x.Method}
		var err error
		switch x.Method {
		case "match_prefix":
			var m MatchResult
			err = x.Answer(&m)
			br := -1
			if m.MambaBranchingSeqlen != nil {
				br = *m.MambaBranchingSeqlen
			}
			cc.ints, cc.nodes = []int{m.DevicePrefixLen, m.FullKVHitLength, br}, []NodeID{m.LastDeviceNode, m.BestMatchNode}
		case "begin_insert", "resume_insert":
			var s struct {
				Actions []struct {
					T       string    `json:"_t"`
					Indices [][]int64 `json:"indices"`
					Tail    *NodeID   `json:"tail_node_id"`
				} `json:"actions"`
				Result *InsertResult `json:"result"`
			}
			err = x.Answer(&s)
			for _, a := range s.Actions {
				if a.T == "FreeDeviceKV" {
					cc.slots = append(cc.slots, a.Indices...)
					continue
				}
				cc.acts = append(cc.acts, a.T)
				if a.Tail != nil {
					cc.nodes = append(cc.nodes, *a.Tail)
				}
			}
			if s.Result != nil {
				cc.ints = []int{s.Result.PrefixLen, b2i(s.Result.MambaExist)}
				cc.nodes = append(cc.nodes, s.Result.LastDeviceNode)
			}
		case "inc_lock_ref":
			var lr LockReceipt
			err = x.Answer(&lr)
			if lr.Delta != nil {
				cc.ints = []int{*lr.Delta}
			}
		case "evict_device_start":
			var n int
			var ct int
			err = x.Arg("request_cnt", &n)
			if err == nil {
				err = x.Arg("component_type", &ct)
			}
			cc.ints = []int{ct, n}
		case "evict_device_end":
			var ct int
			err = x.Arg("component_type", &ct)
			cc.ints = []int{ct}
		case "evict_device_next_node":
			var s struct {
				Node        *NodeID                     `json:"node_id"`
				Tracker     map[ComponentType]int       `json:"tracker"`
				DeviceFrees map[ComponentType][][]int64 `json:"device_frees"`
			}
			var ct int
			err = x.Answer(&s)
			if err == nil {
				err = x.Arg("component_type", &ct)
			}
			cc.ints = []int{ct, s.Tracker[FullComponentType], s.Tracker[MambaComponentType]}
			cc.slots = s.DeviceFrees[MambaComponentType]
			if s.Node != nil {
				cc.nodes = []NodeID{*s.Node}
			}
		case "evict_device_leaf":
			var s struct {
				DeviceFrees map[ComponentType][][]int64 `json:"device_frees"`
				Tracker     map[ComponentType]int       `json:"tracker"`
			}
			err = x.Answer(&s)
			cc.ints = []int{s.Tracker[FullComponentType], s.Tracker[MambaComponentType], len(s.DeviceFrees[FullComponentType])}
			cc.slots = append(slices.Clone(s.DeviceFrees[FullComponentType]), s.DeviceFrees[MambaComponentType]...)
		case "evict_excess_path_states":
			var tail NodeID
			err = x.Arg("tail_node_id", &tail)
			cc.nodes = []NodeID{tail}
			// The slots it frees are the state frees the next steps of the trace show at the
			// pool; the call's own arguments record the empty dicts it was given.
			cc.slots = nil
		case "get_component_device_value":
			var v []int64
			var ct int
			err = x.Answer(&v)
			if err == nil {
				err = x.Arg("component_type", &ct)
			}
			cc.ints, cc.slots = []int{ct}, [][]int64{v}
		case "evictable_size", "full_evictable_size", "full_protected_size":
			var v int
			err = x.Answer(&v)
			cc.ints = []int{v}
		case "total_size":
			var v [2]int
			err = x.Answer(&v)
			cc.ints = v[:]
		case "root_node_handle":
			var n NodeID
			err = x.Answer(&n)
			cc.nodes = []NodeID{n}
		case "collect_full_device_indices":
			var s []int64
			err = x.Answer(&s)
			cc.slots = [][]int64{s}
		case "reset", "has_ongoing_insert", "end_insert", "dec_lock_ref":
		default:
			return nil, fmt.Errorf("seq %d: the replay does not know tree-core call %s", x.Seq, x.Method)
		}
		if err != nil {
			return nil, err
		}
		out = append(out, cc)
	}
	return out, nil
}

func (r *mtReplay) sameCore(got, want []mtCall) error {
	// evict_excess_path_states: SGLang's call hands its frees back in the arguments, which the trace
	// records empty; its figures are checked through the sizes that follow.
	for i := range got {
		if got[i].m == "evict_excess_path_states" {
			got[i].slots = nil
		}
	}
	if len(got) != len(want) {
		return fmt.Errorf("the cache made %d tree-core calls %v; SGLang's made %d %v", len(got), mtNames(got), len(want), mtNames(want))
	}
	for i := range got {
		g, w := got[i], want[i]
		if g.m != w.m || !slices.Equal(g.ints, w.ints) || !slices.Equal(g.acts, w.acts) || len(g.slots) != len(w.slots) || len(g.nodes) != len(w.nodes) {
			return fmt.Errorf("tree-core call %d: %s ints %v acts %v slots %v; SGLang's %s ints %v acts %v slots %v", i, g.m, g.ints, g.acts, g.slots, w.m, w.ints, w.acts, w.slots)
		}
		for j := range g.slots {
			if !slices.Equal(g.slots[j], w.slots[j]) {
				return fmt.Errorf("tree-core call %d %s: slots %v; SGLang's %v", i, g.m, g.slots, w.slots)
			}
		}
		for j := range g.nodes {
			if err := r.pair(w.nodes[j], g.nodes[j]); err != nil {
				return fmt.Errorf("tree-core call %d %s: %w", i, g.m, err)
			}
		}
	}
	return nil
}

func mtNames(cs []mtCall) []string {
	var out []string
	for _, c := range cs {
		out = append(out, c.m)
	}
	return out
}

func loadMambaTreeTraces(t *testing.T) map[string]Trace {
	t.Helper()
	paths, err := testgz.Glob(filepath.Join(MambaTreeGoldenDir, "*.trace.json"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no trace under %s (%v)", MambaTreeGoldenDir, err)
	}
	out := map[string]Trace{}
	for _, p := range paths {
		raw, err := testgz.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var tr Trace
		if err := json.Unmarshal(raw, &tr); err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		out[strings.TrimSuffix(filepath.Base(p), ".trace.json")] = tr
	}
	return out
}

// The tree of the hybrid cache: its component, locks, ledgers, matching, inserting and eviction
// answer, and call the allocator, as SGLang's UnifiedRadixCache with the state component does in
// every sequence of oracle/scenarios-mamba-tree.
func TestMambaTreeAnswersAsSGLangDoes(t *testing.T) {
	trs := loadMambaTreeTraces(t)
	names := make([]string, 0, len(trs))
	for n := range trs {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			n, err := replayMambaTree(name, trs[name])
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%d calls replayed", n)
		})
	}
}

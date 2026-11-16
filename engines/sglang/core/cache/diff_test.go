// Original to blis-sglang.

package cache

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"testing"
)

// The port of UnifiedRadixCache, its Python tree core and FullComponent answers every call
// SGLang's own cache answered, in every trace of the framework (FID-5): the scheduler's calls
// while it ran a scenario, and the driver's in orderings the scheduler does not make. The calls
// the cache makes in turn, at the allocator and the request pool, are SGLang's too, in order.
// The allocator and the request pool are stand-ins that answer as the trace says SGLang's did;
// their own ports are checked against the same traces where they are built.
func TestTheRadixCacheAnswersEveryTracedCallAsSGLangDoes(t *testing.T) {
	replayed := 0
	for name, tr := range loadGoldens(t) {
		t.Run(name, func(t *testing.T) {
			n, err := replayRadix(tr)
			if err != nil {
				t.Fatal(err)
			}
			replayed += n
			t.Logf("%d calls replayed", n)
		})
	}
	if replayed == 0 {
		t.Fatal("no call at the radix cache was replayed")
	}
}

// madeCall is a call the port made at the allocator or the pool, as the trace records SGLang's.
type madeCall struct {
	m    string
	args string
}

// stubAllocator answers AvailableSize with SGLang's answers in turn, and records the frees.
type stubAllocator struct {
	avail []int
	made  []madeCall
	page  int
}

func (a *stubAllocator) AvailableSize() int {
	a.made = append(a.made, madeCall{"available_size", ""})
	if len(a.avail) == 0 {
		return -1
	}
	v := a.avail[0]
	a.avail = a.avail[1:]
	return v
}

func (a *stubAllocator) FreeSegment(s []int64, start int) {
	a.made = append(a.made, madeCall{"free_segment", fmt.Sprint(s, start)})
}

func (a *stubAllocator) FreeSegments(segs []Segment) {
	var parts []string
	for _, s := range segs {
		parts = append(parts, fmt.Sprint(s.Slots, s.StartPos))
	}
	a.made = append(a.made, madeCall{"free_segments", fmt.Sprint(parts)})
}

func (a *stubAllocator) Size() int                                          { return 0 }
func (a *stubAllocator) PageSize() int                                      { return a.page }
func (a *stubAllocator) Alloc(int) ([]int64, bool)                          { panic("the radix cache allocates nothing") }
func (a *stubAllocator) Free([]int64)                                       { a.made = append(a.made, madeCall{"free", ""}) }
func (a *stubAllocator) FreeGroupBegin()                                    {}
func (a *stubAllocator) FreeGroupEnd()                                      {}
func (a *stubAllocator) Clear()                                             {}
func (a *stubAllocator) EvictToFreeTokens(PrefixCache, int)                 {}
func (a *stubAllocator) CheckDecodeCapacity(int, PrefixCache) bool          { return false }
func (a *stubAllocator) RequestSlotReserve(bool, int) int                   { return 0 }
func (a *stubAllocator) CreatePrefillBudget(PrefixCache, int) PrefillBudget { return nil }
func (a *stubAllocator) MaxNewTokensForMemory(int, int, int, *int, *int) (int, bool) {
	return 0, false
}
func (a *stubAllocator) VerifyByteAccounting() []string { return nil }
func (a *stubAllocator) GetKVCache() KVPool             { return nil }

// stubPool holds the rows the trace's writes build, and records the port's writes.
type stubPool struct {
	rows map[int]map[int]int64
	made []madeCall
}

func (p *stubPool) set(row, pos int, slot int64) {
	if p.rows[row] == nil {
		p.rows[row] = map[int]int64{}
	}
	p.rows[row][pos] = slot
}

func (p *stubPool) Write(row, start int, values []int64) {
	if len(values) == 0 {
		return // a write of no slots sets nothing, and is compared as nothing
	}
	p.made = append(p.made, madeCall{"write", fmt.Sprint(row, start, values)})
	for i, v := range values {
		p.set(row, start+i, v)
	}
}

func (p *stubPool) Row(row int) []int64 {
	n := 0
	for pos := range p.rows[row] {
		n = max(n, pos+1)
	}
	out := make([]int64, n)
	for pos, v := range p.rows[row] {
		out[pos] = v
	}
	return out
}

func (p *stubPool) Size() int                          { return 0 }
func (p *stubPool) MaxContextLen() int                 { return 0 }
func (p *stubPool) AvailableSize() int                 { return 0 }
func (p *stubPool) Alloc([]*Req) ([]int, bool)         { panic("the radix cache allocates no row") }
func (p *stubPool) Free(*Req)                          {}
func (p *stubPool) SchedulableTokenCapacity(c int) int { return c }
func (p *stubPool) AllocAuxToLengths(_, _ []int)       {}

// tracedReq is a request as the recorder encodes it.
type tracedReq struct {
	ID             string      `json:"req"`
	KV             ReqKV       `json:"kv"`
	Row            []int64     `json:"row"`
	OriginInputIDs []int64     `json:"origin_input_ids"`
	OutputIDs      []int64     `json:"output_ids"`
	FillIDs        []int64     `json:"full_untruncated_fill_ids"`
	ExtraKey       *string     `json:"extra_key"`
	CacheSalt      *string     `json:"cache_salt"`
	Priority       *int        `json:"priority"`
	SessionID      *string     `json:"session_id"`
	LastNode       *NodeID     `json:"last_node"`
	Lock           *tracedLock `json:"lock"`
	PrefixLen      int         `json:"prefix_len"`
}

type tracedLock struct {
	Node    NodeID `json:"node"`
	Receipt struct {
		Node NodeID `json:"node_id"`
	} `json:"receipt"`
}

type replay struct {
	tr    Trace
	c     *RadixCache
	alloc *stubAllocator
	pool  *stubPool
	nodes *Nodes
	bySeq map[int]Call
	// desc is, for each call, the calls it made at any depth, in order.
	desc map[int][]Call
}

func (r *replay) port(s NodeID) NodeID {
	p, err := r.nodes.Port(s)
	if err != nil {
		panic(err)
	}
	return p
}

func (r *replay) pair(s, p NodeID) error { return r.nodes.Pair(s, p) }

func (r *replay) req(raw json.RawMessage) (*Req, error) {
	var q tracedReq
	if err := json.Unmarshal(raw, &q); err != nil {
		return nil, err
	}
	out := &Req{ID: q.ID, OriginInputIDs: q.OriginInputIDs, OutputIDs: q.OutputIDs, FillIDs: q.FillIDs,
		ExtraKey: q.ExtraKey, CacheSalt: q.CacheSalt, SessionID: q.SessionID, KV: q.KV, PrefixLen: q.PrefixLen}
	if q.Priority != nil {
		out.Priority = *q.Priority
	}
	if q.LastNode != nil {
		n := r.port(*q.LastNode)
		out.LastNode = &n
	}
	if q.Lock != nil {
		out.Lock = &TreeLock{Node: r.port(q.Lock.Node), Receipt: LockReceipt{Node: r.port(q.Lock.Receipt.Node)}}
	}
	if q.KV.ReqPoolIdx != nil {
		got := r.pool.Row(*q.KV.ReqPoolIdx)
		for i, s := range q.Row {
			if i >= len(got) || got[i] != s {
				return nil, fmt.Errorf("request %s: row %d holds %v after the replayed writes; SGLang's held %v",
					q.ID, *q.KV.ReqPoolIdx, got, q.Row)
			}
		}
	}
	return out, nil
}

// within is the calls at iface that the call numbered seq made, at any depth, in order.
func (r *replay) within(seq int, iface Interface, methods ...string) []Call {
	if r.desc == nil {
		r.inCache(Call{})
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

func replayRadix(tr Trace) (int, error) {
	r := &replay{tr: tr, nodes: NewNodes(), pool: &stubPool{rows: map[int]map[int]int64{}}}
	var cfg struct {
		Params struct {
			PageSize int    `json:"page_size"`
			Disable  bool   `json:"disable"`
			Events   bool   `json:"enable_kv_cache_events"`
			Dynamo   bool   `json:"dynamo_kv_event_format"`
			Policy   string `json:"eviction_policy"`
			IsEagle  bool   `json:"is_eagle"`
		} `json:"params"`
	}
	built := false
	n := 0
	for _, c := range tr.Calls {
		switch {
		case c.Iface == ReqToTokenIface && c.Method == "write" && !r.inCache(c):
			slots, err := c.RowWrite()
			if err != nil {
				return n, err
			}
			for _, s := range slots {
				r.pool.set(s.Row, s.Pos, s.Slot)
			}
		case c.Iface == PrefixCacheIface && c.Class == "UnifiedRadixCache" && c.Method == "__init__":
			if built {
				return n, fmt.Errorf("seq %d: a second UnifiedRadixCache", c.Seq)
			}
			if err := c.Arg("params", &cfg.Params); err != nil {
				return n, err
			}
			p := cfg.Params
			r.alloc = &stubAllocator{page: p.PageSize}
			var err error
			r.c, err = NewRadixCache(RadixConfig{PageSize: p.PageSize, Disable: p.Disable, IsEagle: p.IsEagle,
				KVCacheEvents: p.Events, DynamoFormat: p.Dynamo, EvictionPolicy: p.Policy}, r.alloc, r.pool)
			if err != nil {
				return n, err
			}
			built = true
		case c.Iface == PrefixCacheIface && c.Class == "UnifiedRadixCache":
			if !built {
				return n, fmt.Errorf("seq %d: %s before the cache was built", c.Seq, c.Method)
			}
			if err := r.call(c); err != nil {
				return n, fmt.Errorf("seq %d %s: %w", c.Seq, c.Method, err)
			}
			n++
		}
	}
	return n, nil
}

// inCache reports whether c was made by a call at the radix cache.
func (r *replay) inCache(c Call) bool {
	if r.bySeq == nil {
		r.bySeq = map[int]Call{}
		for _, x := range r.tr.Calls {
			r.bySeq[x.Seq] = x
		}
	}
	for c.In != nil {
		c = r.bySeq[*c.In]
		if c.Iface == PrefixCacheIface {
			return true
		}
	}
	return false
}

func decode[T any](raw json.RawMessage) (T, error) {
	var v T
	err := json.Unmarshal(raw, &v)
	return v, err
}

func (r *replay) call(c Call) error {
	// SGLang's answers to the calls this one makes at the allocator, for the stand-in to give.
	r.alloc.avail = nil
	for _, x := range r.within(c.Seq, AllocatorIface, "available_size") {
		v, err := decode[int](x.Ret)
		if err != nil {
			return err
		}
		r.alloc.avail = append(r.alloc.avail, v)
	}
	r.alloc.made, r.pool.made = nil, nil
	if err := r.answer(c); err != nil {
		return err
	}
	// The calls the cache made in turn.
	want, err := r.wantMade(c)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(madeOf(r.alloc.made), madeOf(want.alloc)) {
		return fmt.Errorf("the port called the allocator %v; SGLang's cache called it %v", r.alloc.made, want.alloc)
	}
	if !reflect.DeepEqual(madeOf(r.pool.made), madeOf(want.pool)) {
		return fmt.Errorf("the port wrote the pool %v; SGLang's cache wrote %v", r.pool.made, want.pool)
	}
	return nil
}

func madeOf(m []madeCall) []madeCall {
	if len(m) == 0 {
		return nil
	}
	return m
}

type wantCalls struct{ alloc, pool []madeCall }

func (r *replay) wantMade(c Call) (wantCalls, error) {
	var w wantCalls
	for _, x := range r.within(c.Seq, AllocatorIface, "available_size", "free_segment", "free_segments", "free") {
		switch x.Method {
		case "available_size":
			w.alloc = append(w.alloc, madeCall{"available_size", ""})
		case "free_segment":
			var s []int64
			var start int
			if err := x.Arg("free_index", &s); err != nil {
				return w, err
			}
			if err := x.Arg("start_pos", &start); err != nil {
				return w, err
			}
			w.alloc = append(w.alloc, madeCall{"free_segment", fmt.Sprint(s, start)})
		case "free_segments":
			var segs [][2]json.RawMessage
			if err := x.Arg("segments", &segs); err != nil {
				return w, err
			}
			var parts []string
			for _, sg := range segs {
				s, err := decode[[]int64](sg[0])
				if err != nil {
					return w, err
				}
				start, err := decode[int](sg[1])
				if err != nil {
					return w, err
				}
				parts = append(parts, fmt.Sprint(s, start))
			}
			w.alloc = append(w.alloc, madeCall{"free_segments", fmt.Sprint(parts)})
		case "free":
			w.alloc = append(w.alloc, madeCall{"free", ""})
		}
	}
	for _, x := range r.within(c.Seq, ReqToTokenIface, "write") {
		slots, err := x.RowWrite()
		if err != nil {
			return w, err
		}
		if len(slots) == 0 {
			continue
		}
		var vals []int64
		for _, s := range slots {
			vals = append(vals, s.Slot)
		}
		w.pool = append(w.pool, madeCall{"write", fmt.Sprint(slots[0].Row, slots[0].Pos, vals)})
	}
	return w, nil
}

type tracedKey struct {
	Tokens    []int64 `json:"tokens"`
	ExtraKey  *string `json:"extra_key"`
	CacheSalt *string `json:"cache_salt"`
	Bigram    bool    `json:"bigram"`
}

func (k tracedKey) key() Key {
	return Key(k)
}

func (r *replay) answer(c Call) error {
	ret := func(v any) error { return c.Answer(v) }
	switch c.Method {
	case "reset":
		r.c.Reset()
	case "match_prefix":
		var p struct {
			Params struct {
				Key tracedKey       `json:"key"`
				Req json.RawMessage `json:"req"`
			} `json:"params"`
		}
		if err := c.Arg("params", &p.Params); err != nil {
			return err
		}
		got := r.c.MatchPrefix(MatchPrefixParams{Key: p.Params.Key.key()})
		var want MatchResult
		if err := ret(&want); err != nil {
			return err
		}
		return r.sameMatch(got, want)
	case "touch_prefix":
		var k tracedKey
		if err := c.Arg("key", &k); err != nil {
			return err
		}
		r.c.TouchPrefix(k.key())
	case "insert":
		var p struct {
			Params struct {
				Key           tracedKey `json:"key"`
				Value         []int64   `json:"value"`
				PrevPrefixLen int       `json:"prev_prefix_len"`
				InsertedLen   int       `json:"inserted_len"`
				Priority      int       `json:"priority"`
				SessionID     *string   `json:"session_id"`
			} `json:"params"`
		}
		if err := c.Arg("params", &p.Params); err != nil {
			return err
		}
		got := r.c.Insert(InsertParams{Key: p.Params.Key.key(), Value: p.Params.Value, PrevPrefixLen: p.Params.PrevPrefixLen,
			InsertedLen: p.Params.InsertedLen, Priority: p.Params.Priority, SessionID: p.Params.SessionID})
		var want InsertResult
		if err := ret(&want); err != nil {
			return err
		}
		if got.PrefixLen != want.PrefixLen {
			return fmt.Errorf("prefix_len %d, SGLang's %d", got.PrefixLen, want.PrefixLen)
		}
		return r.pair(want.LastDeviceNode, got.LastDeviceNode)
	case "evict", "evict_for_alloc":
		var p struct {
			Params struct {
				NumTokens int `json:"num_tokens"`
			} `json:"params"`
		}
		if err := c.Arg("params", &p.Params); err != nil {
			return err
		}
		var got EvictResult
		if c.Method == "evict" {
			got = r.c.Evict(EvictParams{NumTokens: p.Params.NumTokens})
		} else {
			got = r.c.EvictForAlloc(EvictParams{NumTokens: p.Params.NumTokens})
		}
		var want EvictResult
		if err := ret(&want); err != nil {
			return err
		}
		if got != want {
			return fmt.Errorf("evicted %d tokens, SGLang %d", got.NumTokensEvicted, want.NumTokensEvicted)
		}
	case "inc_lock_ref":
		var id NodeID
		if err := c.Arg("node_id", &id); err != nil {
			return err
		}
		got := r.c.IncLockRef(r.port(id))
		var want LockReceipt
		if err := ret(&want); err != nil {
			return err
		}
		if (got.Delta == nil) != (want.Delta == nil) || (got.Delta != nil && *got.Delta != *want.Delta) {
			return fmt.Errorf("lock delta %v, SGLang's %v", deref(got.Delta), deref(want.Delta))
		}
	case "dec_lock_ref":
		var id NodeID
		var p struct {
			Node NodeID `json:"node_id"`
		}
		if err := c.Arg("node_id", &id); err != nil {
			return err
		}
		if err := c.Arg("params", &p); err != nil {
			return err
		}
		r.c.DecLockRef(r.port(id), LockReceipt{Node: r.port(p.Node)})
	case "lock":
		var id NodeID
		if err := c.Arg("node_id", &id); err != nil {
			return err
		}
		got := r.c.Lock(r.port(id))
		var want tracedLock
		if err := ret(&want); err != nil {
			return err
		}
		if err := r.pair(want.Node, got.Node); err != nil {
			return err
		}
		return r.pair(want.Receipt.Node, got.Receipt.Node)
	case "unlock":
		var l *tracedLock
		if err := c.Arg("lock", &l); err != nil {
			return err
		}
		if l != nil {
			r.c.Unlock(&TreeLock{Node: r.port(l.Node), Receipt: LockReceipt{Node: r.port(l.Receipt.Node)}})
		}
	case "checkpoint":
		req, err := r.req(c.Args["req"])
		if err != nil {
			return err
		}
		var upTo int
		if err := c.Arg("up_to", &upTo); err != nil {
			return err
		}
		r.c.Checkpoint(req, upTo)
		if len(c.ReqAfter) != 1 {
			return fmt.Errorf("%d requests after", len(c.ReqAfter))
		}
		after, err := decode[tracedReq](c.ReqAfter[0])
		if err != nil {
			return err
		}
		if req.KV.CacheProtectedLen != after.KV.CacheProtectedLen || req.KV.CacheInsertedLen != after.KV.CacheInsertedLen {
			return fmt.Errorf("protected %d inserted %d after; SGLang's %d and %d", req.KV.CacheProtectedLen,
				req.KV.CacheInsertedLen, after.KV.CacheProtectedLen, after.KV.CacheInsertedLen)
		}
		if after.LastNode != nil {
			if err := r.pair(*after.LastNode, *req.LastNode); err != nil {
				return err
			}
		}
		if after.Lock != nil {
			return r.pair(after.Lock.Node, req.Lock.Node)
		}
	case "free_kv_row":
		var kv ReqKV
		var ranges [][2]int
		if err := c.Arg("kv", &kv); err != nil {
			return err
		}
		if err := c.Arg("ranges", &ranges); err != nil {
			return err
		}
		r.c.FreeKVRow(&Req{KV: kv}, ranges)
	case "prefix_device_indices":
		req, err := r.req(c.Args["req"])
		if err != nil {
			return err
		}
		var want []int64
		if err := ret(&want); err != nil {
			return err
		}
		if got := r.c.PrefixDeviceIndices(req); !slices.Equal(got, want) {
			return fmt.Errorf("prefix slots %v, SGLang's %v", got, want)
		}
	case "evictable_size", "protected_size", "full_evictable_size", "full_protected_size", "swa_reprefill_tail_tokens":
		var want int
		if err := ret(&want); err != nil {
			return err
		}
		got := map[string]func() int{"evictable_size": r.c.EvictableSize, "protected_size": r.c.ProtectedSize,
			"full_evictable_size": r.c.FullEvictableSize, "full_protected_size": r.c.FullProtectedSize,
			"swa_reprefill_tail_tokens": func() int { return r.c.SWAReprefillTailTokens(nil) }}[c.Method]()
		if got != want {
			return fmt.Errorf("%d, SGLang's %d", got, want)
		}
	case "total_size":
		var want [2]int
		if err := ret(&want); err != nil {
			return err
		}
		a, b := r.c.TotalSize()
		if [2]int{a, b} != want {
			return fmt.Errorf("%v, SGLang's %v", [2]int{a, b}, want)
		}
	case "supports_swa", "supports_mamba", "supports_prefix_sharing", "supports_fast_match_prefix", "claim_kv_row":
		var want bool
		if err := ret(&want); err != nil {
			return err
		}
		got := map[string]func() bool{"supports_swa": r.c.SupportsSWA, "supports_mamba": r.c.SupportsMamba,
			"supports_prefix_sharing": r.c.SupportsPrefixSharing, "supports_fast_match_prefix": r.c.SupportsFastMatchPrefix,
			"claim_kv_row": func() bool { return r.c.ClaimKVRow(nil) }}[c.Method]()
		if got != want {
			return fmt.Errorf("%v, SGLang's %v", got, want)
		}
	case "take_events":
		var want []Event
		if err := ret(&want); err != nil {
			return err
		}
		got := r.c.TakeEvents()
		if !reflect.DeepEqual(madeEvents(got), madeEvents(want)) {
			return fmt.Errorf("events %s, SGLang's %s", js(got), js(want))
		}
	case "session_records":
		var want map[string]json.RawMessage
		if err := ret(&want); err != nil {
			return err
		}
		if len(want) != len(r.c.SessionRecords()) {
			return fmt.Errorf("session records %v, SGLang's %v", r.c.SessionRecords(), want)
		}
	case "on_release", "maybe_hand_to_session", "flush_pending_backups", "finish_storage_prefetch_admission",
		"release_host_resources":
		// No effect without a session or a host tier; the calls it would make are checked.
	default:
		return fmt.Errorf("no replay for %s", c.Method)
	}
	return nil
}

func madeEvents(e []Event) []Event {
	if len(e) == 0 {
		return nil
	}
	return e
}

func js(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func deref(p *int) any {
	if p == nil {
		return nil
	}
	return *p
}

func (r *replay) sameMatch(got, want MatchResult) error {
	if got.DevicePrefixLen != want.DevicePrefixLen || got.FullKVHitLength != want.FullKVHitLength ||
		got.HostHitLength != want.HostHitLength {
		return fmt.Errorf("matched %d (%d hit, %d host); SGLang %d (%d, %d)", got.DevicePrefixLen, got.FullKVHitLength,
			got.HostHitLength, want.DevicePrefixLen, want.FullKVHitLength, want.HostHitLength)
	}
	for _, p := range [][2]NodeID{{want.LastDeviceNode, got.LastDeviceNode}, {want.BestMatchNode, got.BestMatchNode},
		{want.LastHostNode, got.LastHostNode}} {
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

func (a *stubAllocator) AllocExtend(_, _ []int, _ []int64, _ int) ([]int64, bool) {
	panic("the radix cache allocates nothing")
}
func (a *stubAllocator) AllocDecode(_ []int, _ []int64) ([]int64, bool) {
	panic("the radix cache allocates nothing")
}

func (a *stubAllocator) MambaFullCacheDonor() MambaFullCacheDonor           { return nil }
func (a *stubAllocator) GetExtendAllocationDemand(_, _ []int, n, _ int) int { return n }

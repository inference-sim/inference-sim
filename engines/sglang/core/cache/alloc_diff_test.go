// Original to blis-sglang.

package cache

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"
)

// The allocator and request-pool replay (FID-5, SGL-6): every call the scheduler, the tree cache
// or the driver made at SGLang's allocator and request pool, in every recorded and direct trace,
// is made on the port in the same order, and the port must give SGLang's answer: the exact slots
// an alloc returns, every available size, every capacity figure, every row a request was handed,
// and every row's contents whenever SGLang read one. Calls the allocator made into the tree cache
// are answered from the trace, and the port must make exactly those calls.

// PoolsGoldenDir holds the direct traces at the allocators and pools alone, with their sizing
// (oracle/run_pools.py).
const PoolsGoldenDir = "testdata/golden-pools"

func loadPoolGoldens(t *testing.T) map[string]Trace {
	t.Helper()
	trs, err := LoadTraces(PoolsGoldenDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(trs) == 0 {
		t.Fatalf("no trace under %s", PoolsGoldenDir)
	}
	return trs
}

func TestAllocatorAndRequestPoolAnswerAsSGLang(t *testing.T) {
	all := loadGoldens(t)
	for name, tr := range loadPoolGoldens(t) {
		all["pools/"+name] = tr
	}
	for name, tr := range all {
		t.Run(name, func(t *testing.T) {
			r := newReplay(t, tr)
			r.run()
			if r.entries == 0 {
				t.Fatal("no allocator or request-pool call was replayed")
			}
		})
	}
}

type replay struct {
	t       *testing.T
	tr      Trace
	bySeq   map[int]Call
	alloc   Allocator
	pool    *RequestPool
	kv      KVPool
	entries int
}

// atPools is whether a call is at the allocator or the request pool.
func atPools(c Call) bool { return c.Iface == AllocatorIface || c.Iface == ReqToTokenIface }

func newReplay(t *testing.T, tr Trace) *replay {
	r := &replay{t: t, tr: tr, bySeq: map[int]Call{}}
	for _, c := range tr.Calls {
		r.bySeq[c.Seq] = c
	}
	return r
}

// owner is the nearest call at the allocator or request pool that made c, at any depth, or nil.
func (r *replay) owner(c Call) *Call {
	for c.In != nil {
		p := r.bySeq[*c.In]
		if atPools(p) {
			return &p
		}
		c = p
	}
	return nil
}

func (r *replay) descends(c Call, seq int) bool {
	for c.In != nil {
		if *c.In == seq {
			return true
		}
		c = r.bySeq[*c.In]
	}
	return false
}

func (r *replay) run() {
	for _, c := range r.tr.Calls {
		if r.owner(c) != nil {
			continue // replayed by the call that made it
		}
		r.checkRows(c)
		if atPools(c) || (c.Iface == KVPoolIface && c.Method == "__init__") {
			r.exec(c)
		}
	}
}

// checkRows compares each request row a call's arguments carry, as SGLang read it when the call
// began, with the port's row.
func (r *replay) checkRows(c Call) {
	if r.pool == nil {
		return
	}
	for name, raw := range c.Args {
		for _, q := range tracedReqs(raw) {
			if q.KV.ReqPoolIdx == nil || q.Row == nil {
				continue
			}
			got := r.pool.Row(*q.KV.ReqPoolIdx)
			if len(got) < len(q.Row) || !reflect.DeepEqual(got[:len(q.Row)], q.Row) {
				r.t.Errorf("seq %d %s.%s(%s): request %s's row %d is %v in the port, %v in SGLang",
					c.Seq, c.Iface, c.Method, name, q.Req, *q.KV.ReqPoolIdx, got, q.Row)
			}
		}
	}
}

// tracedReq is a request as the recorder encodes it.
type tracedReq struct {
	Req  string  `json:"req"`
	KV   ReqKV   `json:"kv"`
	Row  []int64 `json:"row"`
	Fill []int64 `json:"full_untruncated_fill_ids"`
}

// tracedReqs finds the requests in an encoded argument: one request or a list of them.
func tracedReqs(raw json.RawMessage) []tracedReq {
	var one tracedReq
	if json.Unmarshal(raw, &one) == nil && one.Req != "" {
		return []tracedReq{one}
	}
	var many []tracedReq
	if json.Unmarshal(raw, &many) == nil {
		var out []tracedReq
		for _, q := range many {
			if q.Req != "" {
				out = append(out, q)
			}
		}
		return out
	}
	return nil
}

func (q tracedReq) toReq() *Req {
	kv := q.KV
	if kv.ReqPoolIdx != nil {
		i := *kv.ReqPoolIdx
		kv.ReqPoolIdx = &i
	}
	return &Req{ID: q.Req, FillIDs: q.Fill, KV: kv}
}

func (r *replay) arg(c Call, name string, v any) {
	r.t.Helper()
	if err := c.Arg(name, v); err != nil {
		r.t.Fatal(err)
	}
}

// want compares the port's answer with SGLang's, both as JSON.
func (r *replay) want(c Call, got any) {
	r.t.Helper()
	g, err := json.Marshal(got)
	if err != nil {
		r.t.Fatal(err)
	}
	var a, b any
	if err := json.Unmarshal(g, &a); err != nil {
		r.t.Fatal(err)
	}
	if err := json.Unmarshal(c.Ret, &b); err != nil {
		r.t.Fatalf("seq %d %s.%s: SGLang's answer %s: %v", c.Seq, c.Iface, c.Method, c.Ret, err)
	}
	if !reflect.DeepEqual(a, b) {
		r.t.Errorf("seq %d %s.%s(%v): the port answers %s, SGLang %s", c.Seq, c.Iface, c.Method, argText(c), g, c.Ret)
	}
}

func argText(c Call) string {
	b, _ := json.Marshal(c.Args)
	if len(b) > 200 {
		b = append(b[:200], "..."...)
	}
	return string(b)
}

// exec makes call c on the port, and checks the answer. A call SGLang raised on must panic.
func (r *replay) exec(c Call) {
	r.entries++
	if c.Method == "__init__" {
		r.build(c)
		return
	}
	if (c.Iface == AllocatorIface && r.alloc == nil) || (c.Iface == ReqToTokenIface && r.pool == nil) {
		r.t.Fatalf("seq %d %s.%s before it was built", c.Seq, c.Iface, c.Method)
	}
	tree := &scriptedTree{r: r, call: c}
	panicked := func() (p any) {
		defer func() { p = recover() }()
		if c.Iface == AllocatorIface {
			r.execAllocator(c, tree)
		} else {
			r.execRequestPool(c)
		}
		return nil
	}()
	if (panicked != nil) != (c.Raise != "") {
		r.t.Errorf("seq %d %s.%s: SGLang raised %q, the port panicked with %v", c.Seq, c.Iface, c.Method, c.Raise, panicked)
	}
	tree.done()
}

func (r *replay) execAllocator(c Call, tree *scriptedTree) {
	a := r.alloc
	switch c.Method {
	case "alloc":
		var n int
		r.arg(c, "need_size", &n)
		got, ok := a.Alloc(n)
		r.wantSlots(c, got, ok)
	case "alloc_extend":
		var prefix, seq []int
		var last []int64
		var n int
		r.arg(c, "prefix_lens_cpu", &prefix)
		r.arg(c, "seq_lens_cpu", &seq)
		r.arg(c, "last_loc", &last)
		r.arg(c, "extend_num_tokens", &n)
		got, ok := a.AllocExtend(prefix, seq, last, n)
		r.wantSlots(c, got, ok)
	case "alloc_decode":
		var seq []int
		var last []int64
		r.arg(c, "seq_lens_cpu", &seq)
		r.arg(c, "last_loc", &last)
		got, ok := a.AllocDecode(seq, last)
		r.wantSlots(c, got, ok)
	case "available_size":
		r.want(c, a.AvailableSize())
	case "free":
		var s []int64
		r.arg(c, "free_index", &s)
		a.Free(s)
	case "free_segment":
		var s []int64
		var start int
		r.arg(c, "free_index", &s)
		r.arg(c, "start_pos", &start)
		a.FreeSegment(s, start)
	case "free_segments":
		var raw [][2]json.RawMessage
		r.arg(c, "segments", &raw)
		segs := make([]Segment, len(raw))
		for i, p := range raw {
			if err := json.Unmarshal(p[0], &segs[i].Slots); err != nil {
				r.t.Fatal(err)
			}
			if err := json.Unmarshal(p[1], &segs[i].StartPos); err != nil {
				r.t.Fatal(err)
			}
		}
		a.FreeSegments(segs)
	case "free_group_begin":
		a.FreeGroupBegin()
	case "free_group_end":
		a.FreeGroupEnd()
	case "clear":
		a.Clear()
	case "evict_to_free_tokens":
		var n int
		r.arg(c, "num_tokens", &n)
		a.EvictToFreeTokens(tree, n)
	case "check_decode_capacity":
		var n int
		r.arg(c, "num_tokens", &n)
		r.want(c, a.CheckDecodeCapacity(n, tree))
	case "request_slot_reserve":
		var has bool
		var n int
		r.arg(c, "has_req_pool_slot", &has)
		r.arg(c, "prefill_tokens", &n)
		r.want(c, a.RequestSlotReserve(has, n))
	case "create_prefill_budget":
		var n int
		r.arg(c, "num_mixed_decode_tokens", &n)
		if b := a.CreatePrefillBudget(tree, n); b == nil {
			r.t.Errorf("seq %d: create_prefill_budget gave no budget; SGLang gave %s", c.Seq, c.Ret)
		}
	case "max_new_tokens_for_memory":
		var in, maxNew, capacity int
		var swa, chunk *int
		r.arg(c, "input_tokens", &in)
		r.arg(c, "max_new_tokens", &maxNew)
		r.arg(c, "token_capacity", &capacity)
		r.arg(c, "sliding_window_size", &swa)
		r.arg(c, "chunk_size", &chunk)
		got, ok := a.MaxNewTokensForMemory(in, maxNew, capacity, swa, chunk)
		if !ok {
			r.want(c, nil)
		} else {
			r.want(c, got)
		}
	case "verify_byte_accounting":
		r.want(c, a.VerifyByteAccounting())
	case "get_kvcache":
		var ret struct {
			T string `json:"_t"`
		}
		if err := c.Answer(&ret); err != nil {
			r.t.Fatal(err)
		}
		if got := a.GetKVCache(); got != r.kv || poolClass(got) != ret.T {
			r.t.Errorf("seq %d: get_kvcache gives a %s, SGLang a %s", c.Seq, poolClass(got), ret.T)
		}
	default:
		r.t.Errorf("seq %d: the replay has no allocator call %s", c.Seq, c.Method)
	}
}

func (r *replay) wantSlots(c Call, got []int64, ok bool) {
	r.t.Helper()
	if !ok {
		r.want(c, nil)
	} else {
		r.want(c, got)
	}
}

func poolClass(p KVPool) string {
	switch p.(type) {
	case *MHAPool:
		return "MHATokenToKVPool"
	case *MLAPool:
		return "MLATokenToKVPool"
	case *DSAPool:
		return "DSATokenToKVPool"
	}
	return fmt.Sprintf("%T", p)
}

func (r *replay) execRequestPool(c Call) {
	p := r.pool
	switch c.Method {
	case "available_size":
		r.want(c, p.AvailableSize())
	case "schedulable_token_capacity":
		var n int
		r.arg(c, "physical_capacity", &n)
		r.want(c, p.SchedulableTokenCapacity(n))
	case "write":
		slots, err := c.RowWrite()
		if err != nil {
			r.t.Fatal(err)
		}
		for _, s := range slots {
			p.Write(s.Row, s.Pos, []int64{s.Slot})
		}
	case "alloc":
		var raw json.RawMessage
		r.arg(c, "reqs", &raw)
		traced := tracedReqs(raw)
		reqs := make([]*Req, len(traced))
		for i, q := range traced {
			reqs[i] = q.toReq()
		}
		got, ok := p.Alloc(reqs)
		if !ok {
			r.want(c, nil)
			return
		}
		r.want(c, got)
		r.wantReqsAfter(c, reqs)
	case "free":
		var raw json.RawMessage
		r.arg(c, "req", &raw)
		traced := tracedReqs(raw)
		if len(traced) != 1 {
			r.t.Fatalf("seq %d: free of %d requests", c.Seq, len(traced))
		}
		req := traced[0].toReq()
		p.Free(req)
		r.wantReqsAfter(c, []*Req{req})
	case "alloc_aux_to_lengths":
		var idx, lens []int
		r.arg(c, "req_pool_indices_cpu", &idx)
		r.arg(c, "target_seq_lens_cpu", &lens)
		p.AllocAuxToLengths(idx, lens)
	case "clear":
		p.Clear()
	default:
		r.t.Errorf("seq %d: the replay has no request-pool call %s", c.Seq, c.Method)
	}
}

// wantReqsAfter compares each request's row index after the call with SGLang's.
func (r *replay) wantReqsAfter(c Call, reqs []*Req) {
	byID := map[string]*Req{}
	for _, q := range reqs {
		byID[q.ID] = q
	}
	for _, raw := range c.ReqAfter {
		for _, after := range tracedReqs(raw) {
			q, ok := byID[after.Req]
			if !ok {
				continue
			}
			if !reflect.DeepEqual(q.KV.ReqPoolIdx, after.KV.ReqPoolIdx) {
				r.t.Errorf("seq %d %s: request %s holds row %v in the port, %v in SGLang", c.Seq, c.Method, q.ID,
					deref(q.KV.ReqPoolIdx), deref(after.KV.ReqPoolIdx))
			}
		}
	}
}

func deref(p *int) any {
	if p == nil {
		return nil
	}
	return *p
}

// build constructs the port's pool, request pool or allocator from SGLang's constructor call, and
// checks the sizing it records.
func (r *replay) build(c Call) {
	var ret map[string]json.RawMessage
	if err := c.Answer(&ret); err != nil {
		r.t.Fatal(err)
	}
	num := func(name string) int {
		var v int
		r.arg(c, name, &v)
		return v
	}
	switch c.Iface {
	case ReqToTokenIface:
		r.pool = NewRequestPool(num("size"), num("max_context_len"))
		r.wantSizing(c, ret, map[string]any{"size": r.pool.Size(), "max_context_len": r.pool.MaxContextLen()})
	case KVPoolIface:
		pool, want := r.buildKVPool(c, num)
		r.kv = pool
		r.wantSizing(c, ret, want)
	case AllocatorIface:
		if r.kv == nil {
			r.t.Fatalf("seq %d: the allocator is built before its KV pool", c.Seq)
		}
		var needSort bool
		r.arg(c, "need_sort", &needSort)
		switch c.Class {
		case "TokenToKVPoolAllocator":
			r.alloc = NewTokenAllocator(num("size"), r.kv, needSort)
		case "PagedTokenToKVPoolAllocator":
			r.alloc = NewPagedAllocator(num("size"), num("page_size"), r.kv, needSort)
		default:
			r.t.Fatalf("seq %d: a %s, which the replay does not build", c.Seq, c.Class)
		}
		r.wantSizing(c, ret, map[string]any{"size": r.alloc.Size(), "page_size": r.alloc.PageSize(),
			"need_sort": needSort})
	}
}

// buildKVPool builds the port's pool from SGLang's constructor call, with the sizing attributes
// the port gives for it.
func (r *replay) buildKVPool(c Call, num func(string) int) (KVPool, map[string]any) {
	var dtype string
	r.arg(c, "dtype", &dtype)
	dt, err := ParseDType(dtype)
	if err != nil {
		r.t.Fatal(err)
	}
	common := func(p KVPool) map[string]any {
		return map[string]any{"size": p.Size(), "page_size": p.PageSize(), "layer_num": p.LayerNum(),
			"dtype": "torch." + string(dt), "store_dtype": "torch." + string(dt.Store())}
	}
	switch c.Class {
	case "MHATokenToKVPool":
		var vhd *int
		r.arg(c, "v_head_dim", &vhd)
		var postCapture bool
		r.arg(c, "post_capture_active", &postCapture)
		pool, err := NewMHAPool(MHAParams{Size: num("size"), PageSize: num("page_size"), DType: dt,
			HeadNum: num("head_num"), HeadDim: num("head_dim"), LayerNum: num("layer_num"), VHeadDim: vhd,
			PostCaptureActive: postCapture})
		if err != nil {
			r.t.Fatal(err)
		}
		want := common(pool)
		want["head_num"], want["head_dim"], want["v_head_dim"] = pool.HeadNum, pool.HeadDim, pool.VHeadDim
		return pool, want
	case "MLATokenToKVPool":
		pool, err := NewMLAPool(MLAParams{Size: num("size"), PageSize: num("page_size"), DType: dt,
			KVLoraRank: num("kv_lora_rank"), QKRopeHeadDim: num("qk_rope_head_dim"), LayerNum: num("layer_num")})
		if err != nil {
			r.t.Fatal(err)
		}
		want := common(pool)
		want["kv_lora_rank"], want["qk_rope_head_dim"] = pool.KVLoraRank, pool.QKRopeHeadDim
		return pool, want
	case "DSATokenToKVPool":
		p := DSAParams{Size: num("size"), PageSize: num("page_size"), DType: dt, KVLoraRank: num("kv_lora_rank"),
			QKRopeHeadDim: num("qk_rope_head_dim"), LayerNum: num("layer_num"), IndexHeadDim: num("index_head_dim"),
			KVCacheDim: num("kv_cache_dim"), IndexKPool: num("index_kpool")}
		r.arg(c, "index_buf_size", &p.IndexBufSize)
		r.arg(c, "skip_topk_layers", &p.SkipTopKLayers)
		pool, err := NewDSAPool(p)
		if err != nil {
			r.t.Fatal(err)
		}
		want := common(pool)
		want["kv_lora_rank"], want["qk_rope_head_dim"] = pool.KVLoraRank, pool.QKRopeHeadDim
		want["index_head_dim"] = pool.IndexHeadDim
		return pool, want
	}
	r.t.Fatalf("seq %d: a %s, which the replay does not build", c.Seq, c.Class)
	return nil, nil
}

// wantSizing compares each sizing attribute the port gives with the one SGLang's constructor set.
func (r *replay) wantSizing(c Call, ret map[string]json.RawMessage, got map[string]any) {
	r.t.Helper()
	for k, v := range got {
		raw, ok := ret[k]
		if !ok {
			r.t.Errorf("seq %d %s: SGLang records no %s", c.Seq, c.Class, k)
			continue
		}
		g, _ := json.Marshal(v)
		var a, b any
		_ = json.Unmarshal(g, &a)
		_ = json.Unmarshal(raw, &b)
		if !reflect.DeepEqual(a, b) {
			r.t.Errorf("seq %d %s: %s is %s in the port, %s in SGLang", c.Seq, c.Class, k, g, raw)
		}
	}
}

// scriptedTree answers the tree-cache calls an allocator call makes, from the trace: each must
// be the next call SGLang's allocator made at the tree cache within that call, and the pool calls
// SGLang's tree cache made within it are made on the port before it answers.
type scriptedTree struct {
	PrefixCache
	r    *replay
	call Call
	next int
}

func (s *scriptedTree) take(method string) Call {
	calls := s.r.tr.Within(s.call.Seq, PrefixCacheIface)
	if s.next >= len(calls) {
		s.r.t.Fatalf("seq %d %s: the port calls the tree cache's %s, which SGLang's allocator did not",
			s.call.Seq, s.call.Method, method)
	}
	c := calls[s.next]
	s.next++
	if c.Method != method {
		s.r.t.Fatalf("seq %d %s: the port calls the tree cache's %s where SGLang called %s",
			s.call.Seq, s.call.Method, method, c.Method)
	}
	// The pool calls the tree cache made while answering, in order.
	for _, d := range s.r.tr.Calls {
		if atPools(d) && s.r.descends(d, c.Seq) {
			if o := s.r.owner(d); o != nil && o.Seq == s.call.Seq {
				s.r.exec(d)
			}
		}
	}
	return c
}

func (s *scriptedTree) done() {
	if n := len(s.r.tr.Within(s.call.Seq, PrefixCacheIface)); s.next != n {
		s.r.t.Errorf("seq %d %s: the port made %d of SGLang's %d calls at the tree cache", s.call.Seq,
			s.call.Method, s.next, n)
	}
}

func (s *scriptedTree) SupportsPrefixSharing() bool {
	var v bool
	if err := s.take("supports_prefix_sharing").Answer(&v); err != nil {
		s.r.t.Fatal(err)
	}
	return v
}

func (s *scriptedTree) SupportsMamba() bool {
	var v bool
	if err := s.take("supports_mamba").Answer(&v); err != nil {
		s.r.t.Fatal(err)
	}
	return v
}

func (s *scriptedTree) EvictForAlloc(p EvictParams) EvictResult {
	c := s.take("evict_for_alloc")
	var args struct {
		NumTokens int `json:"num_tokens"`
	}
	if err := c.Arg("params", &args); err != nil {
		s.r.t.Fatal(err)
	}
	if args.NumTokens != p.NumTokens {
		s.r.t.Errorf("seq %d: the port evicts %d tokens, SGLang %d", c.Seq, p.NumTokens, args.NumTokens)
	}
	var out EvictResult
	if err := c.Answer(&out); err != nil {
		s.r.t.Fatal(err)
	}
	return out
}

func (s *scriptedTree) EvictableSize() int {
	var v int
	if err := s.take("evictable_size").Answer(&v); err != nil {
		s.r.t.Fatal(err)
	}
	return v
}

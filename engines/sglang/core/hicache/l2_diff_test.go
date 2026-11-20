// Original to blis-sglang.

package hicache

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/blis-sglang/core/cache"
)

// The HiCache radix cache (cache.RadixCache with L2 attached) answers every call SGLang's
// UnifiedRadixCache answered while SGLang's own scheduler ran a HiCache scenario with no storage
// backend (FID-5), and makes, in turn, the calls SGLang's made: at the tree core (in order, with
// the same answers, node for node under one consistent pairing), at the controller (the same
// writes, loads and submissions, with the same slots), at the host pool and at the device
// allocator. The controller and the host pool are the ports (S20); the device allocator and the
// request pool are stand-ins that answer as the trace says SGLang's did. Each step boundary falls
// where SGLang's did, so an ack is reaped at the step SGLang reaped it. A scenario with a storage
// backend is refused at start-up, citing HiCache L3's issue.
func TestTheHiCacheRadixCacheAnswersEveryTracedCallAsSGLangDoes(t *testing.T) {
	replayed, refused := 0, 0
	for name, tr := range loadL2Traces(t) {
		if tr.Kind != "recorded" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			n, err := replayL2(tr)
			var ref storageRefused
			if errors.As(err, &ref) {
				if !strings.Contains(ref.err.Error(), StorageRefusalIssue) {
					t.Fatalf("a storage scenario was refused without citing %s: %v", StorageRefusalIssue, ref.err)
				}
				refused++
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			replayed += n
			t.Logf("%d calls replayed", n)
		})
	}
	if replayed == 0 {
		t.Fatal("no call at the HiCache radix cache was replayed")
	}
	if refused == 0 {
		t.Fatal("no storage scenario reached the refusal")
	}
}

// l2TraceDirs are the host tier's own trace directories under goldenDir and the kind each holds:
// SGLang's scheduler running oracle/scenarios/hicache-l2/recorded, and the driver's sequences in
// oracle/scenarios/hicache-l2/direct.
var l2TraceDirs = [][2]string{{"l2-recorded", "recorded"}, {"l2-direct", "direct"}}

// loadL2Traces is every HiCache trace: the framework's and the host tier's own.
func loadL2Traces(t *testing.T) map[string]hiTrace {
	t.Helper()
	out := loadTraces(t)
	for _, d := range l2TraceDirs {
		paths, err := filepath.Glob(filepath.Join(goldenDir, d[0], "*.trace.json"))
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range paths {
			raw, err := os.ReadFile(p)
			if err != nil {
				t.Fatal(err)
			}
			var tr hiTrace
			if err := json.Unmarshal(raw, &tr); err != nil {
				t.Fatalf("%s: %v", p, err)
			}
			if tr.Kind != d[1] {
				t.Fatalf("%s: a %q trace in the %s directory", p, tr.Kind, d[0])
			}
			out[d[0]+"/"+strings.TrimSuffix(filepath.Base(p), ".trace.json")] = tr
		}
	}
	return out
}

type storageRefused struct{ err error }

func (s storageRefused) Error() string { return s.err.Error() }

// --- stand-ins ---

type l2Made struct{ m, args string }

// l2Alloc is the device allocator: AvailableSize and Alloc answer as SGLang's did, and every
// call the cache makes is recorded.
type l2Alloc struct {
	avail  []int
	allocs [][]int64
	made   []l2Made
	page   int
}

func (a *l2Alloc) AvailableSize() int {
	a.made = append(a.made, l2Made{"available_size", ""})
	if len(a.avail) == 0 {
		return -1
	}
	v := a.avail[0]
	a.avail = a.avail[1:]
	return v
}
func (a *l2Alloc) FreeSegment(s []int64, start int) {
	a.made = append(a.made, l2Made{"free_segment", fmt.Sprint(s, start)})
}
func (a *l2Alloc) FreeSegments(segs []cache.Segment) {
	var parts []string
	for _, s := range segs {
		parts = append(parts, fmt.Sprint(s.Slots, s.StartPos))
	}
	a.made = append(a.made, l2Made{"free_segments", fmt.Sprint(parts)})
}
func (a *l2Alloc) Alloc(n int) ([]int64, bool) {
	if len(a.allocs) == 0 {
		return nil, false
	}
	v := a.allocs[0]
	a.allocs = a.allocs[1:]
	return v, v != nil
}
func (a *l2Alloc) SetHicacheTransferDoneEvent(string, *Event)                     {}
func (a *l2Alloc) Size() int                                                      { return 0 }
func (a *l2Alloc) PageSize() int                                                  { return a.page }
func (a *l2Alloc) Free(s []int64)                                                 { a.made = append(a.made, l2Made{"free", fmt.Sprint(s)}) }
func (a *l2Alloc) FreeGroupBegin()                                                {}
func (a *l2Alloc) FreeGroupEnd()                                                  {}
func (a *l2Alloc) Clear()                                                         {}
func (a *l2Alloc) EvictToFreeTokens(cache.PrefixCache, int)                       {}
func (a *l2Alloc) CheckDecodeCapacity(int, cache.PrefixCache) bool                { return false }
func (a *l2Alloc) RequestSlotReserve(bool, int) int                               { return 0 }
func (a *l2Alloc) CreatePrefillBudget(cache.PrefixCache, int) cache.PrefillBudget { return nil }
func (a *l2Alloc) MaxNewTokensForMemory(int, int, int, *int, *int) (int, bool)    { return 0, false }
func (a *l2Alloc) VerifyByteAccounting() []string                                 { return nil }
func (a *l2Alloc) GetKVCache() cache.KVPool                                       { return nil }
func (a *l2Alloc) AllocExtend(_, _ []int, _ []int64, _ int) ([]int64, bool) {
	panic("the radix cache allocates nothing")
}
func (a *l2Alloc) AllocDecode(_ []int, _ []int64) ([]int64, bool) {
	panic("the radix cache allocates nothing")
}
func (a *l2Alloc) MambaFullCacheDonor() cache.MambaFullCacheDonor { return nil }
func (a *l2Alloc) GetExtendAllocationDemand(_, _ []int, n, _ int) int {
	return n
}

// l2Pool holds the rows the trace's writes build, and records the cache's writes.
type l2Pool struct {
	rows map[int]map[int]int64
	made []l2Made
}

func (p *l2Pool) set(row, pos int, slot int64) {
	if p.rows[row] == nil {
		p.rows[row] = map[int]int64{}
	}
	p.rows[row][pos] = slot
}
func (p *l2Pool) Write(row, start int, values []int64) {
	if len(values) == 0 {
		return
	}
	p.made = append(p.made, l2Made{"write", fmt.Sprint(row, start, values)})
	for i, v := range values {
		p.set(row, start+i, v)
	}
}
func (p *l2Pool) Row(row int) []int64 {
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
func (p *l2Pool) Size() int                          { return 0 }
func (p *l2Pool) MaxContextLen() int                 { return 0 }
func (p *l2Pool) AvailableSize() int                 { return 0 }
func (p *l2Pool) Alloc([]*cache.Req) ([]int, bool)   { panic("the radix cache allocates no row") }
func (p *l2Pool) Free(*cache.Req)                    {}
func (p *l2Pool) SchedulableTokenCapacity(c int) int { return c }
func (p *l2Pool) AllocAuxToLengths(_, _ []int)       {}

// --- recorders ---

// figure is one call made at the tree core, the controller or the host pool: its method and the
// figures of its arguments and answer that a replay compares. Node ids are kept as the port's
// (or SGLang's) and compared through the replay's pairing; host-lock uuids likewise.
type figure struct {
	m     string
	ints  []int
	nodes []cache.NodeID
	slots [][]int64
	uuids []int64
	kinds []string
}

func (f figure) String() string {
	return fmt.Sprintf("%s%v%v%v%v", f.m, f.ints, f.nodes, f.slots, f.kinds)
}

type recorder struct{ log []figure }

func (r *recorder) add(f figure) { r.log = append(r.log, f) }

func actionFigure(f *figure, as []cache.CacheAction) {
	for _, a := range as {
		f.kinds = append(f.kinds, string(a.Kind))
		switch a.Kind {
		case cache.FreeDeviceKVAction:
			f.slots = append(f.slots, a.Indices...)
		case cache.BackupKVAction:
			f.nodes = append(f.nodes, a.NodeIDs...)
		case cache.UpdatePendingBackupOnNodeSplitAction:
			f.nodes = append(f.nodes, a.OldNode, a.NewNode)
		}
	}
}

func uuidOf(r cache.LockReceipt) []int64 {
	if r.HostUUID == nil {
		return nil
	}
	return []int64{*r.HostUUID}
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// recCore is the tree core, recording each call the cache makes there.
type recCore struct {
	cache.TreeCore
	recorder
}

func (c *recCore) Reset() { c.TreeCore.Reset(); c.add(figure{m: "reset"}) }
func (c *recCore) RootNodeHandle(k *string) cache.NodeID {
	n := c.TreeCore.RootNodeHandle(k)
	c.add(figure{m: "root_node_handle", nodes: []cache.NodeID{n}})
	return n
}
func (c *recCore) MatchPrefix(p cache.MatchPrefixParams) cache.MatchResult {
	r := c.TreeCore.MatchPrefix(p)
	f := figure{m: "match_prefix", ints: []int{r.DevicePrefixLen, r.FullKVHitLength, r.HostHitLength},
		nodes: []cache.NodeID{r.LastDeviceNode, r.BestMatchNode, r.LastHostNode}}
	actionFigure(&f, r.Actions)
	c.add(f)
	return r
}
func (c *recCore) SupportsFastMatchPrefix() bool {
	c.add(figure{m: "supports_fast_match_prefix"})
	return c.TreeCore.SupportsFastMatchPrefix()
}
func (c *recCore) CollectFullDeviceIndices(from, until cache.NodeID) []int64 {
	s := c.TreeCore.CollectFullDeviceIndices(from, until)
	c.add(figure{m: "collect_full_device_indices", slots: [][]int64{s}})
	return s
}
func insertFigure(m string, s cache.InsertStep) figure {
	f := figure{m: m}
	if s.Result != nil {
		f.ints = []int{s.Result.PrefixLen}
		f.nodes = []cache.NodeID{s.Result.LastDeviceNode}
	}
	actionFigure(&f, s.Actions)
	return f
}
func (c *recCore) BeginInsert(p cache.InsertParams) cache.InsertStep {
	s := c.TreeCore.BeginInsert(p)
	c.add(insertFigure("begin_insert", s))
	return s
}
func (c *recCore) ResumeInsert() cache.InsertStep {
	s := c.TreeCore.ResumeInsert()
	c.add(insertFigure("resume_insert", s))
	return s
}
func (c *recCore) HasOngoingInsert() bool {
	c.add(figure{m: "has_ongoing_insert"})
	return c.TreeCore.HasOngoingInsert()
}
func (c *recCore) EndInsert() []cache.CacheAction {
	a := c.TreeCore.EndInsert()
	f := figure{m: "end_insert"}
	actionFigure(&f, a)
	c.add(f)
	return a
}
func (c *recCore) IncLockRef(n cache.NodeID) cache.LockReceipt {
	r := c.TreeCore.IncLockRef(n)
	c.add(figure{m: "inc_lock_ref", ints: []int{*r.Delta}})
	return r
}
func (c *recCore) DecLockRef(n cache.NodeID, r cache.LockReceipt) {
	c.TreeCore.DecLockRef(n, r)
	c.add(figure{m: "dec_lock_ref", nodes: []cache.NodeID{n}})
}
func (c *recCore) IncHostLockRef(n cache.NodeID) cache.LockReceipt {
	r := c.TreeCore.IncHostLockRef(n)
	c.add(figure{m: "inc_host_lock_ref", nodes: []cache.NodeID{n}, uuids: uuidOf(r)})
	return r
}
func (c *recCore) DecHostLockRef(n cache.NodeID, r cache.LockReceipt) {
	c.TreeCore.DecHostLockRef(n, r)
	c.add(figure{m: "dec_host_lock_ref", nodes: []cache.NodeID{n}, uuids: uuidOf(r)})
}
func (c *recCore) EvictDeviceStart(t cache.ComponentType, n int) {
	c.TreeCore.EvictDeviceStart(t, n)
	c.add(figure{m: "evict_device_start", ints: []int{n}})
}
func (c *recCore) EvictDeviceNextNode(t cache.ComponentType, tr cache.Tracker) cache.EvictionStep {
	s := c.TreeCore.EvictDeviceNextNode(t, tr)
	f := figure{m: "evict_device_next_node"}
	if s.Node != nil {
		f.nodes = []cache.NodeID{*s.Node}
	}
	c.add(f)
	return s
}
func stepFigure(m string, s cache.EvictionStep) figure {
	f := figure{m: m, ints: []int{s.Tracker[cache.FullComponentType], s.UnbackedTokens},
		slots: append(append([][]int64{}, s.DeviceFrees[cache.FullComponentType]...), s.HostFrees[cache.FullComponentType]...),
		nodes: s.BackupKV}
	return f
}
func (c *recCore) EvictDeviceLeaf(n cache.NodeID, wb bool) cache.EvictionStep {
	s := c.TreeCore.EvictDeviceLeaf(n, wb)
	c.add(stepFigure("evict_device_leaf", s))
	return s
}
func (c *recCore) EvictDeviceEnd(t cache.ComponentType) {
	c.TreeCore.EvictDeviceEnd(t)
	c.add(figure{m: "evict_device_end"})
}
func (c *recCore) Demote(n cache.NodeID) cache.EvictionStep {
	s := c.TreeCore.Demote(n)
	c.add(stepFigure("demote", s))
	return s
}
func (c *recCore) DropSubtreeNoHost(n cache.NodeID) cache.EvictionStep {
	s := c.TreeCore.DropSubtreeNoHost(n)
	f := stepFigure("drop_subtree_no_host", s)
	f.ints = append(f.ints, b2i(s.IsDropped))
	c.add(f)
	return s
}
func (c *recCore) DriveHostEviction(t cache.ComponentType, n int) cache.EvictionStep {
	s := c.TreeCore.DriveHostEviction(t, n)
	f := stepFigure("drive_host_eviction", s)
	f.ints = f.ints[:1]
	c.add(f)
	return s
}
func (c *recCore) sized(m string, v int) int { c.add(figure{m: m, ints: []int{v}}); return v }
func (c *recCore) EvictableSize() int        { return c.sized("evictable_size", c.TreeCore.EvictableSize()) }
func (c *recCore) ProtectedSize() int        { return c.sized("protected_size", c.TreeCore.ProtectedSize()) }
func (c *recCore) FullEvictableSize() int {
	return c.sized("full_evictable_size", c.TreeCore.FullEvictableSize())
}
func (c *recCore) FullProtectedSize() int {
	return c.sized("full_protected_size", c.TreeCore.FullProtectedSize())
}
func (c *recCore) IsFullDeviceEvicted(n cache.NodeID) bool {
	v := c.TreeCore.IsFullDeviceEvicted(n)
	c.add(figure{m: "is_full_device_evicted", ints: []int{b2i(v)}})
	return v
}
func (c *recCore) BuildBackupSpec(n cache.NodeID) []int64 {
	s := c.TreeCore.BuildBackupSpec(n)
	c.add(figure{m: "build_backup_spec", slots: [][]int64{s}})
	return s
}
func (c *recCore) CommitBackup(n cache.NodeID, host []int64) {
	c.TreeCore.CommitBackup(n, host)
	c.add(figure{m: "commit_backup", nodes: []cache.NodeID{n}, slots: [][]int64{host}})
}
func (c *recCore) RotationPhaseOf(n cache.NodeID) *int {
	c.add(figure{m: "rotation_phase_of", nodes: []cache.NodeID{n}})
	return c.TreeCore.RotationPhaseOf(n)
}
func (c *recCore) MarkWriteThroughPending(ids []cache.NodeID, ack cache.NodeID) []cache.NodeID {
	out := c.TreeCore.MarkWriteThroughPending(ids, ack)
	c.add(figure{m: "mark_write_through_pending", nodes: append([]cache.NodeID{ack}, out...)})
	return out
}
func (c *recCore) FinishWriteThrough(ids []cache.NodeID, ack cache.NodeID) {
	c.TreeCore.FinishWriteThrough(ids, ack)
	c.add(figure{m: "finish_write_through", nodes: append([]cache.NodeID{ack}, ids...)})
}
func (c *recCore) BuildLoadBackSpec(n cache.NodeID) cache.LoadBackSpec {
	s := c.TreeCore.BuildLoadBackSpec(n)
	c.add(figure{m: "build_load_back_spec", slots: [][]int64{s.Host}, nodes: s.Nodes})
	return s
}
func (c *recCore) CommitLoadBack(n cache.NodeID, device []int64, s cache.LoadBackSpec) []cache.CacheAction {
	a := c.TreeCore.CommitLoadBack(n, device, s)
	c.add(figure{m: "commit_load_back", nodes: []cache.NodeID{n}, slots: [][]int64{device}})
	return a
}
func (c *recCore) FinishLoadBack(n cache.NodeID) {
	c.TreeCore.FinishLoadBack(n)
	c.add(figure{m: "finish_load_back", nodes: []cache.NodeID{n}})
}

// recCtrl is the controller as the cache calls it, recording each call.
type recCtrl struct {
	ControllerPort
	recorder
}

func (c *recCtrl) Write(device []int64, prio *int, node int64, flush bool) ([]int64, bool) {
	h, ok := c.ControllerPort.Write(device, prio, node, flush)
	c.add(figure{m: "write", slots: [][]int64{device, h}, nodes: []cache.NodeID{cache.NodeID(node)}, ints: []int{b2i(flush), b2i(ok)}})
	return h, ok
}
func (c *recCtrl) StartWriting() { c.ControllerPort.StartWriting(); c.add(figure{m: "start_writing"}) }
func (c *recCtrl) Load(host []int64, prio *int, node int64) ([]int64, bool) {
	d, ok := c.ControllerPort.Load(host, prio, node)
	c.add(figure{m: "load", slots: [][]int64{host, d}, nodes: []cache.NodeID{cache.NodeID(node)}, ints: []int{b2i(ok)}})
	return d, ok
}
func (c *recCtrl) StartLoading() (int, error) {
	v, err := c.ControllerPort.StartLoading()
	c.add(figure{m: "start_loading", ints: []int{v}})
	return v, err
}

// recHostPool is the host pool as the cache calls it directly.
type recHostPool struct {
	HostPoolPort
	recorder
}

func (h *recHostPool) AvailableSize() int {
	v := h.HostPoolPort.AvailableSize()
	h.add(figure{m: "available_size", ints: []int{v}})
	return v
}
func (h *recHostPool) Free(idx []int64) (int, error) {
	n, err := h.HostPoolPort.Free(idx)
	h.add(figure{m: "free", slots: [][]int64{idx}, ints: []int{n}})
	return n, err
}
func (h *recHostPool) Destroy() { h.HostPoolPort.Destroy(); h.add(figure{m: "destroy"}) }

// --- the replay ---

type l2Replay struct {
	tr    hiTrace
	l2    *L2
	clock *Clock
	alloc *l2Alloc
	pool  *l2Pool
	core  *recCore
	ctrl  *recCtrl
	host  *recHostPool
	nodes *cache.Nodes
	uuids map[int64]int64
	bySeq map[int]cache.Call
	desc  map[int][]cache.Call
}

func (r *l2Replay) port(s cache.NodeID) cache.NodeID {
	p, err := r.nodes.Port(s)
	if err != nil {
		panic(err)
	}
	return p
}

// pairUUID pairs SGLang's host-lock uuid with the port's: both count up from FULL's range, but
// SGLang's counter is the process's.
func (r *l2Replay) pairUUID(s, p int64) error {
	if q, ok := r.uuids[s]; ok && q != p {
		return fmt.Errorf("SGLang's host-lock uuid %d is the port's %d, and now %d", s, q, p)
	}
	r.uuids[s] = p
	return nil
}

func (r *l2Replay) parent(c cache.Call) *cache.Call {
	if c.In == nil {
		return nil
	}
	p := r.bySeq[*c.In]
	return &p
}

// under is whether c was made, at any depth, inside a call at iface.
func (r *l2Replay) under(c cache.Call, iface cache.Interface) bool {
	for p := r.parent(c); p != nil; p = r.parent(*p) {
		if p.Iface == iface {
			return true
		}
	}
	return false
}

// within is the calls the call numbered seq made at iface, at any depth, in order.
func (r *l2Replay) within(seq int, iface cache.Interface) []cache.Call {
	if r.desc == nil {
		r.desc = map[int][]cache.Call{}
		for _, c := range r.tr.Calls {
			for x := c; x.In != nil; x = r.bySeq[*x.In] {
				r.desc[*x.In] = append(r.desc[*x.In], c)
			}
		}
	}
	var out []cache.Call
	for _, c := range r.desc[seq] {
		if c.Iface == iface {
			out = append(out, c)
		}
	}
	return out
}

func (r *l2Replay) direct(seq int, iface cache.Interface) []cache.Call {
	var out []cache.Call
	for _, c := range r.within(seq, iface) {
		if *c.In == seq {
			out = append(out, c)
		}
	}
	return out
}

func decodeAs[T any](raw json.RawMessage) (T, error) {
	var v T
	err := json.Unmarshal(raw, &v)
	return v, err
}

func replayL2(tr hiTrace) (int, error) {
	r := &l2Replay{tr: tr, nodes: cache.NewNodes(), uuids: map[int64]int64{}, bySeq: map[int]cache.Call{},
		pool: &l2Pool{rows: map[int]map[int]int64{}}, clock: &Clock{}}
	for _, c := range tr.Calls {
		r.bySeq[c.Seq] = c
	}
	if err := r.build(); err != nil {
		return 0, err
	}
	n, b := 0, 0
	for i, c := range tr.Calls {
		for b < len(tr.BoundaryAt) && tr.BoundaryAt[b] <= i {
			if _, err := r.l2.queues.Boundary(); err != nil {
				return n, err
			}
			b++
		}
		if r.under(c, cache.PrefixCacheIface) {
			continue
		}
		var err error
		switch {
		case c.Iface == cache.ReqToTokenIface && c.Method == "write":
			slots, e := c.RowWrite()
			for _, s := range slots {
				r.pool.set(s.Row, s.Pos, s.Slot)
			}
			err = e
		case c.Iface == cache.PrefixCacheIface && c.Class == "UnifiedRadixCache" && c.Method != "__init__":
			err = r.call(c)
			n++
		case c.In == nil && c.Iface == cache.TreeCoreIface && (c.Method == "inc_host_lock_ref" || c.Method == "dec_host_lock_ref"):
			// The scheduler locks a host-matched node through the cache, whose own method the
			// prefix-cache interface does not list (unified_radix_cache.py:1126-1136).
			err = r.call(c)
			n++
		case c.In == nil && c.Iface == cache.Interface(LayerCounterIface) && c.Method == "set_consumer":
			var idx int
			if err = c.Arg("index", &idx); err == nil {
				r.l2.queues.Counter().SetConsumer(idx)
			}
		}
		if err != nil {
			return n, fmt.Errorf("seq %d %s.%s: %w", c.Seq, c.Iface, c.Method, err)
		}
	}
	return n, nil
}

// build makes the cache, its host pool and its controller as SGLang's start-up made them, from
// the constructors' traced arguments.
func (r *l2Replay) build() error {
	var radix, hostInit, ctrlInit *cache.Call
	for i := range r.tr.Calls {
		c := &r.tr.Calls[i]
		switch {
		case c.Iface == cache.PrefixCacheIface && c.Class == "UnifiedRadixCache" && c.Method == "__init__" && radix == nil:
			radix = c
		case c.Iface == cache.Interface(HostPoolIface) && c.Method == "__init__" && hostInit == nil:
			hostInit = c
		case c.Iface == cache.Interface(ControllerIface) && c.Method == "__init__" && ctrlInit == nil:
			ctrlInit = c
		}
	}
	if radix == nil || hostInit == nil || ctrlInit == nil {
		return errors.New("the trace builds no HiCache radix cache")
	}
	var p struct {
		PageSize int    `json:"page_size"`
		Disable  bool   `json:"disable"`
		Events   bool   `json:"enable_kv_cache_events"`
		Dynamo   bool   `json:"dynamo_kv_event_format"`
		Policy   string `json:"eviction_policy"`
		IsEagle  bool   `json:"is_eagle"`
	}
	if err := radix.Arg("params", &p); err != nil {
		return err
	}
	var hs struct {
		Capacity int     `json:"device_capacity_tokens"`
		Ratio    float64 `json:"host_to_device_ratio"`
		SizeGB   float64 `json:"host_size"`
		Page     int     `json:"page_size"`
		Layout   string  `json:"layout"`
	}
	for k, v := range map[string]any{"device_capacity_tokens": &hs.Capacity, "host_to_device_ratio": &hs.Ratio,
		"host_size": &hs.SizeGB, "page_size": &hs.Page, "layout": &hs.Layout} {
		if err := hostInit.Arg(k, v); err != nil {
			return err
		}
	}
	var sizing struct {
		Size         int   `json:"size"`
		SizePerToken int64 `json:"size_per_token"`
	}
	if err := hostInit.Answer(&sizing); err != nil {
		return err
	}
	var cc struct {
		WritePolicy string  `json:"write_policy"`
		IO          string  `json:"io_backend"`
		Layers      int     `json:"transfer_layer_id_max"`
		Mode        string  `json:"host_memory_mode"`
		Storage     *string `json:"storage_backend"`
	}
	for k, v := range map[string]any{"write_policy": &cc.WritePolicy, "io_backend": &cc.IO, "transfer_layer_id_max": &cc.Layers,
		"host_memory_mode": &cc.Mode, "storage_backend": &cc.Storage} {
		if err := ctrlInit.Arg(k, v); err != nil {
			return err
		}
	}
	r.alloc = &l2Alloc{page: p.PageSize}
	rc, err := cache.NewRadixCache(cache.RadixConfig{PageSize: p.PageSize, Disable: p.Disable, IsEagle: p.IsEagle,
		KVCacheEvents: p.Events, DynamoFormat: p.Dynamo, EvictionPolicy: p.Policy}, r.alloc, r.pool)
	if err != nil {
		return err
	}
	host, err := NewHostPool(HostSizing{DeviceCapacityTokens: hs.Capacity, SizePerToken: sizing.SizePerToken,
		HostToDeviceRatio: hs.Ratio, HostSizeGB: hs.SizeGB, PageSize: hs.Page, Layout: hs.Layout})
	if err != nil {
		return err
	}
	if host.Size() != sizing.Size {
		return fmt.Errorf("the host pool holds %d slots; SGLang's held %d", host.Size(), sizing.Size)
	}
	cfg := L2Config{WritePolicy: cc.WritePolicy, IOBackend: cc.IO, Layout: hs.Layout, HostMemoryMode: cc.Mode}
	if cc.Storage != nil {
		cfg.StorageBackend = *cc.Storage
	}
	ctrl, err := NewController(ControllerConfig{WritePolicy: cc.WritePolicy, IOBackend: cc.IO, Layout: hs.Layout,
		PageSize: p.PageSize, TransferLayerIDMax: cc.Layers}, r.clock, host, host.Size(), r.alloc, nil)
	if err != nil {
		return err
	}
	r.l2, err = NewL2(rc, cfg, ctrl, host)
	if err != nil {
		if cfg.StorageBackend != "" {
			return storageRefused{err}
		}
		return err
	}
	rc.WrapCore(func(c cache.TreeCore) cache.TreeCore { r.core = &recCore{TreeCore: c}; return r.core })
	r.l2.WrapController(func(c ControllerPort) ControllerPort { r.ctrl = &recCtrl{ControllerPort: c}; return r.ctrl })
	r.l2.WrapHost(func(h HostPoolPort) HostPoolPort { r.host = &recHostPool{HostPoolPort: h}; return r.host })
	return nil
}

// call replays one call the scheduler made at the cache, and compares what the port answered
// and what it called in turn with SGLang's.
func (r *l2Replay) call(c cache.Call) error {
	r.alloc.avail, r.alloc.allocs = nil, nil
	for _, x := range r.within(c.Seq, cache.AllocatorIface) {
		switch {
		case x.Method == "available_size" && !r.under(x, cache.Interface(ControllerIface)):
			v, err := decodeAs[int](x.Ret)
			if err != nil {
				return err
			}
			r.alloc.avail = append(r.alloc.avail, v)
		case x.Method == "alloc":
			v, err := decodeAs[[]int64](x.Ret)
			if err != nil {
				return err
			}
			r.alloc.allocs = append(r.alloc.allocs, v)
		}
	}
	r.alloc.made, r.pool.made = nil, nil
	r.core.log, r.ctrl.log, r.host.log = nil, nil, nil
	if c.Raise != "" {
		if !l2Panics(func() { _ = r.answer(c) }) {
			return fmt.Errorf("SGLang raised %s and the port answered", c.Raise)
		}
		return nil
	}
	if err := r.answer(c); err != nil {
		return err
	}
	for _, chk := range []struct {
		iface cache.Interface
		got   []figure
	}{{cache.TreeCoreIface, r.core.log}, {cache.Interface(ControllerIface), r.ctrl.log}, {cache.Interface(HostPoolIface), r.host.log}} {
		want, err := r.wantFigures(c.Seq, chk.iface)
		if err != nil {
			return err
		}
		if c.Iface == cache.TreeCoreIface && chk.iface == cache.TreeCoreIface {
			// A call the scheduler made at the tree core through the cache is itself the cache's.
			f, _, err := r.wantFigure(chk.iface, c)
			if err != nil {
				return err
			}
			want = []figure{f}
		}
		if err := r.same(chk.iface, chk.got, want); err != nil {
			return err
		}
	}
	wa, wp, err := r.wantMade(c)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(nilIfEmpty(r.alloc.made), nilIfEmpty(wa)) {
		return fmt.Errorf("the port called the allocator %v; SGLang's cache called it %v", r.alloc.made, wa)
	}
	if !reflect.DeepEqual(nilIfEmpty(r.pool.made), nilIfEmpty(wp)) {
		return fmt.Errorf("the port wrote the request pool %v; SGLang's cache wrote %v", r.pool.made, wp)
	}
	return nil
}

func nilIfEmpty(m []l2Made) []l2Made {
	if len(m) == 0 {
		return nil
	}
	return m
}

func l2Panics(f func()) (p bool) {
	defer func() { p = recover() != nil }()
	f()
	return false
}

// wantMade is the allocator and request-pool calls SGLang's cache made, at any depth, outside
// its controller.
func (r *l2Replay) wantMade(c cache.Call) ([]l2Made, []l2Made, error) {
	var alloc, pool []l2Made
	for _, x := range r.within(c.Seq, cache.AllocatorIface) {
		if r.under(x, cache.Interface(ControllerIface)) {
			continue
		}
		switch x.Method {
		case "available_size":
			alloc = append(alloc, l2Made{"available_size", ""})
		case "free_segment":
			var s []int64
			var start int
			if err := x.Arg("free_index", &s); err != nil {
				return nil, nil, err
			}
			if err := x.Arg("start_pos", &start); err != nil {
				return nil, nil, err
			}
			alloc = append(alloc, l2Made{"free_segment", fmt.Sprint(s, start)})
		case "free_segments":
			var segs [][2]json.RawMessage
			if err := x.Arg("segments", &segs); err != nil {
				return nil, nil, err
			}
			var parts []string
			for _, sg := range segs {
				s, err := decodeAs[[]int64](sg[0])
				if err != nil {
					return nil, nil, err
				}
				st, err := decodeAs[int](sg[1])
				if err != nil {
					return nil, nil, err
				}
				parts = append(parts, fmt.Sprint(s, st))
			}
			alloc = append(alloc, l2Made{"free_segments", fmt.Sprint(parts)})
		case "free":
			var s []int64
			if err := x.Arg("free_index", &s); err != nil {
				return nil, nil, err
			}
			alloc = append(alloc, l2Made{"free", fmt.Sprint(s)})
		}
	}
	for _, x := range r.within(c.Seq, cache.ReqToTokenIface) {
		if x.Method != "write" {
			continue
		}
		slots, err := x.RowWrite()
		if err != nil {
			return nil, nil, err
		}
		if len(slots) == 0 {
			continue
		}
		var vals []int64
		for _, s := range slots {
			vals = append(vals, s.Slot)
		}
		pool = append(pool, l2Made{"write", fmt.Sprint(slots[0].Row, slots[0].Pos, vals)})
	}
	return alloc, pool, nil
}

// tracedAction is a cache action as the recorder encodes it.
type tracedAction struct {
	T          string          `json:"_t"`
	Indices    [][]int64       `json:"indices"`
	NodeIDs    []cache.NodeID  `json:"node_ids"`
	AckID      *cache.NodeID   `json:"ack_id"`
	OldNode    cache.NodeID    `json:"old_node_id"`
	NewNode    cache.NodeID    `json:"new_node_id"`
	NewChild   cache.NodeID    `json:"new_child_node_id"`
	Components json.RawMessage `json:"component_type"`
}

func tracedActionFigure(f *figure, as []tracedAction) {
	for _, a := range as {
		f.kinds = append(f.kinds, a.T)
		switch a.T {
		case "FreeDeviceKV":
			f.slots = append(f.slots, a.Indices...)
		case "BackupKV":
			f.nodes = append(f.nodes, a.NodeIDs...)
		case "UpdatePendingBackupOnNodeSplit":
			f.nodes = append(f.nodes, a.OldNode, a.NewNode)
		}
	}
}

type tracedReceipt struct {
	Node  cache.NodeID     `json:"node_id"`
	Host  map[string]int64 `json:"component_host_lock_uuids"`
	Delta *int             `json:"delta"`
}

func (t tracedReceipt) uuids() []int64 {
	if u, ok := t.Host["FULL"]; ok {
		return []int64{u}
	}
	return nil
}

type tracedStep struct {
	DeviceFrees map[string][][]int64 `json:"device_frees"`
	HostFrees   map[string][][]int64 `json:"host_frees"`
	Tracker     map[string]int       `json:"tracker"`
	Unbacked    int                  `json:"unbacked_tokens"`
	BackupKV    *tracedAction        `json:"backup_kv"`
	NodeID      *cache.NodeID        `json:"node_id"`
	IsDropped   bool                 `json:"is_dropped"`
}

func (s tracedStep) figure(m string) figure {
	f := figure{m: m, ints: []int{s.Tracker["FULL"], s.Unbacked},
		slots: append(append([][]int64{}, s.DeviceFrees["FULL"]...), s.HostFrees["FULL"]...)}
	if s.BackupKV != nil {
		f.nodes = s.BackupKV.NodeIDs
	}
	return f
}

// wantFigures is the calls at iface that the call numbered seq made itself, as figures in
// SGLang's ids.
//
// The host pool's are the cache's at any depth outside its controller: FULL frees host values
// through the component (components/full.py:519-523).
func (r *l2Replay) wantFigures(seq int, iface cache.Interface) ([]figure, error) {
	var out []figure
	calls := r.direct(seq, iface)
	if iface == cache.Interface(HostPoolIface) {
		calls = nil
		for _, x := range r.within(seq, iface) {
			if !r.under(x, cache.Interface(ControllerIface)) {
				calls = append(calls, x)
			}
		}
	}
	for _, x := range calls {
		f, skip, err := r.wantFigure(iface, x)
		if err != nil {
			return nil, fmt.Errorf("seq %d %s: %w", x.Seq, x.Method, err)
		}
		if !skip {
			out = append(out, f)
		}
	}
	return out, nil
}

func (r *l2Replay) wantFigure(iface cache.Interface, x cache.Call) (figure, bool, error) {
	f := figure{m: x.Method}
	var err error
	switch iface {
	case cache.Interface(HostPoolIface):
		switch x.Method {
		case "available_size":
			var v int
			err = x.Answer(&v)
			f.ints = []int{v}
		case "free":
			var idx []int64
			var n int
			if err = x.Arg("indices", &idx); err == nil {
				err = x.Answer(&n)
			}
			f.slots, f.ints = [][]int64{idx}, []int{n}
		case "destroy":
		default:
			return f, false, fmt.Errorf("no replay for the cache's host-pool call %s", x.Method)
		}
		return f, false, err
	case cache.Interface(ControllerIface):
		switch x.Method {
		case "write":
			var dev []int64
			var node cache.NodeID
			var flush bool
			var host *[]int64
			for k, v := range map[string]any{"device_indices": &dev, "node_id": &node, "flush": &flush} {
				if err = x.Arg(k, v); err != nil {
					return f, false, err
				}
			}
			if err = x.Answer(&host); err != nil {
				return f, false, err
			}
			var h []int64
			if host != nil {
				h = *host
			}
			f.slots, f.nodes, f.ints = [][]int64{dev, h}, []cache.NodeID{node}, []int{b2i(flush), b2i(host != nil)}
		case "load":
			var hostIdx []int64
			var node cache.NodeID
			var dev *[]int64
			if err = x.Arg("host_indices", &hostIdx); err != nil {
				return f, false, err
			}
			if err = x.Arg("node_id", &node); err != nil {
				return f, false, err
			}
			if err = x.Answer(&dev); err != nil {
				return f, false, err
			}
			var d []int64
			if dev != nil {
				d = *dev
			}
			f.slots, f.nodes, f.ints = [][]int64{hostIdx, d}, []cache.NodeID{node}, []int{b2i(dev != nil)}
		case "start_writing":
		case "start_loading":
			var v int
			err = x.Answer(&v)
			f.ints = []int{v}
		default:
			return f, false, fmt.Errorf("no replay for the cache's controller call %s", x.Method)
		}
		return f, false, err
	}
	// The tree core.
	switch x.Method {
	case "reset", "has_ongoing_insert", "supports_fast_match_prefix", "evict_device_end":
	case "set_hicache_enabled":
		return f, true, nil
	case "root_node_handle":
		var n cache.NodeID
		err = x.Answer(&n)
		f.nodes = []cache.NodeID{n}
	case "match_prefix":
		var m cache.MatchResult
		var acts struct {
			A []tracedAction `json:"cache_actions"`
		}
		if err = x.Answer(&m); err == nil {
			err = x.Answer(&acts)
		}
		f.ints = []int{m.DevicePrefixLen, m.FullKVHitLength, m.HostHitLength}
		f.nodes = []cache.NodeID{m.LastDeviceNode, m.BestMatchNode, m.LastHostNode}
		tracedActionFigure(&f, acts.A)
	case "collect_full_device_indices":
		var s []int64
		err = x.Answer(&s)
		f.slots = [][]int64{s}
	case "begin_insert", "resume_insert":
		var s struct {
			Actions []tracedAction      `json:"actions"`
			Result  *cache.InsertResult `json:"result"`
		}
		err = x.Answer(&s)
		if s.Result != nil {
			f.ints, f.nodes = []int{s.Result.PrefixLen}, []cache.NodeID{s.Result.LastDeviceNode}
		}
		tracedActionFigure(&f, s.Actions)
	case "end_insert":
		var a []tracedAction
		err = x.Answer(&a)
		tracedActionFigure(&f, a)
	case "inc_lock_ref":
		var lr tracedReceipt
		err = x.Answer(&lr)
		if lr.Delta != nil {
			f.ints = []int{*lr.Delta}
		}
	case "dec_lock_ref":
		var n cache.NodeID
		err = x.Arg("node_id", &n)
		f.nodes = []cache.NodeID{n}
	case "inc_host_lock_ref":
		var n cache.NodeID
		var lr tracedReceipt
		if err = x.Arg("node_id", &n); err == nil {
			err = x.Answer(&lr)
		}
		f.nodes, f.uuids = []cache.NodeID{n}, lr.uuids()
	case "dec_host_lock_ref":
		var n cache.NodeID
		var lr tracedReceipt
		if err = x.Arg("node_id", &n); err == nil {
			err = x.Arg("params", &lr)
		}
		f.nodes, f.uuids = []cache.NodeID{n}, lr.uuids()
	case "evict_device_start":
		var n int
		err = x.Arg("request_cnt", &n)
		f.ints = []int{n}
	case "evict_device_next_node":
		var s tracedStep
		err = x.Answer(&s)
		if s.NodeID != nil {
			f.nodes = []cache.NodeID{*s.NodeID}
		}
	case "evict_device_leaf", "demote":
		var s tracedStep
		err = x.Answer(&s)
		f = s.figure(x.Method)
	case "drop_subtree_no_host":
		var s tracedStep
		err = x.Answer(&s)
		f = s.figure(x.Method)
		f.ints = append(f.ints, b2i(s.IsDropped))
	case "drive_host_eviction":
		var s tracedStep
		err = x.Answer(&s)
		f = s.figure(x.Method)
		f.ints = f.ints[:1]
	case "evictable_size", "protected_size", "full_evictable_size", "full_protected_size":
		var v int
		err = x.Answer(&v)
		f.ints = []int{v}
	case "is_full_device_evicted":
		var v bool
		err = x.Answer(&v)
		f.ints = []int{b2i(v)}
	case "build_backup_spec":
		var s [2]json.RawMessage
		if err = x.Answer(&s); err == nil {
			var dev []int64
			dev, err = decodeAs[[]int64](s[0])
			f.slots = [][]int64{dev}
		}
	case "commit_backup":
		var n cache.NodeID
		var h []int64
		if err = x.Arg("node_id", &n); err == nil {
			err = x.Arg("host_indices", &h)
		}
		f.nodes, f.slots = []cache.NodeID{n}, [][]int64{h}
	case "rotation_phase_of":
		var n cache.NodeID
		err = x.Arg("node_id", &n)
		f.nodes = []cache.NodeID{n}
	case "mark_write_through_pending":
		var ack cache.NodeID
		var out []cache.NodeID
		if err = x.Arg("ack_id", &ack); err == nil {
			err = x.Answer(&out)
		}
		f.nodes = append([]cache.NodeID{ack}, out...)
	case "finish_write_through":
		var ack cache.NodeID
		var ids []cache.NodeID
		if err = x.Arg("ack_id", &ack); err == nil {
			err = x.Arg("node_ids", &ids)
		}
		f.nodes = append([]cache.NodeID{ack}, ids...)
	case "build_load_back_spec":
		var s [2]json.RawMessage
		if err = x.Answer(&s); err == nil {
			var kv struct {
				Host  []int64        `json:"host_indices"`
				Nodes []cache.NodeID `json:"nodes_to_load"`
			}
			err = json.Unmarshal(s[0], &kv)
			f.slots, f.nodes = [][]int64{kv.Host}, kv.Nodes
		}
	case "commit_load_back":
		var n cache.NodeID
		var d []int64
		if err = x.Arg("node_id", &n); err == nil {
			err = x.Arg("device_indices", &d)
		}
		f.nodes, f.slots = []cache.NodeID{n}, [][]int64{d}
	case "finish_load_back":
		var n cache.NodeID
		err = x.Arg("anchor_node_id", &n)
		f.nodes = []cache.NodeID{n}
	default:
		return f, false, fmt.Errorf("no replay for the cache's tree-core call %s", x.Method)
	}
	return f, false, err
}

func (r *l2Replay) same(iface cache.Interface, got, want []figure) error {
	names := func(fs []figure) []string {
		var out []string
		for _, f := range fs {
			out = append(out, f.m)
		}
		return out
	}
	if len(got) != len(want) {
		return fmt.Errorf("the cache made %d %s calls %v; SGLang's made %d %v", len(got), iface, names(got), len(want), names(want))
	}
	for i := range got {
		g, w := got[i], want[i]
		if g.m != w.m || !slices.Equal(g.ints, w.ints) || !slices.Equal(g.kinds, w.kinds) || len(g.slots) != len(w.slots) ||
			len(g.nodes) != len(w.nodes) || len(g.uuids) != len(w.uuids) {
			return fmt.Errorf("%s call %d: %v; SGLang's %v", iface, i, g, w)
		}
		for j := range g.slots {
			if !slices.Equal(g.slots[j], w.slots[j]) {
				return fmt.Errorf("%s call %d %s: slots %v; SGLang's %v", iface, i, g.m, g.slots, w.slots)
			}
		}
		for j := range g.nodes {
			if err := r.nodes.Pair(w.nodes[j], g.nodes[j]); err != nil {
				return fmt.Errorf("%s call %d %s: %w", iface, i, g.m, err)
			}
		}
		for j := range g.uuids {
			if err := r.pairUUID(w.uuids[j], g.uuids[j]); err != nil {
				return fmt.Errorf("%s call %d %s: %w", iface, i, g.m, err)
			}
		}
	}
	return nil
}

// tracedReq is a request as the recorder encodes it.
type tracedReq struct {
	ID             string        `json:"req"`
	KV             cache.ReqKV   `json:"kv"`
	Row            []int64       `json:"row"`
	OriginInputIDs []int64       `json:"origin_input_ids"`
	OutputIDs      []int64       `json:"output_ids"`
	FillIDs        []int64       `json:"full_untruncated_fill_ids"`
	ExtraKey       *string       `json:"extra_key"`
	CacheSalt      *string       `json:"cache_salt"`
	Priority       *int          `json:"priority"`
	SessionID      *string       `json:"session_id"`
	LastNode       *cache.NodeID `json:"last_node"`
	Lock           *tracedLock   `json:"lock"`
	PrefixLen      int           `json:"prefix_len"`
}

type tracedLock struct {
	Node    cache.NodeID  `json:"node"`
	Receipt tracedReceipt `json:"receipt"`
}

func (r *l2Replay) receipt(t tracedReceipt) cache.LockReceipt {
	out := cache.LockReceipt{Node: r.port(t.Node)}
	if u := t.uuids(); u != nil {
		p, ok := r.uuids[u[0]]
		if !ok {
			panic(fmt.Sprintf("SGLang's host-lock uuid %d was never named in an answer", u[0]))
		}
		out.HostUUID = &p
	}
	return out
}

func (r *l2Replay) lock(t *tracedLock) *cache.TreeLock {
	if t == nil {
		return nil
	}
	return &cache.TreeLock{Node: r.port(t.Node), Receipt: r.receipt(t.Receipt)}
}

func (r *l2Replay) req(raw json.RawMessage) (*cache.Req, error) {
	q, err := decodeAs[tracedReq](raw)
	if err != nil {
		return nil, err
	}
	out := &cache.Req{ID: q.ID, OriginInputIDs: q.OriginInputIDs, OutputIDs: q.OutputIDs, FillIDs: q.FillIDs,
		ExtraKey: q.ExtraKey, CacheSalt: q.CacheSalt, SessionID: q.SessionID, KV: q.KV, PrefixLen: q.PrefixLen,
		Lock: r.lock(q.Lock)}
	if q.Priority != nil {
		out.Priority = *q.Priority
	}
	if q.LastNode != nil {
		n := r.port(*q.LastNode)
		out.LastNode = &n
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

func (r *l2Replay) key(raw json.RawMessage) (cache.Key, error) { return decodeAs[cache.Key](raw) }

func (r *l2Replay) answer(c cache.Call) error {
	l := r.l2
	wantInt := func(got int) error {
		var want int
		if err := c.Answer(&want); err != nil {
			return err
		}
		if got != want {
			return fmt.Errorf("%d, SGLang's %d", got, want)
		}
		return nil
	}
	wantBool := func(got bool) error {
		var want bool
		if err := c.Answer(&want); err != nil {
			return err
		}
		if got != want {
			return fmt.Errorf("%v, SGLang's %v", got, want)
		}
		return nil
	}
	switch c.Method {
	case "inc_host_lock_ref":
		var n cache.NodeID
		if err := c.Arg("node_id", &n); err != nil {
			return err
		}
		got := l.IncHostLockRef(r.port(n))
		var want tracedReceipt
		if err := c.Answer(&want); err != nil {
			return err
		}
		if (got.HostUUID == nil) != (want.uuids() == nil) {
			return fmt.Errorf("host lock uuid %v, SGLang's %v", uuidOf(got), want.uuids())
		}
		if got.HostUUID != nil {
			return r.pairUUID(want.uuids()[0], *got.HostUUID)
		}
	case "dec_host_lock_ref":
		var n cache.NodeID
		var p tracedReceipt
		if err := c.Arg("node_id", &n); err != nil {
			return err
		}
		if err := c.Arg("params", &p); err != nil {
			return err
		}
		l.DecHostLockRef(r.port(n), r.receipt(p))
	case "match_prefix":
		var p struct {
			Key json.RawMessage `json:"key"`
		}
		if err := c.Arg("params", &p); err != nil {
			return err
		}
		k, err := r.key(p.Key)
		if err != nil {
			return err
		}
		got := l.MatchPrefix(cache.MatchPrefixParams{Key: k})
		var want cache.MatchResult
		if err := c.Answer(&want); err != nil {
			return err
		}
		if got.DevicePrefixLen != want.DevicePrefixLen || got.FullKVHitLength != want.FullKVHitLength ||
			got.HostHitLength != want.HostHitLength {
			return fmt.Errorf("matched %d (%d hit, %d host); SGLang %d (%d, %d)", got.DevicePrefixLen, got.FullKVHitLength,
				got.HostHitLength, want.DevicePrefixLen, want.FullKVHitLength, want.HostHitLength)
		}
		for _, pr := range [][2]cache.NodeID{{want.LastDeviceNode, got.LastDeviceNode}, {want.BestMatchNode, got.BestMatchNode},
			{want.LastHostNode, got.LastHostNode}} {
			if err := r.nodes.Pair(pr[0], pr[1]); err != nil {
				return err
			}
		}
	case "touch_prefix":
		var raw json.RawMessage
		if err := c.Arg("key", &raw); err != nil {
			return err
		}
		k, err := r.key(raw)
		if err != nil {
			return err
		}
		l.TouchPrefix(k)
	case "insert":
		var p struct {
			Key           json.RawMessage `json:"key"`
			Value         []int64         `json:"value"`
			PrevPrefixLen int             `json:"prev_prefix_len"`
			InsertedLen   int             `json:"inserted_len"`
			Priority      *int            `json:"priority"`
			SessionID     *string         `json:"session_id"`
		}
		if err := c.Arg("params", &p); err != nil {
			return err
		}
		k, err := r.key(p.Key)
		if err != nil {
			return err
		}
		ip := cache.InsertParams{Key: k, Value: p.Value, PrevPrefixLen: p.PrevPrefixLen, InsertedLen: p.InsertedLen, SessionID: p.SessionID}
		if p.Priority != nil {
			ip.Priority = *p.Priority
		}
		got := l.Insert(ip)
		var want cache.InsertResult
		if err := c.Answer(&want); err != nil {
			return err
		}
		if got.PrefixLen != want.PrefixLen {
			return fmt.Errorf("prefix_len %d, SGLang's %d", got.PrefixLen, want.PrefixLen)
		}
		return r.nodes.Pair(want.LastDeviceNode, got.LastDeviceNode)
	case "evict", "evict_for_alloc":
		var p struct {
			NumTokens int `json:"num_tokens"`
		}
		if err := c.Arg("params", &p); err != nil {
			return err
		}
		var got cache.EvictResult
		if c.Method == "evict" {
			got = l.Evict(cache.EvictParams{NumTokens: p.NumTokens})
		} else {
			got = l.EvictForAlloc(cache.EvictParams{NumTokens: p.NumTokens})
		}
		var want cache.EvictResult
		if err := c.Answer(&want); err != nil {
			return err
		}
		if got != want {
			return fmt.Errorf("evicted %d tokens, SGLang %d", got.NumTokensEvicted, want.NumTokensEvicted)
		}
	case "inc_lock_ref":
		var n cache.NodeID
		if err := c.Arg("node_id", &n); err != nil {
			return err
		}
		l.IncLockRef(r.port(n))
	case "dec_lock_ref":
		var n cache.NodeID
		var p tracedReceipt
		if err := c.Arg("node_id", &n); err != nil {
			return err
		}
		if err := c.Arg("params", &p); err != nil {
			return err
		}
		l.DecLockRef(r.port(n), r.receipt(p))
	case "lock":
		var n cache.NodeID
		if err := c.Arg("node_id", &n); err != nil {
			return err
		}
		got := l.Lock(r.port(n))
		var want tracedLock
		if err := c.Answer(&want); err != nil {
			return err
		}
		return r.nodes.Pair(want.Node, got.Node)
	case "unlock":
		var t *tracedLock
		if err := c.Arg("lock", &t); err != nil {
			return err
		}
		l.Unlock(r.lock(t))
	case "checkpoint":
		req, err := r.req(c.Args["req"])
		if err != nil {
			return err
		}
		var upTo int
		if err := c.Arg("up_to", &upTo); err != nil {
			return err
		}
		l.Checkpoint(req, upTo)
		after, err := decodeAs[tracedReq](c.ReqAfter[0])
		if err != nil {
			return err
		}
		if req.KV.CacheProtectedLen != after.KV.CacheProtectedLen || req.KV.CacheInsertedLen != after.KV.CacheInsertedLen {
			return fmt.Errorf("protected %d inserted %d after; SGLang's %d and %d", req.KV.CacheProtectedLen,
				req.KV.CacheInsertedLen, after.KV.CacheProtectedLen, after.KV.CacheInsertedLen)
		}
		if after.LastNode != nil {
			if err := r.nodes.Pair(*after.LastNode, *req.LastNode); err != nil {
				return err
			}
		}
		if after.Lock != nil {
			return r.nodes.Pair(after.Lock.Node, req.Lock.Node)
		}
	case "free_kv_row":
		var kv cache.ReqKV
		var ranges [][2]int
		if err := c.Arg("kv", &kv); err != nil {
			return err
		}
		if err := c.Arg("ranges", &ranges); err != nil {
			return err
		}
		l.FreeKVRow(&cache.Req{KV: kv}, ranges)
	case "prefix_device_indices":
		req, err := r.req(c.Args["req"])
		if err != nil {
			return err
		}
		var want []int64
		if err := c.Answer(&want); err != nil {
			return err
		}
		if got := l.PrefixDeviceIndices(req); !slices.Equal(got, want) {
			return fmt.Errorf("prefix slots %v, SGLang's %v", got, want)
		}
	case "evictable_size":
		return wantInt(l.EvictableSize())
	case "protected_size":
		return wantInt(l.ProtectedSize())
	case "swa_reprefill_tail_tokens":
		return wantInt(l.SWAReprefillTailTokens(nil))
	case "supports_swa":
		return wantBool(l.SupportsSWA())
	case "supports_mamba":
		return wantBool(l.SupportsMamba())
	case "supports_prefix_sharing":
		return wantBool(l.SupportsPrefixSharing())
	case "supports_fast_match_prefix":
		return wantBool(l.SupportsFastMatchPrefix())
	case "claim_kv_row":
		return wantBool(l.ClaimKVRow(nil))
	case "load_back_is_cache_owned":
		return wantBool(l.LoadBackIsCacheOwned())
	case "session_records":
		var want map[string]json.RawMessage
		if err := c.Answer(&want); err != nil {
			return err
		}
		if len(want) != len(l.SessionRecords()) {
			return fmt.Errorf("session records %v, SGLang's %v", l.SessionRecords(), want)
		}
	case "check_hicache_events":
		l.CheckHicacheEvents()
	case "flush_pending_backups":
		l.FlushPendingBackups()
	case "ready_to_load_host_cache":
		v, err := l.ReadyToLoadHostCache()
		if err != nil {
			return err
		}
		return wantInt(v)
	case "release_host_resources":
		l.ReleaseHostResources()
	case "init_load_back":
		var p struct {
			Best     cache.NodeID    `json:"best_match_node"`
			HostHit  int             `json:"host_hit_length"`
			MemQuota *int            `json:"mem_quota"`
			Req      json.RawMessage `json:"req"`
		}
		if err := c.Arg("params", &p); err != nil {
			return err
		}
		req, err := r.req(p.Req)
		if err != nil {
			return err
		}
		n, node := l.InitLoadBack(InitLoadBackParams{BestMatchNode: r.port(p.Best), HostHitLength: p.HostHit, MemQuota: p.MemQuota, Req: req})
		var want [2]json.RawMessage
		if err := c.Answer(&want); err != nil {
			return err
		}
		wn, err := decodeAs[int](want[0])
		if err != nil {
			return err
		}
		wnode, err := decodeAs[cache.NodeID](want[1])
		if err != nil {
			return err
		}
		if n != wn {
			return fmt.Errorf("loaded %d tokens, SGLang %d", n, wn)
		}
		return r.nodes.Pair(wnode, node)
	case "on_release", "maybe_hand_to_session", "finish_storage_prefetch_admission":
		// No effect outside a session and without storage; the calls they would make are checked.
	default:
		return fmt.Errorf("no replay for %s", c.Method)
	}
	return nil
}

// The HiCache radix cache does what SGLang's did on each of the driver's own sequences: for each
// entry, the calls made in turn at the tree core, the controller, the host pool and the device
// allocator are SGLang's, a call the prefix-cache interface records answers as SGLang's did, and
// afterwards the controller's queues, acks and host free list are SGLang's. The sequences reach
// what the scheduler does not: a load-back polled before its copy is done, host eviction, and
// write-back's blocking write check.
func TestTheHiCacheRadixCacheFollowsEveryDriverSequenceAsSGLangDoes(t *testing.T) {
	replayed := 0
	for name, tr := range loadL2Traces(t) {
		if tr.Kind != "direct" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			sq, err := loadSequence(tr)
			if err != nil {
				t.Fatal(err)
			}
			if !sq.drivesCache() {
				t.Log("the sequence drives the controller alone; the controller's replay covers it")
				return
			}
			n, err := replayL2Direct(tr, sq)
			var ref storageRefused
			if errors.As(err, &ref) {
				if !strings.Contains(ref.err.Error(), StorageRefusalIssue) {
					t.Fatalf("a storage sequence was refused without citing %s: %v", StorageRefusalIssue, ref.err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			replayed += n
			t.Logf("%d entries replayed", n)
		})
	}
	if replayed == 0 {
		t.Fatal("no driver sequence at the HiCache radix cache was replayed")
	}
}

type seqEntry struct {
	Boundary  bool   `json:"boundary"`
	On        string `json:"on"`
	M         string `json:"m"`
	As        string `json:"as"`
	Node      string `json:"node"`
	NumTokens int    `json:"num_tokens"`
	MemQuota  *int   `json:"mem_quota"`
	WriteBack bool   `json:"write_back"`
	Raises    string `json:"raises"`
}

type sequence struct {
	Calls []seqEntry `json:"calls"`
}

func (s sequence) drivesCache() bool {
	for _, c := range s.Calls {
		if c.On == "cache" {
			return true
		}
	}
	return false
}

// loadSequence reads the driver's sequence a direct trace was made from: the scenario its
// provenance lists among its inputs, in this repository's oracle.
func loadSequence(tr hiTrace) (sequence, error) {
	var sq sequence
	ins, _ := tr.Provenance["oracle_inputs"].(map[string]any)
	var paths []string
	for p := range ins {
		paths = append(paths, p)
	}
	slices.Sort(paths)
	for _, p := range paths {
		if !strings.HasPrefix(p, "oracle/scenarios/") {
			continue
		}
		raw, err := readRepoFile(p)
		if err != nil {
			return sq, err
		}
		return sq, json.Unmarshal(raw, &sq)
	}
	return sq, errors.New("the trace names no sequence among its inputs")
}

// madeByCache is whether c is a call the cache's own code made: at the top of the driver's
// entry (an unrecorded cache method such as load_back) or inside a recorded cache call.
func (r *l2Replay) madeByCache(c cache.Call) bool {
	p := r.parent(c)
	if c.Iface == cache.Interface(HostPoolIface) {
		return !r.under(c, cache.Interface(ControllerIface))
	}
	return p == nil || p.Iface == cache.PrefixCacheIface
}

func replayL2Direct(tr hiTrace, sq sequence) (int, error) {
	r := &l2Replay{tr: tr, nodes: cache.NewNodes(), uuids: map[int64]int64{}, bySeq: map[int]cache.Call{},
		pool: &l2Pool{rows: map[int]map[int]int64{}}, clock: &Clock{}}
	for _, c := range tr.Calls {
		r.bySeq[c.Seq] = c
	}
	if err := r.build(); err != nil {
		return 0, err
	}
	if len(tr.States) != len(sq.Calls)+1 {
		return 0, fmt.Errorf("%d states for %d entries", len(tr.States), len(sq.Calls))
	}
	bound := map[string]cache.MatchResult{}
	allocs := map[string][]int64{}
	n := 0
	for i, e := range sq.Calls {
		lo, hi := tr.States[i].After, tr.States[i+1].After
		span := tr.Calls[lo+1 : hi+1]
		if err := r.entry(e, span, bound, allocs); err != nil {
			return n, fmt.Errorf("entry %d (%s.%s): %w", i, e.On, e.M, err)
		}
		if err := r.sameState(tr.States[i+1].Controller); err != nil {
			return n, fmt.Errorf("after entry %d (%s.%s): %w", i, e.On, e.M, err)
		}
		n++
	}
	return n, nil
}

func (r *l2Replay) entry(e seqEntry, span []cache.Call, bound map[string]cache.MatchResult, allocs map[string][]int64) error {
	if e.Boundary {
		_, err := r.l2.queues.Boundary()
		return err
	}
	l := r.l2
	switch e.On {
	case "allocator":
		// The driver's own allocation: the slots SGLang's allocator gave.
		for _, c := range span {
			if c.In == nil && c.Iface == cache.AllocatorIface && c.Method == e.M && e.M == "alloc" && e.As != "" {
				v, err := decodeAs[[]int64](c.Ret)
				if err != nil {
					return err
				}
				allocs[e.As] = v
			}
		}
		return nil
	case "controller":
		switch e.M {
		case "has_inflight_device_transfers":
			l.queues.HasInflightDeviceTransfers()
		case "start_writing":
			l.queues.StartWriting()
		case "start_loading":
			if _, err := l.queues.StartLoading(); err != nil {
				return err
			}
		default:
			return fmt.Errorf("no replay for the driver's controller call %s", e.M)
		}
		return nil
	case "cache":
	default:
		return fmt.Errorf("no target %q", e.On)
	}
	// SGLang's answers to the calls the cache makes at the allocator.
	r.alloc.avail, r.alloc.allocs = nil, nil
	for _, x := range span {
		if x.Iface != cache.AllocatorIface {
			continue
		}
		switch {
		case x.Method == "available_size" && !r.under(x, cache.Interface(ControllerIface)):
			v, err := decodeAs[int](x.Ret)
			if err != nil {
				return err
			}
			r.alloc.avail = append(r.alloc.avail, v)
		case x.Method == "alloc":
			v, err := decodeAs[[]int64](x.Ret)
			if err != nil {
				return err
			}
			r.alloc.allocs = append(r.alloc.allocs, v)
		}
	}
	r.alloc.made, r.pool.made = nil, nil
	r.core.log, r.ctrl.log, r.host.log = nil, nil, nil
	var rec *cache.Call
	for k := range span {
		if span[k].In == nil && span[k].Iface == cache.PrefixCacheIface && span[k].Method == e.M {
			rec = &span[k]
			break
		}
	}
	if rec != nil {
		// A method the prefix-cache interface records: driven with SGLang's own arguments.
		if err := r.answer(*rec); err != nil {
			return err
		}
		if e.As != "" && e.M == "match_prefix" {
			var want cache.MatchResult
			if err := rec.Answer(&want); err != nil {
				return err
			}
			bound[e.As] = cache.MatchResult{BestMatchNode: r.port(want.BestMatchNode), LastDeviceNode: r.port(want.LastDeviceNode)}
		}
	} else {
		switch e.M {
		case "load_back":
			m, ok := bound[e.Node]
			if !ok {
				return fmt.Errorf("no match bound as %q", e.Node)
			}
			l.LoadBack(m.BestMatchNode, e.MemQuota)
		case "evict_host":
			l.EvictHost(e.NumTokens)
		case "writing_check":
			l.WritingCheck(e.WriteBack)
		case "loading_check":
			l.LoadingCheck()
		default:
			return fmt.Errorf("no replay for the driver's cache call %s", e.M)
		}
	}
	for _, chk := range []struct {
		iface cache.Interface
		got   []figure
	}{{cache.TreeCoreIface, r.core.log}, {cache.Interface(ControllerIface), r.ctrl.log}, {cache.Interface(HostPoolIface), r.host.log}} {
		var want []figure
		for _, x := range span {
			if x.Iface != chk.iface || !r.madeByCache(x) || (x.In == nil && rec != nil) {
				continue
			}
			f, skip, err := r.wantFigure(chk.iface, x)
			if err != nil {
				return err
			}
			if !skip {
				want = append(want, f)
			}
		}
		if err := r.same(chk.iface, chk.got, want); err != nil {
			return err
		}
	}
	var wantAlloc []l2Made
	for _, x := range span {
		if x.Iface != cache.AllocatorIface || r.under(x, cache.Interface(ControllerIface)) {
			continue
		}
		switch x.Method {
		case "available_size":
			wantAlloc = append(wantAlloc, l2Made{"available_size", ""})
		case "free_segment":
			var s []int64
			var start int
			if err := x.Arg("free_index", &s); err != nil {
				return err
			}
			if err := x.Arg("start_pos", &start); err != nil {
				return err
			}
			wantAlloc = append(wantAlloc, l2Made{"free_segment", fmt.Sprint(s, start)})
		}
	}
	var gotAlloc []l2Made
	for _, m := range r.alloc.made {
		if m.m == "available_size" || m.m == "free_segment" {
			gotAlloc = append(gotAlloc, m)
		}
	}
	if !reflect.DeepEqual(nilIfEmpty(gotAlloc), nilIfEmpty(wantAlloc)) {
		return fmt.Errorf("the port called the allocator %v; SGLang's cache called it %v", gotAlloc, wantAlloc)
	}
	return nil
}

// sameState compares the controller's queues, acks and host free list with SGLang's.
func (r *l2Replay) sameState(want ctrlState) error {
	q := r.l2.queues
	if got := r.l2.host.AvailableSize(); got != want.HostFree {
		return fmt.Errorf("host free %d, SGLang's %d", got, want.HostFree)
	}
	r.host.log = nil
	if len(q.WriteQueue) != len(want.WriteQueue) || len(q.LoadQueue) != len(want.LoadQueue) ||
		len(q.AckWriteQueue) != len(want.AckWriteQueue) || len(q.AckLoadQueue) != len(want.AckLoadQueue) {
		return fmt.Errorf("queues %d/%d, acks %d/%d; SGLang's %d/%d, %d/%d", len(q.WriteQueue), len(q.LoadQueue),
			len(q.AckWriteQueue), len(q.AckLoadQueue), len(want.WriteQueue), len(want.LoadQueue), len(want.AckWriteQueue),
			len(want.AckLoadQueue))
	}
	for i, op := range q.WriteQueue {
		if !slices.Equal(op.Host, want.WriteQueue[i].Host) || !slices.Equal(op.Device, want.WriteQueue[i].Device) {
			return fmt.Errorf("write %d copies %v to %v; SGLang's %v to %v", i, op.Device, op.Host,
				want.WriteQueue[i].Device, want.WriteQueue[i].Host)
		}
	}
	for i, a := range q.AckWriteQueue {
		if a.Finish.Query() != want.AckWriteQueue[i].Done {
			return fmt.Errorf("write ack %d done %v; SGLang's %v", i, a.Finish.Query(), want.AckWriteQueue[i].Done)
		}
	}
	for i, a := range q.AckLoadQueue {
		if a.Finish.Query() != want.AckLoadQueue[i].Done {
			return fmt.Errorf("load ack %d done %v; SGLang's %v", i, a.Finish.Query(), want.AckLoadQueue[i].Done)
		}
	}
	return nil
}

// readRepoFile reads a file of this repository by its path from the root.
func readRepoFile(p string) ([]byte, error) {
	return os.ReadFile(filepath.Join("..", "..", filepath.FromSlash(p)))
}

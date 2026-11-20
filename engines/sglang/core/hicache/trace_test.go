// Original to blis-sglang.

package hicache

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	blissglang "github.com/inference-sim/inference-sim/engines/sglang"
	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
	"github.com/inference-sim/inference-sim/engines/sglang/internal/testgz"
)

// goldenDir is where the HiCache framework's traces are committed: recorded/ from SGLang's
// scheduler running oracle/scenarios/hicache/recorded, direct/ from the driver's sequences in
// oracle/scenarios/hicache/direct (oracle/run_hicache.py).
const goldenDir = "testdata/golden-fid5"

// hiTrace is a trace oracle/run_hicache.py writes: the calls, where each step boundary fell, the
// thread bodies the runner ran, and (direct) the controller's state after each sequence entry.
type hiTrace struct {
	Provenance map[string]any `json:"provenance"`
	Kind       string         `json:"kind"`
	Calls      []cache.Call   `json:"calls"`
	BoundaryAt []int          `json:"boundary_at"`
	BodiesRun  [][]any        `json:"bodies_run"`
	States     []traceState   `json:"states"`
}

type traceState struct {
	After      int       `json:"after"`
	Boundaries int       `json:"boundaries"`
	Controller ctrlState `json:"controller"`
}

type opState struct {
	NodeIDs  []int64 `json:"node_ids"`
	Host     []int64 `json:"host_indices"`
	Device   []int64 `json:"device_indices"`
	Priority int     `json:"priority"`
}

type ackState struct {
	NodeIDs   []int64 `json:"node_ids"`
	NumTokens int     `json:"num_tokens"`
	Done      bool    `json:"done"`
}

type ctrlState struct {
	WriteQueue    []opState  `json:"write_queue"`
	LoadQueue     []opState  `json:"load_queue"`
	AckWriteQueue []ackState `json:"ack_write_queue"`
	AckLoadQueue  []ackState `json:"ack_load_queue"`
	LayerCounter  struct {
		ProducerIndex int    `json:"producer_index"`
		ConsumerIndex int    `json:"consumer_index"`
		Finished      []bool `json:"finished"`
	} `json:"layer_counter"`
	HostFree int            `json:"host_free"`
	Storage  map[string]int `json:"storage"`
	// AckBackupCompletedTokens is each acked backup's completed tokens, front first (with storage).
	AckBackupCompletedTokens []int `json:"ack_backup_completed_tokens"`
}

func loadTraces(t *testing.T) map[string]hiTrace {
	t.Helper()
	out := map[string]hiTrace{}
	for _, kind := range []string{"recorded", "direct"} {
		paths, err := testgz.Glob(filepath.Join(goldenDir, kind, "*.trace.json"))
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range paths {
			raw, err := testgz.ReadFile(p)
			if err != nil {
				t.Fatal(err)
			}
			var tr hiTrace
			if err := json.Unmarshal(raw, &tr); err != nil {
				t.Fatalf("%s: %v", p, err)
			}
			if tr.Kind != kind {
				t.Fatalf("%s: a %q trace in the %s directory", p, tr.Kind, kind)
			}
			out[kind+"/"+strings.TrimSuffix(filepath.Base(p), ".trace.json")] = tr
		}
	}
	if len(out) == 0 {
		t.Fatalf("no trace under %s", goldenDir)
	}
	return out
}

// The Go type that ports each HiCache interface.
var portOf = map[Interface]reflect.Type{
	ControllerIface:   reflect.TypeFor[ControllerPort](),
	LayerCounterIface: reflect.TypeFor[LayerCounterPort](),
	L2TransferIface:   reflect.TypeFor[L2TransferPort](),
	HostPoolIface:     reflect.TypeFor[HostPoolPort](),
	StorageIface:      reflect.TypeFor[Storage](),
}

var (
	_ ControllerPort   = (*Controller)(nil)
	_ LayerCounterPort = (*LayerDoneCounter)(nil)
	_ L2TransferPort   = (*L2Engine)(nil)
	_ HostPoolPort     = (*HostPool)(nil)
)

// FID-5: the HiCache framework is traced at each of its own interfaces by both kinds of trace,
// SGLang's scheduler running a scenario and the driver's own sequences.
func TestEachHiCacheInterfaceHasRecordedAndDirectTraces(t *testing.T) {
	have := map[string]bool{}
	for _, tr := range loadTraces(t) {
		for _, c := range tr.Calls {
			have[tr.Kind+" "+string(c.Iface)] = true
		}
	}
	for iface := range portOf {
		for _, kind := range []string{"recorded", "direct"} {
			if !have[kind+" "+string(iface)] {
				t.Errorf("no %s trace calls %s", kind, iface)
			}
		}
	}
}

// Every trace is SGLang's own output at the release this build models, with a manifest version,
// and its boundaries fall in call order.
func TestHiCacheTracesAreFromThePinnedSGLang(t *testing.T) {
	for name, tr := range loadTraces(t) {
		if got := tr.Provenance["sglang_commit"]; got != blissglang.Upstream.Commit {
			t.Errorf("%s: recorded at SGLang %v, and the build models %s", name, got, blissglang.Upstream.Commit)
		}
		if v, _ := tr.Provenance["mock_version"].(string); !strings.HasPrefix(v, "gpu-mock/") {
			t.Errorf("%s: no manifest version (%v)", name, tr.Provenance["mock_version"])
		}
		if !slices.IsSorted(tr.BoundaryAt) {
			t.Errorf("%s: boundaries out of call order: %v", name, tr.BoundaryAt)
		}
	}
}

// Every method SGLang's HiCache classes were called on is a method of that interface's Go port:
// a call the port has no method for fails here rather than pass unnoticed.
func TestEveryTracedHiCacheCallIsAMethodOfThePort(t *testing.T) {
	for name, tr := range loadTraces(t) {
		for _, c := range tr.Calls {
			port, ok := portOf[Interface(c.Iface)]
			if !ok || c.Method == "__init__" {
				continue
			}
			if _, ok := port.MethodByName(cache.GoName(c.Method)); !ok {
				t.Errorf("%s: SGLang's %s.%s has no Go method %s.%s", name, c.Iface, c.Method, port.Name(), cache.GoName(c.Method))
			}
		}
	}
}

// The step-driven runner's order is the port's: at every boundary of every trace, the thread
// bodies SGLang's runner ran are those the port's Boundary runs, in the same order.
func TestBoundaryRunsTheThreadBodiesInSGLangsStartOrder(t *testing.T) {
	clock := &Clock{}
	host, err := NewHostPool(HostSizing{DeviceCapacityTokens: 8, HostToDeviceRatio: 2, PageSize: 1, Layout: "layer_first"})
	if err != nil {
		t.Fatal(err)
	}
	c, err := NewController(ControllerConfig{WritePolicy: "write_through", IOBackend: "direct", Layout: "layer_first", PageSize: 1,
		TransferLayerIDMax: 1}, clock, host, host.Size(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.AttachStorage("file", &replayStorage{}, StorageConfig{}, 256); err != nil {
		t.Fatal(err)
	}
	port, err := c.Boundary()
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for name, tr := range loadTraces(t) {
		byBoundary := map[float64][]string{}
		for _, r := range tr.BodiesRun {
			k, _ := r[0].(float64)
			body, _ := r[1].(string)
			byBoundary[k] = append(byBoundary[k], body)
		}
		for k, bodies := range byBoundary {
			if !slices.Equal(bodies, port) {
				t.Errorf("%s: at boundary %v SGLang ran %v; the port runs %v", name, k, bodies, port)
			}
			checked++
		}
	}
	if checked == 0 {
		t.Fatal("no trace ran a thread body")
	}
}

// stopsAt is each trace whose replay stops short of its end, at the call it stops at: a call
// the port refuses until HiCache L3 storage (refused). Every other trace replays to its end.
var stopsAt = map[string]string{
	"direct/prefetch-from-storage": "call 103, controller.submit_prefetch",
}

// The port of HiCacheController, LayerDoneCounter and HostKVCache answers every call SGLang's
// answered, in every trace of the framework: the radix cache's and the driver's calls at the
// controller and the host pool, and, at each step boundary, the thread bodies the port's
// Boundary runs. The calls the port makes in turn at the host pool, the transfer engine, the
// device allocator and storage are SGLang's, in order; the allocator and storage are stand-ins
// that answer as the trace says SGLang's did. After each entry of a direct sequence, the
// controller's queues, acks, layer counter and host free list are SGLang's. A trace replays to
// its end, or to the refused call stopsAt names for it.
func TestTheControllerAnswersEveryTracedCallAsSGLangDoes(t *testing.T) {
	replayed := 0
	traces := loadTraces(t)
	for name := range stopsAt {
		if _, ok := traces[name]; !ok {
			t.Errorf("stopsAt names %s, which is not a trace", name)
		}
	}
	for name, tr := range traces {
		t.Run(name, func(t *testing.T) {
			r, err := newReplay(tr)
			if err != nil {
				t.Fatal(err)
			}
			n, stop, err := r.run()
			if err != nil {
				t.Fatal(err)
			}
			replayed += n
			want, mayStop := stopsAt[name]
			switch {
			case stop == "" && mayStop:
				t.Errorf("%d calls replayed to the end; the replay should stop at %s", n, want)
			case stop != "" && !mayStop:
				t.Errorf("%d calls replayed; stopped at %s, and this trace should replay to its end", n, stop)
			case stop != "" && !strings.HasPrefix(stop, want+":"):
				t.Errorf("%d calls replayed; stopped at %s, not at %s", n, stop, want)
			default:
				t.Logf("%d calls replayed; boundaries %d of %d", n, r.boundaries, len(tr.BoundaryAt))
			}
		})
	}
	if replayed == 0 {
		t.Fatal("no call at the controller was replayed")
	}
}

// A replay checks the port's own step boundary: a Boundary that runs a thread body twice, or
// skips one, fails the replay of a trace whose bodies ran, which the correct Boundary passes.
func TestTheReplayFailsABoundaryThatRunsABodyTwiceOrSkipsOne(t *testing.T) {
	tr, ok := loadTraces(t)["direct/backup-to-storage"]
	if !ok {
		t.Fatal("no direct/backup-to-storage trace")
	}
	mutants := map[string]func(c *Controller) func() ([]string, error){
		"the port's Boundary": func(c *Controller) func() ([]string, error) { return c.Boundary },
		"backup_thread_func twice": func(c *Controller) func() ([]string, error) {
			return func() ([]string, error) {
				ran, err := c.Boundary()
				if c.EnableStorage {
					c.BackupThreadFunc()
					ran = append(ran, "backup_thread_func")
				}
				return ran, err
			}
		},
		"backup_thread_func skipped": func(c *Controller) func() ([]string, error) {
			return func() ([]string, error) {
				c.clock.boundaries++
				return []string{"prefetch_thread_func", "prefetch_io_aux_func", "prefetch_sync_thread_func"}, nil
			}
		},
		"backup_thread_func skipped, named as run": func(c *Controller) func() ([]string, error) {
			return func() ([]string, error) {
				c.clock.boundaries++
				return slices.Clone(threadBodies), nil
			}
		},
	}
	for name, mutant := range mutants {
		r, err := newReplay(tr)
		if err != nil {
			t.Fatal(err)
		}
		r.boundary = mutant(r.ctrl)
		_, stop, err := r.run()
		if name == "the port's Boundary" {
			if err != nil || stop != "" {
				t.Errorf("%s: the replay failed (%v) or stopped (%q)", name, err, stop)
			}
			continue
		}
		if err == nil {
			t.Errorf("a Boundary with %s passed the replay", name)
		} else {
			t.Logf("%s: %v", name, err)
		}
	}
}

// --- replay ---

type replayLog struct{ calls []string }

func (l *replayLog) add(iface Interface, m string, args string, ret any) {
	r, _ := json.Marshal(ret)
	l.calls = append(l.calls, fmt.Sprintf("%s.%s(%s)=%s", iface, m, args, r))
}

func js(v ...any) string {
	parts := make([]string, len(v))
	for i, x := range v {
		b, _ := json.Marshal(x)
		parts[i] = string(b)
	}
	return strings.Join(parts, ",")
}

// recHost is the host pool port, recording each call made on it.
type recHost struct {
	*HostPool
	log *replayLog
}

func (h *recHost) Alloc(n int) ([]int64, bool) {
	out, ok := h.HostPool.Alloc(n)
	var ret any
	if ok {
		ret = out
	}
	h.log.add(HostPoolIface, "alloc", js(n), ret)
	return out, ok
}

func (h *recHost) Free(idx []int64) (int, error) {
	n, err := h.HostPool.Free(idx)
	var ret any = n
	if err != nil {
		ret = "raise"
	}
	h.log.add(HostPoolIface, "free", js(idx), ret)
	return n, err
}

func (h *recHost) AvailableSize() int {
	n := h.HostPool.AvailableSize()
	h.log.add(HostPoolIface, "available_size", "", n)
	return n
}

func (h *recHost) PrepareTransferIndices(host, device []int64, io string) ([]int64, []int64) {
	a, b := h.HostPool.PrepareTransferIndices(host, device, io)
	h.log.add(HostPoolIface, "prepare_transfer_indices", js(host, device), [][]int64{a, b})
	return a, b
}

func (h *recHost) BackupFromDeviceAllLayerPhysical(host, device []int64, io string) {
	h.log.add(HostPoolIface, "backup_from_device_all_layer_physical", js(host, device), nil)
}

func (h *recHost) LoadToDevicePerLayerPhysical(host, device []int64, layer int, io string) {
	h.log.add(HostPoolIface, "load_to_device_per_layer_physical", js(host, device, layer), nil)
}

func (h *recHost) GetDataPage(index int64, flat bool) int64 {
	h.log.add(HostPoolIface, "get_data_page", js(index), nil)
	return h.HostPool.GetDataPage(index, flat)
}

// recL2 is the transfer engine, recording each submission before its copies.
type recL2 struct {
	eng *L2Engine
	log *replayLog
}

// transferArgs is a submission's transfers as the replay log writes them: [host, device] each.
func transferArgs(ts []Transfer) [][2][]int64 {
	out := make([][2][]int64, 0, len(ts))
	for _, t := range ts {
		out = append(out, [2][]int64{t.Host, t.Device})
	}
	return out
}

func (l *recL2) SubmitDeviceToHost(ts []Transfer) *Event {
	l.log.add(L2TransferIface, "submit_device_to_host", js(transferArgs(ts)), nil)
	return l.eng.SubmitDeviceToHost(ts)
}

func (l *recL2) SubmitHostToDevice(ts []Transfer, layers int, done func(int)) *Event {
	l.log.add(L2TransferIface, "submit_host_to_device", js(transferArgs(ts), layers), nil)
	return l.eng.SubmitHostToDevice(ts, layers, done)
}

// replayAlloc answers the device allocator's calls as the trace says SGLang's did.
type replayAlloc struct {
	answers [][]int64 // the next alloc answers, nil for a refused alloc
	log     *replayLog
}

func (a *replayAlloc) Alloc(n int) ([]int64, bool) {
	var out []int64
	if len(a.answers) > 0 {
		out, a.answers = a.answers[0], a.answers[1:]
	}
	var ret any
	if out != nil {
		ret = out
	}
	a.log.add(Interface(cache.AllocatorIface), "alloc", js(n), ret)
	return out, out != nil
}

func (a *replayAlloc) SetHicacheTransferDoneEvent(kind string, e *Event) {
	a.log.add(Interface(cache.AllocatorIface), "set_hicache_transfer_done_event", js(kind), nil)
}

// replayStorage answers storage calls as the trace says SGLang's backend did.
type replayStorage struct {
	setAnswers []bool
	log        *replayLog
}

func (s *replayStorage) add(m, args string, ret any) {
	if s.log != nil {
		s.log.add(StorageIface, m, args, ret)
	}
}
func (s *replayStorage) RegisterMemPoolHost(HostPoolPort) { s.add("register_mem_pool_host", "", nil) }
func (s *replayStorage) RegisterMemHostPoolV2(_ HostPoolPort, name string) {
	s.add("register_mem_host_pool_v2", js(name), nil)
}
func (s *replayStorage) OnHostPoolsRegistered() { s.add("on_host_pools_registered", "", nil) }
func (s *replayStorage) BatchSet(keys []string, pages []int64) bool {
	ok := false
	if len(s.setAnswers) > 0 {
		ok, s.setAnswers = s.setAnswers[0], s.setAnswers[1:]
	}
	s.add("batch_set", js(keys), ok)
	return ok
}
func (s *replayStorage) BatchSetV1(keys []string, host []int64, _ *StorageExtraInfo) []bool {
	panic("the file backend is not zero-copy")
}
func (s *replayStorage) BatchExistsV2([]string, *StorageExtraInfo) int          { panic("not traced") }
func (s *replayStorage) BatchGetV2([]string, []int64, *StorageExtraInfo) []bool { panic("not traced") }
func (s *replayStorage) BatchSetV2([]string, []int64, *StorageExtraInfo) []bool { panic("not traced") }
func (s *replayStorage) BatchGetV1([]string, []int64, *StorageExtraInfo) []bool { panic("not traced") }
func (s *replayStorage) Get(string, int64) bool                                 { panic("not traced") }
func (s *replayStorage) BatchGet([]string, []int64) []bool                      { panic("not traced") }
func (s *replayStorage) Set(string, int64) bool                                 { panic("not traced") }
func (s *replayStorage) Exists(string) bool                                     { panic("not traced") }
func (s *replayStorage) BatchExists([]string, *StorageExtraInfo) int            { panic("not traced") }
func (s *replayStorage) Clear()                                                 {}
func (s *replayStorage) GetStats() map[string]float64                           { return nil }

type replay struct {
	tr      hiTrace
	bySeq   map[int]cache.Call
	ctrl    *Controller
	host    *recHost
	alloc   *replayAlloc
	storage *replayStorage
	log     *replayLog
	// boundaries is how many of the trace's boundaries the replay has applied.
	boundaries int
	// bodies is each boundary's thread body calls (boundaryBodies); bodySeq marks them, since the
	// port's Boundary makes them, not the replay.
	bodies  [][]cache.Call
	bodySeq map[int]bool
	// boundary is the port's step boundary: the controller's Boundary.
	boundary func() ([]string, error)
}

// madeBy is whether call c was made, at any depth, by a call at one of ifaces.
func (r *replay) madeBy(c cache.Call, ifaces ...Interface) bool {
	for x := c; x.In != nil; {
		p := r.bySeq[*x.In]
		if slices.Contains(ifaces, Interface(p.Iface)) {
			return true
		}
		x = p
	}
	return false
}

// descendants is the calls c made at the interfaces the replay compares, at any depth, in order.
func (r *replay) descendants(c cache.Call) []cache.Call {
	var out []cache.Call
	for _, d := range r.tr.Calls {
		if d.Seq <= c.Seq || !slices.Contains([]Interface{HostPoolIface, L2TransferIface, StorageIface, LayerCounterIface,
			Interface(cache.AllocatorIface)}, Interface(d.Iface)) {
			continue
		}
		for x := d; x.In != nil; x = r.bySeq[*x.In] {
			if *x.In == c.Seq {
				out = append(out, d)
				break
			}
		}
	}
	return out
}

func generic(raw json.RawMessage) any {
	var v any
	_ = json.Unmarshal(raw, &v)
	return v
}

func (r *replay) arg(c cache.Call, name string) any { return generic(c.Args[name]) }

// expect is how a traced call made by SGLang's controller reads, in the replay log's form. It
// is false, with no error, for a layer counter call, which the replay reads from the counter
// itself; a call at another compared interface that it cannot write is an error, so no call the
// port should have made is dropped unseen.
func (r *replay) expect(c cache.Call) (string, bool, error) {
	var args string
	ret := generic(c.Ret)
	if c.Raise != "" {
		ret = "raise"
	}
	cannot := fmt.Errorf("the replay cannot compare %s.%s (call %d)", c.Iface, c.Method, c.Seq)
	switch Interface(c.Iface) {
	case HostPoolIface:
		switch c.Method {
		case "alloc":
			args = js(r.arg(c, "need_size"))
		case "free":
			args = js(r.arg(c, "indices"))
		case "available_size":
		case "prepare_transfer_indices", "backup_from_device_all_layer_physical":
			args = js(r.arg(c, "host_indices"), r.arg(c, "device_indices"))
		case "load_to_device_per_layer_physical":
			args = js(r.arg(c, "host_indices"), r.arg(c, "device_indices"), r.arg(c, "layer_id"))
		case "get_data_page":
			args, ret = js(r.arg(c, "index")), nil
		default:
			return "", false, cannot
		}
		if c.Method == "backup_from_device_all_layer_physical" || c.Method == "load_to_device_per_layer_physical" {
			ret = nil
		}
	case L2TransferIface:
		// Each transfer's host and device slots, in order; a submission may hold none
		// (hybrid_cache_controller.py:872-883@297e54fd).
		ts, ok := r.arg(c, "transfers").([]any)
		if !ok {
			return "", false, cannot
		}
		pairs := make([][2]any, 0, len(ts))
		for _, x := range ts {
			t, _ := x.(map[string]any)
			pairs = append(pairs, [2]any{t["host_indices"], t["device_indices"]})
		}
		switch c.Method {
		case "submit_device_to_host":
			args = js(pairs)
		case "submit_host_to_device":
			args = js(pairs, r.arg(c, "transfer_layer_id_max"))
		default:
			return "", false, cannot
		}
		ret = nil
	case Interface(cache.AllocatorIface):
		switch c.Method {
		case "alloc":
			args = js(r.arg(c, "need_size"))
		case "set_hicache_transfer_done_event":
			key, _ := r.arg(c, "transfer_key").([]any)
			if len(key) != 2 {
				return "", false, cannot
			}
			args, ret = js(key[1]), nil
		default:
			return "", false, cannot
		}
	case StorageIface:
		switch c.Method {
		case "batch_set":
			args = js(r.arg(c, "keys"))
		case "register_mem_pool_host", "on_host_pools_registered":
			ret = nil
		case "register_mem_host_pool_v2":
			args, ret = js(r.arg(c, "host_pool_name")), nil
		default:
			return "", false, cannot
		}
	case LayerCounterIface:
		// The layer counter's calls inside the controller are read from the counter itself.
		return "", false, nil
	default:
		return "", false, cannot
	}
	var parsed any
	b, _ := json.Marshal(ret)
	_ = json.Unmarshal(b, &parsed)
	rb, _ := json.Marshal(parsed)
	return fmt.Sprintf("%s.%s(%s)=%s", c.Iface, c.Method, args, rb), true, nil
}

// expected is the calls SGLang's controller made at the compared interfaces inside calls, in
// order. It loads the device allocator's and storage's answers there into their stand-ins.
func (r *replay) expected(calls ...cache.Call) ([]string, error) {
	var want []string
	r.alloc.answers = nil
	if r.storage != nil {
		r.storage.setAnswers = nil
	}
	for _, c := range calls {
		for _, d := range r.descendants(c) {
			if d.Iface == cache.AllocatorIface && d.Method == "alloc" {
				var out []int64
				if d.Raise == "" && string(d.Ret) != "null" {
					_ = json.Unmarshal(d.Ret, &out)
				}
				r.alloc.answers = append(r.alloc.answers, out)
			}
			if d.Iface == cache.Interface(StorageIface) && d.Method == "batch_set" {
				if r.storage == nil {
					return nil, fmt.Errorf("call %d: storage called with no storage attached", d.Seq)
				}
				var ok bool
				_ = json.Unmarshal(d.Ret, &ok)
				r.storage.setAnswers = append(r.storage.setAnswers, ok)
			}
			s, ok, err := r.expect(d)
			if err != nil {
				return nil, err
			}
			if ok {
				want = append(want, s)
			}
		}
	}
	return want, nil
}

func newReplay(tr hiTrace) (*replay, error) {
	r := &replay{tr: tr, bySeq: map[int]cache.Call{}, log: &replayLog{}}
	for _, c := range tr.Calls {
		r.bySeq[c.Seq] = c
	}
	var hostInit, ctrlInit, kvInit *cache.Call
	var storageInits []cache.Call
	for i := range tr.Calls {
		c := tr.Calls[i]
		switch {
		case c.Iface == cache.Interface(HostPoolIface) && c.Method == "__init__":
			hostInit = &tr.Calls[i]
		case c.Iface == cache.Interface(ControllerIface) && c.Method == "__init__":
			ctrlInit = &tr.Calls[i]
		case c.Iface == cache.KVPoolIface && c.Method == "__init__" && kvInit == nil:
			kvInit = &tr.Calls[i]
		case c.Iface == cache.Interface(StorageIface) && c.Method == "__init__":
			storageInits = append(storageInits, c)
		}
	}
	if hostInit == nil || ctrlInit == nil || kvInit == nil {
		return nil, fmt.Errorf("the trace builds no host pool, controller or KV pool")
	}
	var ha struct {
		DeviceCapacityTokens int     `json:"device_capacity_tokens"`
		Ratio                float64 `json:"host_to_device_ratio"`
		HostSize             float64 `json:"host_size"`
		PageSize             int     `json:"page_size"`
		Layout               string  `json:"layout"`
		HostTokenCapacity    *int    `json:"host_token_capacity"`
	}
	b, _ := json.Marshal(hostInit.Args)
	if err := json.Unmarshal(b, &ha); err != nil {
		return nil, err
	}
	var hs struct {
		Size         int   `json:"size"`
		PageNum      int   `json:"page_num"`
		SizePerToken int64 `json:"size_per_token"`
	}
	if err := hostInit.Answer(&hs); err != nil {
		return nil, err
	}
	host, err := NewHostPool(HostSizing{DeviceCapacityTokens: ha.DeviceCapacityTokens, SizePerToken: hs.SizePerToken,
		HostToDeviceRatio: ha.Ratio, HostSizeGB: ha.HostSize, PageSize: ha.PageSize, Layout: ha.Layout,
		HostTokenCapacity: ha.HostTokenCapacity})
	if err != nil {
		return nil, err
	}
	if host.Size() != hs.Size || host.PageNum() != hs.PageNum {
		return nil, fmt.Errorf("host pool: the port sizes %d slots in %d pages; SGLang's has %d in %d",
			host.Size(), host.PageNum(), hs.Size, hs.PageNum)
	}
	var ca struct {
		WritePolicy        string `json:"write_policy"`
		IOBackend          string `json:"io_backend"`
		PageSize           int    `json:"page_size"`
		StorageBackend     string `json:"storage_backend"`
		PrefetchThreshold  int    `json:"prefetch_threshold"`
		TransferLayerIDMax *int   `json:"transfer_layer_id_max"`
	}
	b, _ = json.Marshal(ctrlInit.Args)
	if err := json.Unmarshal(b, &ca); err != nil {
		return nil, err
	}
	var kv struct {
		LayerNum int `json:"layer_num"`
	}
	if err := kvInit.Answer(&kv); err != nil {
		return nil, err
	}
	layers := kv.LayerNum
	if ca.TransferLayerIDMax != nil {
		layers = *ca.TransferLayerIDMax
	}
	r.host = &recHost{HostPool: host, log: r.log}
	r.alloc = &replayAlloc{log: r.log}
	clock := &Clock{}
	l2 := &recL2{eng: &L2Engine{Host: r.host, IOBackend: ca.IOBackend, Clock: clock}, log: r.log}
	r.ctrl, err = NewController(ControllerConfig{WritePolicy: ca.WritePolicy, IOBackend: ca.IOBackend, Layout: ha.Layout,
		PageSize: ca.PageSize, TransferLayerIDMax: layers}, clock, r.host, host.Size(), r.alloc, l2)
	if err != nil {
		return nil, err
	}
	r.boundary = r.ctrl.Boundary
	if r.bodies, err = boundaryBodies(tr); err != nil {
		return nil, err
	}
	r.bodySeq = map[int]bool{}
	for _, bs := range r.bodies {
		for _, b := range bs {
			r.bodySeq[b.Seq] = true
		}
	}
	if len(storageInits) > 0 {
		var sa struct {
			Config struct {
				TPRank  int  `json:"tp_rank"`
				TPSize  int  `json:"tp_size"`
				MLA     bool `json:"is_mla_model"`
				DCPRank int  `json:"dcp_rank"`
			} `json:"storage_config"`
		}
		b, _ := json.Marshal(storageInits[0].Args)
		if err := json.Unmarshal(b, &sa); err != nil {
			return nil, err
		}
		r.storage = &replayStorage{log: r.log}
		if err := r.ctrl.AttachStorage(ca.StorageBackend, r.storage,
			StorageConfig{TPRank: sa.Config.TPRank, TPSize: sa.Config.TPSize, IsMLAModel: sa.Config.MLA, DCPRank: sa.Config.DCPRank},
			ca.PrefetchThreshold); err != nil {
			return nil, err
		}
		// The registration the attach made is compared with SGLang's, inside its __init__.
		var want []string
		for _, d := range r.descendants(*ctrlInit) {
			if d.Iface == cache.Interface(StorageIface) && d.Method != "__init__" {
				s, _, err := r.expect(d)
				if err != nil {
					return nil, err
				}
				want = append(want, s)
			}
		}
		if !slices.Equal(r.log.calls, want) {
			return nil, fmt.Errorf("attach: the port called %v; SGLang called %v", r.log.calls, want)
		}
		r.log.calls = nil
	}
	return r, nil
}

// driven is whether c is a call the replay makes itself: one at the controller (every recorded
// controller call comes from outside it) that is not a thread body the runner ran at a boundary,
// or one at the host pool or the layer counter that the controller and the transfer engine did
// not make.
func (r *replay) driven(c cache.Call) bool {
	switch Interface(c.Iface) {
	case ControllerIface:
		return c.Method != "__init__" && !r.bodySeq[c.Seq]
	case HostPoolIface, LayerCounterIface:
		return c.Method != "__init__" && !r.madeBy(c, ControllerIface, L2TransferIface)
	}
	return false
}

// refused is each controller method the port refuses, with why.
var refused = map[string]string{
	"submit_prefetch":      "HiCache L3 storage prefetch (inference-sim/blis-sglang#40)",
	"allocate_storage_hit": "HiCache L3 storage prefetch (inference-sim/blis-sglang#40)",
}

// threadBodies are the controller's thread bodies, which only the runner calls.
var threadBodies = []string{"prefetch_thread_func", "prefetch_io_aux_func", "prefetch_sync_thread_func", "backup_thread_func"}

// boundaryBodies is, for each boundary of the trace, the thread body calls the runner made
// there: the top-level controller calls from where the boundary fell that are thread bodies.
func boundaryBodies(tr hiTrace) ([][]cache.Call, error) {
	out := make([][]cache.Call, len(tr.BoundaryAt))
	for k, at := range tr.BoundaryAt {
		for _, c := range tr.Calls {
			if c.Seq < at || c.In != nil {
				continue
			}
			if c.Iface != cache.Interface(ControllerIface) || !slices.Contains(threadBodies, c.Method) {
				break
			}
			out[k] = append(out[k], c)
		}
	}
	for _, c := range tr.Calls {
		if c.In == nil && c.Iface == cache.Interface(ControllerIface) && slices.Contains(threadBodies, c.Method) {
			found := false
			for _, bs := range out {
				found = found || slices.ContainsFunc(bs, func(b cache.Call) bool { return b.Seq == c.Seq })
			}
			if !found {
				return nil, fmt.Errorf("call %d: thread body %s ran outside a step boundary", c.Seq, c.Method)
			}
		}
	}
	return out, nil
}

// applyBoundary applies boundary k: the port's Boundary runs, and the bodies it runs and the
// calls they make at the host pool and storage must be the bodies SGLang's runner ran there and
// the calls they made, in order.
func (r *replay) applyBoundary(k int) error {
	bodies := r.bodies[k]
	want, err := r.expected(bodies...)
	if err != nil {
		return err
	}
	var wantRan []string
	for _, b := range bodies {
		if b.Raise != "" {
			return fmt.Errorf("call %d: SGLang's %s raised %q", b.Seq, b.Method, b.Raise)
		}
		wantRan = append(wantRan, b.Method)
	}
	r.log.calls = nil
	ran, err := r.boundary()
	if err != nil {
		return err
	}
	if !slices.Equal(ran, wantRan) {
		return fmt.Errorf("the port ran %v; SGLang's runner ran %v", ran, wantRan)
	}
	if !slices.Equal(r.log.calls, want) {
		return fmt.Errorf("the port's bodies called\n  %s\nSGLang's called\n  %s", strings.Join(r.log.calls, "\n  "),
			strings.Join(want, "\n  "))
	}
	return nil
}

func (r *replay) run() (int, string, error) {
	n := 0
	states := r.tr.States
	si := 0
	// advance compares each state taken before call upTo and applies each boundary that fell
	// before it, in the order the trace took them.
	advance := func(upTo int) error {
		for {
			if si < len(states) && states[si].After < upTo && states[si].Boundaries <= r.boundaries {
				s := states[si]
				if s.Boundaries != r.boundaries {
					return fmt.Errorf("the state after call %d is at boundary %d; the replay is at %d", s.After, s.Boundaries, r.boundaries)
				}
				if err := r.compareState(s); err != nil {
					return fmt.Errorf("after call %d, boundary %d: %w", s.After, s.Boundaries, err)
				}
				si++
				continue
			}
			if r.boundaries < len(r.tr.BoundaryAt) && r.tr.BoundaryAt[r.boundaries] <= upTo {
				if err := r.applyBoundary(r.boundaries); err != nil {
					return fmt.Errorf("boundary %d: %w", r.boundaries+1, err)
				}
				r.boundaries++
				continue
			}
			return nil
		}
	}
	for _, c := range r.tr.Calls {
		if err := advance(c.Seq); err != nil {
			return n, "", err
		}
		if !r.driven(c) {
			continue
		}
		if why, ok := refused[c.Method]; ok {
			return n, fmt.Sprintf("call %d, %s.%s: %s", c.Seq, c.Iface, c.Method, why), nil
		}
		if err := r.call(c); err != nil {
			return n, "", fmt.Errorf("call %d %s.%s: %w", c.Seq, c.Iface, c.Method, err)
		}
		n++
	}
	if err := advance(int(^uint(0) >> 1)); err != nil {
		return n, "", err
	}
	if si != len(states) || r.boundaries != len(r.tr.BoundaryAt) {
		return n, "", fmt.Errorf("the replay compared %d of %d states and applied %d of %d boundaries", si, len(states),
			r.boundaries, len(r.tr.BoundaryAt))
	}
	return n, "", nil
}

func ints(v any) []int64 {
	xs, _ := v.([]any)
	if v == nil {
		return nil
	}
	out := make([]int64, len(xs))
	for i, x := range xs {
		f, _ := x.(float64)
		out[i] = int64(f)
	}
	return out
}

func strs(v any) []string {
	xs, _ := v.([]any)
	if v == nil {
		return nil
	}
	out := make([]string, len(xs))
	for i, x := range xs {
		out[i], _ = x.(string)
	}
	return out
}

func optInt(v any) *int {
	f, ok := v.(float64)
	if !ok {
		return nil
	}
	i := int(f)
	return &i
}

// call replays one driven call and compares the port's answer and the calls it made.
func (r *replay) call(c cache.Call) error {
	// The device allocator and storage answer as SGLang's did inside this call.
	want, err := r.expected(c)
	if err != nil {
		return err
	}
	r.log.calls = nil
	var got any
	raised := false
	ctrl := r.ctrl
	producerBefore := ctrl.counter.ProducerIndex
	switch Interface(c.Iface) {
	case ControllerIface:
		switch c.Method {
		case "write":
			if r.arg(c, "extra_pools") != nil {
				return fmt.Errorf("a write with side pools is not modelled (inference-sim/blis-sglang#74)")
			}
			if r.arg(c, "start_owner") != nil {
				return fmt.Errorf("a write with a page-interleaved shard owner is not modelled (inference-sim/blis-sglang#44)")
			}
			flush, _ := r.arg(c, "flush").(bool)
			node, _ := r.arg(c, "node_id").(float64)
			if h, ok := ctrl.Write(ints(r.arg(c, "device_indices")), optInt(r.arg(c, "priority")), int64(node), flush); ok {
				got = h
			}
		case "load":
			if r.arg(c, "extra_pools") != nil || r.arg(c, "device_indices") != nil || r.arg(c, "anchor_indices") != nil {
				return fmt.Errorf("a load into caller-owned slots or with side pools is not modelled (inference-sim/blis-sglang#74)")
			}
			node, _ := r.arg(c, "node_id").(float64)
			if d, ok := ctrl.Load(ints(r.arg(c, "host_indices")), optInt(r.arg(c, "priority")), int64(node)); ok {
				got = d
			}
		case "start_writing":
			ctrl.StartWriting()
		case "start_loading":
			p, err := ctrl.StartLoading()
			if err != nil {
				raised = true
			}
			got = p
		case "evict_host":
			backupOnly, _ := r.arg(c, "backup_only").(bool)
			k, err := ctrl.EvictHost(ints(r.arg(c, "host_indices")), backupOnly)
			raised = err != nil
			got = k
		case "has_inflight_device_transfers":
			got = ctrl.HasInflightDeviceTransfers()
		case "write_storage":
			if r.arg(c, "extra_pools") != nil {
				return fmt.Errorf("a storage backup with side pools is not modelled (inference-sim/blis-sglang#74)")
			}
			got = ctrl.WriteStorage(ints(r.arg(c, "host_indices")), ints(r.arg(c, "token_ids")), strs(r.arg(c, "hash_value")),
				strs(r.arg(c, "prefix_keys")))
		case "prefetch_rate_limited":
			got = ctrl.PrefetchRateLimited()
		case "get_prefetch_submission":
			if s := ctrl.GetPrefetchSubmission(""); s != nil {
				got = s
			}
		case "append_host_mem_release":
			if r.arg(c, "extra_pools") != nil && len(r.arg(c, "extra_pools").([]any)) > 0 {
				return fmt.Errorf("side-pool releases are not modelled (inference-sim/blis-sglang#74)")
			}
			ctrl.AppendHostMemRelease(ints(r.arg(c, "host_indices")))
		case "parse_storage_backend_extra_config":
			s, _ := r.arg(c, "storage_backend_extra_config").(string)
			cfg, err := ctrl.ParseStorageBackendExtraConfig(s)
			if err != nil {
				raised = true
			}
			got = []any{cfg.Extra, cfg.PrefetchThreshold, cfg.PrefetchTimeoutBase, cfg.PrefetchTimeoutPerKiToken, cfg.PassPrefixKeys}
		default:
			return fmt.Errorf("the replay has no call controller.%s", c.Method)
		}
	case HostPoolIface:
		switch c.Method {
		case "available_size":
			got = r.host.HostPool.AvailableSize()
		case "free":
			k, err := r.host.HostPool.Free(ints(r.arg(c, "indices")))
			raised = err != nil
			got = k
		case "alloc":
			n, _ := r.arg(c, "need_size").(float64)
			if h, ok := r.host.HostPool.Alloc(int(n)); ok {
				got = h
			}
		case "destroy":
			r.host.Destroy()
		case "clear":
			r.host.Clear()
		default:
			return fmt.Errorf("the replay has no call host_pool.%s", c.Method)
		}
	case LayerCounterIface:
		switch c.Method {
		case "set_consumer":
			i, _ := r.arg(c, "index").(float64)
			ctrl.counter.SetConsumer(int(i))
		case "wait_until":
			i, _ := r.arg(c, "threshold").(float64)
			ctrl.counter.WaitUntil(int(i))
		case "reset":
			ctrl.counter.Reset()
		default:
			return fmt.Errorf("the replay has no call layer_counter.%s", c.Method)
		}
	}
	if raised != (c.Raise != "") {
		return fmt.Errorf("the port raised %v; SGLang raised %q", raised, c.Raise)
	}
	if !raised {
		var want any
		_ = json.Unmarshal(c.Ret, &want)
		gb, _ := json.Marshal(got)
		var gotJSON any
		_ = json.Unmarshal(gb, &gotJSON)
		if !reflect.DeepEqual(gotJSON, want) {
			return fmt.Errorf("the port answered %s; SGLang answered %s", gb, c.Ret)
		}
	}
	// The layer counter's update_producer inside start_loading: the index the port moved to.
	for _, d := range r.descendants(c) {
		if d.Iface == cache.Interface(LayerCounterIface) && d.Method == "update_producer" {
			if ctrl.counter.ProducerIndex == producerBefore && d.Raise == "" {
				return fmt.Errorf("SGLang's controller advanced its layer counter; the port's did not")
			}
		}
	}
	if !slices.Equal(r.log.calls, want) {
		return fmt.Errorf("the port called\n  %s\nSGLang called\n  %s", strings.Join(r.log.calls, "\n  "),
			strings.Join(want, "\n  "))
	}
	return nil
}

// compareState compares the controller with SGLang's after a sequence entry. The radix cache
// reaps acks and drains storage queues from their fronts (S21, S22): what SGLang's queue no longer
// holds is reaped from the port's front, and the rest must be SGLang's.
func (r *replay) compareState(s traceState) error {
	c := r.ctrl
	ops := func(q []*CacheOperation) []opState {
		out := []opState{}
		for _, o := range q {
			out = append(out, opState{NodeIDs: o.NodeIDs, Host: o.Host, Device: o.Device, Priority: o.Priority})
		}
		return out
	}
	acks := func(q []*Ack) []ackState {
		out := []ackState{}
		for _, a := range q {
			out = append(out, ackState{NodeIDs: a.NodeIDs, NumTokens: a.NumTokens, Done: a.Finish.Query()})
		}
		return out
	}
	// The radix cache reaps an ack only once its finish event is done, front first
	// (unified_radix_cache.py:3404-3410@297e54fd): an ack the port reaps here must be done.
	reap := func(q []*Ack, want int) ([]*Ack, error) {
		if len(q) < want {
			return q, fmt.Errorf("the port holds %d acks; SGLang's %d", len(q), want)
		}
		for i, a := range q[:len(q)-want] {
			if !a.Finish.Query() {
				return q, fmt.Errorf("SGLang's radix cache reaped ack %d (nodes %v), which the port has not finished", i, a.NodeIDs)
			}
		}
		return q[len(q)-want:], nil
	}
	var err error
	if c.AckWriteQueue, err = reap(c.AckWriteQueue, len(s.Controller.AckWriteQueue)); err != nil {
		return fmt.Errorf("write acks: %w", err)
	}
	if c.AckLoadQueue, err = reap(c.AckLoadQueue, len(s.Controller.AckLoadQueue)); err != nil {
		return fmt.Errorf("load acks: %w", err)
	}
	eq := func(what string, got, want any) error {
		g, _ := json.Marshal(got)
		w, _ := json.Marshal(want)
		if string(g) != string(w) {
			return fmt.Errorf("%s: the port's %s; SGLang's %s", what, g, w)
		}
		return nil
	}
	nonNil := func(q []opState) []opState {
		if q == nil {
			return []opState{}
		}
		return q
	}
	nonNilA := func(q []ackState) []ackState {
		if q == nil {
			return []ackState{}
		}
		return q
	}
	finished := []bool{}
	for _, e := range c.counter.Events {
		finished = append(finished, e.Finish().Query())
	}
	for _, chk := range []error{
		eq("write queue", ops(c.WriteQueue), nonNil(s.Controller.WriteQueue)),
		eq("load queue", ops(c.LoadQueue), nonNil(s.Controller.LoadQueue)),
		eq("write acks", acks(c.AckWriteQueue), nonNilA(s.Controller.AckWriteQueue)),
		eq("load acks", acks(c.AckLoadQueue), nonNilA(s.Controller.AckLoadQueue)),
		eq("layer counter", []int{c.counter.ProducerIndex, c.counter.ConsumerIndex},
			[]int{s.Controller.LayerCounter.ProducerIndex, s.Controller.LayerCounter.ConsumerIndex}),
		eq("layer events finished", finished, s.Controller.LayerCounter.Finished),
		eq("host free slots", r.host.HostPool.AvailableSize(), s.Controller.HostFree),
	} {
		if chk != nil {
			return chk
		}
	}
	if st := s.Controller.Storage; st != nil {
		// The radix cache's share: what it reaped from the queues' fronts.
		if k := len(c.AckBackupQueue) - st["ack_backup_queue"]; k >= 0 {
			c.AckBackupQueue = c.AckBackupQueue[k:]
		}
		if k := len(c.HostMemReleaseQueue) - st["host_mem_release_queue"]; k >= 0 {
			c.HostMemReleaseQueue = c.HostMemReleaseQueue[k:]
		}
		completed := []int{}
		for _, op := range c.AckBackupQueue {
			completed = append(completed, op.CompletedTokens)
		}
		want := s.Controller.AckBackupCompletedTokens
		if want == nil {
			return fmt.Errorf("the trace records no acked backup's completed tokens")
		}
		if err := eq("acked backups' completed tokens", completed, want); err != nil {
			return err
		}
		got := map[string]int{"backup_queue": len(c.BackupQueue), "ack_backup_queue": len(c.AckBackupQueue),
			"host_mem_release_queue": len(c.HostMemReleaseQueue), "prefetch_queue": 0, "prefetch_buffer": 0,
			"prefetch_sync_queue": 0, "prefetch_hit_queue": 0, "ack_prefetch_queue": 0,
			"prefetch_tokens_occupied": c.PrefetchTokensOccupied}
		if err := eq("storage queues", got, st); err != nil {
			return err
		}
	}
	return nil
}

// SGLang's radix cache reaps only finished acks: a state whose queue no longer holds an ack the
// port has not finished fails the comparison, and one the port has finished is reaped.
func TestAnAckSGLangReapedMustBeDoneInThePort(t *testing.T) {
	for _, done := range []bool{false, true} {
		clock := &Clock{}
		host, err := NewHostPool(HostSizing{DeviceCapacityTokens: 8, HostToDeviceRatio: 1, PageSize: 1, Layout: "layer_first"})
		if err != nil {
			t.Fatal(err)
		}
		c, err := NewController(ControllerConfig{WritePolicy: "write_through", IOBackend: "direct", Layout: "layer_first",
			PageSize: 1, TransferLayerIDMax: 1}, clock, host, host.Size(), nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		finish := NewEvent(clock)
		finish.Record()
		if done {
			if _, err := c.Boundary(); err != nil {
				t.Fatal(err)
			}
		}
		c.AckWriteQueue = []*Ack{{Finish: finish, NodeIDs: []int64{1}, NumTokens: 1}}
		r := &replay{ctrl: c, host: &recHost{HostPool: host, log: &replayLog{}}}
		var s traceState
		s.Controller.LayerCounter.ProducerIndex, s.Controller.LayerCounter.ConsumerIndex = -1, -1
		s.Controller.LayerCounter.Finished = []bool{true, true, true}
		s.Controller.HostFree = host.AvailableSize()
		err = r.compareState(s)
		if done && err != nil {
			t.Errorf("a finished ack SGLang reaped: %v", err)
		}
		if !done && (err == nil || !strings.Contains(err.Error(), "not finished")) {
			t.Errorf("an unfinished ack SGLang reaped: got %v; want the reap refused", err)
		}
	}
}

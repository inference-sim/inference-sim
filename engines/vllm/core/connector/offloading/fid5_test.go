// Original to blis-vllm.

package offloading

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	blisvllm "github.com/inference-sim/blis-vllm"
)

// FID-5 (design §7.6): the offload framework's interfaces, OffloadingManager, OffloadingSpec and the
// CPU tier's CachePolicy, are checked at the interface itself. Each trace under
// testdata/golden-fid5 is vLLM's own calls and answers, either recorded while a real feature runs
// a scenario (recorded-*, oracle/run_kv_interfaces.py) or made by this repository's driver in
// orderings no feature makes (direct-*, oracle/drive_kv_interfaces.py, sequences in
// oracle/scenarios-fid5). The port receives the same calls in the same order, and every answer
// must be vLLM's.

type fid5Call struct {
	M    string                     `json:"m"`
	Args map[string]json.RawMessage `json:"args"`
	Ret  json.RawMessage            `json:"ret"`
}

type fid5Trace struct {
	Provenance map[string]any `json:"provenance"`
	Spec       []fid5Call     `json:"spec"`
	Manager    []fid5Call     `json:"manager"`
	Policy     []fid5Call     `json:"policy"`
}

// pendingFID5 is testdata/golden-fid5/golden-fid5.pending.json: the traces wait for a named issue, for the
// release the build models now only.
type pendingFID5 struct {
	Release  string `json:"release"`
	WaitsFor string `json:"waits_for"`
	Why      string `json:"why"`
}

func loadFID5(t *testing.T) map[string]fid5Trace {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join("testdata", "golden-fid5", "*.fid5.json"))
	if err != nil {
		t.Fatal(err)
	}
	if raw, err := os.ReadFile(filepath.Join("testdata", "golden-fid5.pending.json")); err == nil {
		var p pendingFID5
		if err := json.Unmarshal(raw, &p); err != nil {
			t.Fatalf("golden-fid5.pending.json: %v", err)
		}
		switch {
		case len(paths) > 0:
			t.Fatalf("testdata/golden-fid5 holds %d trace(s) and golden-fid5.pending.json: remove golden-fid5.pending.json", len(paths))
		case p.Release != blisvllm.Upstream.Tag:
			t.Fatalf("golden-fid5.pending.json defers the traces for %s, and the build models %s: record them (%s)",
				p.Release, blisvllm.Upstream.Tag, p.WaitsFor)
		case !strings.HasPrefix(p.WaitsFor, "https://github.com/inference-sim/blis-vllm/issues/"):
			t.Fatalf("golden-fid5.pending.json waits for %q, not a blis-vllm issue", p.WaitsFor)
		}
		// Reported, not passed: the FID-5 traces of this release are not committed (X16).
		t.Skipf("FID-5 offload interface traces pending for %s until %s: %s", p.Release, p.WaitsFor, p.Why)
	}
	out := map[string]fid5Trace{}
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var tr fid5Trace
		if err := json.Unmarshal(raw, &tr); err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		out[filepath.Base(p)] = tr
	}
	return out
}

// Both kinds of trace exist for each interface, so neither can silently be missing.
func TestEachOffloadInterfaceHasRecordedAndDirectTraces(t *testing.T) {
	have := map[string]bool{}
	for name, tr := range loadFID5(t) {
		kind := strings.SplitN(name, "-", 2)[0]
		for iface, calls := range map[string][]fid5Call{"spec": tr.Spec, "manager": tr.Manager, "policy": tr.Policy} {
			if len(calls) > 0 {
				have[kind+" "+iface] = true
			}
		}
	}
	for _, k := range []string{"recorded spec", "recorded manager", "recorded policy", "direct spec", "direct manager", "direct policy"} {
		if !have[k] {
			t.Errorf("no %s trace in testdata/golden-fid5", k)
		}
	}
}

func TestThePortAnswersEveryOffloadInterfaceCallAsVLLMDoes(t *testing.T) {
	for name, tr := range loadFID5(t) {
		t.Run(name, func(t *testing.T) {
			if v, _ := tr.Provenance["mock_version"].(string); v == "" {
				t.Error("the trace records no mock version")
			}
			numChunks := replaySpec(t, tr.Spec)
			if len(tr.Manager) > 0 {
				if numChunks < 0 {
					t.Fatal("manager calls with no spec that built the manager")
				}
				replayManager(t, numChunks, tr.Manager)
			}
			replayPolicy(t, tr.Policy)
		})
	}
}

func arg[T any](t *testing.T, c fid5Call, k string) T {
	t.Helper()
	var v T
	raw, ok := c.Args[k]
	if !ok {
		t.Fatalf("%s: no argument %q", c.M, k)
	}
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("%s %s: %v", c.M, k, err)
	}
	return v
}

func key(t *testing.T, h string) Key {
	t.Helper()
	b, err := hex.DecodeString(h)
	if err != nil {
		t.Fatal(err)
	}
	return Key(b)
}

func keys(t *testing.T, hs []string) []Key {
	out := make([]Key, len(hs))
	for i, h := range hs {
		out[i] = key(t, h)
	}
	return out
}

func hexes(ks []Key) []string {
	out := make([]string, len(ks))
	for i, k := range ks {
		out[i] = hex.EncodeToString([]byte(k))
	}
	return out
}

// same compares the port's answer, as JSON, with vLLM's.
func same(t *testing.T, i int, c fid5Call, got any) {
	t.Helper()
	g, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var a, b any
	if err := json.Unmarshal(g, &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(c.Ret, &b); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("call %d %s(%s): the port answers %s, vLLM %s", i, c.M, args(c), g, c.Ret)
	}
}

func args(c fid5Call) string {
	var parts []string
	for k, v := range c.Args {
		parts = append(parts, k+"="+string(v))
	}
	slices.Sort(parts)
	return strings.Join(parts, ", ")
}

// replaySpec checks each get_manager: the chunk count the port sizes the CPU tier with
// (NumChunks), and that the manager is the CPU tier the port runs. It returns the last count, or
// -1 with no spec call.
func replaySpec(t *testing.T, calls []fid5Call) int {
	n := -1
	for i, c := range calls {
		spec := arg[string](t, c, "spec")
		copies := arg[int](t, c, "world_size")
		if arg[bool](t, c, "replicated_layout") {
			copies = 1
		}
		got := NumChunks(arg[int64](t, c, "cpu_bytes_to_use"), arg[int64](t, c, "worker_kv_bytes_per_block"), copies,
			arg[int](t, c, "blocks_per_chunk"), arg[int64](t, c, "alignment"))
		var ret struct {
			Manager   string `json:"manager"`
			NumChunks int    `json:"num_chunks"`
		}
		if err := json.Unmarshal(c.Ret, &ret); err != nil {
			t.Fatal(err)
		}
		if got != ret.NumChunks {
			t.Fatalf("spec call %d (%s): the port sizes %d chunks, vLLM %d", i, args(c), got, ret.NumChunks)
		}
		// The port runs both registered specs as its CPU tier (Check), TieringOffloadingSpec only with
		// no secondary tiers, over the LRU policy.
		if err := Check(Options{SpecName: spec, SecondaryTiers: arg[int](t, c, "secondary_tiers"),
			EvictionPolicy: arg[string](t, c, "eviction_policy")}); err != nil {
			t.Fatalf("spec call %d: vLLM ran a spec the port refuses: %v", i, err)
		}
		want := map[string]string{"CPUOffloadingSpec": "CPUOffloadingManager", "TieringOffloadingSpec": "TieringOffloadingManager"}[spec]
		if ret.Manager != want {
			t.Fatalf("spec call %d: %s built a %s, want %s", i, spec, ret.Manager, want)
		}
		n = got
	}
	return n
}

// cpuStatNames are the CPU tier's get_stats metrics (vllm/v1/kv_offload/cpu/common.py:8-13@5df63a72).
const (
	statUsage = "vllm:kv_offload_cpu_cache_usage_perc"
	statWrite = "vllm:kv_offload_cpu_cache_write_usage_perc"
	statRead  = "vllm:kv_offload_cpu_cache_read_usage_perc"
	statAlloc = "vllm:kv_offload_cpu_allocation_size"
)

func replayManager(t *testing.T, numChunks int, calls []fid5Call) {
	m := newCPUManager(numChunks)
	acc := map[string]*reqAccess{}
	access := func(c fid5Call) *reqAccess {
		name := arg[string](t, c, "ctx")
		if acc[name] == nil {
			acc[name] = newReqAccess()
		}
		return acc[name]
	}
	for i, c := range calls {
		switch c.M {
		case "on_new_request":
			same(t, i, c, map[string]string{"policy": m.onNewRequest(access(c))})
		case "lookup":
			same(t, i, c, [...]string{"MISS", "HIT", "HIT_PENDING"}[m.lookup(key(t, arg[string](t, c, "key")))])
		case "get_load_source":
			same(t, i, c, m.loadSource())
		case "prepare_load":
			same(t, i, c, m.prepareLoad(keys(t, arg[[]string](t, c, "keys")), access(c)))
		case "touch":
			m.touch(keys(t, arg[[]string](t, c, "keys")))
			same(t, i, c, nil)
		case "complete_load":
			m.completeLoad(keys(t, arg[[]string](t, c, "keys")))
			same(t, i, c, nil)
		case "prepare_store":
			stored, ids, evicted, ok := m.prepareStore(keys(t, arg[[]string](t, c, "keys")), access(c))
			if !ok {
				same(t, i, c, nil)
				break
			}
			same(t, i, c, map[string]any{"keys_to_store": hexes(stored), "ids": append([]int{}, ids...),
				"evicted": hexes(evicted)})
		case "complete_store":
			m.completeStore(keys(t, arg[[]string](t, c, "keys")), arg[bool](t, c, "success"))
			same(t, i, c, nil)
		case "on_request_finished":
			a := access(c)
			for h, p := range arg[map[string]int](t, c, "positions") {
				a.positions[key(t, h)] = p
			}
			m.requestFinished(a)
			same(t, i, c, nil)
		case "on_schedule_end", "shutdown":
			// OffloadingManager's defaults, which the CPU manager keeps: nothing to do
			// (vllm/v1/kv_offload/base.py:395-421@5df63a72).
			same(t, i, c, nil)
		case "has_pending_work":
			same(t, i, c, m.hasPendingWork())
		case "take_events":
			// The CPU tier publishes no events in the port; vLLM's must be none too.
			same(t, i, c, []any{})
		case "reset_cache":
			m.resetCache()
			same(t, i, c, nil)
		case "get_stats":
			compareStats(t, i, c, m.stats())
		default:
			t.Fatalf("call %d: OffloadingManager.%s is not replayed", i, c.M)
		}
	}
}

func compareStats(t *testing.T, i int, c fid5Call, st cpuStats) {
	t.Helper()
	var got map[string]map[string]any
	if err := json.Unmarshal(c.Ret, &got); err != nil {
		t.Fatalf("call %d get_stats: %v", i, err)
	}
	want := map[string]map[string]any{statUsage: {"()": st.usage}, statWrite: {"()": st.write}, statRead: {"()": st.read}}
	if len(st.allocations) > 0 {
		var sizes []any
		for _, a := range st.allocations {
			sizes = append(sizes, float64(a))
		}
		want[statAlloc] = map[string]any{"()": sizes}
	}
	if !reflect.DeepEqual(got, want) {
		g, _ := json.Marshal(want)
		t.Fatalf("call %d get_stats: the port reports %s, vLLM %s", i, g, c.Ret)
	}
}

func replayPolicy(t *testing.T, calls []fid5Call) {
	p := newLRU()
	for i, c := range calls {
		switch c.M {
		case "insert":
			ch := arg[struct {
				ID  int `json:"id"`
				Ref int `json:"ref"`
			}](t, c, "chunk")
			p.insert(key(t, arg[string](t, c, "key")), &chunk{refCnt: ch.Ref, id: ch.ID})
			same(t, i, c, nil)
		case "get":
			// The chunk's ref_cnt is the manager's to change, outside the policy; its slot is the
			// policy's answer.
			var want *struct {
				ID int `json:"id"`
			}
			if err := json.Unmarshal(c.Ret, &want); err != nil {
				t.Fatal(err)
			}
			got := p.get(key(t, arg[string](t, c, "key")))
			if (got == nil) != (want == nil) || (got != nil && got.id != want.ID) {
				t.Fatalf("call %d get(%s): the port answers %v, vLLM %s", i, args(c), got, c.Ret)
			}
		case "remove":
			p.remove(key(t, arg[string](t, c, "key")))
			same(t, i, c, nil)
		case "clear":
			p.clear()
			same(t, i, c, nil)
		case "mark_evictable":
			p.markEvictable(key(t, arg[string](t, c, "key")))
			same(t, i, c, nil)
		case "mark_non_evictable":
			p.markNonEvictable(key(t, arg[string](t, c, "key")))
			same(t, i, c, nil)
		case "touch":
			p.touchReq(keys(t, arg[[]string](t, c, "keys")))
			same(t, i, c, nil)
		case "on_store_miss":
			p.onStoreMiss(keys(t, arg[[]string](t, c, "keys")))
			same(t, i, c, nil)
		case "on_request_finished":
			var groups [][]Key
			for _, g := range arg[[][]string](t, c, "groups") {
				groups = append(groups, keys(t, g))
			}
			pos := map[Key]int{}
			for h, v := range arg[map[string]int](t, c, "positions") {
				pos[key(t, h)] = v
			}
			p.onRequestFinished(groups, pos)
			same(t, i, c, nil)
		case "evict":
			prot := map[Key]bool{}
			for _, k := range keys(t, arg[[]string](t, c, "protected")) {
				prot[k] = true
			}
			out := p.evict(arg[int](t, c, "n"), prot)
			if out == nil {
				same(t, i, c, nil)
				break
			}
			got := []map[string]any{}
			for _, e := range out {
				got = append(got, map[string]any{"key": hex.EncodeToString([]byte(e.key)), "id": e.chunk.id})
			}
			same(t, i, c, got)
		default:
			t.Fatalf("call %d: CachePolicy.%s is not replayed", i, c.M)
		}
	}
}

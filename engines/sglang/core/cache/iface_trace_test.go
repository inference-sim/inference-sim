// Original to blis-sglang.

package cache

import (
	"reflect"
	"strings"
	"testing"

	blissglang "github.com/inference-sim/inference-sim/engines/sglang"
)

// The Go interface that ports each framework interface.
var portOf = map[Interface]reflect.Type{
	PrefixCacheIface: reflect.TypeFor[PrefixCache](),
	TreeCoreIface:    reflect.TypeFor[TreeCore](),
	ComponentIface:   reflect.TypeFor[Component](),
	AllocatorIface:   reflect.TypeFor[Allocator](),
	ReqToTokenIface:  reflect.TypeFor[ReqToTokenPool](),
	KVPoolIface:      reflect.TypeFor[KVPool](),
}

func loadGoldens(t *testing.T) map[string]Trace {
	t.Helper()
	trs, err := LoadTraces(GoldenDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(trs) == 0 {
		t.Fatalf("no trace under %s", GoldenDir)
	}
	return trs
}

// FID-5: the framework is traced at its own interfaces, by both kinds of trace. A recorded trace
// is a real feature's calls (SGLang's scheduler running a scenario); a direct trace is the
// driver's, in orderings no feature makes. Each interface the scheduler drives directly is
// driven by both.
func TestEachCacheInterfaceHasRecordedAndDirectTraces(t *testing.T) {
	have := map[string]bool{}
	for name, tr := range loadGoldens(t) {
		kind := tr.Kind
		_ = name
		for _, c := range tr.Calls {
			have[kind+" "+string(c.Iface)] = true
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

// Every trace is SGLang's own output at the release this build models, from the oracle the tree
// holds: its provenance names the pinned commit, and it names a manifest version.
func TestCacheTracesAreFromThePinnedSGLang(t *testing.T) {
	for name, tr := range loadGoldens(t) {
		if got := tr.Provenance["sglang_commit"]; got != blissglang.Upstream.Commit {
			t.Errorf("%s: recorded at SGLang %v, and the build models %s", name, got, blissglang.Upstream.Commit)
		}
		if v, _ := tr.Provenance["mock_version"].(string); !strings.HasPrefix(v, "gpu-mock/") {
			t.Errorf("%s: no manifest version (%v)", name, tr.Provenance["mock_version"])
		}
		for _, c := range tr.Calls {
			if c.In != nil && *c.In >= c.Seq {
				t.Errorf("%s: call %d is made by call %d, which comes after it", name, c.Seq, *c.In)
			}
		}
	}
}

// Every method SGLang's classes were called on, at any interface, in any trace, is a method of
// that interface's Go port: a call the port has no method for is a part of the framework the
// port does not model, and fails here rather than pass unnoticed. Construction (__init__) is the
// pools' sizing, which their constructors port.
func TestEveryTracedCallIsAMethodOfThePort(t *testing.T) {
	for name, tr := range loadGoldens(t) {
		for iface, methods := range tr.Methods() {
			port, ok := portOf[iface]
			if !ok {
				t.Errorf("%s: calls at %q, which is not a framework interface", name, iface)
				continue
			}
			for _, m := range methods {
				if m == "__init__" {
					continue
				}
				if _, ok := port.MethodByName(GoName(m)); !ok {
					t.Errorf("%s: SGLang's %s.%s has no Go method %s.%s", name, iface, m, port.Name(), GoName(m))
				}
			}
		}
	}
}

// The pairing of node ids that replays use: one consistent bijection, refusing a node paired
// twice either way.
func TestNodePairingIsOneToOne(t *testing.T) {
	n := NewNodes()
	if err := n.Pair(10, 1); err != nil {
		t.Fatal(err)
	}
	if err := n.Pair(10, 1); err != nil {
		t.Errorf("pairing the same nodes again: %v", err)
	}
	if err := n.Pair(10, 2); err == nil {
		t.Error("SGLang's node 10 paired with a second port node")
	}
	if err := n.Pair(11, 1); err == nil {
		t.Error("the port's node 1 paired with a second SGLang node")
	}
	if p, err := n.Port(10); err != nil || p != 1 {
		t.Errorf("Port(10) = %d, %v; want 1", p, err)
	}
	if _, err := n.Port(12); err == nil {
		t.Error("a node no answer named has a port node")
	}
}

// The calls a replay drives a port with, and the calls it expects the port to make, come from
// the trace's own nesting: a radix cache's insert frees its duplicate slots at the allocator.
func TestTraceNestingNamesTheCallsAPortMakes(t *testing.T) {
	tr, ok := loadGoldens(t)["direct/insert-over-a-cached-prefix"]
	if !ok {
		t.Fatal("no direct/insert-over-a-cached-prefix trace")
	}
	var inserts []Call
	for _, c := range tr.Top(PrefixCacheIface) {
		if c.Method == "insert" {
			inserts = append(inserts, c)
		}
	}
	if len(inserts) < 2 {
		t.Fatalf("%d top-level inserts, want at least 2", len(inserts))
	}
	// The second insert shares six tokens with the first and brings its own slots for them:
	// the tree keeps its own and frees the six new ones.
	var freed []int64
	for _, c := range tr.Calls {
		if c.Iface == AllocatorIface && c.Method == "free_segment" && isWithin(tr, c, inserts[1].Seq) {
			var s []int64
			if err := c.Arg("free_index", &s); err != nil {
				t.Fatal(err)
			}
			freed = append(freed, s...)
		}
	}
	var p struct {
		Value []int64 `json:"value"`
	}
	if err := inserts[1].Arg("params", &p); err != nil {
		t.Fatal(err)
	}
	var r InsertResult
	if err := inserts[1].Answer(&r); err != nil {
		t.Fatal(err)
	}
	if r.PrefixLen != 6 || !reflect.DeepEqual(freed, p.Value[:6]) {
		t.Errorf("the second insert matched %d tokens and freed %v; want 6 and %v", r.PrefixLen, freed, p.Value[:6])
	}
}

// isWithin reports whether c was made, at any depth, by the call numbered seq.
func isWithin(tr Trace, c Call, seq int) bool {
	bySeq := map[int]Call{}
	for _, x := range tr.Calls {
		bySeq[x.Seq] = x
	}
	for c.In != nil {
		if *c.In == seq {
			return true
		}
		c = bySeq[*c.In]
	}
	return false
}

func TestGoNameMapsSGLangMethodNames(t *testing.T) {
	for m, want := range map[string]string{
		"match_prefix": "MatchPrefix", "free_kv_row": "FreeKVRow", "get_kvcache": "GetKVCache",
		"window_reprefill_tail_tokens": "WindowReprefillTailTokens", "__init__": "New",
	} {
		if got := GoName(m); got != want {
			t.Errorf("GoName(%q) = %q, want %q", m, got, want)
		}
	}
}

// A replay rebuilds the request pool from the writes: each write names the positions it sets,
// in both of the forms SGLang indexes the pool with. Replaying every write of a scheduler run
// gives each request's row as SGLang's checkpoint read it.
func TestRowWritesReplayIntoTheRowsCheckpointReads(t *testing.T) {
	tr, ok := loadGoldens(t)["recorded/tiny"]
	if !ok {
		t.Fatal("no recorded/tiny trace")
	}
	rows := map[int]map[int]int64{}
	var checked, sliced int
	for _, c := range tr.Calls {
		switch {
		case c.Iface == ReqToTokenIface && c.Method == "write":
			slots, err := c.RowWrite()
			if err != nil {
				t.Fatal(err)
			}
			if len(slots) > 1 && slots[0].Row == slots[len(slots)-1].Row {
				sliced++
			}
			for _, s := range slots {
				if rows[s.Row] == nil {
					rows[s.Row] = map[int]int64{}
				}
				rows[s.Row][s.Pos] = s.Slot
			}
		case c.Iface == PrefixCacheIface && c.Method == "checkpoint" && c.In == nil:
			var req struct {
				KV  ReqKV   `json:"kv"`
				Row []int64 `json:"row"`
			}
			if err := c.Arg("req", &req); err != nil {
				t.Fatal(err)
			}
			for pos, slot := range req.Row {
				if got := rows[*req.KV.ReqPoolIdx][pos]; got != slot {
					t.Errorf("checkpoint (seq %d): row %d position %d holds slot %d after the replayed writes; SGLang read %d",
						c.Seq, *req.KV.ReqPoolIdx, pos, got, slot)
				}
			}
			checked++
		}
	}
	if checked == 0 || sliced == 0 {
		t.Errorf("%d checkpoints checked against %d sliced writes; the trace must have both", checked, sliced)
	}
}

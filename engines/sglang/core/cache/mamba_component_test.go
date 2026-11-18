package cache

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// MambaComponentGoldenDir holds the traces oracle/run_mamba_component.py records at SGLang's
// MambaComponent request-side lifecycle (inference-sim/blis-sglang#128, piece A1).
const MambaComponentGoldenDir = "testdata/golden-pools/mamba-component"

// treeSlots stands in for the mamba states the tree holds: an eviction frees the oldest.
type treeSlots struct {
	pool      *HybridRequestPool
	slots     []int64
	evictions []map[string]any
}

func (t *treeSlots) EvictForAlloc(n int) {
	t.evictions = append(t.evictions, map[string]any{"m": "evict_for_alloc", "mamba_num": n, "num_tokens": 0})
	for i := 0; i < n && len(t.slots) > 0; i++ {
		t.pool.Mamba.Free([]int64{t.slots[0]})
		t.slots = t.slots[1:]
	}
}

func mambaComponentNames(t *testing.T) []string {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(MambaComponentGoldenDir, "*.trace.json"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no trace under %s (%v)", MambaComponentGoldenDir, err)
	}
	var names []string
	for _, p := range paths {
		names = append(names, strings.TrimSuffix(filepath.Base(p), ".trace.json"))
	}
	return names
}

func intPtrOf(v any) *int {
	if v == nil {
		return nil
	}
	n := asInt(v)
	return &n
}

// MambaComponent's request-side lifecycle, driven by the scenarios' calls, answers and leaves the
// donated state, the request's slots and the pool as SGLang's own MambaComponent does, including
// the eviction it asks for when the mamba pool is exhausted.
func TestMambaLifecycleAnswersAsSGLang(t *testing.T) {
	driven := 0
	for _, name := range mambaComponentNames(t) {
		var sc struct {
			Pools []struct {
				Args map[string]any `json:"args"`
			} `json:"pools"`
			Component struct {
				ExtraBuffer bool `json:"extra_buffer"`
				TreeSlots   int  `json:"tree_mamba_slots"`
			} `json:"component"`
			Calls []map[string]any `json:"calls"`
		}
		var tr struct {
			Calls []map[string]any `json:"calls"`
		}
		for path, into := range map[string]any{
			filepath.Join("..", "..", "oracle", "scenarios-mamba-component", name+".json"): &sc,
			filepath.Join(MambaComponentGoldenDir, name+".trace.json"):                     &tr,
		} {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(raw, into); err != nil {
				t.Fatalf("%s: %v", path, err)
			}
		}
		pa := sc.Pools[0].Args
		overlap := true
		if v, ok := pa["enable_overlap_schedule"]; ok {
			overlap = asBool(v)
		}
		pool := NewHybridRequestPool(asInt(pa["size"]), asInt(pa["mamba_size"]), asInt(pa["max_context_len"]),
			asBool(pa["enable_mamba_extra_buffer"]), asBool(pa["enable_mamba_extra_buffer_lazy"]), overlap)
		tree := &treeSlots{pool: pool}
		if sc.Component.TreeSlots > 0 {
			tree.slots, _ = pool.Mamba.Alloc(sc.Component.TreeSlots)
		}
		life := &MambaLifecycle{Pool: pool, ExtraBuffer: sc.Component.ExtraBuffer, Evictor: tree}
		reqs := map[string]*Req{}
		inserts := map[string]*MambaInsert{}
		results := map[string]*MambaInsertResult{}
		k := 0
		for _, c := range sc.Calls {
			switch c["op"] {
			case "req":
				reqs[c["as"].(string)] = &Req{ID: c["as"].(string)}
				continue
			case "set":
				r := reqs[c["req"].(string)]
				if r.KV.Mamba == nil {
					r.KV.Mamba = &MambaKV{}
				}
				for key, v := range c["set"].(map[string]any) {
					switch key {
					case "kv_allocated_len":
						r.KV.KVAllocatedLen = asInt(v)
					case "mamba_last_track_seqlen":
						r.KV.Mamba.LastTrackSeqlen = intPtrOf(v)
					case "mamba_prev_track_seqlen":
						r.KV.Mamba.PrevTrackSeqlen = intPtrOf(v)
					default:
						t.Fatalf("%s: set %s is not modelled", name, key)
					}
				}
				continue
			case "insert_params":
				inserts[c["as"].(string)] = &MambaInsert{}
				continue
			case "insert_result":
				results[c["as"].(string)] = &MambaInsertResult{Exist: asBool(c["set"].(map[string]any)["mamba_exist"])}
				continue
			}
			rec := tr.Calls[k]
			k++
			args, _ := c["args"].(map[string]any)
			get := func(key, kind string) any {
				v, ok := args[key]
				if !ok {
					return nil
				}
				return v.(map[string]any)[kind]
			}
			var ret any
			var raised string
			switch {
			case c["on"] == "h" && c["m"] == "alloc":
				var rs []*Req
				for _, n := range args["reqs"].(map[string]any)["reqs"].([]any) {
					rs = append(rs, reqs[n.(string)])
				}
				if rows, err := allocOrAssert(pool, rs); err != nil {
					raised = "AssertionError"
				} else if rows != nil {
					ret = rows
				}
			case c["m"] == "prepare_for_caching_req":
				n, err := life.PrepareForCachingReq(reqs[get("req", "req").(string)], inserts[get("insert_params", "insert_params").(string)],
					asInt(args["token_ids_len"]), asBool(args["is_finished"]))
				if err != nil {
					raised = "AssertionError"
				}
				ret = n
			case c["m"] == "cleanup_after_caching_req":
				var res *MambaInsertResult
				if n := get("insert_result", "insert_result"); n != nil {
					res = results[n.(string)]
				}
				var ins *MambaInsert
				if n := get("insert_params", "insert_params"); n != nil {
					ins = inserts[n.(string)]
				}
				if err := life.CleanupAfterCachingReq(reqs[get("req", "req").(string)], asBool(args["is_finished"]), res, ins); err != nil {
					raised = "AssertionError"
				}
			default:
				t.Fatalf("%s: call %v is not modelled", name, c["m"])
			}
			if (rec["raises"] != nil) != (raised != "") {
				t.Fatalf("%s call %d (%v): port raised %q, SGLang %v", name, k, c["m"], raised, rec["raises"])
			}
			if raised == "" {
				if got, want := mustJSON(t, ret), mustJSON(t, rec["ret"]); got != want {
					t.Errorf("%s call %d (%v): answer %s, SGLang %s", name, k, c["m"], got, want)
				}
			}
			objs := map[string]any{}
			for n, p := range inserts {
				o := map[string]any{"mamba_keep_idx": nil, "mamba_value": nil}
				if p.KeepIdx != nil {
					o["mamba_keep_idx"] = *p.KeepIdx
				}
				if p.Value != nil {
					o["mamba_value"] = p.Value
				}
				objs[n] = o
			}
			for n, r := range results {
				objs[n] = map[string]any{"mamba_exist": r.Exist}
			}
			if got, want := mustJSON(t, objs), mustJSON(t, rec["objs"]); got != want {
				t.Errorf("%s call %d (%v): insert objects %s, SGLang %s", name, k, c["m"], got, want)
			}
			ev := tree.evictions
			if ev == nil {
				ev = []map[string]any{}
			}
			if got, want := mustJSON(t, ev), mustJSON(t, rec["evictions"]); got != want {
				t.Errorf("%s call %d (%v): evictions %s, SGLang %s", name, k, c["m"], got, want)
			}
			tree.evictions = nil
			held := tree.slots
			if held == nil {
				held = []int64{}
			}
			if got, want := mustJSON(t, held), mustJSON(t, rec["tree_mamba_slots"]); got != want {
				t.Errorf("%s call %d (%v): tree-held slots %s, SGLang %s", name, k, c["m"], got, want)
			}
			checkMambaState(t, name, k, c, pool, rec, reqs)
			checkTrackSeqlens(t, name, k, c, rec, reqs)
			driven++
		}
	}
	if driven == 0 {
		t.Fatal("no call was driven")
	}
}

func checkTrackSeqlens(t *testing.T, name string, k int, c map[string]any, rec map[string]any, reqs map[string]*Req) {
	t.Helper()
	rs, _ := rec["reqs"].(map[string]any)
	for rn, want := range rs {
		m := reqs[rn].KV.Mamba
		var last, prev any
		if m != nil && m.LastTrackSeqlen != nil {
			last = *m.LastTrackSeqlen
		}
		if m != nil && m.PrevTrackSeqlen != nil {
			prev = *m.PrevTrackSeqlen
		}
		w := want.(map[string]any)
		if mustJSON(t, last) != mustJSON(t, w["mamba_last_track_seqlen"]) || mustJSON(t, prev) != mustJSON(t, w["mamba_prev_track_seqlen"]) {
			t.Errorf("%s call %d (%v): %s tracks %v/%v, SGLang %v/%v", name, k, c["m"], rn, last, prev, w["mamba_last_track_seqlen"], w["mamba_prev_track_seqlen"])
		}
	}
}

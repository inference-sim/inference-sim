// Original to blis-sglang.

package cache

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// MambaGoldenDir holds the traces oracle/run_mamba_pools.py records at SGLang's hybrid pools, the
// state budget and the request-slot cap (inference-sim/blis-sglang#94).
const MambaGoldenDir = "testdata/golden-pools/mamba"

type mambaScenario struct {
	ServerArgs map[string]any `json:"server_args"`
	PageSize   int            `json:"page_size"`
	Config     struct {
		Fields map[string]any `json:"fields"`
	} `json:"config"`
	Pools []struct {
		As    string         `json:"as"`
		Class string         `json:"class"`
		Args  map[string]any `json:"args"`
	} `json:"pools"`
	Calls   []map[string]any `json:"calls"`
	Budgets []struct {
		Name          string         `json:"name"`
		TotalRestGiB  float64        `json:"total_rest_memory_gib"`
		ServerArgs    map[string]any `json:"server_args"`
		SpecAlgorithm *string        `json:"spec_algorithm"`
		TokenCapacity *int           `json:"token_capacity"`
		ContextLen    int            `json:"context_len"`
		AttnDPSize    int            `json:"attn_dp_size"`
	} `json:"budgets"`
}

type mambaTrace struct {
	Facts struct {
		PerReq       int64    `json:"mamba_cache_per_req"`
		Layers       []int    `json:"layers"`
		Full         []int    `json:"full_attention_layer_ids"`
		Conv         [][2]int `json:"conv"`
		Temporal     [3]int   `json:"temporal"`
		ConvDType    string   `json:"conv_dtype"`
		TemporalType string   `json:"temporal_dtype"`
		KHeads       int      `json:"num_k_heads_per_tp"`
		Ring16       int64    `json:"replayssm_ring_bytes_per_req_16"`
	} `json:"facts"`
	Calls  []json.RawMessage `json:"calls"`
	Sizing []struct {
		Pool          string  `json:"pool"`
		Class         string  `json:"class"`
		Bytes         int64   `json:"bytes"`
		MambaSlots    int     `json:"mamba_slots"`
		TrackingSlots int     `json:"tracking_slots"`
		InitialTrack  int     `json:"initial_tracking_slots"`
		KVSizeBytes   []int64 `json:"kv_size_bytes"`
	} `json:"sizing"`
	Budgets []struct {
		Name       string   `json:"name"`
		Rest       *float64 `json:"rest_gib"`
		Size       *int     `json:"max_mamba_cache_size"`
		Ratio      *int     `json:"ratio"`
		MaxNumReqs *int     `json:"max_num_reqs"`
		Raises     string   `json:"raises"`
	} `json:"budgets"`
}

func loadMamba(t *testing.T, name string) (mambaScenario, mambaTrace) {
	t.Helper()
	var sc mambaScenario
	var tr mambaTrace
	for path, into := range map[string]any{
		filepath.Join("..", "..", "oracle", "scenarios-mamba-pools", name+".json"): &sc,
		filepath.Join(MambaGoldenDir, name+".trace.json"):                          &tr,
	} {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(raw, into); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
	}
	return sc, tr
}

func mambaGoldenNames(t *testing.T) []string {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(MambaGoldenDir, "*.trace.json"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no trace under %s (%v)", MambaGoldenDir, err)
	}
	var names []string
	for _, p := range paths {
		names = append(names, strings.TrimSuffix(filepath.Base(p), ".trace.json"))
	}
	return names
}

func asInt(v any) int {
	f, ok := v.(float64)
	if !ok {
		return 0
	}
	return int(f)
}

func asBool(v any) bool { b, _ := v.(bool); return b }

func (s mambaScenario) linearConfig() LinearAttnConfig {
	f := s.Config.Fields
	return LinearAttnConfig{NumHiddenLayers: asInt(f["num_hidden_layers"]), FullAttentionInterval: asInt(f["full_attention_interval"]),
		KeyHeadDim: asInt(f["linear_key_head_dim"]), NumKeyHeads: asInt(f["linear_num_key_heads"]),
		ValueHeadDim: asInt(f["linear_value_head_dim"]), NumValueHeads: asInt(f["linear_num_value_heads"]),
		ConvKernel: asInt(f["linear_conv_kernel_dim"])}
}

func (s mambaScenario) stateParams(t *testing.T, ssm *string) LinearStateParams {
	t.Helper()
	if v, ok := s.ServerArgs["mamba_ssm_dtype"].(string); ok && ssm == nil {
		ssm = &v
	}
	conv, temporal, err := MambaStateDType(ssm, asStr(s.Config.Fields["mamba_ssm_dtype"]))
	if err != nil {
		t.Fatal(err)
	}
	p, err := NewLinearStateParams(s.linearConfig(), max(1, asInt(s.ServerArgs["tp_size"])), conv, temporal)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func asStr(v any) string { s, _ := v.(string); return s }

// The state of one request in one layer, and the layers that hold it, are what SGLang's own
// configuration class computes from the model's linear-attention fields at the scenario's tensor
// parallelism: shapes, dtypes, bytes per request and the ReplaySSM rings.
func TestLinearStateIsSGLangs(t *testing.T) {
	for _, name := range mambaGoldenNames(t) {
		sc, tr := loadMamba(t, name)
		p := sc.stateParams(t, nil)
		f := tr.Facts
		if !reflect.DeepEqual(p.Conv, f.Conv) || p.Temporal != f.Temporal || p.NumKHeadsPerTP != f.KHeads {
			t.Errorf("%s: state shape %v %v (%d k heads), SGLang %v %v (%d)", name, p.Conv, p.Temporal, p.NumKHeadsPerTP,
				f.Conv, f.Temporal, f.KHeads)
		}
		if p.PerReqBytes() != f.PerReq || p.ReplayRingBytesPerReq(16) != f.Ring16 {
			t.Errorf("%s: %d bytes per request (ring %d), SGLang %d (%d)", name, p.PerReqBytes(), p.ReplayRingBytesPerReq(16),
				f.PerReq, f.Ring16)
		}
		if !slices.Equal(p.Layers, f.Layers) || !slices.Equal(sc.linearConfig().FullAttentionLayerIDs(), f.Full) {
			t.Errorf("%s: layers differ from SGLang's", name)
		}
	}
}

// Every pool SGLang built, built again by the port from the same arguments, holds the bytes
// SGLang's buffers hold, and the hybrid request pool charges the tracking slots SGLang does.
func TestHybridPoolBytesAreSGLangs(t *testing.T) {
	seen := 0
	for _, name := range mambaGoldenNames(t) {
		sc, tr := loadMamba(t, name)
		p := sc.stateParams(t, nil)
		for i, pool := range sc.Pools {
			want := tr.Sizing[i]
			if want.Pool != pool.As {
				t.Fatalf("%s: pool %d is %s in the trace, %s in the scenario", name, i, want.Pool, pool.As)
			}
			a := pool.Args
			switch pool.Class {
			case "MambaPool", "HybridReqToTokenPool":
				mp := MambaPoolParams{State: p, Size: asInt(a["size"]), SpecStateSize: asInt(a["spec_state_size"]),
					SpeculativeDraftTokens: asInt(a["speculative_num_draft_tokens"]), EagleTopK: asInt(a["speculative_eagle_topk"]),
					ReplayDecode: asBool(a["enable_linear_replayssm"]), ReplaySpec: asBool(a["enable_linear_replayssm_spec"]),
					ReplayCacheLen: asInt(a["linear_replayssm_cache_len"])}
				if pool.Class == "HybridReqToTokenPool" {
					mp.Size, mp.SpecStateSize = asInt(a["mamba_size"]), asInt(a["mamba_spec_state_size"])
				}
				got, err := MambaPoolBytes(mp)
				if err != nil {
					t.Fatal(err)
				}
				if got != want.Bytes {
					t.Errorf("%s/%s: %d bytes, SGLang %d", name, pool.As, got, want.Bytes)
				}
				seen++
			case "HybridLinearKVPool":
				kv, err := NewHybridKVPool(HybridKVParams{Size: asInt(a["size"]), PageSize: asInt(a["page_size"]), DType: BFloat16,
					HeadNum: asInt(a["head_num"]), HeadDim: asInt(a["head_dim"]), FullAttentionLayers: tr.Facts.Full})
				if err != nil {
					t.Fatal(err)
				}
				k, v := kv.KVSizeBytes()
				if k != want.KVSizeBytes[0] || v != want.KVSizeBytes[1] {
					t.Errorf("%s/%s: K %d V %d bytes, SGLang %v", name, pool.As, k, v, want.KVSizeBytes)
				}
				seen++
			}
		}
	}
	if seen == 0 {
		t.Fatal("no pool was checked")
	}
}

// The hybrid request pool, driven by the scenario's calls, answers and leaves its slots, maps and
// requests as SGLang's pool does after every call, including the assertion it raises when the
// mamba pool is exhausted.
func TestHybridRequestPoolAnswersAsSGLang(t *testing.T) {
	driven := 0
	for _, name := range mambaGoldenNames(t) {
		sc, tr := loadMamba(t, name)
		if len(sc.Calls) == 0 {
			continue
		}
		pa := sc.Pools[0].Args
		extra := asBool(pa["enable_mamba_extra_buffer"])
		overlap := true
		if v, ok := pa["enable_overlap_schedule"]; ok {
			overlap = asBool(v)
		}
		pool := NewHybridRequestPool(asInt(pa["size"]), asInt(pa["mamba_size"]), asInt(pa["max_context_len"]), extra,
			asBool(pa["enable_mamba_extra_buffer_lazy"]), overlap)
		if w := tr.Sizing[0]; pool.PingPongSize() != w.TrackingSlots || pool.InitialTrackingSlots() != w.InitialTrack {
			t.Errorf("%s: %d/%d tracking slots, SGLang %d/%d", name, pool.PingPongSize(), pool.InitialTrackingSlots(),
				w.TrackingSlots, w.InitialTrack)
		}
		reqs := map[string]*Req{}
		get := func(a any) *Req { return reqs[a.(map[string]any)["req"].(string)] }
		var recs []map[string]any
		for _, raw := range tr.Calls {
			var rec map[string]any
			if err := json.Unmarshal(raw, &rec); err != nil {
				t.Fatal(err)
			}
			recs = append(recs, rec)
		}
		k := 0
		for _, c := range sc.Calls {
			switch c["op"] {
			case "req":
				reqs[c["as"].(string)] = &Req{ID: c["as"].(string)}
				continue
			case "set":
				r := reqs[c["req"].(string)]
				for key, v := range c["set"].(map[string]any) {
					if key == "kv_allocated_len" {
						r.KV.KVAllocatedLen = asInt(v)
					}
				}
				continue
			}
			rec := recs[k]
			k++
			args, _ := c["args"].(map[string]any)
			var ret any
			var raised string
			switch c["m"] {
			case "alloc":
				var rs []*Req
				for _, n := range args["reqs"].(map[string]any)["reqs"].([]any) {
					rs = append(rs, reqs[n.(string)])
				}
				rows, err := allocOrAssert(pool, rs)
				if err != nil {
					raised = "AssertionError"
				} else if rows != nil {
					ret = rows
				}
			case "free":
				pool.Free(get(args["req"]))
			case "free_mamba_cache":
				r := get(args["req"])
				row := 0
				if r.KV.ReqPoolIdx != nil {
					row = *r.KV.ReqPoolIdx
				}
				var keep *int
				if v, ok := args["mamba_ping_pong_track_buffer_to_keep"]; ok {
					n := asInt(v)
					keep = &n
				}
				if err := pool.FreeMambaCache(row, r.KV.Mamba, keep); err != nil {
					raised = "AssertionError"
				}
			case "get_mamba_ping_pong_other_idx":
				ret = pool.PingPongOtherIdx(asInt(args["mamba_next_track_idx"]))
			case "get_mamba_ping_pong_keep_idx":
				ret = pool.PingPongKeepIdx(get(args["req"]).KV.Mamba)
			case "donate_mamba_ping_pong_slot":
				r := get(args["req"])
				slot := int64(asInt(args["new_slot"].(map[string]any)["tensor"].([]any)[0]))
				ret = []int64{pool.DonatePingPongSlot(*r.KV.ReqPoolIdx, r.KV.Mamba, slot)}
			case "get_mamba_indices":
				var out []int64
				for _, i := range args["req_indices"].(map[string]any)["tensor"].([]any) {
					out = append(out, pool.ReqToMamba()[asInt(i)])
				}
				ret = out
			case "clear":
				pool.Clear()
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
			checkMambaState(t, name, k, c, pool, rec, reqs)
			driven++
		}
	}
	if driven == 0 {
		t.Fatal("no call was driven")
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	return strings.ReplaceAll(s, "null", "null")
}

func checkMambaState(t *testing.T, name string, k int, c map[string]any, pool *HybridRequestPool, rec map[string]any, reqs map[string]*Req) {
	t.Helper()
	st, _ := rec["state"].(map[string]any)
	got := map[string]any{"req_free": pool.FreeRows(), "mamba_free": pool.Mamba.FreeSlots(), "req_to_mamba": pool.ReqToMamba()}
	if pool.extraBuffer {
		got["ping_pong_map"] = pool.PingPongMap()
	}
	for key, g := range got {
		if mustJSON(t, g) != mustJSON(t, st[key]) {
			t.Errorf("%s call %d (%v): %s is %s, SGLang %s", name, k, c["m"], key, mustJSON(t, g), mustJSON(t, st[key]))
		}
	}
	rs, _ := rec["reqs"].(map[string]any)
	for rn, want := range rs {
		r := reqs[rn]
		w := want.(map[string]any)
		m := r.KV.Mamba
		var idx any
		if r.KV.ReqPoolIdx != nil {
			idx = *r.KV.ReqPoolIdx
		}
		var pool any
		var buf, empty, next, last any
		needsClear := false
		if m != nil {
			needsClear = m.NeedsClear
		}
		if m != nil && m.PoolIdx != nil {
			pool = *m.PoolIdx
		}
		if m != nil && m.PingPongBuffer != nil {
			buf, empty = m.PingPongBuffer, m.PingPongEmpty
		}
		if m != nil && m.NextTrackIdx != nil {
			next, last = *m.NextTrackIdx, *m.LastTrackIdx
		}
		for key, g := range map[string]any{"req_pool_idx": idx, "mamba_pool_idx": pool, "mamba_ping_pong_track_buffer": buf,
			"mamba_ping_pong_empty": empty, "mamba_next_track_idx": next, "mamba_last_track_idx": last,
			"mamba_needs_clear": needsClear} {
			if mustJSON(t, g) != mustJSON(t, w[key]) {
				t.Errorf("%s call %d (%v): %s.%s is %s, SGLang %s", name, k, c["m"], rn, key, mustJSON(t, g), mustJSON(t, w[key]))
			}
		}
	}
}

// The state budget, the state slots charged per request and the request-slot cap are what
// SGLang's configurator computes for each entry of the budget scenario, and a budget too small
// for one request is refused where SGLang refuses it.
func TestMambaBudgetIsSGLangs(t *testing.T) {
	sc, tr := loadMamba(t, "budget")
	if len(tr.Budgets) != len(sc.Budgets) {
		t.Fatalf("%d budgets in the trace, %d in the scenario", len(tr.Budgets), len(sc.Budgets))
	}
	for i, b := range sc.Budgets {
		want := tr.Budgets[i]
		sa := b.ServerArgs
		optInt := func(key string) *int {
			if v, ok := sa[key]; ok {
				n := asInt(v)
				return &n
			}
			return nil
		}
		var ssm *string
		if v, ok := sa["mamba_ssm_dtype"]; ok {
			s := v.(string)
			ssm = &s
		}
		ratio := 0.9
		if v, ok := sa["mamba_full_memory_ratio"]; ok {
			ratio = v.(float64)
		}
		strategy := MambaNoBuffer
		if v, ok := sa["mamba_radix_cache_strategy"]; ok {
			strategy = MambaStrategy(v.(string))
		}
		in := MambaBudget{TotalRestGiB: b.TotalRestGiB, State: sc.stateParams(t, ssm), MaxRunningRequests: optInt("max_running_requests"),
			MaxMambaCacheSize: optInt("max_mamba_cache_size"), DisableRadixCache: asBool(sa["disable_radix_cache"]), AttnDPSize: max(1, b.AttnDPSize),
			FullMemoryRatio: ratio, Strategy: strategy, DisableOverlap: asBool(sa["disable_overlap_schedule"]),
			SpecDraftTokens: asInt(sa["speculative_num_draft_tokens"]), ReplaySpec: asBool(sa["enable_linear_replayssm_spec"]),
			ReplayCacheLen: 16}
		if v, ok := sa["enable_linear_replayssm_spec"]; ok && asBool(v) {
			in.ReplayCacheLen = 16
		}
		size, rest, err := MambaCacheSize(in)
		if want.Raises != "" {
			if err == nil {
				t.Errorf("%s: SGLang raises %s, the port sized %d", b.Name, want.Raises, size)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", b.Name, err)
			continue
		}
		if size != *want.Size || math.Abs(rest-*want.Rest) > 1e-9*math.Max(1, math.Abs(*want.Rest)) {
			t.Errorf("%s: %d slots and %.9f GiB left, SGLang %d and %.9f", b.Name, size, rest, *want.Size, *want.Rest)
		}
		r, err := MambaRatio(in.DisableRadixCache, in.Strategy, in.DisableOverlap)
		if err != nil || r != *want.Ratio {
			t.Errorf("%s: ratio %d (%v), SGLang %d", b.Name, r, err, *want.Ratio)
		}
		if b.TokenCapacity != nil {
			n, err := MaxNumReqsMamba(*b.TokenCapacity, b.ContextLen, in.MaxRunningRequests, in.AttnDPSize, size, r)
			if err != nil || n != *want.MaxNumReqs {
				t.Errorf("%s: %d request slots (%v), SGLang %d", b.Name, n, err, *want.MaxNumReqs)
			}
		}
	}
}

// allocOrAssert is Alloc with the port's panics, which stand for SGLang's assertions, as errors.
func allocOrAssert(p *HybridRequestPool, reqs []*Req) (rows []int, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%v", r)
		}
	}()
	return p.Alloc(reqs)
}

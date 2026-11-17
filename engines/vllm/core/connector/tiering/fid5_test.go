// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Original to blis-vllm.

package tiering_test

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/core/connector/offloading"
	"github.com/inference-sim/blis-vllm/core/connector/tiering"
)

// The FID-5 traces of TieringOffloadingSpec with no secondary tier (core/connector/offloading's
// testdata/golden-fid5, recorded by oracle/drive_kv_interfaces.py at vLLM v0.33.0): vLLM's
// TieringOffloadingManager over its CPU primary tier, called directly. offloading's fid5_test
// replays their spec sizing and policy calls and leaves the manager calls to this package; here
// each manager call is made on tiering.Manager with no tier over the connector's CPU tier
// (tiering.CPU), and must give vLLM's answer.

const fid5Dir = "../offloading/testdata/golden-fid5"

type fid5Call struct {
	M    string                     `json:"m"`
	Args map[string]json.RawMessage `json:"args"`
	Ret  json.RawMessage            `json:"ret"`
}

type fid5Trace struct {
	Spec    []fid5Call `json:"spec"`
	Manager []fid5Call `json:"manager"`
}

func TestTheTieringManagerWithNoTierAnswersAsVLLMDoes(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join(fid5Dir, "direct-tiering-*.fid5.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) == 0 {
		t.Fatalf("no direct-tiering FID-5 traces in %s", fid5Dir)
	}
	for _, p := range paths {
		name := strings.TrimSuffix(filepath.Base(p), ".fid5.json")
		t.Run(name, func(t *testing.T) {
			raw, err := os.ReadFile(p)
			if err != nil {
				t.Fatal(err)
			}
			var tr fid5Trace
			if err := json.Unmarshal(raw, &tr); err != nil {
				t.Fatal(err)
			}
			replayFID5(t, tr)
		})
	}
}

func replayFID5(t *testing.T, tr fid5Trace) {
	if len(tr.Spec) != 1 {
		t.Fatalf("%d spec calls, want the one that built the manager", len(tr.Spec))
	}
	var spec struct {
		Spec  string `json:"spec"`
		Tiers int    `json:"secondary_tiers"`
	}
	specArgs, _ := json.Marshal(tr.Spec[0].Args)
	if err := json.Unmarshal(specArgs, &spec); err != nil {
		t.Fatal(err)
	}
	var built struct {
		Manager         string `json:"manager"`
		NumChunks       int    `json:"num_chunks"`
		KVBytesPerChunk int64  `json:"kv_bytes_per_chunk"`
	}
	var sizing struct {
		BlocksPerChunk int                       `json:"blocks_per_chunk"`
		MaxModelLen    int                       `json:"max_model_len"`
		Groups         []offloading.OffloadGroup `json:"groups"`
	}
	if err := json.Unmarshal(specArgs, &sizing); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(tr.Spec[0].Ret, &built); err != nil {
		t.Fatal(err)
	}
	if spec.Spec != "TieringOffloadingSpec" || spec.Tiers != 0 || built.Manager != "TieringOffloadingManager" {
		t.Fatalf("spec %s with %d tiers built %s, want TieringOffloadingSpec with none building TieringOffloadingManager",
			spec.Spec, spec.Tiers, built.Manager)
	}
	cpu := offloading.NewCPUPrimary(built.NumChunks)
	cpu.SetConfigInfo(offloading.ConfigInfo(built.NumChunks, sizing.BlocksPerChunk, built.KVBytesPerChunk, sizing.MaxModelLen, sizing.Groups))
	m := tiering.New(tiering.CPU(cpu), nil, nil)
	ctxs := map[string]*tiering.ReqContext{}
	arg := func(c fid5Call, k string, v any) {
		t.Helper()
		r, ok := c.Args[k]
		if !ok {
			t.Fatalf("%s: no argument %q", c.M, k)
		}
		if err := json.Unmarshal(r, v); err != nil {
			t.Fatalf("%s %s: %v", c.M, k, err)
		}
	}
	key := func(h string) tiering.Key {
		b, err := hex.DecodeString(h)
		if err != nil {
			t.Fatal(err)
		}
		return tiering.Key(b)
	}
	keys := func(c fid5Call) []tiering.Key {
		var hs []string
		arg(c, "keys", &hs)
		out := make([]tiering.Key, len(hs))
		for i, h := range hs {
			out[i] = key(h)
		}
		return out
	}
	// ctx is the call's request context; its CPU access state holds the token positions vLLM's CPU
	// tier read during the call (prepare_load, prepare_store, on_request_finished).
	ctx := func(c fid5Call) *tiering.ReqContext {
		var name string
		arg(c, "ctx", &name)
		if ctxs[name] == nil {
			var req string
			arg(c, "req", &req)
			ctxs[name] = &tiering.ReqContext{ReqID: req, Filter: tiering.AllTiers, Primary: offloading.NewAccess()}
		}
		r := ctxs[name]
		if _, ok := c.Args["positions"]; ok {
			var pos map[string]int
			arg(c, "positions", &pos)
			for h, p := range pos {
				r.Primary.(*offloading.Access).SetPosition(offloading.Key(key(h)), p)
			}
		}
		return r
	}
	hexes := func(ks []tiering.Key) []string {
		out := make([]string, len(ks))
		for i, k := range ks {
			out[i] = hex.EncodeToString([]byte(k))
		}
		return out
	}
	compared := 0
	for i, c := range tr.Manager {
		var got any
		switch c.M {
		case "on_new_request":
			got = map[string]string{"policy": m.OnNewRequest(ctx(c)).String()}
		case "lookup":
			var h string
			arg(c, "key", &h)
			got = m.Lookup(key(h), ctx(c)).String()
		case "get_load_source":
			var h string
			arg(c, "key", &h)
			got = m.LoadSource(key(h), ctx(c))
		case "prepare_load":
			got = m.PrepareLoad(keys(c), ctx(c))
		case "touch":
			m.Touch(keys(c), ctx(c))
		case "complete_load":
			m.CompleteLoad(keys(c), ctx(c))
		case "prepare_store":
			stored, ids, evicted, ok := m.PrepareStore(keys(c), ctx(c))
			if ok {
				got = map[string]any{"keys_to_store": hexes(stored), "ids": append([]int{}, ids...), "evicted": hexes(evicted)}
			}
		case "complete_store":
			var ok bool
			arg(c, "success", &ok)
			m.CompleteStore(keys(c), ctx(c), ok)
		case "on_request_finished":
			m.OnRequestFinished(ctx(c))
		case "on_schedule_end":
			var e struct {
				New       []string `json:"new_req_ids"`
				Preempted []string `json:"preempted_req_ids"`
			}
			b, _ := json.Marshal(c.Args)
			if err := json.Unmarshal(b, &e); err != nil {
				t.Fatal(err)
			}
			m.OnScheduleEnd(tiering.ScheduleEnd{NewReqIDs: e.New, PreemptedReqIDs: e.Preempted})
		case "has_pending_work":
			got = m.HasPendingWork()
		case "take_events":
			// The port's CPU tier publishes no events (https://github.com/inference-sim/blis-vllm/issues/60),
			// and no recorded sequence makes vLLM's publish any.
			if ev := m.TakeEvents(); len(ev) > 0 {
				t.Fatalf("call %d take_events: the port reports %d events", i, len(ev))
			}
			got = []any{}
		case "reset_cache":
			m.ResetCache()
		case "on_reset_flushed":
			m.OnResetFlushed()
		case "get_stats":
			// A nil payload is vLLM's None.
			if st := m.Stats(); st != nil {
				got = st.Values()
			}
		case "config_info":
			got = m.ConfigInfo()
		case "shutdown":
			// The port holds no thread or file to release.
		default:
			t.Fatalf("call %d: the replay does not drive %q", i, c.M)
		}
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
			t.Fatalf("call %d %s: the port answers %s, vLLM %s", i, c.M, g, c.Ret)
		}
		compared++
	}
	if compared == 0 {
		t.Fatal("no manager call was compared")
	}
}

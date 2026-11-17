// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Original to blis-vllm.

package tiering_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/core/connector/offloading"
	"github.com/inference-sim/blis-vllm/core/connector/tiering"
)

// The replay: each sequence of oracle/scenarios/tiering is run on the port as oracle/run_tiering.py
// ran it on vLLM v0.33.0, and every call at the OffloadingManager interface and at each tier's
// SecondaryTierManager interface must give vLLM's recorded answer. The transport follows the
// oracle's rule: after each call every submitted job runs (loads, then stores, each in submission
// order) and every flushed lookup batch resolves.

const (
	scenarioDir = "../../../oracle/scenarios/tiering"
	goldenDir   = "testdata/golden"
)

// record is one call at an interface, as the trace holds it.
type record struct {
	M    string         `json:"m"`
	Args map[string]any `json:"args"`
	Ret  any            `json:"ret"`
}

func hexKey(k tiering.Key) string { return hex.EncodeToString([]byte(k)) }

func hexKeys(ks []tiering.Key) []string {
	out := make([]string, len(ks))
	for i, k := range ks {
		out[i] = hexKey(k)
	}
	return out
}

func eventsJSON(ev []tiering.Event) []any {
	out := []any{}
	for _, e := range ev {
		var loc any
		if e.Locality != "" {
			loc = string(e.Locality)
		}
		// metadata_only is set only by the CPU tier's failed allocations (vllm/v1/kv_offload/cpu/manager.py:408),
		// whose events are not compared (withoutCPUEvents): a secondary tier's event never sets it.
		out = append(out, map[string]any{"keys": hexKeys(e.Keys), "medium": string(e.Medium), "removed": e.Removed, "locality": loc,
			"metadata_only": false})
	}
	return out
}

// recTier records each call the manager makes on a tier, as run_tiering.py's monitor does.
type recTier struct {
	tiering.Tier
	idx int
	log *[]record
}

func (t *recTier) add(m string, args map[string]any, ret any) {
	if args == nil {
		args = map[string]any{}
	}
	args["tier"] = t.idx
	*t.log = append(*t.log, record{M: m, Args: args, Ret: ret})
}

func jobJSON(j *tiering.TransferJob) map[string]any {
	return map[string]any{"id": j.ID, "keys": hexKeys(j.Keys), "chunk_ids": j.ChunkIDs, "promotion": j.IsPromotion}
}

func (t *recTier) Lookup(k tiering.Key, c *tiering.ReqContext) tiering.Lookup {
	r := t.Tier.Lookup(k, c)
	t.add("lookup", map[string]any{"key": hexKey(k)}, r.String())
	return r
}
func (t *recTier) SubmitStore(j *tiering.TransferJob) {
	t.Tier.SubmitStore(j)
	t.add("submit_store", map[string]any{"job": jobJSON(j)}, nil)
}
func (t *recTier) SubmitLoad(j *tiering.TransferJob) {
	t.Tier.SubmitLoad(j)
	t.add("submit_load", map[string]any{"job": jobJSON(j)}, nil)
}
func (t *recTier) FinishedJobs() []tiering.JobResult {
	rs := t.Tier.FinishedJobs()
	out := []any{}
	for _, r := range rs {
		var ok, bytes any
		if r.SuccessfulKeys != nil {
			ok = hexKeys(r.SuccessfulKeys)
		}
		if r.TransferBytes != 0 {
			bytes = r.TransferBytes
		}
		out = append(out, map[string]any{"id": r.ID, "success": r.Success, "successful_keys": ok, "transfer_bytes": bytes})
	}
	t.add("get_finished_jobs", nil, out)
	return rs
}
func (t *recTier) HasPendingWork() bool {
	r := t.Tier.HasPendingWork()
	t.add("has_pending_work", nil, r)
	return r
}
func (t *recTier) TakeEvents() []tiering.Event {
	ev := t.Tier.TakeEvents()
	t.add("take_events", nil, eventsJSON(ev))
	return ev
}
func (t *recTier) Touch(ks []tiering.Key, c *tiering.ReqContext) {
	t.Tier.Touch(ks, c)
	t.add("touch", map[string]any{"keys": hexKeys(ks)}, nil)
}
func (t *recTier) OnNewRequest(c *tiering.ReqContext) tiering.Policy {
	p := t.Tier.OnNewRequest(c)
	t.add("on_new_request", nil, map[string]any{"policy": p.String()})
	return p
}
func (t *recTier) OnRequestFinished(c *tiering.ReqContext) {
	t.Tier.OnRequestFinished(c)
	t.add("on_request_finished", nil, nil)
}
func (t *recTier) ServeExternal(p tiering.Parent) {
	t.Tier.ServeExternal(p)
	t.add("serve_external_requests", nil, nil)
}
func (t *recTier) OnScheduleEnd(e tiering.ScheduleEnd) {
	t.Tier.OnScheduleEnd(e)
	t.add("on_schedule_end", map[string]any{"new": sorted(e.NewReqIDs), "preempted": sorted(e.PreemptedReqIDs)}, nil)
}
func (t *recTier) DrainJobs() {
	t.Tier.DrainJobs()
	t.add("drain_jobs", nil, nil)
}

func sorted(s []string) []string {
	out := append([]string{}, s...)
	sort.Strings(out)
	return out
}

// keyOf is the oracle's key label: sha256 of the name, then the group as 4 big-endian bytes.
func keyOf(label string) tiering.Key {
	name, group, _ := strings.Cut(label, "/")
	g := 0
	if group != "" {
		for _, ch := range group {
			g = g*10 + int(ch-'0')
		}
	}
	h := sha256.Sum256([]byte(name))
	return tiering.MakeKey(h[:], g)
}

type call struct {
	M         string           `json:"m"`
	Ctx       string           `json:"ctx"`
	Key       string           `json:"key"`
	Keys      []string         `json:"keys"`
	Success   *bool            `json:"success"`
	New       []string         `json:"new"`
	Preempted []string         `json:"preempted"`
	Tier      int              `json:"tier"`
	Filter    []map[string]any `json:"filter"`
}

type traceTier struct {
	Type       string         `json:"type"`
	BlockBytes int64          `json:"block_size_bytes"`
	RunConfig  map[string]any `json:"run_config"`
	Rank       int            `json:"rank"`
	Locality   *string        `json:"locality"`
	Events     bool           `json:"events"`
}

type trace struct {
	Spec []struct {
		Ret struct {
			NumChunks int `json:"num_chunks"`
		} `json:"ret"`
	} `json:"spec"`
	Tiers   []traceTier         `json:"tiers"`
	Files   map[string][]string `json:"files"`
	Manager []record            `json:"manager"`
	Tier    []record            `json:"tier"`
}

func namespace(t traceTier) tiering.Namespace {
	rc := t.RunConfig
	num := func(k string) int { return int(rc[k].(float64)) }
	n := tiering.Namespace{ModelName: rc["model_name"].(string), TokensPerHash: num("tokens_per_hash"),
		BlocksPerFile: num("blocks_per_file"), TPSize: num("tp_size"), PPSize: num("pp_size"), PCPSize: num("pcp_size"),
		DCPSize: num("dcp_size"), Rank: t.Rank, DType: rc["dtype"].(string)}
	for _, g := range rc["kv_cache_groups"].([]any) {
		gm := g.(map[string]any)
		var names []string
		for _, l := range gm["layer_names"].([]any) {
			names = append(names, l.(string))
		}
		n.Groups = append(n.Groups, tiering.KVGroupName{TokensPerBlock: int(gm["tokens_per_block"].(float64)), LayerNames: names})
	}
	// The recorded fields state the layout: parallel_agnostic is written only when False, and
	// replicated_layout only when True (file_mapper.py:63-68@4ba787ec).
	_, notAgnostic := rc["parallel_agnostic"]
	n.ParallelAgnostic = !notAgnostic
	_, n.Replicated = rc["replicated_layout"]
	return n
}

func filterOf(raw []map[string]any) tiering.Filter {
	if raw == nil {
		return tiering.AllTiers
	}
	f := tiering.Filter{Matchers: []tiering.Matcher{}}
	for _, e := range raw {
		var m tiering.Matcher
		if v, ok := e["medium"].(string); ok {
			m.Medium = tiering.Medium(strings.ToUpper(v))
		}
		if v, ok := e["locality"].(string); ok {
			m.Locality = tiering.Locality(strings.ToUpper(v))
		}
		f.Matchers = append(f.Matchers, m)
	}
	return f
}

// normal is a value as JSON decodes it, so a Go answer and a recorded one compare alike.
func normal(t *testing.T, v any) any {
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestTieringTracesReplayEqual(t *testing.T) {
	scenarios, err := filepath.Glob(filepath.Join(scenarioDir, "*.json"))
	if err != nil || len(scenarios) == 0 {
		t.Fatalf("no tiering scenarios in %s (%v)", scenarioDir, err)
	}
	for _, sp := range scenarios {
		name := strings.TrimSuffix(filepath.Base(sp), ".json")
		t.Run(name, func(t *testing.T) { replay(t, sp, filepath.Join(goldenDir, name+".tiering.json")) })
	}
}

func replay(t *testing.T, scenarioPath, tracePath string) {
	var sq struct {
		Calls []call `json:"calls"`
	}
	var tr trace
	for p, v := range map[string]any{scenarioPath: &sq, tracePath: &tr} {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("read %s: %v (record it with oracle/run_tiering.py)", p, err)
		}
		if err := json.Unmarshal(b, v); err != nil {
			t.Fatalf("%s: %v", p, err)
		}
	}
	if len(tr.Spec) != 1 {
		t.Fatalf("the trace records %d spec builds, want 1", len(tr.Spec))
	}
	var tierLog []record
	files := tiering.MemFiles{}
	var fsTiers []*tiering.FSTier
	var tiers []tiering.Tier
	for i, tt := range tr.Tiers {
		if tt.Type != "fs" {
			t.Fatalf("tier %d is %q; this replay builds fs tiers only", i, tt.Type)
		}
		loc := tiering.LocalityNone
		if tt.Locality != nil {
			loc = tiering.Locality(*tt.Locality)
		}
		// The recorded file names are relative to the sequence's directory, under which each tier's
		// root_dir sits: the tier's root is what precedes its model-named directory.
		root := ""
		safe := strings.ReplaceAll(tt.RunConfig["model_name"].(string), "/", "_") + "_"
		for _, fs := range tr.Files {
			if at := strings.Index(fs[i], "/"+safe); at >= 0 && !strings.HasPrefix(fs[i], safe) {
				root = fs[i][:at]
			}
			break
		}
		ft := tiering.NewFSTier(tiering.FSConfig{RootDir: root, Locality: loc, Events: tt.Events, BlockBytes: tt.BlockBytes,
			Namespace: namespace(tt)}, files)
		fsTiers = append(fsTiers, ft)
		tiers = append(tiers, &recTier{Tier: ft, idx: i, log: &tierLog})
	}
	mgr := tiering.New(tiering.CPU(offloading.NewCPUPrimary(tr.Spec[0].Ret.NumChunks)), tiers, nil)
	for label, want := range tr.Files {
		for i, ft := range fsTiers {
			got := strings.TrimPrefix(ft.FileName(keyOf(label)), "/")
			if got != want[i] {
				t.Errorf("tier %d's block file of %s is %q, vLLM's %q", i, label, got, want[i])
			}
		}
	}
	ctxs := map[string]*tiering.ReqContext{}
	ctx := func(c call) *tiering.ReqContext {
		if ctxs[c.Ctx] == nil {
			r := tiering.NewReqContext(c.Ctx)
			r.Filter = filterOf(c.Filter)
			ctxs[c.Ctx] = r
		}
		return ctxs[c.Ctx]
	}
	keys := func(c call) []tiering.Key {
		out := make([]tiering.Key, len(c.Keys))
		for i, k := range c.Keys {
			out[i] = keyOf(k)
		}
		return out
	}
	var manager []record
	for _, c := range sq.Calls {
		var ret any
		recorded := true
		switch c.M {
		case "rm":
			files.Remove(fsTiers[c.Tier].FileName(keyOf(c.Key)))
			recorded = false
		case "put":
			files.Publish(fsTiers[c.Tier].FileName(keyOf(c.Key)))
			recorded = false
		case "on_reset_flushed":
			mgr.OnResetFlushed()
		case "on_new_request":
			ret = map[string]any{"policy": mgr.OnNewRequest(ctx(c)).String()}
		case "lookup":
			ret = mgr.Lookup(keyOf(c.Key), ctx(c)).String()
		case "get_load_source":
			ret = mgr.LoadSource(keyOf(c.Key), ctx(c))
		case "prepare_load":
			ret = mgr.PrepareLoad(keys(c), ctx(c))
		case "touch":
			mgr.Touch(keys(c), ctx(c))
		case "complete_load":
			mgr.CompleteLoad(keys(c), ctx(c))
		case "prepare_store":
			stored, ids, ev, ok := mgr.PrepareStore(keys(c), ctx(c))
			if ok {
				ret = map[string]any{"keys_to_store": hexKeys(stored), "ids": ids, "evicted": hexKeys(ev)}
			}
		case "complete_store":
			mgr.CompleteStore(keys(c), ctx(c), c.Success == nil || *c.Success)
		case "on_request_finished":
			mgr.OnRequestFinished(ctx(c))
		case "on_schedule_end":
			mgr.OnScheduleEnd(tiering.ScheduleEnd{NewReqIDs: c.New, PreemptedReqIDs: c.Preempted})
		case "has_pending_work":
			ret = mgr.HasPendingWork()
		case "take_events":
			ret = eventsJSON(mgr.TakeEvents())
			// The trace's monitor records the primary tier's take_events generator, which the manager's
			// iterates, as a call of its own (kv_interfaces.py starts a record for every generator that
			// is not inside another generator's record). Its answer is the CPU tier's events.
			manager = append(manager, record{M: c.M, Ret: ret})
			ret = []any{}
		case "reset_cache":
			mgr.ResetCache()
		default:
			t.Fatalf("the replay does not drive %q", c.M)
		}
		if recorded {
			manager = append(manager, record{M: c.M, Ret: ret})
		}
		for _, ft := range fsTiers {
			ft.RunAll()
			ft.ResolveLookups()
		}
	}
	// The driver's shutdown is the trace's last manager call; the port holds nothing to release.
	want := tr.Manager
	if n := len(want); n > 0 && want[n-1].M == "shutdown" {
		want = want[:n-1]
	}
	for i := range want {
		if want[i].M == "take_events" {
			want[i].Ret = withoutCPUEvents(want[i].Ret)
		}
	}
	compare(t, "manager", manager, want, false)
	compare(t, "tier", tierLog, tr.Tier, true)
}

// withoutCPUEvents drops the CPU tier's own BlockStored and BlockRemoved events from a recorded
// take_events answer. The port's CPU tier emits none: KV events from the offloading connector are
// https://github.com/inference-sim/blis-vllm/issues/60. The secondary tiers' events are compared.
func withoutCPUEvents(v any) any {
	evs, ok := v.([]any)
	if !ok {
		return v
	}
	out := []any{}
	for _, e := range evs {
		if m, ok := e.(map[string]any); ok && m["medium"] == "CPU" {
			continue
		}
		out = append(out, e)
	}
	return out
}

// stripContext removes the request context's labels, which name the same requests in both traces.
func stripContext(v any) any {
	switch x := v.(type) {
	case map[string]any:
		out := map[string]any{}
		for k, e := range x {
			if k != "ctx" && k != "req" {
				out[k] = stripContext(e)
			}
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			out[i] = stripContext(e)
		}
		return out
	}
	return v
}

// compare checks the calls in order: each method and answer, and for a tier its arguments too
// (without the request context's labels, which name the same requests in both).
func compare(t *testing.T, iface string, got, want []record, args bool) {
	t.Helper()
	for i := 0; i < len(got) && i < len(want); i++ {
		g, w := got[i], want[i]
		if g.M != w.M {
			t.Fatalf("%s call %d is %s, vLLM's %s", iface, i, g.M, w.M)
		}
		if !reflect.DeepEqual(normal(t, g.Ret), normal(t, w.Ret)) {
			t.Fatalf("%s call %d (%s) answered %v, vLLM %v", iface, i, g.M, normal(t, g.Ret), normal(t, w.Ret))
		}
		if args {
			wa := stripContext(normal(t, w.Args))
			if ga := normal(t, g.Args); !reflect.DeepEqual(ga, wa) {
				t.Fatalf("%s call %d (%s) was given %v, vLLM's %v", iface, i, g.M, ga, wa)
			}
		}
	}
	if len(got) != len(want) {
		t.Fatalf("%s: %d calls, vLLM made %d", iface, len(got), len(want))
	}
}

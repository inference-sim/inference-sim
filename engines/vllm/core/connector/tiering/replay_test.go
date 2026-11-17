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
// order) and every flushed lookup batch resolves. A sequence of several instances (p2p) builds one
// manager per instance over one P2PNetwork; after each call every WRITE ends, every control
// message is delivered and every peer registration resolves, unless the sequence holds them.

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
	// poll, for a tier that serves peers, records its parent calls made inside poll_tiers.
	poll *pollRec
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
	if t.poll != nil {
		p = &recParent{Parent: p, poll: t.poll}
	}
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
	// Inst is the instance a call drives (0 by default), Params a new request's
	// kv_transfer_params, and S an advance's seconds.
	Inst   int            `json:"inst"`
	Params map[string]any `json:"params"`
	S      float64        `json:"s"`
}

type traceTier struct {
	Type       string         `json:"type"`
	BlockBytes int64          `json:"block_size_bytes"`
	RunConfig  map[string]any `json:"run_config"`
	Rank       int            `json:"rank"`
	Locality   *string        `json:"locality"`
	Events     bool           `json:"events"`
	// Bucket is an obj tier's store_config bucket; obj tiers naming one bucket share it.
	Bucket string `json:"bucket"`
	// A p2p tier's control identity, primary region, config fingerprint and unbound-store timeout.
	LocalID              string  `json:"local_id"`
	NumBlocks            int     `json:"num_blocks"`
	Fingerprint          string  `json:"fingerprint"`
	UnboundStoreTimeoutS float64 `json:"unbound_store_timeout_s"`
}

type trace struct {
	Spec []struct {
		Args struct {
			SecondaryTiers int `json:"secondary_tiers"`
		} `json:"args"`
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

// pollRec records the manager calls a poll_tiers round makes. poll_tiers is not a recorded call,
// so the trace's monitor records each OffloadingManager call made inside it that is not nested in
// another (kv_interfaces.py starts a record for every interface call outside one): the primary
// tier's complete_load and complete_store as finished jobs end, and the manager's own
// on_new_request, lookup and on_request_finished that a tier serving a peer makes through its
// parent.
type pollRec struct {
	on    bool
	depth int
	log   *[]record
}

// enter starts a call; it is recorded when it is a top-level call inside a poll_tiers round.
func (p *pollRec) enter() bool {
	p.depth++
	return p.on && p.depth == 1
}

func (p *pollRec) leave(top bool, m string, ret any) {
	p.depth--
	if top {
		*p.log = append(*p.log, record{M: m, Ret: ret})
	}
}

// recParent is a tier's parent handle as the monitor sees it during poll_tiers.
type recParent struct {
	tiering.Parent
	poll *pollRec
}

func (p *recParent) OnNewRequest(c *tiering.ReqContext) tiering.Policy {
	top := p.poll.enter()
	r := p.Parent.OnNewRequest(c)
	p.poll.leave(top, "on_new_request", map[string]any{"policy": r.String()})
	return r
}

func (p *recParent) Lookup(k tiering.Key, c *tiering.ReqContext) tiering.Lookup {
	top := p.poll.enter()
	r := p.Parent.Lookup(k, c)
	p.poll.leave(top, "lookup", r.String())
	return r
}

// CreateStoreJob is no interface method; the primary tier's calls inside it are nested.
func (p *recParent) CreateStoreJob(keys []tiering.Key, c *tiering.ReqContext) *tiering.TransferJob {
	p.poll.depth++
	defer func() { p.poll.depth-- }()
	return p.Parent.CreateStoreJob(keys, c)
}

func (p *recParent) OnRequestFinished(c *tiering.ReqContext) {
	top := p.poll.enter()
	p.Parent.OnRequestFinished(c)
	p.poll.leave(top, "on_request_finished", nil)
}

// recPrimary is the primary tier as the monitor sees it during poll_tiers: its interface calls
// there are recorded (PrepareRead is _prepare_load, no interface method).
type recPrimary struct {
	tiering.Primary
	poll *pollRec
}

func (p recPrimary) Lookup(k tiering.Key, c *tiering.ReqContext) tiering.Lookup {
	top := p.poll.enter()
	r := p.Primary.Lookup(k, c)
	p.poll.leave(top, "lookup", r.String())
	return r
}

func (p recPrimary) PrepareRead(ks []tiering.Key, c *tiering.ReqContext) []int {
	p.poll.depth++
	defer func() { p.poll.depth-- }()
	return p.Primary.PrepareRead(ks, c)
}

func (p recPrimary) PrepareStore(ks []tiering.Key, c *tiering.ReqContext) ([]tiering.Key, []int, []tiering.Key, bool) {
	top := p.poll.enter()
	s, ids, ev, ok := p.Primary.PrepareStore(ks, c)
	var ret any
	if ok {
		ret = map[string]any{"keys_to_store": hexKeys(s), "ids": ids, "evicted": hexKeys(ev)}
	}
	p.poll.leave(top, "prepare_store", ret)
	return s, ids, ev, ok
}

func (p recPrimary) CompleteLoad(ks []tiering.Key, c *tiering.ReqContext) {
	top := p.poll.enter()
	p.Primary.CompleteLoad(ks, c)
	p.poll.leave(top, "complete_load", nil)
}

func (p recPrimary) CompleteStore(ks []tiering.Key, c *tiering.ReqContext, ok bool) {
	top := p.poll.enter()
	p.Primary.CompleteStore(ks, c, ok)
	p.poll.leave(top, "complete_store", nil)
}

func (p recPrimary) OnRequestFinished(c *tiering.ReqContext) {
	top := p.poll.enter()
	p.Primary.OnRequestFinished(c)
	p.poll.leave(top, "on_request_finished", nil)
}

// p2pEnv is a sequence's VLLM_P2P_* environment, as P2PConfig takes it.
func p2pEnv(cfg *tiering.P2PConfig, env map[string]any) {
	num := func(k string) (int, bool) {
		v, ok := env[k]
		if !ok {
			return 0, false
		}
		switch x := v.(type) {
		case float64:
			return int(x), true
		case string:
			n := 0
			for _, ch := range x {
				n = n*10 + int(ch-'0')
			}
			return n, true
		}
		return 0, false
	}
	if n, ok := num("VLLM_P2P_MAX_PEERS"); ok {
		cfg.MaxPeers = n
	}
	if n, ok := num("VLLM_P2P_HANDSHAKE_TIMEOUT_S"); ok {
		cfg.HandshakeTimeoutS = n
	}
	if n, ok := num("VLLM_P2P_IDLE_TIMEOUT_S"); ok {
		cfg.IdleTimeoutS, cfg.IdleReapOff = n, n <= 0
	}
}

func replay(t *testing.T, scenarioPath, tracePath string) {
	var sq struct {
		Calls []call         `json:"calls"`
		Env   map[string]any `json:"env"`
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
	if len(tr.Spec) == 0 {
		t.Fatalf("the trace records no spec build")
	}
	// Each instance has its own manager; the trace's tier indices are global, instance 0's tiers
	// first, as each spec build lists its tier count.
	instOf := make([]int, len(tr.Tiers))
	if len(tr.Spec) > 1 {
		at := 0
		for i, sp := range tr.Spec {
			for j := 0; j < sp.Args.SecondaryTiers; j++ {
				if at >= len(instOf) {
					t.Fatalf("the spec builds name more tiers than the trace's %d", len(tr.Tiers))
				}
				instOf[at] = i
				at++
			}
		}
		if at != len(tr.Tiers) {
			t.Fatalf("the spec builds name %d tiers, the trace %d", at, len(tr.Tiers))
		}
	}
	// The virtual clock vLLM's time.monotonic read (run_tiering.py VirtualClock).
	now := 1000.0
	clock := func() float64 { return now }
	net := tiering.NewP2PNetwork()
	hasP2P := false
	var manager []record
	poll := &pollRec{log: &manager}
	var tierLog []record
	files := tiering.MemFiles{}
	var fsTiers []*tiering.FSTier
	tiers := make([][]tiering.Tier, len(tr.Spec))
	// Each tier's built value by trace index: an fs tier's at fsAt, an obj tier's at objAt, a p2p
	// tier's local id at p2pAt.
	fsAt, objAt, p2pAt := map[int]*tiering.FSTier{}, map[int]*tiering.ObjTier{}, map[int]string{}
	var objTiers []*tiering.ObjTier
	buckets := map[string]*tiering.MemBucket{}
	// held are the obj tiers whose transfers the driver holds in flight across calls.
	held := map[*tiering.ObjTier]bool{}
	for i, tt := range tr.Tiers {
		loc := tiering.LocalityNone
		if tt.Locality != nil {
			loc = tiering.Locality(*tt.Locality)
		}
		in := instOf[i]
		switch tt.Type {
		case "obj":
			ot := buildObj(t, i, tt, loc, tr.Files, buckets)
			objAt[i] = ot
			objTiers = append(objTiers, ot)
			tiers[in] = append(tiers[in], &recTier{Tier: ot, idx: i, log: &tierLog})
			continue
		case "p2p":
			hasP2P = true
			cfg := tiering.P2PConfig{Now: clock, LocalID: tt.LocalID, BlockBytes: tt.BlockBytes,
				UnboundStoreTimeoutS: tt.UnboundStoreTimeoutS}
			p2pEnv(&cfg, sq.Env)
			pt := tiering.NewP2PTier(cfg, net.Control(tt.LocalID), net.Data(tt.LocalID, tt.NumBlocks, tt.BlockBytes, tt.Fingerprint))
			p2pAt[i] = tt.LocalID
			tiers[in] = append(tiers[in], &recTier{Tier: pt, idx: i, log: &tierLog, poll: poll})
			continue
		case "fs":
		default:
			t.Fatalf("tier %d is %q; this replay builds fs, obj and p2p tiers only", i, tt.Type)
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
		fsAt[i] = ft
		tiers[in] = append(tiers[in], &recTier{Tier: ft, idx: i, log: &tierLog})
	}
	mgrs := make([]*tiering.Manager, len(tr.Spec))
	for i, sp := range tr.Spec {
		primary := tiering.CPU(offloading.NewCPUPrimary(sp.Ret.NumChunks))
		var mclock func() float64
		if hasP2P {
			primary, mclock = recPrimary{Primary: primary, poll: poll}, clock
		}
		mgrs[i] = tiering.New(primary, tiers[i], mclock)
	}
	for label, want := range tr.Files {
		for i, ot := range objAt {
			if got := ot.ObjectName(keyOf(label)); got != want[i] {
				t.Errorf("tier %d's object key of %s is %q, vLLM's %q", i, label, got, want[i])
			}
		}
		for i, ft := range fsAt {
			got := strings.TrimPrefix(ft.FileName(keyOf(label)), "/")
			if got != want[i] {
				t.Errorf("tier %d's block file of %s is %q, vLLM's %q", i, label, got, want[i])
			}
		}
	}
	// A request's context is per instance and label; its kv_transfer_params are its first call's.
	type ctxKey struct {
		inst  int
		label string
	}
	ctxs := map[ctxKey]*tiering.ReqContext{}
	ctx := func(c call) *tiering.ReqContext {
		k := ctxKey{c.Inst, c.Ctx}
		if ctxs[k] == nil {
			r := tiering.NewReqContext(c.Ctx)
			r.Filter = filterOf(c.Filter)
			r.Params = c.Params
			ctxs[k] = r
		}
		return ctxs[k]
	}
	keys := func(c call) []tiering.Key {
		out := make([]tiering.Key, len(c.Keys))
		for i, k := range c.Keys {
			out[i] = keyOf(k)
		}
		return out
	}
	for _, c := range sq.Calls {
		if c.Inst < 0 || c.Inst >= len(mgrs) {
			t.Fatalf("a call names instance %d of %d", c.Inst, len(mgrs))
		}
		mgr := mgrs[c.Inst]
		var ret any
		recorded := true
		switch c.M {
		case "rm", "put", "deny", "allow", "refuse_next", "hold", "release", "nixl_down", "down":
			recorded = false
			if id, ok := p2pAt[c.Tier]; ok {
				switch c.M {
				case "hold", "release":
					net.HoldWrites(id, c.M == "hold")
				case "nixl_down":
					net.DataDown(id)
				case "down":
					net.Down(id)
				default:
					t.Fatalf("the replay does not drive %q on a p2p tier", c.M)
				}
			} else if ot := objAt[c.Tier]; ot != nil {
				objCall(c, ot, buckets[tr.Tiers[c.Tier].Bucket], held)
			} else if c.M == "rm" {
				files.Remove(fsAt[c.Tier].FileName(keyOf(c.Key)))
			} else {
				files.Publish(fsAt[c.Tier].FileName(keyOf(c.Key)))
			}
		case "advance":
			if c.S < 0 {
				t.Fatalf("the clock does not run backwards")
			}
			now += c.S
			recorded = false
		case "poll_tiers":
			poll.on = true
			mgr.PollTiers()
			poll.on = false
			recorded = false
		case "hold_net", "release_net":
			net.Hold(c.M == "hold_net")
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
		// The obj tiers' transport rule: every transfer in flight ends, in submission order, unless
		// the driver holds the tier's transfers.
		for _, ot := range objTiers {
			if !held[ot] {
				ot.RunAll()
			}
			ot.ResolveLookups()
		}
		// The p2p tiers' (run_tiering.py wait_p2p): every WRITE ends unless its agent is held, every
		// control message is delivered unless the wire is held, and every registration resolves.
		if hasP2P {
			net.RunAll()
		}
	}
	// The driver's shutdown of each instance ends the trace; the port holds nothing to release.
	want := tr.Manager
	for n := len(tr.Spec); n > 0 && len(want) > 0 && want[len(want)-1].M == "shutdown"; n-- {
		want = want[:len(want)-1]
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

// buildObj builds trace tier i, an obj tier, over the bucket its header names. Its prefix is what
// precedes "/<model>_" in a recorded object key (FileMapper's base path is "<prefix>/" + "/" + ...).
func buildObj(t *testing.T, i int, tt traceTier, loc tiering.Locality, files map[string][]string,
	buckets map[string]*tiering.MemBucket) *tiering.ObjTier {
	t.Helper()
	if tt.Bucket == "" {
		t.Fatalf("obj tier %d names no bucket", i)
	}
	if buckets[tt.Bucket] == nil {
		buckets[tt.Bucket] = tiering.NewMemBucket()
	}
	prefix := ""
	safe := "/" + strings.ReplaceAll(tt.RunConfig["model_name"].(string), "/", "_") + "_"
	for _, fs := range files {
		if at := strings.Index(fs[i], safe); at > 0 {
			prefix = strings.TrimSuffix(fs[i][:at], "/")
		}
		break
	}
	return tiering.NewObjTier(tiering.ObjConfig{Prefix: prefix, Locality: loc, Events: tt.Events,
		BlockBytes: tt.BlockBytes, Namespace: namespace(tt)}, buckets[tt.Bucket])
}

// objCall is a driver call on an obj tier's bucket or transport, as run_tiering.py makes it on the
// in-memory NIXL agent.
func objCall(c call, ot *tiering.ObjTier, b *tiering.MemBucket, held map[*tiering.ObjTier]bool) {
	switch c.M {
	case "rm":
		b.Remove(ot.ObjectName(keyOf(c.Key)))
	case "put":
		b.Put(ot.ObjectName(keyOf(c.Key)))
	case "deny", "allow":
		b.Denied = c.M == "deny"
	case "refuse_next":
		ot.RefuseNextSubmission()
	case "hold", "release":
		held[ot] = c.M == "hold"
	}
}

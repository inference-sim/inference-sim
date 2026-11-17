// Original to blis-vllm.

package offloading

import (
	"slices"
	"testing"

	"github.com/inference-sim/blis-vllm/core/kv"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// The connector protocol of vLLM v0.33.0 (offloading/scheduler.py@4ba787ec) as a manager other
// than the CPU tier answers it: RETRY, on_new_request's policy, on_schedule_end,
// has_pending_work and get_load_source. scripted is the CPU tier with the answers a test sets.

type scripted struct {
	cpuTier
	retry   map[Key]bool // keys whose lookup the manager has not resolved
	asked   []Key        // every key looked up, in order
	policy  Policy
	pending bool
	source  string // get_load_source's answer; "" is the CPU tier's
	calls   []string
	ends    []ScheduleEnd
	stores  [][]Key // every prepare_store's keys
}

func (m *scripted) OnNewRequest(a *Access) Policy {
	m.calls = append(m.calls, "on_new_request:"+a.ReqID())
	m.cpuTier.OnNewRequest(a)
	return m.policy
}

func (m *scripted) Lookup(k Key, a *Access) LookupResult {
	m.asked = append(m.asked, k)
	if m.retry[k] {
		return Retry
	}
	return m.cpuTier.Lookup(k, a)
}

func (m *scripted) LoadSource(k Key, a *Access) string {
	if m.source != "" {
		return m.source
	}
	return m.cpuTier.LoadSource(k, a)
}

func (m *scripted) PrepareStore(keys []Key, a *Access) ([]Key, []int, []Key, bool) {
	m.stores = append(m.stores, append([]Key(nil), keys...))
	return m.cpuTier.PrepareStore(keys, a)
}

func (m *scripted) OnRequestFinished(a *Access) {
	m.calls = append(m.calls, "on_request_finished:"+a.ReqID())
	m.cpuTier.OnRequestFinished(a)
}

func (m *scripted) OnScheduleEnd(e ScheduleEnd) {
	m.calls = append(m.calls, "on_schedule_end")
	m.ends = append(m.ends, e)
}

func (m *scripted) HasPendingWork() bool { return m.pending }

const protoBlock = 16

// protoSetup is a scheduler of 16-token blocks with prefix caching and the connector on a
// scripted manager over a 64-chunk CPU tier, one block a chunk. window > 0 adds a sliding-window
// group of that many tokens after the full-attention one.
func protoSetup(t *testing.T, window int) (*sched.Scheduler, *Scheduler, *scripted) {
	t.Helper()
	layers := []kv.Group{{Kind: kv.FullAttention}}
	groups := []Group{{BlockSize: protoBlock}}
	if window > 0 {
		layers = append(layers, kv.Group{Kind: kv.SlidingWindow, Window: window})
		groups = append(groups, Group{BlockSize: protoBlock, Window: window})
	}
	s, err := sched.New(sched.Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 512, BlockSize: protoBlock, NumBlocks: 64,
		MaxModelLen: 1024, EnableChunkedPrefill: true, EnablePrefixCaching: true, ReserveFullSequence: true,
		KVGroups: kv.ResolveGroups(layers, false), MaxConcurrentBatches: 1})
	if err != nil {
		t.Fatal(err)
	}
	m := &scripted{retry: map[Key]bool{}}
	conn, err := New(Config{Groups: groups, TokensPerHash: s.KV.HashBlockSize(), NumChunks: 64, OffloadPromptOnly: true,
		PrefixCaching: true, Workers: 1, Manager: func(cpu *CPUPrimary) (Manager, error) {
			m.cpuTier = cpuTier{cpu.m}
			return m, nil
		}})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetConnector(conn, sched.ConnectorConfig{Consumer: true, Producer: true}); err != nil {
		t.Fatal(err)
	}
	return s, conn, m
}

func prompt(n int, base int64) []int64 {
	p := make([]int64, n)
	for i := range p {
		p[i] = base + int64(i)
	}
	return p
}

// add adds a request of the prompt that samples one token and stops.
func add(t *testing.T, s *sched.Scheduler, id string, p []int64) *sched.Request {
	t.Helper()
	if err := s.Add(&sched.Request{ID: id, Prompt: p, MaxTokens: 1, EOS: -1}); err != nil {
		t.Fatal(err)
	}
	return s.Request(id)
}

// offloaded stores the request's first n chunks of group 0 in the CPU tier, as a finished
// request's stores would have.
func offloaded(t *testing.T, conn *Scheduler, r *sched.Request, n int) []Key {
	t.Helper()
	st := conn.reqs[r.ID]
	conn.updateKeys(st)
	keys := st.groups[0].keys[:n]
	if _, _, _, ok := conn.cpu.prepareStore(keys, st.access.a); !ok {
		t.Fatal("the CPU tier has no room")
	}
	conn.cpu.completeStore(keys, true)
	return keys
}

// stepOnce schedules one step and applies its output with every job done in the step.
func stepOnce(t *testing.T, s *sched.Scheduler, w *worker, i int) sched.Output {
	t.Helper()
	out := s.Schedule()
	o, _ := w.execute(out.KVConnectorMetadata.(Meta), i)
	mock := map[string][]int64{}
	for _, id := range out.Order {
		if !s.Request(id).IsPrefillChunk {
			mock[id] = []int64{7}
		}
	}
	s.UpdateWith(out, mock, o)
	return out
}

func newWorker() *worker {
	return &worker{completesAt: map[int]int{}, jobReq: map[int]string{}, isLoad: map[int]bool{}, workers: 1,
		delay: func(string, bool) int { return 0 }}
}

// A RETRY defers the request (_maximal_prefix_lookup, scheduler.py:797-800@4ba787ec), and the
// scan goes on to the first miss so the manager can start every key's lookup. Once the manager
// resolves the key, the request loads the whole offloaded prefix.
func TestARetryDefersTheRequestAndTheScanGoesOnToTheFirstMiss(t *testing.T) {
	s, conn, m := protoSetup(t, 0)
	r := add(t, s, "b", prompt(5*protoBlock+3, 100))
	keys := offloaded(t, conn, r, 4)
	m.retry[keys[1]] = true

	n, async, deferred := conn.MatchedTokensOrDefer(r, 0)
	if !deferred || n != 0 || async {
		t.Fatalf("with key 1 unresolved: matched %d async %v deferred %v, want deferred", n, async, deferred)
	}
	// Keys 0-3 are offloaded and key 4 is not: the scan asks for 0, 1 (RETRY), 2, 3 and stops at 4.
	if want := conn.reqs[r.ID].groups[0].keys[:5]; !slices.Equal(m.asked, want) {
		t.Errorf("looked up %d keys %x, want the 5 up to and including the first miss", len(m.asked), m.asked)
	}

	delete(m.retry, keys[1])
	n, async, deferred = conn.MatchedTokensOrDefer(r, 0)
	if deferred || n != 4*protoBlock || !async {
		t.Fatalf("resolved: matched %d async %v deferred %v, want %d tokens loaded asynchronously", n, async, deferred, 4*protoBlock)
	}
}

// A RETRY with no hit before it defers too: the lookup is None, not 0 (:803, :1005-1006).
func TestARetryOnTheFirstKeyDefersRatherThanMisses(t *testing.T) {
	s, conn, m := protoSetup(t, 0)
	r := add(t, s, "b", prompt(3*protoBlock+1, 100))
	keys := offloaded(t, conn, r, 3)
	m.retry[keys[0]] = true
	if n, _, deferred := conn.MatchedTokensOrDefer(r, 0); !deferred || n != 0 {
		t.Fatalf("matched %d deferred %v, want deferred", n, deferred)
	}
}

// In a sliding window a RETRY breaks the run as a miss does, and defers the request whichever
// run is found (_sliding_window_lookup, scheduler.py:830-847@4ba787ec).
func TestARetryInASlidingWindowBreaksTheRunAndDefers(t *testing.T) {
	_, conn, m := protoSetup(t, 0)
	a := newAccess("w")
	ks := make([]Key, 5)
	for i := range ks {
		ks[i] = makeKey([]byte{byte(i), 1, 2, 3, 4, 5, 6, 7}, 0)
	}
	conn.cpu.prepareStore(ks, a.a)
	conn.cpu.completeStore(ks, true)

	// Window 2 over five hits: the last run ends at 5.
	if end, deferred := conn.windowLookup(ks, 2, 0, a); end != 5 || deferred {
		t.Fatalf("all hits: end %d deferred %v, want 5 and not deferred", end, deferred)
	}
	// A RETRY on key 3 breaks the run; the scan finds keys 1-2 and still defers.
	m.retry[ks[3]] = true
	if _, deferred := conn.windowLookup(ks, 2, 0, a); !deferred {
		t.Fatal("RETRY right of the run found: not deferred, want deferred")
	}
	// A RETRY on key 4, right of a run of three that needs window 3, is not counted as a hit:
	// the run of keys 1-3 is found, deferred.
	delete(m.retry, ks[3])
	m.retry[ks[4]] = true
	end, deferred := conn.windowLookup(ks, 3, 0, a)
	if end != 4 || !deferred {
		t.Fatalf("RETRY on the last key, window 3: end %d deferred %v, want 4 and deferred", end, deferred)
	}
	// With the RETRY in a window it must fill, no run is long enough and the lookup still defers.
	m.retry = map[Key]bool{ks[1]: true, ks[3]: true}
	if _, deferred := conn.windowLookup(ks, 2, 0, a); !deferred {
		t.Fatal("no run of 2 between RETRYs: not deferred, want deferred")
	}
}

// on_new_request's policy decides whether a prefix hit is stored again: CHUNK_LEVEL skips the
// hit's chunks, REQUEST_LEVEL offloads every chunk of the request (:1468-1472@4ba787ec).
func TestRequestLevelPolicyStoresThePrefixHitAgain(t *testing.T) {
	for _, tc := range []struct {
		policy  Policy
		restore bool
	}{{ChunkLevel, false}, {RequestLevel, true}} {
		s, conn, m := protoSetup(t, 0)
		m.policy = tc.policy
		r := add(t, s, "b", prompt(4*protoBlock+5, 100))
		hit := offloaded(t, conn, r, 2)
		w := newWorker()
		for i := 0; i < 20 && s.HasRequests(); i++ {
			stepOnce(t, s, w, i)
		}
		stored := false
		for _, ks := range m.stores {
			for _, k := range ks {
				stored = stored || slices.Contains(hit, k)
			}
		}
		if stored != tc.restore {
			t.Errorf("policy %d: prefix hit offered to prepare_store %v, want %v (stores %d)", tc.policy, stored, tc.restore, len(m.stores))
		}
	}
}

// on_schedule_end is the last manager call of each step, after on_request_finished, with the
// requests the step scheduled for the first time (build_connector_meta, :2131-2195@4ba787ec).
func TestOnScheduleEndIsEachStepsLastCallWithItsNewRequests(t *testing.T) {
	s, _, m := protoSetup(t, 0)
	// Each samples three tokens, so later steps schedule both again as running requests.
	for _, id := range []string{"a", "b"} {
		if err := s.Add(&sched.Request{ID: id, Prompt: prompt(2*protoBlock+1, int64(id[0])*1000), MaxTokens: 3, EOS: -1}); err != nil {
			t.Fatal(err)
		}
	}
	w := newWorker()
	steps := 0
	for i := 0; i < 20 && s.HasRequests(); i++ {
		stepOnce(t, s, w, i)
		steps++
	}
	if steps < 3 {
		t.Fatalf("ran %d steps, want the requests decoding past their first", steps)
	}
	if len(m.ends) != steps {
		t.Fatalf("on_schedule_end called %d times over %d steps, want once a step", len(m.ends), steps)
	}
	if got := m.ends[0].NewReqIDs; !slices.Equal(got, []string{"a", "b"}) {
		t.Errorf("first step's new requests %v, want [a b]", got)
	}
	for i, e := range m.ends[1:] {
		if len(e.NewReqIDs) != 0 {
			t.Errorf("step %d names %v new, want none: each request is new once", i+1, e.NewReqIDs)
		}
	}
	// Each on_request_finished is followed, in its step, by that step's on_schedule_end.
	for i, c := range m.calls {
		if len(c) > len("on_request_finished") && c[:len("on_request_finished")] == "on_request_finished" {
			if !slices.Contains(m.calls[i+1:], "on_schedule_end") {
				t.Errorf("%s is not followed by on_schedule_end", c)
			}
		}
	}
	if m.calls[len(m.calls)-1] != "on_schedule_end" {
		t.Errorf("last manager call %q, want on_schedule_end", m.calls[len(m.calls)-1])
	}
}

// A request that finishes before the connector tracked it is started and finished on the
// manager (request_finished, :2351-2359@4ba787ec).
func TestAnUntrackedRequestIsStartedAndFinishedOnTheManager(t *testing.T) {
	_, conn, m := protoSetup(t, 0)
	r := &sched.Request{ID: "x", Prompt: prompt(4, 1), MaxTokens: 1}
	if conn.FinishedAllGroups(r, nil) {
		t.Fatal("held the blocks, want freed")
	}
	if want := []string{"on_new_request:x", "on_request_finished:x"}; !slices.Equal(m.calls, want) {
		t.Errorf("manager calls %v, want %v", m.calls, want)
	}
}

// has_pending_work keeps the engine stepping with no request and no job in flight
// (has_pending_push_work, :2198-2216@4ba787ec).
func TestManagerPendingWorkKeepsTheEngineStepping(t *testing.T) {
	s, conn, m := protoSetup(t, 0)
	if conn.HasPendingPushWork() || s.HasRequests() {
		t.Fatal("idle engine with no pending work steps, want idle")
	}
	m.pending = true
	if !conn.HasPendingPushWork() || !s.HasRequests() {
		t.Fatal("manager has pending work: the engine is idle, want it stepping")
	}
}

// get_external_cache_hit_sources splits the accepted hit by each loaded key's tier, the
// outermost where keys overlap, and the uncovered rest as EXTERNAL_UNSPECIFIED
// (scheduler.py:1304-1349@4ba787ec).
func TestExternalHitSourcesSplitTheHitByTier(t *testing.T) {
	s, conn, m := protoSetup(t, 0)
	r := add(t, s, "b", prompt(4*protoBlock+5, 100))
	offloaded(t, conn, r, 3)
	w := newWorker()
	stepOnce(t, s, w, 0)
	m.source = "DISK"
	got := conn.ExternalHitSources(r, 3*protoBlock)
	if len(got) != 1 || got["DISK"] != 3*protoBlock {
		t.Errorf("sources %v, want all %d tokens from DISK", got, 3*protoBlock)
	}
	m.source = ""
	// Two tokens past the loaded keys are covered by none.
	got = conn.ExternalHitSources(r, 3*protoBlock+2)
	if got["HOST"] != 3*protoBlock || got[unspecifiedSource] != 2 {
		t.Errorf("sources %v, want %d HOST and 2 EXTERNAL_UNSPECIFIED", got, 3*protoBlock)
	}
}

// Secondary tiers are refused unless the caller supplies the manager that runs them.
func TestSecondaryTiersNeedTheCallersManager(t *testing.T) {
	o := Options{SpecName: "TieringOffloadingSpec", SecondaryTiers: 1}
	if err := Check(o); err == nil {
		t.Error("secondary tiers with no manager: accepted, want refused")
	}
	o.Manager = true
	if err := Check(o); err != nil {
		t.Errorf("secondary tiers with the caller's manager: %v, want accepted", err)
	}
}

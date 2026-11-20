package multi

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/core/sched"
)

// stub is a child connector that records what the wrapper asks of it and answers from a script.
type stub struct {
	match int
	async bool
	hold  bool // Finished's answer

	asked    int
	newReqs  []string
	allocs   []alloc
	finished []finish
	outputs  []sched.TransferOutcome
}

type alloc struct {
	req      string
	external int
	blocks   []int
}

type finish struct {
	req    string
	blocks any
}

func (s *stub) OnNewRequest(r *sched.Request) { s.newReqs = append(s.newReqs, r.ID) }
func (s *stub) MatchedTokens(*sched.Request, int) (int, bool) {
	s.asked++
	return s.match, s.async
}
func (s *stub) AfterAlloc(r *sched.Request, b sched.AllocatedBlocks, external int) {
	s.allocs = append(s.allocs, alloc{r.ID, external, b.IDs})
}
func (s *stub) Finished(r *sched.Request, ids []int) bool {
	s.finished = append(s.finished, finish{r.ID, ids})
	return s.hold
}
func (s *stub) AfterOutput(o sched.TransferOutcome) { s.outputs = append(s.outputs, o) }

// deferStub also states Deferrer: it can report that it is still looking up its hit.
type deferStub struct {
	*stub
	defer_ bool
}

func (d deferStub) MatchedTokensOrDefer(r *sched.Request, local int) (int, bool, bool) {
	d.asked++
	return d.match, d.async, d.defer_
}

// hmaStub also states SupportsHMA.
type hmaStub struct{ *stub }

func (h hmaStub) FinishedAllGroups(r *sched.Request, ids [][]int) bool {
	h.finished = append(h.finished, finish{r.ID, ids})
	return h.hold
}

type capStub struct {
	*stub
	caps sched.Capabilities
}

func (c capStub) ConnectorModels() sched.Capabilities { return c.caps }

type kvdStub struct {
	*stub
	delivery bool
}

func (k kvdStub) RequiresKVDelivery() bool { return k.delivery }

type pushStub struct {
	*stub
	pending bool
}

func (p pushStub) HasPendingPushWork() bool { return p.pending }

type metaStub struct {
	*stub
	meta any
}

func (m metaStub) BuildMeta(sched.StepOutput) any { return m.meta }

// noArgStub builds its metadata from nothing, as NIXL's scheduler does.
type noArgStub struct {
	*stub
	built *int
}

func (n noArgStub) BuildMeta() { *n.built++ }

func req(id string) *sched.Request { return &sched.Request{ID: id} }

func wrap(t *testing.T, cs ...sched.Connector) *Connector {
	t.Helper()
	var ch []Child
	for i, c := range cs {
		ch = append(ch, Child{Name: fmt.Sprintf("child%d", i), Conn: c})
	}
	m, err := New(ch)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestNewRefusesNoChildAndANilChild(t *testing.T) {
	if _, err := New(nil); err == nil {
		t.Error("a MultiConnector with no child was accepted")
	}
	if _, err := New([]Child{{Name: "x"}}); err == nil || !strings.Contains(err.Error(), `"x"`) {
		t.Errorf("a nil child: %v, want an error naming it", err)
	}
}

// vLLM asks every child (a later one may still be looking its hit up) and gives the load to the
// first child that reports tokens, not the one with the most or the last.
func TestTheFirstChildWithTokensLoadsAndLaterChildrenAreStillAsked(t *testing.T) {
	none, first, last := &stub{}, &stub{match: 5, async: true}, &stub{match: 7, async: false}
	m := wrap(t, none, first, last)
	n, async := m.MatchedTokens(req("r"), 0)
	if n != 5 || !async {
		t.Errorf("matched (%d, async %v), want the first matching child's (5, true)", n, async)
	}
	if i, ok := m.Chosen("r"); !ok || i != 1 {
		t.Errorf("chosen = (%d, %v), want child 1", i, ok)
	}
	for i, s := range []*stub{none, first, last} {
		if s.asked != 1 {
			t.Errorf("child %d was asked %d times, want once: every child is asked, in order", i, s.asked)
		}
	}
}

func TestADeferringChildDefersTheWholeRequest(t *testing.T) {
	hit := &stub{match: 4, async: true}
	looking := deferStub{stub: &stub{}, defer_: true}
	m := wrap(t, hit, looking)
	n, async, deferred := m.MatchedTokensOrDefer(req("r"), 0)
	if !deferred || n != 0 || async {
		t.Errorf("got (%d, %v, deferred %v), want the request deferred with no tokens", n, async, deferred)
	}
	// Through the plain interface a deferral reads as no match.
	if n, _ := m.MatchedTokens(req("r"), 0); n != 0 {
		t.Errorf("MatchedTokens during a deferral = %d, want 0", n)
	}
}

func TestARequestNobodyMatchedHasNoChosenChild(t *testing.T) {
	a, b := &stub{}, &stub{}
	m := wrap(t, a, b)
	if n, async := m.MatchedTokens(req("r"), 0); n != 0 || async {
		t.Errorf("got (%d, %v), want no match", n, async)
	}
	if _, ok := m.Chosen("r"); ok {
		t.Error("a request no child matched has a chosen child")
	}
	m.AfterAlloc(req("r"), sched.AllocatedBlocks{IDs: []int{3, 4}}, 0)
	for i, s := range []*stub{a, b} {
		if !reflect.DeepEqual(s.allocs, []alloc{{"r", 0, []int{3, 4}}}) {
			t.Errorf("child %d saw %+v, want the real blocks and no external tokens", i, s.allocs)
		}
	}
}

func TestAfterAllocGivesTheChosenChildTheExternalTokensAndTheOthersNone(t *testing.T) {
	a, b, c := &stub{}, &stub{match: 8, async: true}, &stub{match: 8, async: true}
	m := wrap(t, a, b, c)
	m.MatchedTokens(req("r"), 0)
	m.AfterAlloc(req("r"), sched.AllocatedBlocks{IDs: []int{1, 2, 3}}, 8)
	want := map[string]int{"a": 0, "b": 8, "c": 0}
	for name, s := range map[string]*stub{"a": a, "b": b, "c": c} {
		if len(s.allocs) != 1 || s.allocs[0].external != want[name] || !reflect.DeepEqual(s.allocs[0].blocks, []int{1, 2, 3}) {
			t.Errorf("child %s saw %+v, want external %d with blocks [1 2 3]", name, s.allocs, want[name])
		}
	}
}

func TestOnNewRequestReachesEveryChild(t *testing.T) {
	a, b := &stub{}, &stub{}
	wrap(t, a, b).OnNewRequest(req("r"))
	if len(a.newReqs) != 1 || len(b.newReqs) != 1 {
		t.Errorf("children saw %v and %v, want one request each", a.newReqs, b.newReqs)
	}
}

// A finished request's blocks are held while any child holds them, every child must hear that
// the request finished (even after one has said it holds the blocks), and the choice is dropped.
func TestFinishedCallsEveryChildOrsTheAnswerAndForgetsTheChoice(t *testing.T) {
	for _, c := range []struct {
		name       string
		holds      []bool
		want       bool
		wantCalled int
	}{
		{"first holds", []bool{true, false}, true, 2},
		{"last holds", []bool{false, true}, true, 2},
		{"none holds", []bool{false, false}, false, 2},
	} {
		t.Run(c.name, func(t *testing.T) {
			a, b := &stub{match: 4, hold: c.holds[0]}, &stub{hold: c.holds[1]}
			m := wrap(t, a, b)
			m.MatchedTokens(req("r"), 0)
			if got := m.Finished(req("r"), []int{9, 8}); got != c.want {
				t.Errorf("Finished = %v, want %v", got, c.want)
			}
			if len(a.finished)+len(b.finished) != c.wantCalled {
				t.Errorf("%d children heard of the finish, want %d", len(a.finished)+len(b.finished), c.wantCalled)
			}
			if _, ok := m.Chosen("r"); ok {
				t.Error("the chosen child was not forgotten")
			}
		})
	}
}

// With every child SupportsHMA each child gets every group's block ids; a child that is not
// SupportsHMA turns the hybrid manager off, so there is one group and request_finished runs.
func TestFinishedAllGroupsFollowsWhetherEveryChildSupportsHMA(t *testing.T) {
	groups := [][]int{{1, 2}, {3}}
	a, b := &stub{}, &stub{hold: true}
	m := wrap(t, hmaStub{a}, hmaStub{b})
	if !m.FinishedAllGroups(req("r"), groups) {
		t.Error("a holding child did not hold the request")
	}
	for i, s := range []*stub{a, b} {
		if len(s.finished) != 1 || !reflect.DeepEqual(s.finished[0].blocks, groups) {
			t.Errorf("child %d heard %+v, want every group's ids", i, s.finished)
		}
	}

	c, d := &stub{}, &stub{}
	m = wrap(t, hmaStub{c}, d)
	m.FinishedAllGroups(req("r"), [][]int{{5, 6}})
	for i, s := range []*stub{c, d} {
		if len(s.finished) != 1 || !reflect.DeepEqual(s.finished[0].blocks, []int{5, 6}) {
			t.Errorf("child %d heard %+v, want group 0's ids through request_finished", i, s.finished)
		}
	}
	defer func() {
		if recover() == nil {
			t.Error("several KV cache groups beside a child that is not SupportsHMA did not stop the run")
		}
	}()
	m.FinishedAllGroups(req("r"), groups)
}

func TestAfterOutputGivesEachChildItsOwnWorkerMeta(t *testing.T) {
	a, b, c := &stub{}, &stub{}, &stub{}
	m := wrap(t, a, b, c)
	m.AfterOutput(sched.TransferOutcome{FinishedSending: []string{"x"}, WorkerMeta: WorkerMeta{nil, "for-b"}})
	for i, want := range []any{nil, "for-b", nil} {
		s := []*stub{a, b, c}[i]
		if len(s.outputs) != 1 || s.outputs[0].WorkerMeta != want {
			t.Errorf("child %d got worker meta %v, want %v", i, s.outputs, want)
		}
		if !reflect.DeepEqual(s.outputs[0].FinishedSending, []string{"x"}) {
			t.Errorf("child %d lost the rest of the output: %+v", i, s.outputs[0])
		}
	}
	// No worker metadata at all (or one that is not a WorkerMeta) is none for every child.
	m.AfterOutput(sched.TransferOutcome{})
	m.AfterOutput(sched.TransferOutcome{WorkerMeta: "not-a-multi-meta"})
	for i, s := range []*stub{a, b, c} {
		for _, o := range s.outputs[1:] {
			if o.WorkerMeta != nil {
				t.Errorf("child %d got worker meta %v with none reported", i, o.WorkerMeta)
			}
		}
	}
}

func TestCapabilitiesAreWhatEveryChildStates(t *testing.T) {
	all := sched.Capabilities{HybridGroups: true, Drafters: true, ConcurrentBatches: true}
	noDrafter := sched.Capabilities{HybridGroups: true, ConcurrentBatches: true}
	hma := func(c sched.Capabilities) sched.Connector { return hmaStub{&stub{}}.withCaps(c) }
	for _, c := range []struct {
		name     string
		children []sched.Connector
		want     sched.Capabilities
	}{
		{"both state all", []sched.Connector{hma(all), hma(all)}, all},
		{"one lacks drafters", []sched.Connector{hma(all), hma(noDrafter)}, noDrafter},
		{"hybrid groups need SupportsHMA from every child", []sched.Connector{hma(all), capStub{&stub{}, all}},
			sched.Capabilities{Drafters: true, ConcurrentBatches: true}},
		{"a child that states nothing", []sched.Connector{hma(all), &stub{}}, sched.Capabilities{}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := wrap(t, c.children...).ConnectorModels(); got != c.want {
				t.Errorf("capabilities %+v, want %+v", got, c.want)
			}
		})
	}
}

// withCaps is a SupportsHMA child that states its capabilities.
type hmaCapStub struct {
	hmaStub
	caps sched.Capabilities
}

func (h hmaCapStub) ConnectorModels() sched.Capabilities { return h.caps }
func (h hmaStub) withCaps(c sched.Capabilities) sched.Connector {
	return hmaCapStub{h, c}
}

func TestRequiresKVDeliveryIsAnyChildWithTheProducerDefault(t *testing.T) {
	for _, c := range []struct {
		name string
		kids []Child
		want bool
	}{
		{"a child that states true", []Child{{Name: "a", Conn: kvdStub{&stub{}, false}}, {Name: "b", Conn: kvdStub{&stub{}, true}}}, true},
		{"children that state false", []Child{{Name: "a", Conn: kvdStub{&stub{}, false}}, {Name: "b", Conn: kvdStub{&stub{}, false}}}, false},
		{"a producer that states nothing", []Child{{Name: "a", Conn: &stub{}, Producer: true}}, true},
		{"a consumer that states nothing", []Child{{Name: "a", Conn: &stub{}}}, false},
		{"a producer whose statement wins", []Child{{Name: "a", Conn: kvdStub{&stub{}, false}, Producer: true}}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, err := New(c.kids)
			if err != nil {
				t.Fatal(err)
			}
			if got := m.RequiresKVDelivery(); got != c.want {
				t.Errorf("RequiresKVDelivery = %v, want %v", got, c.want)
			}
		})
	}
}

func TestPendingPushWorkIsAnyChilds(t *testing.T) {
	idle, busy := pushStub{&stub{}, false}, pushStub{&stub{}, true}
	if wrap(t, idle, &stub{}).HasPendingPushWork() {
		t.Error("pending push work with none pending")
	}
	if !wrap(t, idle, busy).HasPendingPushWork() {
		t.Error("a child's pending push work is not reported")
	}
}

func TestBuildMetaIsEachChildsInOrder(t *testing.T) {
	built := 0
	m := wrap(t, metaStub{&stub{}, "first"}, noArgStub{&stub{}, &built}, &stub{}, metaStub{&stub{}, "last"})
	got, ok := m.BuildMeta(sched.StepOutput{}).(Meta)
	if !ok {
		t.Fatal("BuildMeta did not return a Meta")
	}
	if !reflect.DeepEqual([]any(got), []any{"first", nil, nil, "last"}) {
		t.Errorf("Meta = %v, want each child's result in child order", got)
	}
	if built != 1 {
		t.Errorf("the no-argument BuildMeta child ran %d times, want once", built)
	}
	if got.At(3) != "last" || got.At(4) != nil || got.At(-1) != nil {
		t.Error("Meta.At is not the child's metadata, or nil out of range")
	}
}

// The wrapper over the real NIXL and offloading schedulers on a real sched.Scheduler. A
// prefill-side request (do_remote_decode) whose prompt is in the CPU tier is loaded by the
// offloading child, which is listed second, and the NIXL child holds its blocks once it
// finishes. Request a fills the GPU cache, b evicts it, c asks for a's prompt again.
func TestAPrefillerLoadsFromTheCPUTierAndNIXLHoldsTheFinishedBlocks(t *testing.T) {
	sc := realScenario(t, `[{"name":"NixlConnector","kv_role":"kv_producer"},
	  {"name":"OffloadingConnector","kv_role":"kv_both","extra_config":{"cpu_bytes_to_use":262144,"offload_prompt_only":false}}]`,
		"kv_producer", []string{
			`{"id":"a","arrival_step":0,"prompt":%s,"output":[7,7],"max_tokens":2}`,
			`{"id":"b","arrival_step":6,"prompt":%s,"output":[7,7],"max_tokens":2}`,
			`{"id":"c","arrival_step":16,"prompt":%s,"output":[7],"max_tokens":1,"remote_decode":true,"send_delay":3}`,
		}, [][2]int{{100, 41}, {200, 40}, {100, 41}})
	tr := run(t, sc)

	loaded := firstStep(t, tr, func(s step) bool { return s.Offload != nil && hasLoadFor(s.Offload, "c") })
	if tr.Steps[loaded].Chosen["c"] != 1 {
		t.Errorf("c is loaded from the CPU tier at step %d with chosen %v, want the offloading child (1)", loaded, tr.Steps[loaded].Chosen)
	}
	if !contains(tr.Steps[loaded].WaitingForRemoteKV, "c") && !contains(tr.Steps[loaded+1].WaitingForRemoteKV, "c") {
		t.Errorf("c is not waiting for its CPU load around step %d", loaded)
	}
	done := firstStep(t, tr, func(s step) bool { return contains(s.Finished, "c") })
	if !contains(tr.Steps[done].DelayFreed, "c") {
		t.Errorf("c finished at step %d but NIXL does not hold its blocks (delay_freed %v)", done, tr.Steps[done].DelayFreed)
	}
	if _, ok := tr.Steps[done].Chosen["c"]; ok {
		t.Error("the choice for c outlived the request")
	}
	sent := firstStep(t, tr, func(s step) bool { return contains(s.Transport.Sending, "c") })
	if sent <= done {
		t.Errorf("c's send is reported at step %d, not after it finished at step %d", sent, done)
	}
	for _, s := range tr.Steps[done:sent] {
		if !contains(s.DelayFreed, "c") {
			t.Errorf("step %d: c's blocks were released before the send was reported at step %d", s.Step, sent)
		}
	}
	if contains(tr.Steps[sent].DelayFreed, "c") {
		t.Errorf("c's blocks are still held after the send was reported at step %d", sent)
	}
}

// A decode-side request (do_remote_prefill) is loaded by NIXL when NIXL is listed first, and the
// offloading child, which has nothing for it, is not chosen.
func TestADecoderIsLoadedByNIXLListedFirst(t *testing.T) {
	sc := realScenario(t, `[{"name":"NixlConnector","kv_role":"kv_consumer"},
	  {"name":"OffloadingConnector","kv_role":"kv_both","extra_config":{"cpu_bytes_to_use":262144}}]`,
		"kv_consumer", []string{
			`{"id":"d","arrival_step":0,"prompt":%s,"output":[7,7],"max_tokens":2,"remote_prefill":true,"recv_delay":2}`,
		}, [][2]int{{300, 24}})
	tr := run(t, sc)
	waiting := firstStep(t, tr, func(s step) bool { return contains(s.WaitingForRemoteKV, "d") })
	if got, ok := tr.Steps[waiting].Chosen["d"]; !ok || got != 0 {
		t.Errorf("d waits for remote KV at step %d with chosen %v, want NIXL (0)", waiting, tr.Steps[waiting].Chosen)
	}
	pulled := firstStep(t, tr, func(s step) bool { return contains(s.Transport.Recving, "d") })
	if pulled < waiting {
		t.Errorf("the pull is reported at step %d, before d waited at step %d", pulled, waiting)
	}
	if _, ok := tr.Finished["d"]; !ok {
		t.Error("d never finished")
	}
	for _, s := range tr.Steps {
		if s.Offload != nil && hasLoadFor(s.Offload, "d") {
			t.Fatalf("step %d: the offloading child loaded d, which NIXL was chosen for", s.Step)
		}
	}
}

// realScenario is a small engine (4-token blocks, 14 blocks) with the wrapper over the listed
// children; each request template takes its prompt from the (first token, length) pair.
func realScenario(t *testing.T, children, role string, reqs []string, prompts [][2]int) scenario {
	t.Helper()
	var rs []string
	for i, tpl := range reqs {
		p := make([]int, prompts[i][1])
		for j := range p {
			p[j] = prompts[i][0] + j
		}
		raw, _ := json.Marshal(p)
		rs = append(rs, fmt.Sprintf(tpl, raw))
	}
	doc := fmt.Sprintf(`{"config":{"max_num_seqs":4,"max_num_batched_tokens":32,"block_size":4,"num_blocks":14,
	  "max_model_len":256,"enable_prefix_caching":true,"scheduling_policy":"fcfs",
	  "kv_page":{"num_kv_heads":1,"head_size":512},"transfer":{"load_delay":1,"store_delay":1},
	  "connector":{"name":"MultiConnector","kv_role":%q,"extra_config":{"connectors":%s}}},
	  "requests":[%s],"max_steps":60}`, role, children, strings.Join(rs, ","))
	var sc scenario
	if err := json.Unmarshal([]byte(doc), &sc); err != nil {
		t.Fatal(err)
	}
	return sc
}

func firstStep(t *testing.T, tr trace, pred func(step) bool) int {
	t.Helper()
	for i, s := range tr.Steps {
		if pred(s) {
			return i
		}
	}
	t.Fatalf("no step of the %d-step trace matches", len(tr.Steps))
	return 0
}

func hasLoadFor(o *offState, id string) bool {
	for _, l := range o.Loads {
		if l[1] == id {
			return true
		}
	}
	return false
}

func contains(xs []string, x string) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

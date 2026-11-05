// Package embedtest checks an embed.Constructor against the embed contract, in the way
// testing/fstest checks an fs.FS.
//
// Check drives a router through a fixed scenario on a deterministic Runtime it provides
// (NewRuntime), with the harness playing the engines: each endpoint serves vLLM's metrics at
// the path the router scrapes, reporting either an idle engine or a saturated one. What the
// harness cannot know is how a given router is configured to behave a given way, so the Suite
// supplies one configuration per part of the scenario, each with the properties its field
// documents. A router passes when, so configured, it behaves as the contract says.
//
// The scenario covers a routing decision and its fields; endpoint readiness, address updates
// and removal;
// every RejectKind a request can earn in it (NoEndpoints, Admission, Saturated,
// ContextCancelled, TTLExpired, Unclassified for a malformed body, and Unavailable when the
// Suite supplies a configuration that produces it); response headers and bodies, streamed and
// not; each Termination; the eviction signal; KV-event ingest, removal and clear (with the index
// itself for a router that implements embed.KVIndexInspector); refusal of a
// plugin type through Config.Allow, and of an unaudited one that Allow permits; a request's
// own TTL as its queue-wait budget; a host refusal through Config.Check, before any plugin;
// determinism over a seed; that no draw happens before the Constructor returns; and that Stop
// ends every goroutine and loop.
package embedtest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/embed"
)

// Sheddable is the objective (Request.Objective) of a request the router may shed: a
// configuration in a Suite gives it a negative priority. A request with no objective is not
// sheddable.
const Sheddable = "sheddable"

// Suite is what Check needs to know about one router implementation.
//
// For llm-d's router, for example, the configurations can share one pool (--pool-name, an
// InferencePool selecting Labels) and an InferenceObjective named Sheddable with a negative
// priority, and differ in their EndpointPickerConfig: Route a queue scorer and a max-score
// picker; Hold the same with the flowControl feature gate, a one-request capacity and a
// default request TTL; Evict that with eviction enabled; KV a precise-prefix-cache producer
// at the harness's block size with a prefix-cache scorer, and Engines serving its prompt
// rendering. Refuse is any of them, with RefuseType one of the plugin types it names.
type Suite struct {
	// New is the implementation under test.
	New embed.Constructor

	// Model is the model every request names and every endpoint serves. Default
	// "test-model".
	Model string
	// Labels are set on every endpoint; the configurations' pools select endpoints by them.
	Labels map[string]string
	// Engines serves the router's outbound requests other than metrics scrapes (prompt
	// rendering, for example). nil answers them with 404.
	Engines func(*http.Request) (*http.Response, error)
	// Seed seeds the Runtime's random streams.
	Seed int64
	// Observe is how long the router takes to see a change in what the engines report.
	// Default one second.
	Observe time.Duration

	// Route dispatches a request to a ready endpoint, without flow control. Requests with
	// objective Sheddable have a negative priority, and while every endpoint is saturated a
	// sheddable request is refused (Admission) and any other is still dispatched. The
	// router does not evict.
	Route embed.Config
	// Hold has flow control: while every endpoint is saturated it holds requests, at most
	// one at a time (a second is refused, Saturated), each for a bounded queue-wait budget
	// (FlowControl.MaxQueueTTL) after which it is refused (TTLExpired). A request whose
	// Request.TTL is shorter has that shorter budget.
	Hold embed.Config
	// KV routes by engines' KV events: of idle endpoints, it sends a request to the one whose
	// published blocks hold the longest prefix of the request's prompt. A prompt is token IDs
	// in a completions request.
	KV embed.Config
	// Evict, when set, has flow control with eviction: once every endpoint is saturated, a
	// held request without an objective makes the router evict a dispatched Sheddable one
	// within EvictWithin. nil skips the eviction part.
	Evict *embed.Config
	// Unavailable, when set, refuses every request, Unavailable, while its endpoints are ready
	// and idle: the router can serve none of them and has no more specific reason. A router
	// with no configuration that does so leaves it nil, and the part is skipped.
	Unavailable *embed.Config
	// EvictWithin bounds how long eviction takes. Default ten seconds.
	EvictWithin time.Duration
	// Refuse is a configuration that instantiates a plugin of RefuseType, a type the router
	// hosts (audited, with no reason of its own to refuse it). Check refuses that type
	// through Allow and expects construction to fail with a *embed.RefusalError carrying
	// Allow's reason, and expects the same configuration to construct and route with no
	// policy.
	Refuse     embed.Config
	RefuseType string
	// Unaudited, when set, is a plugin type the router refuses because it has not audited it
	// for hosting, although it would construct it otherwise: Check sets it in the Refuse
	// configuration's place (UnauditedConfig) with an Allow that permits every type, and
	// expects a *embed.RefusalError naming it. Empty skips the check.
	Unaudited       string
	UnauditedConfig embed.Config
}

// Run runs Check and reports each problem as a test error.
func Run(t *testing.T, s Suite) {
	t.Helper()
	for _, p := range Problems(s) {
		t.Error(p)
	}
}

// Check returns the problems Problems finds, joined, or nil when there are none.
func Check(s Suite) error { return errors.Join(Problems(s)...) }

// Problems runs the scenario and returns every departure from the contract it finds, each
// prefixed with the part of the scenario that found it.
func Problems(s Suite) []error {
	if s.Model == "" {
		s.Model = "test-model"
	}
	if s.Observe <= 0 {
		s.Observe = time.Second
	}
	if s.EvictWithin <= 0 {
		s.EvictWithin = 10 * time.Second
	}
	var problems []error
	parts := []struct {
		name string
		run  func(*env)
	}{
		{"route", partRoute},
		{"readiness", partReadiness},
		{"no-endpoints", partNoEndpoints},
		{"admission", partAdmission},
		{"unclassified", partUnclassified},
		{"unavailable", partUnavailable},
		{"responses", partResponses},
		{"determinism", partDeterminism},
		{"hold", partHold},
		{"kv", partKV},
		{"evict", partEvict},
		{"refuse", partRefuse},
		{"unaudited", partUnaudited},
		{"check", partCheck},
	}
	for _, p := range parts {
		e := &env{s: &s, part: p.name}
		func() {
			defer func() {
				if v := recover(); v != nil {
					e.fail("panicked: %v", v)
				}
				e.stop()
			}()
			p.run(e)
		}()
		problems = append(problems, e.problems...)
	}
	return problems
}

// env is one part of the scenario: a runtime, a router on it, and the engines.
type env struct {
	s        *Suite
	part     string
	problems []error
	rt       *Runtime
	r        embed.Router
	cancel   context.CancelFunc
	eps      []embed.Endpoint
	busy     map[string]bool // by address: the engine reports saturation
	blocks   int             // block size of KV events
	started  int             // goroutines live before the router existed
	nextReq  int
	flights  []*outcome
}

func (e *env) fail(format string, args ...any) {
	e.problems = append(e.problems, fmt.Errorf("%s: %s", e.part, fmt.Sprintf(format, args...)))
}

// build constructs the router on a fresh runtime. It reports false, having recorded why, when
// the part cannot go on.
func (e *env) build(cfg embed.Config) bool {
	e.rt = NewRuntime(e.s.Seed)
	e.busy = map[string]bool{}
	e.rt.Engines = e.serve
	ctx, cancel := context.WithCancel(context.Background())
	r, err := e.s.New(ctx, e.rt, cfg)
	if err != nil {
		cancel()
		e.fail("construction failed: %v", err)
		return false
	}
	if r == nil {
		cancel()
		e.fail("construction returned a nil Router and no error")
		return false
	}
	e.r, e.cancel = r, cancel
	if n := e.rt.randCalls; n > 0 {
		e.fail("Rand was called %d times before the Constructor returned, by it or by work it started; no draw may happen until it returns", n)
	}
	e.blocks = 16
	if ix := r.Requirements().PrefixIndexes; len(ix) > 0 && ix[0].BlockSize > 0 {
		e.blocks = ix[0].BlockSize
	}
	return true
}

// stop ends the router and checks that it ended everything it started.
func (e *env) stop() {
	if e.r == nil {
		return
	}
	r := e.r
	e.r = nil
	defer func() {
		if v := recover(); v != nil {
			e.fail("Stop panicked: %v", v)
		}
	}()
	for _, f := range e.flights {
		f.end(e)
	}
	defer e.cancel()
	// Stop ends what the router started without the construction context ending.
	e.rt.stopped = true
	r.Stop()
	e.rt.Settle()
	e.rt.Advance(10 * e.s.Observe)
	if n := e.rt.Live(); n != 0 {
		e.fail("%d goroutines started with Runtime.Go are still live after Stop", n)
	}
	if n := e.rt.loopsStopped; n != 0 {
		e.fail("an Every loop ran %d times after Stop", n)
	}
}

// endpoints adds n ready endpoints, idle, and lets the router observe them.
func (e *env) endpoints(n int) {
	port := 8000
	if p := e.r.Pool().TargetPorts; len(p) > 0 {
		port = p[0]
	}
	for i := range n {
		ep := embed.Endpoint{
			Name:     "endpoint-" + strconv.Itoa(len(e.eps)),
			Labels:   maps.Clone(e.s.Labels),
			NodeName: "node-" + strconv.Itoa(i),
			IP:       "10.0.0." + strconv.Itoa(len(e.eps)+1),
			Port:     port,
		}
		e.eps = append(e.eps, ep)
		e.busy[ep.Address()] = false // its engine serves idle metrics from the start
		if err := e.r.SetEndpoint(ep, true); err != nil {
			e.fail("SetEndpoint(%s): %v", ep.Name, err)
		}
	}
	e.rt.Settle()
	e.rt.Advance(e.s.Observe)
}

// saturate makes every engine report saturation (or idleness) and lets the router observe it.
func (e *env) saturate(on bool) {
	for _, ep := range e.eps {
		e.busy[ep.Address()] = on
	}
	e.rt.Advance(e.s.Observe)
}

func (e *env) addresses() []string {
	out := make([]string, len(e.eps))
	for i, ep := range e.eps {
		out[i] = ep.Address()
	}
	return out
}

// outcome is what a Route call produced.
type outcome struct {
	req      embed.Request
	ctx      context.Context
	cancel   context.CancelFunc
	calls    int
	flight   embed.Flight
	decision embed.Decision
	rej      *embed.Rejection
	ended    bool
}

func (o *outcome) decided() bool { return o.calls > 0 }

func (o *outcome) String() string {
	switch {
	case o.calls == 0:
		return "no outcome yet"
	case o.rej != nil:
		return fmt.Sprintf("rejected %s (%q)", o.rej.Kind, o.rej.Msg)
	default:
		return fmt.Sprintf("dispatched to %s", o.decision.Endpoint)
	}
}

// end ends a dispatched request's response normally, if it has not ended.
func (o *outcome) end(e *env) {
	if o.flight == nil || o.ended {
		return
	}
	o.ended = true
	o.flight.ResponseHeaders(map[string]string{":status": "200", "content-type": "application/json"})
	o.flight.ResponseBody(completion(o.req.ID, e.s.Model), true)
	e.rt.Settle()
}

// request is a completions request whose prompt is tokens.
func (e *env) request(objective string, stream bool, tokens []uint32) embed.Request {
	e.nextReq++
	id := "request-" + strconv.Itoa(e.nextReq)
	body, _ := json.Marshal(map[string]any{"model": e.s.Model, "prompt": tokens, "max_tokens": 8, "stream": stream})
	return embed.Request{ID: id, Path: "/v1/completions", Body: body, Stream: stream, Objective: objective}
}

// route sends req and returns its outcome so far: whatever Route and the work it set off
// produced by the time the runtime has settled.
func (e *env) route(req embed.Request) *outcome {
	o := &outcome{req: req}
	o.ctx, o.cancel = context.WithCancel(context.Background())
	e.r.Route(o.ctx, req, func(f embed.Flight, d embed.Decision, rej *embed.Rejection) {
		o.calls++
		if e.rt.stopped {
			e.fail("%s: decided after Stop; a request undecided when the router stops is never decided", req.ID)
		}
		if o.calls > 1 {
			e.fail("%s: done called %d times; it is called exactly once", req.ID, o.calls)
			return
		}
		o.flight, o.decision, o.rej = f, d, rej
		switch {
		case f == nil && rej == nil:
			e.fail("%s: done called with neither a Flight nor a Rejection", req.ID)
		case f != nil && rej != nil:
			e.fail("%s: done called with both a Flight and a Rejection (%s)", req.ID, rej.Kind)
		case rej != nil && !reflectZero(d):
			e.fail("%s: rejected with a non-zero Decision %+v", req.ID, d)
		}
		if f != nil {
			e.flights = append(e.flights, o)
		}
	})
	e.rt.Settle()
	e.rt.Advance(0)
	return o
}

func reflectZero(d embed.Decision) bool {
	return d.Endpoint == "" && d.Prefill == "" && len(d.Encoders) == 0 && len(d.Scores) == 0 &&
		d.IncomingModel == "" && d.TargetModel == ""
}

// wantDispatched checks that o was dispatched to one of addrs, and the decision's fields.
func (e *env) wantDispatched(o *outcome, addrs []string) bool {
	if o.flight == nil {
		e.fail("%s: want a dispatch to one of %v, got %s", o.req.ID, addrs, o)
		return false
	}
	d := o.decision
	if !slices.Contains(addrs, d.Endpoint) {
		e.fail("%s: dispatched to %q, which is not one of the ready endpoints %v", o.req.ID, d.Endpoint, addrs)
	}
	if d.Prefill != "" {
		e.fail("%s: Prefill %q from a configuration that does not disaggregate", o.req.ID, d.Prefill)
	}
	if len(d.Encoders) != 0 {
		e.fail("%s: Encoders %v for a request with no media", o.req.ID, d.Encoders)
	}
	for addr := range d.Scores {
		if !slices.Contains(e.addresses(), addr) {
			e.fail("%s: a score for %q, which is no endpoint", o.req.ID, addr)
		}
	}
	if d.IncomingModel != e.s.Model {
		e.fail("%s: IncomingModel %q, want the model the request named, %q", o.req.ID, d.IncomingModel, e.s.Model)
	}
	if d.TargetModel != "" && d.TargetModel != e.s.Model {
		e.fail("%s: TargetModel %q with no rewrite configured", o.req.ID, d.TargetModel)
	}
	return true
}

// wantRejected checks that o was rejected with kind.
func (e *env) wantRejected(o *outcome, kind embed.RejectKind) {
	switch {
	case o.rej == nil:
		e.fail("%s: want a %s rejection, got %s", o.req.ID, kind, o)
	case o.rej.Kind != kind:
		e.fail("%s: want a %s rejection, got %s (%q)", o.req.ID, kind, o.rej.Kind, o.rej.Msg)
	case o.rej.Msg == "":
		e.fail("%s: a %s rejection with no message", o.req.ID, kind)
	}
}

// wantHeld checks that o has no outcome yet.
func (e *env) wantHeld(o *outcome) bool {
	if o.decided() {
		e.fail("%s: want it held, got %s", o.req.ID, o)
		return false
	}
	return true
}

// prompt is n blocks of distinct tokens starting at first.
func (e *env) prompt(first uint32, n int) []uint32 {
	out := make([]uint32, n*e.blocks)
	for i := range out {
		out[i] = first + uint32(i)
	}
	return out
}

// --- engines ---

// serve plays the engines: a GET of a metrics path returns vLLM's series for the engine at
// the request's host; anything else goes to Suite.Engines.
func (e *env) serve(req *http.Request) (*http.Response, error) {
	if req.Method == http.MethodGet && e.isMetricsPath(req.URL.Path) {
		busy, ok := e.busy[req.URL.Host]
		if !ok {
			return Respond(req, http.StatusNotFound, "text/plain", "no engine at "+req.URL.Host+"\n"), nil
		}
		return Respond(req, http.StatusOK, "text/plain; version=0.0.4", e.metrics(busy)), nil
	}
	if e.s.Engines != nil {
		return e.s.Engines(req)
	}
	return Respond(req, http.StatusNotFound, "text/plain", "not served\n"), nil
}

func (e *env) isMetricsPath(path string) bool {
	if path == "/metrics" {
		return true
	}
	if e.r == nil {
		return false
	}
	for _, src := range e.r.Requirements().MetricsSources {
		if src.Path == path {
			return true
		}
	}
	return false
}

// metrics is a vLLM engine's metrics, idle or saturated (a long queue and a full cache).
func (e *env) metrics(busy bool) string {
	waiting, running, kv := 0, 0, 0.0
	if busy {
		waiting, running, kv = 128, 256, 1.0
	}
	var b strings.Builder
	gauge := func(name string, v any) {
		fmt.Fprintf(&b, "# TYPE %s gauge\n%s{model_name=%q} %v\n", name, name, e.s.Model, v)
	}
	gauge("vllm:num_requests_waiting", waiting)
	gauge("vllm:num_requests_running", running)
	gauge("vllm:kv_cache_usage_perc", kv)
	fmt.Fprintf(&b, "# TYPE vllm:num_preemptions_total counter\nvllm:num_preemptions_total{model_name=%q} 0\n", e.s.Model)
	fmt.Fprintf(&b, "# TYPE vllm:cache_config_info gauge\nvllm:cache_config_info{block_size=\"%d\",num_gpu_blocks=\"4096\"} 1\n", e.blocks)
	return b.String()
}

func completion(id, model string) []byte {
	b, _ := json.Marshal(map[string]any{"id": "cmpl-" + id, "object": "text_completion", "model": model,
		"choices": []map[string]any{{"index": 0, "text": "", "finish_reason": "stop"}},
		"usage":   map[string]int{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
	return b
}

func chunk(id, model string, final bool) []byte {
	c := map[string]any{"id": "cmpl-" + id, "object": "text_completion", "model": model,
		"choices": []map[string]any{{"index": 0, "text": "", "finish_reason": nil}}}
	if !final {
		b, _ := json.Marshal(c)
		return []byte("data: " + string(b) + "\n\n")
	}
	c["choices"] = []map[string]any{{"index": 0, "text": "", "finish_reason": "stop"}}
	c["usage"] = map[string]int{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
	b, _ := json.Marshal(c)
	return []byte("data: " + string(b) + "\n\ndata: [DONE]\n\n")
}

package embedtest

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/embed"
)

// reset discards the part's router and endpoints, so it can build another.
func (e *env) reset() {
	e.stop()
	e.eps, e.flights, e.nextReq = nil, nil, 0
}

// partRoute: a request reaches a ready endpoint, and the router describes itself.
func partRoute(e *env) {
	if !e.build(e.s.Route) {
		return
	}
	pool := e.r.Pool()
	if pool.Name == "" || pool.Namespace == "" || pool.Kind == "" {
		e.fail("Pool() %+v: want a namespace, a name and a kind", pool)
	}
	if fc := e.r.FlowControl(); fc != (embed.FlowControl{}) {
		e.fail("FlowControl() = %+v for a configuration without flow control; want all zero", fc)
	}
	if e.r.Requirements().Disaggregates {
		e.fail("Requirements().Disaggregates for a configuration that does not disaggregate")
	}
	e.endpoints(2)
	for range 4 {
		o := e.route(e.request("", false, e.prompt(1, 1)))
		if !e.wantDispatched(o, e.addresses()) {
			continue
		}
		if o.flight.EvictionSignal() != nil {
			e.fail("%s: an EvictionSignal from a configuration that does not evict", o.req.ID)
		}
		o.end(e)
	}
}

// partReadiness: the router serves only ready endpoints, and stops serving removed ones.
func partReadiness(e *env) {
	if !e.build(e.s.Route) {
		return
	}
	e.endpoints(2)
	a, b := e.eps[0], e.eps[1]
	set := func(ep embed.Endpoint, ready bool) {
		if err := e.r.SetEndpoint(ep, ready); err != nil {
			e.fail("SetEndpoint(%s, %v): %v", ep.Name, ready, err)
		}
		e.rt.Settle()
		e.rt.Advance(e.s.Observe)
	}
	only := func(want embed.Endpoint, why string) {
		for range 4 {
			o := e.route(e.request("", false, e.prompt(1, 1)))
			if o.flight != nil && o.decision.Endpoint != want.Address() {
				e.fail("%s: dispatched to %s %s; want %s", o.req.ID, o.decision.Endpoint, why, want.Address())
			} else if o.flight == nil {
				e.fail("%s: want a dispatch to %s, got %s", o.req.ID, want.Address(), o)
			}
			o.end(e)
		}
	}
	set(a, false)
	only(b, "after it was made not ready")
	set(a, true)
	set(a, true) // setting an unchanged endpoint again changes nothing
	moved := a
	moved.IP = "10.0.1.1"
	set(moved, true) // SetEndpoint updates an endpoint by name: its address too
	e.eps[0] = moved
	if err := e.r.RemoveEndpoint(b.Name); err != nil {
		e.fail("RemoveEndpoint(%s): %v", b.Name, err)
	}
	e.rt.Settle()
	e.rt.Advance(e.s.Observe)
	only(moved, "after the other was removed, at the address it was updated to")
	if err := e.r.RemoveEndpoint("no-such-endpoint"); err != nil {
		e.fail("RemoveEndpoint of an unknown name: %v; want no error", err)
	}
}

// partNoEndpoints: with no endpoint to serve it, a request is refused, NoEndpoints.
func partNoEndpoints(e *env) {
	if !e.build(e.s.Route) {
		return
	}
	e.wantRejected(e.route(e.request("", false, e.prompt(1, 1))), embed.NoEndpoints)
	e.endpoints(1)
	if err := e.r.SetEndpoint(e.eps[0], false); err != nil {
		e.fail("SetEndpoint: %v", err)
	}
	e.rt.Settle()
	e.rt.Advance(e.s.Observe)
	e.wantRejected(e.route(e.request("", false, e.prompt(1, 1))), embed.NoEndpoints)
}

// partAdmission: while every endpoint is saturated, a sheddable request is refused,
// Admission, and one that is not is still dispatched.
func partAdmission(e *env) {
	if !e.build(e.s.Route) {
		return
	}
	e.endpoints(2)
	e.saturate(true)
	e.wantRejected(e.route(e.request(Sheddable, false, e.prompt(1, 1))), embed.Admission)
	o := e.route(e.request("", false, e.prompt(1, 1)))
	if e.wantDispatched(o, e.addresses()) {
		o.end(e)
	}
	e.saturate(false)
	o = e.route(e.request(Sheddable, false, e.prompt(1, 1)))
	if o.flight == nil {
		e.fail("%s: a sheddable request to idle endpoints: want a dispatch, got %s", o.req.ID, o)
	}
	o.end(e)
}

// partUnclassified: a request whose body is not JSON is no routing outcome the contract names.
func partUnclassified(e *env) {
	if !e.build(e.s.Route) {
		return
	}
	e.endpoints(1)
	req := e.request("", false, nil)
	req.Body = []byte("{not json")
	e.wantRejected(e.route(req), embed.Unclassified)
}

// partUnavailable: a configuration that can serve no request refuses one, Unavailable, and not
// Unclassified: a host counts the two differently.
func partUnavailable(e *env) {
	if e.s.Unavailable == nil {
		e.skipped = "Suite.Unavailable is nil: no configuration that serves no request was supplied, so the Unavailable outcome is not checked"
		return
	}
	if !e.build(*e.s.Unavailable) {
		return
	}
	e.endpoints(2)
	e.wantRejected(e.route(e.request("", false, e.prompt(1, 1))), embed.Unavailable)
}

// partResponses: the router sees each response's headers and body, streamed or not, and each
// way a response can end early.
func partResponses(e *env) {
	if !e.build(e.s.Route) {
		return
	}
	e.endpoints(1)
	headers := func(o *outcome, ct string) {
		in := map[string]string{":status": "200", "content-type": ct}
		out := o.flight.ResponseHeaders(in)
		if out == nil {
			e.fail("%s: ResponseHeaders returned nil; want the headers after the router's handling", o.req.ID)
			return
		}
		for k, v := range in {
			if out[k] != v {
				e.fail("%s: ResponseHeaders returned %s=%q; want every header passed in, %s=%q", o.req.ID, k, out[k], k, v)
			}
		}
		e.rt.Settle()
	}

	if o := e.route(e.request("", false, e.prompt(1, 1))); e.wantDispatched(o, e.addresses()) {
		headers(o, "application/json")
		o.flight.ResponseBody(completion(o.req.ID, e.s.Model), true)
		o.ended = true
		e.rt.Settle()
	}
	if o := e.route(e.request("", true, e.prompt(1, 1))); e.wantDispatched(o, e.addresses()) {
		headers(o, "text/event-stream")
		o.flight.ResponseBody(chunk(o.req.ID, e.s.Model, false), false)
		e.rt.Settle()
		o.flight.ResponseBody(chunk(o.req.ID, e.s.Model, true), true)
		o.ended = true
		e.rt.Settle()
	}
	for _, cause := range []embed.Termination{embed.ClientDisconnect, embed.Evicted, embed.Error} {
		o := e.route(e.request("", true, e.prompt(1, 1)))
		if !e.wantDispatched(o, e.addresses()) {
			continue
		}
		if cause != embed.Error {
			headers(o, "text/event-stream")
		}
		o.flight.Terminate(cause)
		o.ended = true
		e.rt.Settle()
	}
	// The router still routes after every kind of ending.
	if o := e.route(e.request("", false, e.prompt(1, 1))); e.wantDispatched(o, e.addresses()) {
		o.end(e)
	}
}

// partDeterminism: on the same seed and the same calls, the router decides the same way.
func partDeterminism(e *env) {
	run := func() []string {
		if !e.build(e.s.Route) {
			return nil
		}
		e.endpoints(4)
		var got []string
		for i := range 13 { // not a multiple of the endpoints, so a pick order that carries over between runs shows
			o := e.route(e.request("", false, e.prompt(uint32(1+100*i), 1+i%3)))
			got = append(got, o.String())
			o.end(e)
		}
		e.reset()
		return got
	}
	first, second := run(), run()
	if first != nil && second != nil && !slices.Equal(first, second) {
		e.fail("two runs on seed %d made different decisions:\n  %s\n  %s", e.s.Seed,
			strings.Join(first, ", "), strings.Join(second, ", "))
	}
}

// partHold: flow control holds a request while the endpoints are saturated, refuses one over
// its capacity (Saturated), releases one whose client gives up (ContextCancelled) or whose
// budget runs out (TTLExpired), dispatches a held one once capacity returns, and decides
// nothing it still holds when it stops.
func partHold(e *env) {
	if !e.build(e.s.Hold) {
		return
	}
	fc := e.r.FlowControl()
	if !fc.Enabled || fc.MaxQueueTTL <= 0 {
		e.fail("FlowControl() = %+v; the Hold configuration has flow control with a bounded budget", fc)
		return
	}
	release := fc.MaxQueueTTL + fc.ExpirySweep + e.s.Observe
	e.endpoints(1)
	e.saturate(true)

	held := e.route(e.request("", false, e.prompt(1, 1)))
	if !e.wantHeld(held) {
		return
	}
	e.wantRejected(e.route(e.request("", false, e.prompt(1, 1))), embed.Saturated)

	held.cancel()
	e.rt.Settle()
	e.rt.Advance(fc.ExpirySweep + e.s.Observe)
	e.wantRejected(held, embed.ContextCancelled)

	expiring := e.route(e.request("", false, e.prompt(1, 1)))
	if e.wantHeld(expiring) {
		e.rt.Advance(release)
		e.wantRejected(expiring, embed.TTLExpired)
	}

	// A request's own TTL, shorter than the configured budget, is its budget.
	if short := fc.MaxQueueTTL / 4; short > 0 {
		req := e.request("", false, e.prompt(1, 1))
		req.TTL = short
		own := e.route(req)
		if e.wantHeld(own) {
			e.rt.Advance(short + fc.ExpirySweep + e.s.Observe)
			if !own.decided() {
				e.fail("%s: Request.TTL %v did not bound its wait: still held after %v (the configured budget is %v)",
					own.req.ID, short, short+fc.ExpirySweep+e.s.Observe, fc.MaxQueueTTL)
				e.rt.Advance(release)
			}
			e.wantRejected(own, embed.TTLExpired)
		}
	}

	waiting := e.route(e.request("", false, e.prompt(1, 1)))
	if e.wantHeld(waiting) {
		e.saturate(false)
		e.rt.Advance(e.s.Observe)
		if e.wantDispatched(waiting, e.addresses()) {
			waiting.end(e)
		}
	}

	// A request still held when the router stops is never decided (stop checks it).
	e.saturate(true)
	e.wantHeld(e.route(e.request("", false, e.prompt(1, 1))))
}

// partKV: the router's view of each engine's cache is the KV events it was given: stored
// blocks attract a request with that prefix, and removed or cleared blocks no longer do.
func partKV(e *env) {
	if !e.build(e.s.KV) {
		return
	}
	if !e.r.WantsKVEvents() {
		e.fail("WantsKVEvents() is false for the KV configuration")
		return
	}
	if len(e.r.Requirements().PrefixIndexes) == 0 {
		e.fail("Requirements().PrefixIndexes is empty for the KV configuration")
	}
	e.endpoints(2)
	a, b := e.eps[0].Address(), e.eps[1].Address()
	const blocks = 4
	prompt := e.prompt(1, blocks)
	ingest := func(addr string, ev ...embed.KVEvent) {
		if err := e.r.IngestKVEvents(e.ctx(), addr, e.s.Model, embed.KVBatch{Timestamp: e.rt.Now(), Events: ev}); err != nil {
			e.fail("IngestKVEvents(%s): %v", addr, err)
		}
		e.rt.Settle()
		e.rt.Advance(e.s.Observe)
	}
	goes := func(want, why string) {
		o := e.route(e.request("", false, prompt))
		if o.flight == nil {
			e.fail("%s: want a dispatch to %s (%s), got %s", o.req.ID, want, why, o)
			return
		}
		if o.decision.Endpoint != want {
			e.fail("%s: dispatched to %s; want %s, %s", o.req.ID, o.decision.Endpoint, want, why)
		}
		o.end(e)
	}

	inspector, _ := e.r.(embed.KVIndexInspector)
	holds := func(addr string, want []uint64, when string) {
		if inspector == nil {
			return
		}
		got, err := inspector.KVBlocks(addr, "")
		if err != nil {
			e.fail("KVBlocks(%s) %s: %v", addr, when, err)
			return
		}
		if len(got) != len(want) {
			e.fail("KVBlocks(%s) %s holds %d blocks; want %d (%v)", addr, when, len(got), len(want), want)
		}
		for _, h := range want {
			if !got[h] {
				e.fail("KVBlocks(%s) %s lacks block %#x", addr, when, h)
			}
		}
	}

	ingest(b, e.stored(prompt, blocks, 0))
	holds(b, e.hashes(0, blocks), "after its blocks were stored")
	goes(b, "the only endpoint that published the prompt's blocks")

	ingest(b, embed.BlockRemoved{Hashes: e.hashes(0, blocks)})
	holds(b, nil, "after its blocks were removed")
	ingest(a, e.stored(prompt, 2, 0))
	goes(a, "the other endpoint removed the blocks it had published")

	ingest(b, e.stored(prompt, blocks, 0))
	goes(b, "it published the longer prefix")
	if err := e.r.ClearKVEvents(e.ctx(), b); err != nil {
		e.fail("ClearKVEvents(%s): %v", b, err)
	}
	e.rt.Settle()
	e.rt.Advance(e.s.Observe)
	holds(b, nil, "after it was cleared")
	goes(a, "the other endpoint's blocks were cleared")
}

// stored is a BlockStored of the first n blocks of prompt, keyed from base.
func (e *env) stored(prompt []uint32, n int, base uint64) embed.BlockStored {
	return embed.BlockStored{Hashes: e.hashes(base, n), Tokens: prompt[:n*e.blocks], BlockSize: e.blocks}
}

// hashes are n engine block keys from base: non-zero, as zero is vLLM's "no parent".
func (e *env) hashes(base uint64, n int) []uint64 {
	out := make([]uint64, n)
	for i := range out {
		out[i] = 0x1000 + base + uint64(i)
	}
	return out
}

func (e *env) ctx() context.Context { return context.Background() }

// partEvict: under saturation, a held request makes the router revoke a dispatched sheddable
// one through its eviction signal.
func partEvict(e *env) {
	if e.s.Evict == nil {
		e.skipped = "Suite.Evict is nil: no configuration with eviction enabled was supplied, so the eviction signal is not checked"
		return
	}
	if !e.build(*e.s.Evict) {
		return
	}
	e.endpoints(1)
	victim := e.route(e.request(Sheddable, true, e.prompt(1, 1)))
	if !e.wantDispatched(victim, e.addresses()) {
		return
	}
	signal := victim.flight.EvictionSignal()
	if signal == nil {
		e.fail("%s: EvictionSignal is nil from a configuration that evicts", victim.req.ID)
		return
	}
	victim.flight.ResponseHeaders(map[string]string{":status": "200", "content-type": "text/event-stream"})
	e.rt.Settle()
	e.saturate(true)
	held := e.route(e.request("", false, e.prompt(1, 1)))
	e.rt.Advance(e.s.EvictWithin)
	select {
	case <-signal:
		victim.flight.Terminate(embed.Evicted)
		victim.ended = true
		e.rt.Settle()
	default:
		e.fail("%s: not evicted within %v of a held request under saturation", victim.req.ID, e.s.EvictWithin)
	}
	e.saturate(false)
	e.rt.Advance(e.s.Observe)
	if held.flight != nil {
		held.end(e)
	}
}

// partCheck: a host check that fails stops construction before any plugin is built: its
// error comes back as is, Allow is never consulted, and nothing is left running. A check that
// passes sees the Requirements and Pool the router then reports.
func partCheck(e *env) {
	errHost := errors.New("refused by the embedtest host check")
	var asked []string
	calls := 0
	cfg := e.s.Refuse
	cfg.Allow = func(typ string) (bool, string) {
		asked = append(asked, typ)
		return false, "Allow consulted after a failing Check"
	}
	cfg.Check = func(embed.Requirements, embed.Pool) error {
		calls++
		return errHost
	}
	e.rt = NewRuntime(e.s.Seed)
	e.rt.Engines = e.serve
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r, err := e.s.New(ctx, e.rt, cfg)
	switch {
	case err == nil:
		e.fail("construction succeeded although Check failed")
		e.r, e.cancel = r, cancel
		e.reset()
	case err != errHost:
		e.fail("construction failed with %q; want Check's error, unwrapped", err)
	}
	if calls != 1 {
		e.fail("Check was called %d times; want once", calls)
	}
	if len(asked) > 0 {
		e.fail("Allow was consulted (%v) after Check failed; Check runs before any plugin", asked)
	}
	if err != nil {
		e.rt.stopped = true
		e.rt.Settle()
		e.rt.Advance(10 * e.s.Observe)
		if n := e.rt.Live(); n != 0 {
			e.fail("a construction Check refused left %d goroutines running", n)
		}
		if n := e.rt.loopsStopped; n != 0 {
			e.fail("a construction Check refused left an Every loop running (%d calls)", n)
		}
	}

	var gotReq embed.Requirements
	var gotPool embed.Pool
	cfg = e.s.Route
	cfg.Check = func(r embed.Requirements, p embed.Pool) error {
		gotReq, gotPool = r, p
		return nil
	}
	if !e.build(cfg) {
		return
	}
	if !reflect.DeepEqual(gotReq, e.r.Requirements()) {
		e.fail("Check saw Requirements %+v; the router reports %+v", gotReq, e.r.Requirements())
	}
	if !reflect.DeepEqual(gotPool, e.r.Pool()) {
		e.fail("Check saw Pool %+v; the router reports %+v", gotPool, e.r.Pool())
	}
}

// partUnaudited: a type the router has not audited is refused even when Allow permits it,
// after Allow was consulted (the router's own stated reasons come first; see Config.Allow).
func partUnaudited(e *env) {
	if e.s.Unaudited == "" {
		e.skipped = "Suite.Unaudited is empty: no unaudited plugin type was supplied, so the refusal of an unaudited type is not checked"
		return
	}
	var asked []string
	cfg := e.s.UnauditedConfig
	cfg.Allow = func(typ string) (bool, string) {
		asked = append(asked, typ)
		return true, ""
	}
	e.rt = NewRuntime(e.s.Seed)
	e.rt.Engines = e.serve
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r, err := e.s.New(ctx, e.rt, cfg)
	var re *embed.RefusalError
	switch {
	case err == nil:
		e.fail("construction succeeded although %s has not been audited; Allow permitting it must not be enough", e.s.Unaudited)
		e.r, e.cancel = r, cancel
		e.reset()
		return
	case !errors.As(err, &re):
		e.fail("construction failed with %v; want an error wrapping *embed.RefusalError", err)
	case re.Type != e.s.Unaudited:
		e.fail("RefusalError %+v; want type %s", *re, e.s.Unaudited)
	case re.Reason == "":
		e.fail("RefusalError for %s has no reason", e.s.Unaudited)
	}
	if !slices.Contains(asked, e.s.Unaudited) {
		e.fail("Allow was never asked about %s; an unaudited type with no stated reason is still put to Allow first (consulted %v)", e.s.Unaudited, asked)
	}
}

// partRefuse: a plugin type Allow refuses fails construction with a RefusalError naming it,
// leaving nothing running; the same configuration with no policy constructs.
func partRefuse(e *env) {
	if e.s.RefuseType == "" {
		e.fail("Suite.RefuseType is empty")
		return
	}
	const reason = "refused by the embedtest policy"
	var asked []string
	cfg := e.s.Refuse
	cfg.Allow = func(typ string) (bool, string) {
		asked = append(asked, typ)
		if typ == e.s.RefuseType {
			return false, reason
		}
		return true, ""
	}
	e.rt = NewRuntime(e.s.Seed)
	e.rt.Engines = e.serve
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r, err := e.s.New(ctx, e.rt, cfg)
	var re *embed.RefusalError
	switch {
	case err == nil:
		e.fail("construction succeeded although Allow refused %s", e.s.RefuseType)
		e.r, e.cancel = r, cancel
		e.reset()
	case !errors.As(err, &re):
		e.fail("construction failed with %v; want an error wrapping *embed.RefusalError", err)
	case re.Type != e.s.RefuseType || re.Reason != reason:
		e.fail("RefusalError %+v; want type %s and Allow's reason %q", *re, e.s.RefuseType, reason)
	}
	if err != nil {
		// A Constructor that fails has stopped what it started, without its context ending.
		e.rt.stopped = true
		e.rt.Settle()
		e.rt.Advance(10 * e.s.Observe)
		if n := e.rt.Live(); n != 0 {
			e.fail("a failed construction left %d goroutines running", n)
		}
		if n := e.rt.loopsStopped; n != 0 {
			e.fail("a failed construction left an Every loop running (%d calls)", n)
		}
	}
	if !slices.Contains(asked, e.s.RefuseType) {
		e.fail("Allow was never asked about %s; consulted %v", e.s.RefuseType, asked)
	}
	cfg.Allow = nil
	if e.build(cfg) {
		e.endpoints(1)
		o := e.route(e.request("", false, e.prompt(1, 1)))
		if o.flight == nil {
			e.fail("with no policy the Refuse configuration should route; got %s", o)
		}
		o.end(e)
	}
}

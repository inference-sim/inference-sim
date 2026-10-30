package sim

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/engine"
)

// P/D handoffs through engine.Engine. A disaggregated prefill reaches its engine as
// engine.Request.RemoteDecode: the engine keeps its KV for the decode pull and reports it in
// Result.Handoffs, and the host ends the hold with TransferDone (ReleaseHandoff). A decode
// sub-request reaches its engine as RemotePrefill when its transfer may start (AddRemote): the
// engine holds it and starts its pull by its own admission rule, a TransferLoad among a batch's
// TransfersStarted, which the host prices (OnPullStarted); when the pull lands, the host reports
// it done (LandRemote), and the engine runs the request.

// AddRemote hands the engine a decode sub-request whose KV a peer's prefill computed, at now.
// An engine with no other work admits it at once, as vLLM's idle engine schedules a new request
// (it is stepped here); one with work admits it in a later schedule.
func (d *EngineDriver) AddRemote(req *Request, now int64) {
	er := d.engineRequest(req)
	er.RemotePrefill = true
	// Whether the engine had other work is read before the handoff reaches it: an engine that
	// queues the handoff as a waiting request (blis-vllm's v0.32.0 engine) would otherwise count
	// it as that work, and an idle engine would never be stepped to start the pull.
	before := d.eng.Snapshot()
	if err := d.eng.Add(er, now); err != nil {
		panic(fmt.Sprintf("engine driver: AddRemote(%s): %v", req.ID, err))
	}
	if d.remote == nil {
		d.remote, d.pulls = map[string]*Request{}, map[string]engine.TransferID{}
	}
	d.remote[req.ID] = req
	if before.Waiting == 0 && before.Running == 0 {
		d.kick(now)
		return
	}
	d.rearmWake(now)
}

// startTransfers reports the pulls a schedule at now started to the host.
func (d *EngineDriver) startTransfers(ts []engine.Transfer, now int64) {
	for _, t := range ts {
		req := d.remote[t.Request]
		if t.Kind != engine.TransferLoad || req == nil {
			panic(fmt.Sprintf("engine driver: the engine started a %s transfer for %q, which is not a waiting handoff",
				t.Kind, t.Request))
		}
		d.pulls[t.Request] = t.ID
		if d.OnPullStarted == nil {
			panic(fmt.Sprintf("engine driver: the engine started the pull of %s, and no host prices it", t.Request))
		}
		d.OnPullStarted(req, int64(t.Tokens), now)
	}
}

// LandRemote reports that req's pull has landed at now: the engine runs it from here.
func (d *EngineDriver) LandRemote(req *Request, now int64) {
	id, ok := d.pulls[req.ID]
	if !ok {
		panic(fmt.Sprintf("engine driver: LandRemote(%s): no pull of it is in flight", req.ID))
	}
	delete(d.pulls, req.ID)
	delete(d.remote, req.ID)
	if err := d.eng.TransferDone(id, true, now); err != nil {
		panic(fmt.Sprintf("engine driver: TransferDone(%d) for %s: %v", id, req.ID, err))
	}
	req.State = StateQueued
	// A remote-prefilled request's prompt was looked up on its prefill engine: none is recorded here.
	d.reqs[req.ID] = &driverReq{req: req, remote: true, lookedUp: true}
	if req.Deadline > 0 && req.Deadline <= d.Horizon {
		d.schedule(&driverTimeoutEvent{at: req.Deadline, req: req})
	}
	d.stepAfterArrivals(now)
}

// FailRemote ends req's pull at now without its KV landing: what it reserved is freed.
func (d *EngineDriver) FailRemote(req *Request, now int64) {
	id, ok := d.pulls[req.ID]
	if !ok {
		panic(fmt.Sprintf("engine driver: FailRemote(%s): no pull of it is in flight", req.ID))
	}
	delete(d.pulls, req.ID)
	delete(d.remote, req.ID)
	if err := d.eng.TransferDone(id, false, now); err != nil {
		panic(fmt.Sprintf("engine driver: TransferDone(%d, failed) for %s: %v", id, req.ID, err))
	}
	d.rearmWake(now)
}

// AbortRemote ends a handoff the engine holds before its pull landed, at now (Finish, abort).
func (d *EngineDriver) AbortRemote(req *Request, now int64) {
	delete(d.pulls, req.ID)
	delete(d.remote, req.ID)
	if err := d.eng.Finish(req.ID, engine.FinishAbort, now); err != nil {
		panic(fmt.Sprintf("engine driver: Finish(%s, abort): %v", req.ID, err))
	}
	d.kick(now)
}

// HandoffBlocks are the KV blocks the engine holds for req's pull (Result.Handoffs).
func (d *EngineDriver) HandoffBlocks(req *Request) []int64 {
	h, ok := d.holds[req.ID]
	if !ok {
		panic(fmt.Sprintf("engine driver: %s left no KV for a pull", req.ID))
	}
	return h.Blocks
}

// ReleaseHandoff ends the hold of req's prefill KV at now: the pull completed, or the hold ran out.
func (d *EngineDriver) ReleaseHandoff(req *Request, now int64) {
	h, ok := d.holds[req.ID]
	if !ok {
		panic(fmt.Sprintf("engine driver: ReleaseHandoff(%s): the engine holds no KV for it", req.ID))
	}
	delete(d.holds, req.ID)
	if err := d.eng.TransferDone(h.Transfer, true, now); err != nil {
		panic(fmt.Sprintf("engine driver: TransferDone(%d) for %s: %v", h.Transfer, req.ID, err))
	}
	// The freed blocks may admit a request waiting for KV, so the engine is stepped now, as for
	// any other call that frees KV (Evict, AbortRemote): the host loop woke a parked engine when
	// a hold released (WakeIfParked).
	d.kick(now)
}

// noteHandoffs keeps the KV holds a completed batch reported.
func (d *EngineDriver) noteHandoffs(hs []engine.Handoff) {
	for _, h := range hs {
		if d.holds == nil {
			d.holds = map[string]engine.Handoff{}
		}
		d.holds[h.ID] = h
	}
}

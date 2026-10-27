// Original to blis-vllm.

package contract

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/engine"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"
)

// A P/D handoff through engine.Engine. The prefill engine keeps a finished disaggregated
// prefill's KV (engine.Request.RemoteDecode) and reports it in Result.Handoffs; the simulator
// ends the hold with TransferDone. The decode engine is handed the request as RemotePrefill and
// holds it among its waiting handoffs, in the order they reached it, until its admission rule
// starts the pull (cengine.EngineCore.AdmitRemoteKV, INV-PD-9): inside a schedule, where vLLM's
// WAITING phase reaches it (SetRemoteKVAdmission), or at once when the engine has no other work,
// as vLLM's idle engine schedules a new request at once. Each pull is a TransferLoad among the
// batch's TransfersStarted, sized by the prompt tokens it must move; when the simulator reports
// it done the pulled blocks enter the prefix cache and the request waits to decode, as
// inference-sim's cluster used to do through the compatibility engine's own methods.
type handoffs struct {
	// waiting are the handoffs the decode engine holds before their pull starts, in arrival order.
	waiting []*cengine.Request
	// pulls are the pulls in flight, and held the finished prefills whose KV is held, by transfer.
	pulls map[engine.TransferID]*cengine.Request
	held  map[engine.TransferID]*cengine.Request
	next  engine.TransferID
	// started are the transfers the current call started, for the batch it returns.
	started []engine.Transfer
	armed   bool
}

func (h *handoffs) newTransfer() engine.TransferID {
	h.next++
	return h.next
}

func (h *handoffs) take() []engine.Transfer {
	s := h.started
	h.started = nil
	return s
}

// addRemote holds a remote-prefilled request among the decode engine's waiting handoffs.
func (e *Engine) addRemote(r engine.Request, now engine.Time) error {
	for _, w := range e.ho.waiting {
		if w.ID == r.ID {
			return fmt.Errorf("contract: request %s added twice", r.ID)
		}
	}
	e.now = now
	req := &cengine.Request{ID: r.ID, InputTokens: tokens(r.Prompt), OutputTokens: tokens(r.Output),
		MaxOutputLen: r.MaxOutputTokens, ArrivalTime: r.ArrivalTime, Adapter: r.Adapter,
		State: cengine.StateWaitingForRemoteKVs, SLOClass: r.Tags[TagSLOClass], IsDecodeSubRequest: true}
	if !e.ho.armed {
		e.ho.armed = true
		e.core.SetRemoteKVAdmission(func(int64) { e.admitRemote() })
	}
	e.ho.waiting = append(e.ho.waiting, req)
	e.core.SetRemoteKVWaiting(len(e.ho.waiting))
	return nil
}

// admitRemote starts the pulls the engine admits, in its own order, and keeps the rest waiting.
func (e *Engine) admitRemote() {
	if len(e.ho.waiting) == 0 {
		return
	}
	started := map[*cengine.Request]bool{}
	for _, a := range e.core.AdmitRemoteKV(e.ho.waiting) {
		id := e.ho.newTransfer()
		if e.ho.pulls == nil {
			e.ho.pulls = map[engine.TransferID]*cengine.Request{}
		}
		e.ho.pulls[id] = a.Req
		started[a.Req] = true
		e.ho.started = append(e.ho.started, engine.Transfer{ID: id, Request: a.Req.ID, Kind: engine.TransferLoad,
			Tokens: int(a.PulledTokens)})
	}
	rest := e.ho.waiting[:0:0]
	for _, w := range e.ho.waiting {
		if !started[w] {
			rest = append(rest, w)
		}
	}
	e.ho.waiting = rest
	e.core.SetRemoteKVWaiting(len(rest))
}

// holdHandoffs reports the finished prefills whose KV the engine keeps for a pull.
func (e *Engine) holdHandoffs(finished []engine.Finished) []engine.Handoff {
	var out []engine.Handoff
	for _, f := range finished {
		req := e.reqs[f.ID]
		if req == nil || !req.RetainKVOnFinish {
			continue
		}
		blocks, _ := e.core.RequestBlocks(req)
		id := e.ho.newTransfer()
		if e.ho.held == nil {
			e.ho.held = map[engine.TransferID]*cengine.Request{}
		}
		e.ho.held[id] = req
		out = append(out, engine.Handoff{ID: f.ID, Transfer: id, Blocks: append([]int64(nil), blocks...)})
	}
	return out
}

// TransferDone ends a handoff's hold, freeing the prefill's KV, or lands a pull: its blocks enter
// the prefix cache and the request waits to decode (EnqueueDecodeSubRequest). A pull that failed
// frees what it reserved.
func (e *Engine) TransferDone(id engine.TransferID, ok bool, now engine.Time) error {
	if req, held := e.ho.held[id]; held {
		delete(e.ho.held, id)
		e.now = now
		e.core.ReleaseKV(req)
		return nil
	}
	req, pulling := e.ho.pulls[id]
	if !pulling {
		return engine.ErrUnknownTransfer
	}
	delete(e.ho.pulls, id)
	e.now = now
	e.core.RemoteKVLanded(req)
	if !ok {
		e.core.ReleaseKV(req)
		return nil
	}
	e.core.CacheRemoteKV(req)
	req.State = cengine.StateQueued
	e.reqs[req.ID] = req
	e.core.EnqueueDecodeSubRequest(req, now)
	return nil
}

// finishRemote ends a handoff the decode engine holds, waiting or pulling: what its pull reserved
// is freed, and the engine's next step publishes the state after it (NoteAbort).
func (e *Engine) finishRemote(id string, now engine.Time) bool {
	for i, w := range e.ho.waiting {
		if w.ID == id {
			e.ho.waiting = append(e.ho.waiting[:i:i], e.ho.waiting[i+1:]...)
			e.core.SetRemoteKVWaiting(len(e.ho.waiting))
			return true
		}
	}
	for tid, req := range e.ho.pulls {
		if req.ID == id {
			delete(e.ho.pulls, tid)
			e.now = now
			e.core.RemoteKVLanded(req)
			e.core.ReleaseKV(req)
			e.core.NoteAbort(now)
			return true
		}
	}
	return false
}

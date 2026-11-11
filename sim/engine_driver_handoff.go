package sim

import (
	"errors"
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/env"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// P/D handoffs through engine.Engine. A disaggregated prefill reaches its engine as
// engine.Request.RemoteDecode: the engine keeps its KV for the decode pull and reports it in
// Result.Handoffs, and the host ends the hold with TransferDone (ReleaseHandoff). A decode
// sub-request reaches its engine as RemotePrefill when its transfer may start (AddRemote): the
// engine holds it and starts its pull by its own admission rule, a TransferLoad among a batch's
// TransfersStarted. The driver prices it through the Environment (priceTransfer, from the
// Move the engine states) and the host shares the links among the pulls in flight
// (OnPullPriced); when the pull lands, the host reports it done (LandRemote), and the engine runs
// the request.

// AddRemote hands the engine a decode sub-request whose KV a peer's prefill computed, at now.
// peer is where that prefill's workers sit, in the prefill pool's own numbering: the engine
// learns where it pulls from only through the request (engine.Request.Peer). An engine with no
// other work admits it at once, as vLLM's idle engine schedules a new request (it is stepped
// here); one with work admits it in a later schedule.
func (d *EngineDriver) AddRemote(req *Request, peer plan.Endpoint, now int64) {
	er := d.engineRequest(req)
	er.RemotePrefill = true
	er.Peer = &peer
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

// startTransfers reports the transfers a schedule at now that ran no forward pass started: its
// offload-tier copies start once the compute already queued ends (startTierTransfer).
func (d *EngineDriver) startTransfers(ts []engine.Transfer, now int64) {
	d.startTransfersAfter(ts, now, max(now, d.BusyUntil()))
}

// startTransfersAfter prices every transfer a schedule at now started, through the Environment,
// hands the pulls to the host with their demand, and starts its offload-tier copies no earlier
// than computeEnd, when the compute stream they wait on frees.
func (d *EngineDriver) startTransfersAfter(ts []engine.Transfer, now, computeEnd int64) {
	for _, t := range ts {
		demand := d.priceTransfer(t)
		if t.Tier != "" {
			d.startTierTransfer(t, demand, computeEnd)
			continue
		}
		req := d.remote[t.Request]
		if t.Kind != engine.TransferLoad || req == nil {
			panic(fmt.Sprintf("engine driver: the engine started a %s transfer for %q, which is not a waiting handoff",
				t.Kind, t.Request))
		}
		d.pulls[t.Request] = t.ID
		if d.OnPullPriced == nil {
			panic(fmt.Sprintf("engine driver: the engine started the pull of %s, and no host runs it", t.Request))
		}
		d.OnPullPriced(req, int64(t.Tokens), demand, now)
	}
}

// priceTransfer is what t demands of the fabric: env.DemandFor(Environment, plan, t), where plan
// is the PoolPlan the engine states (engine.Planned) and the move is the engine's own
// (engine.Transfer.Move). The driver derives nothing: the tier, the endpoints and the direction
// are the engine's, the price the cost model's. An engine that starts a transfer and is not
// Planned, a host with no Environment, and a transfer the contract or the cost model refuses are
// each a failure of the run, never a transfer priced at zero.
func (d *EngineDriver) priceTransfer(t engine.Transfer) kernel.TransferDemand {
	p, ok := d.eng.(engine.Planned)
	if !ok {
		panic(fmt.Sprintf("engine driver: the engine started a %s transfer for %q and states no plan "+
			"(engine.Planned), so the cost model cannot price it", t.Kind, t.Request))
	}
	if d.cfg.Environment == nil {
		panic(fmt.Sprintf("engine driver: the engine started a %s transfer for %q, and the driver has no "+
			"Environment to price it", t.Kind, t.Request))
	}
	demand, err := env.DemandFor(d.cfg.Environment, p.Plan(), t)
	if err != nil {
		panic(fmt.Sprintf("engine driver: transfer %d of %q: %v", t.ID, t.Request, err))
	}
	return demand
}

// TierTicks is how long an offload-tier copy takes with its links to itself, in whole ticks
// (microseconds) truncated and at least one. The truncation is the one the tier copies have
// always had (the engine priced them as the cost model's TierTime in whole microseconds); a P/D
// pull is rounded up instead (cluster.AloneTicks), since a pull is not done until its last byte
// lands. The two rules are kept apart so that moving the pricing to the Environment moves no
// price.
func TierTicks(d kernel.TransferDemand) int64 {
	return max(1, d.AloneTime().Microseconds())
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

// FailRemote ends req's pull at now without its KV landing, the cluster having dropped the
// request (its decode instance went unroutable): the engine ends the request as well (Finish,
// abort) and is told the pull is over, so what the pull reserved is freed and nothing runs for
// it. Failing the pull alone would leave the request to the engine's kv_load_failure_policy,
// which under "recompute" prefills it locally (vllm/v1/core/sched/scheduler.py:3132-3153,
// 3341-3354@5df63a72), a request the cluster no longer counts.
func (d *EngineDriver) FailRemote(req *Request, now int64) {
	if _, ok := d.pulls[req.ID]; !ok {
		panic(fmt.Sprintf("engine driver: FailRemote(%s): no pull of it is in flight", req.ID))
	}
	d.endRemote(req, now)
	// The engine's next output frees what the pull held. FailRemote can run inside a kick (the
	// cluster prices a pull as the engine starts it, startPull), so the step is queued, not run.
	d.stepAfterArrivals(now)
}

// AbortRemote ends a handoff the engine holds before its pull landed, at now (Finish, abort).
func (d *EngineDriver) AbortRemote(req *Request, now int64) {
	d.endRemote(req, now)
	d.kick(now)
}

// endRemote ends a handoff the host gives up on before its pull landed: the engine finishes the
// request (finish_requests, FINISHED_ABORTED), and a pull in flight for it is reported over. vLLM
// delays the free of a request whose load is in flight until the worker reports the load done
// (finish_requests' delay_free_blocks, then _update_from_kv_xfer_finished's free of a finished
// request, scheduler.py:2690-2700, 3226-3231@5df63a72); unreported, its blocks stay held for the
// rest of the run. The pull is reported over at now, which frees the blocks when the host gives
// up rather than when the abandoned read would have ended.
func (d *EngineDriver) endRemote(req *Request, now int64) {
	id, pulling := d.pulls[req.ID]
	delete(d.pulls, req.ID)
	delete(d.remote, req.ID)
	if err := d.eng.Finish(req.ID, engine.FinishAbort, now); err != nil {
		panic(fmt.Sprintf("engine driver: Finish(%s, abort): %v", req.ID, err))
	}
	d.noteFreed(req.ID)
	if pulling {
		// An engine that ends the pull with the request (the compatibility engine's Finish
		// forgets it) no longer knows the transfer, which is not an error here.
		if err := d.eng.TransferDone(id, false, now); err != nil && !errors.Is(err, engine.ErrUnknownTransfer) {
			panic(fmt.Sprintf("engine driver: TransferDone(%d, failed) for %s: %v", id, req.ID, err))
		}
	}
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

// startTierTransfer runs an engine's load from or store to an offload tier (Transfer.Tier,
// blis-vllm's OffloadingConnector) for as long as its demand takes alone, and reports it done
// when it lands. The tier's streams are the engine's (below), so a copy is never slowed by
// another: its price is the demand with every link to itself.
// vLLM's CPU offload worker has one handler a direction (store_handler, load_handler), each
// starting a transfer only after the one before it on that handler ends (stream.wait_event of the
// last transfer's end event), and every transfer's stream first waits on the compute stream
// (stream.wait_stream(current_stream)), so it starts after the forward pass of the step that
// carried it (SingleDirectionOffloadingHandler.transfer_async, vllm/v1/kv_offload/cpu/
// gpu_worker.py:665-670, 834-852@5df63a72): a step's loads are submitted in its post_forward and
// its stores in the next step's pre_forward, before that step's forward is queued
// (vllm/v1/worker/gpu/kv_connector.py:81, 131-138; offloading/worker.py:218-236). Loads therefore never overlap each other, nor stores each other, but a load and a
// store do.
func (d *EngineDriver) startTierTransfer(t engine.Transfer, demand kernel.TransferDemand, after int64) {
	if d.tierFreeAt == nil {
		d.tierFreeAt = map[engine.TransferKind]int64{}
	}
	start := max(after, d.tierFreeAt[t.Kind])
	d.tierFreeAt[t.Kind] = start + TierTicks(demand)
	d.schedule(&driverTierTransferDone{at: d.tierFreeAt[t.Kind], id: t.ID})
}

// driverTierTransferDone reports an offload-tier transfer landed. Transfer completions run after
// the instant's arrivals and before its steps (the engine contract's same-instant order).
type driverTierTransferDone struct {
	at int64
	id engine.TransferID
}

func (e *driverTierTransferDone) Timestamp() int64 { return e.at }
func (e *driverTierTransferDone) Priority() int    { return PriorityQueued }
func (e *driverTierTransferDone) run(d *EngineDriver) {
	if err := d.eng.TransferDone(e.id, true, e.at); err != nil {
		panic(fmt.Sprintf("engine driver: TransferDone(%d) of a tier transfer: %v", e.id, err))
	}
	d.kick(e.at)
}

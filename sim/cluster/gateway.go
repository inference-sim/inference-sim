package cluster

import (
	"container/heap"
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
)

// Gateway is everything between a request's arrival at the cluster and its dispatch to model
// servers: admission, flow control and endpoint picking. The cluster has no routing logic of its
// own; the llm-d router adapter (sim/llmd) is the production implementation.
//
// The cluster calls a Gateway only from its event loop. A Gateway acts on the cluster only
// through GatewayHost, and never re-enters the cluster synchronously from a GatewayHost call:
// work it triggers runs in a fresh event (Schedule).
type Gateway interface {
	// Start is called once, before the first event, with the host the gateway runs on.
	Start(host GatewayHost) error
	// OnArrival hands over a request that has arrived at the cluster.
	OnArrival(req *sim.Request)
	// OnDispatched reports that a dispatched request was accepted by its model server at time at.
	OnDispatched(req *sim.Request, at int64)
	// OnFirstToken reports that the client receives the first output token of req at time at.
	OnFirstToken(req *sim.Request, at int64)
	// OnDone reports that req reached a terminal state; the client observes the end of the
	// response at time at.
	OnDone(req *sim.Request, at int64, outcome DoneOutcome)
	// OnInstanceChange reports an instance that was added or changed lifecycle state.
	OnInstanceChange(inst *InstanceSimulator)
	// Pending returns the number of requests the gateway holds that it has neither dispatched
	// nor rejected.
	Pending() int
	// Stop is called once, after the last event.
	Stop()
}

// IdleFastForwarder is implemented by a gateway whose background work can be advanced over a
// period in which no request is anywhere in the cluster.
type IdleFastForwarder interface {
	// FastForwardIdle is called when the next request work is at target and nothing runs before
	// it but the gateway's own background events.
	FastForwardIdle(target int64)
}

// DoneOutcome is how a dispatched request ended.
type DoneOutcome int

const (
	DoneCompleted DoneOutcome = iota // the response completed (including a length-capped one)
	DoneTimedOut                     // the client timed out
	DoneDropped                      // the model server could not serve it
	DoneEvicted                      // the gateway evicted it in flight
)

// Rejection is the INV-1 bucket a request the gateway did not dispatch lands in.
type Rejection int

const (
	RejectedAdmission       Rejection = iota // an admission check refused the request
	RejectedRouting                          // no endpoint could serve the request
	RejectedGatewayCapacity                  // flow control rejected it: queues at capacity
	RejectedGatewayExpired                   // it waited in flow control past its TTL
	RejectedClientTimeout                    // its client deadline passed while the gateway held it
)

// Placement is where the gateway sends a request. Decode is the endpoint that serves the
// response. A non-empty Prefill disaggregates the request: the prefill runs on Prefill and its KV
// moves to Decode.
type Placement struct {
	Decode  InstanceID
	Prefill InstanceID
	// Encode are the encoders the router chose for a multimodal request, in its order, or none.
	// Encoding costs no simulated time: the choice is recorded (trace,
	// ParentRequest.EncodeInstanceIDs).
	Encode []InstanceID
	// Scores maps instance IDs to the scores the primary scheduling profile gave them, for the
	// decision trace. May be nil.
	Scores map[string]float64
}

// GatewayHost is the cluster as a Gateway sees it.
type GatewayHost interface {
	// Now returns the current simulated time (µs).
	Now() int64
	// Schedule runs fn at time at (>= Now) as a cluster event. Events at one timestamp run in
	// priority order (PriorityGatewayRequest, or PriorityGatewayBackground and above), then in
	// scheduling order. A daemon event does not keep the simulation alive.
	Schedule(at int64, priority int, daemon bool, fn func()) (cancel func())
	// Instances returns every instance in index order, routable or not.
	Instances() []*InstanceSimulator
	// EndpointRanks returns the engines behind the router endpoint id, in rank order: the
	// instance itself, every rank of a data-parallel replica behind vLLM's internal balancer
	// (#1909) for the replica's rank 0, or nil for an instance that is no endpoint (another rank
	// of such a replica). A request dispatched to a replica endpoint is served by the rank the
	// replica's balancer picks.
	EndpointRanks(id InstanceID) []*InstanceSimulator
	// Seed returns the run seed.
	Seed() int64
	// RNG returns the run's partitioned RNG.
	RNG() *sim.PartitionedRNG
	// BlockSizeTokens returns the KV block size of the deployment.
	BlockSizeTokens() int64
	// PoolRole returns an instance's P/D pool role, or 0 when the deployment is not
	// disaggregated.
	PoolRole(id InstanceID) PoolRole
	// Dispatch sends req to its placement now.
	Dispatch(req *sim.Request, p Placement)
	// Reject records that the gateway will not dispatch req.
	Reject(req *sim.Request, why Rejection, reason string)
	// Evict terminates a dispatched request on the instance serving it. It reports whether the
	// request was still running there.
	Evict(req *sim.Request) bool
	// Abort ends the run with err: the gateway met an outcome it cannot account for. Run returns
	// the first error aborted with.
	Abort(err error)
}

// GatewayHoldHorizon is implemented by a gateway whose held requests can be released by its own
// timers (client deadlines, queue-wait budgets). HeldUntil is the latest time any of them can
// still be released that way, given that nothing but those timers has happened since since; until
// then a run waiting only on those timers is not stuck.
type GatewayHoldHorizon interface {
	HeldUntil(since int64) int64
}

// newDefaultGateway, when set, supplies the gateway of a cluster that has none at Run. Only this
// package's tests set it (a small fake); production code always installs the llm-d router.
var newDefaultGateway func() Gateway

// Cluster-event priorities of gateway callbacks at one timestamp.
const (
	// PriorityGatewayRequest is the request path: after the arrival (0), before P/D transfer
	// (4-6) and autoscaling (8), where the cluster's routing decision always ran.
	PriorityGatewayRequest = 2
	// PriorityGatewayBackground and above is the router's background work (metrics
	// collection, timers, housekeeping), after every cluster event at the timestamp.
	PriorityGatewayBackground = 10
)

// CallbackEvent runs a gateway callback.
type CallbackEvent struct {
	at        int64
	priority  int
	daemon    bool
	cancelled bool
	fn        func()
}

func (e *CallbackEvent) Timestamp() int64 { return e.at }
func (e *CallbackEvent) Priority() int    { return e.priority }

func (e *CallbackEvent) Execute(cs *ClusterSimulator) {
	if e.daemon {
		cs.daemonQueued--
	}
	if e.cancelled {
		return
	}
	e.fn()
}

// Now implements GatewayHost.
func (cs *ClusterSimulator) Now() int64 { return cs.clock }

// Seed implements GatewayHost.
func (cs *ClusterSimulator) Seed() int64 { return cs.config.Seed }

// RNG implements GatewayHost.
func (cs *ClusterSimulator) RNG() *sim.PartitionedRNG { return cs.rng }

// BlockSizeTokens implements GatewayHost.
func (cs *ClusterSimulator) BlockSizeTokens() int64 { return cs.config.BlockSizeTokens }

// PoolRole implements GatewayHost.
func (cs *ClusterSimulator) PoolRole(id InstanceID) PoolRole { return cs.poolMembership[string(id)] }

// Schedule implements GatewayHost.
func (cs *ClusterSimulator) Schedule(at int64, priority int, daemon bool, fn func()) func() {
	if at < cs.clock {
		panic(fmt.Sprintf("ClusterSimulator.Schedule: time %d is before the clock %d (INV-3)", at, cs.clock))
	}
	if priority != PriorityGatewayRequest && priority < PriorityGatewayBackground {
		panic(fmt.Sprintf("ClusterSimulator.Schedule: priority %d is reserved for cluster events", priority))
	}
	ev := &CallbackEvent{at: at, priority: priority, daemon: daemon, fn: fn}
	if daemon {
		cs.daemonQueued++
	}
	heap.Push(&cs.clusterEvents, clusterEventEntry{event: ev, seqID: cs.nextSeqID()})
	return func() { ev.cancelled = true }
}

// SetGateway installs the gateway. It must be called before Run.
func (cs *ClusterSimulator) SetGateway(gw Gateway) {
	if cs.hasRun {
		panic("ClusterSimulator: SetGateway must be called before Run()")
	}
	cs.gateway = gw
}

// Dispatch implements GatewayHost.
func (cs *ClusterSimulator) Dispatch(req *sim.Request, p Placement) {
	req.GatewayDispatchTime = cs.clock
	if p.Prefill != "" {
		cs.dispatchDisaggregated(req, p)
		return
	}
	inst := cs.instanceByID(p.Decode)
	if inst == nil {
		panic(fmt.Sprintf("ClusterSimulator.Dispatch: gateway chose unknown instance %q for %s", p.Decode, req.ID))
	}
	cs.recordRouting(req, p)
	cs.injectOnInstance(cs.rankFor(inst), req)
}

// injectOnInstance hands req to inst at the current time.
func (cs *ClusterSimulator) injectOnInstance(inst *InstanceSimulator, req *sim.Request) {
	req.AssignedInstance = string(inst.ID())
	cs.inFlightRequests[string(inst.ID())]++
	warmUpCount := cs.config.InstanceLifecycle.WarmUpRequestCount
	if warmUpCount > 0 && len(inst.WarmUpRequestIDs()) < warmUpCount {
		inst.RecordWarmUpRequest(req.ID)
	}
	inst.InjectRequestOnline(req, cs.clock)
	cs.gateway.OnDispatched(req, cs.clock)
}

// Reject implements GatewayHost.
func (cs *ClusterSimulator) Reject(req *sim.Request, why Rejection, reason string) {
	switch why {
	case RejectedAdmission:
		cs.rejectedRequests++
	case RejectedRouting:
		cs.routingRejections++
	case RejectedGatewayCapacity:
		cs.gatewayRejected++
	case RejectedGatewayExpired:
		cs.gatewayExpired++
	case RejectedClientTimeout:
		cs.gatewayTimedOut++
	default:
		panic(fmt.Sprintf("ClusterSimulator.Reject: unknown rejection %d", why))
	}
	cs.recordRejection(req, why, reason)
	cs.recordUnserved(req) // measured failed_requests and closed-loop drivers (#1938, #1937)
}

// Abort implements GatewayHost.
func (cs *ClusterSimulator) Abort(err error) {
	if cs.abortErr == nil {
		cs.abortErr = err
	}
}

// Evict implements GatewayHost. A disaggregated request is evicted at the stage it is in.
func (cs *ClusterSimulator) Evict(req *sim.Request) bool {
	if parent := cs.parentRequests[req.ID]; parent != nil && parent.OriginalRequest == req {
		if !cs.evictParent(parent) {
			return false
		}
		cs.gatewayEvicted++
		cs.recordShed(req)
		cs.recordUnserved(req)
		cs.parentDone(parent, cs.clock, DoneEvicted)
		return true
	}
	inst := cs.instanceByID(InstanceID(req.AssignedInstance))
	if inst == nil || !inst.EvictRequest(req, cs.clock) {
		return false
	}
	cs.releaseInFlight(req.AssignedInstance)
	cs.gatewayEvicted++
	cs.recordShed(req)
	cs.recordUnserved(req)
	cs.gateway.OnDone(req, cs.clock, DoneEvicted)
	return true
}

// evictParent stops a disaggregated request at its current stage, releasing what that stage
// holds: the prefill sub-request on its prefill engine (no transfer starts); the decode KV
// reserved for a transfer in flight (its completion then finds nothing to promote); or the decode
// sub-request on its decode engine. It reports false when the request is between stages (on its
// way to an engine, or in the step that computes its prefill, whose end starts the transfer) or
// has finished; the gateway retries.
func (cs *ClusterSimulator) evictParent(parent *ParentRequest) bool {
	if parent.gatewayDone {
		return false
	}
	prefill, decode := parent.PrefillSubReq, parent.DecodeSubReq
	switch _, prefilling := cs.pendingPrefillCompletions[parent.PrefillSubReqID]; {
	case prefilling:
		inst := cs.instanceByID(InstanceID(prefill.AssignedInstance))
		if inst == nil || !inst.EvictRequest(prefill, cs.clock) {
			return false // not yet in its prefill engine's queue
		}
		delete(cs.pendingPrefillCompletions, parent.PrefillSubReqID)
		cs.releaseInFlight(prefill.AssignedInstance)
	case decode == nil:
		return false // its prefill is computing in the current step; its transfer starts at the step's end
	case decode.State == sim.StateWaitingForRemoteKVs:
		inst := cs.instanceByID(parent.DecodeInstanceID)
		if inst == nil {
			return false
		}
		inst.AbortReservedKV(decode, cs.clock)
		parent.DecodeSubReq = nil
		cs.pdEvictedAfterPrefill++
	default:
		if _, decoding := cs.pendingDecodeCompletions[parent.DecodeSubReqID]; !decoding {
			return false
		}
		inst := cs.instanceByID(parent.DecodeInstanceID)
		if inst == nil || !inst.EvictRequest(decode, cs.clock) {
			return false
		}
		delete(cs.pendingDecodeCompletions, parent.DecodeSubReqID)
		cs.releaseInFlight(string(parent.DecodeInstanceID))
		cs.pdEvictedAfterPrefill++
	}
	parent.CompletionTime = cs.clock
	return true
}

func (cs *ClusterSimulator) releaseInFlight(id string) {
	cs.inFlightRequests[id]--
	if cs.inFlightRequests[id] < 0 {
		cs.inFlightRequests[id] = 0
	}
}

func (cs *ClusterSimulator) instanceByID(id InstanceID) *InstanceSimulator {
	for _, inst := range cs.instances {
		if inst.ID() == id {
			return inst
		}
	}
	return nil
}

// dispatchDisaggregated starts a disaggregated request: the prefill runs on p.Prefill, then its
// KV moves to p.Decode, which serves the response.
func (cs *ClusterSimulator) dispatchDisaggregated(req *sim.Request, p Placement) {
	if !cs.poolsConfigured() {
		panic(fmt.Sprintf("ClusterSimulator.Dispatch: gateway disaggregated %s on a deployment without P/D pools", req.ID))
	}
	if cs.instanceByID(p.Decode) == nil || cs.instanceByID(p.Prefill) == nil {
		panic(fmt.Sprintf("ClusterSimulator.Dispatch: gateway chose unknown instances (prefill %q, decode %q) for %s", p.Prefill, p.Decode, req.ID))
	}
	cs.recordRouting(req, p)
	if cs.trace != nil {
		cs.trace.RecordDisaggregation(trace.DisaggregationRecord{RequestID: req.ID, Clock: cs.clock, Disaggregate: true})
	}
	parent := NewParentRequest(req, cs.config.BlockSizeTokens)
	parent.DecodeInstanceID = p.Decode
	parent.PrefillInstanceID = p.Prefill
	parent.EncodeInstanceIDs = p.Encode
	cs.parentRequests[parent.ID] = parent

	// The prefill sub-request computes the prompt and completes without output. InputTokens is
	// a slice-header alias of req.InputTokens (#1445).
	prefillSubReq := &sim.Request{
		ID:           parent.PrefillSubReqID,
		InputTokens:  req.InputTokens,
		MaxOutputLen: req.MaxOutputLen,
		Deadline:     req.Deadline,
		PrefixGroup:  req.PrefixGroup,
		State:        sim.StateQueued,
		ArrivalTime:  req.ArrivalTime,
		TenantID:     req.TenantID,
		SLOClass:     req.SLOClass,
		Model:        req.Model,
		Adapter:      req.Adapter, // the router may have rewritten it (InferenceModelRewrite)
		Phase:        req.Phase,
		// A NIXL prefill engine keeps the finished prefill KV for the decode pull (INV-PD-7).
		RetainKVOnFinish: cs.config.PDPrefillKVHold,
	}
	parent.PrefillSubReq = prefillSubReq
	cs.pendingPrefillCompletions[prefillSubReq.ID] = parent.ID
	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &PrefillRoutingEvent{time: cs.clock, request: prefillSubReq, parentReq: parent},
		seqID: cs.nextSeqID(),
	})
	cs.gateway.OnDispatched(req, cs.clock)
}

// parentOf returns the parent of a prefill or decode sub-request, or nil.
func (cs *ClusterSimulator) parentOf(req *sim.Request) *ParentRequest {
	if cs.parentRequests == nil {
		return nil
	}
	if id, ok := cs.pendingPrefillCompletions[req.ID]; ok {
		return cs.parentRequests[id]
	}
	if id, ok := cs.pendingDecodeCompletions[req.ID]; ok {
		return cs.parentRequests[id]
	}
	return nil
}

// parentDone reports the end of a disaggregated request to the gateway.
func (cs *ClusterSimulator) parentDone(parent *ParentRequest, at int64, outcome DoneOutcome) {
	if parent == nil || parent.OriginalRequest == nil || parent.gatewayDone {
		return
	}
	parent.gatewayDone = true
	if parent.DecodeSubReq != nil {
		// The tokens the client received are the decode child's (INV-9: what was generated, never
		// the oracle OutputTokens).
		parent.OriginalRequest.GeneratedTokens = parent.DecodeSubReq.GeneratedTokens
	}
	cs.gateway.OnDone(parent.OriginalRequest, at, outcome)
}

func (cs *ClusterSimulator) recordRouting(req *sim.Request, p Placement) {
	if cs.trace == nil {
		return
	}
	cs.trace.RecordAdmission(trace.AdmissionRecord{RequestID: req.ID, Clock: cs.clock, Admitted: true})
	record := trace.RoutingRecord{
		RequestID:      req.ID,
		Clock:          cs.clock,
		ChosenInstance: string(p.Decode),
		Reason:         "llm-d",
		Scores:         copyScores(p.Scores),
	}
	if k := cs.trace.Config.CounterfactualK; k > 0 && len(p.Scores) > 0 {
		record.Candidates, record.Regret = cs.counterfactual(string(p.Decode), p.Scores, k)
	}
	cs.trace.RecordRouting(record)
	// The encode stage costs nothing (option A, as BLIS's own encode stage did): one record per
	// encoder the router chose, in its order.
	for _, enc := range p.Encode {
		cs.trace.RecordEncodeRouting(trace.EncodeRoutingRecord{ParentRequestID: req.ID, Clock: cs.clock, ChosenInstance: string(enc)})
	}
}

// counterfactual ranks the endpoints the router scored and reports the top k with their state,
// and the regret of the choice: the best score minus the chosen endpoint's score.
func (cs *ClusterSimulator) counterfactual(chosen string, scores map[string]float64, k int) ([]trace.CandidateScore, float64) {
	ids := make([]string, 0, len(scores))
	for id := range scores {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		if scores[ids[i]] != scores[ids[j]] {
			return scores[ids[i]] > scores[ids[j]]
		}
		return ids[i] < ids[j]
	})
	regret := 0.0
	if len(ids) > 0 {
		regret = max(0, scores[ids[0]]-scores[chosen])
	}
	var out []trace.CandidateScore
	for _, id := range ids[:min(k, len(ids))] {
		c := trace.CandidateScore{InstanceID: id, Score: scores[id]}
		if inst := cs.instanceByID(InstanceID(id)); inst != nil {
			c.QueueDepth = inst.QueueDepth()
			c.BatchSize = inst.BatchSize()
			c.InFlightRequests = cs.inFlightRequests[id]
			c.KVUtilization = inst.KVUtilization()
			c.FreeKVBlocks = inst.FreeKVBlocks()
		}
		out = append(out, c)
	}
	return out, regret
}

func (cs *ClusterSimulator) recordRejection(req *sim.Request, why Rejection, reason string) {
	// Shedding is a decision to turn work away under load: an admission denial or a queue-wait
	// expiry. A request no endpoint could serve, or one that found the queues full, is not shed.
	if why == RejectedAdmission || why == RejectedGatewayExpired {
		cs.recordShed(req)
	}
	// A client that gave up is not an admission decision, so it leaves no admission record.
	if cs.trace != nil && why != RejectedClientTimeout {
		cs.trace.RecordAdmission(trace.AdmissionRecord{RequestID: req.ID, Clock: cs.clock, Admitted: false, Reason: reason})
	}
}

// recordShed counts a request the gateway shed against its SLO class.
func (cs *ClusterSimulator) recordShed(req *sim.Request) {
	tier := req.SLOClass
	if tier == "" {
		tier = "standard"
	}
	cs.shedByTier[tier]++
}

func copyScores(scores map[string]float64) map[string]float64 {
	if scores == nil {
		return nil
	}
	out := make(map[string]float64, len(scores))
	for k, v := range scores {
		out[k] = v
	}
	return out
}

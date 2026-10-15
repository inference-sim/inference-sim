package cluster

import (
	"github.com/inference-sim/inference-sim/sim"
)

// fakeGateway is the unit-test stand-in for the llm-d router: it dispatches every request on
// arrival, round-robin over the routable instances that serve the request's model. With P/D
// pools it disaggregates when disaggregate is set, round-robin over each pool (the default
// gateway sets it, so a P/D deployment disaggregates every request unless a test installs
// &fakeGateway{} to keep requests on the decode pool). It holds no
// state between requests beyond the round-robin positions, so cluster behavior tests see the
// cluster's own mechanics, not routing policy.
type fakeGateway struct {
	host         GatewayHost
	next         map[PoolRole]int
	disaggregate bool
	// reject, when set, decides per request whether to reject it as an admission rejection.
	reject func(*sim.Request) bool
	// scores, when set, is reported as Placement.Scores on every dispatch (the decision
	// trace's counterfactual ranks them). It does not influence the choice.
	scores map[string]float64
	// holdUs, when positive, holds every request that long after arrival before deciding
	// on it (a stand-in for a gateway queue delay).
	holdUs int64
	held   int
}

func init() {
	newDefaultGateway = func() Gateway { return &fakeGateway{disaggregate: true} }
}

func (g *fakeGateway) Start(host GatewayHost) error {
	g.host = host
	g.next = map[PoolRole]int{}
	return nil
}

func (g *fakeGateway) OnArrival(req *sim.Request) {
	if g.holdUs > 0 {
		g.held++
		g.host.Schedule(g.host.Now()+g.holdUs, PriorityGatewayRequest, false, func() {
			g.held--
			g.decide(req)
		})
		return
	}
	g.decide(req)
}

func (g *fakeGateway) decide(req *sim.Request) {
	if g.reject != nil && g.reject(req) {
		g.host.Reject(req, RejectedAdmission, "fake gateway")
		return
	}
	pooled := false
	for _, inst := range g.host.Instances() {
		if g.host.PoolRole(inst.ID()) != 0 {
			pooled = true
			break
		}
	}
	if !pooled {
		if id, ok := g.pick(req, 0); ok {
			g.host.Dispatch(req, Placement{Decode: id, Scores: g.scores})
			return
		}
		g.host.Reject(req, RejectedRouting, "no routable instance")
		return
	}
	decode, ok := g.pick(req, PoolRoleDecode)
	if !ok {
		g.host.Reject(req, RejectedRouting, "no routable decode instance")
		return
	}
	p := Placement{Decode: decode, Scores: g.scores}
	if g.disaggregate {
		prefill, ok := g.pick(req, PoolRolePrefill)
		if !ok {
			g.host.Reject(req, RejectedRouting, "no routable prefill instance")
			return
		}
		p.Prefill = prefill
	}
	g.host.Dispatch(req, p)
}

// pick returns the next routable instance with the role (0 = any) serving req's model.
func (g *fakeGateway) pick(req *sim.Request, role PoolRole) (InstanceID, bool) {
	var candidates []*InstanceSimulator
	for _, inst := range g.host.Instances() {
		if !inst.IsRoutable() {
			continue
		}
		if role != 0 && !g.host.PoolRole(inst.ID()).Has(role) {
			continue
		}
		if req.Model != "" && inst.Model != "" && inst.Model != req.Model {
			continue
		}
		candidates = append(candidates, inst)
	}
	if len(candidates) == 0 {
		return "", false
	}
	i := g.next[role] % len(candidates)
	g.next[role]++
	return candidates[i].ID(), true
}

func (g *fakeGateway) OnDispatched(*sim.Request, int64)        {}
func (g *fakeGateway) OnFirstToken(*sim.Request, int64)        {}
func (g *fakeGateway) OnDone(*sim.Request, int64, DoneOutcome) {}
func (g *fakeGateway) OnInstanceChange(*InstanceSimulator)     {}
func (g *fakeGateway) Pending() int                            { return g.held }
func (g *fakeGateway) Stop()                                   {}

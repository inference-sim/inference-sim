// Package llmd runs the llm-d router (the Endpoint Picker) inside the simulation, in place of a
// gateway.
//
// The router is the real llm-d-router code, from the inference-sim fork, reached only through the
// router contract of blis-schemas (package embed) and the fork's constructor for it
// (pkg/embed.New). It is configured exactly as a deployment configures it: the EPP command-line
// flags, the EndpointPickerConfig they name, and the InferencePool, InferenceObjective and
// InferenceModelRewrite manifests. Nothing in this package decides where a request goes; it only
// translates between BLIS and the router:
//
//   - BLIS instances become endpoints of the pool, ready as their lifecycle dictates;
//   - each arrival becomes the request a client sends (an OpenAI body and its routing fields);
//   - the router's outbound HTTP (metrics scrapes, prompt rendering) is served by the simulated
//     engines (server.go);
//   - response headers, the first token and the end of each response reach the router at the
//     simulated times the client would see them;
//   - the router's decision becomes a BLIS dispatch, and its rejections become INV-1 rejection
//     buckets.
//
// Everything runs on a discrete-event runtime (runtime.go, an embed.Runtime) whose clock is the
// simulation clock and whose goroutines are baton-passed coroutines, so the router is
// deterministic (INV-6).
package llmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net"
	"sort"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"github.com/inference-sim/blis-schemas/embed"
	"github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// Options configures the embedded router. EPPFlags, the configuration they name and Manifests
// are the deployment's own router inputs; the remaining fields model the environment.
type Options struct {
	// EPPFlags are the router's command-line flags, verbatim. They name the EndpointPickerConfig
	// with --config-file or --config-text and select the pool with --pool-name or
	// --endpoint-selector.
	EPPFlags []string
	// Manifests are Kubernetes YAML documents. The router reads InferencePool,
	// InferenceObjective and InferenceModelRewrite objects. BLIS also reads
	// explicit Node records for zone/region identity when topology is enabled;
	// other kinds are ignored.
	Manifests []byte
	// Model is the served model name requests carry.
	Model string
	// PodLabels returns the labels of an instance's pod.
	PodLabels func(id cluster.InstanceID, role cluster.PoolRole) map[string]string
	// GatewayLatencyUs is the time from a request's arrival to the router's first look at it:
	// the network and proxy hops before the Endpoint Picker.
	GatewayLatencyUs int64
	// ScrapeJitter bounds the seeded phase offset of each endpoint's metrics collection.
	ScrapeJitter time.Duration
	// KVEventDelayUs is the time from an engine publishing its KV events (at the end of a step)
	// to the router's precise-prefix-cache-producer ingesting them.
	KVEventDelayUs int64
	// Seed, when set, seeds the router's random streams and its prompt canonicalization. Unset,
	// the seed is derived from the run seed. A replay passes the seed its trace recorded.
	Seed *int64
	// Logger receives the router's logs. The zero value discards them. It must never write to
	// stdout (INV-6).
	Logger logr.Logger
}

// DefaultEPPFlags are the router flags BLIS uses when a deployment supplies none: a standalone
// router whose pool is every BLIS pod.
func DefaultEPPFlags() []string {
	return []string{"--endpoint-selector=" + DefaultPoolLabel + "=" + DefaultPoolLabelValue, "--endpoint-target-ports=8000"}
}

// DefaultPoolLabel is the pod label BLIS sets on every instance so the default standalone pool
// selects it.
const (
	DefaultPoolLabel      = "llm-d.ai/inference-serving"
	DefaultPoolLabelValue = "true"
)

// Gateway is the embedded llm-d router. It implements cluster.Gateway.
type Gateway struct {
	opts   Options
	host   cluster.GatewayHost
	rt     *desRuntime
	ctx    context.Context
	cancel context.CancelFunc

	router embed.Router
	req    embed.Requirements
	pool   embed.Pool
	canon  *promptCanon
	// kvHeld counts, per instance, the published holdings (tiers) of each block hash, to release
	// the canonicalizer's pins when the instance's cache is cleared.
	kvHeld map[cluster.InstanceID]map[string]int
	// kvLastDelivery is when the router last receives each instance's KV stream (kvDeliveryTime).
	kvLastDelivery map[cluster.InstanceID]int64
	// canonCapacity, set only by tests, overrides the canonicalizer's chain capacity.
	canonCapacity int
	// nodeLocations are explicit synthetic Node manifests matched to the worker's
	// actual simulated host; the scenario/fabric alone does not define geography.
	nodeLocations map[string]cluster.EndpointTopology
	// topologyRequested guards against silently running an affinity plugin with
	// missing or ambiguous placement (its normal behavior is to fail open).
	topologyRequested bool
	// maxQueueTTL and expirySweepUs bound a held request's wait in flow control (FlowControl).
	maxQueueTTL   int64
	expirySweepUs int64
	// observe, set only by tests, sees each scrape ("scrape", endpoint), each request the router
	// receives ("request", ID), each dispatch ("dispatch", "ID <context error>"), each rejection
	// ("rejected", "ID kind time") and the usage of each completed response ("usage", "ID tokens").
	observe func(kind, detail string)
	// responseHeaderObserved is a test hook for the response headers after the
	// router's handling. The callback must copy the map if it retains it.
	responseHeaderObserved func(requestID string, headers map[string]string)
	// flightsEnded counts the router flights BLIS has ended (a final ResponseBody or Terminate);
	// flightsDispatched those it was handed. Equal at the end of a run: the router tracks no
	// request as in flight.
	flightsDispatched, flightsEnded int

	endpoints map[cluster.InstanceID]*endpoint
	byIP      map[string]cluster.InstanceID
	flights   map[string]*flight
	// evictable are the dispatched flights carrying an eviction signal, in dispatch order.
	evictable []*flight
	pending   int
	// skipIdle is whether idle periods may be fast-forwarded (see FastForwardIdle);
	// neverSkipIdle disables it, for the test that proves it exact.
	skipIdle      bool
	neverSkipIdle bool
}

type endpoint struct {
	inst *cluster.InstanceSimulator
	ep   embed.Endpoint
	// registered is set once the router has been told of the endpoint.
	registered bool
	ready      bool
	// cleared is set when the instance terminates; kvCleared once the router has been told its
	// prefix cache is gone, after which no batch of it is delivered.
	cleared   bool
	kvCleared bool
}

// flight is one request between its arrival and the end of its response.
type flight struct {
	req    *sim.Request
	fl     embed.Flight // the router's flight, once dispatched
	ctx    context.Context
	cancel context.CancelFunc
	stream bool
	prompt []uint32 // the prompt as the router sees it (promptCanon)
	// stopDeadline cancels the client-deadline event (nil without a deadline).
	stopDeadline func()
	// decided is set when the router has returned its decision; finishRequest acts on it at the
	// same instant, so a client deadline at that instant no longer cancels the request.
	decided    bool
	held       bool // counted in Pending: arrived and neither dispatched nor rejected
	dispatched bool
	// evictCh is the router's in-flight eviction signal for the request once it is dispatched
	// (nil when the router never evicts): flow control closes it to revoke the request.
	evictCh    <-chan struct{}
	evicting   bool
	headersOut bool
	firstOut   bool
	done       bool
}

// Seed is the seed the router runs with. It is valid once the router has started.
func (g *Gateway) Seed() int64 { return g.rt.seed }

// Requirements returns what the router's configuration requires of its host, as the router
// reported it when it started (embed.Requirements); the zero value before Start.
func (g *Gateway) Requirements() embed.Requirements { return g.req }

// New builds an embedded router. The router itself is assembled in Start, when the instances
// exist.
func New(opts Options) *Gateway {
	if len(opts.EPPFlags) == 0 {
		opts.EPPFlags = DefaultEPPFlags()
	}
	return &Gateway{opts: opts, endpoints: map[cluster.InstanceID]*endpoint{}, byIP: map[string]cluster.InstanceID{}, flights: map[string]*flight{}}
}

var _ cluster.Gateway = (*Gateway)(nil)

// Start builds the router through its own startup path and registers the instances.
func (g *Gateway) Start(host cluster.GatewayHost) (err error) {
	defer func() {
		if err != nil {
			g.Stop() // release whatever the router started before the failure
		}
	}()
	g.host = host
	seed := streamSeed(host.RNG())
	if g.opts.Seed != nil {
		seed = *g.opts.Seed
	}
	g.rt = newDESRuntime(host, seed, g.opts.ScrapeJitter)
	g.rt.server = &modelServers{g: g, rendered: map[cluster.InstanceID]renderedMetrics{}}
	g.ctx, g.cancel = context.WithCancel(context.Background())

	var logger *slog.Logger
	if g.opts.Logger.GetSink() != nil {
		logger = slog.New(logr.ToSlogHandler(g.opts.Logger))
	}
	router, err := newRouter(g.ctx, g.rt, embed.Config{
		Flags:     g.opts.EPPFlags,
		Manifests: g.opts.Manifests,
		Allow:     allow,
		Check:     g.check,
		Logger:    logger,
	})
	if err != nil {
		return err
	}
	g.router = router
	g.rt.setScope(router.Pool().ID())

	var kvBlocks int64
	for _, inst := range host.Instances() {
		kvBlocks += inst.TotalKVBlocks()
	}
	capacity := int(2 * kvBlocks)
	if g.canonCapacity > 0 {
		capacity = g.canonCapacity
	}
	g.canon = newPromptCanon(g.rt.seed, int(host.BlockSizeTokens()), capacity)
	g.kvHeld = map[cluster.InstanceID]map[string]int{}
	g.kvLastDelivery = map[cluster.InstanceID]int64{}
	for _, inst := range host.Instances() {
		if err := g.checkPreciseLoRA(inst); err != nil {
			return err
		}
	}
	if err := validateTopology(g.req, host); err != nil {
		return err
	}
	g.rt.co.settled = g.checkEvictions
	g.skipIdle = router.IdleSkippable()
	if fc := router.FlowControl(); fc.Enabled {
		g.maxQueueTTL, g.expirySweepUs = toMicros(fc.MaxQueueTTL), toMicros(fc.ExpirySweep)
	}

	g.rt.warm = true
	for _, inst := range host.Instances() {
		if host.EndpointRanks(inst.ID()) == nil {
			continue // a data-parallel rank behind its replica's endpoint (#1909)
		}
		g.OnInstanceChange(inst)
	}
	g.rt.warm = false
	g.rt.co.settle()
	return nil
}

// check is the router's pre-build check (embed.Config.Check): what the configuration needs from
// the simulated engines and placement, compared with what they produce, before any plugin exists.
func (g *Gateway) check(req embed.Requirements, pool embed.Pool) error {
	g.req, g.pool = req, pool
	if len(req.InertFlags) > 0 {
		logrus.Warnf("[llm-d] EPP flags with no effect in a simulation (transport, replicas, telemetry): %s", strings.Join(req.InertFlags, " "))
	}
	if err := checkMetricFlags(req.MetricFlags); err != nil {
		return fmt.Errorf("the router configuration cannot be simulated faithfully: %w", err)
	}
	if err := checkTopologyLabels(req.Topology); err != nil {
		return fmt.Errorf("the router configuration cannot be simulated faithfully: %w", err)
	}
	g.topologyRequested = req.Topology.Requested
	if g.topologyRequested {
		nodes, err := manifestNodes(g.opts.Manifests)
		if err != nil {
			return err
		}
		if g.nodeLocations, err = nodeLocations(nodes); err != nil {
			return fmt.Errorf("simulated Node topology: %w", err)
		}
	}
	var podLabels []map[string]string
	var resolved []cluster.EndpointTopology
	for _, inst := range g.host.Instances() {
		if g.host.EndpointRanks(inst.ID()) == nil {
			continue // a data-parallel rank behind its replica's endpoint (#1909)
		}
		labels, err := g.podLabelsForInstance(inst)
		if err != nil {
			return fmt.Errorf("the router configuration cannot be simulated faithfully: %w", err)
		}
		podLabels = append(podLabels, labels)
		resolved = append(resolved, podLocation(labels))
	}
	if g.topologyRequested {
		if err := validateLocationHierarchy(resolved); err != nil {
			return fmt.Errorf("simulated Node topology: %w", err)
		}
	}
	if err := checkEngineSignals(req, podLabels, int(g.host.BlockSizeTokens())); err != nil {
		return fmt.Errorf("the router configuration cannot be simulated faithfully: %w", err)
	}
	return checkTargetPorts(pool)
}

// OnInstanceChange makes an instance's endpoint ready or not ready, as its lifecycle state
// dictates.
func (g *Gateway) OnInstanceChange(inst *cluster.InstanceSimulator) {
	if g.router == nil {
		return // instances are registered at Start
	}
	ranks := g.host.EndpointRanks(inst.ID())
	if ranks == nil {
		// A data-parallel rank behind its replica's endpoint (#1909) is no pod of its own: the
		// change is the endpoint's, whose readiness needs every rank.
		g.onEndpointRankChange(inst)
		return
	}
	ep, ok := g.endpoints[inst.ID()]
	if !ok {
		if !g.rt.warm {
			// A pod that appears during the run (autoscaling, lifecycle) passes the startup checks
			// too: its labels may select an engine mapping, and its role may make the deployment P/D.
			if err := g.checkLatePod(inst); err != nil {
				g.host.Abort(err)
				return
			}
		}
		var err error
		ep, err = g.newEndpoint(inst)
		if err != nil {
			g.host.Abort(err)
			return
		}
	}
	if inst.State == sim.InstanceStateTerminated && !ep.cleared {
		// Its prefix cache is gone with it. The clear travels the same delayed stream as its
		// events, so a batch already in flight cannot arrive after it and repopulate the index.
		ep.cleared = true
		g.host.Schedule(g.kvDeliveryTime(inst.ID(), g.host.Now()), PriorityKVEvents, true, func() { g.clearKVEvents(ep) })
	}
	ready := true
	for _, r := range ranks {
		ready = ready && r.IsRoutable()
	}
	if ep.registered && ready == ep.ready {
		return
	}
	ep.registered, ep.ready = true, ready
	if err := g.router.SetEndpoint(ep.ep, ready); err != nil {
		panic(fmt.Sprintf("llmd: set endpoint %s: %v", ep.ep.Name, err))
	}
	g.rt.co.settle()
}

// onEndpointRankChange applies a lifecycle change of a data-parallel rank to its replica's
// endpoint, the pod of the replica's rank 0.
func (g *Gateway) onEndpointRankChange(rank *cluster.InstanceSimulator) {
	for id, ep := range g.endpoints {
		for _, r := range g.host.EndpointRanks(id) {
			if r == rank {
				g.OnInstanceChange(ep.inst)
				return
			}
		}
	}
}

// podLabels are the labels of an instance's pod: the default pool label and Options.PodLabels.
func (g *Gateway) podLabels(id cluster.InstanceID, role cluster.PoolRole) map[string]string {
	labels := map[string]string{DefaultPoolLabel: DefaultPoolLabelValue}
	if g.opts.PodLabels != nil {
		for k, v := range g.opts.PodLabels(id, role) {
			labels[k] = v
		}
	}
	return labels
}

// podLabelsForInstance overlays only topology BLIS can establish from the
// actual simulated placement. A user label must not contradict it. When a
// topology plugin is configured, missing or multi-node placement is an error
// instead of the real filter's otherwise silent fail-open behavior.
func (g *Gateway) podLabelsForInstance(inst *cluster.InstanceSimulator) (map[string]string, error) {
	labels := g.podLabels(inst.ID(), g.host.PoolRole(inst.ID()))
	// One EPP endpoint may front several data-parallel ranks on different hosts.
	// Rank zero's location is not the location of that whole endpoint.
	if len(g.host.EndpointRanks(inst.ID())) > 1 {
		if g.topologyRequested {
			return nil, fmt.Errorf("instance %s endpoint spans multiple data-parallel ranks; no single simulated host placement", inst.ID())
		}
		return labels, nil
	}
	t, err := overlayNodeLocation(inst.EndpointTopology(), g.nodeLocations)
	if err != nil {
		return nil, fmt.Errorf("instance %s: %w", inst.ID(), err)
	}
	if g.topologyRequested {
		switch {
		case t.MixedSources:
			return nil, fmt.Errorf("instance %s has both NodePool placement and independent kernel transfer placement", inst.ID())
		case t.NodeCount > 1:
			return nil, fmt.Errorf("instance %s spans %d nodes; one synthetic Pod cannot claim a single host", inst.ID(), t.NodeCount)
		case t.NodeCount != 1:
			return nil, fmt.Errorf("instance %s has no simulated host placement (node count %d)", inst.ID(), t.NodeCount)
		case t.Hostname == "":
			return nil, fmt.Errorf("instance %s has no simulated host placement for topology-aware routing", inst.ID())
		}
	}
	for _, field := range []struct{ key, derived string }{
		{corev1.LabelHostname, t.Hostname},
		{"topology.kubernetes.io/rack", t.Rack},
		{corev1.LabelTopologyZone, t.Zone},
		{corev1.LabelTopologyRegion, t.Region},
	} {
		key, derived := field.key, field.derived
		if stated := labels[key]; stated != "" && stated != derived && (derived != "" || g.topologyRequested) {
			return nil, fmt.Errorf("instance %s Pod label %s=%q conflicts with simulated placement %q", inst.ID(), key, stated, derived)
		}
		if derived != "" {
			labels[key] = derived
		}
	}
	return labels, nil
}

func (g *Gateway) newEndpoint(inst *cluster.InstanceSimulator) (*endpoint, error) {
	idx := len(g.endpoints)
	ip := fmt.Sprintf("10.%d.%d.%d", 1+idx/65536, (idx/256)%256, idx%256)
	port := 8000
	if len(g.pool.TargetPorts) > 0 {
		port = g.pool.TargetPorts[0]
	}
	labels, err := g.podLabelsForInstance(inst)
	if err != nil {
		return nil, err
	}
	nodeName := inst.EndpointTopology().Hostname
	if len(g.host.EndpointRanks(inst.ID())) > 1 {
		nodeName = "" // a grouped endpoint does not occupy one Kubernetes node
	}
	ep := &endpoint{inst: inst, ep: embed.Endpoint{Name: podName(inst.ID()), Labels: labels, NodeName: nodeName, IP: ip, Port: port}}
	g.subscribeKVEvents(inst)
	g.endpoints[inst.ID()] = ep
	g.byIP[ip] = inst.ID()
	return ep, nil
}

// podName turns an instance ID into a valid Kubernetes object name.
func podName(id cluster.InstanceID) string { return strings.ReplaceAll(string(id), "_", "-") }

// instanceAt returns the instance a router target ("ip:port" or "ip") names, or nil.
func (g *Gateway) instanceAt(target string) *cluster.InstanceSimulator {
	host := target
	if h, _, err := net.SplitHostPort(target); err == nil {
		host = h
	}
	id, ok := g.byIP[host]
	if !ok {
		return nil
	}
	return g.endpoints[id].inst
}

func (g *Gateway) servedModel(inst *cluster.InstanceSimulator) string {
	if inst.Model != "" {
		return inst.Model
	}
	return g.opts.Model
}

// Pending implements cluster.Gateway.
func (g *Gateway) Pending() int { return g.pending }

// FastForwardIdle implements cluster.IdleFastForwarder. While no request is anywhere in the router
// or the engines, nothing the router reads changes, so its background loops (metrics collection
// of unchanged engine state, cleanup, flow-control ticks over empty queues) have the effect over
// an idle period that their last ticks have. The cluster calls this when the next request work
// is at target; the router's loops then run only their last ticks before it, each keeping its
// phase. It is skipped when a data source polls at its own cadence.
func (g *Gateway) FastForwardIdle(target int64) {
	if !g.skipIdle || g.neverSkipIdle || len(g.flights) > 0 || g.pending > 0 {
		return
	}
	g.rt.fastForward(target)
}

// Stop implements cluster.Gateway: it ends every router loop and every coroutine.
func (g *Gateway) Stop() {
	if g.cancel == nil || g.rt == nil {
		return
	}
	for _, id := range g.sortedFlights() {
		g.flights[id].cancel()
	}
	g.cancel()
	if g.router != nil {
		g.router.Stop()
	}
	g.rt.co.drain()
}

func (g *Gateway) sortedFlights() []string {
	ids := make([]string, 0, len(g.flights))
	for id := range g.flights {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// LiveCoroutines returns the number of router coroutines that have not finished.
func (g *Gateway) LiveCoroutines() int { return g.rt.co.Live() }

// OnArrival implements cluster.Gateway: the router sees the request after the gateway latency.
func (g *Gateway) OnArrival(req *sim.Request) {
	g.pending++
	f := &flight{req: req, held: true, stream: req.Streaming}
	g.flights[req.ID] = f
	g.host.Schedule(g.host.Now()+g.opts.GatewayLatencyUs, PriorityRequest, false, func() { g.handleRequest(f) })
	if req.Deadline > 0 {
		// The client gives up at its deadline. While the router still holds the request (flow
		// control), that cancels the request's context, as a client disconnect does; once it is
		// dispatched, the engine's own timeout applies. A daemon event: it never prolongs a run.
		f.stopDeadline = g.host.Schedule(max(req.Deadline, g.host.Now()), PriorityRequest, true, func() { g.clientDeadline(f) })
	}
}

// clientDeadline ends a request whose client deadline passed before the router dispatched it.
func (g *Gateway) clientDeadline(f *flight) {
	f.stopDeadline = nil
	if f.dispatched || f.done || f.decided {
		return // decided: dispatched or rejected at this instant; a dispatched request's deadline is the engine's
	}
	if f.ctx == nil { // still on its way to the router (gateway latency)
		f.held = false
		g.pending--
		g.host.Reject(f.req, cluster.RejectedClientTimeout, "client deadline passed before the router received the request")
		g.endFlight(f)
		return
	}
	f.cancel()
	g.rt.co.settle()
}

// handleRequest hands the request to the router as a client sends it: the router parses it and
// decides, possibly after waiting in flow control. Its decision is acted on in a fresh event at
// the instant it is made (Rule G: the router never calls back into the cluster synchronously).
func (g *Gateway) handleRequest(f *flight) {
	if f.done { // its client deadline passed on the way to the router
		return
	}
	// A stdlib cancel context over the gateway's own (itself over Background): the router derives
	// the request's context from it, and cancellation propagates synchronously only through stdlib
	// cancel contexts, so a client deadline reaches the router at its simulated instant (INV-6).
	f.ctx, f.cancel = context.WithCancel(g.ctx)
	f.prompt = g.canon.canonical(f.req.InputTokens)
	if g.observe != nil {
		g.observe("request", f.req.ID)
	}
	if g.router.FlowControl().Enabled {
		f.req.GatewayEnqueueTime = g.host.Now() // the router enqueues it in flow control now
	}
	g.router.Route(f.ctx, g.routeRequest(f), func(fl embed.Flight, d embed.Decision, rej *embed.Rejection) {
		f.decided = true
		g.host.Schedule(g.host.Now(), PriorityRequest, false, func() { g.finishRequest(f, fl, d, rej) })
	})
	g.rt.co.settle()
}

// routeRequest is the request a client sends for f.
func (g *Gateway) routeRequest(f *flight) embed.Request {
	r := embed.Request{
		ID:         f.req.ID,
		Path:       requestPath(f),
		Body:       g.requestBody(f),
		Stream:     f.stream,
		Objective:  f.req.SLOClass,
		FairnessID: f.req.TenantID,
	}
	if f.req.SLOTargetUs > 0 {
		r.TTFTSLO = time.Duration(f.req.SLOTargetUs) * time.Microsecond
	}
	if f.req.Deadline > 0 {
		// The client's lifetime for the request, as it sends it: flow control's queue budget
		// becomes the shorter of this and the configured one (an EDF ordering policy orders by
		// it). The router counts from its own enqueue, no earlier than the send, so the client
		// still gives up first, at its deadline (clientDeadline).
		if ttl, ok := requestTTL(f.req.Deadline - f.req.ArrivalTime); ok {
			r.TTL = ttl
		}
	}
	return r
}

// finishRequest acts on the router's decision for f.
func (g *Gateway) finishRequest(f *flight, fl embed.Flight, d embed.Decision, rej *embed.Rejection) {
	if !f.held {
		return
	}
	f.held = false
	g.pending--
	if rej != nil {
		if g.observe != nil {
			g.observe("rejected", fmt.Sprintf("%s %s %d", f.req.ID, rej.Kind, g.host.Now()))
		}
		g.reject(f, rej)
		return
	}
	decode := g.instanceAt(d.Endpoint)
	if decode == nil {
		panic(fmt.Sprintf("llmd: the router picked %q for %s, which is no simulated endpoint", d.Endpoint, f.req.ID))
	}
	p := cluster.Placement{Decode: decode.ID(), Scores: g.instanceScores(d.Scores)}
	if d.Prefill != "" {
		prefill := g.instanceAt(d.Prefill)
		if prefill == nil {
			panic(fmt.Sprintf("llmd: the router picked prefiller %q for %s, which is no simulated endpoint", d.Prefill, f.req.ID))
		}
		if prefill.ID() != decode.ID() {
			p.Prefill = prefill.ID()
		}
	}
	if len(d.Encoders) > 0 {
		// Every encoder the router chose, in its order, as the encode sidecar uses them all.
		var ids []string
		for _, target := range d.Encoders {
			encode := g.instanceAt(target)
			if encode == nil {
				panic(fmt.Sprintf("llmd: the router picked encoder %q for %s, which is no simulated endpoint", target, f.req.ID))
			}
			p.Encode = append(p.Encode, encode.ID())
			ids = append(ids, string(encode.ID()))
		}
		if g.observe != nil {
			g.observe("encode", fmt.Sprintf("%s %s", f.req.ID, strings.Join(ids, ",")))
		}
	}
	g.applyModelRewrite(f, d)
	f.fl = fl
	f.dispatched = true
	g.flightsDispatched++
	if f.stopDeadline != nil {
		f.stopDeadline()
		f.stopDeadline = nil
	}
	if g.observe != nil {
		g.observe("dispatch", fmt.Sprintf("%s %v", f.req.ID, f.ctx.Err()))
		g.observe("score", fmt.Sprintf("%s %s %g", f.req.ID, decode.ID(), p.Scores[string(decode.ID())]))
	}
	if f.evictCh = fl.EvictionSignal(); f.evictCh != nil {
		g.evictable = append(g.evictable, f)
	}
	g.host.Dispatch(f.req, p)
}

// checkEvictions observes flow control's in-flight eviction signals, as the ext_proc server's
// stream loop does: a dispatched request whose signal has closed is revoked, in a fresh event
// (the router is not re-entered from its own call), in dispatch order.
func (g *Gateway) checkEvictions() {
	live := g.evictable[:0]
	for _, f := range g.evictable {
		if f.done {
			continue
		}
		live = append(live, f)
		if f.evicting {
			continue
		}
		select {
		case <-f.evictCh:
			f.evicting = true
			g.host.Schedule(g.host.Now(), PriorityRequest, false, func() { g.evict(f) })
		default:
		}
	}
	clear(g.evictable[len(live):])
	g.evictable = live
}

// evict revokes a dispatched request the router evicted: the client receives the router's
// immediate response, and the gateway aborts the request on its engine (its KV blocks and adapter
// pin are released), counted as gateway-evicted; the router's response ends with the eviction
// cause (OnDone). A request that finished at this instant has nothing left to revoke.
func (g *Gateway) evict(f *flight) {
	if f.done {
		return
	}
	if g.observe != nil {
		g.observe("evict", f.req.ID)
	}
	if !g.host.Evict(f.req) {
		f.evicting = false // between stages at this instant: the next check retries
		return
	}
	if g.observe != nil {
		g.observe("evicted", f.req.ID)
	}
}

func (g *Gateway) instanceScores(scores map[string]float64) map[string]float64 {
	if len(scores) == 0 {
		return nil
	}
	out := make(map[string]float64, len(scores))
	for addr, s := range scores {
		if inst := g.instanceAt(addr); inst != nil {
			out[string(inst.ID())] = s
		}
	}
	return out
}

// reject maps a router rejection onto the INV-1 bucket of the outcome it describes. An outcome that
// matches no known kind aborts the run: it means BLIS or the router mistranslated something, and
// counting it as shedding would hide that (R1).
func (g *Gateway) reject(f *flight, rej *embed.Rejection) {
	why, ok := rejectionBucket(rej.Kind)
	if !ok {
		g.host.Abort(fmt.Errorf("request %s: the router returned an outcome BLIS cannot classify: %w", f.req.ID, errors.New(rej.Msg)))
		why = cluster.RejectedRouting // keep the books balanced until the run stops
	}
	g.host.Reject(f.req, why, rej.Msg)
	g.endFlight(f)
}

// rejectionBucket maps the kind of a router's rejection onto its INV-1 bucket. The kinds are an
// explicit allow-list:
//
//   - NoEndpoints and Unavailable: no endpoint could serve it (routing);
//   - TTLExpired: it waited in flow control past its queue-wait budget;
//   - Saturated: flow control's queues were at capacity;
//   - ContextCancelled: the client gave up while the router held it (its deadline);
//   - Admission: an admission check refused it (the legacy admission controller shedding a
//     sheddable request, or an admitter plugin).
//
// Unclassified, and any kind a newer contract adds, is not a routing outcome BLIS knows.
func rejectionBucket(k embed.RejectKind) (cluster.Rejection, bool) {
	switch k {
	case embed.NoEndpoints, embed.Unavailable:
		return cluster.RejectedRouting, true
	case embed.TTLExpired:
		return cluster.RejectedGatewayExpired, true
	case embed.Saturated:
		return cluster.RejectedGatewayCapacity, true
	case embed.ContextCancelled:
		return cluster.RejectedClientTimeout, true
	case embed.Admission:
		return cluster.RejectedAdmission, true
	}
	return 0, false
}

// OnDispatched implements cluster.Gateway. A streaming response's headers leave the model
// server as soon as it accepts the request.
func (g *Gateway) OnDispatched(req *sim.Request, at int64) {
	f := g.flights[req.ID]
	if f == nil || !f.stream {
		return
	}
	g.host.Schedule(at, PriorityRequest, false, func() { g.responseHeaders(f) })
}

// OnFirstToken implements cluster.Gateway.
func (g *Gateway) OnFirstToken(req *sim.Request, at int64) {
	f := g.flights[req.ID]
	if f == nil || !f.stream || f.firstOut {
		return
	}
	f.firstOut = true // the client receives the first token once, whatever the engine recomputes
	g.host.Schedule(at, PriorityRequest, false, func() {
		if f.done {
			return
		}
		g.responseHeaders(f)
		f.fl.ResponseBody(firstChunk(g.opts.Model, f, false), false)
		g.rt.co.settle()
	})
}

// OnDone implements cluster.Gateway.
func (g *Gateway) OnDone(req *sim.Request, at int64, outcome cluster.DoneOutcome) {
	f := g.flights[req.ID]
	if f == nil {
		return
	}
	g.host.Schedule(at, PriorityRequest, false, func() { g.endResponse(f, outcome) })
}

func (g *Gateway) responseHeaders(f *flight) {
	if f.headersOut || f.done {
		return
	}
	f.headersOut = true
	ct := "application/json"
	if f.stream {
		ct = "text/event-stream"
	}
	headers := f.fl.ResponseHeaders(map[string]string{":status": "200", "content-type": ct})
	if g.responseHeaderObserved != nil {
		g.responseHeaderObserved(f.req.ID, headers)
	}
	g.rt.co.settle()
}

// endResponse delivers the end of a response to the router, or, for a response that did not
// complete, its termination.
func (g *Gateway) endResponse(f *flight, outcome cluster.DoneOutcome) {
	if f.done {
		return
	}
	if outcome == cluster.DoneCompleted {
		usage := usageOf(f.req)
		if g.observe != nil {
			g.observe("usage", fmt.Sprintf("%s %d", f.req.ID, usage.CompletionTokens))
		}
		g.responseHeaders(f)
		if f.stream {
			if !f.firstOut {
				f.firstOut = true
				f.fl.ResponseBody(firstChunk(g.opts.Model, f, false), false)
			}
			f.fl.ResponseBody(firstChunk(g.opts.Model, f, true, usage), true)
		} else {
			f.fl.ResponseBody(completionBody(g.opts.Model, f, usage), true)
		}
	} else {
		f.fl.Terminate(termination(outcome))
	}
	g.flightsEnded++
	g.endFlight(f)
	g.rt.co.settle()
}

func termination(o cluster.DoneOutcome) embed.Termination {
	switch o {
	case cluster.DoneTimedOut:
		return embed.ClientDisconnect
	case cluster.DoneEvicted:
		return embed.Evicted
	default:
		return embed.Error
	}
}

func (g *Gateway) endFlight(f *flight) {
	f.done = true
	if f.stopDeadline != nil {
		f.stopDeadline()
		f.stopDeadline = nil
	}
	if f.cancel != nil {
		f.cancel()
	}
	delete(g.flights, f.req.ID)
}

// applyModelRewrite makes the engine serve the model the router resolved.
func (g *Gateway) applyModelRewrite(f *flight, d embed.Decision) {
	target := d.TargetModel
	if target == "" || target == d.IncomingModel {
		return
	}
	if target == g.opts.Model {
		f.req.Adapter = ""
		return
	}
	f.req.Adapter = target
}

// --- request and response translation ---

const (
	completionsPath = "/v1/completions"
	chatPath        = "/v1/chat/completions"
)

// requestPath is the API a client sends f to: a multimodal request is a chat completion (its
// media are content parts), any other a token-ID completion.
func requestPath(f *flight) string {
	if f.req.IsMultimodal() {
		return chatPath
	}
	return completionsPath
}

// renderID is the text part of a multimodal request's message: it names the request, so the
// simulated renderer, which has no tokenizer, renders the request's own tokens (renderChat).
func renderID(id string) string { return "blis-request " + id }

// maxTTLMicros is the longest lifetime a Go duration (nanoseconds in an int64) can carry.
const maxTTLMicros = math.MaxInt64 / int64(time.Microsecond/time.Nanosecond)

// requestTTL is a client lifetime of lifeUs microseconds as the request's TTL, which the router
// sends as x-llm-d-inference-ttl. A lifetime that is not positive sends no TTL; one too long for a
// Go duration (over about 292 years) sends none either: it bounds no queue budget the router can
// hold, as with no client deadline.
func requestTTL(lifeUs int64) (time.Duration, bool) {
	if lifeUs <= 0 || lifeUs > maxTTLMicros {
		return 0, false
	}
	return time.Duration(lifeUs) * time.Microsecond, true
}

// requestBody is the OpenAI completions request a client sends for f: the prompt as token IDs,
// the output budget, and the model (the LoRA adapter, when the request names one).
func (g *Gateway) requestBody(f *flight) []byte {
	model := g.opts.Model
	if f.req.Adapter != "" {
		model = f.req.Adapter
	}
	body := map[string]any{"model": model, "prompt": f.prompt, "stream": f.stream}
	if f.req.IsMultimodal() {
		// One content part per modality the request carries, after its text, as a client sends
		// media in a chat message. The renderer gives each its placeholder span (renderChat).
		parts := []map[string]any{{"type": "text", "text": renderID(f.req.ID)}}
		if f.req.ImageTokenCount > 0 {
			parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": "blis://image/" + f.req.ID}})
		}
		if f.req.AudioTokenCount > 0 {
			parts = append(parts, map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "", "format": "wav"}})
		}
		if f.req.VideoTokenCount > 0 {
			parts = append(parts, map[string]any{"type": "video_url", "video_url": map[string]any{"url": "blis://video/" + f.req.ID}})
		}
		body = map[string]any{"model": model, "stream": f.stream,
			"messages": []map[string]any{{"role": "user", "content": parts}}}
	}
	if f.req.MaxOutputLen > 0 {
		body["max_tokens"] = f.req.MaxOutputLen
	}
	b, err := json.Marshal(body)
	if err != nil {
		panic(err)
	}
	return b
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// usageOf is the usage a model server reports for a completed request: the prompt and the
// tokens it actually generated (INV-9: never the request's oracle output).
func usageOf(req *sim.Request) usage {
	p := int(req.InputLen())
	c := req.GeneratedTokens
	return usage{PromptTokens: p, CompletionTokens: c, TotalTokens: p + c}
}

// firstChunk is a streamed response chunk carrying no text: the first, or, final, the last, with
// the usage and the end of the stream. A chat request's chunks are chat-completion chunks.
func firstChunk(model string, f *flight, final bool, u ...usage) []byte {
	choice := map[string]any{"index": 0, "text": "", "finish_reason": nil}
	object, id := "text_completion", "cmpl-"+f.req.ID
	if requestPath(f) == chatPath {
		choice = map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": nil}
		object, id = "chat.completion.chunk", "chatcmpl-"+f.req.ID
	}
	chunk := map[string]any{"id": id, "object": object, "model": model, "choices": []map[string]any{choice}}
	if !final {
		b, _ := json.Marshal(chunk)
		return []byte("data: " + string(b) + "\n\n")
	}
	choice["finish_reason"] = "stop"
	chunk["usage"] = u[0]
	b, _ := json.Marshal(chunk)
	return []byte("data: " + string(b) + "\n\ndata: [DONE]\n\n")
}

func completionBody(model string, f *flight, u usage) []byte {
	choice := map[string]any{"index": 0, "text": "", "finish_reason": "stop"}
	object, id := "text_completion", "cmpl-"+f.req.ID
	if requestPath(f) == chatPath {
		choice = map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": ""}, "finish_reason": "stop"}
		object, id = "chat.completion", "chatcmpl-"+f.req.ID
	}
	b, _ := json.Marshal(map[string]any{"id": id, "object": object, "model": model,
		"choices": []map[string]any{choice}, "usage": u})
	return b
}

// HeldUntil is the latest time at which a request the router holds can still be released by its
// own timers: its client deadline, or the end of its queue-wait budget plus one expiry sweep. A
// change of pool regime (endpoints appearing or leaving) starts a fresh budget; it follows an
// instance event, so a budget counts from the later of the request's enqueue and since, the last
// time anything other than the router's own timers happened. Past it, a run in which nothing else
// happens cannot make progress. It implements cluster.GatewayHoldHorizon.
func (g *Gateway) HeldUntil(since int64) int64 {
	var until int64
	for _, f := range g.flights {
		if !f.held {
			continue
		}
		release := int64(math.MaxInt64)
		if f.req.Deadline > 0 {
			release = f.req.Deadline
		}
		if g.maxQueueTTL > 0 {
			release = min(release, max(f.req.GatewayEnqueueTime, since)+g.maxQueueTTL+g.expirySweepUs)
		}
		if release != math.MaxInt64 {
			until = max(until, release)
		}
	}
	return until
}

// checkLatePod applies the startup checks to a pod that appears during the run.
func (g *Gateway) checkLatePod(inst *cluster.InstanceSimulator) error {
	labels, err := g.podLabelsForInstance(inst)
	if err != nil {
		return fmt.Errorf("instance %s, added during the run: %w", inst.ID(), err)
	}
	resolved := []cluster.EndpointTopology{podLocation(labels)}
	for _, ep := range g.endpoints {
		resolved = append(resolved, podLocation(ep.ep.Labels))
	}
	if g.topologyRequested {
		if err := validateLocationHierarchy(resolved); err != nil {
			return fmt.Errorf("instance %s, added during the run: simulated Node topology: %w", inst.ID(), err)
		}
	}
	if err := checkEngineSignals(g.req, []map[string]string{labels}, int(g.host.BlockSizeTokens())); err != nil {
		return fmt.Errorf("instance %s, added during the run: the router configuration cannot be simulated faithfully: %w", inst.ID(), err)
	}
	if err := validateTopology(g.req, g.host); err != nil {
		return fmt.Errorf("instance %s, added during the run: %w", inst.ID(), err)
	}
	if err := g.checkPreciseLoRA(inst); err != nil {
		return fmt.Errorf("instance %s, added during the run: %w", inst.ID(), err)
	}
	return nil
}

// checkPreciseLoRA refuses a precise-prefix-cache-producer over an engine serving LoRA adapters.
func (g *Gateway) checkPreciseLoRA(inst *cluster.InstanceSimulator) error {
	if len(g.req.PrefixIndexes) > 0 && inst.AdapterCapacity() > 0 {
		return errors.New("the router configuration cannot be simulated faithfully: precise-prefix-cache-producer " +
			"keys blocks by LoRA adapter, but the compatibility engine this simulator runs keys its prefix cache " +
			"without the adapter, so its KV events cannot name the adapter a block belongs to")
	}
	return nil
}

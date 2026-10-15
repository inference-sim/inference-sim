// Package llmd runs the llm-d router (the Endpoint Picker) inside the simulation, in place of a
// gateway.
//
// The router is the real llm-d-router code, from the inference-sim fork that runs it on a
// hosting runtime (pkg/common/hosting). It is configured exactly as a deployment configures it:
// the EPP command-line flags, the EndpointPickerConfig they name, and the InferencePool,
// InferenceObjective and InferenceModelRewrite manifests. Nothing in this package decides where
// a request goes; it only translates between BLIS and the router:
//
//   - BLIS instances become ready pods of the pool, through the router's own pod reconciler;
//   - each arrival becomes the request headers and body Envoy would hand the router, processed by
//     the router's own request handlers and Director;
//   - the router's outbound HTTP (metrics scrapes, prompt rendering) is served by the simulated
//     engines (server.go);
//   - response headers, the first token and the end of each response reach the router at the
//     simulated times the client would see them;
//   - the router's decision becomes a BLIS dispatch, and its errors become INV-1 rejection
//     buckets.
//
// Everything runs on a discrete-event runtime (runtime.go) whose clock is the simulation clock
// and whose goroutines are baton-passed coroutines, so the router is deterministic (INV-6).
package llmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	"github.com/llm-d/llm-d-router/cmd/epp/runner"
	"github.com/llm-d/llm-d-router/pkg/common"
	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/common/hosting"
	"github.com/llm-d/llm-d-router/pkg/epp/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
	poolutil "github.com/llm-d/llm-d-router/pkg/epp/util/pool"
	"github.com/sirupsen/logrus"
	"github.com/spf13/pflag"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/log"
	giev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

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
	// InferenceObjective and InferenceModelRewrite objects; other kinds are ignored.
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
	defaultNamespace      = "default"
)

// embedMu serializes router construction: the router registers its plugin factories and
// feature gates in process-wide registries.
var embedMu sync.Mutex

// Gateway is the embedded llm-d router. It implements cluster.Gateway.
type Gateway struct {
	opts   Options
	host   cluster.GatewayHost
	rt     *desRuntime
	ctx    context.Context
	cancel context.CancelFunc
	logger logr.Logger

	eppOpts *runserver.Options
	emb     *runner.Embedded
	canon   *promptCanon
	// observe, set only by tests, sees each scrape ("scrape", endpoint), each request the router
	// receives ("request", ID) and the usage of each completed response ("usage", "ID tokens").
	observe func(kind, detail string)
	server  *handlers.StreamingServer
	pool    *datalayer.EndpointPool
	gknn    common.GKNN
	kube    client.Client
	pods    *controller.PodReconciler

	endpoints map[cluster.InstanceID]*endpoint
	byIP      map[string]cluster.InstanceID
	flights   map[string]*flight
	pending   int
	// skipIdle is whether idle periods may be fast-forwarded (see FastForwardIdle);
	// neverSkipIdle disables it, for the test that proves it exact.
	skipIdle      bool
	neverSkipIdle bool
}

type endpoint struct {
	inst  *cluster.InstanceSimulator
	pod   *corev1.Pod
	ip    string
	port  int
	ready bool
}

// flight is one request between its arrival and the end of its response.
type flight struct {
	req    *sim.Request
	reqCtx *handlers.RequestContext
	ctx    context.Context
	cancel context.CancelFunc
	stream bool
	prompt []uint32 // the prompt as the router sees it (promptCanon)
	// stopDeadline cancels the client-deadline event (nil without a deadline).
	stopDeadline func()
	held         bool // counted in Pending: arrived and neither dispatched nor rejected
	dispatched   bool
	headersOut   bool
	firstOut     bool
	done         bool
}

// Seed is the seed the router runs with. It is valid once the router has started.
func (g *Gateway) Seed() int64 { return g.rt.seed }

// New builds an embedded router. The router itself is assembled in Start, when the instances
// exist.
func New(opts Options) *Gateway {
	if len(opts.EPPFlags) == 0 {
		opts.EPPFlags = DefaultEPPFlags()
	}
	return &Gateway{opts: opts, endpoints: map[cluster.InstanceID]*endpoint{}, byIP: map[string]cluster.InstanceID{}, flights: map[string]*flight{}}
}

var _ cluster.Gateway = (*Gateway)(nil)

// ParseEPPFlags parses router flags with the router's own flag set, as the EPP binary does.
func ParseEPPFlags(args []string) (*runserver.Options, error) {
	opts := runserver.NewOptions()
	fs := pflag.NewFlagSet("epp", pflag.ContinueOnError)
	opts.AddFlags(fs)
	if err := fs.Parse(args); err != nil {
		return nil, fmt.Errorf("EPP flags: %w", err)
	}
	if fs.NArg() > 0 {
		return nil, fmt.Errorf("EPP flags: unexpected arguments %q", fs.Args())
	}
	if err := opts.Complete(); err != nil {
		return nil, fmt.Errorf("EPP flags: %w", err)
	}
	if err := opts.Validate(); err != nil {
		return nil, fmt.Errorf("EPP flags: %w", err)
	}
	return opts, nil
}

// Start assembles the router through its own startup path and registers the instances.
func (g *Gateway) Start(host cluster.GatewayHost) (err error) {
	defer func() {
		if err != nil {
			g.Stop() // release whatever the router started before the failure
		}
	}()
	g.host = host
	g.logger = g.opts.Logger
	if g.logger.GetSink() == nil {
		g.logger = logr.Discard()
	}
	eppOpts, perr := ParseEPPFlags(g.opts.EPPFlags)
	if perr != nil {
		return perr
	}
	g.eppOpts = eppOpts
	inert, err := checkEPPFlags(g.opts.EPPFlags)
	if err != nil {
		return fmt.Errorf("the router configuration cannot be simulated faithfully: %w", err)
	}
	if len(inert) > 0 {
		logrus.Warnf("[llm-d] EPP flags with no effect in a simulation (transport, replicas, telemetry): %s", strings.Join(inert, " "))
	}
	config := []byte(eppOpts.ConfigText)
	if eppOpts.ConfigFile != "" {
		if config, err = os.ReadFile(eppOpts.ConfigFile); err != nil {
			return fmt.Errorf("EPP flag --config-file: %w", err)
		}
	}
	var podLabels []map[string]string
	for _, inst := range host.Instances() {
		podLabels = append(podLabels, g.podLabels(inst.ID(), host.PoolRole(inst.ID())))
	}
	if err := checkRawConfig(config, podLabels); err != nil {
		return fmt.Errorf("the router configuration cannot be simulated faithfully: %w", err)
	}

	scheme := newScheme()
	res, err := parseManifests(scheme, g.opts.Manifests, poolNamespace(eppOpts))
	if err != nil {
		return err
	}
	if err := g.resolvePool(eppOpts, res); err != nil {
		return err
	}
	if len(g.pool.TargetPorts) > 1 {
		// Several target ports make each pod several endpoints (data-parallel ranks behind one
		// pod). BLIS simulates each rank as its own instance and pod (DP placement), so the ports
		// would name endpoints no engine serves.
		return fmt.Errorf("the pool has %d target ports %v; BLIS simulates each data-parallel rank as its own pod "+
			"with one port, so give the pool a single target port", len(g.pool.TargetPorts), g.pool.TargetPorts)
	}

	seed := streamSeed(host.RNG())
	if g.opts.Seed != nil {
		seed = *g.opts.Seed
	}
	g.rt = newDESRuntime(host, seed, g.gknn.String(), g.opts.ScrapeJitter)
	var kvBlocks int64
	for _, inst := range host.Instances() {
		kvBlocks += inst.TotalKVBlocks()
	}
	g.canon = newPromptCanon(g.rt.seed, int(host.BlockSizeTokens()), int(2*kvBlocks))
	g.rt.server = &modelServers{g: g, rendered: map[cluster.InstanceID]renderedMetrics{}}
	ctx := hosting.Into(log.IntoContext(context.Background(), g.logger), g.rt)
	g.ctx, g.cancel = context.WithCancel(ctx)

	embedMu.Lock()
	ctrl.SetLogger(g.logger)
	emb, eerr := runner.Embed(g.ctx, runner.EmbedOptions{Options: eppOpts, Pool: g.pool, AfterRegister: refuseUnsupportedFactories})
	embedMu.Unlock()
	if eerr != nil {
		return fmt.Errorf("llm-d router: %w", eerr)
	}
	g.emb = emb
	// A backstop: every plugin the configuration holds, by any path, is a supported type.
	if err := validateEmbedded(emb.Config, emb.Handle); err != nil {
		return err
	}
	if err := validateTopology(emb, host); err != nil {
		return err
	}
	if emb.RequestEvictor != nil {
		return errors.New("the router configuration cannot be simulated faithfully: flow-control in-flight eviction " +
			"(enableEviction) revokes dispatched requests, and the simulation does not model that revocation yet")
	}
	g.skipIdle = true
	if emb.Config != nil && emb.Config.DataConfig != nil {
		for _, src := range emb.Config.DataConfig.Sources {
			if disp, ok := src.Plugin.(fwkdl.PollingDispatcher); ok && disp.Interval() > 0 {
				// A source with its own cadence counts base ticks; skipping ticks would shift it.
				g.skipIdle = false
			}
		}
	}
	g.server = handlers.NewStreamingServer(emb.Datastore, emb.Director, emb.Parsers, 0)
	g.server.SetResponders(nil)

	objs := res.objects()
	g.kube = fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
	g.pods = &controller.PodReconciler{Reader: g.kube, Datastore: emb.Datastore}
	if eppOpts.PoolName != "" {
		if err := g.reconcileResources(res, objs); err != nil {
			return err
		}
	} else if len(res.objectives)+len(res.objectivesV1a2)+len(res.rewrites) > 0 {
		return errors.New("InferenceObjective and InferenceModelRewrite manifests have no effect with --endpoint-selector: " +
			"the router runs their reconcilers only for an InferencePool (--pool-name)")
	}

	g.rt.warm = true
	for _, inst := range host.Instances() {
		g.OnInstanceChange(inst)
	}
	g.rt.warm = false
	g.rt.co.settle()
	return nil
}

func poolNamespace(o *runserver.Options) string {
	if o.PoolNamespace != "" {
		return o.PoolNamespace
	}
	return defaultNamespace
}

// resolvePool finds the pool the router serves: the InferencePool named by --pool-name, or the
// standalone pool --endpoint-selector describes.
func (g *Gateway) resolvePool(o *runserver.Options, res *resources) error {
	ns := poolNamespace(o)
	if o.PoolName != "" {
		for _, p := range res.pools {
			if p.Name == o.PoolName && p.Namespace == ns {
				g.pool = poolutil.InferencePoolToEndpointPool(p)
				g.gknn = common.GKNN{
					NamespacedName: types.NamespacedName{Namespace: ns, Name: o.PoolName},
					GroupKind:      schema.GroupKind{Group: o.PoolGroup, Kind: "InferencePool"},
				}
				if len(res.pools) > 1 {
					return fmt.Errorf("the manifests define %d InferencePools; a simulation runs one router for one pool", len(res.pools))
				}
				return nil
			}
		}
		return fmt.Errorf("--pool-name=%s: no InferencePool %s/%s in the manifests", o.PoolName, ns, o.PoolName)
	}
	if len(res.pools) > 0 {
		return errors.New("the manifests define an InferencePool but the EPP flags select a standalone pool (--endpoint-selector); pass --pool-name")
	}
	g.pool = datalayer.NewEndpointPool(ns, "epp")
	g.pool.Selector = o.EndpointSelector
	g.pool.TargetPorts = append(g.pool.TargetPorts, o.EndpointTargetPorts...)
	g.gknn = common.GKNN{NamespacedName: types.NamespacedName{Namespace: ns, Name: "epp"}, GroupKind: schema.GroupKind{Kind: "Deployment", Group: "apps"}}
	return nil
}

// reconcileResources runs the router's own objective and rewrite reconcilers over the manifests,
// as the Kubernetes controllers do at startup.
func (g *Gateway) reconcileResources(res *resources, objs []client.Object) error {
	objRec := &controller.InferenceObjectiveReconciler{
		Reader: g.kube, Datastore: g.emb.Datastore, PoolGKNN: g.gknn,
		PriorityBandControlPlane: g.emb.ControlPlane, PrimaryV1: true,
		// As the runner discovers from the API server: the installed CRD serves both versions.
		WatchV1Alpha2: res.served[schema.GroupVersion(v1alpha2.GroupVersion).WithKind("InferenceObjective")],
	}
	rwRec := &controller.InferenceModelRewriteReconciler{Reader: g.kube, Datastore: g.emb.Datastore, PoolGKNN: g.gknn}
	for _, o := range objs {
		req := ctrl.Request{NamespacedName: types.NamespacedName{Namespace: o.GetNamespace(), Name: o.GetName()}}
		var err error
		switch o.(type) {
		case *giev1.InferencePool:
		case *v1alpha2.InferenceModelRewrite:
			_, err = rwRec.Reconcile(g.ctx, req)
		default:
			_, err = objRec.Reconcile(g.ctx, req)
		}
		if err != nil {
			return fmt.Errorf("reconciling %s/%s: %w", o.GetNamespace(), o.GetName(), err)
		}
	}
	return nil
}

// OnInstanceChange makes an instance's pod ready or not ready, as its lifecycle state dictates,
// and lets the router's pod reconciler update the pool.
func (g *Gateway) OnInstanceChange(inst *cluster.InstanceSimulator) {
	if g.emb == nil {
		return // instances are registered at Start
	}
	ep, ok := g.endpoints[inst.ID()]
	if !ok {
		ep = g.newEndpoint(inst)
	}
	ready := inst.IsRoutable()
	if ok && ready == ep.ready {
		return
	}
	ep.ready = ready
	status := corev1.ConditionFalse
	if ready {
		status = corev1.ConditionTrue
	}
	ep.pod.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: status}}
	if ok {
		// Readiness is pod status, a subresource: a plain Update does not change it.
		if err := g.kube.Status().Update(g.ctx, ep.pod); err != nil {
			panic(fmt.Sprintf("llmd: update pod status %s: %v", ep.pod.Name, err))
		}
	} else if err := g.kube.Create(g.ctx, ep.pod); err != nil {
		panic(fmt.Sprintf("llmd: create pod %s: %v", ep.pod.Name, err))
	}
	if g.emb.Datastore.PoolLabelsMatch(ep.pod.Labels) {
		req := ctrl.Request{NamespacedName: types.NamespacedName{Namespace: ep.pod.Namespace, Name: ep.pod.Name}}
		if _, err := g.pods.Reconcile(g.ctx, req); err != nil {
			panic(fmt.Sprintf("llmd: reconcile pod %s: %v", ep.pod.Name, err))
		}
	}
	g.rt.co.settle()
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

func (g *Gateway) newEndpoint(inst *cluster.InstanceSimulator) *endpoint {
	idx := len(g.endpoints)
	ip := fmt.Sprintf("10.%d.%d.%d", 1+idx/65536, (idx/256)%256, idx%256)
	port := 8000
	if len(g.pool.TargetPorts) > 0 {
		port = g.pool.TargetPorts[0]
	}
	labels := g.podLabels(inst.ID(), g.host.PoolRole(inst.ID()))
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName(inst.ID()), Namespace: g.gknn.Namespace, Labels: labels},
		Status:     corev1.PodStatus{PodIP: ip, HostIP: ip, Phase: corev1.PodRunning},
	}
	ep := &endpoint{inst: inst, pod: pod, ip: ip, port: port}
	g.endpoints[inst.ID()] = ep
	g.byIP[ip] = inst.ID()
	return ep
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
	if f.dispatched || f.done {
		return
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

// handleRequest runs the request phase exactly as the router's ext_proc server does for a
// request whose headers and body have arrived: the header handler, the body parser, then the
// Director, which may wait in flow control.
func (g *Gateway) handleRequest(f *flight) {
	if f.done { // its client deadline passed on the way to the router
		return
	}
	f.ctx, f.cancel = context.WithCancel(g.ctx)
	reqCtx := &handlers.RequestContext{
		Request:  &handlers.Request{ID: f.req.ID, Headers: map[string]string{}, Metadata: map[string]any{}},
		Response: &handlers.Response{Headers: map[string]string{}},
	}
	f.reqCtx = reqCtx
	f.prompt = g.canon.canonical(f.req.InputTokens)
	if g.observe != nil {
		g.observe("request", f.req.ID)
	}
	body := g.requestBody(f)
	if err := g.server.HandleRequestHeaders(f.ctx, reqCtx, requestHeaders(g.requestHeaderMap(f))); err != nil {
		g.finishRequest(f, err)
		return
	}
	reqCtx.Request.RawBody = body
	reqCtx.RequestSize = len(body)
	parser, err := g.emb.Parsers.Resolve(completionsPath)
	if err != nil {
		g.finishRequest(f, errcommon.Error{Code: errcommon.BadRequest, Msg: err.Error()})
		return
	}
	reqCtx.Parser = parser
	parsed, err := parser.ParseRequest(f.ctx, body, reqCtx.Request.Headers)
	if err != nil {
		g.finishRequest(f, errcommon.Error{Code: errcommon.BadRequest, Msg: err.Error()})
		return
	}
	if g.emb.FlowControlEnabled() {
		f.req.GatewayEnqueueTime = g.host.Now() // the Director enqueues it in flow control now
	}
	g.rt.Go(func() {
		_, err := g.emb.Director.HandleRequest(f.ctx, reqCtx, parsed.Body)
		// Rule G: the router never calls back into the cluster synchronously.
		g.host.Schedule(g.host.Now(), PriorityRequest, false, func() { g.finishRequest(f, err) })
	})
	g.rt.co.settle()
}

// finishRequest acts on the router's decision for f.
func (g *Gateway) finishRequest(f *flight, err error) {
	if !f.held {
		return
	}
	f.held = false
	g.pending--
	if err != nil {
		g.reject(f, err)
		return
	}
	reqCtx := f.reqCtx
	targets := strings.Split(reqCtx.TargetEndpoint, ",")
	decode := g.instanceAt(targets[0])
	if decode == nil {
		panic(fmt.Sprintf("llmd: the router picked %q for %s, which is no simulated endpoint", reqCtx.TargetEndpoint, f.req.ID))
	}
	p := cluster.Placement{Decode: decode.ID(), Scores: g.instanceScores(reqCtx.TargetEndpointScores)}
	if hp, ok := metadata.GetLowerCaseHeaderValue(reqCtx.Request.Headers, prefillerHeader); ok && hp != "" {
		prefill := g.instanceAt(strings.Split(hp, ",")[0])
		if prefill == nil {
			panic(fmt.Sprintf("llmd: the router picked prefiller %q for %s, which is no simulated endpoint", hp, f.req.ID))
		}
		if prefill.ID() != decode.ID() {
			p.Prefill = prefill.ID()
		}
	}
	if err := g.applyModelRewrite(f); err != nil {
		panic(err.Error())
	}
	f.dispatched = true
	if f.stopDeadline != nil {
		f.stopDeadline()
		f.stopDeadline = nil
	}
	g.host.Dispatch(f.req, p)
}

// prefillerHeader is the header the disaggregation profile handler sets for the P/D sidecar.
const prefillerHeader = "x-llm-d-prefiller-host-port"

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

// reject maps a router error onto the INV-1 bucket of the outcome it describes. An error that
// matches no known router outcome aborts the run: it means BLIS mistranslated something, and
// counting it as shedding would hide that (R1).
func (g *Gateway) reject(f *flight, err error) {
	why, reason, ok := classify(err)
	if !ok {
		g.host.Abort(fmt.Errorf("request %s: the router returned an outcome BLIS cannot classify: %w", f.req.ID, err))
		why = cluster.RejectedRouting // keep the books balanced until the run stops
	}
	g.host.Reject(f.req, why, reason)
	g.endFlight(f)
}

// classify maps the router's rejection of a request onto its INV-1 bucket. The outcomes are an
// explicit allow-list, keyed on the router's status code and its x-llm-d-request-dropped-reason:
//
//   - rejected-no-endpoints: no endpoint could serve it (routing);
//   - rejected-ttl-expired: it waited in flow control past its queue-wait budget;
//   - rejected-saturated: flow control's queues were at capacity;
//   - rejected-context-cancelled: the client gave up while the router held it (its deadline);
//   - ResourceExhausted with no drop reason: an admission check refused it (the legacy admission
//     controller shedding a sheddable request, or an admitter plugin);
//   - ServiceUnavailable with no drop reason: no endpoint could serve it.
//
// Anything else is not a routing outcome and is reported as unclassified.
func classify(err error) (why cluster.Rejection, reason string, ok bool) {
	var e errcommon.Error
	if !errors.As(err, &e) {
		return 0, err.Error(), false
	}
	switch dropped, hasReason := e.Headers[errcommon.RequestDroppedReasonHeaderKey]; {
	case hasReason:
		switch errcommon.RequestDroppedReason(dropped) {
		case errcommon.RequestDroppedReasonNoEndpoints:
			return cluster.RejectedRouting, e.Msg, true
		case errcommon.RequestDroppedReasonTTLExpired:
			return cluster.RejectedGatewayExpired, e.Msg, true
		case errcommon.RequestDroppedReasonSaturated:
			return cluster.RejectedGatewayCapacity, e.Msg, true
		case errcommon.RequestDroppedReasonContextCancelled:
			return cluster.RejectedClientTimeout, e.Msg, true
		}
	case e.Code == errcommon.ResourceExhausted && !strings.HasPrefix(e.Msg, schedulerFallbackMsg):
		return cluster.RejectedAdmission, e.Msg, true
	case e.Code == errcommon.ServiceUnavailable:
		return cluster.RejectedRouting, e.Msg, true
	}
	return 0, e.Msg, false
}

// schedulerFallbackMsg starts the Director's message for a scheduler failure that carries no
// typed error (requestcontrol/director.go). It shares ResourceExhausted with admission denials but
// is not one; no supported configuration produces it, so it is reported as unclassified.
const schedulerFallbackMsg = "failed to find target endpoint"

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
		g.server.HandleResponseBody(f.ctx, f.reqCtx, firstChunk(g.opts.Model, f.req.ID), false)
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
	g.server.HandleResponseHeaders(f.ctx, f.reqCtx, responseHeaders(map[string]string{":status": "200", "content-type": ct}))
	g.rt.co.settle()
}

// endResponse delivers the end of a response to the router, or, for a response that did not
// complete, the termination the ext_proc server reports when a stream breaks.
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
				g.server.HandleResponseBody(f.ctx, f.reqCtx, firstChunk(g.opts.Model, f.req.ID), false)
			}
			g.server.HandleResponseBody(f.ctx, f.reqCtx, finalChunk(g.opts.Model, f.req.ID, usage), true)
		} else {
			g.server.HandleResponseBody(f.ctx, f.reqCtx, completionBody(g.opts.Model, f.req.ID, usage), true)
		}
	} else {
		f.reqCtx.TerminationCause = terminationCause(outcome)
		g.emb.Director.HandleResponseBody(context.WithoutCancel(f.ctx), f.reqCtx, true)
	}
	g.endFlight(f)
	g.rt.co.settle()
}

func terminationCause(o cluster.DoneOutcome) fwkrc.TerminationCause {
	switch o {
	case cluster.DoneTimedOut:
		return fwkrc.TerminationCauseClientDisconnect
	case cluster.DoneEvicted:
		return fwkrc.TerminationCauseEvicted
	default:
		return fwkrc.TerminationCauseError
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
func (g *Gateway) applyModelRewrite(f *flight) error {
	target := f.reqCtx.TargetModelName
	if target == "" || target == f.reqCtx.IncomingModelName {
		return nil
	}
	if target == g.opts.Model {
		f.req.Adapter = ""
		return nil
	}
	f.req.Adapter = target
	return nil
}

// --- request and response translation ---

const completionsPath = "/v1/completions"

func (g *Gateway) requestHeaderMap(f *flight) map[string]string {
	h := map[string]string{
		":method":      "POST",
		":path":        completionsPath,
		"content-type": "application/json",
		"x-request-id": f.req.ID,
	}
	if f.req.SLOClass != "" {
		h[metadata.ObjectiveKey] = f.req.SLOClass
	}
	if f.req.TenantID != "" {
		h[metadata.FlowFairnessIDKey] = f.req.TenantID
	}
	if f.req.SLOTargetUs > 0 {
		// The header is whole milliseconds; round up so a sub-millisecond target stays a target.
		h[metadata.TTFTSLOHeaderKey] = strconv.FormatInt((f.req.SLOTargetUs+999)/1000, 10)
	}
	return h
}

// requestBody is the OpenAI completions request a client sends for f: the prompt as token IDs,
// the output budget, and the model (the LoRA adapter, when the request names one).
func (g *Gateway) requestBody(f *flight) []byte {
	model := g.opts.Model
	if f.req.Adapter != "" {
		model = f.req.Adapter
	}
	body := map[string]any{"model": model, "prompt": f.prompt, "stream": f.stream}
	if f.req.MaxOutputLen > 0 {
		body["max_tokens"] = f.req.MaxOutputLen
	}
	b, err := json.Marshal(body)
	if err != nil {
		panic(err)
	}
	return b
}

func requestHeaders(h map[string]string) *extProcPb.ProcessingRequest_RequestHeaders {
	return &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: &extProcPb.HttpHeaders{Headers: headerMap(h)}}
}

func responseHeaders(h map[string]string) *extProcPb.ProcessingRequest_ResponseHeaders {
	return &extProcPb.ProcessingRequest_ResponseHeaders{ResponseHeaders: &extProcPb.HttpHeaders{Headers: headerMap(h)}}
}

func headerMap(h map[string]string) *configPb.HeaderMap {
	keys := make([]string, 0, len(h))
	for k := range h {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	m := &configPb.HeaderMap{}
	for _, k := range keys {
		m.Headers = append(m.Headers, &configPb.HeaderValue{Key: k, RawValue: []byte(h[k])})
	}
	return m
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

func firstChunk(model, id string) []byte {
	b, _ := json.Marshal(map[string]any{
		"id": "cmpl-" + id, "object": "text_completion", "model": model,
		"choices": []map[string]any{{"index": 0, "text": "", "finish_reason": nil}},
	})
	return []byte("data: " + string(b) + "\n\n")
}

func finalChunk(model, id string, u usage) []byte {
	b, _ := json.Marshal(map[string]any{
		"id": "cmpl-" + id, "object": "text_completion", "model": model,
		"choices": []map[string]any{{"index": 0, "text": "", "finish_reason": "stop"}},
		"usage":   u,
	})
	return []byte("data: " + string(b) + "\n\ndata: [DONE]\n\n")
}

func completionBody(model, id string, u usage) []byte {
	b, _ := json.Marshal(map[string]any{
		"id": "cmpl-" + id, "object": "text_completion", "model": model,
		"choices": []map[string]any{{"index": 0, "text": "", "finish_reason": "stop"}},
		"usage":   u,
	})
	return b
}

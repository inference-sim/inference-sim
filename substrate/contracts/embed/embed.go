// Package embed is the contract between a simulator that hosts a request router and the
// router it hosts, the counterpart of package kernel: kernel is how a simulator prices work,
// embed is how it lets a real router decide where that work goes.
//
// The boundary is ownership. The host owns simulated time, background work and randomness
// (it implements Runtime), the endpoints' lifecycle (it calls SetEndpoint and
// RemoveEndpoint), and what each engine publishes (it serves the router's outbound HTTP
// through RoundTripper and feeds KV events through IngestKVEvents). The router owns every
// routing decision and the meaning of its configuration. The contract never restates router
// configuration — scorers, weights, profiles — which travels as the deployment's own bytes
// (Config); what a host needs to know about it comes back as facts (Requirements), and the
// host decides what those facts mean for the engines it simulates.
//
// # Determinism
//
// A router is a deterministic function of its Config, the Runtime it runs on and the
// sequence of calls the host makes. It reads time only through Runtime.Now and the timers and
// tickers Runtime makes, starts concurrent work only through Runtime.Go and Runtime.Every,
// blocks only through Runtime.Park, draws randomness only from Runtime.Rand, and reaches the
// network only through the Runtime's RoundTripper. On a deterministic Runtime it makes the
// same decisions every time.
//
// # Calls
//
// The host calls Router and Flight methods from one goroutine at a time, never concurrently,
// and never from inside a callback the router is running. The router calls back into the host
// only through Runtime and through Route's done (see Route).
//
// # Compatibility
//
// Interfaces here are implemented in other repositories, so within a major version none of
// them gains a method; an optional capability is a separate interface (RoundTripper), found by
// type assertion. Struct types may gain fields in a minor version, with the zero value meaning
// what the type meant before the field existed; construct them with keyed literals. A new
// RejectKind or Termination value is a minor change too: a host maps a kind it does not know
// as it maps Unclassified.
//
// Package embedtest checks a Constructor against this contract.
package embed

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"time"
)

// --- What the host implements ---------------------------------------------------------

// Runtime is the environment a hosted router runs on: time, background work, blocking and
// randomness. A host that runs the router on simulated time implements it on its event queue;
// every router action then happens at a simulated instant and in an order the host fixes.
type Runtime interface {
	// Now returns the current time.
	Now() time.Time

	// NewTimer returns a timer that delivers Now on its channel once d has elapsed on the
	// runtime clock, as time.NewTimer does. A non-positive d fires at the current instant.
	NewTimer(d time.Duration) Timer

	// NewTicker returns a ticker that delivers Now on its channel every d on the runtime
	// clock, dropping ticks a slow reader misses, as time.NewTicker does. d must be positive.
	NewTicker(d time.Duration) Ticker

	// Go runs fn concurrently with the caller. The runtime may run fn until it finishes or
	// parks before Go returns.
	Go(fn func())

	// Every calls fn every interval until ctx is done; the first call is one interval after
	// Every is called. name identifies the loop to the runtime (a host may, for example, give
	// metrics collection its own phase). Every does not block. interval must be positive.
	Every(ctx context.Context, name string, interval time.Duration, fn func(context.Context))

	// WithTimeout is context.WithTimeout on the runtime clock.
	WithTimeout(parent context.Context, d time.Duration) (context.Context, context.CancelFunc)

	// WithDeadlineCause is context.WithDeadlineCause on the runtime clock.
	WithDeadlineCause(parent context.Context, deadline time.Time, cause error) (context.Context, context.CancelFunc)

	// Park is called by a goroutine the router started with Go or Every immediately before it
	// blocks on work that runs elsewhere in the router. ready reports whether that work has
	// happened; the goroutine blocks only if ready returns false, and the runtime resumes it
	// once ready returns true. ready reads only router state, and must be cheap.
	Park(ready func() bool)

	// Rand returns the named random stream. The same name returns the same stream.
	Rand(stream string) Rand

	// Local returns the value stored under key, storing init() first if there is none. It
	// holds state the router would otherwise keep process-wide, so two routers on two
	// runtimes never share it.
	Local(key any, init func() any) any
}

// Timer is a timer on the runtime clock, with the methods of k8s.io/utils/clock.Timer.
type Timer interface {
	C() <-chan time.Time
	// Stop prevents the timer from firing. It reports whether the timer was active.
	Stop() bool
	// Reset makes the timer fire d from now. It reports whether the timer was active.
	Reset(d time.Duration) bool
}

// Ticker is a ticker on the runtime clock, with the methods of k8s.io/utils/clock.Ticker.
type Ticker interface {
	C() <-chan time.Time
	Stop()
}

// Rand is a stream of random numbers.
type Rand interface {
	Float64() float64
	// IntN returns a number in [0, n). n must be positive.
	IntN(n int) int
	Shuffle(n int, swap func(i, j int))
}

// RoundTripper is implemented by a Runtime that serves the router's outbound HTTP itself:
// engine metrics, prompt rendering, any request the router would send to a model server. A
// router on a Runtime that does not implement it makes no outbound request, and refuses at
// construction a configuration that needs one.
type RoundTripper interface {
	RoundTrip(*http.Request) (*http.Response, error)
}

// --- What the router implements -------------------------------------------------------

// KVIndexInspector is implemented by a Router that can report its KV-event index, found by
// type assertion. It is for conformance and fidelity tests; a host must not depend on it in
// production.
type KVIndexInspector interface {
	// KVBlocks returns the engine block keys the router's index holds for the endpoint at
	// address endpoint (Endpoint.Address), on tier ("" for every tier). Reading does not
	// perturb the index: no recency, eviction or statistic changes. It returns an error when
	// the router keeps no index for that endpoint.
	KVBlocks(endpoint, tier string) (map[uint64]bool, error)
}

// Config is a router's configuration as a deployment states it. Its bytes are the router's
// own formats, which the contract does not interpret.
type Config struct {
	// Flags are the router's command-line flags, verbatim (a deployment's EPP flags).
	Flags []string
	// Config is the router's configuration file (an EndpointPickerConfig), when Flags do
	// not name one. Naming one in both is an error.
	Config []byte
	// Manifests are the deployment's Kubernetes YAML documents. The router reads the kinds
	// it reads and ignores the others.
	Manifests []byte
	// Allow is the host's plugin policy. For every plugin type the router would instantiate
	// — named in the configuration, reached through a reference, or added as a default — it
	// decides before that plugin's constructor runs, in this order, and refuses the
	// configuration with a *RefusalError carrying the reason:
	//
	//  1. a type the router cannot host, for a reason it states, is refused with that reason,
	//     and Allow is not consulted;
	//  2. otherwise Allow is consulted, and false refuses the type with Allow's reason;
	//  3. otherwise a type the router has not audited for hosting is still refused, even
	//     though Allow permits it.
	//
	// So the host's reason reaches the user for any type it refuses that the router has no
	// reason of its own to refuse, and the router never runs code it has not audited because
	// the host allows it. nil allows every type the router hosts.
	Allow func(pluginType string) (ok bool, reason string)
	// Check is the host's check of the configuration, before the router is built. The
	// router calls it once, after it has parsed Flags, Config and Manifests and resolved its
	// pool, and before it instantiates any plugin or consults Allow, with the Requirements
	// and Pool the returned Router would report. A non-nil error is returned from the
	// Constructor as is, unwrapped, after the router has released everything it started.
	// nil means no host check.
	Check func(Requirements, Pool) error
	// Logger receives the router's logs. nil discards them.
	Logger *slog.Logger
}

// Constructor builds a router on rt. It runs the router's own startup path over cfg and
// returns a router with no endpoints. Background work it starts runs on rt and ends at Stop or
// when ctx is done.
//
// Neither the Constructor nor any work it starts (through rt.Go or rt.Every) may call rt.Rand
// before the Constructor returns; work it starts may draw after it returns. A host derives its
// streams' scope from the constructed router (for example its Pool), so a draw before the
// router exists would come from a stream the host has not yet determined. On a Runtime whose Go
// runs fn at once, work started with Go therefore waits (parks) before its first draw.
//
// A Constructor that fails has stopped whatever it started. A refusal of a plugin type is an
// error that wraps a *RefusalError.
type Constructor func(ctx context.Context, rt Runtime, cfg Config) (Router, error)

// RefusalError is a plugin type the router refused at construction: one it cannot host, one
// Config.Allow refused, or one it has not audited (see Config.Allow for the order).
type RefusalError struct {
	Plugin string // the plugin's name in the configuration (its type when it has none)
	Type   string
	Reason string
}

func (e *RefusalError) Error() string {
	return fmt.Sprintf("plugin %q (type %s): %s", e.Plugin, e.Type, e.Reason)
}

// Router is a constructed router.
type Router interface {
	// Requirements are what the configuration needs from the engines and the host.
	Requirements() Requirements
	// Pool is the pool the router serves.
	Pool() Pool
	// FlowControl describes how long the router can hold a request.
	FlowControl() FlowControl
	// IdleSkippable reports whether, while no request is anywhere in the router or the
	// engines and the engines' published state does not change, every background loop's
	// effect over a period is what its last two ticks produce. A host may then skip the
	// earlier ticks of such a period. False when some loop counts its own ticks.
	IdleSkippable() bool

	// SetEndpoint adds ep, or updates the endpoint with ep's name, and sets its readiness.
	// The router serves an endpoint only while it is ready and matches the pool.
	SetEndpoint(ep Endpoint, ready bool) error
	// RemoveEndpoint removes the named endpoint. Removing an unknown name is not an error.
	RemoveEndpoint(name string) error

	// Route asks the router where req goes. ctx is the request's lifetime and carries only
	// that: cancelling it while the router holds the request is the client giving up (a
	// ContextCancelled rejection). done is called exactly once: with a Flight and a Decision
	// when the router dispatches the request, or with a nil Flight, a zero Decision and a
	// Rejection when it does not. A request still undecided when the router stops is never
	// decided.
	//
	// done may be called before Route returns: a request refused before scheduling (a
	// malformed body) is, and on a Runtime whose Go runs fn before returning so is any
	// request the router decides without waiting. Otherwise it is called later, from work the
	// router runs on its Runtime (a goroutine started with Go, resumed by the host). A host
	// that must not be re-entered from Route defers what done does to its own next event.
	Route(ctx context.Context, req Request, done func(Flight, Decision, *Rejection))

	// WantsKVEvents reports whether the configuration consumes engines' KV events. When it
	// does not, the host need not call IngestKVEvents or ClearKVEvents.
	WantsKVEvents() bool
	// IngestKVEvents delivers one batch of KV events an engine published. endpoint is the
	// engine's address (Endpoint.Address); model is the model the engine serves. Batches
	// from one endpoint are delivered in the order the engine published them.
	IngestKVEvents(ctx context.Context, endpoint, model string, b KVBatch) error
	// ClearKVEvents forgets every block the endpoint at that address published. The host
	// calls it when an engine's cache is gone; a later batch from it starts afresh.
	ClearKVEvents(ctx context.Context, endpoint string) error

	// Stop ends every loop and goroutine the router started. The router is unusable after.
	Stop()
}

// Flight is a dispatched request, from its decision until its response ends. The host
// reports the response to it as the client would see it, then ends it exactly once: with a
// ResponseBody call whose endOfStream is true, or with Terminate. After that the host makes
// no more calls on it.
type Flight interface {
	// ResponseHeaders hands the router the response headers the engine sent and returns the
	// response headers after the router's handling: every header passed in, plus those the
	// router wrote. The caller must not modify the returned map.
	ResponseHeaders(h map[string]string) map[string]string
	// ResponseBody hands the router one chunk of the response body; endOfStream marks the
	// last. A non-streaming response is one chunk with endOfStream true.
	ResponseBody(chunk []byte, endOfStream bool)
	// Terminate ends a response that did not complete.
	Terminate(cause Termination)
	// EvictionSignal is closed when the router revokes the request after dispatch, to
	// reclaim capacity; the host then aborts it on its engine and ends the flight with
	// Terminate(Evicted). nil when the router never evicts.
	EvictionSignal() <-chan struct{}
}

// Termination is why a response ended before it completed.
type Termination string

const (
	// ClientDisconnect is a client that gave up while the response was in progress.
	ClientDisconnect Termination = "client-disconnect"
	// Evicted is a request the router revoked (Flight.EvictionSignal).
	Evicted Termination = "evicted"
	// Error is any other failure: the engine aborted or failed the request.
	Error Termination = "error"
)

// --- Data -----------------------------------------------------------------------------

// Requirements are facts about a router's configuration that a host compares with what it
// produces. They are facts, not verdicts: the router reports what its configuration reads,
// with its own defaults applied, and the host decides per endpoint whether it can supply it.
// They follow from the parsed configuration alone, before any plugin exists (Config.Check
// receives them then), and do not change over the router's life.
type Requirements struct {
	// MetricFlags are the flags naming an engine series that were set, by flag name without
	// dashes, with their values.
	MetricFlags map[string]string
	// InertFlags are the flags that were set and change how a deployed router is served or
	// observed (transport, TLS, replicas, shutdown, telemetry), never what it decides, as
	// "--name", sorted.
	InertFlags []string
	// Extractors are the configuration's engine-metrics extractors, in configuration order.
	Extractors []MetricsExtractor
	// MetricsSources are the configuration's metrics data sources, in configuration order.
	MetricsSources []MetricsSource
	// PrefixIndexes are the configuration's consumers of KV events, in configuration order.
	PrefixIndexes []PrefixIndex
	// Topology is what topology-aware routing reads from endpoint labels.
	Topology Topology
	// Disaggregates reports whether the configuration splits a request between a prefill
	// and a decode endpoint. A host whose deployment has both roles needs it.
	Disaggregates bool
}

// MetricsExtractor maps an engine's metric series onto what the router reads. Each endpoint's
// mapping is chosen by its EngineLabelKey label, or DefaultEngine when it has none.
type MetricsExtractor struct {
	Name string
	// Implicit is set on the extractor the router adds when the configuration declares
	// none.
	Implicit       bool
	EngineLabelKey string
	DefaultEngine  string
	// Engines are the configuration's engine mappings, in configuration order. An engine
	// with no entry here uses the router's built-in mapping for it, if it has one.
	Engines []EngineMapping
}

// EngineMapping is the series an extractor reads for one engine type.
type EngineMapping struct {
	Name string
	// Series maps each of the mapping's keys that is set (queuedRequestsSpec,
	// runningRequestsSpec, kvUsageSpec, loraSpec, cacheInfoSpec, cacheBlockSizeSpec,
	// cacheNumBlocksSpec) to its series, as the configuration spells it (labels included).
	Series map[string]string
	// Custom are the mapping's custom metrics, in configuration order.
	Custom []CustomMetric
}

// CustomMetric is a custom series an engine mapping reads.
type CustomMetric struct {
	Series   string
	Optional bool
}

// MetricsSource is a data source that scrapes each endpoint over HTTP.
type MetricsSource struct {
	Name string
	Path string
}

// PrefixIndex is a configuration element that indexes engines' KV events.
type PrefixIndex struct {
	Name string
	// BlockSize is the number of tokens the index cuts a block key at.
	BlockSize int
	// EngineType is the engine whose KV-event format the index reads ("vllm").
	EngineType string
}

// Topology is what topology-aware routing reads.
type Topology struct {
	// Requested reports whether the configuration routes by topology.
	Requested bool
	// Extractors are the topology extractors, in configuration order.
	Extractors []TopologyExtractor
}

// TopologyExtractor names the endpoint labels a topology extractor reads for each level.
type TopologyExtractor struct {
	Name     string
	Hostname string
	Rack     string
	Zone     string
	Region   string
}

// Pool is the pool a router serves. It follows from the parsed flags and manifests alone,
// before any plugin exists (Config.Check receives it then), and does not change over the
// router's life.
type Pool struct {
	Namespace string
	Name      string
	// Group and Kind name the pool's resource: an InferencePool, or the router's own
	// Deployment for a pool selected by labels.
	Group string
	Kind  string
	// TargetPorts are the ports the router addresses on each endpoint.
	TargetPorts []int
}

// ID is the pool's identity in the form Kubernetes renders a GroupKind and a NamespacedName:
// "Kind.group namespace/name", or "Kind namespace/name" for the core group. A host may key
// per-router state on it.
func (p Pool) ID() string {
	gk := p.Kind
	if p.Group != "" {
		gk += "." + p.Group
	}
	return gk + " " + p.Namespace + "/" + p.Name
}

// FlowControl is how long a router can hold a request before it decides. All zero when flow
// control is off.
type FlowControl struct {
	Enabled bool
	// MaxQueueTTL is the longest bounded queue-wait budget any request can have: the largest
	// positive of the configuration's default and no-endpoint budgets and every priority
	// band's own. Zero when none is bounded.
	MaxQueueTTL time.Duration
	// ExpirySweep is the interval at which expired requests are released, so a request is
	// released no later than its budget plus one sweep.
	ExpirySweep time.Duration
}

// Endpoint is one engine the router can send requests to (a pod).
type Endpoint struct {
	// Name identifies the endpoint (a pod name).
	Name     string
	Labels   map[string]string
	NodeName string
	IP       string
	Port     int
}

// Address is the endpoint's "host:port", the form a router names endpoints by.
func (e Endpoint) Address() string { return net.JoinHostPort(e.IP, strconv.Itoa(e.Port)) }

// Request is a request as a client sends it to the router. The router renders its request
// headers from the fields, in sorted key order: ":method" POST, ":path" Path, "content-type"
// application/json and "x-request-id" ID always, and the headers below for the fields that are
// set.
type Request struct {
	ID string
	// Path is the API path ("/v1/completions", "/v1/chat/completions").
	Path string
	// Body is the request body, verbatim.
	Body []byte
	// Stream reports whether the client asked for a streamed response; it agrees with Body.
	Stream bool
	// Objective names the request's objective (x-llm-d-inference-objective).
	Objective string
	// FairnessID is the request's fairness flow (x-llm-d-inference-fairness-id).
	FairnessID string
	// TTFTSLO is the request's time-to-first-token objective, sent when positive as whole
	// milliseconds rounded up (x-llm-d-slo-ttft-ms).
	TTFTSLO time.Duration
	// TTL is how long the client waits for the request, sent when positive as
	// time.Duration.String (x-llm-d-inference-ttl).
	TTL time.Duration
}

// Decision is where the router sent a request. Endpoints are named by address
// (Endpoint.Address).
type Decision struct {
	// Endpoint serves the request (decodes it, in a disaggregated deployment).
	Endpoint string
	// Prefill prefills it, when the router split the request; "" otherwise.
	Prefill string
	// Encoders encode its media, in the router's order; empty when none.
	Encoders []string
	// Scores are the final scores of the endpoints the router scored, by address.
	Scores map[string]float64
	// IncomingModel is the model the request named; TargetModel the model the router
	// resolved it to (a rewrite or an adapter). Equal, or TargetModel empty, when unchanged.
	IncomingModel string
	TargetModel   string
}

// Rejection is a request the router did not dispatch.
type Rejection struct {
	Kind RejectKind
	// Msg is the router's message. For Unclassified it is the full error text.
	Msg string
}

// RejectKind is why a router did not dispatch a request.
type RejectKind string

const (
	// NoEndpoints: no endpoint could serve the request.
	NoEndpoints RejectKind = "no-endpoints"
	// TTLExpired: it waited in flow control past its queue-wait budget.
	TTLExpired RejectKind = "ttl-expired"
	// Saturated: flow control's queues were at capacity.
	Saturated RejectKind = "saturated"
	// ContextCancelled: the client gave up while the router held it.
	ContextCancelled RejectKind = "context-cancelled"
	// Admission: an admission check refused it (for example a sheddable request while the
	// pool is saturated).
	Admission RejectKind = "admission"
	// Unavailable: the router could not serve it and gave no more specific reason.
	Unavailable RejectKind = "unavailable"
	// Unclassified: any outcome the router does not map to a kind above. A host should treat
	// it as a defect in the router or the hosting, not as load shedding.
	Unclassified RejectKind = "unclassified"
)

// RejectKinds are the kinds, in declaration order.
func RejectKinds() []RejectKind {
	return []RejectKind{NoEndpoints, TTLExpired, Saturated, ContextCancelled, Admission, Unavailable, Unclassified}
}

// KVBatch is one batch of KV events an engine published, in vLLM's vocabulary.
type KVBatch struct {
	// Timestamp is when the engine published the batch, on the runtime clock. A router
	// that keeps vLLM's float-seconds form converts it as float64(Timestamp.UnixNano())/1e9.
	Timestamp time.Time
	Events    []KVEvent
}

// KVEvent is BlockStored or BlockRemoved.
type KVEvent interface{ isKVEvent() }

// BlockStored is a run of blocks entering an engine's cache, each the parent of the next.
type BlockStored struct {
	// Hashes are the blocks' engine keys, in chain order.
	Hashes []uint64
	// Parent is the key of the first block's parent; 0 for none.
	Parent uint64
	// Tokens are the blocks' tokens, BlockSize per block; empty for a tier that records
	// locations only.
	Tokens    []uint32
	BlockSize int
	// Tier is the cache tier ("" for GPU, "cpu", ...).
	Tier string
}

// BlockRemoved is blocks leaving one tier of an engine's cache.
type BlockRemoved struct {
	Hashes []uint64
	Tier   string
}

func (BlockStored) isKVEvent()  {}
func (BlockRemoved) isKVEvent() {}

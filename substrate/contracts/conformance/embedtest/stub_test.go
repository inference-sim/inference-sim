package embedtest_test

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"maps"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/conformance/embedtest"
	"github.com/inference-sim/blis-substrate/contracts/embed"
)

// stubConfig is the stub router's configuration file (embed.Config.Config), in its own JSON.
type stubConfig struct {
	Plugins     []string `json:"plugins"`
	FlowControl bool     `json:"flowControl"`
	TTL         string   `json:"ttl"`       // queue-wait budget
	Unaudited   []string `json:"unaudited"` // types it would build but has not audited
	Capacity    int      `json:"capacity"`  // held requests at most
	Evict       bool     `json:"evict"`
	KV          bool     `json:"kv"`
	Unavailable bool     `json:"unavailable"` // serves nothing
}

// mutation breaks one rule of the contract, to show that the checks notice.
type mutation string

const (
	correct             mutation = ""
	saturatedAsAdmit    mutation = "Saturated reported as Admission"
	ttlAsCancelled      mutation = "TTLExpired reported as ContextCancelled"
	noEndpointsAsUnavl  mutation = "NoEndpoints reported as Unavailable"
	admitNotShed        mutation = "sheddable requests admitted under saturation"
	badBodyDispatched   mutation = "malformed body dispatched"
	randAtConstruction  mutation = "randomness drawn at construction"
	randInStartedWork   mutation = "randomness drawn at construction by a goroutine it started"
	wallClockSeed       mutation = "picks seeded from a counter, not the Runtime"
	leakOnStop          mutation = "Stop leaves a held request's goroutine parked"
	loopAfterStop       mutation = "Stop leaves the scrape loop running"
	headersDropped      mutation = "ResponseHeaders drops the headers passed in"
	ignoresAllow        mutation = "Allow not consulted"
	ignoresClear        mutation = "ClearKVEvents ignored"
	ignoresRemoved      mutation = "BlockRemoved ignored"
	neverEvicts         mutation = "eviction signal never closed"
	routesNotReady      mutation = "not-ready endpoints served"
	removeKeeps         mutation = "RemoveEndpoint ignored"
	doneTwice           mutation = "done called twice"
	noFlowControlFacts  mutation = "FlowControl() reports nothing"
	evictSignalNoEvict  mutation = "EvictionSignal without eviction"
	scoresForStrangers  mutation = "a score for an address that is no endpoint"
	failedConstructLeak mutation = "a refused construction leaves its loop running"
	parkedForever       mutation = "a goroutine parked on nothing Stop ends"
	unavailableAsUncl   mutation = "Unavailable reported as Unclassified"
	requestTTLIgnored   mutation = "Request.TTL ignored"
	unauditedAllowed    mutation = "an unaudited type built because Allow permits it"
	unauditedNoAsk      mutation = "an unaudited type refused without consulting Allow"
	checkWrapped        mutation = "Check's error wrapped"
	checkAfterPlugins   mutation = "Check called after Allow"
	checkIgnored        mutation = "Check never called"
	checkStalePool      mutation = "Check sees another Pool"
	inspectorStale      mutation = "KVBlocks reports removed blocks"
	addressKept         mutation = "SetEndpoint keeps the old address"
	unknownEngineKept   mutation = "a batch naming an unknown engine decoded as vLLM's"
)

var stubCounter int // wallClockSeed's process-wide "randomness"

// newStub returns a Constructor for the stub router with mutation m.
func newStub(m mutation) embed.Constructor {
	return func(ctx context.Context, rt embed.Runtime, cfg embed.Config) (embed.Router, error) {
		var c stubConfig
		if len(cfg.Config) > 0 {
			if err := json.Unmarshal(cfg.Config, &c); err != nil {
				return nil, fmt.Errorf("stub: configuration: %w", err)
			}
		}
		ttl := 30 * time.Second
		if c.TTL != "" {
			d, err := time.ParseDuration(c.TTL)
			if err != nil {
				return nil, err
			}
			ttl = d
		}
		s := &stub{m: m, rt: rt, cfg: c, ttl: ttl, eps: map[string]*stubEndpoint{}}
		s.ctx, s.cancel = context.WithCancel(ctx)
		if m == randAtConstruction {
			rt.Rand("picker").Float64()
		}
		if m == randInStartedWork {
			rt.Go(func() { rt.Rand("startup-jitter").Float64() })
		}
		if cfg.Check != nil && m != checkIgnored && m != checkAfterPlugins {
			if err := s.check(cfg.Check); err != nil {
				return nil, err
			}
		}
		s.startScrape()
		if m == parkedForever {
			rt.Go(func() { rt.Park(func() bool { return false }) })
		}
		for _, typ := range c.Plugins {
			if cfg.Allow == nil || m == ignoresAllow {
				continue
			}
			unaudited := slices.Contains(c.Unaudited, typ)
			if unaudited && m == unauditedNoAsk {
				s.cancel()
				return nil, fmt.Errorf("stub: %w", &embed.RefusalError{Plugin: typ, Type: typ, Reason: "not audited"})
			}
			if ok, reason := cfg.Allow(typ); !ok {
				if m != failedConstructLeak {
					s.cancel()
				}
				return nil, fmt.Errorf("stub: %w", &embed.RefusalError{Plugin: typ, Type: typ, Reason: reason})
			}
			if unaudited && m != unauditedAllowed {
				s.cancel()
				return nil, fmt.Errorf("stub: %w", &embed.RefusalError{Plugin: typ, Type: typ, Reason: "it has not been audited for hosting"})
			}
		}
		if cfg.Check != nil && m == checkAfterPlugins {
			if err := s.check(cfg.Check); err != nil {
				s.cancel()
				return nil, err
			}
		}
		return s, nil
	}
}

func (s *stub) check(check func(embed.Requirements, embed.Pool) error) error {
	pool := s.Pool()
	if s.m == checkStalePool {
		pool.Name = "other"
	}
	err := check(s.Requirements(), pool)
	if err != nil && s.m == checkWrapped {
		return fmt.Errorf("stub: %w", err)
	}
	return err
}

type stub struct {
	m      mutation
	rt     embed.Runtime
	cfg    stubConfig
	ttl    time.Duration
	ctx    context.Context
	cancel context.CancelFunc

	eps     map[string]*stubEndpoint // by name
	order   []string                 // endpoint names in the order they were added
	held    int
	flights []*stubFlight
}

type stubEndpoint struct {
	ep    embed.Endpoint
	ready bool
	busy  bool
	// blocks maps an engine block key to its chain key (the prefix up to and including it).
	blocks map[uint64]uint64
	chains map[uint64]bool
}

func (s *stub) startScrape() {
	scrapeCtx := s.ctx
	if s.m == loopAfterStop || s.m == failedConstructLeak {
		scrapeCtx = context.Background()
	}
	s.rt.Every(scrapeCtx, "metrics", 500*time.Millisecond, func(context.Context) {
		for _, name := range s.order {
			e := s.eps[name]
			if e == nil {
				continue
			}
			e.busy = s.scrape(e.ep)
		}
	})
}

// scrape reports whether the engine at ep says it is saturated (requests are waiting). An
// endpoint whose metrics it cannot read counts as saturated, as llm-d's saturation detector
// treats stale metrics: the stub routes only on metrics it has actually read.
func (s *stub) scrape(ep embed.Endpoint) bool {
	srv, ok := s.rt.(embed.RoundTripper)
	if !ok {
		return true
	}
	req, _ := http.NewRequestWithContext(s.ctx, http.MethodGet, "http://"+ep.Address()+"/metrics", nil)
	resp, err := srv.RoundTrip(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		return true
	}
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "vllm:num_requests_waiting{") {
			f := strings.Fields(line)
			v, _ := strconv.ParseFloat(f[len(f)-1], 64)
			return v > 0
		}
	}
	return true // no waiting series: unreadable
}

func (s *stub) Requirements() embed.Requirements {
	r := embed.Requirements{
		Extractors: []embed.MetricsExtractor{{Name: "metrics", Implicit: true, EngineLabelKey: "engine", DefaultEngine: "vllm"}},
	}
	if s.cfg.KV {
		r.PrefixIndexes = []embed.PrefixIndex{{Name: "prefix", BlockSize: 16, EngineType: "vllm"}}
	}
	return r
}

func (s *stub) Pool() embed.Pool {
	return embed.Pool{Namespace: "default", Name: "stub", Group: "apps", Kind: "Deployment", TargetPorts: []int{8000}}
}

func (s *stub) FlowControl() embed.FlowControl {
	if !s.cfg.FlowControl || s.m == noFlowControlFacts {
		return embed.FlowControl{}
	}
	return embed.FlowControl{Enabled: true, MaxQueueTTL: s.ttl}
}

func (s *stub) IdleSkippable() bool { return true }

func (s *stub) SetEndpoint(ep embed.Endpoint, ready bool) error {
	e, ok := s.eps[ep.Name]
	if !ok {
		// Saturated until its metrics are read, like an endpoint with stale metrics.
		e = &stubEndpoint{busy: true, blocks: map[uint64]uint64{}, chains: map[uint64]bool{}}
		s.eps[ep.Name] = e
		s.order = append(s.order, ep.Name)
	}
	if ok && s.m == addressKept {
		ep.IP, ep.Port = e.ep.IP, e.ep.Port
	}
	e.ep, e.ready = ep, ready
	return nil
}

func (s *stub) RemoveEndpoint(name string) error {
	if s.m == removeKeeps {
		return nil
	}
	delete(s.eps, name)
	s.order = slices.DeleteFunc(s.order, func(n string) bool { return n == name })
	return nil
}

func (s *stub) candidates() []*stubEndpoint {
	var out []*stubEndpoint
	for _, name := range s.order {
		if e := s.eps[name]; e.ready || s.m == routesNotReady {
			out = append(out, e)
		}
	}
	return out
}

func (s *stub) saturated(c []*stubEndpoint) bool {
	for _, e := range c {
		if !e.busy {
			return false
		}
	}
	return true
}

func (s *stub) Route(ctx context.Context, req embed.Request, done func(embed.Flight, embed.Decision, *embed.Rejection)) {
	var body struct {
		Model  string   `json:"model"`
		Prompt []uint32 `json:"prompt"`
	}
	if err := json.Unmarshal(req.Body, &body); err != nil && s.m != badBodyDispatched {
		done(nil, embed.Decision{}, &embed.Rejection{Kind: embed.Unclassified, Msg: "inference error: BadRequest - " + err.Error()})
		return
	}
	reject := func(kind embed.RejectKind, msg string) {
		done(nil, embed.Decision{}, &embed.Rejection{Kind: kind, Msg: msg})
	}
	s.rt.Go(func() {
		c := s.candidates()
		switch {
		case s.cfg.Unavailable:
			kind := embed.Unavailable
			if s.m == unavailableAsUncl {
				kind = embed.Unclassified
			}
			reject(kind, "no serving capacity")
			return
		case len(c) == 0:
			kind := embed.NoEndpoints
			if s.m == noEndpointsAsUnavl {
				kind = embed.Unavailable
			}
			reject(kind, "no endpoints available")
			return
		case s.saturated(c) && !s.cfg.FlowControl && req.Objective == embedtest.Sheddable && s.m != admitNotShed:
			reject(embed.Admission, "system saturated, sheddable request dropped")
			return
		case s.saturated(c) && s.cfg.FlowControl:
			if s.held >= max(1, s.cfg.Capacity) {
				kind := embed.Saturated
				if s.m == saturatedAsAdmit {
					kind = embed.Admission
				}
				reject(kind, "queue at capacity")
				return
			}
			s.held++
			budget := s.ttl
			if req.TTL > 0 && req.TTL < budget && s.m != requestTTLIgnored {
				budget = req.TTL
			}
			wait, stop := s.rt.WithTimeout(ctx, budget)
			if s.cfg.Evict {
				s.evictOne()
			}
			stopped := s.ctx
			if s.m == leakOnStop {
				stopped = context.Background()
			}
			s.rt.Park(func() bool {
				return wait.Err() != nil || stopped.Err() != nil || !s.saturated(s.candidates())
			})
			s.held--
			expired := wait.Err() != nil
			stop()
			switch {
			case stopped.Err() != nil:
				return // never decided
			case ctx.Err() != nil:
				reject(embed.ContextCancelled, "client disconnected")
				return
			case expired:
				kind := embed.TTLExpired
				if s.m == ttlAsCancelled {
					kind = embed.ContextCancelled
				}
				reject(kind, "request timed out in queue")
				return
			}
			c = s.candidates()
		}
		e := s.pick(c, body.Prompt)
		d := embed.Decision{Endpoint: e.ep.Address(), IncomingModel: body.Model, Scores: map[string]float64{e.ep.Address(): 1}}
		if s.m == scoresForStrangers {
			d.Scores["192.0.2.1:8000"] = 0
		}
		f := &stubFlight{s: s, sheddable: req.Objective == embedtest.Sheddable}
		if s.cfg.Evict || s.m == evictSignalNoEvict {
			f.evict = make(chan struct{})
		}
		s.flights = append(s.flights, f)
		done(f, d, nil)
		if s.m == doneTwice {
			done(f, d, nil)
		}
	})
}

// pick is the endpoint holding the longest published prefix of prompt, among idle ones when
// any is idle; ties, and every pick without KV events, go to a random one.
func (s *stub) pick(c []*stubEndpoint, prompt []uint32) *stubEndpoint {
	if idle := slices.DeleteFunc(slices.Clone(c), func(e *stubEndpoint) bool { return e.busy }); len(idle) > 0 {
		c = idle
	}
	if s.cfg.KV {
		keys := chainKeys(prompt, 16)
		best, bestN := []*stubEndpoint(nil), -1
		for _, e := range c {
			n := 0
			for n < len(keys) && e.chains[keys[n]] {
				n++
			}
			switch {
			case n > bestN:
				best, bestN = []*stubEndpoint{e}, n
			case n == bestN:
				best = append(best, e)
			}
		}
		c = best
	}
	if s.m == wallClockSeed {
		stubCounter++
		return c[stubCounter%len(c)]
	}
	return c[s.rt.Rand("picker").IntN(len(c))]
}

func (s *stub) evictOne() {
	if s.m == neverEvicts {
		return
	}
	for _, f := range s.flights {
		if f.sheddable && !f.ended && f.evict != nil && !f.evicted {
			f.evicted = true
			close(f.evict)
			return
		}
	}
}

func chainKeys(tokens []uint32, blockSize int) []uint64 {
	var out []uint64
	var parent uint64
	for i := 0; i+blockSize <= len(tokens); i += blockSize {
		parent = chainKey(parent, tokens[i:i+blockSize])
		out = append(out, parent)
	}
	return out
}

func chainKey(parent uint64, block []uint32) uint64 {
	h := fnv.New64a()
	fmt.Fprint(h, parent, block)
	return h.Sum64()
}

func (s *stub) WantsKVEvents() bool { return s.cfg.KV }

func (s *stub) endpointAt(addr string) *stubEndpoint {
	for _, name := range s.order {
		if e := s.eps[name]; e.ep.Address() == addr {
			return e
		}
	}
	return nil
}

func (s *stub) IngestKVEvents(_ context.Context, endpoint, _ string, b embed.KVBatch) error {
	e := s.endpointAt(endpoint)
	if e == nil {
		return fmt.Errorf("stub: no endpoint %s", endpoint)
	}
	if !b.Known() && s.m != unknownEngineKept {
		return fmt.Errorf("stub: engine %q is not a kind the contract defines", b.Engine)
	}
	for _, ev := range b.Events {
		switch ev := ev.(type) {
		case embed.BlockStored:
			parent := e.blocks[ev.Parent]
			for i, h := range ev.Hashes {
				parent = chainKey(parent, ev.Tokens[i*ev.BlockSize:(i+1)*ev.BlockSize])
				e.blocks[h] = parent
				e.chains[parent] = true
			}
		case embed.BlockRemoved:
			if s.m == ignoresRemoved {
				continue
			}
			for _, h := range ev.Hashes {
				delete(e.chains, e.blocks[h])
				delete(e.blocks, h)
			}
		}
	}
	return nil
}

func (s *stub) ClearKVEvents(_ context.Context, endpoint string) error {
	if e := s.endpointAt(endpoint); e != nil && s.m != ignoresClear {
		e.blocks, e.chains = map[uint64]uint64{}, map[uint64]bool{}
	}
	return nil
}

func (s *stub) Stop() { s.cancel() }

var _ embed.KVIndexInspector = (*stub)(nil)

// KVBlocks implements embed.KVIndexInspector. The stub keeps one tier.
func (s *stub) KVBlocks(endpoint, _ string) (map[uint64]bool, error) {
	e := s.endpointAt(endpoint)
	if e == nil || !s.cfg.KV {
		return nil, fmt.Errorf("stub: no KV index for %s", endpoint)
	}
	out := map[uint64]bool{}
	for h := range e.blocks {
		out[h] = true
	}
	if s.m == inspectorStale {
		out[0x1000] = true
	}
	return out, nil
}

type stubFlight struct {
	s         *stub
	sheddable bool
	evict     chan struct{}
	evicted   bool
	ended     bool
}

func (f *stubFlight) ResponseHeaders(h map[string]string) map[string]string {
	out := maps.Clone(h)
	if f.s.m == headersDropped {
		out = map[string]string{}
	}
	out["x-stub-served"] = "true"
	return out
}

func (f *stubFlight) ResponseBody(_ []byte, endOfStream bool) {
	if endOfStream {
		f.ended = true
	}
}

func (f *stubFlight) Terminate(embed.Termination) { f.ended = true }

func (f *stubFlight) EvictionSignal() <-chan struct{} {
	return f.evict
}

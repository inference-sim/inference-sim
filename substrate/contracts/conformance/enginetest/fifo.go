package enginetest

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
)

// FIFO is a deliberately small engine that is not a model of any real server: first come
// first served, chunked prefill, one decode token a step plus optional drafts, and no KV
// cache. It exists to prove the contract is engine-neutral. It is configured, built and
// driven through the same Spec, Factory and Engine a real engine uses, so a simulator
// whose tests run it has shown that a second engine plugs in with no change to the
// simulator or to this module.
//
// Two request tags exercise the timer and transfer halves of the contract:
// "fifo.hold_until" (an instant before which the request may not be admitted, reported
// through NextWake) and "fifo.load_tokens" (tokens to load before it runs, started as a
// transfer at its first admission attempt).
type FIFO struct {
	cfg      FIFOConfig
	waiting  []*fifoReq
	running  []*fifoReq
	byID     map[string]*fifoReq
	inFlight []engine.Batch
	nextID   engine.BatchID
	nextXfer engine.TransferID
	xfers    map[engine.TransferID]*fifoReq
	finished int
}

// FIFOConfig is FIFO's configuration block.
type FIFOConfig struct {
	MaxRunning  int `yaml:"max_running"`
	TokenBudget int `yaml:"token_budget"`
	// InFlight is how many batches may be in flight; zero means one.
	InFlight int `yaml:"in_flight,omitempty"`
	// Drafts is the speculative tokens each decode verifies.
	Drafts int `yaml:"drafts,omitempty"`
	// StepBaseUs and StepPerTokenUs price a step as base + per-token x tokens.
	StepBaseUs     int64 `yaml:"step_base_us"`
	StepPerTokenUs int64 `yaml:"step_per_token_us"`
	// Stages, above one, splits each step evenly into that many pipeline stages
	// (engine.Pipelined): what a simulator overlaps when it runs more than one batch in
	// flight on a pipeline. Zero or one is a single stage.
	Stages int `yaml:"stages,omitempty"`
}

type fifoReq struct {
	r         engine.Request
	computed  int
	out       int
	holdUntil engine.Time
	load      int
	loading   bool
	loaded    bool
	inFlight  bool
	done      bool
}

// FIFOKind is the kind FIFOFactory builds.
const FIFOKind = "enginetest-fifo"

// FIFOFactory builds FIFO engines. It is not registered by default; a test registers it.
type FIFOFactory struct{}

// Kind implements engine.Factory.
func (FIFOFactory) Kind() string { return FIFOKind }

// New implements engine.Factory, decoding the block strictly: an unknown key is an error.
// It decodes the node itself rather than a re-encoding of it, so an error names the line of
// the document the block came from.
func (FIFOFactory) New(s engine.Spec) (engine.Engine, error) {
	if s.Kind != FIFOKind {
		return nil, fmt.Errorf("fifo: spec kind %q is not %q", s.Kind, FIFOKind)
	}
	block := &s.Config
	if block.Kind == yaml.DocumentNode && len(block.Content) == 1 {
		block = block.Content[0]
	}
	if block.Kind == yaml.MappingNode {
		known := map[string]bool{}
		t := reflect.TypeOf(FIFOConfig{})
		for i := range t.NumField() {
			known[strings.Split(t.Field(i).Tag.Get("yaml"), ",")[0]] = true
		}
		for i := 0; i+1 < len(block.Content); i += 2 {
			if k := block.Content[i]; !known[k.Value] {
				return nil, fmt.Errorf("fifo: config: line %d: field %s not found in FIFOConfig", k.Line, k.Value)
			}
		}
	}
	var cfg FIFOConfig
	if err := block.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("fifo: config: %w", err)
	}
	return NewFIFO(cfg)
}

// NewFIFO builds a FIFO engine.
func NewFIFO(cfg FIFOConfig) (*FIFO, error) {
	if cfg.MaxRunning < 1 || cfg.TokenBudget < 1 || cfg.InFlight < 0 || cfg.Drafts < 0 ||
		cfg.StepBaseUs < 0 || cfg.StepPerTokenUs < 0 || cfg.Stages < 0 {
		return nil, fmt.Errorf("fifo: invalid config %+v", cfg)
	}
	if cfg.TokenBudget < 1+cfg.Drafts {
		return nil, fmt.Errorf("fifo: a token budget of %d cannot hold one decode of %d tokens", cfg.TokenBudget, 1+cfg.Drafts)
	}
	if cfg.InFlight == 0 {
		cfg.InFlight = 1
	}
	return &FIFO{cfg: cfg, byID: map[string]*fifoReq{}, xfers: map[engine.TransferID]*fifoReq{}}, nil
}

// Add implements engine.Engine.
func (f *FIFO) Add(r engine.Request, _ engine.Time) error {
	if len(r.Prompt) == 0 {
		return fmt.Errorf("fifo: request %s has an empty prompt", r.ID)
	}
	if _, dup := f.byID[r.ID]; dup {
		return fmt.Errorf("fifo: request %s added twice", r.ID)
	}
	q := &fifoReq{r: r}
	if v, ok := r.Tags["fifo.hold_until"]; ok {
		t, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return fmt.Errorf("fifo: request %s: fifo.hold_until: %w", r.ID, err)
		}
		q.holdUntil = t
	}
	if v, ok := r.Tags["fifo.load_tokens"]; ok {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			return fmt.Errorf("fifo: request %s: fifo.load_tokens %q", r.ID, v)
		}
		q.load = n
	}
	f.byID[r.ID] = q
	f.waiting = append(f.waiting, q)
	return nil
}

// Schedule implements engine.Engine.
func (f *FIFO) Schedule(now engine.Time) (engine.Batch, error) {
	if len(f.inFlight) >= f.cfg.InFlight {
		return engine.Batch{}, engine.ErrNoCapacity
	}
	var b engine.Batch
	budget := f.cfg.TokenBudget
	for _, q := range f.running {
		if q.inFlight || budget == 0 {
			continue
		}
		if s, ok := f.share(q, budget); ok {
			budget -= s.Tokens
			b.Reqs = append(b.Reqs, s)
			q.inFlight = true
		}
	}
	kept := f.waiting[:0]
	for _, q := range f.waiting {
		switch {
		case len(f.running) >= f.cfg.MaxRunning || budget == 0 || q.holdUntil > now || q.loading:
			kept = append(kept, q)
		case q.load > 0 && !q.loaded:
			f.nextXfer++
			q.loading = true
			f.xfers[f.nextXfer] = q
			b.TransfersStarted = append(b.TransfersStarted, engine.Transfer{
				ID: f.nextXfer, Request: q.r.ID, Kind: engine.TransferLoad, Tokens: q.load,
			})
			kept = append(kept, q)
		default:
			s, _ := f.share(q, budget)
			budget -= s.Tokens
			b.Reqs = append(b.Reqs, s)
			q.inFlight = true
			f.running = append(f.running, q)
		}
	}
	f.waiting = kept
	if b.Empty() {
		return b, nil
	}
	f.nextID++
	b.ID = f.nextID
	f.inFlight = append(f.inFlight, b)
	return b, nil
}

// share is q's part of a batch with budget tokens left: the next prefill chunk, or one
// decode with its drafts.
func (f *FIFO) share(q *fifoReq, budget int) (engine.Scheduled, bool) {
	s := engine.Scheduled{ID: q.r.ID, Computed: q.computed + q.out, PromptLen: len(q.r.Prompt)}
	if q.computed < len(q.r.Prompt) {
		s.Tokens = min(len(q.r.Prompt)-q.computed, budget)
		return s, true
	}
	if budget < 1+f.cfg.Drafts {
		return s, false
	}
	s.Tokens, s.Drafts = 1+f.cfg.Drafts, f.cfg.Drafts
	return s, true
}

// StepTime implements engine.Engine. A pipelined FIFO's step is at least one microsecond a
// stage, so StageTimes always sums to it.
func (f *FIFO) StepTime(b engine.Batch) engine.Duration {
	return max(1, engine.Duration(max(f.cfg.Stages, 1)), f.cfg.StepBaseUs+f.cfg.StepPerTokenUs*int64(b.Tokens()))
}

// StageTimes implements engine.Pipelined: the step split as evenly as microseconds allow,
// the remainder on the last stage, each stage at least one microsecond.
func (f *FIFO) StageTimes(b engine.Batch) []engine.Duration {
	n := max(f.cfg.Stages, 1)
	total := max(f.StepTime(b), engine.Duration(n))
	out := make([]engine.Duration, n)
	for i := range out {
		out[i] = total / engine.Duration(n)
	}
	out[n-1] += total % engine.Duration(n)
	return out
}

// Complete implements engine.Engine.
func (f *FIFO) Complete(id engine.BatchID, outcome engine.Outcome, _ engine.Time) (engine.Result, error) {
	idx := -1
	for i, b := range f.inFlight {
		if b.ID == id {
			idx = i
		}
	}
	switch {
	case idx < 0:
		return engine.Result{}, engine.ErrUnknownBatch
	case idx > 0:
		return engine.Result{}, engine.ErrOutOfOrder
	}
	b := f.inFlight[0]
	f.inFlight = f.inFlight[1:]
	res := engine.Result{Emitted: map[string]int{}}
	for _, s := range b.Reqs {
		q := f.byID[s.ID]
		q.inFlight = false
		if q.done {
			continue // finished while its batch ran
		}
		emitted := 0
		if s.Drafts == 0 && q.computed < len(q.r.Prompt) {
			q.computed += s.Tokens
			if q.computed == len(q.r.Prompt) && len(q.r.Output) > 0 {
				emitted = 1
				res.FirstTokens = append(res.FirstTokens, q.r.ID)
			}
		} else {
			emitted = 1 + min(outcome.Accepted[s.ID], s.Drafts)
		}
		emitted = min(emitted, len(q.r.Output)-q.out)
		if q.r.MaxOutputTokens > 0 {
			emitted = min(emitted, q.r.MaxOutputTokens-q.out)
		}
		q.out += emitted
		if emitted > 0 {
			res.Emitted[q.r.ID] = emitted
		}
		if q.computed < len(q.r.Prompt) {
			continue
		}
		switch {
		case q.out >= len(q.r.Output):
			res.Finished = append(res.Finished, f.finish(q, engine.FinishStop))
		case q.r.MaxOutputTokens > 0 && q.out >= q.r.MaxOutputTokens:
			res.Finished = append(res.Finished, f.finish(q, engine.FinishLength))
		}
	}
	return res, nil
}

func (f *FIFO) finish(q *fifoReq, why engine.FinishReason) engine.Finished {
	q.done = true
	f.finished++
	f.running = remove(f.running, q)
	f.waiting = remove(f.waiting, q)
	return engine.Finished{ID: q.r.ID, Reason: why, OutputTokens: q.out}
}

func remove(qs []*fifoReq, q *fifoReq) []*fifoReq {
	for i, x := range qs {
		if x == q {
			return append(qs[:i], qs[i+1:]...)
		}
	}
	return qs
}

// TransferDone implements engine.Engine. A failed load is treated as a miss: the request
// runs and computes its prompt, which FIFO always does anyway.
func (f *FIFO) TransferDone(id engine.TransferID, _ bool, _ engine.Time) error {
	q, ok := f.xfers[id]
	if !ok {
		return engine.ErrUnknownTransfer
	}
	delete(f.xfers, id)
	q.loading, q.loaded = false, true
	return nil
}

// Finish implements engine.Engine.
func (f *FIFO) Finish(id string, why engine.FinishReason, _ engine.Time) error {
	q, ok := f.byID[id]
	if !ok || q.done {
		return nil
	}
	f.finish(q, why)
	return nil
}

// NextWake implements engine.Engine: the earliest hold that ends after now.
func (f *FIFO) NextWake(now engine.Time) (engine.Time, bool) {
	var at engine.Time
	found := false
	for _, q := range f.waiting {
		if q.holdUntil > now && (!found || q.holdUntil < at) {
			at, found = q.holdUntil, true
		}
	}
	return at, found
}

// InFlightLimit implements engine.Engine.
func (f *FIFO) InFlightLimit() int { return f.cfg.InFlight }

// Fields implements engine.Engine. The test engine reads one field, its in-flight bound,
// and models it; a real engine returns the list of its whole upstream release.
func (f *FIFO) Fields() fields.FieldList {
	return fields.FieldList{
		Kind:   fields.Kind,
		Engine: fields.Engine{Kind: "fifo", Version: "0"},
		Fields: []fields.Field{{
			Name:   "max_concurrent_batches",
			Source: fields.SourceConfig,
			Status: fields.StatusModelled,
			Evidence: fields.Evidence{
				Traces: []string{"contracts/conformance/enginetest/enginetest_test.go"},
			},
			Estimation: fields.Estimation{Rule: fields.RuleNone},
		}},
	}
}

// Metrics implements engine.Engine.
func (f *FIFO) Metrics(now engine.Time) engine.Metrics {
	one := func(name string, kind engine.MetricKind, v int) engine.Family {
		return engine.Family{Name: name, Kind: kind, Samples: []engine.Sample{{Name: name, Value: float64(v)}}}
	}
	return engine.Metrics{At: now, Families: []engine.Family{
		one("fifo_waiting", engine.Gauge, len(f.waiting)),
		one("fifo_running", engine.Gauge, len(f.running)),
		one("fifo_finished_total", engine.Counter, f.finished),
	}}
}

// Snapshot implements engine.Engine.
func (f *FIFO) Snapshot() engine.State {
	return engine.State{Waiting: len(f.waiting), Running: len(f.running), InFlight: len(f.inFlight)}
}

var _ engine.Engine = (*FIFO)(nil)

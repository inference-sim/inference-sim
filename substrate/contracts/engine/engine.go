// Package engine is the interface an inference-engine model implements and a
// discrete-event simulator calls: the counterpart of package kernel for scheduling
// rather than pricing.
//
// The boundary follows from who owns time. The simulator owns the clock, the event queue
// and the decision of when to step. An engine owns every scheduling, KV-cache and
// connector decision, and the metrics those decisions produce. An engine never reads a
// clock and never schedules an event: every call carries the simulated instant as a
// value, and where an engine rule makes work wait until a later instant, the engine
// reports that instant through NextWake and the simulator decides what to do with it.
//
// Nothing here names an engine. A configuration reaches an engine as an opaque Spec that
// the simulator hands over unread, so a second engine implements this interface in its
// own repository without changing this package or the simulator. Package enginetest holds
// the contract checks that every implementation, and every fake a simulator's tests
// build, runs against itself.
package engine

import (
	"errors"
	"fmt"
)

// Time is a simulated instant in microseconds since the start of a run. It is a value,
// never a clock reading.
type Time = int64

// Duration is a simulated span in microseconds.
type Duration = int64

// Engine is one simulated engine instance: the state machine behind one server process.
//
// Calls are sequential. The simulator never calls an engine from two goroutines, and an
// engine starts none.
//
// Every call carries the instant it happens at. The instants of Add, Schedule,
// TransferDone and Finish never decrease. Complete carries the instant its batch's step
// ends, which a simulator that applies a step's results as the step starts passes before
// it delivers the inputs that arrive during the step. An engine therefore stamps what a
// batch makes visible at Complete's instant, and must not assume a later call's instant
// is later than it.
//
// After every call that can change engine state (Add, Schedule, Complete, TransferDone,
// Finish, and a wake firing) the simulator calls NextWake again. That is the whole timer
// protocol: an engine has no other way to ask for future attention, and needs none.
type Engine interface {
	// Add hands the engine a request that has arrived at now. The engine decides when it
	// runs; Add schedules nothing.
	Add(r Request, now Time) error

	// Schedule decides one batch at now: which requests run, how many tokens each, which
	// are preempted, and which KV transfers start. It returns ErrNoCapacity when as many
	// batches as InFlightLimit allows are already in flight. A batch that runs requests
	// gets an ID greater than every earlier batch of this engine, and stays in flight until
	// Complete names it. A batch that runs none (Empty) is not in flight, but may still
	// start transfers, which the simulator prices like any other.
	Schedule(now Time) (Batch, error)

	// StepTime prices a batch that runs requests: how long its forward pass takes, at least
	// one microsecond, so a step always advances time. An Empty batch runs no forward pass,
	// so a simulator does not price it.
	StepTime(b Batch) Duration

	// Complete applies the outcome of an in-flight batch, whose step ends at end: tokens
	// sampled and drafts accepted. It returns what became visible: first tokens, finishes
	// and KV events. Batches complete in the order they were scheduled; Complete returns
	// ErrOutOfOrder otherwise and ErrUnknownBatch for an ID not in flight.
	Complete(id BatchID, outcome Outcome, end Time) (Result, error)

	// TransferDone reports that an asynchronous KV load or store the engine started has
	// finished at now, successfully or not. An ID the engine does not know is an error.
	TransferDone(id TransferID, ok bool, now Time) error

	// Finish ends a request at now for a reason the engine did not decide: a client abort
	// or a timeout. It frees what the request holds. Finishing an unknown or an already
	// finished request is a no-op, as the engine's finish_requests is.
	Finish(id string, why FinishReason, now Time) error

	// NextWake is the earliest instant after now at which engine state may change without
	// another external input: a deferred admission becoming possible, a cooldown expiring,
	// a deadline passing. False means there is no such instant. See the wake protocol
	// below.
	NextWake(now Time) (Time, bool)

	// InFlightLimit is how many batches may be in flight at once: the engine's own bound
	// (for the engine, max_concurrent_batches), fixed at construction and at least one.
	InFlightLimit() int

	// Metrics returns the engine's exported metrics as of now.
	Metrics(now Time) Metrics

	// Snapshot returns the engine's current state for routing and tests. It is a copy the
	// caller may keep.
	Snapshot() State
}

// Pipelined is implemented by an engine whose batches run through more than one pipeline
// stage, each stage on its own devices: vLLM with pipeline_parallel_size > 1. It is
// optional; a simulator that does not look for it times every batch as one step of
// StepTime on one device, which serializes the stages and so is conservative.
//
// StageTimes splits StepTime(b) by stage, in pipeline order: one entry per stage, each at
// least one microsecond, summing exactly to StepTime(b). A simulator that honours it starts
// stage s of a batch once stage s-1 of that batch has ended and stage s has finished the
// batch before it, and ends the batch when its last stage ends: a new batch can enter the
// pipeline every slowest-stage time, while each batch's output still takes the whole sum.
// An engine with one stage returns a single entry equal to StepTime(b).
type Pipelined interface {
	StageTimes(b Batch) []Duration
}

// The wake protocol. NextWake answers one question: with no new input, when could this
// engine's state change? The simulator owns everything that follows from the answer.
//
//   - It asks after every state-changing call, so an answer is never stale by more than
//     one call. An engine therefore never cancels a wake: it reports a different instant,
//     or false, the next time it is asked.
//   - An instant earlier than the pending wake supersedes it; a later one, or false,
//     replaces it. A superseded wake is dropped without running.
//   - When a wake fires, the simulator steps the engine at the wake instant.
//   - An engine with unfinished work, no batch in flight, no transfer outstanding and an
//     empty Schedule must report a wake. Otherwise nothing would ever step it again. This
//     is the liveness law enginetest checks.
//   - At one instant a simulator processes, in this order: arrivals; transfer
//     completions; batch completions; wakes and steps; timeouts last, so a request that
//     completes at the instant its deadline passes has completed. An engine may rely on
//     that order.
//
// The order is stated here rather than left to each simulator because the same events
// processed in a different order at one instant are a different run.

// ErrNoCapacity is returned by Schedule when InFlightLimit batches are in flight.
var ErrNoCapacity = errors.New("engine: no free in-flight batch slot")

// ErrOutOfOrder is returned by Complete when a batch is completed before an earlier one.
var ErrOutOfOrder = errors.New("engine: batches complete in the order they were scheduled")

// ErrUnknownBatch is returned by Complete for an ID that is not in flight.
var ErrUnknownBatch = errors.New("engine: no such batch in flight")

// ErrUnknownTransfer is returned by TransferDone for an ID the engine did not start.
var ErrUnknownTransfer = errors.New("engine: no such transfer outstanding")

// BatchID identifies one scheduled batch for the life of the engine.
type BatchID uint64

// TransferID identifies one asynchronous KV transfer for the life of the engine.
type TransferID uint64

// Request is what an engine is told about a request: its identity, its prompt, the
// output the simulator has assigned it, and the scheduling facts the simulator owns.
type Request struct {
	ID string
	// Prompt is the prompt's token IDs, in order. An engine may hash them for its prefix
	// cache; it must not modify them.
	Prompt []int32
	// Output is the output the simulator assigned the request: what the model will sample,
	// one of the inputs a simulator estimates rather than an engine decides. An engine
	// reads Output[i] only once token i is sampled, and never consults the length of
	// Output for an admission, ordering or preemption decision: only to stop the request
	// where its output ends, as a real engine stops at an end-of-sequence token.
	Output []int32
	// MaxOutputTokens is the client's output budget; zero means none beyond the engine's
	// own context limit.
	MaxOutputTokens int
	// Priority orders requests under a priority policy; lower is more urgent.
	Priority int
	// ArrivalTime is when the request arrived, which FCFS and priority policies break ties
	// by. It may be earlier than the instant of the Add call that delivers it.
	ArrivalTime Time
	// Adapter names a LoRA adapter, or is empty for the base model.
	Adapter string
	// CacheSalt isolates the request's prefix-cache entries from requests with another
	// salt; an empty salt shares with every unsalted request.
	CacheSalt string
	// Tags are opaque facts a simulator passes through for an engine that reads them. An
	// engine ignores keys it does not know.
	Tags map[string]string
	// RemoteDecode marks a disaggregated prefill: the request computes its prompt here and a
	// peer engine decodes it, pulling its KV. When it finishes, the engine holds that KV for
	// the pull and reports it in Result.Handoffs, until the simulator ends the handoff
	// (TransferDone with Handoff.Transfer). An engine that does not disaggregate refuses it.
	RemoteDecode bool
	// RemotePrefill marks a disaggregated decode: a peer engine computed the prompt, and this
	// engine loads its KV rather than computing it. The engine decides when the load starts (a
	// TransferLoad among a Batch's TransfersStarted), and runs the request once the simulator
	// reports that transfer done. An engine that does not disaggregate refuses it.
	RemotePrefill bool
}

// Batch is one scheduling decision.
type Batch struct {
	ID BatchID
	// Reqs are the requests that run in this batch, with their tokens, in the engine's own
	// order.
	Reqs []Scheduled
	// Preempted are the requests this decision evicted, in eviction order.
	Preempted []string
	// TransfersStarted are the asynchronous KV transfers this decision started. The
	// simulator prices each and reports its end with TransferDone.
	TransfersStarted []Transfer
	// Finished are requests this decision ended without running them, with why: a request
	// whose asynchronous KV load failed under a fail-the-request policy, which the engine
	// finishes at its next schedule. A request finished here never appears in a Result.
	Finished []Finished
}

// Empty reports whether the batch runs nothing.
func (b Batch) Empty() bool { return len(b.Reqs) == 0 }

// Tokens returns the batch's total scheduled tokens.
func (b Batch) Tokens() int {
	n := 0
	for _, r := range b.Reqs {
		n += r.Tokens
	}
	return n
}

// Scheduled is one request's share of a batch.
type Scheduled struct {
	ID string
	// Tokens is how many tokens run for this request in this batch, including any
	// speculative drafts it verifies.
	Tokens int
	// Computed is how many tokens were already computed before this batch, including
	// prefix-cache hits. It is what separates a chunked-prefill tail from a decode.
	Computed int
	// PromptLen is the request's prompt length.
	PromptLen int
	// CachedTokens is the part of Computed that a prefix-cache hit supplied at this
	// batch's admission, zero otherwise.
	CachedTokens int
	// PrefixLookup reports that the engine looked the request's prompt up in its prefix
	// cache at this batch's admission, so CachedTokens is that lookup's hit, zero for a miss.
	// An engine with no prefix cache, or one that skipped the lookup, leaves it false. A
	// simulator counts a hit rate over the lookups a request's first admission made.
	PrefixLookup bool
	// Drafts is how many of Tokens are speculative tokens being verified.
	Drafts int
}

// Transfer is one asynchronous KV transfer an engine started.
type Transfer struct {
	ID TransferID
	// Request is the request the transfer serves.
	Request string
	// Kind says which way the KV moves.
	Kind TransferKind
	// Tokens and Blocks size the transfer.
	Tokens int
	Blocks int
	// Tier names the offload tier or remote peer as the engine's configuration names it.
	Tier string
}

// TransferKind says which way a transfer moves KV.
type TransferKind string

const (
	// TransferLoad moves KV into the engine's cache: an offload hit or a remote prefill.
	TransferLoad TransferKind = "load"
	// TransferStore moves KV out of it: an offload write or a handoff to a peer.
	TransferStore TransferKind = "store"
)

// Outcome is what executing a batch produced that the engine did not decide, supplied by
// the simulator from inputs it owns.
type Outcome struct {
	// Accepted maps a request that verified drafts to how many of them the target model
	// accepted, from the simulator's calibrated acceptance. A request absent from it
	// accepted none. An engine that verified no drafts ignores it.
	Accepted map[string]int
}

// Result is what a completed batch made visible.
type Result struct {
	// Emitted maps each request that produced output tokens in this batch to how many.
	Emitted map[string]int
	// FirstTokens are the requests that produced their first output token in this batch.
	FirstTokens []string
	// Finished are the requests that finished in this batch, with why.
	Finished []Finished
	// KVEvents are the prefix-cache changes this batch caused, in order.
	KVEvents []KVEvent
	// Handoffs are the disaggregated prefills (Request.RemoteDecode) this batch finished whose
	// KV the engine holds for a peer's pull, in the order they finished.
	Handoffs []Handoff
}

// Handoff is the KV a finished disaggregated prefill left for a peer engine to pull. The
// engine holds it until the simulator reports Transfer done: when the peer's pull has landed,
// or when the hold has run out.
type Handoff struct {
	ID string
	// Transfer is the handoff, which the simulator ends with TransferDone.
	Transfer TransferID
	// Blocks are the engine's IDs of the KV blocks held, opaque to the simulator, which may
	// count distinct blocks across handoffs: a block a prefix cache shares appears in several.
	Blocks []int64
}

// Finished is one request an engine finished.
type Finished struct {
	ID           string
	Reason       FinishReason
	OutputTokens int
}

// FinishReason says why a request finished.
type FinishReason string

const (
	FinishStop    FinishReason = "stop"    // the assigned output ended
	FinishLength  FinishReason = "length"  // the output budget or the context limit
	FinishAbort   FinishReason = "abort"   // the client went away
	FinishTimeout FinishReason = "timeout" // a deadline passed
	FinishError   FinishReason = "error"   // the engine refused or failed the request
)

// KVEvent is one change to the set of KV blocks an engine's prefix cache can serve.
type KVEvent struct {
	Removed bool
	Hash    string
	Parent  string
	Tokens  []int32
	// Tier is where the block is cached: empty for the device cache, else the tier's name.
	Tier string
}

// State is an engine's current state as a router or a test sees it.
type State struct {
	Waiting  int
	Running  int
	InFlight int
	// KVBlocksTotal and KVBlocksUsed describe the device KV pool, in the engine's blocks.
	KVBlocksTotal int64
	KVBlocksUsed  int64
	// KVBlockSize is tokens per block.
	KVBlockSize int
}

// Metrics is an engine's exported metrics: every series it exposes, by family. A family's
// name and labels are the engine's own. A simulator reads the ones it routes on and
// copies the rest to its report untouched.
type Metrics struct {
	At       Time
	Families []Family
}

// Family is one metric family: a name, its kind and its samples.
type Family struct {
	Name    string
	Kind    MetricKind
	Help    string
	Samples []Sample
}

// MetricKind is a Prometheus metric type.
type MetricKind string

const (
	Gauge     MetricKind = "gauge"
	Counter   MetricKind = "counter"
	Histogram MetricKind = "histogram"
	Info      MetricKind = "info"
)

// Sample is one labelled value. A histogram family carries one sample per bucket with an
// "le" label, then a _sum and a _count sample, as Prometheus exposes them.
type Sample struct {
	Name   string
	Labels map[string]string
	Value  float64
}

// Family returns the named family and whether it exists.
func (m Metrics) Family(name string) (Family, bool) {
	for _, f := range m.Families {
		if f.Name == name {
			return f, true
		}
	}
	return Family{}, false
}

// Value returns the value of a family that has exactly one sample, the shape of the
// gauges and counters a router reads.
func (m Metrics) Value(name string) (float64, error) {
	f, ok := m.Family(name)
	if !ok {
		return 0, fmt.Errorf("engine: no metric family %q", name)
	}
	if len(f.Samples) != 1 {
		return 0, fmt.Errorf("engine: metric family %q has %d samples, not one", name, len(f.Samples))
	}
	return f.Samples[0].Value, nil
}

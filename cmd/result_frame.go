package cmd

// The experimental frame (design §8.6, DEV-7): what a run fed in, when it measured, and what
// it measured, recorded in every Result. This file is the one place the frame is built, from
// the run's resolved options, so that moving it onto the frame field of the contracts Result
// (blis-substrate contracts/spec/result) is a change of where it is written, not of what it
// holds.
//
// blis-schemas v0.10.0's Result has no frame field and is decoded strictly, so the frame is
// written under engine.frame: engine is the one open section every Result carries, and it
// already holds the simulator's whole-run records (engine.cluster.whole_run, .agentic). No
// consumer iterates engine's keys (blis-config-search reads engine.cluster and
// engine.instances).

import (
	"math"
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/spec/result"
)

// The generator kinds a frame names: how the run's requests were produced.
const (
	generatorWorkloadSpec  = "workload_spec"  // run --workload-spec
	generatorDistribution  = "distribution"   // run --workload distribution: the token-length flags
	generatorPreset        = "preset"         // run --workload <catalog preset>
	generatorConcurrency   = "concurrency"    // run --concurrency: closed-loop virtual users
	generatorTrace         = "trace"          // replay --trace-header/--trace-data
	generatorAgenticReplay = "agentic_replay" // replay --agentic-scenario
)

// frameRandomGenerator names the random-number generator every draw comes from, with its
// derivation, so that a change of generator, which changes every draw, changes the frame
// (§8.4). The derivation is sim.PartitionedRNG's (sim/rng.go): math/rand's source per named
// subsystem, seeded with the seed for the workload subsystem and with the seed XOR the
// FNV-1a 64 hash of the subsystem's name for every other one.
const frameRandomGenerator = "math/rand rngSource; sim.PartitionedRNG: workload subsystem seeded by the seed, " +
	"every other subsystem by seed XOR fnv1a64(name)"

// frameGenerator is what a command tells the frame about its request source.
type frameGenerator struct {
	// Kind is one of the generator* constants.
	Kind string
	// Source identifies the workload: a workload spec's or a trace data file's sha256, a
	// preset's name, or empty for the distribution and concurrency generators, whose
	// parameters are the run's flags (input.run and overrides).
	Source string
	// Seed is the seed the workload's draws used. A workload spec run without --seed uses the
	// spec's own seed.
	Seed int64
	// SessionMode is a trace replay's --session-mode (fixed, closed-loop or
	// fixed-accumulate); empty for a run.
	SessionMode string
	// Warmup names the warm-up rule, "" for none: an agentic replay's warm-up mode.
	Warmup string
}

// resultFrame is the experimental frame, as recorded.
type resultFrame struct {
	Generator    frameGeneratorRecord `json:"generator"`
	InitialState frameInitialState    `json:"initial_state"`
	Phases       framePhases          `json:"phases"`
	StopRule     frameStopRule        `json:"stop_rule"`
	Acceptance   frameAcceptance      `json:"acceptance"`
	// Metrics are the identifiers of the metric definitions the run reports: the keys of the
	// Result's metric_defs, sorted.
	Metrics []string `json:"metrics"`
}

type frameGeneratorRecord struct {
	Kind        string `json:"kind"`
	Source      string `json:"source,omitempty"`
	Seed        int64  `json:"seed"`
	SessionMode string `json:"session_mode,omitempty"`
	// Random names the random-number generator and how its streams are seeded.
	Random string `json:"random"`
}

type frameInitialState struct {
	// Caches is how every engine's KV cache starts. BLIS starts every engine empty; an
	// agentic replay's warm-up fills it during the run (phases.warmup).
	Caches string `json:"caches"`
}

type framePhases struct {
	// Warmup is the rule for the warm-up phase the measured window excludes.
	Warmup string `json:"warmup"`
	// MeasuredStart and MeasuredStop are the rules that bound the measured window
	// (result.Window.Start and .Stop).
	MeasuredStart string `json:"measured_start"`
	MeasuredStop  string `json:"measured_stop"`
	// Drain is what happens to requests still in flight when the run stops.
	Drain string `json:"drain"`
}

type frameStopRule struct {
	// Rule is request_count, horizon, both, or exhaustion (neither: the run ends when no
	// event remains).
	Rule string `json:"rule"`
	// NumRequests is the request cap the generator was given, 0 for none.
	NumRequests int `json:"num_requests,omitempty"`
	// HorizonUs is the horizon the run stopped at, 0 for none.
	HorizonUs int64 `json:"horizon_us,omitempty"`
}

type frameAcceptance struct {
	// Conditions are the conditions under which the run produces figures. A run that fails
	// one produces no Result (emitRunResult refuses it).
	Conditions []string `json:"conditions"`
	// UnfinishedAtStop counts the requests that had not finished when the run stopped:
	// queued, running, or held by the router.
	UnfinishedAtStop int `json:"unfinished_at_stop"`
}

// The phase and stop rules, as written in the frame.
const (
	frameNoWarmup         = "none"
	frameCachesEmpty      = "empty"
	frameDrainNone        = "none: a request unfinished at the horizon is counted unfinished, not drained"
	frameDrainToEnd       = "every request runs to its end: the run stops when no event remains"
	frameStopRequestCount = "request_count"
	frameStopHorizon      = "horizon"
	frameStopBoth         = "both"
	frameStopExhaustion   = "exhaustion"
	// frameConditionSimulatedTime is the one acceptance condition emitRunResult enforces.
	frameConditionSimulatedTime = "simulated time advanced: at least one request arrived, or a horizon bounds the run"
)

// buildFrame is the run's experimental frame. numRequestsCap is the request cap the generator
// was given (0 for none); horizonUs the horizon the run stopped at (math.MaxInt64 or 0 for
// none).
func buildFrame(g frameGenerator, r *result.Result, numRequestsCap int, horizonUs int64) resultFrame {
	bounded := horizonUs > 0 && horizonUs < math.MaxInt64
	stop := frameStopRule{NumRequests: max(numRequestsCap, 0)}
	switch {
	case numRequestsCap > 0 && bounded:
		stop.Rule = frameStopBoth
	case numRequestsCap > 0:
		stop.Rule = frameStopRequestCount
	case bounded:
		stop.Rule = frameStopHorizon
	default:
		stop.Rule = frameStopExhaustion
	}
	drain := frameDrainToEnd
	if bounded {
		stop.HorizonUs = horizonUs
		drain = frameDrainNone
	}
	warmup := g.Warmup
	if warmup == "" {
		warmup = frameNoWarmup
	}
	metrics := make([]string, 0, len(r.MetricDefs))
	for k := range r.MetricDefs {
		metrics = append(metrics, k)
	}
	sort.Strings(metrics)
	q := r.Requests
	return resultFrame{
		Generator: frameGeneratorRecord{Kind: g.Kind, Source: g.Source, Seed: g.Seed,
			SessionMode: g.SessionMode, Random: frameRandomGenerator},
		InitialState: frameInitialState{Caches: frameCachesEmpty},
		Phases:       framePhases{Warmup: warmup, MeasuredStart: r.Window.Start, MeasuredStop: r.Window.Stop, Drain: drain},
		StopRule:     stop,
		Acceptance: frameAcceptance{Conditions: []string{frameConditionSimulatedTime},
			UnfinishedAtStop: q.StillQueued + q.StillRunning + q.GatewayQueueDepth},
		Metrics: metrics,
	}
}

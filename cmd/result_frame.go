package cmd

// The experimental frame (design §8.6, DEV-7): what a run fed in, when it measured, and what
// it measured, recorded in every Result's frame. This file is the one place the frame is
// built, from the run's resolved options.

import (
	"math"
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/spec/result"

	"github.com/inference-sim/inference-sim/sim"
)

// The generator kinds a frame names: how the run's requests were produced. They are written
// nowhere in the frame itself (contracts' Generator states a workload binding and a load); a
// command uses them to choose its binding.
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
// (§8.4, DEV-6).
const frameRandomGenerator = sim.StreamGenerator

// frameGenerator is what a command tells the frame about its request source.
type frameGenerator struct {
	// Kind is one of the generator* constants.
	Kind string
	// Source identifies the workload: a workload spec's or a trace data file's sha256, a
	// preset's name, or empty for the distribution and concurrency generators.
	Source string
	// Seed is the seed the workload's draws used. A workload spec run without --seed uses the
	// spec's own seed.
	Seed int64
	// SessionMode is a trace replay's --session-mode; empty for a run.
	SessionMode string
	// Warmup names an agentic replay's warm-up mode, "" for none.
	Warmup string
}

// frameUnfinishedThreshold is the acceptance condition every run states: the fraction of
// generated requests unfinished at the horizon may be at most 1, which every run meets. It is
// recorded so a reader sees the observed fraction; a frame that bounds it more tightly is the
// workload's to state.
const frameUnfinishedThreshold = 1.0

// buildFrame fills r.Frame from the generator, the request cap the generator was given
// (0 for none) and the horizon the run stopped at (math.MaxInt64 or 0 for none). It runs
// after the metrics and the window are set, since frame.metrics lists the figures and the
// measured phase names the window's rules.
func buildFrame(g frameGenerator, r *result.Result, rules windowRules, numRequestsCap int, horizonUs int64) {
	f := &r.Frame
	f.Kind = result.FrameTerminating
	f.Generator.Seed = g.Seed
	f.Generator.RNG = frameRandomGenerator
	f.InitialState = result.InitialState{Caches: result.CachesEmpty}

	bounded := horizonUs > 0 && horizonUs < math.MaxInt64
	f.Stop = result.StopRule{Requests: max(numRequestsCap, 0)}
	if bounded {
		f.Stop.HorizonS = horizonSeconds(horizonUs)
	}
	f.Phases.Warmup = result.WarmupPhase{Rule: result.WarmupNone}
	if g.Warmup != "" {
		f.Phases.Warmup.Rule = result.WarmupMarked
	}
	f.Phases.Measured = result.MeasuredPhase{Start: rules.start, Stop: rules.stop}
	f.Phases.Drain = result.DrainPhase{Rule: result.DrainUntilComplete}
	if bounded {
		f.Phases.Drain.Rule = result.DrainNone
	}

	q := r.Requests
	observed := 0.0
	if q.Generated > 0 {
		observed = float64(q.Unfinished()) / float64(q.Generated)
	}
	f.Acceptance = []result.AcceptanceCondition{{Condition: result.ConditionUnfinishedAtMost,
		Threshold: frameUnfinishedThreshold, Observed: observed, Met: observed <= frameUnfinishedThreshold}}

	f.Metrics = make([]string, 0, len(r.Metrics))
	for k := range r.Metrics {
		f.Metrics = append(f.Metrics, k)
	}
	sort.Strings(f.Metrics)
}

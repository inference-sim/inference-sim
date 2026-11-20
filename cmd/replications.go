package cmd

// An interval on request (DEV-11, COV-10, design §8.8). One replication is the default of every
// run; --replications N (N at least five) asks for an interval. The replication driver runs the
// same command N times as child processes, each with its own seed drawn from a named stream of
// the run's --seed (DEV-6), reads each child's Result, and writes one Result whose figures are
// the means over the replications with a Student t interval. It then continues one replication
// at a time until every figure's interval is no wider than --interval-width of the figure, or
// --max-replications ran. The interval covers the randomness of the workload only.
//
// The driver is a process boundary rather than a loop in the run command because the run
// command resolves its configuration into package state and is not re-entrant; a child is the
// same code path as a single run, so each replication is byte-identical to the run its seed
// would give alone (INV-6).

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strings"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"

	sim "github.com/inference-sim/inference-sim/sim"
)

// The interval request flags of `run`.
var (
	replications    int     // --replications: 1 (the default) or at least five
	maxReplications int     // --max-replications: the stopping rule's limit
	intervalWidth   float64 // --interval-width: the stopping rule's relative width
)

const (
	defaultMaxReplications = 30
	defaultIntervalWidth   = 0.10
	// intervalLevel is the confidence level of every interval the driver reports.
	intervalLevel = 0.95
)

func registerIntervalFlags(cmd *cobra.Command) {
	cmd.Flags().IntVar(&replications, "replications", 1,
		"Replications (runs that differ only in their seed): 1 reports point estimates that state they have no interval; "+
			"at least 5 requests an interval on every figure (design §8.8)")
	cmd.Flags().IntVar(&maxReplications, "max-replications", defaultMaxReplications,
		"With --replications >= 5: the most replications run while an interval is wider than --interval-width")
	cmd.Flags().Float64Var(&intervalWidth, "interval-width", defaultIntervalWidth,
		"With --replications >= 5: replications continue until every figure's 95% interval is no wider than this fraction of the figure")
}

// replicationFlags are the flags the driver consumes or must not pass to a child.
var replicationFlags = map[string]bool{
	"replications": true, "max-replications": true, "interval-width": true,
	// Output flags: the driver writes the one document; a child writes JSON to a pipe.
	"format": true, "output": true, "view": true,
}

// replicationRefused are the flags that describe one run and have no meaning across replications.
var replicationRefused = map[string]string{
	"from-result":       "reproduces one run",
	"trace-output":      "exports the trace of one run",
	"saturation-report": "writes the verdict of one run",
}

// replicationSeed is the scenario seed of replication k (from 0): the first draw of the stream
// "replication|seed|k" under the run's seed (DEV-6).
func replicationSeed(base int64, k int) int64 {
	return sim.NewStream(base, sim.StreamName{Component: "replication", Purpose: "seed", Entity: fmt.Sprintf("r%d", k)}).Int63()
}

// checkReplicationFlags refuses an interval request that cannot be honoured, before any run.
func checkReplicationFlags(cmd *cobra.Command) {
	if replications != 1 && replications < result.MinIntervalReplications {
		logrus.Fatalf("--replications %d: an interval is computed from at least %d replications; use 1 for a point estimate or %d or more for an interval (design §8.8)",
			replications, result.MinIntervalReplications, result.MinIntervalReplications)
	}
	if replications == 1 {
		for _, f := range []string{"max-replications", "interval-width"} {
			if cmd.Flags().Changed(f) {
				logrus.Fatalf("--%s needs --replications of %d or more: one replication reports a point estimate with no interval", f, result.MinIntervalReplications)
			}
		}
		return
	}
	for f, why := range replicationRefused {
		if cmd.Flags().Changed(f) {
			logrus.Fatalf("--%s cannot be combined with --replications: it %s", f, why)
		}
	}
	if maxReplications < replications {
		logrus.Fatalf("--max-replications %d is below --replications %d", maxReplications, replications)
	}
	if !(intervalWidth > 0) || math.IsInf(intervalWidth, 0) {
		logrus.Fatalf("--interval-width %v: a positive fraction of the figure", intervalWidth)
	}
}

// childArgs is the command line of one replication: the run's own flags, the replication's
// seed, and a JSON document on stdout.
func childArgs(fs *pflag.FlagSet, seedValue int64) []string {
	args := []string{"run", "--format", string(result.FormatJSON), fmt.Sprintf("--seed=%d", seedValue)}
	fs.Visit(func(f *pflag.Flag) {
		if replicationFlags[f.Name] || f.Name == "seed" {
			return
		}
		if sv, ok := f.Value.(pflag.SliceValue); ok {
			for _, e := range sv.GetSlice() {
				args = append(args, fmt.Sprintf("--%s=%s", f.Name, e))
			}
			return
		}
		args = append(args, fmt.Sprintf("--%s=%s", f.Name, f.Value.String()))
	})
	return args
}

// replicationCommand builds the process of one replication. A test replaces it to run the
// test binary's own CLI leg.
var replicationCommand = func(args []string) (*exec.Cmd, error) {
	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("locating the blis executable to run a replication: %w", err)
	}
	return exec.Command(exe, args...), nil
}

// runReplication runs one replication and reads its Result.
func runReplication(args []string) (*result.Result, error) {
	c, err := replicationCommand(args)
	if err != nil {
		return nil, err
	}
	var out, errBuf bytes.Buffer
	c.Stdout, c.Stderr = &out, &errBuf
	if err := c.Run(); err != nil {
		return nil, fmt.Errorf("%w\n%s", err, strings.TrimSpace(errBuf.String()))
	}
	var r result.Result
	if err := result.Unmarshal(out.Bytes(), &r); err != nil {
		return nil, fmt.Errorf("reading the replication's Result: %w", err)
	}
	return &r, nil
}

// runReplications is `run` with --replications of five or more: it drives the replications,
// writes the one Result, and returns.
func runReplications(cmd *cobra.Command) {
	snapshotResultFlags("run", cmd)
	var docs []*result.Result
	var seeds []int64
	for {
		k := len(docs)
		s := replicationSeed(seed, k)
		r, err := runReplication(childArgs(cmd.Flags(), s))
		if err != nil {
			logrus.Fatalf("replication %d of the interval (seed %d): %v", k+1, s, err)
		}
		docs, seeds = append(docs, r), append(seeds, s)
		if k+1 < replications {
			continue
		}
		agg, stopped, err := aggregateReplications(docs, seeds, intervalPlan{
			Min: replications, Max: maxReplications, Width: intervalWidth,
		})
		if err != nil {
			logrus.Fatalf("%v", err)
		}
		if !stopped {
			continue
		}
		agg.Frame.Generator.Seed = seed
		agg.Input.Options = resultInputOptions
		agg.Name = fmt.Sprintf("%s-n%d", resultName(agg), len(docs))
		emitResult(agg, nil)
		return
	}
}

// intervalPlan is the stopping rule.
type intervalPlan struct {
	Min, Max int
	Width    float64
}

// aggregateReplications is the Result for the replications so far and whether the stopping rule
// stops there. Replications that disagree on which figures they report are refused: they are
// not replications of one measurement.
func aggregateReplications(docs []*result.Result, seeds []int64, plan intervalPlan) (*result.Result, bool, error) {
	first := *docs[0]
	n := len(docs)
	figures := make(map[string]result.Figure, len(first.Metrics))
	widest := 0.0
	for i, d := range docs {
		if len(d.Metrics) != len(first.Metrics) {
			return nil, false, fmt.Errorf("replication %d (seed %d) reports %d figures and replication 1 reports %d: the replications are not of one measurement", i+1, seeds[i], len(d.Metrics), len(first.Metrics))
		}
	}
	for id, f0 := range first.Metrics {
		vals := make([]float64, n)
		samples := 0
		for i, d := range docs {
			f, ok := d.Metrics[id]
			if !ok {
				return nil, false, fmt.Errorf("replication %d (seed %d) reports no %s although replication 1 does: the replications are not of one measurement", i+1, seeds[i], id)
			}
			vals[i] = f.Value
			samples += f.Samples
		}
		mean, low, high := tInterval(vals, intervalLevel)
		f0.Value, f0.Samples = mean, samples
		f0.Interval = &result.FigureInterval{Low: low, High: high}
		figures[id] = f0
		widest = math.Max(widest, relativeWidth(mean, low, high))
	}
	stopWidth := widest <= plan.Width
	stopped := stopWidth || n >= plan.Max
	first.Metrics = figures
	first.Estimate = result.Estimate{
		Replications: n,
		Interval: &result.IntervalEstimate{
			Level: intervalLevel, Method: result.IntervalMethodStudentT, Covers: result.IntervalCovers,
			Seeds: append([]int64(nil), seeds...), MinReplications: plan.Min, MaxReplications: plan.Max,
			TargetRelativeWidth: plan.Width, RelativeWidthReached: widest,
			StoppedBy: map[bool]result.IntervalStop{true: result.StoppedByWidth, false: result.StoppedByLimit}[stopWidth],
		},
	}
	return &first, stopped, nil
}

// relativeWidth is the interval's width as a fraction of the mean. A figure of zero has zero
// width (every figure is non-negative, so a zero mean means every replication reported zero).
func relativeWidth(mean, low, high float64) float64 {
	if mean == 0 {
		return 0
	}
	return (high - low) / math.Abs(mean)
}

// tInterval is the mean of vals and the two-sided Student t interval of the mean at level.
func tInterval(vals []float64, level float64) (mean, low, high float64) {
	n := float64(len(vals))
	mean = stat.Mean(vals, nil)
	if len(vals) < 2 {
		return mean, mean, mean
	}
	sd := stat.StdDev(vals, nil)
	t := distuv.StudentsT{Mu: 0, Sigma: 1, Nu: n - 1}.Quantile(0.5 + level/2)
	half := t * sd / math.Sqrt(n)
	return mean, mean - half, mean + half
}

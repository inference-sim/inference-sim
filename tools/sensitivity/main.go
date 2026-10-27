// Command sensitivity measures how much each ASSUMED engine setting moves the score.
//
// # Why this exists
//
// NVIDIA's snapshot publishes a deployment's framework, precision, serving mode,
// speculation method and five parallelism widths. It publishes NO engine settings. They are
// taken from the measured runs' own engine logs where those exist (-engine-settings), and
// from vLLM's own defaults where a run passed nothing -- but a resolved default is not a
// measurement, and some points have no log at all.
//
// Those defaults were justified for the earlier step-time comparison on the grounds that a
// constant cancels in a ratio. That is true of a ratio of two StepTime calls and FALSE once
// a scheduler is in the loop: max_num_seqs and the token budget bound the resident batch,
// and the resident batch is what sets the shape of the concurrency response. A justification
// that held in one regime was carried into another where it does not hold.
//
// The generalisable defence is not a better guess. It is to MEASURE how much each assumed
// value moves the answer, so a reader knows which conclusions rest on a guess and by how
// much. A setting that moves the score by a tenth of a point is a footnote; one that moves
// it by several points bounds what the comparison can claim.
//
// This command reports a one-at-a-time sweep: hold everything else at the scored
// configuration, vary one setting, report the resulting shape MAPE. It does not choose a
// value, and choosing the best one would be fitting to the evaluation set.
//
// # The draw
//
// It also reports how much the score moves with the WORKLOAD DRAW. Every point replays the
// benchmark's own seed-0 length vectors (testdata/inferencex-workloads/seed0.json); this
// re-scores with the vectors numpy draws at seeds 1 to 4 from the same intervals, generated
// by the same script. The spread is the draw-sensitivity an inexact workload would carry --
// what scoring against BLIS's own Go draw used to add -- and what exact reproduction removes.
//
// Usage:
//
//	go run ./tools/sensitivity -framework vllm
package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/vendored"
	"github.com/inference-sim/inference-sim/tools/harness"
)

// The kernel adapter's defaults are this repository's vendored catalog, registry and pinned
// kernel (sim/vendored).
func init() { kernelmodel.SetVendoredRoot(vendored.Root()) }

func main() {
	corpusPath := flag.String("corpus",
		kernelmodel.MeasurementPath("aisimulate_e2e.json"), "")
	scenarios := flag.String("scenarios",
		kernelmodel.DefaultScenarios(), "")
	catalog := flag.String("catalog", kernelmodel.DefaultCatalog(), "")
	registry := flag.String("registry", kernelmodel.DefaultRegistry(), "")
	framework := flag.String("framework", "vllm", "")
	settings := flag.String("engine-settings",
		kernelmodel.MeasurementPath("inferencex_engine_settings.json"),
		"the settings each measured run was launched with; the baseline is the scored configuration")
	draws := flag.Bool("draws", true, "also re-score with the numpy seed-1..4 length fixtures")
	monotoneOnly := flag.Bool("monotone", true, "score only sweeps with a monotone measurement")
	blis := flag.String("blis", "", "path of the blis binary every point is simulated with (`go build -o blis main.go`); required")
	gaps := flag.String("gaps", "", "write the coverage-gap report here (default stderr; never stdout)")
	flag.Parse()
	if err := harness.RequireBlis(*blis); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := kernelmodel.RequireCorpora(map[string]string{"corpus": *corpusPath, "engine-settings": *settings}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	c, err := harness.LoadCorpus(*corpusPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	eset, err := harness.LoadEngineSettings(*settings)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	base := harness.Config{
		Repos: kernelmodel.Repos{
			Scenarios: *scenarios, Catalog: *catalog, Registry: *registry,
		},
		Blis: *blis, Seed: 42, EngineSettings: eset,
	}

	// score returns the TPOT and TTFT shape MAPE (anchor excluded, each side normalised to its
	// own lowest concurrency) and the number of TPOT points.
	score := func(cfg harness.Config) (float64, float64, int) {
		var errs, ttftErrs []float64
		for _, sw := range c.Sweeps {
			if sw.Framework != *framework || len(sw.Points) < 2 {
				continue
			}
			if *monotoneOnly && !sw.MonotoneMeasurement() {
				continue
			}
			var anchor, anchorTTFT float64
			ok := true
			var local, localTTFT []float64
			for i, p := range sw.Points {
				obs, err := harness.Run(sw, p.Concurrency, cfg)
				if err == nil && obs.MeanTPOTUs <= 0 {
					err = fmt.Errorf("no time per output token observed")
				}
				if err != nil {
					// Reported, not swallowed: a sweep dropped from one configuration but not
					// another changes the population a delta is taken over.
					fmt.Fprintf(os.Stderr, "sensitivity: %s %s c=%d: %v\n", sw.Scenario, sw.Label, p.Concurrency, err)
					ok = false
					break
				}
				if i == 0 {
					anchor, anchorTTFT = obs.MeanTPOTUs, obs.MeanTTFTUs
					continue // the anchor's shape error is zero by construction
				}
				local = append(local, math.Abs(obs.MeanTPOTUs/anchor/p.MeasuredRelative-1)*100)
				if p.MeasuredTTFTRelative > 0 {
					localTTFT = append(localTTFT, math.Abs(obs.MeanTTFTUs/anchorTTFT/p.MeasuredTTFTRelative-1)*100)
				}
			}
			if ok {
				errs = append(errs, local...)
				ttftErrs = append(ttftErrs, localTTFT...)
			}
		}
		return harness.MAPE2(errs), harness.MAPE2(ttftErrs), len(errs)
	}

	fmt.Printf("Sensitivity of the TPOT shape MAPE to settings the snapshot does not publish\n")
	fmt.Printf("framework=%s monotone-only=%v; anchor excluded; every scoreable sweep (all tiers)\n\n",
		*framework, *monotoneOnly)

	baseMAPE, baseTTFT, n := score(base)
	fmt.Printf("%-34s %6s %8s %8s\n", "setting", "n", "MAPE", "delta")
	fmt.Printf("%-34s %6d %7.2f%% %8s\n", "scored configuration (baseline)", n, baseMAPE, "-")

	type variant struct {
		label string
		apply func(*harness.Config)
	}
	variants := []variant{
		{"max_num_seqs x2", func(c *harness.Config) { c.MaxNumSeqsScale = 2 }},
		{"max_num_seqs x4", func(c *harness.Config) { c.MaxNumSeqsScale = 4 }},
		{"max_num_seqs /2", func(c *harness.Config) { c.MaxNumSeqsScale = 0.5 }},
		{"token budget x2", func(c *harness.Config) { c.TokenBudgetScale = 2 }},
		{"token budget /2", func(c *harness.Config) { c.TokenBudgetScale = 0.5 }},
	}
	populationsDiffer := false
	for _, v := range variants {
		cfg := base
		v.apply(&cfg)
		m, _, vn := score(cfg)
		fmt.Printf("%-34s %6d %7.2f%% %+7.2f\n", v.label, vn, m, m-baseMAPE)
		if vn != n {
			fmt.Fprintf(os.Stderr, "sensitivity: %q scored %d points against the baseline's %d, so "+
				"its delta compares different sweeps\n", v.label, vn, n)
			populationsDiffer = true
		}
	}

	// The former "admission: seqs only (no KV)" row set a KV budget large enough never to
	// bind. Points are now simulated by `blis replay`, which sizes the KV pool from the kernel
	// and offers no override, so that row cannot be expressed and is reported, not faked
	// (#1902).
	fmt.Printf("%-34s %6s %8s %8s\n", "admission: seqs only (no KV)", "-", "n/a", "n/a")
	fmt.Printf("  (not expressible: blis replay sizes KV from the kernel and has no override)\n")

	if *draws {
		fmt.Printf("\nThe workload draw. Seed 0 is the benchmark's own (the scored baseline above);\n")
		fmt.Printf("seeds 1-4 are other numpy draws from the same intervals, same request count.\n\n")
		fmt.Printf("%-10s %6s %10s %8s %10s %8s\n", "numpy seed", "n", "TPOT shape", "delta", "TTFT shape", "delta")
		fmt.Printf("%-10d %6d %9.2f%% %8s %9.2f%% %8s\n", 0, n, baseMAPE, "-", baseTTFT, "-")
		tp, tt := []float64{baseMAPE}, []float64{baseTTFT}
		fixtures, err := harness.LoadDrawFixtures()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		for _, ws := range fixtures {
			seed := ws.Seed
			cfg := base
			cfg.Workloads = ws
			m, mt, vn := score(cfg)
			fmt.Printf("%-10d %6d %9.2f%% %+8.2f %9.2f%% %+8.2f\n", seed, vn, m, m-baseMAPE, mt, mt-baseTTFT)
			if vn != n {
				fmt.Fprintf(os.Stderr, "sensitivity: seed %d scored %d points against the baseline's %d\n", seed, vn, n)
				populationsDiffer = true
			}
			tp, tt = append(tp, m), append(tt, mt)
		}
		fmt.Printf("%-10s %6s %9.2f%% %8s %9.2f%%\n", "range", "", spread(tp), "", spread(tt))
		fmt.Printf("\nThe range is how far a figure scored against a draw other than the benchmark's\n")
		fmt.Printf("own can move for that reason alone.\n")
	}

	fmt.Printf("\nA setting whose delta is small is a footnote. One whose delta is large bounds\n")
	fmt.Printf("what this comparison can claim, because its value is a guess: the snapshot\n")
	fmt.Printf("states it nowhere. No value is chosen here -- choosing the best would be\n")
	fmt.Printf("fitting to the evaluation set.\n")
	if err := harness.AssessCoverage(c, base).WriteGaps(*gaps); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if populationsDiffer {
		os.Exit(1)
	}
}

// spread is max - min.
func spread(xs []float64) float64 {
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, x := range xs {
		lo, hi = math.Min(lo, x), math.Max(hi, x)
	}
	return hi - lo
}

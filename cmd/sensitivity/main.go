// Command sensitivity measures how much each ASSUMED engine setting moves the score.
//
// # Why this exists
//
// NVIDIA's snapshot publishes a deployment's framework, precision, serving mode,
// speculation method and five parallelism widths. It publishes NO engine settings. The
// scenario files this comparison reads therefore default max_num_seqs, the token budget,
// block size, memory utilization and the cudagraph mode.
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
// This command reports a one-at-a-time sweep: hold everything else at the scenario's value,
// vary one setting, report the resulting shape MAPE. It does not choose a value, and
// choosing the best one would be fitting to the evaluation set.
//
// Usage:
//
//	go run ./cmd/sensitivity -framework vllm
package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/kernelmodel/harness"
)

func main() {
	corpusPath := flag.String("corpus",
		kernelmodel.MeasurementPath("aisimulate_e2e.json"), "")
	scenarios := flag.String("scenarios",
		kernelmodel.DefaultScenarios(), "")
	catalog := flag.String("catalog", kernelmodel.DefaultCatalog(), "")
	registry := flag.String("registry", kernelmodel.DefaultRegistry(), "")
	framework := flag.String("framework", "vllm", "")
	sessions := flag.Int("sessions", 40, "")
	monotoneOnly := flag.Bool("monotone", true, "score only sweeps with a monotone measurement")
	blis := flag.String("blis", "", "path of the blis binary every point is simulated with (`go build -o blis main.go`); required")
	gaps := flag.String("gaps", "", "write the coverage-gap report here (default stderr; never stdout)")
	flag.Parse()
	if err := harness.RequireBlis(*blis); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := kernelmodel.RequireCorpora(map[string]string{"corpus": *corpusPath}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	c, err := harness.LoadCorpus(*corpusPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	base := harness.Config{
		Repos: kernelmodel.Repos{
			Scenarios: *scenarios, Catalog: *catalog, Registry: *registry,
		},
		Blis: *blis, SessionsPerPoint: *sessions, Seed: 42,
	}

	score := func(cfg harness.Config) (float64, int) {
		var errs []float64
		for _, sw := range c.Sweeps {
			if sw.Framework != *framework || len(sw.Points) < 2 {
				continue
			}
			if *monotoneOnly && !sw.MonotoneMeasurement() {
				continue
			}
			var anchor float64
			ok := true
			var local []float64
			for i, p := range sw.Points {
				obs, err := harness.Run(sw, p.Concurrency, cfg)
				if err == nil && obs.MeanITLUs <= 0 {
					err = fmt.Errorf("no inter-token latency observed")
				}
				if err != nil {
					// Reported, not swallowed: a sweep dropped from one configuration but not
					// another changes the population a delta is taken over.
					fmt.Fprintf(os.Stderr, "sensitivity: %s %s c=%d: %v\n", sw.Scenario, sw.Label, p.Concurrency, err)
					ok = false
					break
				}
				if i == 0 {
					anchor = obs.MeanITLUs
				}
				local = append(local, math.Abs(obs.MeanITLUs/anchor/p.MeasuredRelative-1)*100)
			}
			if ok {
				errs = append(errs, local...)
			}
		}
		return harness.MAPE2(errs), len(errs)
	}

	fmt.Printf("Sensitivity of the shape MAPE to settings the snapshot does not publish\n")
	fmt.Printf("framework=%s sessions=%d monotone-only=%v\n\n", *framework, *sessions, *monotoneOnly)

	baseMAPE, n := score(base)
	fmt.Printf("%-34s %6s %8s %8s\n", "setting", "n", "MAPE", "delta")
	fmt.Printf("%-34s %6d %7.2f%% %8s\n", "scenario values (baseline)", n, baseMAPE, "-")

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
		m, vn := score(cfg)
		fmt.Printf("%-34s %6d %7.2f%% %+7.2f\n", v.label, vn, m, m-baseMAPE)
		if vn != n {
			fmt.Fprintf(os.Stderr, "sensitivity: %q scored %d points against the baseline's %d, so "+
				"its delta compares different sweeps\n", v.label, vn, n)
			populationsDiffer = true
		}
	}

	// The former "admission: seqs only (no KV)" row set a KV budget large enough never to
	// bind. Points are now simulated by `blis run`, which sizes the KV pool from the kernel
	// and offers no override, so that row cannot be expressed and is reported, not faked
	// (#1902).
	fmt.Printf("%-34s %6s %8s %8s\n", "admission: seqs only (no KV)", "-", "n/a", "n/a")
	fmt.Printf("  (not expressible: blis run sizes KV from the kernel and has no override)\n")

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

// Command signedprobe emits one CSV row per direct-corpus point: the measured latency, the
// kernel's prediction, and the keys a regime question needs -- model, chip, concurrency,
// workload shape and parallelism. It answers "is the error one-sided, and where" without
// an aggregate hiding the answer.
//
// Temporary: this exists to characterise a residual, not to score anything. metricscore
// remains the scoring command.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strconv"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/kernelmodel/harness"
)

func main() {
	corpusPath := flag.String("corpus", "", "")
	scenarios := flag.String("scenarios", "", "")
	catalog := flag.String("catalog", kernelmodel.DefaultCatalog(), "")
	registry := flag.String("registry", kernelmodel.DefaultRegistry(), "")
	absolutes := flag.String("absolutes", "", "")
	settings := flag.String("engine-settings", "", "")
	tier := flag.String("config-tier", "",
		"restrict to \"measured\" sweeps (the run's own command line) or \"resolved\" "+
			"(vLLM defaults). Same test metricscore applies, so the two agree on membership.")
	blis := flag.String("blis", "", "path of the blis binary every point is simulated with (`go build -o blis main.go`); required")
	gaps := flag.String("gaps", "", "write the coverage-gap report here (default stderr; never stdout)")
	flag.Parse()
	if err := harness.RequireBlis(*blis); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	c, err := harness.LoadCorpus(*corpusPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	abs, err := harness.LoadAbsolutes(*absolutes)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	eset, err := harness.LoadEngineSettings(*settings)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	cfg := harness.Config{
		Repos: kernelmodel.Repos{
			Scenarios: *scenarios, Catalog: *catalog, Registry: *registry,
		},
		Blis:           *blis,
		EngineSettings: eset,
		Seed:           42,
	}

	w := csv.NewWriter(os.Stdout)
	write := func(row []string) {
		if err := w.Write(row); err != nil {
			fmt.Fprintf(os.Stderr, "signedprobe: writing stdout: %v\n", err)
			os.Exit(1)
		}
	}
	write([]string{
		"scenario", "model", "gpu", "label", "concurrency",
		"meas_tpot_us", "pred_tpot_us", "meas_ttft_us", "pred_ttft_us",
	})
	var total, noAbs, noTPOT, noTTFT, runErr, emitted int
	for _, sw := range c.Sweeps {
		if *tier != "" {
			// The same membership test cmd/metricscore uses, so a slice here and a score
			// there cover the same sweeps.
			rec := eset.For(sw)
			measured := rec != nil && rec.At(sw.Points[0].Concurrency) != nil
			if (*tier == "measured") != measured {
				continue
			}
		}
		for _, p := range sw.Points {
			total++
			a := abs.For(sw)
			if a == nil {
				noAbs++
				continue
			}
			mt, ok := a.At(harness.MetricTPOT, p.Concurrency)
			if !ok {
				noTPOT++
				continue
			}
			// A missing TTFT absolute is skipped, not written as 0: a zero measured TTFT is a
			// value, and a consumer would score it.
			mf, ok := a.At(harness.MetricTTFT, p.Concurrency)
			if !ok {
				noTTFT++
				continue
			}
			o, err := harness.Run(sw, p.Concurrency, cfg)
			if err != nil {
				runErr++
				if runErr == 1 {
					fmt.Fprintf(os.Stderr, "first run error (%s c=%d): %v\n",
						sw.Scenario, p.Concurrency, err)
				}
				continue
			}
			if o.MeanITLUs <= 0 || o.MeanTTFTUs <= 0 {
				runErr++
				fmt.Fprintf(os.Stderr, "no latency observed (%s c=%d)\n", sw.Scenario, p.Concurrency)
				continue
			}
			emitted++
			write([]string{
				sw.Scenario, sw.Model, sw.GPU, sw.Label,
				strconv.Itoa(p.Concurrency),
				strconv.FormatFloat(mt, 'f', 3, 64),
				strconv.FormatFloat(o.MeanITLUs, 'f', 3, 64),
				strconv.FormatFloat(mf, 'f', 3, 64),
				strconv.FormatFloat(o.MeanTTFTUs, 'f', 3, 64),
			})
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		fmt.Fprintf(os.Stderr, "signedprobe: writing stdout: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr,
		"points=%d emitted=%d skipped: no_absolutes=%d no_tpot=%d no_ttft=%d run_error=%d\n",
		total, emitted, noAbs, noTPOT, noTTFT, runErr)
	if err := harness.AssessCoverage(c, cfg).WriteGaps(*gaps); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	// A probe that emitted nothing printed only its header: that is a failure, not a result.
	if emitted == 0 {
		fmt.Fprintln(os.Stderr, "signedprobe: no point could be emitted")
		os.Exit(1)
	}
}

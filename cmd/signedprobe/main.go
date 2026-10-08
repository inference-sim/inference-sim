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
	hwConfig := flag.String("hardware-config", "hardware_config.json", "")
	defaults := flag.String("defaults", "defaults.yaml", "")
	flag.Parse()

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
		Admission: harness.AdmissionKernelKV,
		Backends: harness.BackendPaths{
			Catalog: *catalog, HWConfig: *hwConfig, Defaults: *defaults,
		},
		Estimator:      harness.EstimatorKernel,
		EngineSettings: eset,
		Seed:           42,
	}

	w := csv.NewWriter(os.Stdout)
	defer w.Flush()
	_ = w.Write([]string{
		"scenario", "model", "gpu", "label", "concurrency",
		"meas_tpot_us", "pred_tpot_us", "meas_ttft_us", "pred_ttft_us",
	})
	var total, noAbs, noTPOT, runErr, emitted int
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
			mf, _ := a.At(harness.MetricTTFT, p.Concurrency)
			o, err := harness.Run(sw, p.Concurrency, cfg)
			if err != nil {
				runErr++
				if runErr == 1 {
					fmt.Fprintf(os.Stderr, "first run error (%s c=%d): %v\n",
						sw.Scenario, p.Concurrency, err)
				}
				continue
			}
			emitted++
			_ = w.Write([]string{
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
	fmt.Fprintf(os.Stderr,
		"points=%d emitted=%d skipped: no_absolutes=%d no_tpot=%d run_error=%d\n",
		total, emitted, noAbs, noTPOT, runErr)
}

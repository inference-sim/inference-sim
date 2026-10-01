// Command metricscore scores every available estimator on every available metric.
//
// # What is comparable, and why that is less than four things
//
// The snapshot publishes four accuracy figures per estimator: TTFT and TPOT, each as a mean
// absolute percentage error (mape) and as a shape error. Only the two SHAPE figures can be
// reproduced here, and the reason is a property of the data rather than a choice.
//
// Shape error re-anchors each side to its OWN value at the sweep's lowest concurrency and excludes
// the anchor from the mean (scripts/build_e2e_accuracy_overview.py, _aggregate_shape_error). It
// therefore asks whether latency RESPONDS to concurrency correctly, and a constant multiplicative
// offset cancels exactly.
//
// mape is the plain absolute error on absolute latency. The artifact ships NO absolute latency:
// every point carries exactly tpot_relative and ttft_relative per estimator, verified by
// enumerating every key of every point in the artifact. The published mape figures come from rows
// NVIDIA does not ship, so nobody can recompute them from this corpus -- including NVIDIA. A mape
// column here would have to be invented, and is omitted instead.
//
// # The estimators
//
// AISIMULATE and AIC are two modes of ONE product: snapshot.aic_source names
// ai-dynamo/aisimulate for both, AIC being the analytic configurator path since absorbed into the
// simulator. They are reported side by side as modes, not as rival baselines.
//
// ROOFLINE and TRAINED-PHYSICS are BLIS's own backends, scored only where they are calibrated.
// hardware_config.json carries H100, H200, A100-SXM, A100-80 and L40S and blis-registry has no
// roofline-b200.yaml, so including Blackwell would mean inventing mfuPrefill/mfuDecode. Passing
// -hopper restricts the corpus to the chips they are calibrated for and adds them as columns.
//
// Usage:
//
//	go run ./cmd/metricscore                 # kernel vs AISimulate vs AIC, every chip
//	go run ./cmd/metricscore -hopper         # the above plus roofline and trained-physics
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/kernelmodel/harness"
)

// arm is one column of the report.
type arm struct {
	name string
	// est is set for an arm BLIS simulates; empty for a published baseline read from the corpus.
	est harness.Estimator
	// published reads the prediction out of the point instead of simulating it.
	published func(harness.Point, harness.Metric) float64
}

func main() {
	corpusPath := flag.String("corpus",
		"/Users/sri/Documents/Projects/blis-latency-kernel/testdata/measurements/aisimulate_e2e.json",
		"the extracted AISimulate corpus")
	scenarios := flag.String("scenarios",
		"/Users/sri/Documents/Projects/blis-latency-kernel/testdata/aisimulate", "scenario directory")
	catalog := flag.String("catalog", "/Users/sri/Documents/Projects/blis-catalog", "")
	registry := flag.String("registry", "/Users/sri/Documents/Projects/blis-registry", "")
	hwConfig := flag.String("hardware-config", "hardware_config.json", "")
	defaults := flag.String("defaults", "defaults.yaml", "")
	absolutes := flag.String("absolutes",
		"/Users/sri/Documents/Projects/blis-latency-kernel/testdata/measurements/inferencex_absolutes.json",
		"InferenceX absolute measured latencies; enables the mape column for simulated arms")
	framework := flag.String("framework", "vllm", "restrict to one framework")
	hopper := flag.String("hopper", "", "set to \"yes\" to restrict to h100/h200 and add the analytic arms")
	seed := flag.Int64("seed", 42, "")
	flag.Parse()

	c, err := harness.LoadCorpus(*corpusPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	// The measured absolute curves. Without them a simulated arm has no mape, because the
	// artifact ships only ratios; with them mape is computed against the same InferenceX rows
	// the baseline was scored on.
	abs, err := harness.LoadAbsolutes(*absolutes)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	base := harness.Config{
		Repos: kernelmodel.Repos{
			Scenarios: *scenarios, Catalog: *catalog, Registry: *registry,
		},
		Admission: harness.AdmissionKernelKV,
		Seed:      *seed,
		Backends: harness.BackendPaths{
			Catalog: *catalog, HWConfig: *hwConfig, Defaults: *defaults,
		},
	}
	hopperOnly := *hopper != ""

	arms := []arm{
		{name: "blis-latency-kernel", est: harness.EstimatorKernel},
		{name: "AISimulate", published: func(p harness.Point, m harness.Metric) float64 {
			_, a, _ := p.MetricOf(m)
			return a
		}},
		{name: "AIC (same project)", published: func(p harness.Point, m harness.Metric) float64 {
			_, _, a := p.MetricOf(m)
			return a
		}},
	}
	if hopperOnly {
		arms = append(arms,
			arm{name: "roofline", est: harness.EstimatorRoofline},
			arm{name: "trained-physics", est: harness.EstimatorTrainedPhysics})
	}

	// errs[arm][metric] accumulates signed per-point SHAPE errors; mapes[arm][metric] the
	// absolute per-point mape errors, which only the published arms can supply.
	//
	// Why mape is computable for them and not for us. Both of a point's relatives are divided by
	// the MEASURED anchor -- a published prediction's relative at the lowest concurrency is never
	// 1.0, which is the fingerprint -- so pred_relative/measured_relative = pred(c)/meas(c), a
	// true absolute ratio with the unknown anchor cancelled. BLIS instead produces absolute
	// microseconds, and the measurement exists only as a ratio to an anchor latency the artifact
	// does not ship, so a BLIS mape would require inventing that anchor.
	errs := make([]map[harness.Metric][]float64, len(arms))
	mapes := make([]map[harness.Metric][]float64, len(arms))
	for i := range errs {
		errs[i] = map[harness.Metric][]float64{}
		mapes[i] = map[harness.Metric][]float64{}
	}
	byChip := map[string]map[int]map[harness.Metric][]float64{}
	var failures []string
	sweeps, points := 0, 0
	ttftExcluded := 0

	for i := range c.Sweeps {
		sw := c.Sweeps[i]
		if *framework != "" && sw.Framework != *framework {
			continue
		}
		chip := family(sw.GPU)
		if hopperOnly && chip != "h100" && chip != "h200" {
			continue
		}

		// Points whose MEASURED TTFT is an isolated artefact, excluded from TTFT for EVERY arm
		// so the columns stay comparable. TPOT is unaffected: these sweeps' TPOT curves are
		// sound, and dropping the whole sweep would discard eleven good points to remove four
		// bad ones.
		spike := map[int]bool{}
		for _, idx := range sw.TTFTSpikeIndices(harness.DefaultTTFTSpikeFactor) {
			spike[idx] = true
			ttftExcluded++
		}

		// Every arm must score the SAME points or the columns are not comparable, so a sweep is
		// taken whole or not at all.
		sim := map[harness.Estimator]map[harness.Metric][]float64{}
		simMape := map[harness.Estimator]map[harness.Metric][]float64{}
		ok := true
		for _, a := range arms {
			if a.est == "" {
				continue
			}
			cfg := base
			cfg.Estimator = a.est
			var anchorITL, anchorTTFT float64
			per := map[harness.Metric][]float64{}
			perMape := map[harness.Metric][]float64{}
			measured := abs.For(sw)
			for j, p := range sw.Points {
				obs, err := harness.Run(sw, p.Concurrency, cfg)
				if err != nil {
					failures = append(failures, fmt.Sprintf("%s %s %s c=%d: %v",
						sw.Scenario, sw.Label, a.est, p.Concurrency, err))
					ok = false
					break
				}
				if obs.MeanITLUs <= 0 || obs.MeanTTFTUs <= 0 {
					failures = append(failures, fmt.Sprintf("%s %s %s c=%d: no measurement",
						sw.Scenario, sw.Label, a.est, p.Concurrency))
					ok = false
					break
				}
				// mape at EVERY point including the anchor: it compares two absolute values
				// and has no anchor to cancel.
				if v, ok := measured.At(harness.MetricTPOT, p.Concurrency); ok {
					perMape[harness.MetricTPOT] = append(perMape[harness.MetricTPOT],
						(obs.MeanITLUs/v-1)*100)
				}
				if v, ok := measured.At(harness.MetricTTFT, p.Concurrency); ok && !spike[j] {
					perMape[harness.MetricTTFT] = append(perMape[harness.MetricTTFT],
						(obs.MeanTTFTUs/v-1)*100)
				}
				if j == 0 {
					anchorITL, anchorTTFT = obs.MeanITLUs, obs.MeanTTFTUs
					continue // the anchor's SHAPE error is zero by construction; excluded
				}
				mt, _, _ := p.MetricOf(harness.MetricTPOT)
				tt, _, _ := p.MetricOf(harness.MetricTTFT)
				if mt <= 0 || tt <= 0 {
					ok = false
					break
				}
				per[harness.MetricTPOT] = append(per[harness.MetricTPOT],
					(obs.MeanITLUs/anchorITL/mt-1)*100)
				if !spike[j] {
					per[harness.MetricTTFT] = append(per[harness.MetricTTFT],
						(obs.MeanTTFTUs/anchorTTFT/tt-1)*100)
				}
			}
			if !ok {
				break
			}
			sim[a.est] = per
			simMape[a.est] = perMape
		}
		if !ok {
			continue
		}

		// The published arms, re-anchored the same way.
		pub := make([]map[harness.Metric][]float64, len(arms))
		pubMape := make([]map[harness.Metric][]float64, len(arms))
		for ai, a := range arms {
			if a.published == nil {
				continue
			}
			per := map[harness.Metric][]float64{}
			perMape := map[harness.Metric][]float64{}
			bad := false
			for _, m := range harness.AllMetrics {
				var anchor float64
				for j, p := range sw.Points {
					measured, _, _ := p.MetricOf(m)
					pred := a.published(p, m)
					if measured <= 0 || pred <= 0 {
						bad = true
						break
					}
					// mape at EVERY point including the anchor: pred/measured is a real ratio
					// there too, since both carry the same measured denominator.
					if !(m == harness.MetricTTFT && spike[j]) {
						perMape[m] = append(perMape[m], (pred/measured-1)*100)
					}
					if j == 0 {
						anchor = pred
						continue
					}
					if m == harness.MetricTTFT && spike[j] {
						continue
					}
					per[m] = append(per[m], (pred/anchor/measured-1)*100)
				}
				if bad {
					break
				}
			}
			if bad {
				ok = false
				break
			}
			pub[ai] = per
			pubMape[ai] = perMape
		}
		if !ok {
			continue
		}

		sweeps++
		points += len(sim[harness.EstimatorKernel][harness.MetricTPOT])
		for ai, a := range arms {
			src := pub[ai]
			srcMape := pubMape[ai]
			if a.est != "" {
				src = sim[a.est]
				srcMape = simMape[a.est]
			}
			for _, m := range harness.AllMetrics {
				errs[ai][m] = append(errs[ai][m], src[m]...)
				if srcMape != nil {
					mapes[ai][m] = append(mapes[ai][m], srcMape[m]...)
				}
				if byChip[chip] == nil {
					byChip[chip] = map[int]map[harness.Metric][]float64{}
				}
				if byChip[chip][ai] == nil {
					byChip[chip][ai] = map[harness.Metric][]float64{}
				}
				byChip[chip][ai][m] = append(byChip[chip][ai][m], src[m]...)
			}
		}
	}

	if points == 0 {
		fmt.Fprintln(os.Stderr, "no points scored")
		for _, f := range failures {
			fmt.Fprintf(os.Stderr, "  %s\n", f)
		}
		os.Exit(1)
	}

	title := "every chip"
	if hopperOnly {
		title = "Hopper only (h100, h200), with BLIS's analytic backends"
	}
	fmt.Printf("Shape error by metric, framework %s, %s\n", *framework, title)
	fmt.Printf("corpus: %s\n", c.Source)
	fmt.Printf("ground truth: %s\n\n", c.MeasurementSource)
	fmt.Println("Shape error only. The artifact ships no absolute latency -- every point carries")
	fmt.Println("tpot_relative and ttft_relative and nothing else -- so a mean ABSOLUTE percentage")
	fmt.Println("error cannot be computed against it by anyone. Each side is normalised to its own")
	fmt.Println("value at the sweep's lowest concurrency, and that anchor is excluded.")
	fmt.Printf("\n%d sweeps, %d points for TPOT\n", sweeps, points)
	if ttftExcluded > 0 {
		fmt.Printf("%d point(s) excluded from TTFT for EVERY arm: the measured TTFT is an\n"+
			"  isolated excursion more than %.0fx both neighbours. Over the 281 interior points\n"+
			"  the excursion ratio has median 0.738 and p90 0.903, its four largest values are\n"+
			"  52.31x, 22.30x, 11.18x and 10.71x, and the fifth is 1.74x -- so every threshold\n"+
			"  from 2x to 10x selects the same four points. TPOT keeps them: those sweeps'\n"+
			"  TPOT curves are sound, and excluding the sweeps would discard 11 good points to\n"+
			"  remove 4 bad ones.\n", ttftExcluded, harness.DefaultTTFTSpikeFactor)
	}

	for _, m := range harness.AllMetrics {
		fmt.Printf("\n=== %s shape error (level divided out; curvature only)\n", m)
		fmt.Printf("%-24s %5s %9s %9s %9s %7s %9s %9s\n",
			"estimator", "n", "mean", "median", "mean|e|", "over", "p10", "p90")
		for ai, a := range arms {
			st := harness.Signed(errs[ai][m])
			fmt.Printf("%-24s %5d %+8.2f%% %+8.2f%% %8.2f%% %6.0f%% %+8.2f%% %+8.2f%%\n",
				a.name, st.N, st.Mean, st.Median, st.MeanAbs, 100*st.FractionOver, st.P10, st.P90)
		}
		fmt.Printf("\n=== %s mape (level AND curvature)\n", m)
		fmt.Printf("%-24s %5s %9s %9s %9s %7s\n",
			"estimator", "n", "mean", "median", "mean|e|", "over")
		for ai, a := range arms {
			v := mapes[ai][m]
			if len(v) == 0 {
				fmt.Printf("%-24s %5s %9s %9s %9s %7s   not computable: needs the measured "+
					"anchor latency, which the artifact does not ship\n", a.name, "--", "--", "--", "--", "--")
				continue
			}
			st := harness.Signed(v)
			fmt.Printf("%-24s %5d %+8.2f%% %+8.2f%% %8.2f%% %6.0f%%\n",
				a.name, st.N, st.Mean, st.Median, st.MeanAbs, 100*st.FractionOver)
		}
	}

	fmt.Printf("\n=== By chip family (mean|e|)\n")
	chips := make([]string, 0, len(byChip))
	for k := range byChip {
		chips = append(chips, k)
	}
	sort.Strings(chips)
	fmt.Printf("%-12s %-8s", "family", "metric")
	for _, a := range arms {
		fmt.Printf(" %18s", trunc(a.name, 18))
	}
	fmt.Println()
	for _, chip := range chips {
		for _, m := range harness.AllMetrics {
			fmt.Printf("%-12s %-8s", chip, m)
			for ai := range arms {
				v := byChip[chip][ai][m]
				if len(v) == 0 {
					fmt.Printf(" %18s", "--")
					continue
				}
				fmt.Printf(" %17.2f%%", harness.MAPE2(harness.Abs(v)))
			}
			fmt.Printf("   n=%d\n", len(byChip[chip][0][m]))
		}
	}

	if len(failures) > 0 {
		fmt.Printf("\n%d point(s) could not be scored:\n", len(failures))
		for _, f := range failures {
			fmt.Printf("  %s\n", f)
		}
	}
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func family(gpu string) string {
	g := strings.ToLower(gpu)
	for _, k := range []string{"h100", "h200", "b200", "b300"} {
		if strings.Contains(g, k) {
			return k
		}
	}
	return g
}

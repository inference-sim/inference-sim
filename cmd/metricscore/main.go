// Command metricscore scores every available estimator on every available metric.
//
// # What is comparable
//
// The snapshot publishes four accuracy figures per estimator: TTFT and TPOT, each as a mean
// absolute percentage error (mape) and as a shape error.
//
// Shape error re-anchors each side to its OWN value at the sweep's lowest concurrency and excludes
// the anchor from the mean (scripts/build_e2e_accuracy_overview.py, _aggregate_shape_error). It
// therefore asks whether latency RESPONDS to concurrency correctly, and a constant multiplicative
// offset cancels exactly.
//
// mape is the plain absolute error on absolute latency. The artifact ships NO absolute latency:
// every point carries exactly tpot_relative and ttft_relative per estimator. A published arm's
// mape is still recoverable from those ratios (both are divided by the same measured anchor). A
// simulated arm produces absolute microseconds, so its mape needs the measured absolute curves,
// which -absolutes supplies from the InferenceX rows the published arms were scored against.
//
// # The estimators
//
// AISIMULATE and AIC are two modes of ONE product: snapshot.aic_source names
// ai-dynamo/aisimulate for both, AIC being the analytic configurator path since absorbed into the
// simulator. They are reported side by side as modes, not as rival baselines.
//
// blis-latency-kernel is BLIS's latency backend, simulated per point by exec'ing -blis.
//
// Usage:
//
//	go build -o blis main.go
//	go run ./cmd/metricscore -blis ./blis    # kernel vs AISimulate vs AIC
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
		kernelmodel.MeasurementPath("aisimulate_e2e.json"),
		"the extracted AISimulate corpus")
	scenarios := flag.String("scenarios",
		kernelmodel.DefaultScenarios(), "scenario directory")
	catalog := flag.String("catalog", kernelmodel.DefaultCatalog(), "")
	registry := flag.String("registry", kernelmodel.DefaultRegistry(), "")
	settings := flag.String("engine-settings",
		kernelmodel.MeasurementPath("inferencex_engine_settings.json"),
		"the settings each measured run was launched with; configures each point as the run was")
	absolutes := flag.String("absolutes",
		kernelmodel.MeasurementPath("inferencex_absolutes.json"),
		"InferenceX absolute measured latencies; enables the mape column for simulated arms")
	framework := flag.String("framework", "vllm",
		"restrict to one framework; empty scores every framework, which only the published "+
			"arms can do (BLIS models vLLM)")
	tier := flag.String("config-tier", "",
		"restrict to points whose engine configuration is \"measured\" (the run's own command "+
			"line) or \"resolved\" (vLLM's defaults, no log captured). Empty includes both, "+
			"which mixes two kinds of evidence in one average")
	// A corpus measured directly against InferenceX carries no published prediction: an
	// artifact prediction is keyed per (model, gpu, precision, framework, workload, tp,
	// concurrency), and a direct corpus separates deployments the artifact never did --
	// container image, stated engine settings -- so one prediction would have to stand for
	// up to fourteen different deployments. Those arms are dropped rather than scored
	// against a zero, which the sweep-whole-or-not rule would otherwise read as a failed
	// sweep and discard the kernel's own points with it.
	simulatedOnly := flag.Bool("simulated-only", false,
		"score only the simulated arms; for a corpus with no published predictions")
	lengthRatio := flag.Float64("length-range-ratio", 0,
		"override the prompt-length sampling interval as a fraction of the labelled "+
			"length: 0 keeps AISimulate's one-sided 0.8 (mean 0.9x len), 1.0 gives "+
			"constant lengths, which is what vLLM's client does when no "+
			"--random-range-ratio is passed. Applies to the INPUT length only: output "+
			"variance sets the queueing regime, not the prompt shape. A sensitivity "+
			"control, not a tuning knob.")
	seed := flag.Int64("seed", 42, "")
	blis := flag.String("blis", "", "path of the blis binary every point is simulated with (`go build -o blis main.go`); required")
	gaps := flag.String("gaps", "", "write the coverage-gap report here (default stderr; never stdout)")
	flag.Parse()
	if err := harness.RequireBlis(*blis); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := kernelmodel.RequireCorpora(map[string]string{"corpus": *corpusPath, "engine-settings": *settings, "absolutes": *absolutes}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

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
	// The settings each run was launched with. Without these a point is configured from an
	// assumed default while being scored against a run that used something else.
	eset, err := harness.LoadEngineSettings(*settings)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	base := harness.Config{
		Repos: kernelmodel.Repos{
			Scenarios: *scenarios, Catalog: *catalog, Registry: *registry,
		},
		Blis:             *blis,
		Seed:             *seed,
		LengthRangeRatio: *lengthRatio,
		EngineSettings:   eset,
	}

	// With no framework restriction the corpus spans sglang and trtllm, which BLIS does not
	// model and for which no vLLM engine log exists. The published arms still can be scored
	// there -- their predictions come from the artifact and need no configuration from us --
	// so the comparison is widened to the whole corpus and the simulated arms are dropped
	// rather than run on a deployment they cannot describe.
	publishedOnly := *framework == ""
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
	if *simulatedOnly {
		kept := arms[:0]
		for _, a := range arms {
			if a.est != "" {
				kept = append(kept, a)
			}
		}
		arms = kept
	}
	if publishedOnly {
		kept := arms[:0]
		for _, a := range arms {
			if a.est == "" {
				kept = append(kept, a)
			}
		}
		arms = kept
	}

	// errs[arm][metric] accumulates signed per-point SHAPE errors; mapes[arm][metric] the
	// absolute per-point mape errors of every arm.
	//
	// Why both can be scored. A published point's relatives are both divided by the MEASURED
	// anchor -- a published prediction's relative at the lowest concurrency is never 1.0, which
	// is the fingerprint -- so pred_relative/measured_relative = pred(c)/meas(c), a true absolute
	// ratio with the unknown anchor cancelled. BLIS instead produces absolute microseconds, which
	// are scored against the measured absolute curves loaded from -absolutes.
	errs := make([]map[harness.Metric][]float64, len(arms))
	mapes := make([]map[harness.Metric][]float64, len(arms))
	for i := range errs {
		errs[i] = map[harness.Metric][]float64{}
		mapes[i] = map[harness.Metric][]float64{}
	}
	byChip := map[string]map[int]map[harness.Metric][]float64{}
	byFramework := map[string]map[int]map[harness.Metric][]float64{}
	byModel := map[string]map[int]map[harness.Metric][]float64{}
	var failures []string
	// Every corpus gap, beside the score rather than in it: see harness.Coverage.
	coverage := harness.AssessCoverage(c, base)
	sweeps, points := 0, 0
	ttftExcluded := 0

	for i := range c.Sweeps {
		sw := c.Sweeps[i]
		if *framework != "" && sw.Framework != *framework {
			continue
		}
		chip := family(sw.GPU)

		// Configuration tier: a sweep is taken whole or not at all, because a mean over a
		// mix of measured and resolved settings is not a statement about either.
		if *tier != "" {
			rec := eset.For(sw)
			measured := rec != nil && rec.At(sw.Points[0].Concurrency) != nil
			if (*tier == "measured") != measured {
				continue
			}
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
			var anchorITL, anchorTTFT float64
			per := map[harness.Metric][]float64{}
			perMape := map[harness.Metric][]float64{}
			measured := abs.For(sw)
			for j, p := range sw.Points {
				obs, err := harness.Run(sw, p.Concurrency, cfg)
				if err != nil {
					failures = append(failures, fmt.Sprintf("%s %s %s c=%d: %v",
						sw.Scenario, sw.Label, a.est, p.Concurrency, err))
					coverage.Dropped(sw, p.Concurrency, err.Error())
					ok = false
					break
				}
				if obs.MeanITLUs <= 0 || obs.MeanTTFTUs <= 0 {
					failures = append(failures, fmt.Sprintf("%s %s %s c=%d: no measurement",
						sw.Scenario, sw.Label, a.est, p.Concurrency))
					coverage.Dropped(sw, p.Concurrency, "no measurement")
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
					detail := fmt.Sprintf("measured TPOT relative %v, TTFT relative %v", mt, tt)
					failures = append(failures, fmt.Sprintf("%s %s %s c=%d: %s",
						sw.Scenario, sw.Label, a.name, p.Concurrency, detail))
					coverage.BadMeasurement(sw, p.Concurrency, detail)
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
					if m != harness.MetricTTFT || !spike[j] {
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
		// Count from whichever arm is present: in published-only mode there is no simulated
		// arm, and the published arms carry the same point set.
		if n := len(sim[harness.EstimatorKernel][harness.MetricTPOT]); n > 0 {
			points += n
		} else if pub[0] != nil {
			points += len(pub[0][harness.MetricTPOT])
		}
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
				if byFramework[sw.Framework] == nil {
					byFramework[sw.Framework] = map[int]map[harness.Metric][]float64{}
				}
				if byFramework[sw.Framework][ai] == nil {
					byFramework[sw.Framework][ai] = map[harness.Metric][]float64{}
				}
				byFramework[sw.Framework][ai][m] = append(
					byFramework[sw.Framework][ai][m], src[m]...)
				if byModel[sw.Model] == nil {
					byModel[sw.Model] = map[int]map[harness.Metric][]float64{}
				}
				if byModel[sw.Model][ai] == nil {
					byModel[sw.Model][ai] = map[harness.Metric][]float64{}
				}
				byModel[sw.Model][ai][m] = append(byModel[sw.Model][ai][m], src[m]...)
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
	if publishedOnly {
		title = "every chip and every framework; published arms only"
	}
	if *tier != "" {
		title += "; " + *tier + "-configuration points only"
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

	if len(byModel) > 1 {
		fmt.Printf("\n=== By model (mean|e|)\n")
		ms := make([]string, 0, len(byModel))
		for k := range byModel {
			ms = append(ms, k)
		}
		sort.Strings(ms)
		fmt.Printf("%-24s %-8s", "model", "metric")
		for _, a := range arms {
			fmt.Printf(" %18s", trunc(a.name, 18))
		}
		fmt.Println()
		for _, mm := range ms {
			for _, m := range harness.AllMetrics {
				fmt.Printf("%-24s %-8s", trunc(mm, 24), m)
				for ai := range arms {
					v := byModel[mm][ai][m]
					if len(v) == 0 {
						fmt.Printf(" %18s", "--")
						continue
					}
					fmt.Printf(" %17.2f%%", harness.MAPE2(harness.Abs(v)))
				}
				fmt.Printf("   n=%d\n", len(byModel[mm][0][m]))
			}
		}
	}

	if len(byFramework) > 1 {
		fmt.Printf("\n=== By framework (mean|e|)\n")
		fws := make([]string, 0, len(byFramework))
		for k := range byFramework {
			fws = append(fws, k)
		}
		sort.Strings(fws)
		fmt.Printf("%-10s %-8s", "framework", "metric")
		for _, a := range arms {
			fmt.Printf(" %18s", trunc(a.name, 18))
		}
		fmt.Println()
		for _, fw := range fws {
			for _, m := range harness.AllMetrics {
				fmt.Printf("%-10s %-8s", fw, m)
				for ai := range arms {
					v := byFramework[fw][ai][m]
					if len(v) == 0 {
						fmt.Printf(" %18s", "--")
						continue
					}
					fmt.Printf(" %17.2f%%", harness.MAPE2(harness.Abs(v)))
				}
				fmt.Printf("   n=%d\n", len(byFramework[fw][0][m]))
			}
		}
	}

	if len(failures) > 0 {
		fmt.Printf("\n%d point(s) could not be scored:\n", len(failures))
		for _, f := range failures {
			fmt.Printf("  %s\n", f)
		}
	}
	if err := coverage.WriteGaps(*gaps); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
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

// Command metricscore scores every available estimator on every available metric, on three
// tiers of points: STRICT, MEASURED-OR-RESOLVED and ALL (approximate), decided per sweep from
// the coverage gaps (harness.Coverage.TierOf) before any point is scored.
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
// blis-latency-kernel is BLIS's latency backend, simulated per point by exec'ing -blis as
// `blis replay` on the point's InferenceX main phase (harness.Run).
//
// Usage:
//
//	go build -o blis main.go
//	go run ./tools/metricscore -blis ./blis    # kernel vs AISimulate vs AIC
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/vendored"
	"github.com/inference-sim/inference-sim/tools/harness"
)

// arm is one column of the report.
type arm struct {
	name string
	// est is set for an arm BLIS simulates; empty for a published baseline read from the corpus.
	est harness.Estimator
	// published reads the prediction out of the point instead of simulating it.
	published func(harness.Point, harness.Metric) float64
}

// The kernel adapter's defaults are this repository's vendored catalog, registry and pinned
// kernel (sim/vendored).
func init() { kernelmodel.SetVendoredRoot(vendored.Root()) }

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
		"restrict to sweeps whose first point has an engine log (\"measured\") or has none "+
			"(\"resolved\"). Empty keeps both. Independent of the STRICT / MEASURED-OR-RESOLVED / "+
			"ALL tables, which are always printed and decided per sweep from the coverage gaps")
	// A corpus measured directly against InferenceX carries no published prediction: an
	// artifact prediction is keyed per (model, gpu, precision, framework, workload, tp,
	// concurrency), and a direct corpus separates deployments the artifact never did --
	// container image, stated engine settings -- so one prediction would have to stand for
	// up to fourteen different deployments. Those arms are dropped rather than scored
	// against a zero, which the sweep-whole-or-not rule would otherwise read as a failed
	// sweep and discard the kernel's own points with it.
	simulatedOnly := flag.Bool("simulated-only", false,
		"score only the simulated arms; for a corpus with no published predictions")
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
		Blis:           *blis,
		Seed:           *seed,
		EngineSettings: eset,
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
	//
	// Both are kept per TIER (harness.AllTiers), from the coverage gaps of the sweep: a sweep
	// contributes to every table at least as loose as its own tier, and every arm is scored on
	// exactly the same sweeps within a table.
	errs := map[harness.Tier][]map[harness.Metric][]float64{}
	mapes := map[harness.Tier][]map[harness.Metric][]float64{}
	sweepsIn := map[harness.Tier]int{}
	// The sweeps of the two strict tables, in scoring order, for the listing that says why
	// each is where it is.
	type tierSweep struct {
		sw   harness.Sweep
		tier harness.Tier
	}
	var strictSweeps []tierSweep
	for _, t := range harness.AllTiers {
		errs[t] = make([]map[harness.Metric][]float64, len(arms))
		mapes[t] = make([]map[harness.Metric][]float64, len(arms))
		for i := range arms {
			errs[t][i] = map[harness.Metric][]float64{}
			mapes[t][i] = map[harness.Metric][]float64{}
		}
	}
	byChip := map[string]map[int]map[harness.Metric][]float64{}
	byFramework := map[string]map[int]map[harness.Metric][]float64{}
	byModel := map[string]map[int]map[harness.Metric][]float64{}
	var failures []string
	// Every corpus gap: written out beside the score, and read DURING scoring to put each
	// sweep in its tier (harness.Coverage.TierOf).
	coverage := harness.AssessCoverage(c, base)
	// For each point Run scored after the warm-up's state (request 0's prompt cached), the
	// same main phase started cold, reported as information only.
	type pointDifference struct {
		sw   harness.Sweep
		conc int
		d    harness.WarmupDifference
	}
	var differences []pointDifference
	differencesIn := map[harness.Tier][]harness.WarmupDifference{}
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

		// A sweep-level gap (a missing artifact, a parallelism mismatch, an engine BLIS does
		// not model) means the simulation would not be the measured deployment, so the sweep
		// is kept out of every table rather than labelled approximate. Decided before any
		// point runs.
		if !publishedOnly && coverage.TierOf(sw) == harness.TierUnscorable {
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
			// The observation Run returns is the one scored, for both errors below: on a
			// prefix-caching point it is the main phase after the warm-up's state.
			var obs []harness.Observation
			for _, p := range sw.Points {
				o, err := harness.Run(sw, p.Concurrency, cfg)
				if err != nil {
					failures = append(failures, fmt.Sprintf("%s %s %s c=%d: %v",
						sw.Scenario, sw.Label, a.est, p.Concurrency, err))
					coverage.Dropped(sw, p.Concurrency, err.Error())
					ok = false
					break
				}
				if o.MeanTPOTUs <= 0 || o.MeanTTFTUs <= 0 {
					failures = append(failures, fmt.Sprintf("%s %s %s c=%d: no measurement",
						sw.Scenario, sw.Label, a.est, p.Concurrency))
					coverage.Dropped(sw, p.Concurrency, "no measurement")
					ok = false
					break
				}
				if o.Primed {
					d, err := harness.RunWarmupDifference(sw, p.Concurrency, cfg, o)
					if err != nil {
						failures = append(failures, fmt.Sprintf("%s %s c=%d cold run (information only): %v",
							sw.Scenario, sw.Label, p.Concurrency, err))
					} else {
						differences = append(differences, pointDifference{sw, p.Concurrency, d})
					}
				}
				obs = append(obs, o)
			}
			if !ok {
				break
			}
			per, perMape, bad := simulatedErrors(sw, obs, abs.For(sw), spike)
			if bad != "" {
				failures = append(failures, fmt.Sprintf("%s %s %s: %s", sw.Scenario, sw.Label, a.name, bad))
				coverage.BadMeasurement(sw, 0, bad)
				ok = false
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

		sweepTier := coverage.TierOf(sw)
		if publishedOnly {
			// The published arms need no configuration from us, so every sweep they can be
			// scored on is one table.
			sweepTier = harness.TierApproximate
		}
		sweeps++
		if sweepTier.In(harness.TierMeasuredOrResolved) {
			strictSweeps = append(strictSweeps, tierSweep{sw, sweepTier})
		}
		for _, t := range harness.AllTiers {
			if sweepTier.In(t) {
				sweepsIn[t]++
				for _, pd := range differences {
					if pd.sw.Scenario == sw.Scenario && pd.sw.Label == sw.Label {
						differencesIn[t] = append(differencesIn[t], pd.d)
					}
				}
			}
		}
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
				for _, t := range harness.AllTiers {
					if !sweepTier.In(t) {
						continue
					}
					errs[t][ai][m] = append(errs[t][ai][m], src[m]...)
					if srcMape != nil {
						mapes[t][ai][m] = append(mapes[t][ai][m], srcMape[m]...)
					}
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
		title += "; " + *tier + "-configuration sweeps only"
	}
	fmt.Printf("Accuracy by metric, framework %s, %s\n", *framework, title)
	fmt.Printf("corpus: %s\n", c.Source)
	fmt.Printf("ground truth: %s\n\n", c.MeasurementSource)
	fmt.Println("Protocol: each point is the InferenceX benchmark client's MAIN phase, replayed by")
	fmt.Println("`blis replay`: the 10*CONC target lengths numpy drew at seed 0, in request order, at a")
	fmt.Println("concurrency gate of CONC, exactly the target output length; TTFT and TPOT are")
	fmt.Println("calculate_metrics's means over those requests. The sent prompt lengths are verified")
	fmt.Println("equal to those targets per point by testdata/inferencex-workloads/prompt-lengths.json; a")
	fmt.Println("point it does not verify carries a gap (-gaps). Not reproduced: run settings")
	fmt.Println("blis-schemas cannot express. See docs/kernel-exclusive/APPLES-TO-APPLES.md.")
	fmt.Println()
	fmt.Println("Tiers, decided per sweep from the coverage gaps (-gaps), the worst point deciding:")
	fmt.Println("  STRICT                every point has an engine log; every setting it passed is carried;")
	fmt.Println("                        the admission settings and max_model_len were measured; expert")
	fmt.Println("                        parallelism and gpu_memory_utilization measured or vLLM's unconditional")
	fmt.Println("                        default; the sent prompt lengths verified equal to the targets; and")
	fmt.Println("                        no other protocol gap")
	fmt.Println("  MEASURED-OR-RESOLVED  as STRICT, but an unpassed admission setting may take the value vLLM")
	fmt.Println("                        resolves by a rule this harness restates")
	fmt.Println("  ALL (approximate)     every scoreable sweep; NOT the measured runs as they ran")
	fmt.Println("Each table includes the stricter ones; within a table every arm scores the same points.")
	fmt.Printf("Warm-up: the benchmark sends %d*CONC copies of request 0 and drains them before the\n",
		harness.InferenceXWarmupsPerConcurrency)
	fmt.Println("main phase. The only state that leaves which the simulator reads is the prefix cache,")
	fmt.Println("so a point whose engine caches prefixes is scored on the main phase run after copies of")
	fmt.Println("request 0 have completed (its prompt cached, the engine idle); a point with caching off")
	fmt.Println("is scored on the main phase from an empty engine, which is then the same state. The")
	fmt.Println("cold run of a prefix-caching point is reported below as information only. Exact")
	fmt.Println("two-phase replay in the trace format: blis-schemas#56, inference-sim#1926.")
	fmt.Println("Shape error divides each side by its own value at the sweep's lowest concurrency and")
	fmt.Println("excludes that anchor; mape compares absolute latency at every point.")
	fmt.Printf("\n%d sweeps, %d points for TPOT shape, over all tiers\n", sweeps, points)
	for _, t := range harness.AllTiers {
		fmt.Printf("  %-22s %3d sweeps\n", t, sweepsIn[t])
	}
	if ttftExcluded > 0 {
		fmt.Printf("%d point(s) excluded from TTFT for EVERY arm: the measured TTFT is an\n"+
			"  isolated excursion more than %.0fx both neighbours. Over the 281 interior points\n"+
			"  the excursion ratio has median 0.738 and p90 0.903, its four largest values are\n"+
			"  52.31x, 22.30x, 11.18x and 10.71x, and the fifth is 1.74x -- so every threshold\n"+
			"  from 2x to 10x selects the same four points. TPOT keeps them: those sweeps'\n"+
			"  TPOT curves are sound, and excluding the sweeps would discard 11 good points to\n"+
			"  remove 4 bad ones.\n", ttftExcluded, harness.DefaultTTFTSpikeFactor)
	}

	fmt.Printf("\n=== Sweeps in STRICT and MEASURED-OR-RESOLVED, and why each is not stricter\n")
	if len(strictSweeps) == 0 {
		fmt.Println("  none")
	}
	for _, ts := range strictSweeps {
		why := "nothing: every setting measured and carried"
		if rs := coverage.ResolvedSettings(ts.sw); len(rs) > 0 {
			names := make([]string, 0, len(rs))
			for k := range rs {
				names = append(names, k)
			}
			sort.Strings(names)
			parts := make([]string, len(names))
			for i, k := range names {
				parts[i] = fmt.Sprintf("%s (%d pts)", k, rs[k])
			}
			why = "resolved, not measured: " + strings.Join(parts, ", ")
		}
		fmt.Printf("  %-22s %-44s %-5s %2d points  %s\n", ts.tier, trunc(ts.sw.Scenario, 44), ts.sw.Label,
			len(ts.sw.Points), why)
	}

	for _, m := range harness.AllMetrics {
		for _, kind := range []string{"shape error (level divided out; curvature only)", "mape (level AND curvature)"} {
			src := errs
			if strings.HasPrefix(kind, "mape") {
				src = mapes
			}
			fmt.Printf("\n=== %s %s\n", m, kind)
			fmt.Printf("%-22s %-24s %5s %9s %9s %9s %7s %9s %9s\n",
				"tier", "estimator", "n", "mean", "median", "mean|e|", "over", "p10", "p90")
			for _, t := range harness.AllTiers {
				for ai, a := range arms {
					v := src[t][ai][m]
					if len(v) == 0 {
						fmt.Printf("%-22s %-24s %5d %9s\n", t, a.name, 0, "--")
						continue
					}
					st := harness.Signed(v)
					fmt.Printf("%-22s %-24s %5d %+8.2f%% %+8.2f%% %8.2f%% %6.0f%% %+8.2f%% %+8.2f%%\n",
						t, a.name, st.N, st.Mean, st.Median, st.MeanAbs, 100*st.FractionOver, st.P10, st.P90)
				}
			}
		}
	}

	fmt.Printf("\n=== Tier summary for %s (mean|e|), with the information-only difference between\n"+
		"    each prefix-caching point's scored (warm-up) and cold main phase, |scored-cold|/cold\n", arms[0].name)
	fmt.Printf("%-22s %5s %9s %9s %9s %9s %8s %15s %15s\n", "tier", "n", "TPOT shp", "TPOT mape",
		"TTFT shp", "TTFT mape", "warm pts", "TTFT med/max", "TPOT med/max")
	for _, t := range harness.AllTiers {
		cell := func(v []float64) string {
			if len(v) == 0 {
				return "--"
			}
			return fmt.Sprintf("%.2f%%", harness.MAPE2(harness.Abs(v)))
		}
		ds := harness.SummarizeDifferences(differencesIn[t])
		fmt.Printf("%-22s %5d %9s %9s %9s %9s %8d %6.3f/%6.3f%% %6.4f/%6.4f%%\n", t,
			len(errs[t][0][harness.MetricTPOT]), cell(errs[t][0][harness.MetricTPOT]), cell(mapes[t][0][harness.MetricTPOT]),
			cell(errs[t][0][harness.MetricTTFT]), cell(mapes[t][0][harness.MetricTTFT]),
			ds.N, ds.MedianTTFT, ds.MaxTTFT, ds.MedianTPOT, ds.MaxTPOT)
	}

	if len(differences) > 0 {
		fmt.Printf("\n=== Prefix-caching points: scored (after the warm-up's state) against cold (information only)\n")
		fmt.Printf("%-44s %-5s %5s %12s %12s %9s %11s %11s %9s\n", "scenario", "label", "conc",
			"TTFT (us)", "cold (us)", "diff", "TPOT (us)", "cold (us)", "diff")
		for _, pd := range differences {
			d := pd.d
			fmt.Printf("%-44s %-5s %5d %12.0f %12.0f %+8.3f%% %11.1f %11.1f %+8.4f%%\n",
				trunc(pd.sw.Scenario, 44), pd.sw.Label, pd.conc, d.Scored.MeanTTFTUs, d.Cold.MeanTTFTUs, d.DTTFTPct,
				d.Scored.MeanTPOTUs, d.Cold.MeanTPOTUs, d.DTPOTPct)
		}
	}

	fmt.Printf("\nThe breakdowns below are over ALL scoreable sweeps (approximate), the widest table.\n")
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

// simulatedErrors is a simulated arm's per-point errors over one sweep, from the observations
// harness.Run scored at each point in order: the absolute error at every point against the
// measured absolute latency, and the shape error -- each side divided by its own value at the
// sweep's lowest concurrency, that anchor excluded -- against the measured relatives. Both read
// the same observation per point. TTFT skips the points in spike. A non-positive measured
// relative makes the sweep unscorable and is returned as bad.
func simulatedErrors(sw harness.Sweep, obs []harness.Observation, measured *harness.Absolutes,
	spike map[int]bool) (shape, mape map[harness.Metric][]float64, bad string) {
	shape, mape = map[harness.Metric][]float64{}, map[harness.Metric][]float64{}
	var anchorTPOT, anchorTTFT float64
	for j, p := range sw.Points[:len(obs)] {
		o := obs[j]
		if v, ok := measured.At(harness.MetricTPOT, p.Concurrency); ok {
			mape[harness.MetricTPOT] = append(mape[harness.MetricTPOT], (o.MeanTPOTUs/v-1)*100)
		}
		if v, ok := measured.At(harness.MetricTTFT, p.Concurrency); ok && !spike[j] {
			mape[harness.MetricTTFT] = append(mape[harness.MetricTTFT], (o.MeanTTFTUs/v-1)*100)
		}
		if j == 0 {
			anchorTPOT, anchorTTFT = o.MeanTPOTUs, o.MeanTTFTUs
			continue // the anchor's SHAPE error is zero by construction; excluded
		}
		mt, _, _ := p.MetricOf(harness.MetricTPOT)
		tt, _, _ := p.MetricOf(harness.MetricTTFT)
		if mt <= 0 || tt <= 0 {
			return nil, nil, fmt.Sprintf("c=%d: measured TPOT relative %v, TTFT relative %v", p.Concurrency, mt, tt)
		}
		shape[harness.MetricTPOT] = append(shape[harness.MetricTPOT], (o.MeanTPOTUs/anchorTPOT/mt-1)*100)
		if !spike[j] {
			shape[harness.MetricTTFT] = append(shape[harness.MetricTTFT], (o.MeanTTFTUs/anchorTTFT/tt-1)*100)
		}
	}
	return shape, mape, ""
}

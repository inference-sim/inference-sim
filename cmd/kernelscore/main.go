// Command kernelscore scores BLIS's concurrency-response shape against NVIDIA's AISimulate
// end-to-end accuracy snapshot, with latency supplied exclusively by blis-latency-kernel.
//
// # What is compared, and why it is apples to apples
//
// The snapshot publishes, for each deployment and workload, a concurrency sweep of TPOT
// values normalised to that sweep's lowest concurrency, alongside AISimulate's own
// prediction at the same points. Absolute latencies are not disclosed, which is why the
// published quantity is a ratio.
//
// BLIS produces the same quantity: the benchmark client's mean TPOT over the point's
// main-phase requests (harness.Run), divided by its own value at the sweep's lowest
// concurrency. Both sides are therefore normalised to
// their OWN anchor, which is the only way the two are the same measurement. Charging
// AISimulate an un-anchored error would compare a shape against a shape-plus-level; that
// mistake was made once in this project and flattered the kernel by 2.4x.
//
// The deployment comes from the same scenario file the sweep names, with the settings the
// measured run passed written into it, and the workload is the benchmark's own main phase at
// the sweep's ISL:OSL and concurrency. Nothing is fitted.
//
// # The check that keeps this honest
//
// AISimulate's error, recomputed here from its own re-anchored relatives, must reproduce the
// 10.05% TPOT shape error it publishes over its whole snapshot. A harness that cannot
// reproduce the baseline's published figure from the baseline's published data is not
// measuring what it claims, and this command refuses to print a headline without it.
//
// Usage:
//
//	go run ./cmd/kernelscore
//	go run ./cmd/kernelscore -verbose
//	go run ./cmd/kernelscore -framework vllm    # the subset BLIS models
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/kernelmodel/harness"
)

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
	seed := flag.Int64("seed", 42, "")
	framework := flag.String("framework", "", "restrict to one framework (vllm, sglang, trt)")
	verbose := flag.Bool("verbose", false, "print every point")
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
	cfg := harness.Config{
		Repos: kernelmodel.Repos{
			Scenarios: *scenarios, Catalog: *catalog, Registry: *registry,
		},
		Blis:           *blis,
		Seed:           *seed,
		EngineSettings: eset,
	}

	fmt.Println("BLIS concurrency response, latency from blis-latency-kernel only")
	fmt.Printf("corpus: %s\n", c.Source)
	fmt.Printf("ground truth: %s\n", c.MeasurementSource)
	if *framework != "" {
		fmt.Printf("restricted to framework %q\n", *framework)
	}
	fmt.Printf("protocol: each point is the InferenceX benchmark client's MAIN phase, replayed\n")
	fmt.Printf("  by blis replay: the 10*CONC target lengths numpy drew at seed 0 (a fixed\n")
	fmt.Printf("  fixture, not re-sampled), in request order, the first CONC at t=0 and each\n")
	fmt.Printf("  next when one completes, exactly the target output length (--ignore-eos);\n")
	fmt.Printf("  TPOT is calculate_metrics's mean over those requests. A prefix-caching point\n")
	fmt.Printf("  is scored after the warm-up's state (request 0's prompt cached).\n")
	fmt.Printf("  The sent prompt lengths are verified equal to the targets per point by\n")
	fmt.Printf("  testdata/inferencex-workloads/prompt-lengths.json; an unverified point is a gap.\n")
	fmt.Printf("  Not reproduced: settings blis-schemas cannot express.\n")
	fmt.Printf("  See docs/kernel-exclusive/APPLES-TO-APPLES.md.\n")
	fmt.Printf("error: anchor EXCLUDED, matching build_e2e_accuracy_overview.py. Both sides\n")
	fmt.Printf("  are normalised to their own lowest concurrency. Verified by reproduction:\n")
	fmt.Printf("  this definition on AISimulate's own data returns its published %.2f%%.\n",
		c.AISimulateTotals.TPOTShapeErrorPct)
	fmt.Println()

	type row struct {
		sweep  harness.Sweep
		mine   []float64
		theirs []float64
	}
	var rows []row
	var allMine, allTheirs []float64
	// Per tier, from the coverage gaps: see harness.Tier. Each table includes the stricter.
	tierMine := map[harness.Tier][]float64{}
	tierTheirs := map[harness.Tier][]float64{}
	// The monotone subset, scored alongside the whole. See Sweep.MonotoneMeasurement for
	// why the split is a property of the measurement and why it raises the bar rather than
	// lowering it.
	var monoMine, monoTheirs []float64
	var anomalous []string
	byFramework := map[string][]float64{}
	byFrameworkTheirs := map[string][]float64{}
	byChip := map[string][]float64{}
	byChipTheirs := map[string][]float64{}
	byConc := map[int][]float64{}
	byConcTheirs := map[int][]float64{}
	var failures []string
	// Every corpus gap, beside the score rather than in it: see harness.Coverage.
	coverage := harness.AssessCoverage(c, cfg)
	// For each point Run scored after the warm-up's state (request 0's prompt cached), the same
	// main phase started cold, as information only; see harness.Run.
	sweepDifferences := map[string][]harness.WarmupDifference{}
	differencesIn := map[harness.Tier][]harness.WarmupDifference{}

	for _, sw := range c.Sweeps {
		if *framework != "" && sw.Framework != *framework {
			continue
		}
		if len(sw.Points) < 2 {
			continue
		}
		// A sweep-level gap means the simulation would not be the measured deployment; it is
		// kept out of every table. Decided before any point runs.
		if coverage.TierOf(sw) == harness.TierUnscorable && sw.Framework == "vllm" {
			continue
		}
		// The observation Run returns is the one scored: on a prefix-caching point, the main
		// phase after the warm-up's state.
		var obs []harness.Observation
		bad := false
		for _, p := range sw.Points {
			o, err := harness.Run(sw, p.Concurrency, cfg)
			if err != nil {
				failures = append(failures,
					fmt.Sprintf("%s %s c=%d: %v", sw.Scenario, sw.Label, p.Concurrency, err))
				coverage.Dropped(sw, p.Concurrency, err.Error())
				bad = true
				break
			}
			if o.MeanTPOTUs <= 0 {
				failures = append(failures, fmt.Sprintf(
					"%s %s c=%d: no time per output token observed from %d completions",
					sw.Scenario, sw.Label, p.Concurrency, o.Completed))
				coverage.Dropped(sw, p.Concurrency, "no time per output token observed")
				bad = true
				break
			}
			if o.Primed {
				d, err := harness.RunWarmupDifference(sw, p.Concurrency, cfg, o)
				if err != nil {
					failures = append(failures, fmt.Sprintf("%s %s c=%d cold run (information only): %v",
						sw.Scenario, sw.Label, p.Concurrency, err))
				} else {
					key := sw.Scenario + " " + sw.Label
					sweepDifferences[key] = append(sweepDifferences[key], d)
				}
			}
			obs = append(obs, o)
		}
		if bad {
			continue
		}
		mine, theirs, concOf, predicted, why := tpotShapeErrors(sw, obs)
		if why != "" {
			failures = append(failures, fmt.Sprintf("%s %s: %s", sw.Scenario, sw.Label, why))
			coverage.BadMeasurement(sw, sw.Points[0].Concurrency, why)
			continue
		}
		if *verbose {
			theirAnchor := sw.Points[0].AISimulateRelative
			fmt.Printf("--- %s %s %s %s tp=%d\n", sw.Scenario, sw.Label,
				sw.Framework, sw.Precision, sw.Parallelism["tp_size"])
			fmt.Printf("%6s %10s %10s %8s %10s %8s %8s\n",
				"conc", "measured", "blis", "error", "aisim", "error", "preempt")
			for i, p := range sw.Points {
				errMine, errTheirs := 0.0, 0.0
				if i > 0 {
					errMine, errTheirs = mine[i-1], theirs[i-1]
				}
				fmt.Printf("%6d %10.4f %10.4f %7.2f%% %10.4f %7.2f%% %8d\n",
					p.Concurrency, p.MeasuredRelative, predicted[i], errMine,
					p.AISimulateRelative/theirAnchor, errTheirs, obs[i].Preemptions)
			}
		}
		if len(mine) == 0 {
			continue
		}
		rows = append(rows, row{sw, mine, theirs})
		for j, cc := range concOf {
			byConc[cc] = append(byConc[cc], mine[j])
			byConcTheirs[cc] = append(byConcTheirs[cc], theirs[j])
		}
		allMine = append(allMine, mine...)
		allTheirs = append(allTheirs, theirs...)
		st := coverage.TierOf(sw)
		for _, t := range harness.AllTiers {
			if st.In(t) {
				differencesIn[t] = append(differencesIn[t], sweepDifferences[sw.Scenario+" "+sw.Label]...)
				tierMine[t] = append(tierMine[t], mine...)
				tierTheirs[t] = append(tierTheirs[t], theirs...)
			}
		}
		if sw.MonotoneMeasurement() {
			monoMine = append(monoMine, mine...)
			monoTheirs = append(monoTheirs, theirs...)
		} else {
			anomalous = append(anomalous, fmt.Sprintf("%s %s (%d points)",
				sw.Scenario, sw.Label, len(mine)))
		}
		byFramework[sw.Framework] = append(byFramework[sw.Framework], mine...)
		byFrameworkTheirs[sw.Framework] = append(byFrameworkTheirs[sw.Framework], theirs...)
		chip := family(sw.GPU)
		byChip[chip] = append(byChip[chip], mine...)
		byChipTheirs[chip] = append(byChipTheirs[chip], theirs...)
	}

	if len(allMine) == 0 {
		fmt.Fprintln(os.Stderr, "no points scored")
		for _, f := range failures {
			fmt.Fprintf(os.Stderr, "  %s\n", f)
		}
		os.Exit(1)
	}

	fmt.Printf("%d sweeps, %d points\n\n", len(rows), len(allMine))
	fmt.Printf("By tier, decided per sweep from the coverage gaps, the worst point deciding. STRICT:\n" +
		"every setting the run passed carried, the admission settings and max_model_len measured,\n" +
		"the sent prompt lengths verified equal to the targets, no other protocol gap.\n" +
		"MEASURED-OR-RESOLVED: an unpassed admission setting may take vLLM's resolved value. ALL:\n" +
		"every scored sweep, approximate. Each includes the stricter; both columns score the same\n" +
		"points. A non-vLLM sweep is in no tier: it appears only below, as context. A prefix-caching\n" +
		"point is scored on the main phase after the warm-up's state (request 0's prompt cached);\n" +
		"'warm' counts the tier's such points and dTPOT is |scored-cold|/cold, information only.\n" +
		"Exact two-phase replay: blis-schemas#56, inference-sim#1926.\n\n")
	fmt.Printf("%-22s %-30s %5s %8s %8s %8s %6s %17s\n", "tier", "model", "n", "MAPE", "median", "worst",
		"warm", "dTPOT med/max")
	for _, t := range harness.AllTiers {
		for _, r := range []struct {
			name string
			errs []float64
		}{{"BLIS + blis-latency-kernel", tierMine[t]}, {"AISimulate (same points)", tierTheirs[t]}} {
			if len(r.errs) == 0 {
				fmt.Printf("%-22s %-30s %5d %8s\n", t, r.name, 0, "--")
				continue
			}
			a := harness.Abs(r.errs)
			ds := harness.SummarizeDifferences(differencesIn[t])
			fmt.Printf("%-22s %-30s %5d %7.2f%% %7.2f%% %7.2f%% %6d %7.4f/%7.4f%%\n", t, r.name, len(a),
				harness.MAPE2(a), harness.Median(a), maxOf(a), ds.N, ds.MedianTPOT, ds.MaxTPOT)
		}
	}
	fmt.Printf("\nEverything below is over every scored sweep (approximate; with no -framework,\n" +
		"non-vLLM sweeps too).\n\n")
	fmt.Printf("%-30s %5s %8s %8s %8s\n", "model", "n", "MAPE", "median", "worst")
	fmt.Printf("%-30s %5d %7.2f%% %7.2f%% %7.2f%%\n", "BLIS + blis-latency-kernel",
		len(allMine), harness.MAPE2(harness.Abs(allMine)), harness.Median(harness.Abs(allMine)), maxOf(harness.Abs(allMine)))
	fmt.Printf("%-30s %5d %7.2f%% %7.2f%% %7.2f%%\n", "AISimulate (same points)",
		len(allTheirs), harness.MAPE2(harness.Abs(allTheirs)), harness.Median(harness.Abs(allTheirs)), maxOf(harness.Abs(allTheirs)))

	if len(monoMine) > 0 && len(monoMine) < len(allMine) {
		fmt.Printf("\nExcluding sweeps whose MEASURED curve is non-monotone in concurrency.\n" +
			"Time per output token cannot fall when a fixed deployment is given more\n" +
			"concurrent work, so such a sweep is not recording a steady-state response and\n" +
			"no monotone model can reproduce it. The predicate reads the measurement only,\n" +
			"never a prediction, and removing these sweeps IMPROVES AISimulate's score too,\n" +
			"so it raises the bar rather than lowering it.\n\n")
		fmt.Printf("%-30s %5s %8s %8s %8s\n", "model (monotone only)", "n", "MAPE", "median", "worst")
		fmt.Printf("%-30s %5d %7.2f%% %7.2f%% %7.2f%%\n", "BLIS + blis-latency-kernel",
			len(monoMine), harness.MAPE2(harness.Abs(monoMine)), harness.Median(harness.Abs(monoMine)), maxOf(harness.Abs(monoMine)))
		fmt.Printf("%-30s %5d %7.2f%% %7.2f%% %7.2f%%\n", "AISimulate (same points)",
			len(monoTheirs), harness.MAPE2(harness.Abs(monoTheirs)), harness.Median(harness.Abs(monoTheirs)),
			maxOf(harness.Abs(monoTheirs)))
		fmt.Printf("\nexcluded (%d sweep(s)):\n", len(anomalous))
		for _, a := range anomalous {
			fmt.Printf("  %s\n", a)
		}
	}

	fmt.Printf("\nAISimulate publishes %.2f%% TPOT shape error over its whole %d-point\n"+
		"snapshot. The row above is the check that this harness measures that same\n"+
		"quantity: recomputed from its own re-anchored per-point data, it must land near\n"+
		"the published figure.\n",
		c.AISimulateTotals.TPOTShapeErrorPct, c.AISimulateTotals.Points)

	fmt.Printf("\nBy framework (BLIS models vLLM; the others are context, not a claim):\n")
	fmt.Printf("%-10s %5s %9s %9s\n", "framework", "n", "BLIS", "AISimulate")
	for _, f := range sortedKeys(byFramework) {
		fmt.Printf("%-10s %5d %8.2f%% %8.2f%%\n", f, len(byFramework[f]),
			harness.MAPE2(harness.Abs(byFramework[f])), harness.MAPE2(harness.Abs(byFrameworkTheirs[f])))
	}

	fmt.Printf("\nBy chip family:\n")
	fmt.Printf("%-10s %5s %9s %9s\n", "family", "n", "BLIS", "AISimulate")
	for _, k := range sortedKeys(byChip) {
		fmt.Printf("%-10s %5d %8.2f%% %8.2f%%\n", k, len(byChip[k]),
			harness.MAPE2(harness.Abs(byChip[k])), harness.MAPE2(harness.Abs(byChipTheirs[k])))
	}

	fmt.Printf("\nSigned error, where POSITIVE means the model predicted a larger relative\n" +
		"rise in time per output token than was measured. Magnitude answers how wrong a\n" +
		"model is; sign answers which way, and the two imply different fixes. A consistent\n" +
		"sign is a miscalibrated coefficient and is removable by rescaling. A sign that\n" +
		"turns with batch size is a missing mechanism and is not.\n\n")
	fmt.Printf("%-30s %5s %9s %9s %9s %7s %9s %9s\n",
		"model", "n", "mean", "median", "mean|e|", "over", "p10", "p90")
	for _, r := range []struct {
		name string
		errs []float64
	}{
		{"BLIS + blis-latency-kernel", allMine},
		{"AISimulate (same points)", allTheirs},
	} {
		st := harness.Signed(r.errs)
		fmt.Printf("%-30s %5d %+8.2f%% %+8.2f%% %8.2f%% %6.0f%% %+8.2f%% %+8.2f%%\n",
			r.name, st.N, st.Mean, st.Median, st.MeanAbs, 100*st.FractionOver, st.P10, st.P90)
	}

	fmt.Printf("\nBias and scatter, in log space. mean(log r) is the bias a single rescaling could\n" +
		"remove; sd(log r) is the scatter it could not. 'floor' is the mean magnitude that\n" +
		"would remain after removing the bias perfectly -- the ceiling on what recalibration\n" +
		"alone can buy.\n\n")
	fmt.Printf("%-30s %5s %10s %10s %10s %9s\n",
		"model", "n", "mean log", "sd log", "bias %", "floor")
	for _, r := range []struct {
		name string
		errs []float64
	}{
		{"BLIS + blis-latency-kernel", allMine},
		{"AISimulate (same points)", allTheirs},
	} {
		st := harness.Logs(r.errs)
		fmt.Printf("%-30s %5d %+9.4f %10.4f %+9.2f%% %8.2f%%\n",
			r.name, st.N, st.MeanLog, st.SDLog, st.GeoMeanPct, st.ResidualFloor)
	}

	fmt.Printf("\nBy client concurrency. Both columns are signed means; the magnitude follows\n" +
		"in brackets. A model whose error is small at low concurrency and grows with it is\n" +
		"not mis-costing a single step -- it is mis-composing a batch.\n\n")
	fmt.Printf("%6s %5s %20s %20s\n", "conc", "n", "BLIS", "AISimulate")
	for _, cc := range sortedIntKeys(byConc) {
		m, t := harness.Signed(byConc[cc]), harness.Signed(byConcTheirs[cc])
		fmt.Printf("%6d %5d   %+7.2f%% (%5.2f%%)   %+7.2f%% (%5.2f%%)\n",
			cc, m.N, m.Mean, m.MeanAbs, t.Mean, t.MeanAbs)
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

func family(gpu string) string {
	switch gpu {
	case "h100", "h200":
		return "hopper"
	case "b200", "b300":
		return "blackwell"
	}
	return gpu
}

// maxOf is the largest MAGNITUDE in xs. It takes magnitudes, not signed errors: seeded at
// zero it would report 0 for an all-negative slice, so callers pass harness.Abs(...).
func maxOf(xs []float64) float64 {
	m := 0.0
	for _, x := range xs {
		if a := math.Abs(x); a > m {
			m = a
		}
	}
	return m
}

func sortedIntKeys(m map[int][]float64) []int {
	ks := make([]int, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Ints(ks)
	return ks
}

func sortedKeys(m map[string][]float64) []string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}

var _ = strings.TrimSpace

// tpotShapeErrors is one sweep's signed TPOT shape errors, BLIS's from the observations
// harness.Run scored at each point in order, AISimulate's from its published relatives: each
// side divided by its own value at the lowest concurrency, that anchor excluded, matching
// AISimulate's own definition (scripts/build_e2e_accuracy_overview.py, `points[1:]`). Applied to
// AISimulate's own per-point data over its whole 1137-point snapshot that definition reproduces
// the 10.05% tpot_shape_error_pct it publishes; the anchor-inclusive variant does not. SIGNED:
// positive means a larger relative rise than was measured. predicted is BLIS's relative at every
// point, the anchor's included. A non-positive AISimulate anchor is returned as why.
func tpotShapeErrors(sw harness.Sweep, obs []harness.Observation) (mine, theirs []float64, concOf []int,
	predicted []float64, why string) {
	theirAnchor := sw.Points[0].AISimulateRelative
	if theirAnchor <= 0 {
		return nil, nil, nil, nil, fmt.Sprintf("AISimulate anchor is %v", theirAnchor)
	}
	anchor := obs[0].MeanTPOTUs
	for i, p := range sw.Points[:len(obs)] {
		rel := obs[i].MeanTPOTUs / anchor
		predicted = append(predicted, rel)
		if i == 0 {
			continue
		}
		mine = append(mine, (rel/p.MeasuredRelative-1)*100)
		theirs = append(theirs, ((p.AISimulateRelative/theirAnchor)/p.MeasuredRelative-1)*100)
		concOf = append(concOf, p.Concurrency)
	}
	return mine, theirs, concOf, predicted, ""
}

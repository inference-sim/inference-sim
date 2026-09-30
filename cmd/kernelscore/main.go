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
// BLIS produces the same quantity: mean inter-token latency at a fixed deployment, divided
// by its own value at the sweep's lowest concurrency. Both sides are therefore normalised to
// their OWN anchor, which is the only way the two are the same measurement. Charging
// AISimulate an un-anchored error would compare a shape against a shape-plus-level; that
// mistake was made once in this project and flattered the kernel by 2.4x.
//
// The deployment comes from the same scenario file the sweep names, and the workload from
// the sweep's own ISL:OSL and concurrency. Nothing is fitted.
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
		"/Users/sri/Documents/Projects/blis-latency-kernel/testdata/measurements/aisimulate_e2e.json",
		"the extracted AISimulate corpus")
	scenarios := flag.String("scenarios",
		"/Users/sri/Documents/Projects/blis-latency-kernel/testdata/aisimulate", "scenario directory")
	catalog := flag.String("catalog", "/Users/sri/Documents/Projects/blis-catalog", "")
	registry := flag.String("registry", "/Users/sri/Documents/Projects/blis-registry", "")
	sessions := flag.Int("sessions", 120, "requests completed per point")
	seed := flag.Int64("seed", 42, "")
	framework := flag.String("framework", "", "restrict to one framework (vllm, sglang, trt)")
	verbose := flag.Bool("verbose", false, "print every point")
	flag.Parse()

	c, err := harness.LoadCorpus(*corpusPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	cfg := harness.Config{
		Repos: kernelmodel.Repos{
			Scenarios: *scenarios, Catalog: *catalog, Registry: *registry,
		},
		Admission:        harness.AdmissionKernelKV,
		SessionsPerPoint: *sessions,
		Seed:             *seed,
	}

	fmt.Println("BLIS concurrency response, latency from blis-latency-kernel only")
	fmt.Printf("corpus: %s\n", c.Source)
	fmt.Printf("ground truth: %s\n", c.MeasurementSource)
	if *framework != "" {
		fmt.Printf("restricted to framework %q\n", *framework)
	}
	fmt.Println()

	type row struct {
		sweep  harness.Sweep
		mine   []float64
		theirs []float64
	}
	var rows []row
	var allMine, allTheirs []float64
	byFramework := map[string][]float64{}
	byFrameworkTheirs := map[string][]float64{}
	byChip := map[string][]float64{}
	byChipTheirs := map[string][]float64{}
	var failures []string

	for _, sw := range c.Sweeps {
		if *framework != "" && sw.Framework != *framework {
			continue
		}
		if len(sw.Points) < 2 {
			continue
		}
		var mine, theirs []float64
		var anchor, theirAnchor float64
		bad := false
		for i, p := range sw.Points {
			obs, err := harness.Run(sw, p.Concurrency, cfg)
			if err != nil {
				failures = append(failures,
					fmt.Sprintf("%s %s c=%d: %v", sw.Scenario, sw.Label, p.Concurrency, err))
				bad = true
				break
			}
			if obs.MeanITLUs <= 0 {
				failures = append(failures, fmt.Sprintf(
					"%s %s c=%d: no inter-token latency observed from %d completions",
					sw.Scenario, sw.Label, p.Concurrency, obs.Completed))
				bad = true
				break
			}
			if i == 0 {
				anchor = obs.MeanITLUs
				theirAnchor = p.AISimulateRelative
				if theirAnchor <= 0 {
					failures = append(failures, fmt.Sprintf(
						"%s %s: AISimulate anchor is %v", sw.Scenario, sw.Label, theirAnchor))
					bad = true
					break
				}
			}
			predicted := obs.MeanITLUs / anchor
			errMine := math.Abs(predicted/p.MeasuredRelative-1) * 100
			errTheirs := math.Abs((p.AISimulateRelative/theirAnchor)/p.MeasuredRelative-1) * 100
			mine = append(mine, errMine)
			theirs = append(theirs, errTheirs)
			if *verbose {
				if i == 0 {
					fmt.Printf("--- %s %s %s %s tp=%d\n", sw.Scenario, sw.Label,
						sw.Framework, sw.Precision, sw.Parallelism["tp_size"])
					fmt.Printf("%6s %10s %10s %8s %10s %8s %8s\n",
						"conc", "measured", "blis", "error", "aisim", "error", "resident")
				}
				fmt.Printf("%6d %10.4f %10.4f %7.2f%% %10.4f %7.2f%% %8.1f\n",
					p.Concurrency, p.MeasuredRelative, predicted, errMine,
					p.AISimulateRelative/theirAnchor, errTheirs, obs.MeanResident)
			}
		}
		if bad || len(mine) == 0 {
			continue
		}
		rows = append(rows, row{sw, mine, theirs})
		allMine = append(allMine, mine...)
		allTheirs = append(allTheirs, theirs...)
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
	fmt.Printf("%-30s %5s %8s %8s %8s\n", "model", "n", "MAPE", "median", "worst")
	fmt.Printf("%-30s %5d %7.2f%% %7.2f%% %7.2f%%\n", "BLIS + blis-latency-kernel",
		len(allMine), harness.MAPE2(allMine), harness.Median(allMine), maxOf(allMine))
	fmt.Printf("%-30s %5d %7.2f%% %7.2f%% %7.2f%%\n", "AISimulate (same points)",
		len(allTheirs), harness.MAPE2(allTheirs), harness.Median(allTheirs), maxOf(allTheirs))

	fmt.Printf("\nAISimulate publishes %.2f%% TPOT shape error over its whole %d-point\n"+
		"snapshot. The row above is the check that this harness measures that same\n"+
		"quantity: recomputed from its own re-anchored per-point data, it must land near\n"+
		"the published figure.\n",
		c.AISimulateTotals.TPOTShapeErrorPct, c.AISimulateTotals.Points)

	fmt.Printf("\nBy framework (BLIS models vLLM; the others are context, not a claim):\n")
	fmt.Printf("%-10s %5s %9s %9s\n", "framework", "n", "BLIS", "AISimulate")
	for _, f := range sortedKeys(byFramework) {
		fmt.Printf("%-10s %5d %8.2f%% %8.2f%%\n", f, len(byFramework[f]),
			harness.MAPE2(byFramework[f]), harness.MAPE2(byFrameworkTheirs[f]))
	}

	fmt.Printf("\nBy chip family:\n")
	fmt.Printf("%-10s %5s %9s %9s\n", "family", "n", "BLIS", "AISimulate")
	for _, k := range sortedKeys(byChip) {
		fmt.Printf("%-10s %5d %8.2f%% %8.2f%%\n", k, len(byChip[k]),
			harness.MAPE2(byChip[k]), harness.MAPE2(byChipTheirs[k]))
	}

	if len(failures) > 0 {
		fmt.Printf("\n%d point(s) could not be scored:\n", len(failures))
		for _, f := range failures {
			fmt.Printf("  %s\n", f)
		}
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

func maxOf(xs []float64) float64 {
	m := 0.0
	for _, x := range xs {
		if x > m {
			m = x
		}
	}
	return m
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

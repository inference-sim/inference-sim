// Command estimatorscore scores FOUR estimators on one subset: blis-latency-kernel, NVIDIA's
// AISimulate, and BLIS's own roofline and trained-physics backends.
//
// # Why this is Hopper-only
//
// Roofline reads mfuPrefill/mfuDecode from hardware_config.json, which carries H100, H200,
// A100-SXM, A100-80 and L40S. There is no Blackwell entry, and blis-registry has no
// roofline-b200.yaml or roofline-b300.yaml. Scoring roofline on b200 would mean inventing a
// calibration input to fill a column, so the subset is the chips it is actually calibrated for.
// The kernel-against-AISimulate headline in cmd/kernelscore covers every chip and is unaffected.
//
// # What differs between arms, and what does not
//
// Step time, and nothing else. The KV budget comes from the KERNEL's memory methods for every
// arm, as do the engine settings and the dp width, so the resident batch is decided identically
// and the comparison cannot be contaminated by admission behaviour. The host per-token cost is
// the kernel's for every arm too: the metric is mean inter-token latency, and the registry
// already carries the same 45.9 us/token magnitude the trained-physics fit produced.
//
// Error is defined exactly as cmd/kernelscore defines it: each side normalised to its OWN
// value at the sweep's lowest concurrency, with the anchor excluded from the mean.
//
// Usage:
//
//	go run ./cmd/estimatorscore
//	go run ./cmd/estimatorscore -verbose
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

// hopperChips are the parts every arm in this comparison is calibrated for.
var hopperChips = map[string]bool{"h100": true, "h200": true}

func main() {
	corpusPath := flag.String("corpus",
		kernelmodel.MeasurementPath("aisimulate_e2e.json"),
		"the extracted AISimulate corpus")
	scenarios := flag.String("scenarios",
		kernelmodel.DefaultScenarios(), "scenario directory")
	catalog := flag.String("catalog", kernelmodel.DefaultCatalog(), "")
	registry := flag.String("registry", kernelmodel.DefaultRegistry(), "")
	hwConfig := flag.String("hardware-config", "hardware_config.json",
		"roofline/trained-physics hardware calibration")
	defaults := flag.String("defaults", "defaults.yaml", "trained-physics coefficients")
	seed := flag.Int64("seed", 42, "")
	verbose := flag.Bool("verbose", false, "print every point")
	curves := flag.String("curves", "",
		"instead of scoring, print each arm's predicted curve and admission facts for one "+
			"scenario (e.g. gpt-oss-120b-h200-fp4-vllm-tp4.yaml)")
	flag.Parse()
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
		Admission: harness.AdmissionKernelKV,
		Seed:      *seed,
		Backends: harness.BackendPaths{
			Catalog: *catalog, HWConfig: *hwConfig, Defaults: *defaults,
		},
	}

	fmt.Println("Four estimators on the Hopper vLLM subset")
	fmt.Printf("corpus: %s\n", c.Source)
	fmt.Printf("ground truth: %s\n\n", c.MeasurementSource)
	fmt.Println("Subset: framework vllm, chips h100 and h200. Roofline has no Blackwell")
	fmt.Println("  calibration (hardware_config.json carries H100/H200/A100/L40S only), so a")
	fmt.Println("  Blackwell column would require inventing mfuPrefill/mfuDecode.")
	fmt.Println("Held fixed across arms: the KV budget and engine settings come from the")
	fmt.Println("  kernel for every arm, as does the 45.9 us/token host cost. Only the")
	fmt.Println("  forward-pass model differs.")
	fmt.Println("Activation precision: gpt-oss-120b (mxfp4) and minimax-m2.5 (fp8) declare no")
	fmt.Println("  torch_dtype/dtype. vLLM resolves that case to the platform's first")
	fmt.Println("  supported dtype -- bfloat16 on capability >= 80 -- so the analytic arms")
	fmt.Println("  use 2 bytes, which is what the engine being modelled runs.")
	fmt.Println()

	if *curves != "" {
		if err := printCurves(c, base, *curves); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}

	// Signed per-point errors, per arm. Positive means the arm predicted a larger relative
	// rise in time per output token than was measured.
	errs := map[harness.Estimator][]float64{}
	var theirs []float64
	byScenario := map[string]map[harness.Estimator][]float64{}
	failures := []string{}
	sweeps := 0
	points := 0

	for i := range c.Sweeps {
		sw := c.Sweeps[i]
		if sw.Framework != "vllm" || !hopperChips[family(sw.GPU)] {
			continue
		}
		// Every arm must score the SAME points, or the columns are not comparable. An arm
		// that fails on any point disqualifies the sweep for all of them.
		perArm := map[harness.Estimator][]float64{}
		var theirsHere []float64
		ok := true
		for _, e := range harness.AllEstimators {
			cfg := base
			cfg.Estimator = e
			var anchor, theirAnchor float64
			for j, p := range sw.Points {
				obs, err := harness.Run(sw, p.Concurrency, cfg)
				if err != nil {
					failures = append(failures,
						fmt.Sprintf("%s %s %s c=%d: %v", sw.Scenario, sw.Label, e, p.Concurrency, err))
					ok = false
					break
				}
				if obs.MeanITLUs <= 0 {
					failures = append(failures, fmt.Sprintf(
						"%s %s %s c=%d: no measurement", sw.Scenario, sw.Label, e, p.Concurrency))
					ok = false
					break
				}
				if j == 0 {
					anchor = obs.MeanITLUs
					theirAnchor = p.AISimulateRelative
					if theirAnchor <= 0 {
						ok = false
						break
					}
					continue // the anchor's error is zero by construction; excluded
				}
				perArm[e] = append(perArm[e], (obs.MeanITLUs/anchor/p.MeasuredRelative-1)*100)
				if e == harness.EstimatorKernel {
					theirsHere = append(theirsHere,
						((p.AISimulateRelative/theirAnchor)/p.MeasuredRelative-1)*100)
				}
				if *verbose && e == harness.EstimatorKernel {
					fmt.Printf("  %-44s %-5s c=%-5d measured %.4f\n",
						sw.Scenario, sw.Label, p.Concurrency, p.MeasuredRelative)
				}
			}
			if !ok {
				break
			}
		}
		if !ok {
			continue
		}
		sweeps++
		points += len(perArm[harness.EstimatorKernel])
		theirs = append(theirs, theirsHere...)
		key := strings.TrimSuffix(sw.Scenario, ".yaml") + " " + sw.Label
		byScenario[key] = map[harness.Estimator][]float64{}
		for _, e := range harness.AllEstimators {
			errs[e] = append(errs[e], perArm[e]...)
			byScenario[key][e] = perArm[e]
		}
	}

	if points == 0 {
		fmt.Fprintln(os.Stderr, "no points scored")
		for _, f := range failures {
			fmt.Fprintf(os.Stderr, "  %s\n", f)
		}
		os.Exit(1)
	}

	fmt.Printf("%d sweeps, %d points\n\n", sweeps, points)
	fmt.Printf("%-24s %5s %9s %9s %9s %7s %9s %9s\n",
		"estimator", "n", "mean", "median", "mean|e|", "over", "p10", "p90")
	report := func(name string, v []float64) {
		st := harness.Signed(v)
		fmt.Printf("%-24s %5d %+8.2f%% %+8.2f%% %8.2f%% %6.0f%% %+8.2f%% %+8.2f%%\n",
			name, st.N, st.Mean, st.Median, st.MeanAbs, 100*st.FractionOver, st.P10, st.P90)
	}
	report("blis-latency-kernel", errs[harness.EstimatorKernel])
	report("AISimulate", theirs)
	report("roofline", errs[harness.EstimatorRoofline])
	report("trained-physics", errs[harness.EstimatorTrainedPhysics])

	fmt.Printf("\nBy scenario (mean|e|):\n")
	fmt.Printf("%-50s %5s %9s %9s %9s %9s\n",
		"scenario", "n", "kernel", "aisim", "roofline", "trained")
	keys := make([]string, 0, len(byScenario))
	for k := range byScenario {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		m := byScenario[k]
		fmt.Printf("%-50s %5d %8.2f%% %8s %8.2f%% %8.2f%%\n", k,
			len(m[harness.EstimatorKernel]),
			harness.MAPE2(harness.Abs(m[harness.EstimatorKernel])), "--",
			harness.MAPE2(harness.Abs(m[harness.EstimatorRoofline])),
			harness.MAPE2(harness.Abs(m[harness.EstimatorTrainedPhysics])))
	}

	if len(failures) > 0 {
		fmt.Printf("\n%d point(s) could not be scored:\n", len(failures))
		for _, f := range failures {
			fmt.Printf("  %s\n", f)
		}
	}
}

// printCurves prints each arm's predicted curve for one scenario, alongside the admission facts
// the arms must share. It exists because a mean error cannot distinguish a bad model from a broken
// harness: only the shape of the curve, and the evidence that every arm saw the same KV pool and
// completed the same requests, can.
func printCurves(c *harness.Corpus, base harness.Config, scenario string) error {
	for i := range c.Sweeps {
		sw := c.Sweeps[i]
		if sw.Scenario != scenario || sw.Framework != "vllm" {
			continue
		}
		type armRun struct {
			e      harness.Estimator
			raw    []float64
			blocks []int64
			done   []int
		}
		var arms []*armRun
		for _, e := range harness.AllEstimators {
			a := &armRun{e: e}
			cfg := base
			cfg.Estimator = e
			for _, p := range sw.Points {
				obs, err := harness.Run(sw, p.Concurrency, cfg)
				if err != nil {
					return fmt.Errorf("%s %s c=%d: %w", scenario, e, p.Concurrency, err)
				}
				a.raw = append(a.raw, obs.MeanITLUs)
				a.blocks = append(a.blocks, obs.KVBlocks)
				a.done = append(a.done, obs.Completed)
			}
			arms = append(arms, a)
		}

		fmt.Printf("=== %s %s\n\n", sw.Scenario, sw.Label)
		fmt.Println("Predicted curve, each arm normalised to its OWN value at the lowest")
		fmt.Println("concurrency -- the same quantity the score compares:")
		fmt.Printf("\n%12s %10s", "concurrency", "measured")
		for _, a := range arms {
			fmt.Printf(" %16s", a.e)
		}
		fmt.Println()
		for j, p := range sw.Points {
			fmt.Printf("%12d %10.4f", p.Concurrency, p.MeasuredRelative)
			for _, a := range arms {
				fmt.Printf(" %16.4f", a.raw[j]/a.raw[0])
			}
			fmt.Println()
		}

		fmt.Println("\nAbsolute mean inter-token latency (us), which is what explains the shapes:")
		fmt.Printf("\n%-18s %12s %12s\n", "arm", "anchor", "highest")
		for _, a := range arms {
			fmt.Printf("%-18s %12.1f %12.1f\n", a.e, a.raw[0], a.raw[len(a.raw)-1])
		}

		fmt.Println("\nAdmission, which every arm must share for this to be a step-time")
		fmt.Println("comparison at all:")
		fmt.Printf("\n%12s %12s %12s   %s\n", "concurrency", "KV blocks", "completed", "identical across arms")
		for j, p := range sw.Points {
			same := true
			for _, a := range arms {
				if a.blocks[j] != arms[0].blocks[j] || a.done[j] != arms[0].done[j] {
					same = false
				}
			}
			mark := "yes"
			if !same {
				mark = "NO -- the comparison is contaminated"
			}
			fmt.Printf("%12d %12d %12d   %s\n",
				p.Concurrency, arms[0].blocks[j], arms[0].done[j], mark)
		}
		return nil
	}
	return fmt.Errorf("scenario %q not found in the vllm corpus", scenario)
}

// family maps a GPU name to its architecture family.
func family(gpu string) string {
	g := strings.ToLower(gpu)
	switch {
	case strings.Contains(g, "h100"):
		return "h100"
	case strings.Contains(g, "h200"):
		return "h200"
	case strings.Contains(g, "b200"):
		return "b200"
	case strings.Contains(g, "b300"):
		return "b300"
	}
	return g
}

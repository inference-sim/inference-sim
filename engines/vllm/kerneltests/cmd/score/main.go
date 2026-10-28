// Moved from blis-latency-kernel/cmd/score/main.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

// Command score reports how well the kernel predicts published measurements.
//
// The governing requirement is that accuracy is reported rather than asserted. This scores
// predicted inter-token latency against every operating point in the report corpus that a
// step-time model can be held to, and prints the error per point, per model, and in
// aggregate — with the conditions and the gaps stated rather than left implicit.
//
// # What is scored
//
// ITL, and only ITL. At steady-state decode one step serves one token per resident request,
// so the interval between a request's tokens is one step plus the host's per-token work.
// That is nearly a direct reading of a step time.
//
// TTFT is not scored: it is dominated by queue waiting, which belongs to a scheduler this
// kernel deliberately does not model. Throughput is not scored for the same reason — it is
// what a scheduler achieves given step times, not a step time.
//
// # What the corpus supplies, and what it does not
//
// The reports were written to compare deployments, not to calibrate a cost model, so no
// report states everything a prediction needs. They divide by what they omit:
//
//	The Nemotron sweep states the resident batch (`running`) and the preemption count but
//	no input length, so context is derived from the reported input and output token rates.
//
//	The Kimi sweep states neither directly; it gives effective concurrency, which is the
//	resident batch, and its arm label states the context bound.
//
// Every row records which of its inputs was stated and which was derived, and the report
// below marks the derived ones. A point whose batch or context had to be derived is scored
// and flagged, not dropped: the alternative is scoring one model.
//
// # Usage
//
//	go run ./cmd/score
//	go run ./cmd/score -verbose
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-vllm/kerneltests/internal/harness"
)

// point is one operating point, as extracted from a report.
type point struct {
	Scenario    string   `json:"scenario"`
	Report      string   `json:"report"`
	Series      string   `json:"series"`
	Label       string   `json:"label"`
	Concurrency int      `json:"concurrency"`
	ITLms       float64  `json:"itl_ms"`
	TTFTms      *float64 `json:"ttft_ms"`
	ISL         *float64 `json:"isl"`
	OSL         *float64 `json:"osl"`
	KVPct       *float64 `json:"kv_pct"`
	HitPct      *float64 `json:"hit_pct"`
	Preempted   *float64 `json:"preempted"`
	Running     *float64 `json:"running"`
	Waiting     *float64 `json:"waiting"`
	InTPS       *float64 `json:"in_tps"`
	OutTPS      *float64 `json:"out_tps"`
	AcceptPct   *float64 `json:"accept_pct"`
	EffConc     *float64 `json:"effconc"`
	E2Ems       *float64 `json:"e2e_ms"`
}

// resolved is a point with its batch and context determined, and the derivation recorded.
type resolved struct {
	point
	batch         int
	context       int
	batchStated   bool
	contextStated bool
	// replicas is how many engine instances the reported figures are summed across. A
	// per-step batch is the resident total divided by it.
	replicas int
	inScope  bool
	why      string
}

func main() {
	catalog := flag.String("catalog", harness.DefaultCatalog(), "")
	registry := flag.String("registry", harness.DefaultRegistry(), "")
	data := flag.String("data",
		filepath.Join(harness.DefaultMeasurements(), "scoreable.json"), "")
	verbose := flag.Bool("verbose", false, "print every point, not only a summary")
	flag.Parse()

	raw, err := os.ReadFile(*data)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\nThe measurement corpora are not distributed with this "+
			"repository: regenerate them from their sources with scripts/extract_*, or point "+
			"-data or BLIS_MEASUREMENTS at a copy you hold.\n", err)
		os.Exit(1)
	}
	var points []point
	if err := json.Unmarshal(raw, &points); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	kernels := map[string]*latencykernel.Kernel{}
	scenarios := map[string]int{} // replica count per scenario
	for _, p := range points {
		if _, ok := kernels[p.Scenario]; ok {
			continue
		}
		repos := harness.Repos{
			Scenarios: "testdata", Catalog: *catalog, Registry: *registry,
		}
		k, err := harness.Open(p.Scenario, repos)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", p.Scenario, err)
			os.Exit(1)
		}
		replicas, err := harness.Replicas(p.Scenario, repos)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", p.Scenario, err)
			os.Exit(1)
		}
		kernels[p.Scenario] = k
		scenarios[p.Scenario] = replicas
	}

	byScenario := map[string][]float64{}
	var all []float64
	rows := make([]resolved, 0, len(points))
	for _, p := range points {
		r := resolve(p, scenarios[p.Scenario])
		rows = append(rows, r)
		if !r.inScope {
			continue
		}
		k := kernels[p.Scenario]
		predicted := predict(k, r)
		relative := math.Abs(predicted/p.ITLms - 1)
		all = append(all, relative)
		byScenario[p.Scenario] = append(byScenario[p.Scenario], relative)
	}

	fmt.Println("Predicted inter-token latency against the published report corpus")
	fmt.Printf("%d operating points from %d reports, %d scenarios, %d models\n\n",
		len(points), countDistinct(points, func(p point) string { return p.Report }),
		len(kernels),
		countDistinct(points, func(p point) string { return p.Scenario }))

	if *verbose {
		fmt.Printf("%-30s %5s %9s %10s %8s %9s  %s\n",
			"scenario / arm", "conc", "measured", "predicted", "error", "bound by", "note")
		for _, r := range rows {
			if !r.inScope {
				fmt.Printf("%-30s %5d %8.2fms %10s %8s %9s  %s\n",
					short(r.Scenario), r.Concurrency, r.ITLms, "-", "-", "-", r.why)
				continue
			}
			k := kernels[r.Scenario]
			predicted := predict(k, r)
			e := k.StepTime(harness.DecodeBatch(r.batch, r.context))
			fmt.Printf("%-30s %5d %8.2fms %9.2fms %+7.1f%% %9s  %s\n",
				short(r.Scenario), r.Concurrency, r.ITLms, predicted,
				(predicted/r.ITLms-1)*100, e.Bottleneck, r.why)
		}
		fmt.Println()
	}

	fmt.Printf("%-34s %4s %7s %8s %7s\n", "scenario", "n", "MAPE", "median", "worst")
	names := make([]string, 0, len(byScenario))
	for n := range byScenario {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		report(short(n), byScenario[n])
	}
	fmt.Println()
	report("ALL IN-SCOPE POINTS", all)
	skipped := 0
	for _, r := range rows {
		if !r.inScope {
			skipped++
		}
	}
	fmt.Printf("%-34s %4d  (reasons above with -verbose)\n", "out of scope", skipped)

	fmt.Println(`
Conditions, and what the corpus does not state:

  * NO report states gpu_memory_utilization, so the KV budget each engine actually had
    is unknown. Every scenario assumes 0.9. This does not enter an ITL prediction
    directly; it decides which concurrencies were feasible at all.

  * The Nemotron sweep states the resident batch and zero preemptions but no input
    length, so context is derived from the input and output token rates. A derived
    context is marked.

  * The Kimi sweep states neither; effective concurrency stands in for the batch and the
    arm label states the 256k context bound. Both are marked as derived.

  * Speculative arms are scored at their reported acceptance rate where one is given. An
    accepted draft token costs no extra step, so a step serves more than one token per
    request and ITL falls below the step time; the prediction divides by the mean tokens
    per step for that reason.

  * DeepSeek-V4.1-Flash has two reports in the corpus and no model in the catalog, so its
    26 points cannot be scored. GLM-5.3's reports carry no per-level latency series.`)
}

// resolve determines a point's batch and context, and whether it is in scope.
func resolve(p point, replicas int) resolved {
	r := resolved{point: p, replicas: replicas}
	if r.replicas < 1 {
		r.replicas = 1
	}

	// The resident batch. `running` is the engine's own count, summed across replicas, so
	// a per-step batch divides by the replica count. Falling back to client concurrency is
	// only valid while the queue is empty.
	switch {
	case p.Running != nil && *p.Running > 0:
		r.batch = int(math.Round(*p.Running / float64(r.replicas)))
		r.batchStated = true
	case p.EffConc != nil && *p.EffConc > 0:
		// Effective concurrency is "requests genuinely in flight", which is the resident
		// batch by another name — and unlike the client level it already excludes what is
		// waiting. Also summed across replicas.
		r.batch = int(math.Round(*p.EffConc / float64(r.replicas)))
		r.batchStated = true
	case p.Concurrency > 0:
		r.batch = p.Concurrency / r.replicas
	}
	if r.batch < 1 {
		r.batch = 1
	}

	// The context each request carries: its input plus roughly half its output, since a
	// request grows through its decode phase.
	switch {
	case p.ISL != nil && *p.ISL > 0:
		out := 0.0
		if p.OSL != nil {
			out = *p.OSL
		}
		r.context = int(*p.ISL + out/2)
		r.contextStated = true
	case p.InTPS != nil && p.OutTPS != nil && *p.OutTPS > 0 &&
		p.EffConc != nil && *p.EffConc > 0 && p.E2Ems != nil && *p.E2Ems > 0:
		// Neither length is stated, but both are derivable from rates the report does give,
		// without assuming either.
		//
		// The per-request output rate is outtps / effconc. Multiplied by the end-to-end
		// latency that gives the mean output length; scaled by the input:output token-rate
		// ratio it gives the mean input length. On the Nemotron sweep this yields inputs of
		// 36k to 51k tokens and outputs of 350 to 660 — a long-context agentic workload,
		// which a nominal few-hundred-token assumption would have missed by two orders of
		// magnitude.
		perRequestOut := *p.OutTPS / *p.EffConc
		outLen := perRequestOut * *p.E2Ems / 1000
		inLen := outLen * (*p.InTPS / *p.OutTPS)
		r.context = int(inLen + outLen/2)
	}

	switch {
	case r.context <= 0:
		r.why = "no context stated or derivable"
	case p.Preempted != nil && *p.Preempted > 0:
		r.why = fmt.Sprintf("%.0f preemptions: scheduler-bound", *p.Preempted)
	case p.Waiting != nil && *p.Waiting > 1:
		r.why = fmt.Sprintf("%.1f queued: prefill shares these steps", *p.Waiting)
	case !r.batchStated && p.TTFTms != nil && *p.TTFTms > 200:
		// No stated batch AND a TTFT well past its flat band means requests are queueing
		// and their prefill chunks share decode steps. Client concurrency then overstates
		// the resident batch and measured ITL is not a step time.
		r.why = fmt.Sprintf("TTFT %.0fms with no stated batch: prefill shares these steps",
			*p.TTFTms)
	default:
		r.inScope = true
		var marks string
		if !r.batchStated {
			marks += " batch=concurrency"
		}
		if !r.contextStated {
			marks += " ctx derived"
		}
		r.why = marks
	}
	return r
}

func predict(k *latencykernel.Kernel, r resolved) float64 {
	e := k.StepTime(harness.DecodeBatch(r.batch, r.context))
	// Host per-token work runs off the forward pass but lands on the interval anyway.
	// NoOverlap is the edge the measured evidence selects -- see harness.TimePerOutputToken
	// and blis-registry's docs/band-selection.md.
	step := (e.NoOverlap + k.OutputTokenOverhead()).Seconds() * 1e3
	// With speculation, an accepted draft yields a token without its own step, so the
	// observed interval is the step divided by the mean tokens it produced.
	if r.AcceptPct != nil && *r.AcceptPct > 0 {
		perStep := 1 + *r.AcceptPct/100
		return step / perStep
	}
	return step
}

func report(label string, errs []float64) {
	if len(errs) == 0 {
		fmt.Printf("%-34s %4d  no points in scope\n", label, 0)
		return
	}
	sorted := append([]float64(nil), errs...)
	sort.Float64s(sorted)
	var sum float64
	for _, e := range sorted {
		sum += e
	}
	fmt.Printf("%-34s %4d %6.1f%% %7.1f%% %6.1f%%\n", label, len(sorted),
		sum/float64(len(sorted))*100, sorted[len(sorted)/2]*100, sorted[len(sorted)-1]*100)
}

func short(s string) string {
	return s[:len(s)-len(filepath.Ext(s))]
}

func countDistinct[T any](xs []T, key func(T) string) int {
	seen := map[string]bool{}
	for _, x := range xs {
		seen[key(x)] = true
	}
	return len(seen)
}

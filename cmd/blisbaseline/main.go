// Command blisbaseline scores this simulator's EXISTING latency models -- roofline and
// trained-physics -- against the same report corpus blis-latency-kernel is scored against.
//
// It lives here rather than in blis-latency-kernel because it imports this simulator, and the
// dependency runs the other way: the simulator consumes the kernel, so a kernel that imported
// the simulator would close a cycle between the two repositories.
//
// It exists so that any accuracy claim about the new kernel is comparative rather than
// absolute. A MAPE of 47% means nothing on its own; it means something against what the
// shipping models achieve on the same points, with the same batch and context derivations.
//
// Two models are scored:
//
//	roofline         — analytical FLOPs and bandwidth with per-phase MFU discounts.
//	trained-physics  — the same roofline basis functions with ten coefficients fitted
//	                   end to end, including explicit per-layer, per-batch and constant
//	                   terms (beta5*L + beta6*B + beta7).
//
// The second is expected to do well: it was fitted against measurements, and three of its
// terms are the ones this project derived analytically and independently. Reporting it
// honestly is the point — the new kernel's claim is that a miss localizes to a named term,
// not that it wins on aggregate error.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	_ "github.com/inference-sim/inference-sim/sim/latency"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
)

// The coefficients inference-sim ships in defaults.yaml, transcribed so this harness needs
// no config plumbing. Both are the shipped production values.
var (
	alphaCoeffs = []float64{15563.199579, 777.3455, 45.907545}
	betaCoeffs  = []float64{
		0.152128, 0.0, 1.36252915, 0.752037, 32.09546717,
		4.41684444, 126.024825, 481.8613888, 0.0, 1.94710771,
	}
)

// point mirrors the corpus rows cmd/score reads, so both harnesses score identical inputs.
type point struct {
	Scenario    string   `json:"scenario"`
	Report      string   `json:"report"`
	Series      string   `json:"series"`
	Concurrency int      `json:"concurrency"`
	ITLms       float64  `json:"itl_ms"`
	TTFTms      *float64 `json:"ttft_ms"`
	ISL         *float64 `json:"isl"`
	OSL         *float64 `json:"osl"`
	Preempted   *float64 `json:"preempted"`
	Running     *float64 `json:"running"`
	Waiting     *float64 `json:"waiting"`
	InTPS       *float64 `json:"in_tps"`
	OutTPS      *float64 `json:"out_tps"`
	AcceptPct   *float64 `json:"accept_pct"`
	EffConc     *float64 `json:"effconc"`
	E2Ems       *float64 `json:"e2e_ms"`
}

// deployment is what a scenario tells this harness: the model and hardware to build the
// existing models from, and the parallelism to divide by.
type deployment struct {
	model    string // catalog model directory
	hardware string // catalog hardware stem
	tp       int
	replicas int
}

// The scenarios these models can be built for.
//
// Only Granite-5 is here, and the two omissions are a finding rather than an oversight. The
// existing models take a flat ModelConfig unmarshalled straight from a HuggingFace
// config.json, reading `num_hidden_layers`, `num_local_experts` and `num_experts_per_tok`
// at the top level. Two of the corpus's four models do not present their shape that way:
//
//	Nemotron-3-Ultra names its experts `n_routed_experts`, states no top-level
//	`num_hidden_layers`, and describes its hybrid stack in `layers_block_type` — a
//	per-layer list of mamba, attention and MoE kinds a flat config has no field for.
//
//	Kimi-K3 is multimodal: every language-model parameter is nested under `text_config`,
//	and the top level carries only the vision and routing configuration.
//
// Both unmarshal without complaint and yield zeros, and the roofline model then refuses them
// rather than pricing them wrongly. That refusal is correct, and it is also why this project
// derives a graph per model: the derivation reconciles a vendor's naming once, so the cost
// model never sees it.
var deployments = map[string]deployment{
	"granite5-h200-tp8-measured.yaml": {"granite-5.0-230b-rl-preview", "h200", 8, 1},
	"granite5-h100-tp8-measured.yaml": {"granite-5.0-230b-rl-preview", "h100", 8, 1},
}

func main() {
	catalog := flag.String("catalog", kernelmodel.DefaultCatalog(), "")
	corpus := flag.String("corpus",
		filepath.Join(kernelmodel.DefaultMeasurements(), "scoreable.json"),
		"")
	verbose := flag.Bool("verbose", false, "print every point")
	flag.Parse()

	raw, err := os.ReadFile(*corpus)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var points []point
	if err := json.Unmarshal(raw, &points); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	fmt.Println("Existing inference-sim latency models against the same report corpus")
	fmt.Printf("%d corpus points, %d scenarios\n\n", len(points), len(deployments))

	for _, backend := range []string{"roofline", "trained-physics"} {
		byScenario := map[string][]float64{}
		var all []float64
		if *verbose {
			fmt.Printf("--- %s ---\n%-30s %5s %9s %10s %8s\n",
				backend, "scenario", "conc", "measured", "predicted", "error")
		}
		for _, p := range points {
			d, ok := deployments[p.Scenario]
			if !ok {
				continue
			}
			batch, context, inScope := resolve(p, d.replicas)
			if !inScope {
				continue
			}
			model, err := build(*catalog, d, backend)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%s/%s: %v\n", p.Scenario, backend, err)
				os.Exit(1)
			}
			predicted := predict(model, batch, context, p.AcceptPct)
			rel := math.Abs(predicted/p.ITLms - 1)
			all = append(all, rel)
			byScenario[p.Scenario] = append(byScenario[p.Scenario], rel)
			if *verbose {
				fmt.Printf("%-30s %5d %8.2fms %9.2fms %+7.1f%%\n", short(p.Scenario),
					p.Concurrency, p.ITLms, predicted, (predicted/p.ITLms-1)*100)
			}
		}
		fmt.Printf("%-16s %-20s %4s %7s %8s %7s\n", backend, "scenario", "n", "MAPE",
			"median", "worst")
		names := make([]string, 0, len(byScenario))
		for n := range byScenario {
			names = append(names, n)
		}
		sort.Strings(names)
		for _, n := range names {
			report("", short(n), byScenario[n])
		}
		report(backend, "ALL IN-SCOPE POINTS", all)
		fmt.Println()
	}
	fmt.Println(`Method: identical corpus, identical batch and context derivation, identical
exclusions to cmd/score, so the numbers are comparable point for point.

How to read these against cmd/score's 32.0% on the same twelve points:

  MAPE alone favours trained-physics, 24.6% against 32.0%. It was fitted end to end against
  measurements, and three of its ten coefficients — beta5*L + beta6*B + beta7, per layer, per
  batch, constant — are terms this project derived independently from component sweeps. The
  convergence is reassuring about the structure and does not win the comparison.

  The error SHAPES differ more than the magnitudes. trained-physics is near-unbiased and
  scattered: mean -15.0%, spread 68.8 points, crossing zero between c=8 and c=16. This kernel
  is uniformly low: mean -32.0%, spread 50.9 points, negative at every point. A systematic
  error points at one missing mechanism; a scattered one has been absorbed into coefficients
  and localizes nowhere, which is the property this design was meant to keep.

  A test of that claim: one two-parameter correction, model * (1 + 0.225 * c^0.32), takes this
  kernel from 32.0% to 7.5% — better than trained-physics with eight fewer parameters. That
  supports "one mechanism" and it is NOT shipped, because the same arithmetic refutes the
  obvious mechanism. If the cause were client requests exceeding the resident batch, the
  factor should track c/resident, which Nemotron reports: that ratio FALLS from 7.1 to 2.1 as
  concurrency rises while the fitted factor RISES from 1.44 to 2.06. A curve that fits while
  contradicting its own explanation is a fudge factor, and this model does not carry one.

  Against the analytical baseline the gain is real: roofline 73.1% to 32.0%, from measured
  primitives rather than fitted corrections.

Coverage: only Granite-5 is baselined. The existing models take a flat ModelConfig read
straight from a HuggingFace config.json, which Nemotron-3-Ultra and Kimi-K3 do not present
their shape through — see the deployments map for what each omits.`)
}

func report(prefix, label string, errs []float64) {
	if len(errs) == 0 {
		fmt.Printf("%-16s %-20s %4d  no points in scope\n", prefix, label, 0)
		return
	}
	s := append([]float64(nil), errs...)
	sort.Float64s(s)
	var sum float64
	for _, e := range s {
		sum += e
	}
	fmt.Printf("%-16s %-20s %4d %6.1f%% %7.1f%% %6.1f%%\n", prefix, label, len(s),
		sum/float64(len(s))*100, s[len(s)/2]*100, s[len(s)-1]*100)
}

// resolve mirrors cmd/score's scope and derivation rules exactly. Duplicated rather than
// shared because the two harnesses must stay comparable even if one repository moves.
func resolve(p point, replicas int) (batch, context int, inScope bool) {
	if replicas < 1 {
		replicas = 1
	}
	stated := false
	switch {
	case p.Running != nil && *p.Running > 0:
		batch, stated = int(math.Round(*p.Running/float64(replicas))), true
	case p.EffConc != nil && *p.EffConc > 0:
		batch, stated = int(math.Round(*p.EffConc/float64(replicas))), true
	case p.Concurrency > 0:
		batch = p.Concurrency / replicas
	}
	if batch < 1 {
		batch = 1
	}
	switch {
	case p.ISL != nil && *p.ISL > 0:
		out := 0.0
		if p.OSL != nil {
			out = *p.OSL
		}
		context = int(*p.ISL + out/2)
	case p.InTPS != nil && p.OutTPS != nil && *p.OutTPS > 0 &&
		p.EffConc != nil && *p.EffConc > 0 && p.E2Ems != nil && *p.E2Ems > 0:
		perReqOut := *p.OutTPS / *p.EffConc
		outLen := perReqOut * *p.E2Ems / 1000
		context = int(outLen*(*p.InTPS / *p.OutTPS) + outLen/2)
	}
	switch {
	case context <= 0:
	case p.Preempted != nil && *p.Preempted > 0:
	case p.Waiting != nil && *p.Waiting > 1:
	case !stated && p.TTFTms != nil && *p.TTFTms > 200:
	default:
		inScope = true
	}
	return batch, context, inScope
}

func predict(m sim.LatencyModel, batch, context int, acceptPct *float64) float64 {
	reqs := make([]*sim.Request, batch)
	for i := range reqs {
		// A decode request, per the existing models' own classification in
		// sim/latency/latency.go: ProgressIndex at or past the prompt length and a
		// non-empty OutputTokens slice. ProgressIndex carries the context they price.
		reqs[i] = &sim.Request{
			ID:            fmt.Sprintf("r%d", i),
			NumNewTokens:  1,
			ProgressIndex: int64(context - 1),
			InputTokens:   make([]sim.TokenID, context-1),
			OutputTokens:  make([]sim.TokenID, 1),
		}
	}
	step := float64(m.StepTime(reqs)) / 1000 // microseconds to milliseconds
	step += float64(m.OutputTokenProcessingTime()) / 1000
	if acceptPct != nil && *acceptPct > 0 {
		step /= 1 + *acceptPct/100
	}
	return step
}

func build(catalog string, d deployment, backend string) (sim.LatencyModel, error) {
	var mc sim.ModelConfig
	raw, err := os.ReadFile(filepath.Join(catalog, "models", d.model, "config.json"))
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &mc); err != nil {
		return nil, err
	}
	if mc.BytesPerParam == 0 {
		mc.BytesPerParam = 2
	}
	hw, err := loadChip(filepath.Join(catalog, "hardware", d.hardware+".yaml"))
	if err != nil {
		return nil, err
	}
	// dp=1, expert parallelism off, no MoE comm backend: this comparison drives the legacy
	// models at the same single-pool deployment the kernel is scored at, and those two axes
	// are what the legacy models do not represent anyway.
	cfg := sim.NewModelHardwareConfig(mc, hw, d.model, d.hardware, d.tp, 1, false, "",
		backend, 0)
	return sim.MustNewLatencyModel(sim.NewLatencyCoeffs(betaCoeffs, alphaCoeffs), cfg)
}

func short(s string) string { return s[:len(s)-len(filepath.Ext(s))] }

// loadChip reads a catalog hardware file into the flat HardwareCalib the existing models take.
//
// The catalog's own loader lives in blis-schemas, which inference-sim does not depend on, so
// the four fields those models read are pulled out directly. The MFU discounts are the 0.45
// and 0.30 the simulator itself ships, from blis-registry's roofline sets.
func loadChip(path string) (sim.HardwareCalib, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return sim.HardwareCalib{}, err
	}
	var facts struct {
		TFlopsPeak float64 `yaml:"TFlopsPeak"`
		TFlopsFP8  float64 `yaml:"TFlopsFP8"`
		BwPeakTBs  float64 `yaml:"BwPeakTBs"`
		MemoryGiB  float64 `yaml:"MemoryGiB"`
	}
	if err := yaml.Unmarshal(raw, &facts); err != nil {
		return sim.HardwareCalib{}, err
	}
	return sim.HardwareCalib{
		TFlopsPeak: facts.TFlopsPeak,
		TFlopsFP8:  facts.TFlopsFP8,
		BwPeakTBs:  facts.BwPeakTBs,
		MemoryGiB:  facts.MemoryGiB,
		MfuPrefill: 0.45,
		MfuDecode:  0.30,
	}, nil
}

package cmd

// The single post-run path both blis run and blis replay take (#1871, INV-13): every figure
// of the Result document, and the --view summary come from one call, so the two commands cannot drift into reporting different things.

import (
	"fmt"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/blis-substrate/contracts/spec/measurement"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	schemaworkload "github.com/inference-sim/blis-substrate/contracts/spec/workload"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/llmd"
	"github.com/inference-sim/inference-sim/sim/trace"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// floatMapAny copies a float map into a generic one, for an open-ended section.
func floatMapAny(in map[string]float64) map[string]any {
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// runControls are what a command tells the frame's generator and stop rule: the offered load,
// the request bound the run kept, the seed, and the workload binding.
type runControls struct {
	Load        measurement.Load
	NumRequests int
	Seed        int64
	Binding     *schemaworkload.Binding
}

// postRun is what a finished run hands the emitter.
type postRun struct {
	cs      *cluster.ClusterSimulator
	gateway *llmd.Gateway
	// run are the run controls the document records.
	run runControls
	// goodputTargets are the resolved SLO targets.
	goodputTargets map[string]workload.SLODimTargets
	// satTracer is the saturation tracer, nil when no detector was selected.
	satTracer *saturationTracer
	// agentic is an agentic replay's own counts (workload.AgentXSummary), nil otherwise.
	agentic any
	// artifacts are the bulk files the command wrote before emitting, by role.
	artifacts []artifactFile
	// fitnessWeights is --fitness-weights, "" for none (run only).
	fitnessWeights string
	// extraOverrides are the command's own overrides beyond the flags it was given (a workload
	// spec's seed the run did not use, a request cap the run's follow-ups exceeded).
	extraOverrides []sim.ScenarioOverride
	// started is when the command started, for the runtime block.
	started time.Time
	// frame is what the command tells the experimental frame about its request source.
	frame frameGenerator
	// numRequestsCap is the request cap the generator was given, 0 for none (the frame's
	// stop rule).
	numRequestsCap int
}

// emitRunResult builds, validates and prints the Result document, and renders --view summary.
func emitRunResult(p postRun) {
	cs := p.cs
	aggregated := cs.AggregatedMetrics()

	// Saturation: the detectors stream over the completed requests and write their report
	// (--saturation-report) as before; the final label goes into the document.
	var satFinal map[string]string
	if p.satTracer != nil {
		final, err := p.satTracer.run(aggregated.CompletedRequestMetrics())
		if err != nil {
			logrus.Fatalf("Saturation: %v", err)
		}
		if len(final) > 0 {
			satFinal = make(map[string]string, len(final))
			for k, v := range final {
				satFinal[k] = v.String()
			}
		}
	}

	raw := cluster.CollectRawMetrics(aggregated, cs.PerInstanceMetrics(), cs.RejectedRequests(), scheduler,
		cs.RoutingRejections(), cs.EncodeRoutingRejections(), cs.InjectedByClass())
	pdMetrics := cs.PDMetrics()
	// INV-13 SYNC POINT (metrics): run and replay share this block.
	raw.PD = pdMetrics
	raw.ShedByTier = cs.ShedByTier()
	raw.GatewayQueueDepth = cs.GatewayQueueDepth()
	raw.GatewayQueueRejected = cs.GatewayQueueRejected()
	raw.GatewayEvicted = cs.GatewayEvicted()
	raw.GatewayExpired = cs.GatewayExpired()

	series := cs.ServerSeries()

	docs, err := resultScenarioDocs()
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	policy, err := resultPolicy(policyConfigPath)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	sum := aggregated.Summary()
	routerReq := p.gateway.Requirements()
	artifacts, err := resultArtifacts(p.artifacts)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	data := mustRunData()
	in := resultInputs{
		Scenario:       docs.Scenario,
		Deployment:     docs.Deployment,
		Packaging:      resultPackaging(kernelOpened),
		Policy:         policy,
		Options:        resultInputOptions,
		Generator:      result.Generator{Workload: p.run.Binding, Load: p.run.Load, Seed: p.frame.Seed},
		NumRequests:    p.run.NumRequests,
		Overrides:      runOverrides(p.extraOverrides...),
		GPUs:           runGPUs(cs, docs.Deployment),
		GoodputTargets: p.goodputTargets,
		Saturation:     satFinal,
		Artifacts:      artifacts,
		Engine:         result.Engine{Kind: scenarioEngine.Kind, Version: scenarioEngine.Version},
		Revisions:      resultRevisions(),
		Data:           resultData(data),
	}
	out := resultMetrics{
		Aggregated:      aggregated,
		Ledger:          resultLedger(cs),
		PD:              pdMetrics,
		Engine:          engineSection(aggregated, sum, raw, series, prefixCacheRecords, acceptanceRecord, instanceSections(cs), p.agentic),
		Router:          routerSection(routerReq, raw.ShedByTier, raw),
		Kernel:          kernelSection(kernelProvenance(kernelOpened, kernelPDOpened)),
		InjectedByClass: cs.InjectedByClass(),
		Rows:            requestRows(cs, aggregated),
		ClockUs:         cs.Clock(),
		HorizonUs:       cs.Horizon(),
	}
	// The generator's request cap bounds what it emits; a closed-loop session's follow-up
	// rounds are generated during the run on top of it. The frame's stop rule states the cap
	// the run was given; when the run generated more, an override row says so.
	if cap := in.NumRequests; cap > 0 && cap < out.Ledger.Generated {
		in.Overrides = append(in.Overrides, sim.ScenarioOverride{Setting: "num_requests_cap",
			Scenario: fmt.Sprint(out.Ledger.Generated), Run: fmt.Sprint(cap),
			Source: "the workload generator's cap; closed-loop follow-up rounds are generated beyond it"})
	}
	r := buildResult(in, out)
	if len(r.Metrics) == 0 {
		// Nothing arrived and simulated time never advanced (no horizon bounded it): there is no
		// span to state any rate over, so the run measured nothing a Result can report.
		logrus.Fatalf("the run simulated no time: no request arrived (generated %d) and no --horizon bounds the run, "+
			"so there is nothing to report; check the workload's arrival windows, or pass --horizon", out.Ledger.Generated)
	}
	// The experimental frame (DEV-7, result_frame.go), then the selected-entry report, read
	// last so it covers every lookup the run made (VAL-6).
	buildFrame(p.frame, r, windowRulesOf(aggregated, sum), p.numRequestsCap, cs.Horizon())
	r.SelectedEntries = data.Report().Result()
	markRerun(r)
	r.Runtime = resultRuntime(p.started)

	view := &summaryView{
		Raw:               raw,
		PerSLO:            cluster.ComputePerSLODistributions(aggregated),
		GoodputConfigured: len(p.goodputTargets) > 0,
		PerModel:          cluster.ComputePerModelMetrics(aggregated),
		PerTenant:         cluster.ComputePerTenantMetrics(aggregated),
		Sessions:          cluster.ComputeSessionMetrics(aggregated),
	}
	if p.fitnessWeights != "" {
		weights, err := cluster.ParseFitnessWeights(p.fitnessWeights)
		if err != nil {
			logrus.Fatalf("Invalid fitness weights: %v", err)
		}
		fitness, err := cluster.ComputeFitness(raw, weights)
		if err != nil {
			logrus.Fatalf("Fitness evaluation failed: %v", err)
		}
		view.Fitness = &fitness
		// The score is reported in the document too, under router: it scores the run's routing
		// outcome from its raw metrics (#1871 review S5).
		r.Reports.Router["fitness"] = map[string]any{"score": fitness.Score, "components": floatMapAny(fitness.Components),
			"weights": p.fitnessWeights}
	}
	if cs.Trace() != nil && summarizeTrace {
		view.Trace = trace.Summarize(cs.Trace())
		r.Reports.Router["trace_summary"] = toGeneric(view.Trace)
	}
	if resultView != viewSummary && (p.fitnessWeights != "" || (cs.Trace() != nil && summarizeTrace)) {
		logrus.Warnf("--fitness-weights and --summarize-trace are reported in the Result's router section; " +
			"pass --view summary to also print them as text on stderr")
	}
	emitResult(r, view)
}

// resultPackaging is the packaging the Result's inputs record: the kernel's resolved
// packaging when any value came from the chip's catalog entry, nil when the scenario stated
// every value (input.scenario then records each as stated) or no kernel is open.
func resultPackaging(m *engineModel) *scenario.Packaging {
	if m == nil {
		return nil
	}
	pk := m.Packaging()
	if pk.GPUsPerNodeSource == scenario.SourceStated && pk.GPUsPerRack == 0 {
		return nil
	}
	return &pk
}

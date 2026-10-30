package cmd

// The single post-run path both blis run and blis replay take (#1871, INV-13): every figure
// of the Result document, the deprecated --metrics-path file, and the --view summary come
// from one call, so the two commands cannot drift into reporting different things.

import (
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/blis-schemas/spec/result"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/llmd"
	"github.com/inference-sim/inference-sim/sim/trace"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// postRun is what a finished run hands the emitter.
type postRun struct {
	cs      *cluster.ClusterSimulator
	gateway *llmd.Gateway
	// run are the run controls the document records.
	run result.Run
	// goodputTargets are the resolved SLO targets.
	goodputTargets map[string]workload.SLODimTargets
	// satTracer is the saturation tracer, nil when no detector was selected.
	satTracer *saturationTracer
	// metricsPath is the deprecated --metrics-path file, "" for none.
	metricsPath string
	// metricsFlag names the flag metricsPath came from, for the deprecation log.
	metricsFlag string
	// extraFileOptions are the command's own file-only blocks (replay's agentic summary).
	extraFileOptions []sim.EmitOption
	// agentic is an agentic replay's own counts (workload.AgentXSummary), nil otherwise.
	agentic any
	// artifacts are the bulk files the command wrote before emitting, by role.
	artifacts []artifactFile
	// fitnessWeights is --fitness-weights, "" for none (run only).
	fitnessWeights string
	// started is when the command started, for the runtime block.
	started time.Time
}

// emitRunResult builds, validates and prints the Result document, writes the deprecated
// --metrics-path file, and renders --view summary.
func emitRunResult(p postRun) {
	cs := p.cs
	aggregated := cs.AggregatedMetrics()

	// Saturation: the detectors stream over the completed requests and write their report
	// (--saturation-report) as before; the final label goes into the document.
	var satFinal map[string]string
	var satAny any
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
			satAny = final
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

	// The deprecated --metrics-path file keeps its content for one release (#1871): the
	// whole-run MetricsOutput with goodput and saturation, and the file-only blocks.
	series := cs.ServerSeries()
	if p.metricsPath != "" {
		warnDeprecatedFile(p.metricsFlag, p.metricsPath)
		legacy := aggregated.BuildOutput("cluster")
		emitGoodput(&legacy, aggregated, cs.InjectedByClass(), float64(aggregated.SimEndedTime)/1e6, p.goodputTargets)
		if satAny != nil {
			legacy.Saturation = satAny
		}
		extra := append([]sim.EmitOption{sim.WithPD(pdMetrics.Output()), sim.WithServerSeries(series)}, p.extraFileOptions...)
		if err := aggregated.WriteMetricsFile(legacy, p.metricsPath,
			append(resultsFileEmitOptions(p.metricsPath, resolvedCatalogRoot), extra...)...); err != nil {
			logrus.Fatalf("--metrics-path: %v", err)
		}
	}

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
	files := p.artifacts
	if p.metricsPath != "" {
		files = append(files, artifactFile{role: "metrics_file", path: p.metricsPath, rows: len(aggregated.Requests)})
	}
	artifacts, err := resultArtifacts(files)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	in := resultInputs{
		Scenario:       docs.Scenario,
		Deployment:     docs.Deployment,
		Policy:         policy,
		Run:            p.run,
		Overrides:      scenarioOverrides,
		GPUs:           runGPUs(cs, docs.Deployment),
		GoodputTargets: p.goodputTargets,
		Saturation:     satFinal,
		Artifacts:      artifacts,
		Revisions:      resultRevisions(resolvedCatalogRoot, kernelRegistry, docs.Scenario.EngineVersion),
	}
	out := resultMetrics{
		Aggregated:      aggregated,
		Ledger:          resultLedger(cs),
		PD:              pdMetrics,
		Engine:          engineSection(aggregated, sum, raw, series, prefixCacheRecords, acceptanceRecord, instanceSections(cs), p.agentic),
		Router:          routerSection(routerReq, raw.ShedByTier, raw),
		Kernel:          kernelSection(kernelProvenance(kernelOpened, kernelPDOpened)),
		InjectedByClass: cs.InjectedByClass(),
		ClockUs:         cs.Clock(),
		HorizonUs:       cs.Horizon(),
	}
	// blis-schemas reads num_requests as the most requests the run generates. BLIS's
	// --num-requests caps the requests the workload generator emits; a closed-loop session's
	// follow-up rounds are generated during the run on top of them. The document records the
	// larger, so it states a bound the run kept.
	if in.Run.NumRequests < out.Ledger.Generated {
		in.Run.NumRequests = out.Ledger.Generated
	}
	r := buildResult(in, out)
	if len(r.Metrics) == 0 {
		// Nothing arrived and simulated time never advanced (no horizon bounded it): there is no
		// span to state any rate over, so the run measured nothing a Result can report.
		logrus.Fatalf("the run simulated no time: no request arrived (generated %d) and no --horizon bounds the run, "+
			"so there is nothing to report; check the workload's arrival windows, or pass --horizon", out.Ledger.Generated)
	}
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
	}
	if cs.Trace() != nil && summarizeTrace {
		view.Trace = trace.Summarize(cs.Trace())
	}
	emitResult(r, view)
}

package cmd

// The Result document's input record, revisions and component sections (#1871): what a run
// was given, the versions it read, and what the engine, router and kernel reported.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"time"

	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v3"

	blisschemas "github.com/inference-sim/inference-sim/substrate/contracts"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/measurement"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/scenario"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/trace"
	schemaworkload "github.com/inference-sim/inference-sim/substrate/contracts/spec/workload"
	"github.com/inference-sim/inference-sim/substrate/load"
	embed "github.com/llm-d/llm-d-router/blisembed"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// Module path whose version a Result records beyond this module's own.
const modulePathRouter = "github.com/llm-d/llm-d-router"

// resultRevisions are the versions of the BLIS modules the run was built from.
func resultRevisions() result.Revisions {
	info, _ := debug.ReadBuildInfo()
	return revisionsFrom(info)
}

// revisionsFrom is resultRevisions over given build information, for tests. The substrate and
// the engines are packages of the simulator's module, so the substrate and engine_module
// revisions are the simulator's own build; only the router fork is a separate module. A module
// the build information does not carry is "unknown" for the simulator and the substrate, which
// the document requires.
func revisionsFrom(info *debug.BuildInfo) result.Revisions {
	v := versionsFrom(info)
	r := result.Revisions{Simulator: v.BLIS, Substrate: v.Schemas, LLMD: llmdCommit()}
	if info != nil {
		r.EngineModule = v.BLIS
		for _, dep := range info.Deps {
			if dep.Path == modulePathRouter {
				r.Router = depVersion(dep)
			}
		}
	}
	return r
}

// resultBuild is the build the Result's data record names (design §3.3): the simulator's
// release tag when the binary is a tagged release, else "unreleased". Go stamps a module's
// version only into a binary built by `go install …@version`, which inference-sim's replace
// directive forbids (X11), so a release build sets buildRelease with -ldflags.
func resultBuild() string {
	if releaseTag.MatchString(buildRelease) {
		return buildRelease
	}
	return result.BuildUnreleased
}

// buildRelease is the release tag a release build stamps:
//
//	go build -ldflags "-X github.com/inference-sim/inference-sim/cmd.buildRelease=v0.16.0"
var buildRelease string

// releaseTag is a plain release tag, vMAJOR.MINOR.PATCH.
var releaseTag = regexp.MustCompile(`^v[0-9]+\.[0-9]+\.[0-9]+$`)

// resultData is the Result's record of the catalog and registry the run read (design §3.3,
// X8): each one's source, content hash, overlay hashes and combined hash, the build, and
// whether the run used released data. Released data is a tagged simulator release that read
// the embedded copy of both with no overlay; the embedded data is the release's own substrate/
// directory.
func resultData(d *load.Data) result.Data {
	out := result.Data{Catalog: d.Catalog.Result(), Registry: d.Registry.Result(), Build: resultBuild()}
	out.ReleasedData = out.Build != result.BuildUnreleased &&
		d.Catalog.Source == load.SourceEmbedded && d.Registry.Source == load.SourceEmbedded &&
		len(d.Catalog.Overlays) == 0 && len(d.Registry.Overlays) == 0
	return out
}

// resultScenarioDocs reads the scenario and deployment the run used, through blis-schemas' own
// strict bundle loader (the same decoding the kernel applies).
func resultScenarioDocs() (resultDocs, error) {
	path := engineScenarioPath() // the SGLang scenario itself, not its host-side twin
	sc, dep, err := blisschemas.LoadBundle(path)
	if err != nil {
		return resultDocs{}, fmt.Errorf("re-reading the scenario for the Result document: %w", err)
	}
	return resultDocs{Scenario: sc, Deployment: dep}, nil
}

// resultPolicy reads the --policy-config bundle verbatim, as a generic map; nil without one.
func resultPolicy(path string) (map[string]any, error) {
	if path == "" {
		return nil, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("--policy-config %s: %w", path, err)
	}
	var m map[string]any
	if err := yaml.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("--policy-config %s: %w", path, err)
	}
	return m, nil
}

// specLoad is a workload spec's offered load: the sum of its clients' concurrency when any
// client is closed-loop, else its aggregate rate. A spec in absolute-rate mode (aggregate rate
// 0: each client or cohort states its own trace_rate per window) states no single rate; its
// load is the one the requests arrived at (arrivalLoad).
func specLoad(spec *workload.WorkloadSpec, arrivals []int64) measurement.Load {
	if spec == nil {
		return arrivalLoad(arrivals)
	}
	users := 0
	for _, c := range spec.Clients {
		users += c.Concurrency
	}
	if users > 0 {
		return measurement.Load{Concurrency: users}
	}
	if spec.AggregateRate > 0 {
		return measurement.Load{RequestRate: spec.AggregateRate}
	}
	return arrivalLoad(arrivals)
}

// traceLoad is a trace's offered load: its requests over its arrival span, the rate the
// replay injects them at.
func traceLoad(tr *workload.TraceV2) measurement.Load {
	if tr == nil {
		return measurement.Load{}
	}
	at := make([]int64, len(tr.Records))
	for i, r := range tr.Records {
		at[i] = r.ArrivalTimeUs
	}
	return arrivalLoad(at)
}

// arrivalLoad is the rate requests arrived at, given their arrival times in ticks: their count
// over the span from the first to the last. When every request arrives at once the load is the
// whole set in flight, a concurrency.
func arrivalLoad(at []int64) measurement.Load {
	if len(at) == 0 {
		return measurement.Load{}
	}
	first, last := at[0], at[0]
	for _, a := range at {
		first, last = min(first, a), max(last, a)
	}
	if last <= first {
		return measurement.Load{Concurrency: len(at)}
	}
	return measurement.Load{RequestRate: float64(len(at)) / (float64(last-first) / 1e6)}
}

// horizonSeconds is a horizon in seconds, nil for an unbounded one.
func horizonSeconds(ticks int64) *float64 {
	if ticks <= 0 || ticks == math.MaxInt64 {
		return nil
	}
	s := float64(ticks) / 1e6
	return &s
}

// traceBinding is the workload binding of a replayed trace: its data file and header.
// The data file is named as an artifact is (artifactRef: no host directory layout) and pinned
// by its sha256; a file that cannot be read is an error (R1).
func traceBinding(dataPath string, h workload.TraceHeader, rows int) (*schemaworkload.Binding, error) {
	d, err := fileDigest(dataPath)
	if err != nil {
		return nil, fmt.Errorf("--trace-data %s: %w", dataPath, err)
	}
	mode, err := resultTraceMode(h.Mode)
	if err != nil {
		return nil, err
	}
	// The binding names the trace by its mode, seed and goodput targets; the deployment
	// inputs the header records (offload, router, cross-node span) are recorded in the
	// Result's input and reports, not restated here.
	ref := &schemaworkload.TraceRef{Data: artifactRef(dataPath), SHA256: d, Rows: rows, Header: trace.Header{
		Version: h.Version, TimeUnit: h.TimeUnit, Mode: mode, WorkloadSeed: h.WorkloadSeed,
		GoodputSLOTargets: h.GoodputSLOTargets,
	}}
	return &schemaworkload.Binding{Trace: ref}, nil
}

// resultTraceMode is a trace header mode in the contracts' vocabulary (contracts/spec/trace),
// which has the three modes a trace can have. A mode outside the set is refused.
func resultTraceMode(mode trace.Mode) (trace.Mode, error) {
	switch mode {
	case workload.TraceModeGenerated:
		return trace.ModeGenerated, nil
	case workload.TraceModeReplayed:
		return trace.ModeReplayed, nil
	case workload.TraceModeConverted:
		return trace.ModeConverted, nil
	}
	return "", fmt.Errorf("trace header mode %q has no Result mode", mode)
}

// resultLedger is the fate count (DEV-9) from the cluster's fate ledger: every generated
// request in exactly one fate, drops by reason. The cluster's checkFates has already held the
// ledger to INV-1's buckets.
func resultLedger(cs *cluster.ClusterSimulator) result.Requests {
	t := cs.Fates().Tally()
	q := result.Requests{
		Generated:         t.Created,
		PlannedNotIssued:  t.PlannedNotIssued,
		Completed:         t.ByFate[sim.FateCompleted],
		TimedOut:          t.ByFate[sim.FateTimedOut],
		InFlightAtHorizon: t.ByFate[sim.FateInFlightAtHorizon],
		QueuedAtHorizon:   t.ByFate[sim.FateQueuedAtHorizon],
		NotStarted:        t.ByFate[sim.FateNotStarted],
		LengthCapped:      cs.AggregatedMetrics().LengthCappedRequests,
	}
	for reason, n := range t.Drops {
		if n > 0 {
			if q.Dropped == nil {
				q.Dropped = map[result.DropReason]int{}
			}
			q.Dropped[result.DropReason(reason)] = n
		}
	}
	return q
}

// runGPUs is the number of GPUs the run's instances occupy: each instance its own pool's
// tensor x pipeline x prefill-context-parallel x data-parallel width, the GPUs one engine
// instance (a whole data-parallel group) spans.
// An instance whose role names no pool (a test cluster) counts its tensor-parallel degree.
func runGPUs(cs *cluster.ClusterSimulator, dep *deployment.Deployment) int {
	all, _ := runGPUsByRole(cs, dep)
	return all
}

// runDecodeGPUs is the GPUs of the decode pool's instances, 0 for a run with none.
func runDecodeGPUs(cs *cluster.ClusterSimulator, dep *deployment.Deployment) int {
	_, decode := runGPUsByRole(cs, dep)
	return decode
}

// runGPUsByRole is runGPUs, and the part of it the decode instances occupy.
func runGPUsByRole(cs *cluster.ClusterSimulator, dep *deployment.Deployment) (all, decode int) {
	gpus, isDecode := instanceGPUs(cs, dep)
	for id, g := range gpus {
		all += g
		if isDecode[id] {
			decode += g
		}
	}
	return all, decode
}

// poolGPUs is the GPUs one instance of each pool role occupies, tp x pp x pcp x dp
// (sim.GPUsPerInstance): the one count node-pool placement, node span, cost and the Result's GPU
// totals read. Empty without a deployment.
func poolGPUs(dep *deployment.Deployment) map[deployment.Role]int {
	perRole := map[deployment.Role]int{}
	if dep != nil {
		for _, p := range dep.Pools {
			perRole[p.Role] = sim.GPUsPerInstance(p.Parallel.TP, p.Parallel.PP, p.Parallel.PCP, p.Parallel.DP)
		}
	}
	return perRole
}

// stateInstanceGPUs states in config the GPUs one instance of each pool occupies, so a node pool
// reserves, spans and bills an instance for its whole layout and not its tensor-parallel degree
// alone (#2260). A role the deployment has no pool for keeps the unstated count (its TP).
func stateInstanceGPUs(config *cluster.DeploymentConfig) {
	docs, err := resultScenarioDocs()
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	perRole := poolGPUs(docs.Deployment)
	if g, ok := perRole[deployment.RoleColocated]; ok {
		config.GPUs = g
	}
	if g, ok := perRole[deployment.RolePrefill]; ok && config.Roles.Prefill != nil {
		config.Roles.Prefill.GPUs = g
	}
	if g, ok := perRole[deployment.RoleDecode]; ok && config.Roles.Decode != nil {
		config.Roles.Decode.GPUs = g
	}
}

// instanceGPUs is the GPUs each instance occupies, by instance id, and which of them are
// decode instances.
func instanceGPUs(cs *cluster.ClusterSimulator, dep *deployment.Deployment) (gpus map[string]int, decode map[string]bool) {
	perRole := poolGPUs(dep)
	gpus, decode = map[string]int{}, map[string]bool{}
	membership := cs.PoolMembership()
	for _, inst := range cs.Instances() {
		role := deployment.RoleColocated
		if r, ok := membership[string(inst.ID())]; ok {
			switch r {
			case cluster.PoolRolePrefill:
				role = deployment.RolePrefill
			case cluster.PoolRoleDecode:
				role = deployment.RoleDecode
			}
		}
		g, ok := perRole[role]
		if !ok {
			g = max(inst.TPDegree, 1)
		}
		gpus[string(inst.ID())] = g
		decode[string(inst.ID())] = role == deployment.RoleDecode
	}
	return gpus, decode
}

// engineSection is what the engine side of the run reported: under "cluster" the aggregate
// (the
// step-interval series the legacy itl_* keys held, preemptions, KV allocation failures, the
// prefix-cache hit rate, the acceptance record and the server series), under each pool role
// its resolved prefix-cache record, its memory, and under "instances" each instance's own figures
// by id.
func engineSection(m *sim.Metrics, sum sim.MeasuredSummary, raw *cluster.RawMetrics, series *sim.ServerSeriesOutput,
	prefix map[string]sim.PrefixCacheRecord, acceptance *sim.AcceptanceRecord, poolAcceptance map[string]*sim.AcceptanceRecord, perInstance map[string]any, agentic any, poolMemory map[string]any) map[string]any {
	c := map[string]any{
		"preemptions":                        m.PreemptionCount,
		"kv_allocation_failures":             m.KVAllocationFailures,
		"scheduling_delay_p99_ms":            schedulingDelayP99(m),
		"cache_hit_rate":                     sum.CacheHitRate,
		"cache_hit_rate_whole_run":           m.CacheHitRate,
		"cpu_tier_hit_rate":                  sum.CPUTierHitRate,
		"kv_capacity_tokens_per_replica":     sum.KVCapacityTokensPerReplica,
		"in_flight_requests_mean":            sum.InFlightRequestsMean,
		"in_flight_requests_max":             sum.InFlightRequestsMax,
		"gateway_queue_delay_mean_ms":        sum.GatewayQueueDelayMeanMs,
		"gateway_queue_delay_p90_ms":         sum.GatewayQueueDelayP90Ms,
		"step_interval_mean_ms":              sum.StepIntervalMeanMs,
		"step_interval_p50_ms":               sum.StepIntervalP50Ms,
		"step_interval_p90_ms":               sum.StepIntervalP90Ms,
		"step_interval_p95_ms":               sum.StepIntervalP95Ms,
		"step_interval_p99_ms":               sum.StepIntervalP99Ms,
		"step_interval_percentile_method":    sim.PercentileMethodNumpyLinear,
		"total_input_tokens":                 m.TotalInputTokens,
		"engine_admissions":                  m.EngineAccepted,
		"total_output_tokens":                m.TotalOutputTokens,
		"simulated_duration_s":               float64(m.SimEndedTime) / 1e6,
		"measured_cancelled_requests":        sum.CancelledRequests,
		"measured_failed_requests":           sum.FailedRequests,
		"measured_context_overflow_requests": sum.ContextOverflowRequests,
	}
	if raw != nil {
		c["preemption_rate"] = raw.PreemptionRate
		c["kv_thrashing_rate"] = raw.KVThrashingRate
	}
	whole := m.BuildOutput("cluster")
	if len(whole.Adapters) > 0 {
		c["adapters"] = toGeneric(whole.Adapters)
	}
	// The whole-run figures stdout printed before the Result (inference-sim/inference-sim#2467 section A: "whole-run
	// values, when wanted, are a second named window"; schema v1 states one window, so they are
	// the engine's): latencies over every completed request, the per-decode-step interval the
	// legacy itl_* keys held, and the rates over the simulated run.
	c["whole_run"] = map[string]any{
		"ttft_mean_ms": whole.TTFTMeanMs, "ttft_p90_ms": whole.TTFTP90Ms, "ttft_p95_ms": whole.TTFTP95Ms, "ttft_p99_ms": whole.TTFTP99Ms,
		"e2e_mean_ms": whole.E2EMeanMs, "e2e_p90_ms": whole.E2EP90Ms, "e2e_p95_ms": whole.E2EP95Ms, "e2e_p99_ms": whole.E2EP99Ms,
		"step_interval_mean_ms": whole.ITLMeanMs, "step_interval_p90_ms": whole.ITLP90Ms,
		"step_interval_p95_ms": whole.ITLP95Ms, "step_interval_p99_ms": whole.ITLP99Ms,
		"tokens_per_sec": whole.TokensPerSec, "responses_per_sec": whole.ResponsesPerSec,
	}
	if agentic != nil {
		c["agentic"] = toGeneric(agentic)
	}
	if series != nil {
		c["server_series"] = toGeneric(series)
	}
	if acceptance != nil {
		c["acceptance"] = toGeneric(acceptance)
	}
	out := map[string]any{"cluster": c}
	if len(perInstance) > 0 {
		out["instances"] = perInstance
	}
	for role, rec := range prefix {
		out[role] = map[string]any{"prefix_cache": toGeneric(rec)}
	}
	// Pools that draft differently each state their own acceptance (inference-sim#2125); the
	// cluster states none then, as there is no one acceptance of the run.
	for role, rec := range poolAcceptance {
		section, _ := out[role].(map[string]any)
		if section == nil {
			section = map[string]any{}
			out[role] = section
		}
		section["acceptance"] = toGeneric(rec)
	}
	// Each pool's memory as the engine that sized it left it, so a Result states the deployment a
	// reader can reproduce without the run.
	for role, mem := range poolMemory {
		section, _ := out[role].(map[string]any)
		if section == nil {
			section = map[string]any{}
			out[role] = section
		}
		section["memory"] = mem
	}
	return out
}

// instanceSections are each engine instance's own figures, keyed by instance id, as a
// multi-instance run printed them: the whole-run MetricsOutput of that instance, and under
// "measured_window" the same instance's figures over the run's measured window, so a consumer can
// judge the worst instance on the window and not on the warmup-inflated whole run (#2087). The
// window is the aggregate's (sum), and the figures are the definitions of the cluster's, taken
// over the requests this instance served (Metrics.SummaryOver, as the breakdowns do). A run with
// no measured window states none.
//
// A disaggregated run's instances hold sub-request records (a prefill leg and a decode leg per
// user request), which are not what a user measured, so its figures are the user-level records
// the cluster's own figures use (ClusterSimulator.projectPDMetrics), selected per instance
// (instanceUserWindow): a decode instance states the user requests it completed, and a prefill
// instance the user requests it prefilled.
func instanceSections(cs *cluster.ClusterSimulator, sum sim.MeasuredSummary) map[string]any {
	insts := cs.Instances()
	if len(insts) <= 1 {
		return nil
	}
	// In a disaggregated run, the parent ids each prefill-only instance prefilled.
	roles := cs.PoolMembership()
	var prefilled map[string]map[string]bool
	if roles != nil {
		prefilled = map[string]map[string]bool{}
		for _, p := range cs.ParentRequests() {
			id := string(p.PrefillInstanceID)
			if prefilled[id] == nil {
				prefilled[id] = map[string]bool{}
			}
			prefilled[id][p.ID] = true
		}
	}
	out := make(map[string]any, len(insts))
	for _, inst := range insts {
		id := string(inst.ID())
		block := toGeneric(inst.Metrics().BuildOutput(id))
		if m, ok := block.(map[string]any); ok && sum.MeasuredWindowS > 0 {
			m["measured_window"] = instanceMeasuredWindow(cs, inst, sum, roles, prefilled)
		}
		out[id] = block
	}
	return out
}

// instanceMeasuredWindow is one instance's measured_window block. In a colocated run the
// instance's own records are the users' and the figures are taken from them. In a disaggregated
// run they are taken from the user-level records of the aggregate:
//
//   - an instance that completed requests (a decode instance, or any instance serving a request
//     whole) states the user requests it completed: their TTFT and E2E are the user's, from
//     arrival, and measured_completed_requests counts them, so the instances' counts add up to
//     the window's;
//   - an instance that only prefilled states the user requests it prefilled, as those users saw
//     them: their TTFT and E2E include the decode leg, so a slow prefill instance shows in its
//     users' latency. It completes no request (measured_completed_requests is 0), produces no
//     token (output_tok_s is 0) and has no inter-token gaps; measured_user_requests is the number
//     of requests behind its figures. A request is counted once under measured_completed_requests
//     and once more under the measured_user_requests of its prefill instance, which is why the
//     latter is a separate key.
//
// An instance of a shared prefill-decode pool takes the first reading (it completes requests).
// A disaggregated instance states its pool role and measured_user_requests, so a reader can tell
// the semantics apart; a colocated instance states neither (its block is as it was, plus the
// time-per-output-token and inter-token figures).
func instanceMeasuredWindow(cs *cluster.ClusterSimulator, inst *cluster.InstanceSimulator, sum sim.MeasuredSummary, roles map[string]cluster.PoolRole, prefilled map[string]map[string]bool) map[string]any {
	id := string(inst.ID())
	// The cluster's inter-token figures keep a request only when it completes by an explicit window's
	// end; a derived window ends at the last completion and cuts nothing.
	var gapEnd int64
	if sum.MeasuredWindowSource == sim.WindowSourceExplicit {
		gapEnd = sum.WindowEndUs
	}
	if roles == nil {
		w := inst.Metrics().SummaryOver(func(sim.RequestMetrics) bool { return true }, sum.WindowStartUs, sum.WindowEndUs)
		mw := measuredWindowFigures(sum, w)
		addClientITL(mw, inst.Metrics().ClientITLOver(gapEnd))
		return mw
	}
	agg := cs.AggregatedMetrics()
	role := roles[id]
	if role.Has(cluster.PoolRolePrefill) && !role.Has(cluster.PoolRoleDecode) {
		mine := prefilled[id]
		w := agg.SummaryOver(func(rm sim.RequestMetrics) bool { return mine[rm.ID] }, sum.WindowStartUs, sum.WindowEndUs)
		mw := measuredWindowFigures(sum, w)
		mw["role"] = "prefill"
		mw["measured_user_requests"] = w.MeasuredCompletedRequests
		mw["measured_completed_requests"] = 0
		mw["output_tok_s"] = 0.0
		return mw
	}
	w := agg.SummaryOver(func(rm sim.RequestMetrics) bool { return rm.HandledBy == id }, sum.WindowStartUs, sum.WindowEndUs)
	mw := measuredWindowFigures(sum, w)
	mw["role"] = role.String()
	mw["measured_user_requests"] = w.MeasuredCompletedRequests
	addClientITL(mw, inst.Metrics().ClientITLOver(gapEnd))
	return mw
}

// measuredWindowFigures are the window bounds and the request-level figures of w, the
// definitions of the cluster's: TTFT, E2E, output throughput and time per output token.
func measuredWindowFigures(sum, w sim.MeasuredSummary) map[string]any {
	mw := map[string]any{
		"window_start_us": sum.WindowStartUs, "window_end_us": sum.WindowEndUs,
		"measured_completed_requests": w.MeasuredCompletedRequests,
		"ttft_mean_ms":                w.TTFTMeanMs, "ttft_p50_ms": w.TTFTP50Ms, "ttft_p90_ms": w.TTFTP90Ms, "ttft_p95_ms": w.TTFTP95Ms, "ttft_p99_ms": w.TTFTP99Ms,
		"e2e_mean_ms": w.E2EMeanMs, "e2e_p50_ms": w.E2EP50Ms, "e2e_p90_ms": w.E2EP90Ms, "e2e_p95_ms": w.E2EP95Ms, "e2e_p99_ms": w.E2EP99Ms,
		"output_tok_s": w.OutputThroughputTokS,
		// The time per output token, (latency - TTFT) / (OSL - 1) per request (the cluster's
		// tpot_*): a request of one token has none, so tpot_samples can be below the requests.
		"tpot_samples": w.RequestITLCount,
	}
	if w.RequestITLCount > 0 {
		mw["tpot_mean_ms"], mw["tpot_p50_ms"], mw["tpot_p90_ms"] = w.RequestITLMeanMs, w.RequestITLP50Ms, w.RequestITLP90Ms
		mw["tpot_p95_ms"], mw["tpot_p99_ms"] = w.RequestITLP95Ms, w.RequestITLP99Ms
	}
	return mw
}

// addClientITL adds the client-observed inter-token latency (the cluster's itl_*) an instance
// streamed over the window; none when it streamed no gap.
func addClientITL(mw map[string]any, g sim.ClientITL) {
	mw["itl_samples"] = g.Count
	if g.Count > 0 {
		mw["itl_mean_ms"], mw["itl_p50_ms"], mw["itl_p90_ms"] = g.MeanMs, g.P50Ms, g.P90Ms
		mw["itl_p95_ms"], mw["itl_p99_ms"] = g.P95Ms, g.P99Ms
	}
}

// schedulingDelayP99 is the p99 of per-request scheduling delay in ms, as MetricsOutput
// reported it.
func schedulingDelayP99(m *sim.Metrics) float64 {
	vals := make([]float64, 0, len(m.RequestSchedulingDelays))
	for _, v := range m.RequestSchedulingDelays {
		vals = append(vals, float64(v))
	}
	sort.Float64s(vals)
	return sim.CalculatePercentile(vals, 99)
}

// routerSection is what the embedded router reported: its requirements, and the requests
// it shed by SLO class.
func routerSection(req embed.Requirements, shed map[string]int, anomalies *cluster.RawMetrics) map[string]any {
	r := map[string]any{"requirements": toGeneric(req)}
	if len(shed) > 0 {
		s := map[string]any{}
		for k, v := range shed {
			s[k] = v
		}
		r["shed_by_class"] = s
	}
	if anomalies != nil {
		r["priority_inversions"] = anomalies.PriorityInversions
		r["hol_blocking_events"] = anomalies.HOLBlockingEvents
	}
	return r
}

// kernelSection is the latency kernel's per-pool coefficient provenance.
func kernelSection(p map[string]sim.KernelPoolProvenance) map[string]any {
	if len(p) == 0 {
		return nil
	}
	out := map[string]any{}
	for role, prov := range p {
		out[role] = toGeneric(prov)
	}
	return out
}

// toGeneric re-expresses v as the generic maps and slices a YAML encoder writes, through
// its JSON form: v's json tags are the keys these components report under. Numbers are decoded
// exactly: an integer stays an int64 (a float64 would round a count above 2^53), anything else
// a float64. A value that does not survive the trip is a programming error, so it panics.
func toGeneric(v any) any {
	data, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("toGeneric: %v", err))
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var out any
	if err := dec.Decode(&out); err != nil {
		panic(fmt.Sprintf("toGeneric: %v", err))
	}
	return exactNumbers(out)
}

// exactNumbers replaces each json.Number in v with an int64 when it is an integer, else a
// float64.
func exactNumbers(v any) any {
	switch x := v.(type) {
	case map[string]any:
		for k, e := range x {
			x[k] = exactNumbers(e)
		}
	case []any:
		for i, e := range x {
			x[i] = exactNumbers(e)
		}
	case json.Number:
		if i, err := x.Int64(); err == nil {
			return i
		}
		f, err := x.Float64()
		if err != nil {
			panic(fmt.Sprintf("toGeneric: number %q: %v", x, err))
		}
		return f
	}
	return v
}

// resultRuntime is the host- and wall-clock-dependent block.
func resultRuntime(started time.Time) *result.Runtime {
	return &result.Runtime{
		WallClockS: time.Since(started).Seconds(),
		StartedUTC: started.UTC().Format(time.RFC3339),
		Host:       map[string]string{"os": runtime.GOOS, "arch": runtime.GOARCH, "cpus": strconv.Itoa(runtime.NumCPU())},
	}
}

// resultDocs are the scenario and deployment documents a run read.
type resultDocs struct {
	Scenario   *scenario.Scenario
	Deployment *deployment.Deployment
}

// artifactFile is a bulk output a run wrote: its role in the document, path and row count.
type artifactFile struct {
	role string
	path string
	rows int
}

// resultArtifacts references each file by path and sha256. A file that cannot be read is
// an error: the document would otherwise point at nothing.
func resultArtifacts(files []artifactFile) (map[string]result.Artifact, error) {
	if len(files) == 0 {
		return nil, nil
	}
	out := make(map[string]result.Artifact, len(files))
	for _, f := range files {
		d, err := fileDigest(f.path)
		if err != nil {
			return nil, fmt.Errorf("artifact %s (%s): %w", f.role, f.path, err)
		}
		a := result.Artifact{Ref: artifactRef(f.path), SHA256: d}
		if f.rows > 0 {
			rows := f.rows
			a.Rows = &rows
		}
		out[f.role] = a
	}
	return out, nil
}

// artifactRef is path relative to the Result document, as blis-schemas states an artifact's
// ref: relative to --output's directory. With no --output the document has no location, so a
// relative path is recorded as the command was given it (cleaned) and an absolute one by its
// base name: a host's directory layout is not part of the result (INV-6), and the sha256
// identifies the file.
func artifactRef(path string) string {
	if resultOutput == "" {
		if filepath.IsAbs(path) {
			return filepath.Base(path)
		}
		return filepath.ToSlash(filepath.Clean(path))
	}
	absBase, err1 := filepath.Abs(filepath.Dir(resultOutput))
	absPath, err2 := filepath.Abs(path)
	if err1 != nil || err2 != nil {
		return filepath.Base(path)
	}
	rel, err := filepath.Rel(absBase, absPath)
	if err != nil {
		return filepath.Base(path)
	}
	return filepath.ToSlash(rel)
}

// traceArtifacts are the two files of an exported trace.
func traceArtifacts(prefix string, rows int) []artifactFile {
	if prefix == "" {
		return nil
	}
	return []artifactFile{
		{role: "trace_header", path: prefix + ".yaml"},
		{role: "trace_data", path: prefix + ".csv", rows: rows},
	}
}

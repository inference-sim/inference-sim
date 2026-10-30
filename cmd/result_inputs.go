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
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"time"

	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v3"

	blisschemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/embed"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/measurement"
	"github.com/inference-sim/blis-schemas/spec/result"
	"github.com/inference-sim/blis-schemas/spec/scenario"
	schemaworkload "github.com/inference-sim/blis-schemas/spec/workload"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// Module paths whose versions a Result records.
const (
	modulePathRouter = "github.com/llm-d/llm-d-router"
	modulePathEngine = "github.com/inference-sim/blis-vllm"
)

// resultRevisions are the versions of every component and data repository the run read.
// catalogRoot and registryRoot are the clone roots; a root that is not a git checkout is
// recorded as unknown, never guessed.
func resultRevisions(catalogRoot, registryRoot, engineVersion string) result.Revisions {
	info, _ := debug.ReadBuildInfo()
	return revisionsFrom(info, catalogRoot, registryRoot, engineVersion)
}

// revisionsFrom is resultRevisions over given build information, for tests.
func revisionsFrom(info *debug.BuildInfo, catalogRoot, registryRoot, engineVersion string) result.Revisions {
	v := versionsFrom(info)
	r := result.Revisions{BLIS: v.BLIS, BLISSchemas: v.Schemas, BLISLatencyKernel: v.Kernel}
	if r.BLISLatencyKernel == "unknown" {
		r.BLISLatencyKernel = ""
	}
	engineRelease := ""
	if info != nil {
		for _, dep := range info.Deps {
			switch dep.Path {
			case modulePathRouter:
				r.LLMDRouter = depVersion(dep)
			case modulePathEngine:
				engineRelease = depVersion(dep)
			}
		}
	}
	if engineVersion != "" {
		r.Engine = &result.Engine{Kind: "vllm", UpstreamVersion: engineVersion, Release: engineRelease}
	}
	r.Catalog = dataRepoRevision(catalogRoot)
	r.Registry = dataRepoRevision(registryRoot)
	return r
}

// dataRepoRevision is a data repository's git revision, with "-dirty" when its tree has
// uncommitted content; nil when no root was given. A root that is not a git checkout is
// recorded by sim.UnknownCatalogRevision.
func dataRepoRevision(root string) *result.DataRepo {
	if root == "" {
		return nil
	}
	rev, dirty, err := catalogGitState(root)
	if err != nil {
		logrus.Debugf("revision of %s: %v", root, err)
		return &result.DataRepo{Revision: sim.UnknownCatalogRevision}
	}
	if dirty {
		rev += "-dirty"
	}
	return &result.DataRepo{Revision: rev}
}

// resultScenarioDocs reads the scenario and deployment the run used, through blis-schemas' own
// strict bundle loader (the same decoding the kernel applies).
func resultScenarioDocs() (resultDocs, error) {
	path := filepath.Join(kernelScenarioDir, kernelScenario)
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
func traceLoad(trace *workload.TraceV2) measurement.Load {
	if trace == nil {
		return measurement.Load{}
	}
	at := make([]int64, len(trace.Records))
	for i, r := range trace.Records {
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
func traceBinding(dataPath string, h workload.TraceHeader, rows int) *schemaworkload.Binding {
	ref := &schemaworkload.TraceRef{Data: dataPath, Rows: rows, Header: schemaworkload.TraceHeader{
		Version: h.Version, TimeUnit: schemaworkload.TimeUnit(h.TimeUnit), Mode: schemaworkload.Mode(h.Mode),
		WorkloadSeed: h.WorkloadSeed,
	}}
	if d, err := fileDigest(dataPath); err == nil {
		ref.SHA256 = d
	}
	if len(h.GoodputSLOTargets) > 0 {
		ref.Header.GoodputSLOTargets = map[string]schemaworkload.SLODimTargets{}
		for cls, t := range h.GoodputSLOTargets {
			ref.Header.GoodputSLOTargets[cls] = schemaworkload.SLODimTargets{TTFTMs: t.TTFTMs, ITLMs: t.ITLMs, E2EMs: t.E2EMs}
		}
	}
	return &schemaworkload.Binding{Trace: ref}
}

// resultLedger is INV-1's ledger from the cluster's own counters: arrivals are observed at
// the cluster boundary, independently of the terminal buckets, so the validator's two
// clauses check the run rather than restate a sum.
func resultLedger(cs *cluster.ClusterSimulator) result.Requests {
	m := cs.AggregatedMetrics()
	rejected := cs.RejectedRequests()
	arrivals := cs.Arrivals()
	return result.Requests{
		Generated:               arrivals,
		Rejected:                rejected,
		Injected:                arrivals - rejected,
		Completed:               m.CompletedRequests,
		StillQueued:             m.StillQueued,
		StillRunning:            m.StillRunning,
		DroppedUnservable:       m.DroppedUnservable,
		TimedOut:                m.TimedOutRequests,
		RoutingRejections:       cs.RoutingRejections(),
		GatewayQueueDepth:       cs.GatewayQueueDepth(),
		GatewayQueueRejected:    cs.GatewayQueueRejected(),
		GatewayEvicted:          cs.GatewayEvicted(),
		GatewayExpired:          cs.GatewayExpired(),
		EncodeRoutingRejections: cs.EncodeRoutingRejections(),
		LengthCapped:            m.LengthCappedRequests,
	}
}

// runGPUs is the number of GPUs the run's instances occupy: each instance its own pool's
// tensor x pipeline x prefill-context-parallel width, the GPUs one engine replica spans.
// Data parallelism is already one instance per rank (DP-as-placement), so it is not a factor.
// An instance whose role names no pool (a test cluster) counts its tensor-parallel degree.
func runGPUs(cs *cluster.ClusterSimulator, dep *deployment.Deployment) int {
	perRole := map[deployment.Role]int{}
	if dep != nil {
		for _, p := range dep.Pools {
			perRole[p.Role] = max(p.Parallel.TP, 1) * max(p.Parallel.PP, 1) * max(p.Parallel.PCP, 1)
		}
	}
	membership := cs.PoolMembership()
	n := 0
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
		if g, ok := perRole[role]; ok {
			n += g
		} else {
			n += max(inst.TPDegree, 1)
		}
	}
	return n
}

// engineSection is what the engine side of the run reported: under "cluster" the aggregate
// (the
// step-interval series the legacy itl_* keys held, preemptions, KV allocation failures, the
// prefix-cache hit rate, the acceptance record and the server series), under each pool role
// its resolved prefix-cache record, and under "instances" each instance's own figures by id.
func engineSection(m *sim.Metrics, sum sim.MeasuredSummary, raw *cluster.RawMetrics, series *sim.ServerSeriesOutput,
	prefix map[string]sim.PrefixCacheRecord, acceptance *sim.AcceptanceRecord, perInstance map[string]any, agentic any) map[string]any {
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
	// The whole-run figures stdout printed before the Result (blis-schemas#41 section A: "whole-run
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
	return out
}

// instanceSections are each engine instance's own whole-run figures, keyed by instance id,
// as a multi-instance run printed them: the MetricsOutput of that instance.
func instanceSections(cs *cluster.ClusterSimulator) map[string]any {
	insts := cs.Instances()
	if len(insts) <= 1 {
		return nil
	}
	out := make(map[string]any, len(insts))
	for _, inst := range insts {
		out[string(inst.ID())] = toGeneric(inst.Metrics().BuildOutput(string(inst.ID())))
	}
	return out
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

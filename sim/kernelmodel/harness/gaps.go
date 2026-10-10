package harness

// coverage.go reports what the score does NOT cover, and why.
//
// A MAPE over the points that ran says nothing about the points that did not, or that ran
// with a setting the real benchmark did not use. This file walks the whole corpus and names,
// for every sweep or point that cannot be scored or is scored only approximately, the cause
// and the repository whose addition would close it. Every check reads the corpus, the
// scenario, the catalog, the registry or the extracted engine settings; none is a list of
// known-missing things, so the report shrinks by itself as those repositories grow.
//
// It never runs a simulation. Points that fail inside `blis run` are added by the scorer
// that ran them (Dropped).

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
)

// The repositories a gap can be closed in.
const (
	OwnerCatalog      = "blis-catalog"
	OwnerRegistry     = "blis-registry"
	OwnerSchemas      = "blis-schemas"
	OwnerKernel       = "blis-latency-kernel"
	OwnerSim          = "inference-sim"
	OwnerMeasurements = "measurements (InferenceX engine-settings extraction)"
)

// Gap is one sweep (Concurrency 0) or point that the score omits or approximates.
type Gap struct {
	Scenario    string
	Label       string
	Concurrency int
	// Points is how many corpus points the gap covers: all of a sweep's for a sweep-level
	// gap, one for a point-level gap.
	Points int
	Cause  string
	Owner  string
	Detail string
}

// Coverage is the gap report for one corpus.
type Coverage struct {
	Gaps []Gap
}

// appliedSettings are the measured engine settings the harness writes into a point's scenario
// (applyPointConfig). A passed setting outside this set is simulated at the scenario's value.
var appliedSettings = map[string]bool{
	"max_num_seqs": true, "max_num_batched_tokens": true, "block_size": true,
	"enable_prefix_caching": true,
}

// AssessCoverage walks every sweep and point of the corpus and records each gap. cfg supplies
// the artifact roots and the engine settings, exactly as a scoring run would use them.
func AssessCoverage(c *Corpus, cfg Config) *Coverage {
	cv := &Coverage{}
	for _, sw := range c.Sweeps {
		cv.assessSweep(sw, cfg)
	}
	return cv
}

func (cv *Coverage) sweepGap(sw Sweep, cause, owner, detail string) {
	cv.Gaps = append(cv.Gaps, Gap{Scenario: sw.Scenario, Label: sw.Label, Points: len(sw.Points),
		Cause: cause, Owner: owner, Detail: detail})
}

func (cv *Coverage) pointGap(sw Sweep, conc int, cause, owner, detail string) {
	cv.Gaps = append(cv.Gaps, Gap{Scenario: sw.Scenario, Label: sw.Label, Concurrency: conc, Points: 1,
		Cause: cause, Owner: owner, Detail: detail})
}

// Dropped records a sweep the scorer could not score because one of its points failed --
// `blis run` refused or errored, the run did not complete every request it was sent (a
// timeout, a drop, a length cap), or it observed nothing. A scorer takes a sweep whole or
// not at all, so the gap covers every point of the sweep; the failing point and its reason
// are the detail.
func (cv *Coverage) Dropped(sw Sweep, conc int, reason string) {
	if i := strings.LastIndex(reason, "\n"); i >= 0 {
		reason = strings.TrimSpace(reason[i+1:])
	}
	cv.sweepGap(sw, "sweep dropped: a point failed in blis run", OwnerSim,
		fmt.Sprintf("c=%d: %s", conc, reason))
}

// BadMeasurement records a sweep the scorer could not score because the measurement itself
// is unusable at a point -- a non-positive relative or anchor -- so no error against it can be
// computed. The gap is the measurement extraction's, not the simulator's.
func (cv *Coverage) BadMeasurement(sw Sweep, conc int, detail string) {
	cv.sweepGap(sw, "sweep dropped: a measured value is not usable", OwnerMeasurements,
		fmt.Sprintf("c=%d: %s", conc, detail))
}

func (cv *Coverage) assessSweep(sw Sweep, cfg Config) {
	// The engine. BLIS models vLLM's scheduler, and the engine settings come from vLLM's
	// args line, so a measurement taken under another engine is not apples-to-apples.
	// Not a return: the deployment checks below still apply, so a non-vLLM sweep whose model
	// is also missing from the catalog shows both gaps.
	vllm := sw.Framework == "vllm"
	if !vllm {
		cv.sweepGap(sw, "engine framework is not vLLM (BLIS models vLLM)", OwnerSim, sw.Framework)
	}
	if sw.Serving != "" && sw.Serving != "aggregated" {
		cv.sweepGap(sw, "serving mode the harness does not drive", OwnerSim, sw.Serving)
		return
	}
	if sw.SpecMethod != "" && sw.SpecMethod != "none" {
		cv.sweepGap(sw, "speculative method the harness does not drive", OwnerSim, sw.SpecMethod)
		return
	}
	if _, _, err := sw.ISLOSL(); err != nil {
		cv.sweepGap(sw, "workload label is not <isl>:<osl>", OwnerMeasurements, err.Error())
		return
	}

	// The deployment: the scenario file and everything it names.
	sc, dep, err := latencykernel.LoadBundle(filepath.Join(cfg.Repos.Scenarios, sw.Scenario))
	if err != nil {
		cv.sweepGap(sw, "no loadable scenario for the sweep", OwnerKernel, err.Error())
		return
	}
	if missing, owner, what := missingArtifact(sc.Model, sc.Cluster.Hardware, sc.Cluster.Fabric,
		sc.Coefficients, cfg.Repos); missing != "" {
		cv.sweepGap(sw, what, owner, missing)
		return
	}
	m, err := kernelmodel.Open(sw.Scenario, cfg.Repos)
	if err != nil {
		cv.sweepGap(sw, "kernel refuses the deployment", OwnerKernel, err.Error())
		return
	}
	eng, err := m.Engine()
	if err != nil {
		cv.sweepGap(sw, "scenario states no usable engine", OwnerKernel, err.Error())
		return
	}
	if len(dep.Pools) != 1 {
		cv.sweepGap(sw, "scenario states more than one pool", OwnerKernel, fmt.Sprintf("%d pools", len(dep.Pools)))
		return
	}
	pool := dep.Pools[0]

	// The corpus's own topology record against the scenario that claims to reproduce it.
	d := m.Deployment()
	for _, chk := range []struct {
		key   string
		state int
	}{{"tp_size", d.TP}, {"attention_dp_size", d.DP}, {"pp_size", max(pool.Parallel.PP, 1)}} {
		if v, ok := sw.Parallelism[chk.key]; ok && v > 0 && v != chk.state {
			cv.sweepGap(sw, "scenario parallelism differs from the corpus topology", OwnerKernel,
				fmt.Sprintf("%s: corpus %d, scenario %d", chk.key, v, chk.state))
		}
	}

	if !vllm {
		return // no vLLM args line exists for another engine, so there is nothing per point to compare
	}

	// Per point: where each admission setting came from, and every setting the run passed
	// that the simulation does not carry.
	rec := cfg.EngineSettings.For(sw)
	for _, p := range sw.Points {
		pc := resolveAdmission(sw, p.Concurrency, eng, d, cfg.EngineSettings)
		for _, f := range []struct {
			name string
			from SettingSource
		}{{"max_num_seqs", pc.SeqsFrom}, {"max_num_batched_tokens", pc.TokensFrom},
			{"block_size", pc.BlockFrom}, {"enable_prefix_caching", pc.PrefixFrom}} {
			if f.from != SourceMeasured {
				cv.pointGap(sw, p.Concurrency, "measured setting absent: "+f.name+" ("+string(f.from)+" value used)",
					OwnerMeasurements, "")
			}
		}
		ps := rec.At(p.Concurrency)
		if ps == nil {
			continue
		}
		keys := make([]string, 0, len(ps.PassedRaw))
		for k := range ps.PassedRaw {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			// "model" names the served checkpoint; the scenario's model identity is what
			// the catalog checks above already cover.
			if appliedSettings[k] || k == "model" {
				continue
			}
			passed := decodeAny(ps.PassedRaw[k])
			stated, known := schemaValue(pool, k)
			switch {
			case !known:
				cv.pointGap(sw, p.Concurrency, "run setting blis-schemas cannot express: "+k, OwnerSchemas,
					fmt.Sprintf("run passed %v", passed))
			case !reflect.DeepEqual(passed, stated):
				cv.pointGap(sw, p.Concurrency, "run setting not carried into the simulation: "+k, OwnerSim,
					fmt.Sprintf("run passed %v, scenario states %v", passed, describe(stated)))
			}
		}
	}
}

// missingArtifact names the first catalog or registry file the scenario needs and does not
// find, at the paths the kernel's own loader reads (latencykernel.OpenInputs).
func missingArtifact(model, hardware, fabric string, coefficients []string, r kernelmodel.Repos) (path, owner, cause string) {
	exists := func(p string) bool { _, err := os.Stat(p); return err == nil }
	if p := filepath.Join(r.Catalog, "models", model, "graph.yaml"); !exists(p) {
		return model, OwnerCatalog, "model not in blis-catalog"
	}
	if p := filepath.Join(r.Catalog, "hardware", hardware+".yaml"); !exists(p) {
		return hardware, OwnerCatalog, "hardware not in blis-catalog"
	}
	if fabric != "" {
		if p := filepath.Join(r.Catalog, "networks", fabric+".yaml"); !exists(p) {
			return fabric, OwnerCatalog, "fabric not in blis-catalog"
		}
	}
	for _, name := range coefficients {
		if p := filepath.Join(r.Registry, "coefficients", name+".yaml"); !exists(p) {
			return name, OwnerRegistry, "coefficient set not in blis-registry"
		}
	}
	return "", "", ""
}

// schemaValue returns the pool's value for a setting named by its blis-schemas YAML key, in
// Engine or Parallelism, decoded the way a JSON measurement is (so the two compare), and
// whether the schema has such a field at all. An unstated pointer field reads as nil.
func schemaValue(pool deployment.Pool, key string) (any, bool) {
	for _, v := range []reflect.Value{reflect.ValueOf(pool.Engine), reflect.ValueOf(pool.Parallel)} {
		t := v.Type()
		for i := 0; i < t.NumField(); i++ {
			tag := strings.Split(t.Field(i).Tag.Get("yaml"), ",")[0]
			if tag != key {
				continue
			}
			f := v.Field(i)
			if f.Kind() == reflect.Pointer && f.IsNil() {
				return nil, true
			}
			if f.IsZero() {
				return nil, true // omitempty: unstated
			}
			raw, err := json.Marshal(f.Interface())
			if err != nil {
				return nil, true
			}
			return decodeAny(raw), true
		}
	}
	return nil, false
}

func decodeAny(raw json.RawMessage) any {
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return string(raw)
	}
	return v
}

func describe(v any) string {
	if v == nil {
		return "nothing (engine default)"
	}
	return fmt.Sprint(v)
}

// Write prints every gap, one line each, then a summary of points per cause, largest first,
// so a reader sees which addition would widen apples-to-apples coverage the most. A point
// with several gaps counts once under each.
func (cv *Coverage) Write(w io.Writer) error {
	var b strings.Builder
	fmt.Fprintf(&b, "=== Coverage gaps: corpus points the score omits or approximates\n")
	for _, g := range cv.Gaps {
		where := "sweep"
		if g.Concurrency > 0 {
			where = fmt.Sprintf("c=%d", g.Concurrency)
		}
		line := fmt.Sprintf("gap %s %s %s [%s] %s", g.Scenario, g.Label, where, g.Owner, g.Cause)
		if g.Detail != "" {
			line += ": " + g.Detail
		}
		fmt.Fprintln(&b, line)
	}
	type key struct{ cause, owner string }
	points := map[key]int{}
	// Point-level gaps whose cause varies only in a detail are already one cause; sum points.
	for _, g := range cv.Gaps {
		points[key{g.Cause, g.Owner}] += g.Points
	}
	keys := make([]key, 0, len(points))
	for k := range points {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if points[keys[i]] != points[keys[j]] {
			return points[keys[i]] > points[keys[j]]
		}
		return keys[i].cause < keys[j].cause
	})
	fmt.Fprintf(&b, "\n=== Coverage gap summary (points per cause, largest first)\n")
	fmt.Fprintf(&b, "%7s  %-52s %s\n", "points", "owner", "cause")
	for _, k := range keys {
		fmt.Fprintf(&b, "%7d  %-52s %s\n", points[k], k.owner, k.cause)
	}
	_, err := io.WriteString(w, b.String())
	return err
}

// WriteGaps writes the report to path, or to stderr when path is empty -- never to stdout,
// which carries the score tables.
func (cv *Coverage) WriteGaps(path string) error {
	if path == "" {
		return cv.Write(os.Stderr)
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := cv.Write(f); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// Package prereg holds the calibration acceptance gates a refit is fixed against before it fits
// (M3.2 plan Table W row 18; design VAL-6 and section 12.3).
//
// A pre-registration is a data file under registry/gates/preregistrations/. It names the
// coefficient sets a refit may rewrite, the sources it may read and the roles it may read them
// at, its success criteria and their margins, and pins what it was registered against: the
// partition, the seed's hash, the gate defaults and each source's revision and cut-points. The
// thresholds come from registry/gates/defaults.yaml (requirements section 1.5) unless a record
// names the merged pull request in which a maintainer set another.
//
// Verify checks the committed files against the ledger, the coefficient sets and the engines'
// code IDs. Admit is what a fit calls before it reads a source: it refuses a source, role or
// pin the record does not cover. ScanScripts finds fit scripts that name a source no record
// covers. No function here reads a unit or runs a fit.
package prereg

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/substrate/contracts/check"
	"github.com/inference-sim/inference-sim/substrate/ledger"
)

const (
	FileDefaults    = "defaults.yaml"
	DirRecords      = "preregistrations"
	KindDefaults    = "CalibrationGateDefaults"
	KindRecord      = "CalibrationPreRegistration"
	ClassKernel     = "kernel-keyed"
	ClassEnginePath = "engine-path"
)

// Step is what a fit does with a role: it fits on one share and selects on others (X8).
type Step string

const (
	StepFit    Step = "fit"
	StepSelect Step = "select"
)

// Defaults is defaults.yaml: the section 1.5 thresholds and the rules every record shares.
type Defaults struct {
	Kind     string `yaml:"kind"`
	Baseline string `yaml:"baseline"`
	Error    struct {
		Definition string   `yaml:"definition"`
		Metrics    []string `yaml:"metrics"`
		Per        string   `yaml:"per"`
	} `yaml:"error"`
	Slices    []string `yaml:"slices"`
	MinPoints int      `yaml:"min_points"`
	Margin    Margin   `yaml:"margin"`
	Test      struct {
		Method       string  `yaml:"method"`
		Resamples    int     `yaml:"resamples"`
		Alpha        float64 `yaml:"alpha"`
		Correction   string  `yaml:"correction"`
		ResampleOver string  `yaml:"resample_over"`
	} `yaml:"test"`
	Heldout struct {
		Evaluations int    `yaml:"evaluations"`
		After       string `yaml:"after"`
	} `yaml:"heldout"`
	Revision struct {
		Retries                      int    `yaml:"retries"`
		RetryNeedsNewPreregistration bool   `yaml:"retry_needs_new_preregistration"`
		Then                         string `yaml:"then"`
		NegativeResultNote           bool   `yaml:"negative_result_note"`
	} `yaml:"revision"`
	RolesDrawnPer    string              `yaml:"roles_drawn_per"`
	TrainingRoles    []ledger.Role       `yaml:"training_roles"`
	FitReads         []ledger.Role       `yaml:"fit_reads"`
	SelectReads      []ledger.Role       `yaml:"select_reads"`
	RoleKeyFields    map[string][]string `yaml:"role_key_fields"`
	ForbiddenSources []string            `yaml:"forbidden_sources"`
	GatedSources     []string            `yaml:"gated_sources"`
	FitScriptDirs    []string            `yaml:"fit_script_dirs"`
	LaneFields       []string            `yaml:"lane_fields"`
	// NotPreregistered lists coefficient sets no record covers, each with the reason. It is not
	// a threshold, so it is outside the digest the records pin.
	NotPreregistered []Uncovered `yaml:"not_preregistered_sets"`
}

// Uncovered is a coefficient set no pre-registration covers yet.
type Uncovered struct {
	Set    string `yaml:"set"`
	Reason string `yaml:"reason"`
}

// Margin is the non-inferiority margin d of each criterion (VAL-6).
type Margin struct {
	Error    float64 `yaml:"error"`
	Distance float64 `yaml:"distance"`
}

// Record is one pre-registration: one refit family.
type Record struct {
	Kind             string `yaml:"kind"`
	ID               string `yaml:"id"`
	PartitionVersion int    `yaml:"partition_version"`
	Hypothesis       string `yaml:"hypothesis"`
	Layer            string `yaml:"layer"`
	PredictedChange  string `yaml:"predicted_change"`

	// CoefficientClass is kernel-keyed (shared by the engines, design section 3.2) or
	// engine-path (keyed by the engine's CodeID, so one engine's).
	CoefficientClass string   `yaml:"coefficient_class"`
	CoefficientSets  []string `yaml:"coefficient_sets"`
	TargetEngines    []string `yaml:"target_engines"`
	// EngineCodeIDs is the engine-path record's evidence that it is one engine's: the CodeIDs
	// of the host and launch code, as registry/inputs/engine_code_ids.json states them.
	EngineCodeIDs map[string]CodeIDs `yaml:"engine_code_ids,omitempty"`

	Sources     []string            `yaml:"sources"`
	TargetLanes []map[string]string `yaml:"target_lanes"`
	Criteria    []Criterion         `yaml:"criteria"`
	Metrics     []string            `yaml:"metrics"`
	MinPoints   int                 `yaml:"min_points"`
	Substream   string              `yaml:"substream"`
	// MeasuredSHA256 is the hash of the measured table the evaluation reads, which the gate
	// fixes with the base gate-candidate run. Empty until then.
	MeasuredSHA256    string   `yaml:"measured_sha256"`
	ThresholdOverride Override `yaml:"threshold_override,omitempty"`
	Pins              Pins     `yaml:"pins"`
}

// CodeIDs are an engine's host-path and launch-path CodeIDs.
type CodeIDs struct {
	Host   string `yaml:"host"`
	Graphs string `yaml:"graphs"`
}

// Criterion is one success criterion and its margin.
type Criterion struct {
	Criterion string  `yaml:"criterion"`
	Margin    float64 `yaml:"margin"`
}

// Override records the thresholds a maintainer set in a merged pull request (requirements 1.5).
// A zero Override means every threshold is the default; otherwise it gives all of them.
type Override struct {
	PullRequest string `yaml:"pull_request,omitempty"`
	SetBy       string `yaml:"set_by,omitempty"`
	MinPoints   int    `yaml:"min_points,omitempty"`
	Margin      Margin `yaml:"margin,omitempty"`
}

func (o Override) active() bool { return o != (Override{}) }

// Pins is what a record was registered against. A record whose pins differ from the files they
// were taken from is stale: the family must be registered again before it fits.
type Pins struct {
	PartitionVersion int               `yaml:"partition_version"`
	SeedSHA256       string            `yaml:"seed_sha256"`
	DefaultsSHA256   string            `yaml:"defaults_sha256"`
	Sources          map[string]string `yaml:"sources"`
	CutPoints        map[string]string `yaml:"cut_points"`
}

// Registry is the loaded gate directory.
type Registry struct {
	Defaults Defaults
	Records  []Record
}

// Env locates what Verify checks the records against.
type Env struct {
	LedgerDir       string // catalog/ledger
	CoefficientsDir string // registry/coefficients
	EngineCodeIDs   string // registry/inputs/engine_code_ids.json
}

func decodeFile(path string, dst any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return validate.DecodeStrict(path, raw, dst)
}

// Load reads defaults.yaml and every record in dir strictly (SUB-12). A record's file name is
// its id.
func Load(dir string) (*Registry, error) {
	var r Registry
	if err := decodeFile(filepath.Join(dir, FileDefaults), &r.Defaults); err != nil {
		return nil, err
	}
	paths, err := filepath.Glob(filepath.Join(dir, DirRecords, "*.yaml"))
	if err != nil {
		return nil, err
	}
	sort.Strings(paths)
	var errs []error
	for _, p := range paths {
		var rec Record
		if err := decodeFile(p, &rec); err != nil {
			errs = append(errs, err)
			continue
		}
		if want := strings.TrimSuffix(filepath.Base(p), ".yaml"); rec.ID != want {
			errs = append(errs, fmt.Errorf("%s: id %q, but the file name says %q", p, rec.ID, want))
		}
		r.Records = append(r.Records, rec)
	}
	return &r, errors.Join(errs...)
}

// Record returns the record with the id, or nil.
func (r *Registry) Record(id string) *Record {
	for i := range r.Records {
		if r.Records[i].ID == id {
			return &r.Records[i]
		}
	}
	return nil
}

// DefaultsDigest is the SHA-256 of the thresholds and rules of the defaults: everything but the
// list of sets no record covers.
func (d Defaults) DefaultsDigest() (string, error) {
	d.NotPreregistered = nil
	b, err := json.Marshal(d)
	if err != nil {
		return "", err
	}
	return ledger.ID(map[string]any{"defaults": string(b)})
}

// generic turns v into the JSON-shaped value ledger.Canonical accepts.
func generic(v any) (any, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var out any
	if err := dec.Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

// ComputePins takes the pins of the given sources from the ledger in ledgerDir and the defaults.
func ComputePins(ledgerDir string, d Defaults, sources []string) (Pins, error) {
	cfg, err := ledger.LoadConfig(ledgerDir)
	if err != nil {
		return Pins{}, err
	}
	srcs, err := ledger.LoadSources(ledgerDir)
	if err != nil {
		return Pins{}, err
	}
	dd, err := d.DefaultsDigest()
	if err != nil {
		return Pins{}, err
	}
	p := Pins{PartitionVersion: cfg.PartitionVersion, SeedSHA256: cfg.Seed.SHA256, DefaultsSHA256: dd,
		Sources: map[string]string{}, CutPoints: map[string]string{}}
	for _, id := range sources {
		sp, err := sourcePin(cfg, srcs, id)
		if err != nil {
			return Pins{}, err
		}
		p.Sources[id], p.CutPoints[id] = sp.revision, sp.cutPoints
	}
	return p, nil
}

type sourcePins struct{ revision, cutPoints string }

func sourcePin(cfg *ledger.Config, srcs *ledger.Sources, id string) (sourcePins, error) {
	s := srcs.Lookup(id)
	rule, ok := cfg.Rules[id]
	if s == nil || !ok {
		return sourcePins{}, fmt.Errorf("source %q: not in the ledger's sources.yaml and ledger.yaml", id)
	}
	rev, err := generic(map[string]any{"revision": s.Revision, "artifacts": s.Artifacts})
	if err != nil {
		return sourcePins{}, err
	}
	cp, err := generic(rule)
	if err != nil {
		return sourcePins{}, err
	}
	var out sourcePins
	if out.revision, err = ledger.ID(rev); err != nil {
		return sourcePins{}, err
	}
	if out.cutPoints, err = ledger.ID(cp); err != nil {
		return sourcePins{}, err
	}
	return out, nil
}

var (
	idRe        = regexp.MustCompile(`^[a-z0-9][a-z0-9.-]{2,63}$`)
	metricRe    = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)
	substreamRe = regexp.MustCompile(`^[a-z0-9][a-z0-9._/-]{2,63}$`)
	hex64Re     = regexp.MustCompile(`^[0-9a-f]{64}$`)
	prURLRe     = regexp.MustCompile(`^https://github\.com/inference-sim/[A-Za-z0-9._-]+/pull/[0-9]+$`)
)

func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) && x >= 0 }

func contains[T comparable](xs []T, x T) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

func sentence(errs *[]error, at, s string) {
	if len(strings.TrimSpace(s)) < 10 {
		*errs = append(*errs, fmt.Errorf("%s: a sentence is required (at least 10 characters)", at))
	}
}

// Verify checks the defaults and every record. It fails on a record that is malformed, that
// departs from a default without a merged pull request, that is stale against the ledger, or
// that leaves a coefficient set or a gated source uncovered.
func (r *Registry) Verify(env Env) error {
	errs := r.verifyDefaults(env)
	cfg, err := ledger.LoadConfig(env.LedgerDir)
	if err != nil {
		return errors.Join(append(errs, err)...)
	}
	srcs, err := ledger.LoadSources(env.LedgerDir)
	if err != nil {
		return errors.Join(append(errs, err)...)
	}
	engines, err := readCodeIDs(env.EngineCodeIDs)
	if err != nil {
		errs = append(errs, err)
	}
	seen := map[string]bool{}
	sets := map[string][]string{}
	covered := map[string]bool{}
	for i := range r.Records {
		rec := &r.Records[i]
		if seen[rec.ID] {
			errs = append(errs, fmt.Errorf("%s: two records have this id", rec.ID))
		}
		seen[rec.ID] = true
		errs = append(errs, r.verifyRecord(rec, cfg, srcs, engines)...)
		for _, s := range rec.CoefficientSets {
			sets[s] = append(sets[s], rec.ID)
		}
		for _, s := range rec.Sources {
			covered[s] = true
		}
	}
	for _, g := range r.Defaults.GatedSources {
		if !covered[g] {
			errs = append(errs, fmt.Errorf("source %s is gated and no pre-registration covers it", g))
		}
	}
	errs = append(errs, r.verifySetCoverage(env, sets)...)
	return errors.Join(errs...)
}

func (r *Registry) verifyDefaults(env Env) []error {
	d := &r.Defaults
	var errs []error
	e := func(f string, a ...any) { errs = append(errs, fmt.Errorf("defaults: "+f, a...)) }
	if d.Kind != KindDefaults {
		e("kind: %q, want %q", d.Kind, KindDefaults)
	}
	if d.Baseline == "" || d.Error.Definition == "" || d.Error.Per == "" || len(d.Error.Metrics) == 0 || len(d.Slices) == 0 {
		e("baseline, error and slices are required")
	}
	if d.MinPoints < 1 {
		e("min_points: %d, want at least 1", d.MinPoints)
	}
	if !finite(d.Margin.Error) || !finite(d.Margin.Distance) {
		e("margin: finite and zero or more")
	}
	if d.Test.Resamples < 1 || d.Test.Alpha <= 0 || d.Test.Alpha >= 1 || d.Test.Method == "" || d.Test.Correction == "" {
		e("test: method, correction, resamples and an alpha in (0, 1) are required")
	}
	if d.Heldout.Evaluations != 1 || d.Heldout.After != "selection" {
		e("heldout: the held-out verdict is read once, after selection (section 12.3)")
	}
	if d.Revision.Retries != 1 || !d.Revision.RetryNeedsNewPreregistration || d.Revision.Then != "revert" {
		e("revision: one pre-registered retry, then revert (requirements 1.5)")
	}
	if d.RolesDrawnPer != "role-key" {
		e("roles_drawn_per: %q, want role-key (requirements 1.4)", d.RolesDrawnPer)
	}
	for name, roles := range map[string][]ledger.Role{"fit_reads": d.FitReads, "select_reads": d.SelectReads} {
		if len(roles) == 0 {
			e("%s: empty", name)
		}
		for _, role := range roles {
			if !contains(d.TrainingRoles, role) {
				e("%s: %q is not a training role %v: a fit never reads held, anchor or report", name, role, d.TrainingRoles)
			}
		}
	}
	for _, a := range d.FitReads {
		if contains(d.SelectReads, a) {
			e("fit_reads and select_reads share %q: selection reads the share the fit did not", a)
		}
	}
	for _, g := range d.GatedSources {
		if contains(d.ForbiddenSources, g) {
			e("source %s is both gated and forbidden", g)
		}
	}
	cfg, err := ledger.LoadConfig(env.LedgerDir)
	if err == nil {
		for _, s := range append(append([]string{}, d.GatedSources...), d.ForbiddenSources...) {
			if _, ok := cfg.Rules[s]; !ok {
				e("source %s is not in ledger.yaml", s)
			}
		}
		for s := range d.RoleKeyFields {
			if _, ok := cfg.Rules[s]; !ok {
				e("role_key_fields: source %s is not in ledger.yaml", s)
			}
		}
	}
	for i, u := range d.NotPreregistered {
		if u.Set == "" || strings.TrimSpace(u.Reason) == "" {
			e("not_preregistered_sets[%d]: a set and a reason are required", i)
		}
	}
	return errs
}

func (r *Registry) verifyRecord(rec *Record, cfg *ledger.Config, srcs *ledger.Sources, engines map[string]CodeIDs) []error {
	d := &r.Defaults
	var errs []error
	e := func(f string, a ...any) { errs = append(errs, fmt.Errorf(rec.ID+": "+f, a...)) }
	if rec.Kind != KindRecord {
		e("kind: %q, want %q", rec.Kind, KindRecord)
	}
	if !idRe.MatchString(rec.ID) {
		e("id: lowercase letters, digits, dots and dashes, 3 to 64 characters")
	}
	for f, s := range map[string]string{"hypothesis": rec.Hypothesis, "layer": rec.Layer, "predicted_change": rec.PredictedChange} {
		sentence(&errs, rec.ID+"."+f, s)
	}
	if rec.PartitionVersion != cfg.PartitionVersion {
		e("partition_version: %d, but the ledger is at %d", rec.PartitionVersion, cfg.PartitionVersion)
	}
	errs = append(errs, r.verifyScope(rec, engines)...)
	errs = append(errs, r.verifySources(rec, cfg)...)
	errs = append(errs, r.verifyGate(rec)...)
	if rec.MeasuredSHA256 != "" && !hex64Re.MatchString(rec.MeasuredSHA256) {
		e("measured_sha256: empty until the base gate-candidate run fixes it, then a SHA-256")
	}
	if ov := rec.ThresholdOverride; ov.active() {
		if !prURLRe.MatchString(ov.PullRequest) || ov.SetBy == "" || ov.MinPoints < 1 || !finite(ov.Margin.Error) || !finite(ov.Margin.Distance) {
			e("threshold_override: a threshold other than the default needs the merged pull request (https://github.com/inference-sim/<repo>/pull/N), who set it, and all of min_points and margin")
		}
	}
	errs = append(errs, r.verifyPins(rec, cfg, srcs, d)...)
	return errs
}

func (r *Registry) verifyScope(rec *Record, engines map[string]CodeIDs) []error {
	var errs []error
	e := func(f string, a ...any) { errs = append(errs, fmt.Errorf(rec.ID+": "+f, a...)) }
	if len(rec.CoefficientSets) == 0 {
		e("coefficient_sets: at least one")
	}
	if len(rec.TargetEngines) == 0 {
		e("target_engines: at least one")
	}
	for _, g := range rec.TargetEngines {
		if _, ok := engines[g]; !ok && engines != nil {
			e("target_engines: %q is not an engine in engine_code_ids.json", g)
		}
	}
	switch rec.CoefficientClass {
	case ClassKernel:
		if len(rec.EngineCodeIDs) != 0 {
			e("engine_code_ids: a kernel-keyed record is shared across engines and names no engine's code")
		}
		if len(rec.TargetEngines) < 2 {
			e("target_engines: a kernel-keyed coefficient is shared, so every engine is a lane it must not harm (design section 3.2): %v", rec.TargetEngines)
		}
	case ClassEnginePath:
		if len(rec.TargetEngines) != 1 {
			e("target_engines: an engine-path record is one engine's, got %v", rec.TargetEngines)
			break
		}
		g := rec.TargetEngines[0]
		if len(rec.EngineCodeIDs) != 1 || rec.EngineCodeIDs[g] == (CodeIDs{}) {
			e("engine_code_ids: exactly the CodeIDs of %s", g)
		} else if want, ok := engines[g]; ok && rec.EngineCodeIDs[g] != want {
			e("engine_code_ids[%s]: %+v, but engine_code_ids.json states %+v", g, rec.EngineCodeIDs[g], want)
		}
	default:
		e("coefficient_class: %q, want %s or %s", rec.CoefficientClass, ClassKernel, ClassEnginePath)
	}
	return errs
}

func (r *Registry) verifySources(rec *Record, cfg *ledger.Config) []error {
	d := &r.Defaults
	var errs []error
	e := func(f string, a ...any) { errs = append(errs, fmt.Errorf(rec.ID+": "+f, a...)) }
	if len(rec.Sources) == 0 {
		e("sources: at least one")
	}
	seen := map[string]bool{}
	for _, s := range rec.Sources {
		if seen[s] {
			e("sources: %s is listed twice", s)
		}
		seen[s] = true
		rule, ok := cfg.Rules[s]
		switch {
		case contains(d.ForbiddenSources, s):
			e("sources: %s is anchor-only and is never fit, selected or validated (requirements 1.4)", s)
		case !ok:
			e("sources: %s has no rule in ledger.yaml", s)
		default:
			have := map[ledger.Role]bool{}
			for _, p := range rule.CutPoints {
				have[p.Role] = true
			}
			for _, role := range append(append([]ledger.Role{}, d.FitReads...), d.SelectReads...) {
				if !have[role] {
					e("sources: %s has no %s share, but the fit or selection reads it", s, role)
				}
			}
		}
	}
	return errs
}

func (r *Registry) verifyGate(rec *Record) []error {
	d := &r.Defaults
	var errs []error
	e := func(f string, a ...any) { errs = append(errs, fmt.Errorf(rec.ID+": "+f, a...)) }
	if len(rec.TargetLanes) == 0 {
		e("target_lanes: at least one lane")
	}
	laneEngines := map[string]bool{}
	for i, lane := range rec.TargetLanes {
		if len(lane) == 0 {
			e("target_lanes[%d]: empty", i)
		}
		for k, v := range lane {
			if !contains(d.LaneFields, k) || v == "" {
				e("target_lanes[%d].%s: a lane field is one of %v with a non-empty value", i, k, d.LaneFields)
			}
		}
		laneEngines[lane["engine"]] = true
	}
	for _, g := range rec.TargetEngines {
		if !laneEngines[g] {
			e("target_lanes: no lane for engine %s", g)
		}
	}
	for g := range laneEngines {
		if !contains(rec.TargetEngines, g) {
			e("target_lanes: engine %q is not a target engine of the record", g)
		}
	}
	if len(rec.Metrics) == 0 {
		e("metrics: at least one")
	}
	for _, m := range rec.Metrics {
		if !metricRe.MatchString(m) || !contains(d.Error.Metrics, m) {
			e("metrics: %q is not one of the default metrics %v", m, d.Error.Metrics)
		}
	}
	if !substreamRe.MatchString(rec.Substream) {
		e("substream: the gate's bootstrap substream name (X6), 3 to 64 characters of [a-z0-9._/-]")
	}
	wantMin, wantMargin := d.MinPoints, d.Margin
	if rec.ThresholdOverride.active() {
		wantMin, wantMargin = rec.ThresholdOverride.MinPoints, rec.ThresholdOverride.Margin
	}
	if rec.MinPoints != wantMin {
		e("min_points: %d, but the %s is %d", rec.MinPoints, source(rec), wantMin)
	}
	seen := map[string]bool{}
	for _, c := range rec.Criteria {
		var want float64
		switch c.Criterion {
		case "error":
			want = wantMargin.Error
		case "distance":
			want = wantMargin.Distance
		default:
			e("criteria: %q, want error or distance", c.Criterion)
			continue
		}
		if seen[c.Criterion] {
			e("criteria: %s is listed twice", c.Criterion)
		}
		seen[c.Criterion] = true
		if !finite(c.Margin) || c.Margin != want {
			e("criteria: %s margin %v, but the %s is %v", c.Criterion, c.Margin, source(rec), want)
		}
	}
	if !seen["error"] || !seen["distance"] {
		e("criteria: both error and distance are required")
	}
	return errs
}

func source(rec *Record) string {
	if rec.ThresholdOverride.active() {
		return "threshold the override sets"
	}
	return "section 1.5 default"
}

func (r *Registry) verifyPins(rec *Record, cfg *ledger.Config, srcs *ledger.Sources, d *Defaults) []error {
	var errs []error
	e := func(f string, a ...any) { errs = append(errs, fmt.Errorf(rec.ID+": "+f, a...)) }
	p := rec.Pins
	if p.PartitionVersion != cfg.PartitionVersion {
		e("pins.partition_version: %d, the ledger is at %d", p.PartitionVersion, cfg.PartitionVersion)
	}
	if p.SeedSHA256 != cfg.Seed.SHA256 {
		e("pins.seed_sha256: the seed changed since the record was registered")
	}
	if dd, err := d.DefaultsDigest(); err != nil {
		errs = append(errs, err)
	} else if p.DefaultsSHA256 != dd {
		e("pins.defaults_sha256: the defaults changed since the record was registered (now %s)", dd)
	}
	for _, s := range rec.Sources {
		sp, err := sourcePin(cfg, srcs, s)
		if err != nil {
			continue
		}
		if p.Sources[s] != sp.revision {
			e("pins.sources[%s]: the source's pinned revision changed (now %s); register the family again before it fits", s, sp.revision)
		}
		if p.CutPoints[s] != sp.cutPoints {
			e("pins.cut_points[%s]: the source's cut-points changed (now %s); register the family again before it fits", s, sp.cutPoints)
		}
	}
	for s := range p.Sources {
		if !contains(rec.Sources, s) {
			e("pins.sources[%s]: pins a source the record does not list", s)
		}
	}
	for s := range p.CutPoints {
		if !contains(rec.Sources, s) {
			e("pins.cut_points[%s]: pins a source the record does not list", s)
		}
	}
	return errs
}

// verifySetCoverage requires every coefficient set under CoefficientsDir to be covered by a
// record or listed with a reason, and every set a record names to exist. A kernel-keyed set is
// in one record; an engine-path set may be in one record per engine.
func (r *Registry) verifySetCoverage(env Env, sets map[string][]string) []error {
	var errs []error
	paths, err := filepath.Glob(filepath.Join(env.CoefficientsDir, "*.yaml"))
	if err != nil || len(paths) == 0 {
		return []error{fmt.Errorf("no coefficient sets under %s", env.CoefficientsDir)}
	}
	on := map[string]bool{}
	for _, p := range paths {
		on[strings.TrimSuffix(filepath.Base(p), ".yaml")] = true
	}
	unc := map[string]bool{}
	for _, u := range r.Defaults.NotPreregistered {
		unc[u.Set] = true
		if !on[u.Set] {
			errs = append(errs, fmt.Errorf("defaults: not_preregistered_sets names %s, which is not a coefficient set", u.Set))
		}
		if _, in := sets[u.Set]; in {
			errs = append(errs, fmt.Errorf("defaults: %s is listed as not pre-registered, and record(s) %v cover it", u.Set, sets[u.Set]))
		}
	}
	for set := range on {
		if _, in := sets[set]; !in && !unc[set] {
			errs = append(errs, fmt.Errorf("coefficient set %s is in no pre-registration and is not listed in not_preregistered_sets: a refit of it would read sources no record covers", set))
		}
	}
	for set, ids := range sets {
		if !on[set] {
			errs = append(errs, fmt.Errorf("%v: coefficient set %s does not exist under %s", ids, set, env.CoefficientsDir))
		}
		if len(ids) < 2 {
			continue
		}
		engineOf := map[string]bool{}
		for _, id := range ids {
			rec := r.Record(id)
			if rec.CoefficientClass != ClassEnginePath {
				errs = append(errs, fmt.Errorf("coefficient set %s is in %v: only an engine-path set is in more than one record", set, ids))
				break
			}
			for _, g := range rec.TargetEngines {
				if engineOf[g] {
					errs = append(errs, fmt.Errorf("coefficient set %s is in two records for engine %s", set, g))
				}
				engineOf[g] = true
			}
		}
	}
	return errs
}

func readCodeIDs(path string) (map[string]CodeIDs, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Engines map[string]struct {
			Host   string `json:"host"`
			Graphs string `json:"graphs"`
		} `json:"engines"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	out := map[string]CodeIDs{}
	for k, v := range doc.Engines {
		out[k] = CodeIDs{Host: v.Host, Graphs: v.Graphs}
	}
	return out, nil
}

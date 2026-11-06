package ledger

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"sort"
)

// The candidate job's only output is a predictions file with this schema and size limit, which
// the seed-holding job reads with a strict loader as untrusted input (D7).
const (
	// MaxPredictionsBytes bounds the predictions file.
	MaxPredictionsBytes = 64 << 20
	// MinAggregateUnits is the fewest units an aggregate may cover (D7).
	MinAggregateUnits = 10
)

// Predictions is the candidate job's output: one prediction per unit it was given.
type Predictions struct {
	Kind string `json:"kind"` // "Predictions"
	// Candidate is the commit the candidate job built.
	Candidate string `json:"candidate"`
	// Harness is the scoring harness's digest, for whole-run predictions (design §13.5).
	Harness     string       `json:"harness,omitempty"`
	Predictions []Prediction `json:"predictions"`
}

// Prediction is one unit's predicted values, by metric name. A unit the candidate could not
// run has no values and a reason.
type Prediction struct {
	Source string             `json:"source"`
	UnitID string             `json:"unit_id"`
	Values map[string]float64 `json:"values,omitempty"`
	Reason string             `json:"reason,omitempty"`
}

var metricName = regexpMust(`^[a-z][a-z0-9_]{0,63}$`)

// ReadPredictions reads a predictions file strictly: at most MaxPredictionsBytes, no unknown
// field, no trailing data, each unit once, each value finite, each name a short identifier,
// the candidate commit equal to wantCandidate, the head of the run the file came from, which
// the caller learns from Actions and not from the file (X22), every prediction for a unit in want (the units the job handed the candidate) and every unit
// of want predicted: coverage is complete and fixed by the job, never chosen by the file
// (D7). It never trusts the file to say which units are which role.
func ReadPredictions(r io.Reader, want map[string]bool, wantCandidate string) (*Predictions, error) {
	if !gitCommit.MatchString(wantCandidate) {
		return nil, fmt.Errorf("predictions: expected candidate %q is not a full commit", wantCandidate)
	}
	raw, err := io.ReadAll(io.LimitReader(r, MaxPredictionsBytes+1))
	if err != nil {
		return nil, err
	}
	if len(raw) > MaxPredictionsBytes {
		return nil, fmt.Errorf("predictions file exceeds %d bytes", MaxPredictionsBytes)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var p Predictions
	if err := dec.Decode(&p); err != nil {
		return nil, fmt.Errorf("predictions: %w", err)
	}
	if dec.More() {
		return nil, errors.New("predictions: trailing data")
	}
	if p.Kind != "Predictions" {
		return nil, fmt.Errorf("predictions: kind %q, want \"Predictions\"", p.Kind)
	}
	if !gitCommit.MatchString(p.Candidate) {
		return nil, fmt.Errorf("predictions: candidate %q is not a full commit", p.Candidate)
	}
	if p.Candidate != wantCandidate {
		return nil, fmt.Errorf("predictions: candidate %s is not the commit %s of the run they came from (X22)", p.Candidate, wantCandidate)
	}
	seen := map[string]bool{}
	for i, x := range p.Predictions {
		k := x.Source + "/" + x.UnitID
		switch {
		case !want[k]:
			return nil, fmt.Errorf("predictions[%d]: unit %s was not handed to the candidate", i, k)
		case seen[k]:
			return nil, fmt.Errorf("predictions[%d]: unit %s predicted twice", i, k)
		case len(x.Values) == 0 && x.Reason == "":
			return nil, fmt.Errorf("predictions[%d]: no values and no reason", i)
		case len(x.Values) > 0 && x.Reason != "":
			return nil, fmt.Errorf("predictions[%d]: values and a reason; a reason means no prediction", i)
		case len(x.Reason) > 200:
			return nil, fmt.Errorf("predictions[%d]: reason longer than 200 bytes", i)
		}
		seen[k] = true
		for name, v := range x.Values {
			if !metricName.MatchString(name) {
				return nil, fmt.Errorf("predictions[%d]: metric name %q", i, name)
			}
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("predictions[%d].values.%s: not finite", i, name)
			}
		}
	}
	if len(seen) != len(want) {
		for k := range want {
			if !seen[k] {
				return nil, fmt.Errorf("predictions: %d of %d handed units have no prediction (first missing: %s); an artifact that omits a unit is refused whole (D7)", len(want)-len(seen), len(want), k)
			}
		}
	}
	return &p, nil
}

// Aggregate is one reported figure over a lane of units. It names no unit.
//
// What this protects, and what it does not (D7, design §10.2). Protected: no unit is named, no
// figure covers fewer than MinAggregateUnits units, and the units a figure covers, and so Units,
// are fixed by the seed and the measured data alone, never by the candidate's predictions.
// Not protected: Value moves with the predictions. A candidate that resubmits with one unit's
// prediction corrupted can read the change in Value and learn whether that unit is in the
// reported role. The design counts and restricts only the held role (the gate job runs for a
// pre-registered change, and every further evaluation is counted); it promises no such limit
// for the select and error jobs. Today a run of the select job is neither counted nor limited
// to main-branch candidates.
type Aggregate struct {
	Lane   string  `json:"lane"`
	Metric string  `json:"metric"`
	Units  int     `json:"units"`
	Value  float64 `json:"value"`
	// NotBlind marks a figure over units on a legacy side. The old labels and the side rules
	// are public, so anyone can tell which of those units are held; such a figure is reported
	// apart from the blind one and never pooled with it (design §10.2).
	NotBlind bool `json:"not_blind"`
}

// Report is what a seed-holding job publishes: aggregates only.
type Report struct {
	Kind       string      `json:"kind"` // "AggregateReport"
	Job        string      `json:"job"`  // release, select, error or gate
	Candidate  string      `json:"candidate"`
	Aggregates []Aggregate `json:"aggregates"`
	// Evaluated is false when no unit of the role was predicted; such a report decides nothing.
	Evaluated bool `json:"evaluated"`
	// Evaluation counts this change's evaluations on held units, for the gate (design §10.2).
	Evaluation int `json:"evaluation,omitempty"`
}

// CheckReport refuses a seed-holding job's output that names a unit or reports an aggregate
// over fewer than MinAggregateUnits units (D7). raw is the output as it would be published.
// unitIDs holds every unit_id the job saw, so an identifier hidden in any field is caught.
func CheckReport(raw []byte, unitIDs []string) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var r Report
	if err := dec.Decode(&r); err != nil {
		return fmt.Errorf("report: %w", err)
	}
	var errs []error
	for i, a := range r.Aggregates {
		if a.Units < MinAggregateUnits {
			errs = append(errs, fmt.Errorf("aggregates[%d] (%s, %s): covers %d units, fewer than %d (D7)", i, a.Lane, a.Metric, a.Units, MinAggregateUnits))
		}
	}
	sort.Strings(unitIDs)
	for _, id := range unitIDs {
		if bytes.Contains(raw, []byte(id)) || (len(id) >= 16 && bytes.Contains(raw, []byte(id[:16]))) {
			errs = append(errs, fmt.Errorf("report names unit %s; a seed-holding job reports aggregates only (D7)", id))
			break
		}
	}
	return errors.Join(errs...)
}

func regexpMust(s string) *regexp.Regexp { return regexp.MustCompile(s) }

// Measured is the measured values a seed-holding job reads from the pinned source with its
// own code from main, by "<source>/<unit_id>" and metric, with each unit's lane.
type Measured struct {
	Values map[string]map[string]float64
	Lanes  map[string]string
}

// AggregatePredictions reduces predictions to aggregates over the units whose role is role:
// the mean absolute percentage error of each metric in each lane. Units on a legacy side
// (onLegacySide) and the others are reduced apart, into figures marked not blind and blind.
// What a figure covers does not depend on the candidate. A unit of the role is every handed
// unit the seed puts in it; a metric is every metric measured for that unit. A prediction that
// is absent (a reason, or a value left out) scores as a 100% error, so omitting a prediction
// can lower neither the figure nor the units it covers, and it cannot be used to learn a unit's
// role from a change in a count. The one exception is an artifact with no value at all, which
// decides nothing and reports nothing. Within each, a lane with fewer than MinAggregateUnits units is merged into the lane "other",
// and if "other" is still too small it is not reported (design §10.2). The units of other
// roles are dropped before anything is computed, so a candidate's predictions on them change
// nothing.
func AggregatePredictions(job, role string, p *Predictions, m *Measured, roleOf func(source, unitID string) (Role, error), onLegacySide func(source, unitID string) bool) (*Report, error) {
	type acc struct {
		sum float64
		n   int
	}
	type key struct {
		notBlind     bool
		lane, metric string
	}
	sums := map[key]*acc{}
	predicted := false
	for _, x := range p.Predictions {
		predicted = predicted || len(x.Values) > 0
	}
	if !predicted {
		return &Report{Kind: "AggregateReport", Job: job, Candidate: p.Candidate, Aggregates: []Aggregate{}}, nil
	}
	for _, x := range p.Predictions {
		k := x.Source + "/" + x.UnitID
		r, err := roleOf(x.Source, x.UnitID)
		if err != nil {
			return nil, err
		}
		if string(r) != role {
			continue
		}
		nb := onLegacySide != nil && onLegacySide(x.Source, x.UnitID)
		for metric, meas := range m.Values[k] {
			if meas == 0 {
				continue
			}
			pred := 0.0 // an absent prediction is a 100% error
			if v, ok := x.Values[metric]; ok {
				pred = v
			}
			kk := key{nb, m.Lanes[k], metric}
			a := sums[kk]
			if a == nil {
				a = &acc{}
				sums[kk] = a
			}
			a.sum += math.Abs(pred-meas) / math.Abs(meas) * 100
			a.n++
		}
	}
	keys := make([]key, 0, len(sums))
	for k := range sums {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := keys[i], keys[j]
		if a.notBlind != b.notBlind {
			return !a.notBlind
		}
		if a.lane != b.lane {
			return a.lane < b.lane
		}
		return a.metric < b.metric
	})
	other := map[key]*acc{}
	var otherKeys []key
	rep := &Report{Kind: "AggregateReport", Job: job, Candidate: p.Candidate, Aggregates: []Aggregate{}}
	for _, k := range keys {
		a := sums[k]
		if a.n < MinAggregateUnits {
			ok := key{k.notBlind, "other", k.metric}
			o := other[ok]
			if o == nil {
				o = &acc{}
				other[ok] = o
				otherKeys = append(otherKeys, ok)
			}
			o.sum += a.sum
			o.n += a.n
			continue
		}
		rep.Aggregates = append(rep.Aggregates, Aggregate{Lane: k.lane, Metric: k.metric + "_mape", Units: a.n, Value: a.sum / float64(a.n), NotBlind: k.notBlind})
	}
	for _, k := range otherKeys {
		if o := other[k]; o.n >= MinAggregateUnits {
			rep.Aggregates = append(rep.Aggregates, Aggregate{Lane: "other", Metric: k.metric + "_mape", Units: o.n, Value: o.sum / float64(o.n), NotBlind: k.notBlind})
		}
	}
	rep.Evaluated = len(rep.Aggregates) > 0
	return rep, nil
}

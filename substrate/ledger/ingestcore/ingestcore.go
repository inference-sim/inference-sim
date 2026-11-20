// Package ingestcore holds the part of ledger ingestion that does not depend on the source: it
// turns identified records into units by the rules of design §10.2, and accounts for every
// record in the ingest report. A source's package decides what a record is, what identifies it
// and what it measured; this package decides which records are duplicates, which are merged
// and which are left out.
package ingestcore

import (
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/substrate/ledger"
)

// Candidate is one source record, ready to be a measurement of a unit.
type Candidate struct {
	// Record is the source's row id.
	Record string
	// Identity is the unit's identifying fields; RoleKey is the part of it that near-copies share.
	Identity map[string]any
	RoleKey  map[string]any
	// Measured is the record's measured values, hashed as ledger.MeasuredID does.
	Measured map[string]any
	// Recorded is what the record states beyond its identity: it is kept with the measurement.
	Recorded map[string]any
	// Facet is the value the report counts units by.
	Facet string
}

// Report is the ingest report for one source revision (design §10.2). Its keys are those of the
// InferenceX report; UnitsBy counts units per value of the facet FacetName.
type Report struct {
	Kind     string `yaml:"kind"` // "IngestReport"
	Source   string `yaml:"source"`
	Revision string `yaml:"revision"`
	// Input is the SHA-256 of the exported records the report was generated from.
	Input         string              `yaml:"input"`
	Records       int                 `yaml:"records"`
	Units         int                 `yaml:"units"`
	RoleKeys      int                 `yaml:"role_keys"`
	Duplicates    int                 `yaml:"duplicates"`
	Merged        int                 `yaml:"merged"`
	JoinedUnits   int                 `yaml:"joined_units"`
	JoinedRecords int                 `yaml:"joined_records"`
	LeftOut       map[string][]string `yaml:"left_out"`
	FacetName     string              `yaml:"facet"`
	UnitsBy       map[string]int      `yaml:"units_by"`
}

// SameMeasurementTwoRoleKeys is the reason for a record whose measured values appear under two
// identities with different role keys: nothing says which unit it belongs to.
const SameMeasurementTwoRoleKeys = "same_measurement_two_role_keys"

// Ingest turns the candidates, in the source's record order, into units and the report.
// Candidates with the same identity and measured values are one measurement, kept under the
// first record and listing the others as duplicates. Candidates with the same identity and
// different measured values are one unit holding each measurement. The same measured values
// under two identities are one unit when the identities share a role key, under the first
// record's identity, and are left out when they do not. leftOut holds the records the source's
// package has already refused, by reason; records is the number the source held in all.
func Ingest(source, revision, input, facet string, records int, cands []Candidate, leftOut map[string][]string) ([]ledger.Unit, *Report, error) {
	rep := &Report{Kind: "IngestReport", Source: source, Revision: revision, Input: input,
		Records: records, LeftOut: map[string][]string{}, FacetName: facet, UnitsBy: map[string]int{}}
	for why, ids := range leftOut {
		rep.LeftOut[why] = append([]string(nil), ids...)
	}
	type rec struct {
		seq      int
		c        *Candidate
		unitID   string
		roleKey  string
		measured string
	}
	recs := make([]*rec, 0, len(cands))
	for i := range cands {
		c := &cands[i]
		uid, err := ledger.ID(c.Identity)
		if err != nil {
			return nil, nil, fmt.Errorf("record %s: %w", c.Record, err)
		}
		rkID, err := ledger.ID(c.RoleKey)
		if err != nil {
			return nil, nil, fmt.Errorf("record %s: %w", c.Record, err)
		}
		m, err := ledger.MeasuredID(c.Measured)
		if err != nil {
			return nil, nil, fmt.Errorf("record %s: %w", c.Record, err)
		}
		recs = append(recs, &rec{seq: i, c: c, unitID: uid, roleKey: rkID, measured: m})
	}

	byMeasured := map[string][]*rec{}
	var measuredOrder []string
	for _, x := range recs {
		if _, ok := byMeasured[x.measured]; !ok {
			measuredOrder = append(measuredOrder, x.measured)
		}
		byMeasured[x.measured] = append(byMeasured[x.measured], x)
	}
	target := map[*rec]*rec{}
	leave := map[*rec]bool{}
	for _, mk := range measuredOrder {
		group := byMeasured[mk]
		ids, keys := map[string]bool{}, map[string]bool{}
		for _, x := range group {
			ids[x.unitID] = true
			keys[x.roleKey] = true
		}
		if len(ids) < 2 {
			continue
		}
		if len(keys) > 1 {
			for _, x := range group {
				leave[x] = true
			}
			continue
		}
		for _, x := range group[1:] {
			if x.unitID != group[0].unitID {
				target[x] = group[0]
			}
		}
	}

	type meas struct {
		kept               *rec
		duplicates, merged []*rec
	}
	type unit struct {
		first *rec
		meas  []*meas
		byM   map[string]*meas
	}
	units := map[string]*unit{}
	var order []string
	for _, x := range recs {
		if leave[x] {
			rep.LeftOut[SameMeasurementTwoRoleKeys] = append(rep.LeftOut[SameMeasurementTwoRoleKeys], x.c.Record)
			continue
		}
		owner := x
		if t, ok := target[x]; ok {
			owner = t
		}
		u := units[owner.unitID]
		if u == nil {
			u = &unit{first: owner, byM: map[string]*meas{}}
			units[owner.unitID] = u
			order = append(order, owner.unitID)
		}
		m := u.byM[x.measured]
		switch {
		case m == nil:
			m = &meas{kept: x}
			u.byM[x.measured] = m
			u.meas = append(u.meas, m)
		case owner != x:
			m.merged = append(m.merged, x)
			rep.Merged++
		default:
			m.duplicates = append(m.duplicates, x)
			rep.Duplicates++
		}
	}

	out := make([]ledger.Unit, 0, len(units))
	roleKeys := map[string]bool{}
	for _, uid := range order {
		u := units[uid]
		lu := ledger.Unit{UnitID: uid, RoleKey: u.first.roleKey, Identity: u.first.c.Identity}
		n := 0
		sort.SliceStable(u.meas, func(i, j int) bool { return u.meas[i].kept.seq < u.meas[j].kept.seq })
		for _, m := range u.meas {
			rc := m.kept.c.Recorded
			if rc == nil {
				rc = map[string]any{}
			}
			lm := ledger.Measurement{Record: m.kept.c.Record, Measured: m.kept.measured,
				Duplicates: []string{}, Merged: []ledger.MergedRecord{}, Recorded: rc}
			for _, d := range m.duplicates {
				lm.Duplicates = append(lm.Duplicates, d.c.Record)
			}
			for _, d := range m.merged {
				lm.Merged = append(lm.Merged, ledger.MergedRecord{Record: d.c.Record, UnitID: d.unitID})
			}
			n += 1 + len(m.duplicates) + len(m.merged)
			lu.Measurements = append(lu.Measurements, lm)
		}
		if len(lu.Measurements) > 1 {
			rep.JoinedUnits++
			rep.JoinedRecords += n
		}
		roleKeys[lu.RoleKey] = true
		rep.UnitsBy[u.first.c.Facet]++
		out = append(out, lu)
	}
	rep.Units = len(out)
	rep.RoleKeys = len(roleKeys)
	sort.Slice(out, func(i, j int) bool { return out[i].UnitID < out[j].UnitID })
	left := 0
	for _, ids := range rep.LeftOut {
		left += len(ids)
	}
	kept := 0
	for _, u := range out {
		for _, m := range u.Measurements {
			kept += 1 + len(m.Duplicates) + len(m.Merged)
		}
	}
	if got := kept + left; got != rep.Records {
		return nil, nil, fmt.Errorf("ingest accounted for %d records of %d", got, rep.Records)
	}
	if err := ledger.CheckMeasurementsUnique(out); err != nil {
		return nil, nil, err
	}
	return out, rep, nil
}

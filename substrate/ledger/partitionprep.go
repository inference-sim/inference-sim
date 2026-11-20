package ledger

import (
	"errors"
	"fmt"
	"sort"
)

// PrepKind is the kind of a partition-prep report.
const PrepKind = "PartitionPrep"

// PrepReport is the D2 join report that prepares the next partition version (M3.2 row 26).
// It says which units of an anchor-only source were never read under a training role, from
// the consumers' unit-reads records alone: it needs no seed and assigns no role, so any
// maintainer can run it. It is a job-local file, never committed: the units it lists are the
// ones the next ledger.yaml leaves out.
type PrepReport struct {
	Kind   string `yaml:"kind"`
	Source string `yaml:"source"`
	// Consumers lists each record the report read, so a reader can see which readers it covers.
	Consumers []PrepConsumer `yaml:"consumers"`
	Units     int            `yaml:"units"`
	// Anchor counts the units that are anchors in the next version: never read under a training role.
	Anchor int `yaml:"anchor"`
	// Conflicts counts units read under two roles across the records (D2).
	Conflicts int `yaml:"conflicts"`
	// LeftOut lists, sorted, the units read under a training role: they get no role.
	LeftOut []string `yaml:"left_out"`
}

// PrepConsumer is one consumer record the report read.
type PrepConsumer struct {
	Consumer string `yaml:"consumer"`
	Revision string `yaml:"revision"`
	Reads    int    `yaml:"reads"`
}

// PrepareAnchorSource joins the consumers' records against one source's unit list and finds
// the units read under a training role (fit, select or validate). A source's unit that no
// record mentions was never read, so only a complete set of records can show that: require
// names the consumers whose record must be present, and a missing one is refused, because a
// consumer that is absent has not shown it read nothing.
func PrepareAnchorSource(source string, units []Unit, records []*Reads, require []string) (*PrepReport, error) {
	var errs []error
	if len(records) == 0 {
		return nil, errors.New("no consumer record: a join over nothing shows no unit was never read (X22)")
	}
	have := map[string]bool{}
	rep := &PrepReport{Kind: PrepKind, Source: source, Units: len(units), LeftOut: []string{}}
	known := map[string]bool{}
	for _, u := range units {
		known[u.UnitID] = true
	}
	trained := map[string]bool{}
	for _, rec := range records {
		have[rec.Consumer] = true
		n := 0
		for _, u := range rec.Reads {
			if u.Source != source {
				continue
			}
			n++
			if !known[u.UnitID] {
				errs = append(errs, fmt.Errorf("consumer %s reads %s unit %s, which the unit list does not hold", rec.Consumer, source, u.UnitID))
				continue
			}
			if contains(trainingRoles, u.Role) {
				trained[u.UnitID] = true
			}
		}
		rep.Consumers = append(rep.Consumers, PrepConsumer{Consumer: rec.Consumer, Revision: rec.Revision, Reads: n})
	}
	sort.Slice(rep.Consumers, func(i, j int) bool { return rep.Consumers[i].Consumer < rep.Consumers[j].Consumer })
	for _, name := range require {
		if !have[name] {
			errs = append(errs, fmt.Errorf("the record of consumer %q is missing; without it no unit is shown to have been never read", name))
		}
	}
	conflicts, err := Join(records, nil)
	if err != nil {
		return nil, err
	}
	for _, c := range conflicts {
		if c.Source == source {
			rep.Conflicts++
		}
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	for id := range trained {
		rep.LeftOut = append(rep.LeftOut, id)
	}
	sort.Strings(rep.LeftOut)
	rep.Anchor = len(units) - len(rep.LeftOut)
	return rep, nil
}

// ProposeNextPartition is the ledger.yaml a maintainer reviews for the next partition version,
// from the current one: the next number and the same seed record, InferenceX unchanged, every
// other source with a held cut-point moved to training roles, and the anchor-only sources set
// to anchor with the units the report leaves out. It is a proposal for a pull request; it
// writes no real version, and the version it describes is drawn by the same secret as this one.
func ProposeNextPartition(cur *Config, reports map[string]*PrepReport) (*Config, error) {
	next := &Config{Kind: cur.Kind, PartitionVersion: cur.PartitionVersion + 1, Seed: cur.Seed, Rules: map[string]SourceRule{}}
	for src, r := range cur.Rules {
		next.Rules[src] = SourceRule{CutPoints: append(CutPoints(nil), r.CutPoints...), Legacy: r.Legacy, LeftOut: r.LeftOut}
	}
	for _, src := range sortedKeys(next.Rules) {
		r := next.Rules[src]
		switch {
		case contains(HeldSources, src):
		case contains(AnchorOnlySources, src):
			rep := reports[src]
			if rep == nil {
				return nil, fmt.Errorf("%s: no partition-prep report; run the join report on its unit list first", src)
			}
			r = SourceRule{CutPoints: append(CutPoints(nil), anchorOnly...), LeftOut: rep.LeftOut}
		case hasRole(r.CutPoints, RoleHeld):
			r = SourceRule{CutPoints: TrainingCutPoints()}
		case len(r.Legacy) != 0:
			return nil, fmt.Errorf("%s: has a legacy side and is not a held source", src)
		}
		next.Rules[src] = r
	}
	for _, src := range AnchorOnlySources {
		if _, ok := next.Rules[src]; !ok {
			return nil, fmt.Errorf("%s: the ledger has no rule for it", src)
		}
	}
	if err := next.Validate(); err != nil {
		return nil, fmt.Errorf("the proposal is not a valid ledger.yaml: %w", err)
	}
	return next, nil
}

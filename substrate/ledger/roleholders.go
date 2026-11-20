package ledger

import (
	"fmt"
	"sort"
)

// RoleHoldersFrom is the first partition version in which each role has fixed holders
// (M3.2 row 26, requirements Table DC "Partition version"). Version 1 kept the roles of the
// earlier design and is still read as it was written.
const RoleHoldersFrom = 2

// HeldSources are the only sources that have a held cut-point from RoleHoldersFrom: InferenceX
// alone is held out, for the whole-run gate (D9). Every other source is training data or an
// anchor.
var HeldSources = []string{"inferencex"}

// AnchorSources are the only sources that have an anchor cut-point from RoleHoldersFrom.
var AnchorSources = []string{"inferencex", "configiq"}

// AnchorOnlySources are the AnchorSources whose every unit is an anchor, except the units a
// rule leaves out (SourceRule.LeftOut): a source read as training data before the roles were
// fixed, now used only as an anchor.
var AnchorOnlySources = []string{"configiq"}

// anchorOnly is the cut-points of an anchor-only source: the whole draw range is anchor.
var anchorOnly = CutPoints{{Role: RoleAnchor, From: 0, To: 99}}

// TrainingCutPoints is the cut-points of a source used only as training data: the
// proportions of aisimulate-fpm.
func TrainingCutPoints() CutPoints {
	return CutPoints{
		{Role: RoleFit, From: 0, To: 69},
		{Role: RoleSelect, From: 70, To: 84},
		{Role: RoleValidate, From: 85, To: 99},
	}
}

func (c *Config) checkRoleHolders() []error {
	var errs []error
	if c.PartitionVersion < RoleHoldersFrom {
		for _, src := range sortedKeys(c.Rules) {
			if len(c.Rules[src].LeftOut) != 0 {
				errs = append(errs, fmt.Errorf("rules.%s.left_out: units are left out from partition version %d", src, RoleHoldersFrom))
			}
		}
		return errs
	}
	for _, src := range sortedKeys(c.Rules) {
		r := c.Rules[src]
		for i, p := range r.CutPoints {
			if p.Role == RoleHeld && !contains(HeldSources, src) {
				errs = append(errs, fmt.Errorf("rules.%s.cut_points[%d].role: held; from partition version %d only %v has a held cut-point", src, i, RoleHoldersFrom, HeldSources))
			}
			if p.Role == RoleAnchor && !contains(AnchorSources, src) {
				errs = append(errs, fmt.Errorf("rules.%s.cut_points[%d].role: anchor; from partition version %d only %v have an anchor cut-point", src, i, RoleHoldersFrom, AnchorSources))
			}
		}
		if contains(AnchorOnlySources, src) {
			if fmt.Sprint(r.CutPoints) != fmt.Sprint(anchorOnly) || len(r.Legacy) != 0 {
				errs = append(errs, fmt.Errorf("rules.%s: an anchor-only source has the one cut-point {anchor 0..99} and no legacy side", src))
			}
		}
		if len(r.LeftOut) != 0 && !contains(AnchorOnlySources, src) {
			errs = append(errs, fmt.Errorf("rules.%s.left_out: only %v leave units out", src, AnchorOnlySources))
		}
		for i, id := range r.LeftOut {
			switch {
			case !hex64.MatchString(id):
				errs = append(errs, fmt.Errorf("rules.%s.left_out[%d]: %q is not a unit_id (64 lowercase hex digits)", src, i, id))
			case i > 0 && id <= r.LeftOut[i-1]:
				errs = append(errs, fmt.Errorf("rules.%s.left_out[%d]: %s is not above the unit before it; list the units sorted, each once", src, i, id))
			}
		}
	}
	for _, src := range HeldSources {
		if r, ok := c.Rules[src]; ok && !hasRole(r.CutPoints, RoleHeld) {
			errs = append(errs, fmt.Errorf("rules.%s.cut_points: %s has no held cut-point; it is the one held source", src, src))
		}
	}
	return errs
}

func hasRole(c CutPoints, r Role) bool {
	for _, p := range c {
		if p.Role == r {
			return true
		}
	}
	return false
}

// leftOut reports whether unitID is one of the units the rule leaves out.
func (r SourceRule) leftOut(unitID string) bool {
	i := sort.SearchStrings(r.LeftOut, unitID)
	return i < len(r.LeftOut) && r.LeftOut[i] == unitID
}

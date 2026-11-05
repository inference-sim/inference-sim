package ledger

import (
	"errors"
	"fmt"
	"sort"
)

// Assigner computes roles. It exists only where the seed does: in the release, select, error
// and gate jobs (D7). Its results are never written to a committed file, except the published
// roles the release job writes.
type Assigner struct {
	seed     Seed
	rules    map[string]SourceRule
	roleKeys map[string]RoleKeyFunc
	legacy   *LegacyMap
	// sides gives each role key that holds legacy units the side its new units take.
	sides map[string]Side
}

// RoleKeyFunc returns the role key of a unit's identity, for one source.
type RoleKeyFunc func(identity map[string]any) (map[string]any, error)

// NewAssigner checks the seed against the ledger's recorded hash and returns an Assigner.
// roleKeys gives each source's role-key rule, so the HMAC's message is the canonical JSON of
// the role key itself (design §10.2).
func NewAssigner(cfg *Config, legacy *LegacyMap, roleKeys map[string]RoleKeyFunc, hexSeed string) (*Assigner, error) {
	if !cfg.Seed.Created() {
		return nil, errors.New("ledger.yaml records no seed; a ledger owner creates it (catalog/ledger/README.md)")
	}
	seed, err := ParseSeed(hexSeed, cfg.Seed.SHA256)
	if err != nil {
		return nil, err
	}
	a := &Assigner{seed: seed, rules: cfg.Rules, roleKeys: roleKeys, legacy: legacy, sides: map[string]Side{}}
	if legacy != nil {
		for rk, side := range legacy.RoleKeySides {
			a.sides[rk] = side
		}
	}
	return a, nil
}

// LegacySide returns the legacy side a unit is on, if any: its own side if it is in the legacy
// table, else the side of its role key if that holds legacy units. A unit on a legacy side is
// not blind, because the old labels and the side rules are public (design §10.2).
func (a *Assigner) LegacySide(source string, u *Unit) (Side, bool) {
	if a.legacy == nil || a.legacy.Source != source {
		return "", false
	}
	if s, ok := a.legacy.Units[u.UnitID]; ok {
		return s, true
	}
	s, ok := a.sides[u.RoleKey]
	return s, ok
}

// Role returns a unit's role. A unit on a legacy side keeps that side, and the draw chooses
// within it with the side's cut-point. Any other unit is drawn with the source's cut-points.
func (a *Assigner) Role(source string, u *Unit) (Role, error) {
	rule, ok := a.rules[source]
	if !ok {
		return "", fmt.Errorf("source %q has no cut-points in ledger.yaml", source)
	}
	rkf, ok := a.roleKeys[source]
	if !ok {
		return "", fmt.Errorf("source %q has no role-key rule in this build", source)
	}
	rk, err := rkf(u.Identity)
	if err != nil {
		return "", fmt.Errorf("unit %s: %w", u.UnitID, err)
	}
	msg, err := Canonical(rk)
	if err != nil {
		return "", fmt.Errorf("unit %s: role key: %w", u.UnitID, err)
	}
	if id := hashHex(msg); id != u.RoleKey {
		return "", fmt.Errorf("unit %s: role_key %s is not the hash of its role key (%s)", u.UnitID, u.RoleKey, id)
	}
	d := a.seed.Draw(msg)
	side, legacy := a.LegacySide(source, u)
	if !legacy {
		return rule.CutPoints.Role(d)
	}
	cp, ok := rule.Legacy[side]
	if !ok {
		return "", fmt.Errorf("source %q has no cut-points for legacy side %s", source, side)
	}
	return cp.Role(d)
}

// Published is one of the three lists the release job writes: fit.yaml, anchor.yaml or
// report.yaml under catalog/ledger/published/.
type Published struct {
	Kind             string   `yaml:"kind"` // "PublishedUnits"
	Role             Role     `yaml:"role"`
	PartitionVersion int      `yaml:"partition_version"`
	SeedSHA256       string   `yaml:"seed_sha256"`
	Units            []string `yaml:"units"`
}

// Release computes the three published lists from every unit of every source. It returns
// only fit, anchor and report: the members of the other roles never leave the job.
func (a *Assigner) Release(cfg *Config, units map[string][]Unit) (map[Role]*Published, error) {
	out := map[Role]*Published{}
	for _, r := range []Role{RoleFit, RoleAnchor, RoleReport} {
		out[r] = &Published{Kind: "PublishedUnits", Role: r, PartitionVersion: cfg.PartitionVersion,
			SeedSHA256: cfg.Seed.SHA256, Units: []string{}}
	}
	for _, src := range sortedKeys(units) {
		for i := range units[src] {
			r, err := a.Role(src, &units[src][i])
			if err != nil {
				return nil, err
			}
			if p, ok := out[r]; ok {
				p.Units = append(p.Units, units[src][i].UnitID)
			}
		}
	}
	for _, p := range out {
		sort.Strings(p.Units)
	}
	return out, nil
}

// CheckCitesOnlyFit is D1's release check: a fitted coefficient cites only units whose role
// is fit. cites maps each coefficient to the unit_ids it cites; fit is the release's fit list.
// A unit not on the fit list is select, validate, held, anchor or report, and none may have
// fitted a coefficient; a legacy coefficient citing a unit now in select is refitted from the
// fit list before the release (design §10.2). The registry has no field that cites unit_ids
// yet (D1's target is M2), so the release job has nothing to pass it until that field lands.
func CheckCitesOnlyFit(cites map[string][]string, fit *Published) error {
	if fit == nil || fit.Role != RoleFit {
		return errors.New("the check needs the release's fit list")
	}
	on := make(map[string]bool, len(fit.Units))
	for _, id := range fit.Units {
		on[id] = true
	}
	var errs []error
	for _, name := range sortedKeys(cites) {
		for _, id := range cites[name] {
			if !on[id] {
				errs = append(errs, fmt.Errorf("coefficient %s cites unit %s, which is not on the fit list; refit it from fit units (D1)", name, id))
				break
			}
		}
	}
	return errors.Join(errs...)
}

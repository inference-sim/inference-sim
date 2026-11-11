package ledger

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
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

// PublishedKind is the kind of a published list.
const PublishedKind = "PublishedUnits"

// Check is what a published list can know about itself: its kind, that its role is one the
// release publishes and the one its file name states, the partition and seed it was drawn
// under, and that its units are unit identifiers, sorted and listed once. file is the role
// the file name states.
func (p *Published) Check(file Role, cfg *Config) error {
	var errs []error
	if p.Kind != PublishedKind {
		errs = append(errs, fmt.Errorf("kind %q, want %q", p.Kind, PublishedKind))
	}
	if !p.Role.Published() {
		errs = append(errs, fmt.Errorf("role %q is not published; only fit, anchor and report are", p.Role))
	} else if p.Role != file {
		errs = append(errs, fmt.Errorf("role %q in the file for %q", p.Role, file))
	}
	if !cfg.Seed.Created() {
		errs = append(errs, errors.New("a list is published, but ledger.yaml records no seed; the lists are drawn from the seed"))
	} else if p.SeedSHA256 != cfg.Seed.SHA256 {
		errs = append(errs, fmt.Errorf("seed_sha256 %q is not the seed ledger.yaml records, %q", p.SeedSHA256, cfg.Seed.SHA256))
	}
	if p.PartitionVersion != cfg.PartitionVersion {
		errs = append(errs, fmt.Errorf("partition_version %d, but ledger.yaml is at %d", p.PartitionVersion, cfg.PartitionVersion))
	}
	if len(p.Units) == 0 && p.Role.mustBeNonempty() {
		errs = append(errs, fmt.Errorf("the %s list holds no unit; a list the dashboard or the fits read is never empty (COV-2, D1)", p.Role))
	}
	for i, id := range p.Units {
		switch {
		case !hex64.MatchString(id):
			errs = append(errs, fmt.Errorf("units[%d] %q is not a unit_id (64 lowercase hex digits)", i, id))
		case i > 0 && id == p.Units[i-1]:
			errs = append(errs, fmt.Errorf("units[%d] %s is listed twice", i, id))
		case i > 0 && id < p.Units[i-1]:
			errs = append(errs, fmt.Errorf("units[%d] %s is out of order; the list is sorted", i, id))
		}
	}
	return errors.Join(errs...)
}

// CheckReleasable is the release gate's check of the three lists (D7, COV-2): all three are
// present, and the fit list and the anchor list each hold a unit. The anchor list is what the
// dashboard scores (COV-2); the fit list is what every fitted coefficient cites (D1). The
// report list may be empty, because the partition's cut-points give the report role no range
// until a source draws one. A tree that has not yet published its lists is fine to check and
// to merge; it is not fine to release.
func CheckReleasable(lists map[Role]*Published) error {
	var errs []error
	for _, r := range []Role{RoleFit, RoleAnchor, RoleReport} {
		p := lists[r]
		switch {
		case p == nil:
			errs = append(errs, fmt.Errorf("the %s list is not published; a release carries fit, anchor and report", r))
		case len(p.Units) == 0 && r.mustBeNonempty():
			errs = append(errs, fmt.Errorf("the %s list holds no unit; a release publishes a nonempty %s list", r, r))
		}
	}
	return errors.Join(errs...)
}

// CheckRelease loads the ledger at dir the way a release reads it, without the role-key rules
// of the ingest tools (ledger check runs those), and applies CheckReleasable to the published
// lists. A release whose tree holds no lists, or an empty fit or anchor list, is refused.
func CheckRelease(dir string) error {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return err
	}
	srcs, err := LoadSources(dir)
	if err != nil {
		return err
	}
	units := map[string][]Unit{}
	for _, src := range srcs.Sources {
		if src.Units == "" {
			continue
		}
		us, _, err := ReadUnitList(filepath.Join(dir, filepath.FromSlash(src.Units)))
		if err != nil {
			return err
		}
		units[src.ID] = us
	}
	lists, err := LoadPublished(dir, cfg, units)
	if err != nil {
		return err
	}
	return CheckReleasable(lists)
}

// LoadPublished reads the lists the release job wrote under dir/published/ and checks them
// against the ledger: each list's own checks, every unit on it is a unit of some source, and
// no unit is on two lists (one role per unit, D4). The lists are optional until the first
// release publishes them; a missing list is absent from the result, but a directory holding
// some lists and not all of the three is refused, because the dashboard and the candidate
// jobs read them as a set.
func LoadPublished(dir string, cfg *Config, units map[string][]Unit) (map[Role]*Published, error) {
	out := map[Role]*Published{}
	var errs []error
	for _, r := range []Role{RoleFit, RoleAnchor, RoleReport} {
		path := filepath.Join(dir, DirPublished, string(r)+".yaml")
		raw, err := os.ReadFile(path)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		var p Published
		if err := decodeStrict(path, raw, &p); err != nil {
			errs = append(errs, err)
			continue
		}
		if err := p.Check(r, cfg); err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", path, err))
		}
		out[r] = &p
	}
	if len(out) == 0 {
		return out, errors.Join(errs...)
	}
	if len(out) != 3 {
		errs = append(errs, fmt.Errorf("%s holds %d of the three published lists; the release job writes fit, anchor and report together", DirPublished, len(out)))
	}
	known := map[string]bool{}
	for _, us := range units {
		for _, u := range us {
			known[u.UnitID] = true
		}
	}
	on := map[string]Role{}
	for _, r := range []Role{RoleFit, RoleAnchor, RoleReport} {
		p := out[r]
		if p == nil {
			continue
		}
		for _, id := range p.Units {
			if !known[id] {
				errs = append(errs, fmt.Errorf("%s list: unit %s is in no source's unit list", r, id))
				break
			}
		}
		for _, id := range p.Units {
			if other, dup := on[id]; dup {
				errs = append(errs, fmt.Errorf("unit %s is on the %s list and the %s list; a unit has one role", id, other, r))
				break
			}
			on[id] = r
		}
	}
	return out, errors.Join(errs...)
}

// Release computes the three published lists from every unit of every source. It returns
// only fit, anchor and report: the members of the other roles never leave the job.
func (a *Assigner) Release(cfg *Config, units map[string][]Unit) (map[Role]*Published, error) {
	out := map[Role]*Published{}
	for _, r := range []Role{RoleFit, RoleAnchor, RoleReport} {
		out[r] = &Published{Kind: PublishedKind, Role: r, PartitionVersion: cfg.PartitionVersion,
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
	if err := CheckReleasable(out); err != nil {
		return nil, fmt.Errorf("the release job would publish lists a release refuses: %w", err)
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

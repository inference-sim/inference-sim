package ledger

import (
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
)

// LegacyMap is legacy-map.yaml: the partition made before the ledger existed, carried over as
// each legacy unit's side (design §10.2, plan §3.3). It holds labels, not roles: FIT or HELD.
type LegacyMap struct {
	Kind   string `yaml:"kind"` // "LegacyMap"
	Source string `yaml:"source"`
	// Partition pins the old partition file.
	Partition LegacyPartition `yaml:"partition"`
	// Points counts the old partition's points and what became of them.
	Points LegacyCounts `yaml:"points"`
	// Units gives each legacy unit's side. In the file it is a sorted list of
	// "<unit_id> <side>" lines (UnitSides), because a YAML mapping of tens of thousands of
	// keys decodes in quadratic time.
	Units     map[string]Side `yaml:"-"`
	UnitSides []string        `yaml:"units"`
	// BothLabels lists the units whose legacy records carry both labels. Each takes FIT,
	// because a coefficient may already cite its FIT records (D1).
	BothLabels []string `yaml:"both_labels"`
	// RoleKeySides gives, for each role key that holds legacy units, the side a new unit of
	// that role key takes: the side its legacy units hold, FIT if they hold both. It is
	// computed from Units and the unit list (SetRoleKeySides), not stored.
	RoleKeySides map[string]Side `yaml:"-"`
	// NewUnitsTakingSide lists the units outside Units whose role key holds legacy units.
	NewUnitsTakingSide []string `yaml:"new_units_taking_side"`
	// PointsFile names the gzipped JSON lines that map each legacy point to its units.
	PointsFile string `yaml:"points_file"`
}

// LegacyPartition pins the old partition file by commit and content hash.
type LegacyPartition struct {
	URL    string `yaml:"url"`
	SHA256 string `yaml:"sha256"`
	Seed   int64  `yaml:"seed"`
}

// LegacyCounts says how the old partition's points map to units.
type LegacyCounts struct {
	Total int `yaml:"total"`
	// Mapped counts points with at least one unit.
	Mapped int `yaml:"mapped"`
	// ToSeveralUnits counts points whose records fall in more than one unit.
	ToSeveralUnits int `yaml:"to_several_units"`
	// NotInSource counts points with no record in the pinned source revision; they were drawn
	// from data the revision does not hold.
	NotInSource int `yaml:"not_in_source"`
	// AllLeftOut counts points whose every record the ingest left out.
	AllLeftOut int `yaml:"all_left_out"`
}

// LegacyPoint is one line of the points file.
type LegacyPoint struct {
	Point string   `json:"point"`
	Side  Side     `json:"side"`
	Units []string `json:"units"`
}

// Validate checks the map on its own, and fills Units from UnitSides.
func (m *LegacyMap) Validate() error {
	var errs []error
	m.Units = make(map[string]Side, len(m.UnitSides))
	for i, line := range m.UnitSides {
		id, side, ok := strings.Cut(line, " ")
		if !ok {
			errs = append(errs, fmt.Errorf("units[%d]: %q is not \"<unit_id> <side>\"", i, line))
			continue
		}
		if i > 0 && m.UnitSides[i-1] >= line {
			errs = append(errs, fmt.Errorf("units[%d]: out of order or repeated", i))
		}
		m.Units[id] = Side(side)
	}
	if m.Kind != "LegacyMap" {
		errs = append(errs, fmt.Errorf("kind: %q, want \"LegacyMap\"", m.Kind))
	}
	if !hex64.MatchString(m.Partition.SHA256) || !pinnedURL.MatchString(m.Partition.URL) {
		errs = append(errs, errors.New("partition: pin the old partition by a URL at a commit and its sha256 (D3)"))
	}
	for id, s := range m.Units {
		if !hex64.MatchString(id) || !s.Valid() {
			errs = append(errs, fmt.Errorf("units.%s: %q; a legacy unit maps a unit_id to FIT or HELD", id, s))
		}
	}
	for _, id := range m.BothLabels {
		if m.Units[id] != SideFit {
			errs = append(errs, fmt.Errorf("both_labels: unit %s carries both labels and so takes FIT, but units gives %q", id, m.Units[id]))
		}
	}
	c := m.Points
	if c.Mapped+c.NotInSource+c.AllLeftOut != c.Total {
		errs = append(errs, fmt.Errorf("points: mapped %d + not_in_source %d + all_left_out %d != total %d", c.Mapped, c.NotInSource, c.AllLeftOut, c.Total))
	}
	return errors.Join(errs...)
}

// BuildLegacy maps legacy points to units. pointUnits gives, for each legacy point present in
// the source revision, the units its records fell in (empty when every record was left out).
// labels gives each legacy point's side. units is the source's unit list.
func BuildLegacy(source string, part LegacyPartition, labels map[string]Side, pointUnits map[string][]string, units []Unit, pointsFile string) (*LegacyMap, []LegacyPoint) {
	m := &LegacyMap{Kind: "LegacyMap", Source: source, Partition: part, Units: map[string]Side{},
		BothLabels: []string{}, RoleKeySides: map[string]Side{}, NewUnitsTakingSide: []string{}, PointsFile: pointsFile}
	m.Points.Total = len(labels)
	sides := map[string]map[Side]bool{}
	var points []LegacyPoint
	for _, p := range sortedKeys(labels) {
		us, inSource := pointUnits[p]
		switch {
		case !inSource:
			m.Points.NotInSource++
		case len(us) == 0:
			m.Points.AllLeftOut++
		default:
			m.Points.Mapped++
			if len(us) > 1 {
				m.Points.ToSeveralUnits++
			}
		}
		sorted := append([]string{}, us...)
		sort.Strings(sorted)
		points = append(points, LegacyPoint{Point: p, Side: labels[p], Units: sorted})
		for _, u := range sorted {
			if sides[u] == nil {
				sides[u] = map[Side]bool{}
			}
			sides[u][labels[p]] = true
		}
	}
	for u, s := range sides {
		if s[SideFit] {
			m.Units[u] = SideFit
			if s[SideHeld] {
				m.BothLabels = append(m.BothLabels, u)
			}
		} else {
			m.Units[u] = SideHeld
		}
	}
	sort.Strings(m.BothLabels)
	for u, s := range m.Units {
		m.UnitSides = append(m.UnitSides, u+" "+string(s))
	}
	sort.Strings(m.UnitSides)
	m.NewUnitsTakingSide = m.SetRoleKeySides(units)
	return m, points
}

// SetRoleKeySides computes RoleKeySides from Units and the source's unit list, and returns
// the units outside the table whose role key holds legacy units.
func (m *LegacyMap) SetRoleKeySides(units []Unit) []string {
	m.RoleKeySides = map[string]Side{}
	for _, u := range units {
		if s, ok := m.Units[u.UnitID]; ok {
			if prev, seen := m.RoleKeySides[u.RoleKey]; !seen || (prev == SideHeld && s == SideFit) {
				m.RoleKeySides[u.RoleKey] = s
			}
		}
	}
	newUnits := []string{}
	for _, u := range units {
		if _, ok := m.Units[u.UnitID]; !ok {
			if _, held := m.RoleKeySides[u.RoleKey]; held {
				newUnits = append(newUnits, u.UnitID)
			}
		}
	}
	sort.Strings(newUnits)
	return newUnits
}

// LoadLegacyMap reads and validates legacy-map.yaml in dir. It returns nil, nil when the file
// does not exist.
func LoadLegacyMap(dir string) (*LegacyMap, error) {
	path := filepath.Join(dir, FileLegacyMap)
	if !exists(path) {
		return nil, nil
	}
	var m LegacyMap
	if err := decodeStrictFile(path, &m); err != nil {
		return nil, err
	}
	if err := m.Validate(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &m, nil
}

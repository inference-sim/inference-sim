package ledger

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
)

// Unit is one line of a committed unit list: the unit's identifier, its role key, its
// identifying fields, and each measurement it holds. It has no role field, and a line that
// carries one is refused when the list is read (D4).
type Unit struct {
	UnitID       string         `json:"unit_id"`
	RoleKey      string         `json:"role_key"`
	Identity     map[string]any `json:"identity"`
	Measurements []Measurement  `json:"measurements"`
}

// Measurement is one measurement of a unit: the source record kept for it, the hash of its
// measured values, the records that repeat it, and what the kept record states beyond the
// identifying fields.
type Measurement struct {
	// Record is the source's row id of the record kept: the smallest among the records that
	// hold this measurement under this identity.
	Record string `json:"record"`
	// Measured is the SHA-256 of the canonical JSON of the record's measured values.
	Measured string `json:"measured"`
	// Duplicates are the other records with the same identity and measured values, by row id.
	Duplicates []string `json:"duplicates"`
	// Merged are records with the same measured values under another identity with the same
	// role key, which design §10.2 folds into this unit, each with the unit_id its own
	// identity would have had.
	Merged []MergedRecord `json:"merged"`
	// Recorded holds fields the source records that are not identifying, such as the image
	// digest and the run attempt.
	Recorded map[string]any `json:"recorded"`
}

// MergedRecord is a record folded into a unit under another identity.
type MergedRecord struct {
	Record string `json:"record"`
	UnitID string `json:"unit_id"`
}

// Check verifies a unit on its own: unit_id is the identifier of its identity, role_key is the
// identifier of roleKey(identity), and every measurement is well formed.
func (u *Unit) Check(roleKey func(map[string]any) (map[string]any, error)) error {
	id, err := ID(u.Identity)
	if err != nil {
		return fmt.Errorf("unit %s: identity: %w", u.UnitID, err)
	}
	if id != u.UnitID {
		return fmt.Errorf("unit %s: its identity hashes to %s", u.UnitID, id)
	}
	rk, err := roleKey(u.Identity)
	if err != nil {
		return fmt.Errorf("unit %s: %w", u.UnitID, err)
	}
	rkID, err := ID(rk)
	if err != nil {
		return fmt.Errorf("unit %s: role key: %w", u.UnitID, err)
	}
	if rkID != u.RoleKey {
		return fmt.Errorf("unit %s: role_key is %s, but its identity gives %s", u.UnitID, u.RoleKey, rkID)
	}
	if len(u.Measurements) == 0 {
		return fmt.Errorf("unit %s: holds no measurement", u.UnitID)
	}
	seen := map[string]bool{}
	for i, m := range u.Measurements {
		if !hex64.MatchString(m.Measured) {
			return fmt.Errorf("unit %s: measurements[%d].measured: not a SHA-256", u.UnitID, i)
		}
		if seen[m.Measured] {
			return fmt.Errorf("unit %s: measurements[%d]: the same measured values appear twice; one is a duplicate of the other", u.UnitID, i)
		}
		seen[m.Measured] = true
		if m.Record == "" {
			return fmt.Errorf("unit %s: measurements[%d].record: required", u.UnitID, i)
		}
	}
	return nil
}

// EncodeUnits writes units as canonical JSON lines, sorted by unit_id, and gzips them with a
// fixed header (no name, no time), so the same units give the same bytes.
func EncodeUnits(w io.Writer, units []Unit) error {
	sorted := append([]Unit(nil), units...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].UnitID < sorted[j].UnitID })
	zw, err := gzip.NewWriterLevel(w, gzip.BestCompression)
	if err != nil {
		return err
	}
	for i := range sorted {
		line, err := CanonicalStruct(sorted[i])
		if err != nil {
			return fmt.Errorf("unit %s: %w", sorted[i].UnitID, err)
		}
		if _, err := zw.Write(append(line, '\n')); err != nil {
			return err
		}
	}
	return zw.Close()
}

// UnitListBytes returns the uncompressed canonical lines EncodeUnits writes. The byte-for-byte
// check of D4 compares these, so a change of compressor cannot fail it and a change of content
// cannot pass it.
func UnitListBytes(units []Unit) ([]byte, error) {
	var gz bytes.Buffer
	if err := EncodeUnits(&gz, units); err != nil {
		return nil, err
	}
	return gunzip(gz.Bytes())
}

// Gzip writes b to w, gzipped with a fixed header (no name, no time), so the same bytes give
// the same file.
func Gzip(w io.Writer, b []byte) error {
	zw, err := gzip.NewWriterLevel(w, gzip.BestCompression)
	if err != nil {
		return err
	}
	if _, err := zw.Write(b); err != nil {
		return err
	}
	return zw.Close()
}

// Gunzip returns the content of a gzipped file.
func Gunzip(b []byte) ([]byte, error) { return gunzip(b) }

func gunzip(b []byte) ([]byte, error) {
	zr, err := gzip.NewReader(bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	out, err := io.ReadAll(zr)
	if cerr := zr.Close(); err == nil {
		err = cerr
	}
	return out, err
}

// ReadUnitList reads a committed unit list (gzipped canonical JSON lines). It refuses a line
// with an unknown key, such as a role (SUB-12, D4), a line that is not in canonical form, and
// lines out of unit_id order or repeated. It returns the units and the uncompressed bytes.
func ReadUnitList(path string) ([]Unit, []byte, error) {
	gz, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	raw, err := gunzip(gz)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	var units []Unit
	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	n := 0
	for sc.Scan() {
		n++
		line := sc.Bytes()
		var u Unit
		dec := json.NewDecoder(bytes.NewReader(line))
		dec.DisallowUnknownFields()
		dec.UseNumber()
		if err := dec.Decode(&u); err != nil {
			return nil, nil, fmt.Errorf("%s:%d: %w", path, n, err)
		}
		canon, err := CanonicalStruct(u)
		if err != nil {
			return nil, nil, fmt.Errorf("%s:%d: %w", path, n, err)
		}
		if !bytes.Equal(canon, line) {
			return nil, nil, fmt.Errorf("%s:%d: not in canonical form", path, n)
		}
		if len(units) > 0 && units[len(units)-1].UnitID >= u.UnitID {
			return nil, nil, fmt.Errorf("%s:%d: unit %s is out of order or repeated", path, n, u.UnitID)
		}
		units = append(units, u)
	}
	if err := sc.Err(); err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	return units, raw, nil
}

// CanonicalStruct encodes a value through encoding/json and then canonically, so struct tags
// name the keys and every number keeps its text.
func CanonicalStruct(v any) ([]byte, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var g any
	if err := dec.Decode(&g); err != nil {
		return nil, err
	}
	return Canonical(g)
}

// CheckMeasurementsUnique enforces that each measurement belongs to exactly one unit: no
// measured-value hash appears in two units of one list (design §10.2).
func CheckMeasurementsUnique(units []Unit) error {
	owner := map[string]string{}
	var errs []error
	for _, u := range units {
		for _, m := range u.Measurements {
			if prev, ok := owner[m.Measured]; ok {
				errs = append(errs, fmt.Errorf("measured values %s belong to units %s and %s; a measurement belongs to one unit", m.Measured, prev, u.UnitID))
				continue
			}
			owner[m.Measured] = u.UnitID
		}
	}
	return errors.Join(errs...)
}

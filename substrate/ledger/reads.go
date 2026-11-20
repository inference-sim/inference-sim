package ledger

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
)

// Reads is one consumer's record of the units it read, and under which role (D2). Every
// consumer writes one: each fitting script, the select, error and gate jobs, and the
// dashboard's scoring runs. A record names roles, so it is never committed: each consumer
// uploads it as a CI artifact, and only the nightly join, inside the gate job, reads them all.
type Reads struct {
	Kind     string `yaml:"kind" json:"kind"` // "UnitReads"
	Consumer string `yaml:"consumer" json:"consumer"`
	// Revision is the consumer's commit or release that read the units.
	Revision string     `yaml:"revision" json:"revision"`
	Reads    []UnitRead `yaml:"reads" json:"reads"`
}

// UnitRead is one unit read under one role.
type UnitRead struct {
	Source string `yaml:"source" json:"source"`
	UnitID string `yaml:"unit_id" json:"unit_id"`
	Role   Role   `yaml:"role" json:"role"`
}

// Validate checks a record on its own.
func (r *Reads) Validate() error {
	var errs []error
	if r.Kind != "UnitReads" {
		errs = append(errs, fmt.Errorf("kind: %q, want \"UnitReads\"", r.Kind))
	}
	if r.Consumer == "" {
		errs = append(errs, errors.New("consumer: required"))
	}
	if err := CheckPinned("revision", r.Revision); err != nil {
		errs = append(errs, err)
	}
	for i, u := range r.Reads {
		if !hex64.MatchString(u.UnitID) || u.Source == "" || !u.Role.Valid() {
			errs = append(errs, fmt.Errorf("reads[%d]: needs a source, a 64-digit unit_id and one of %v", i, Roles))
		}
	}
	return errors.Join(errs...)
}

// DecodeReads reads one consumer's record from JSON, strictly: unknown fields are refused, and
// so is trailing data (SUB-12). A record comes from a job's artifact and is untrusted input.
func DecodeReads(r io.Reader) (*Reads, error) {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var out Reads
	if err := dec.Decode(&out); err != nil {
		return nil, err
	}
	if dec.More() {
		return nil, errors.New("trailing data after the record")
	}
	if err := out.Validate(); err != nil {
		return nil, err
	}
	return &out, nil
}

// Conflict is a unit read under two roles.
type Conflict struct {
	Source string
	UnitID string
	// Roles lists each role with the consumers that read the unit under it.
	Roles map[Role][]string
}

func (c Conflict) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "%s unit %s read as", c.Source, c.UnitID)
	for _, r := range Roles {
		if cs, ok := c.Roles[r]; ok {
			fmt.Fprintf(&b, " %s by %v;", r, cs)
		}
	}
	return b.String()
}

// Join is D2's nightly join: it fails if any unit was read under two roles across all
// consumers. It also fails if any record reads a unit under a role other than the one the
// seed assigns, when assign is given (inside the gate job), so a consumer that read a held
// unit as fit is caught even if no other consumer read it.
func Join(records []*Reads, assign func(source, unitID string) (Role, error)) ([]Conflict, error) {
	by := map[[2]string]map[Role]map[string]bool{}
	for _, rec := range records {
		for _, u := range rec.Reads {
			k := [2]string{u.Source, u.UnitID}
			if by[k] == nil {
				by[k] = map[Role]map[string]bool{}
			}
			if by[k][u.Role] == nil {
				by[k][u.Role] = map[string]bool{}
			}
			by[k][u.Role][rec.Consumer] = true
		}
	}
	keys := make([][2]string, 0, len(by))
	for k := range by {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i][0]+keys[i][1] < keys[j][0]+keys[j][1] })
	var out []Conflict
	for _, k := range keys {
		roles := by[k]
		c := Conflict{Source: k[0], UnitID: k[1], Roles: map[Role][]string{}}
		for r, cs := range roles {
			for name := range cs {
				c.Roles[r] = append(c.Roles[r], name)
			}
			sort.Strings(c.Roles[r])
		}
		bad := len(roles) > 1
		if !bad && assign != nil {
			want, err := assign(k[0], k[1])
			if err != nil {
				return nil, err
			}
			if _, ok := roles[want]; !ok {
				c.Roles[want] = append(c.Roles[want], "(assigned)")
				bad = true
			}
		}
		if bad {
			out = append(out, c)
		}
	}
	return out, nil
}

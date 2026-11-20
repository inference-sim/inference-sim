package ledger

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/big"
)

// Role is what a unit may be used for (design §10.2).
type Role string

const (
	RoleFit      Role = "fit"
	RoleSelect   Role = "select"
	RoleValidate Role = "validate"
	RoleHeld     Role = "held"
	RoleAnchor   Role = "anchor"
	RoleReport   Role = "report"
)

// Roles lists every role, in the order the design's table gives them.
var Roles = []Role{RoleFit, RoleSelect, RoleValidate, RoleHeld, RoleAnchor, RoleReport}

// Valid reports whether r is one of the six roles.
func (r Role) Valid() bool {
	for _, x := range Roles {
		if r == x {
			return true
		}
	}
	return false
}

// Published reports whether the release job publishes r's members: fit, anchor and report.
// The members of select, validate and held are known only inside the jobs that hold the seed.
func (r Role) Published() bool { return r == RoleFit || r == RoleAnchor || r == RoleReport }

// mustBeNonempty reports whether a published list of r holds a unit in every release: the
// fit list, which the fits cite (D1), and the anchor list, which the dashboard scores (COV-2).
func (r Role) mustBeNonempty() bool { return r == RoleFit || r == RoleAnchor }

// Side is a legacy label: the side of the partition made before the ledger existed.
type Side string

const (
	SideFit  Side = "FIT"
	SideHeld Side = "HELD"
)

// Valid reports whether s is FIT or HELD.
func (s Side) Valid() bool { return s == SideFit || s == SideHeld }

// Roles a legacy side may give: FIT gives fit or select, HELD gives held or anchor (D4).
func (s Side) roles() []Role {
	if s == SideFit {
		return []Role{RoleFit, RoleSelect}
	}
	return []Role{RoleHeld, RoleAnchor}
}

// SeedSize is the seed's length in bytes: 256 random bits (D7).
const SeedSize = 32

// Seed is the role seed. It exists only inside a job that holds it.
type Seed struct{ key []byte }

// ParseSeed reads a seed from its hex form and checks it against the committed hash. A seed
// that does not hash to want is refused, so a job cannot draw roles with the wrong secret.
func ParseSeed(hexSeed, want string) (Seed, error) {
	key, err := hex.DecodeString(hexSeed)
	if err != nil {
		return Seed{}, fmt.Errorf("seed: not hex: %v", err)
	}
	if len(key) != SeedSize {
		return Seed{}, fmt.Errorf("seed: %d bytes, want %d (256 bits, D7)", len(key), SeedSize)
	}
	sum := sha256.Sum256(key)
	if got := hex.EncodeToString(sum[:]); got != want {
		return Seed{}, fmt.Errorf("seed: its SHA-256 is %s, but the ledger records %s", got, want)
	}
	return Seed{key: key}, nil
}

// Draw returns HMAC-SHA-256(seed, message) mod 100, the digest read as a big-endian unsigned
// integer. The message is the canonical JSON of the role key (design §10.2).
func (s Seed) Draw(message []byte) int {
	m := hmac.New(sha256.New, s.key)
	m.Write(message)
	n := new(big.Int).SetBytes(m.Sum(nil))
	return int(new(big.Int).Mod(n, big.NewInt(100)).Int64())
}

// CutPoint sends the draws From..To, both included, to Role.
type CutPoint struct {
	Role Role `yaml:"role"`
	From int  `yaml:"from"`
	To   int  `yaml:"to"`
}

// CutPoints is a partition of 0..99 into roles, in increasing order.
type CutPoints []CutPoint

// Role returns the role the draw d falls in.
func (c CutPoints) Role(d int) (Role, error) {
	for _, p := range c {
		if d >= p.From && d <= p.To {
			return p.Role, nil
		}
	}
	return "", fmt.Errorf("draw %d falls in no cut-point", d)
}

// check reports whether c covers 0..99 exactly once, in order, with roles from allowed.
func (c CutPoints) check(at string, allowed []Role) error {
	if len(c) == 0 {
		return fmt.Errorf("%s: no cut-points", at)
	}
	next := 0
	for i, p := range c {
		if !contains(allowed, p.Role) {
			return fmt.Errorf("%s[%d].role: %q is not one of %v", at, i, p.Role, allowed)
		}
		if p.From != next {
			return fmt.Errorf("%s[%d].from: %d, want %d: cut-points cover 0..99 in order with no gap or overlap", at, i, p.From, next)
		}
		if p.To < p.From || p.To > 99 {
			return fmt.Errorf("%s[%d].to: %d is outside %d..99", at, i, p.To, p.From)
		}
		next = p.To + 1
	}
	if next != 100 {
		return fmt.Errorf("%s: cut-points end at %d, want 99", at, next-1)
	}
	return nil
}

func contains[T comparable](xs []T, x T) bool {
	for _, y := range xs {
		if x == y {
			return true
		}
	}
	return false
}

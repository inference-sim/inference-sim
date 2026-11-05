// Package ledger holds the data ledger of design §10: the committed sources, units, cut-points,
// legacy table and alias tables under catalog/ledger/, the keyed rule that gives each unit its
// role, and the checks that keep roles out of committed files.
//
// It is not a contracts package. Only inference-sim and the seed-holding jobs import it.
//
// A unit's role is computed, never stored, except in the three published lists the release
// job writes (fit, anchor, report) and in the legacy labels of legacy-map.yaml (D4). Nothing
// here writes a select, validate or held role to a file; Assign returns roles to the caller,
// and only a job that holds the seed can call it.
package ledger

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Canonical returns the canonical JSON encoding of v: UTF-8, object keys sorted by byte
// order, no insignificant space. It is the encoding accuracy-dashboard's contract.py
// canonical_dumps produces (json.dumps with sort_keys, separators (",", ":") and
// ensure_ascii=False), so a unit_id and the dashboard's point_id agree byte for byte.
//
// v may hold only nil, bool, string, int, int64 and json.Number numbers that are integers,
// []any and map[string]any. A float is refused: two encoders format floats differently, and
// no identifying field of any source is a float.
func Canonical(v any) ([]byte, error) {
	var b bytes.Buffer
	if err := encode(&b, v, ""); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// ID returns the hex SHA-256 of Canonical(v).
func ID(v any) (string, error) {
	c, err := Canonical(v)
	if err != nil {
		return "", err
	}
	return hashHex(c), nil
}

func hashHex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// MeasuredID returns the hex SHA-256 of the canonical encoding of a record's measured values.
// It differs from ID in one way: a json.Number with a fraction or an exponent is written as
// the literal text the source stored, because a measurement is a float and re-formatting it
// would make the hash depend on a formatter. Two records hash equal exactly when they hold the
// same keys and the same number literals.
func MeasuredID(v any) (string, error) {
	var b bytes.Buffer
	if err := encodeMeasured(&b, v, ""); err != nil {
		return "", err
	}
	sum := sha256.Sum256(b.Bytes())
	return hex.EncodeToString(sum[:]), nil
}

func encodeMeasured(b *bytes.Buffer, v any, at string) error {
	switch x := v.(type) {
	case json.Number:
		if !numberLiteral.MatchString(string(x)) {
			return fmt.Errorf("%s: %q is not a JSON number", where(at), x)
		}
		b.WriteString(string(x))
		return nil
	case []any:
		b.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				b.WriteByte(',')
			}
			if err := encodeMeasured(b, e, fmt.Sprintf("%s[%d]", at, i)); err != nil {
				return err
			}
		}
		b.WriteByte(']')
		return nil
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteByte(',')
			}
			writeString(b, k)
			b.WriteByte(':')
			if err := encodeMeasured(b, x[k], strings.TrimPrefix(at+"."+k, ".")); err != nil {
				return err
			}
		}
		b.WriteByte('}')
		return nil
	}
	return encode(b, v, at)
}

var numberLiteral = regexp.MustCompile(`^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$`)

func encode(b *bytes.Buffer, v any, at string) error {
	switch x := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if x {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case int:
		b.WriteString(strconv.Itoa(x))
	case int64:
		b.WriteString(strconv.FormatInt(x, 10))
	case json.Number:
		if _, err := strconv.ParseInt(string(x), 10, 64); err != nil {
			return fmt.Errorf("%s: %s is not an integer; identifying fields hold no fractions", where(at), x)
		}
		b.WriteString(string(x))
	case float64:
		if x == math.Trunc(x) {
			return fmt.Errorf("%s: %v is a float; pass an integer as int", where(at), x)
		}
		return fmt.Errorf("%s: %v is a float; identifying fields hold no fractions", where(at), x)
	case string:
		if !utf8.ValidString(x) {
			return fmt.Errorf("%s: string is not valid UTF-8", where(at))
		}
		writeString(b, x)
	case []any:
		b.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				b.WriteByte(',')
			}
			if err := encode(b, e, fmt.Sprintf("%s[%d]", at, i)); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteByte(',')
			}
			writeString(b, k)
			b.WriteByte(':')
			if err := encode(b, x[k], strings.TrimPrefix(at+"."+k, ".")); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	default:
		return fmt.Errorf("%s: cannot encode a %T canonically", where(at), v)
	}
	return nil
}

func where(at string) string {
	if at == "" {
		return "value"
	}
	return at
}

// writeString writes s as Python's json.dumps(ensure_ascii=False) does: the escapes \" \\ \n
// \r \t \b \f, \u00XX for the other control characters below 0x20, and every other character
// as itself. Python does not escape 0x7f, U+2028, U+2029, '/', '<', '>' or '&', and neither
// does this.
func writeString(b *bytes.Buffer, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

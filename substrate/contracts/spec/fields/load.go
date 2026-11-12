package fields

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// The keys each level of a field list declares. Parse checks every key against these by
// exact spelling: encoding/json alone would match a key case-insensitively, so "Kind" would
// read as "kind".
var (
	listKeys       = keySet("kind", "engine", "fields")
	engineKeys     = keySet("kind", "version")
	fieldKeys      = keySet("name", "source", "status", "evidence", "value_classes", "estimation")
	evidenceKeys   = keySet("traces", "proof", "issue", "note")
	estimationKeys = keySet("rule", "setting", "reason", "preserves", "valid_when")
	classesKeys    = keySet("type", "values", "covered", "thresholds", "covered_classes")
	thresholdKeys  = keySet("value", "closed")
	clauseKeys     = keySet("value_in", "value_list_has_any", "value_list_has_none",
		"value_not_prefixed", "tp_group_nodes", "compute_capability")
	comparisonKeys = keySet("eq", "gt", "gte", "lt", "lte")
)

func keySet(keys ...string) map[string]bool {
	m := map[string]bool{}
	for _, k := range keys {
		m[k] = true
	}
	return m
}

// Load reads and validates the field list at path.
func Load(path string) (*FieldList, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	l, err := Parse(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return l, nil
}

// Parse reads a field list strictly and validates it. It refuses a key the schema does not
// declare, a key that appears twice in one object, anything after the document, and a number
// that does not convert: an integer outside int64, or a decimal outside float64's range,
// overflow and underflow alike (SUB-12). JSON has no spelling for NaN or an infinity, so a
// non-finite number is a syntax error. Numbers in a substitute setting are kept as
// json.Number, with their exact text, so Marshal writes back what was read.
func Parse(data []byte) (*FieldList, error) {
	if err := checkStructure(data); err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	dec.DisallowUnknownFields()
	var l FieldList
	if err := dec.Decode(&l); err != nil {
		return nil, fmt.Errorf("field list: %w", err)
	}
	if err := l.Validate(); err != nil {
		return nil, err
	}
	return &l, nil
}

// Marshal writes the list's canonical bytes: JSON indented by two spaces, object keys in
// the declared order for the list's own types and sorted within a setting, no HTML
// escaping, and a final newline.
func (l *FieldList) Marshal() ([]byte, error) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(l); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// checkStructure walks the document's tokens and refuses an unknown or duplicated key, an
// inexact number and trailing data, naming where each is.
func checkStructure(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	w := &walker{dec: dec}
	if err := w.value("", levelList); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		if err == nil {
			return errors.New("field list: data after the document")
		}
		return fmt.Errorf("field list: after the document: %w", err)
	}
	return nil
}

// level is which part of the schema a JSON value is.
type level int

const (
	levelList level = iota
	levelEngine
	levelFields
	levelField
	levelEvidence
	levelEstimation
	levelClasses
	levelThresholds
	levelThreshold
	levelValidWhen
	levelClause
	levelComparison
	levelFree // inside a setting, or a list of trace paths: any keys, still checked for duplicates
)

type walker struct{ dec *json.Decoder }

func (w *walker) value(path string, lv level) error {
	tok, err := w.dec.Token()
	if err != nil {
		return fmt.Errorf("field list: at %s: %w", where(path), err)
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			return w.object(path, lv)
		case '[':
			return w.array(path, lv)
		}
		return fmt.Errorf("field list: at %s: unexpected %q", where(path), t)
	case json.Number:
		return exactNumber(path, t)
	}
	return nil
}

func (w *walker) object(path string, lv level) error {
	allowed := map[level]map[string]bool{levelList: listKeys, levelEngine: engineKeys,
		levelField: fieldKeys, levelEvidence: evidenceKeys, levelEstimation: estimationKeys,
		levelClasses: classesKeys, levelThreshold: thresholdKeys, levelClause: clauseKeys,
		levelComparison: comparisonKeys}[lv]
	seen := map[string]bool{}
	for w.dec.More() {
		tok, err := w.dec.Token()
		if err != nil {
			return fmt.Errorf("field list: at %s: %w", where(path), err)
		}
		key := tok.(string)
		if seen[key] {
			return fmt.Errorf("field list: at %s: key %q appears twice", where(path), key)
		}
		seen[key] = true
		if lv != levelFree && !allowed[key] {
			return fmt.Errorf("field list: at %s: unknown key %q", where(path), key)
		}
		child := levelFree
		switch {
		case lv == levelList && key == "engine":
			child = levelEngine
		case lv == levelList && key == "fields":
			child = levelFields
		case lv == levelField && key == "evidence":
			child = levelEvidence
		case lv == levelField && key == "estimation":
			child = levelEstimation
		case lv == levelField && key == "value_classes":
			child = levelClasses
		case lv == levelClasses && key == "thresholds":
			child = levelThresholds
		case lv == levelEstimation && key == "valid_when":
			child = levelValidWhen
		case lv == levelClause && (key == "tp_group_nodes" || key == "compute_capability"):
			child = levelComparison
		}
		at := join(path, key)
		if lv == levelField && key == "name" {
			// Read the name so later messages about this field can name it.
			tok, err := w.dec.Token()
			if err != nil {
				return fmt.Errorf("field list: at %s: %w", where(at), err)
			}
			if s, ok := tok.(string); ok {
				path = renamed(path, s)
				continue
			}
			if d, ok := tok.(json.Delim); ok && (d == '{' || d == '[') {
				return fmt.Errorf("field list: at %s: must be a string", where(at))
			}
			continue
		}
		if err := w.value(at, child); err != nil {
			return err
		}
	}
	_, err := w.dec.Token() // the closing brace
	return err
}

func (w *walker) array(path string, lv level) error {
	for i := 0; w.dec.More(); i++ {
		child := levelFree
		switch lv {
		case levelFields:
			child = levelField
		case levelThresholds:
			child = levelThreshold
		case levelValidWhen:
			child = levelClause
		}
		if err := w.value(fmt.Sprintf("%s[%d]", path, i), child); err != nil {
			return err
		}
	}
	_, err := w.dec.Token() // the closing bracket
	return err
}

// exactNumber refuses a number whose value float64 or int64 cannot hold: an integer literal
// outside int64, or a decimal that overflows or underflows float64.
func exactNumber(path string, n json.Number) error {
	s := n.String()
	if !strings.ContainsAny(s, ".eE") {
		if _, err := strconv.ParseInt(s, 10, 64); err != nil {
			return fmt.Errorf("field list: at %s: integer %s does not fit in 64 bits", where(path), s)
		}
		return nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || (f == 0 && nonZeroMantissa(s)) {
		return fmt.Errorf("field list: at %s: number %s is outside the range of a 64-bit float", where(path), s)
	}
	return nil
}

// nonZeroMantissa reports whether a JSON number's digits before its exponent are not all
// zero, so a parse to zero is an underflow rather than the value zero.
func nonZeroMantissa(s string) bool {
	if i := strings.IndexAny(s, "eE"); i >= 0 {
		s = s[:i]
	}
	return strings.Trim(s, "-+0.") != ""
}

func join(path, key string) string {
	if path == "" {
		return key
	}
	return path + "." + key
}

// renamed adds a field's name to its path, "fields[3]" becoming "fields[3](max_num_seqs)".
func renamed(path, name string) string {
	if name == "" || !strings.HasSuffix(path, "]") {
		return path
	}
	return path + "(" + name + ")"
}

func where(path string) string {
	if path == "" {
		return "the top level"
	}
	return path
}

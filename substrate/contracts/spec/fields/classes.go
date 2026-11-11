package fields

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"

	validate "github.com/inference-sim/blis-substrate/contracts/check"
)

// ValueClasses are the classes of a field's values that the field's evidence covers (design
// VLM-3). A scoring harness grades a run G1 only when each modelled field the run sets has a
// value in a covered class, and G2 when it sets one outside every covered class (design
// §11.3). The class is the unit of evidence: a scenario that sets a field to 128 says nothing
// about 4096 when the upstream compares the field with 1024.
//
// The kind of field decides the classes.
//
//   - A boolean field has two classes, "true" and "false". Covered names those the evidence
//     sets.
//   - An enumerated field has one class per value. Covered names the values the evidence
//     sets. Values, when given, is every value the upstream accepts.
//   - An ordered numeric field has one class per interval between the thresholds the
//     upstream compares the field with, generated from the upstream's comparisons. n
//     thresholds make n+1 classes, numbered from 0 for the values below the first. Each
//     threshold says which side takes the value equal to it, because `x > 8` and `x >= 8`
//     compare differently at 8. CoveredClasses names the classes the evidence sets a value in.
//
// A proven-inert field records the classes its probes cover, and its probes cover every class
// (a probe in one class shows nothing about another): see Field.Validate.
type ValueClasses struct {
	// Type says which of the three kinds the field is.
	Type ValueType `json:"type" yaml:"type"`
	// Values is every value an enumerated field accepts. Optional for a modelled field, which
	// may be an enumeration the engine cannot list in full; required for a proven-inert one,
	// whose probes must cover all of them.
	Values []string `json:"values,omitempty" yaml:"values,omitempty"`
	// Covered is the values of a boolean or enumerated field that the evidence sets, as
	// strings: "true" and "false" for a boolean.
	Covered []string `json:"covered,omitempty" yaml:"covered,omitempty"`
	// Thresholds are the values a numeric field is compared with, strictly increasing.
	Thresholds []Threshold `json:"thresholds,omitempty" yaml:"thresholds,omitempty"`
	// CoveredClasses are the numbered classes of a numeric field that the evidence sets a
	// value in, increasing.
	CoveredClasses []int `json:"covered_classes,omitempty" yaml:"covered_classes,omitempty"`
}

// ValueType is the kind of a field's values.
type ValueType string

const (
	// ValueBoolean: the field is true or false.
	ValueBoolean ValueType = "boolean"
	// ValueEnumerated: the field takes one of a set of named values.
	ValueEnumerated ValueType = "enumerated"
	// ValueNumeric: the field is an ordered number the upstream compares with thresholds.
	ValueNumeric ValueType = "numeric"
)

// Threshold is one value a numeric field is compared with.
type Threshold struct {
	// Value is the number the upstream compares the field with.
	Value float64 `json:"value" yaml:"value"`
	// Closed says which class a value equal to Value falls in: "above" for a comparison
	// such as x >= Value, "below" for x > Value.
	Closed Side `json:"closed" yaml:"closed"`
}

// Side is the class a threshold's own value falls in.
type Side string

const (
	// SideBelow: the value equal to the threshold is in the lower class.
	SideBelow Side = "below"
	// SideAbove: the value equal to the threshold is in the upper class.
	SideAbove Side = "above"
)

// classCount is how many classes the field has, or 0 when the count is not finite (an
// enumeration with no complete list).
func (c *ValueClasses) classCount() int {
	switch c.Type {
	case ValueBoolean:
		return 2
	case ValueNumeric:
		return len(c.Thresholds) + 1
	case ValueEnumerated:
		return len(c.Values)
	}
	return 0
}

// validate checks a field's value classes. status decides whether they are required.
func (c *ValueClasses) validate(p *validate.Problems, status Status) {
	need := status == StatusModelled || status == StatusProvenInert
	switch {
	case c == nil && need:
		p.Field("value_classes", "required for status %q: the classes of values the evidence covers (design VLM-3)", status)
		return
	case c == nil:
		return
	case !need:
		p.Field("value_classes", "not allowed for status %q: a field with no evidence covers no class", status)
		return
	}
	switch c.Type {
	case ValueBoolean, ValueEnumerated, ValueNumeric:
	default:
		p.Field("value_classes.type", "must be %q, %q or %q, got %q", ValueBoolean, ValueEnumerated, ValueNumeric, c.Type)
		return
	}
	numeric := c.Type == ValueNumeric
	if !numeric && (len(c.Thresholds) > 0 || len(c.CoveredClasses) > 0) {
		p.Field("value_classes", "thresholds and covered_classes are for a numeric field, not a %s one", c.Type)
	}
	if numeric && (len(c.Values) > 0 || len(c.Covered) > 0) {
		p.Field("value_classes", "values and covered are for a boolean or enumerated field, not a numeric one")
	}
	if c.Type != ValueEnumerated && len(c.Values) > 0 {
		p.Field("value_classes.values", "only an enumerated field lists its values")
	}
	if numeric {
		c.validateNumeric(p)
	} else {
		c.validateDiscrete(p)
	}
	if status == StatusProvenInert {
		c.validateProbesCoverEveryClass(p)
	}
}

func (c *ValueClasses) validateDiscrete(p *validate.Problems) {
	if len(c.Covered) == 0 {
		p.Field("value_classes.covered", "must name at least one value the evidence sets")
	}
	known := map[string]bool{}
	switch c.Type {
	case ValueBoolean:
		known = map[string]bool{"true": true, "false": true}
	case ValueEnumerated:
		seen := map[string]bool{}
		for i, v := range c.Values {
			switch {
			case v == "":
				p.Field(fmt.Sprintf("value_classes.values[%d]", i), "must not be empty")
			case seen[v]:
				p.Field(fmt.Sprintf("value_classes.values[%d]", i), "names %q twice", v)
			}
			seen[v] = true
		}
		if len(c.Values) > 0 {
			known = seen
		} else {
			known = nil
		}
	}
	seen := map[string]bool{}
	for i, v := range c.Covered {
		at := fmt.Sprintf("value_classes.covered[%d]", i)
		switch {
		case v == "":
			p.Field(at, "must not be empty")
		case seen[v]:
			p.Field(at, "names %q twice", v)
		case known != nil && !known[v]:
			if c.Type == ValueBoolean {
				p.Field(at, "a boolean's values are \"true\" and \"false\", got %q", v)
			} else {
				p.Field(at, "%q is not among the field's values", v)
			}
		}
		seen[v] = true
	}
}

func (c *ValueClasses) validateNumeric(p *validate.Problems) {
	for i, t := range c.Thresholds {
		at := fmt.Sprintf("value_classes.thresholds[%d]", i)
		if math.IsNaN(t.Value) || math.IsInf(t.Value, 0) {
			p.Field(at+".value", "must be finite")
		} else if i > 0 && t.Value <= c.Thresholds[i-1].Value {
			p.Field(at+".value", "thresholds are strictly increasing; %v follows %v", t.Value, c.Thresholds[i-1].Value)
		}
		if t.Closed != SideBelow && t.Closed != SideAbove {
			p.Field(at+".closed", "must be %q or %q, got %q", SideBelow, SideAbove, t.Closed)
		}
	}
	if len(c.CoveredClasses) == 0 {
		p.Field("value_classes.covered_classes", "must name at least one class the evidence sets a value in")
	}
	n := len(c.Thresholds) + 1
	for i, k := range c.CoveredClasses {
		at := fmt.Sprintf("value_classes.covered_classes[%d]", i)
		switch {
		case k < 0 || k >= n:
			p.Field(at, "class %d does not exist: %d thresholds make classes 0 to %d", k, len(c.Thresholds), n-1)
		case i > 0 && k <= c.CoveredClasses[i-1]:
			p.Field(at, "classes are listed once each, increasing; %d follows %d", k, c.CoveredClasses[i-1])
		}
	}
}

// validateProbesCoverEveryClass refuses a proven-inert field whose probes leave a class
// untested. An enumeration must list its values for the check to be possible.
func (c *ValueClasses) validateProbesCoverEveryClass(p *validate.Problems) {
	if c.Type == ValueEnumerated && len(c.Values) == 0 {
		p.Field("value_classes.values", "required for a proven-inert enumerated field: the probes must cover every value, and the list says which they are")
		return
	}
	covered := len(c.Covered)
	if c.Type == ValueNumeric {
		covered = len(c.CoveredClasses)
	}
	if n := c.classCount(); covered < n {
		p.Field("value_classes", "a proven-inert field's probes cover every class: %d of %d are covered", covered, n)
	}
}

// Covers reports whether value falls in a class the field's evidence covers. value is a bool,
// a string, an integer or float type, or a json.Number; a boolean field takes a bool or the
// string "true" or "false", an enumerated field a string, and a numeric field a number or
// the text of one. A value of the wrong kind is an error, not a "no": the harness has read
// the field wrongly.
func (c *ValueClasses) Covers(value any) (bool, error) {
	if c == nil {
		return false, fmt.Errorf("the field records no value classes")
	}
	switch c.Type {
	case ValueBoolean:
		var s string
		switch v := value.(type) {
		case bool:
			s = strconv.FormatBool(v)
		case string:
			s = v
		default:
			return false, fmt.Errorf("a boolean field takes a bool, not %T", value)
		}
		if s != "true" && s != "false" {
			return false, fmt.Errorf("a boolean field takes true or false, not %q", s)
		}
		return contains(c.Covered, s), nil
	case ValueEnumerated:
		s, ok := value.(string)
		if !ok {
			return false, fmt.Errorf("an enumerated field takes a string, not %T", value)
		}
		return contains(c.Covered, s), nil
	case ValueNumeric:
		x, err := number(value)
		if err != nil {
			return false, err
		}
		k := c.ClassOf(x)
		for _, covered := range c.CoveredClasses {
			if covered == k {
				return true, nil
			}
		}
		return false, nil
	}
	return false, fmt.Errorf("unknown value type %q", c.Type)
}

// ClassOf is the number of the class a numeric value falls in: the count of thresholds the
// value is above, a value equal to a threshold being above it when the threshold is closed
// above.
func (c *ValueClasses) ClassOf(x float64) int {
	k := 0
	for _, t := range c.Thresholds {
		if x > t.Value || (x == t.Value && t.Closed == SideAbove) {
			k++
		}
	}
	return k
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func number(v any) (float64, error) {
	var f float64
	switch x := v.(type) {
	case json.Number:
		return number(string(x))
	case string:
		var err error
		if f, err = strconv.ParseFloat(strings.TrimSpace(x), 64); err != nil {
			return 0, fmt.Errorf("a numeric field takes a finite number, not %q", x)
		}
	case nil, bool:
		return 0, fmt.Errorf("a numeric field takes a number, not %T", v)
	default:
		rv := reflect.ValueOf(v)
		switch rv.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			f = float64(rv.Int())
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			f = float64(rv.Uint())
		case reflect.Float32, reflect.Float64:
			f = rv.Float()
		default:
			return 0, fmt.Errorf("a numeric field takes a number, not %T", v)
		}
	}
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, fmt.Errorf("a numeric field takes a finite number, not %v", f)
	}
	return f, nil
}

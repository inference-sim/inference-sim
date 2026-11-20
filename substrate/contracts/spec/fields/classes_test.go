package fields

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// list wraps one field in a list, as a document, so a case goes through Parse.
func list(field string) []byte {
	return []byte(`{"kind":"FieldList","engine":{"kind":"vllm","version":"0.32.0"},"fields":[` + field + `]}`)
}

const (
	numericModelled = `{"name":"max_num_seqs","source":"config","status":"modelled","evidence":{"traces":["t.json"]},
	  "value_classes":{"type":"numeric","thresholds":[{"value":1,"closed":"above"},{"value":256,"closed":"below"}],"covered_classes":[1,2]},
	  "estimation":{"rule":"leave_unset"}}`
	boolModelled = `{"name":"enable_x","source":"config","status":"modelled","evidence":{"traces":["t.json"]},
	  "value_classes":{"type":"boolean","covered":["true"]},"estimation":{"rule":"leave_unset"}}`
	enumModelled = `{"name":"backend","source":"config","status":"modelled","evidence":{"traces":["t.json"]},
	  "value_classes":{"type":"enumerated","values":["a","b","c"],"covered":["a","b"]},"estimation":{"rule":"leave_unset"}}`
	enumOpen = `{"name":"backend","source":"config","status":"modelled","evidence":{"traces":["t.json"]},
	  "value_classes":{"type":"enumerated","covered":["a"]},"estimation":{"rule":"leave_unset"}}`
	boolInert = `{"name":"enable_x","source":"config","status":"proven_inert","evidence":{"proof":"TestX"},
	  "value_classes":{"type":"boolean","covered":["true","false"]},"estimation":{"rule":"leave_unset"}}`
)

func TestEachKindOfValueClassParses(t *testing.T) {
	for name, f := range map[string]string{"numeric": numericModelled, "boolean": boolModelled,
		"enumerated": enumModelled, "enumerated without a full list": enumOpen, "proven inert": boolInert} {
		if _, err := Parse(list(f)); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}

// A modelled field with no class is refused (design VLM-3), and so is every way the classes
// can be wrong. Each case is a known-bad document (SUB-2).
func TestValueClassesAreRefusedWhenWrong(t *testing.T) {
	rep := func(src, old, new string) []byte {
		if strings.Count(src, old) != 1 {
			t.Fatalf("%q occurs %d times", old, strings.Count(src, old))
		}
		return list(strings.Replace(src, old, new, 1))
	}
	noClasses := func(src string) []byte {
		i := strings.Index(src, `"value_classes"`)
		j := strings.Index(src, `"estimation"`)
		return list(src[:i] + src[j:])
	}
	for _, c := range []struct {
		name string
		doc  []byte
		want []string
	}{
		{"a modelled field with no class", noClasses(numericModelled),
			[]string{"fields[0](max_num_seqs).value_classes", "required for status \"modelled\""}},
		{"a proven-inert field with no class", noClasses(boolInert),
			[]string{"value_classes", "required for status \"proven_inert\""}},
		{"a not-modelled field with classes", list(`{"name":"a","source":"config","status":"not_modelled","evidence":{"issue":"o/r#1"},
		  "value_classes":{"type":"boolean","covered":["true"]},"estimation":{"rule":"none"}}`),
			[]string{"value_classes", "not allowed for status \"not_modelled\""}},
		{"an inert-unproven field with classes", list(`{"name":"a","source":"env","status":"inert_unproven","evidence":{"note":"n"},
		  "value_classes":{"type":"boolean","covered":["true"]},"estimation":{"rule":"none"}}`),
			[]string{"value_classes", "not allowed for status \"inert_unproven\""}},
		{"an unknown type", rep(boolModelled, `"type":"boolean"`, `"type":"integer"`), []string{"value_classes.type", "\"integer\""}},
		{"no type", rep(boolModelled, `"type":"boolean",`, ``), []string{"value_classes.type"}},
		{"a boolean covering nothing", rep(boolModelled, `"covered":["true"]`, `"covered":[]`), []string{"value_classes.covered", "at least one"}},
		{"a boolean covering a value it does not take", rep(boolModelled, `"covered":["true"]`, `"covered":["yes"]`), []string{"covered[0]", "\"true\" and \"false\""}},
		{"a value covered twice", rep(boolModelled, `"covered":["true"]`, `"covered":["true","true"]`), []string{"covered[1]", "twice"}},
		{"an empty covered value", rep(enumModelled, `"covered":["a","b"]`, `"covered":["a",""]`), []string{"covered[1]", "empty"}},
		{"an enumerated value outside the list", rep(enumModelled, `"covered":["a","b"]`, `"covered":["a","z"]`), []string{"covered[1]", "not among the field's values"}},
		{"a repeated enumerated value", rep(enumModelled, `"values":["a","b","c"]`, `"values":["a","b","b"]`), []string{"values[2]", "twice"}},
		{"values on a boolean", rep(boolModelled, `"covered":["true"]`, `"values":["true"],"covered":["true"]`), []string{"value_classes.values", "only an enumerated field"}},
		{"thresholds on a boolean", rep(boolModelled, `"covered":["true"]`, `"covered":["true"],"thresholds":[{"value":1,"closed":"above"}]`), []string{"thresholds and covered_classes are for a numeric field"}},
		{"covered values on a numeric field", rep(numericModelled, `"covered_classes":[1,2]`, `"covered_classes":[1,2],"covered":["1"]`), []string{"values and covered are for a boolean or enumerated field"}},
		{"thresholds out of order", rep(numericModelled, `"value":256`, `"value":0`), []string{"thresholds[1].value", "increasing; 0 follows 1"}},
		{"a repeated threshold", rep(numericModelled, `{"value":256,"closed":"below"}`, `{"value":1,"closed":"above"}`), []string{"thresholds[1].value", "empty class"}},
		{"a repeated threshold closed below", rep(numericModelled, `{"value":1,"closed":"above"}`, `{"value":256,"closed":"below"}`), []string{"thresholds[1].value", "empty class"}},
		{"x > v listed before x >= v", rep(numericModelled, `{"value":1,"closed":"above"},{"value":256,"closed":"below"}`, `{"value":9,"closed":"below"},{"value":9,"closed":"above"}`), []string{"thresholds[1].closed", "listed before"}},
		{"a threshold with no side", rep(numericModelled, `{"value":256,"closed":"below"}`, `{"value":256}`), []string{"thresholds[1].closed"}},
		{"a threshold with an unknown side", rep(numericModelled, `"closed":"below"`, `"closed":"left"`), []string{"thresholds[1].closed", "\"left\""}},
		{"a numeric field covering no class", rep(numericModelled, `"covered_classes":[1,2]`, `"covered_classes":[]`), []string{"covered_classes", "at least one"}},
		{"a class that does not exist", rep(numericModelled, `"covered_classes":[1,2]`, `"covered_classes":[1,3]`), []string{"covered_classes[1]", "does not exist"}},
		{"a negative class", rep(numericModelled, `"covered_classes":[1,2]`, `"covered_classes":[-1]`), []string{"covered_classes[0]", "does not exist"}},
		{"classes out of order", rep(numericModelled, `"covered_classes":[1,2]`, `"covered_classes":[2,1]`), []string{"covered_classes[1]", "increasing"}},
		{"a proven-inert boolean probed at one value", rep(boolInert, `"covered":["true","false"]`, `"covered":["true"]`), []string{"value_classes", "probes cover every class: 1 of 2"}},
		{"a proven-inert numeric probed in two of three classes", list(`{"name":"n","source":"config","status":"proven_inert","evidence":{"proof":"p"},
		  "value_classes":{"type":"numeric","thresholds":[{"value":1,"closed":"above"},{"value":9,"closed":"above"}],"covered_classes":[0,2]},
		  "estimation":{"rule":"none"}}`), []string{"probes cover every class: 2 of 3"}},
		{"a proven-inert enumeration with no full list", list(`{"name":"e","source":"config","status":"proven_inert","evidence":{"proof":"p"},
		  "value_classes":{"type":"enumerated","covered":["a"]},"estimation":{"rule":"none"}}`), []string{"value_classes.values", "required for a proven-inert enumerated field"}},
		{"a proven-inert enumeration probed at some values", list(`{"name":"e","source":"config","status":"proven_inert","evidence":{"proof":"p"},
		  "value_classes":{"type":"enumerated","values":["a","b"],"covered":["a"]},"estimation":{"rule":"none"}}`), []string{"probes cover every class: 1 of 2"}},
		{"an unknown key in the classes", rep(boolModelled, `"type":"boolean",`, `"type":"boolean","default":true,`), []string{"value_classes", "unknown key \"default\""}},
		{"an unknown key in a threshold", rep(numericModelled, `{"value":256,"closed":"below"}`, `{"value":256,"closed":"below","x":1}`), []string{"thresholds[1]", "unknown key \"x\""}},
		{"a repeated key in a threshold", rep(numericModelled, `{"value":256,"closed":"below"}`, `{"value":256,"value":3,"closed":"below"}`), []string{"key \"value\" appears twice"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := Parse(c.doc)
			if err == nil {
				t.Fatal("Parse accepted it")
			}
			for _, w := range c.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error does not mention %q:\n%v", w, err)
				}
			}
		})
	}
}

// ClassOf puts a value equal to a threshold on the side the threshold names. The two
// comparisons x >= 8 and x > 8 differ at 8 and nowhere else.
func TestAThresholdNamesTheSideOfItsOwnValue(t *testing.T) {
	ge := &ValueClasses{Type: ValueNumeric, Thresholds: []Threshold{{Value: 8, Closed: SideAbove}}}
	gt := &ValueClasses{Type: ValueNumeric, Thresholds: []Threshold{{Value: 8, Closed: SideBelow}}}
	for _, c := range []struct {
		x      float64
		ge, gt int
	}{{7.9, 0, 0}, {8, 1, 0}, {8.1, 1, 1}} {
		if got := ge.ClassOf(c.x); got != c.ge {
			t.Errorf("x >= 8: ClassOf(%v) = %d, want %d", c.x, got, c.ge)
		}
		if got := gt.ClassOf(c.x); got != c.gt {
			t.Errorf("x > 8: ClassOf(%v) = %d, want %d", c.x, got, c.gt)
		}
	}
}

func TestCoversTellsACoveredValueFromOneOutsideEveryClass(t *testing.T) {
	l, err := Parse(list(numericModelled))
	if err != nil {
		t.Fatal(err)
	}
	num := l.Fields[0].ValueClasses // classes: <1 | 1..256 | >256; covers the second and third
	for _, c := range []struct {
		v    any
		want bool
	}{{0, false}, {0.5, false}, {1, true}, {256, true}, {257, true}, {int64(4096), true},
		{json.Number("0"), false}, {json.Number("300"), true}, {"128", true}, {"-3", false}} {
		got, err := num.Covers(c.v)
		if err != nil || got != c.want {
			t.Errorf("Covers(%#v) = %v, %v; want %v", c.v, got, err, c.want)
		}
	}
	for _, c := range []struct {
		v    any
		want bool
	}{{int32(300), true}, {uint64(1 << 63), true}, {float32(0.5), false}, {uint8(1), true}, {int8(-1), false}} {
		if got, err := num.Covers(c.v); err != nil || got != c.want {
			t.Errorf("Covers(%T %v) = %v, %v; want %v", c.v, c.v, got, err, c.want)
		}
	}
	for _, bad := range []any{true, "many", "NaN", "Inf", nil, math.NaN(), math.Inf(1), math.Inf(-1), float32(math.NaN()), []int{1}} {
		if _, err := num.Covers(bad); err == nil {
			t.Errorf("a numeric field accepted %#v", bad)
		}
	}

	b, _ := Parse(list(boolModelled)) // covers true only
	for v, want := range map[any]bool{true: true, false: false, "true": true, "false": false} {
		if got, err := b.Fields[0].ValueClasses.Covers(v); err != nil || got != want {
			t.Errorf("boolean Covers(%#v) = %v, %v; want %v", v, got, err, want)
		}
	}
	if _, err := b.Fields[0].ValueClasses.Covers("yes"); err == nil {
		t.Error("a boolean field accepted \"yes\"")
	}

	e, _ := Parse(list(enumModelled)) // covers a and b of a, b, c
	for v, want := range map[string]bool{"a": true, "b": true, "c": false, "z": false} {
		if got, err := e.Fields[0].ValueClasses.Covers(v); err != nil || got != want {
			t.Errorf("enumerated Covers(%q) = %v, %v; want %v", v, got, err, want)
		}
	}
	if _, err := e.Fields[0].ValueClasses.Covers(3); err == nil {
		t.Error("an enumerated field accepted a number")
	}
	var none *ValueClasses
	if _, err := none.Covers(1); err == nil {
		t.Error("a field with no classes answered")
	}
}

// The boundary vectors the reference page states, so a restatement in another language can be
// checked against the same table.
func TestTheDocumentedBoundaryVectors(t *testing.T) {
	above := &ValueClasses{Type: ValueNumeric, Thresholds: []Threshold{{Value: 8, Closed: SideAbove}}}
	below := &ValueClasses{Type: ValueNumeric, Thresholds: []Threshold{{Value: 8, Closed: SideBelow}}}
	two := &ValueClasses{Type: ValueNumeric, Thresholds: []Threshold{{Value: 1, Closed: SideAbove}, {Value: 256, Closed: SideBelow}}}
	both := &ValueClasses{Type: ValueNumeric, Thresholds: []Threshold{{Value: 8, Closed: SideAbove}, {Value: 8, Closed: SideBelow}}}
	for _, c := range []struct {
		vc   *ValueClasses
		x    float64
		want int
	}{{above, 7.999, 0}, {above, 8, 1}, {above, 8.001, 1}, {below, 7.999, 0}, {below, 8, 0}, {below, 8.001, 1},
		{two, 0, 0}, {two, 1, 1}, {two, 256, 1}, {two, 257, 2},
		{both, 7.999, 0}, {both, 8, 1}, {both, 8.001, 2}} {
		if got := c.vc.ClassOf(c.x); got != c.want {
			t.Errorf("%v at %v: class %d, want %d", c.vc.Thresholds, c.x, got, c.want)
		}
	}
}

// An upstream that compares a field with x >= 8 in one place and x > 8 in another sends three
// values to three different sets of decisions: below 8, 8, and above 8. The list states both
// thresholds, and the three values are three classes (design VLM-3).
func TestOneFieldMayUseBothComparisonsAtOneValue(t *testing.T) {
	both := &ValueClasses{Type: ValueNumeric, Thresholds: []Threshold{{Value: 8, Closed: SideAbove}, {Value: 8, Closed: SideBelow}}}
	for _, c := range []struct {
		x    float64
		want int
	}{{7.999, 0}, {8, 1}, {8.001, 2}} {
		if got := both.ClassOf(c.x); got != c.want {
			t.Errorf("ClassOf(%v) = %d, want %d", c.x, got, c.want)
		}
	}
	doc := func(status, covered string) []byte {
		evidence := `{"traces":["t.json"]}`
		if status == "proven_inert" {
			evidence = `{"proof":"p"}`
		}
		return list(`{"name":"n","source":"config","status":"` + status + `","evidence":` + evidence + `,
		  "value_classes":{"type":"numeric","thresholds":[{"value":8,"closed":"above"},{"value":8,"closed":"below"}],"covered_classes":` + covered + `},
		  "estimation":{"rule":"none"}}`)
	}
	// Modelled and covering only the class at 8: 7 and 9 are outside every covered class.
	l, err := Parse(doc("modelled", "[1]"))
	if err != nil {
		t.Fatalf("both comparisons at one value were refused: %v", err)
	}
	vc := l.Fields[0].ValueClasses
	for _, c := range []struct {
		x    float64
		want bool
	}{{7, false}, {8, true}, {9, false}} {
		if got, err := vc.Covers(c.x); err != nil || got != c.want {
			t.Errorf("Covers(%v) = %v, %v; want %v", c.x, got, err, c.want)
		}
	}
	// Proven inert: probes in the classes below and above 8 leave 8 itself untested.
	if _, err := Parse(doc("proven_inert", "[0,2]")); err == nil || !strings.Contains(err.Error(), "probes cover every class: 2 of 3") {
		t.Errorf("probes below and above 8 but not at 8 were accepted as covering every class: %v", err)
	}
	if _, err := Parse(doc("proven_inert", "[0,1,2]")); err != nil {
		t.Errorf("probes in all three classes were refused: %v", err)
	}
}

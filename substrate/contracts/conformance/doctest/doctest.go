// Package doctest holds the malformed-document set of SUB-12: from one valid document of a
// type, it makes the documents every BLIS loader of that type must refuse. A repository
// that loads a BLIS document runs Cases against its loader (Run), the way testing/fstest
// checks an fs.FS.
//
// The cases are:
//
//   - an unknown key;
//   - a repeated key;
//   - a fraction in an integer field (yaml.v3 truncates it, blis-schemas#114);
//   - an integer a float64 field cannot hold exactly (yaml.v3 rounds it);
//   - a non-finite number, .nan and .inf, in a float field;
//   - a recursive alias (blis-schemas#107).
//
// A case that needs a field the document does not have, such as an integer field, is not
// made for it; Run reports which cases ran, and a caller checks that across its loaders
// every case ran at least once.
package doctest

import (
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// Case is one malformed document.
type Case struct {
	Name string
	Doc  []byte
	// Want is a substring the refusal must contain: the key or the value it is about.
	Want string
}

// Cases returns the malformed documents made from valid, a YAML document that decodes into
// target (a pointer to the document's Go type).
func Cases(valid []byte, target any) ([]Case, error) {
	parse := func() (*yaml.Node, error) {
		var doc yaml.Node
		if err := yaml.Unmarshal(valid, &doc); err != nil {
			return nil, err
		}
		if doc.Kind != yaml.DocumentNode || len(doc.Content) != 1 || doc.Content[0].Kind != yaml.MappingNode {
			return nil, fmt.Errorf("the valid document's top level is not a mapping")
		}
		return &doc, nil
	}
	root, err := parse()
	if err != nil {
		return nil, err
	}
	top := root.Content[0]
	var out []Case
	add := func(name, want string, doc *yaml.Node) error {
		b, err := yaml.Marshal(doc)
		if err != nil {
			return err
		}
		out = append(out, Case{Name: name, Doc: b, Want: want})
		return nil
	}

	// An unknown key, at the top level; in the first entry's value for a document whose
	// top level is a map of named entries, where any top-level key is an entry's name.
	if d, err := parse(); err == nil {
		m := d.Content[0]
		if tt := reflect.TypeOf(target); tt.Kind() == reflect.Pointer && tt.Elem().Kind() == reflect.Map &&
			len(m.Content) > 1 && m.Content[1].Kind == yaml.MappingNode {
			m = m.Content[1]
		}
		m.Content = append(m.Content, scalar("!!str", "zz_not_a_field"), scalar("!!int", "1"))
		if err := add("unknown key", "zz_not_a_field", d); err != nil {
			return nil, err
		}
	}
	// A repeated key: the first top-level key again, with its own value.
	if d, err := parse(); err == nil {
		m := d.Content[0]
		m.Content = append(m.Content, m.Content[0], m.Content[1])
		if err := add("repeated key", m.Content[0].Value, d); err != nil {
			return nil, err
		}
	}
	// The number cases, at the first integer and the first float field found.
	t := reflect.TypeOf(target)
	if path := find(top, t, isInt); path != nil {
		d, _ := parse()
		n := follow(d.Content[0], path)
		n.Tag, n.Value = "!!float", "1.5"
		if err := add("fraction in an integer field", "1.5", d); err != nil {
			return nil, err
		}
	}
	if path := find(top, t, isFloat); path != nil {
		for _, c := range []struct{ name, tag, value string }{
			{"integer a float64 cannot hold", "!!int", "9007199254740993"},
			{"NaN", "!!float", ".nan"},
			{"infinity", "!!float", "-.inf"},
		} {
			d, _ := parse()
			n := follow(d.Content[0], path)
			n.Tag, n.Value, n.Style = c.tag, c.value, 0
			if err := add(c.name, c.value, d); err != nil {
				return nil, err
			}
		}
	}
	// A recursive alias, as the value of the first top-level key. yaml.Marshal cannot write
	// a cyclic tree, so the line is spliced into the text.
	b, err := yaml.Marshal(root)
	if err != nil {
		return nil, err
	}
	key := top.Content[0].Value
	lines := strings.Split(string(b), "\n")
	var buf bytes.Buffer
	replaced := false
	skipping := false
	for _, l := range lines {
		switch {
		case !replaced && strings.HasPrefix(l, key+":"):
			fmt.Fprintf(&buf, "%s: &loop [*loop]\n", key)
			replaced, skipping = true, true
		case skipping && (strings.HasPrefix(l, " ") || strings.HasPrefix(l, "-")):
		default:
			skipping = false
			if l != "" {
				buf.WriteString(l + "\n")
			}
		}
	}
	out = append(out, Case{Name: "recursive alias", Doc: buf.Bytes(), Want: "loop"})
	return out, nil
}

// Run loads each case with load and fails t for a case that loads, or whose refusal does
// not contain the case's Want. It returns the names of the cases run.
func Run(t *testing.T, valid []byte, target any, load func([]byte) error) []string {
	t.Helper()
	if err := load(valid); err != nil {
		t.Fatalf("the valid document does not load: %v", err)
	}
	cases, err := Cases(valid, target)
	if err != nil {
		t.Fatal(err)
	}
	var ran []string
	for _, c := range cases {
		ran = append(ran, c.Name)
		err := load(c.Doc)
		switch {
		case err == nil:
			t.Errorf("%s: loaded:\n%s", c.Name, c.Doc)
		case !strings.Contains(err.Error(), c.Want):
			t.Errorf("%s: refused, but the error does not name %q: %v", c.Name, c.Want, err)
		}
	}
	return ran
}

func scalar(tag, v string) *yaml.Node { return &yaml.Node{Kind: yaml.ScalarNode, Tag: tag, Value: v} }

func isInt(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return true
	}
	return false
}

func isFloat(t reflect.Type) bool { return t.Kind() == reflect.Float64 }

// find returns the path, as child indices, to the first scalar in n whose field type
// satisfies want, descending by the Go type as the decoder would.
func find(n *yaml.Node, t reflect.Type, want func(reflect.Type) bool) []int {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch n.Kind {
	case yaml.ScalarNode:
		if want(t) && (n.ShortTag() == "!!int" || n.ShortTag() == "!!float") {
			return []int{}
		}
	case yaml.MappingNode:
		switch t.Kind() {
		case reflect.Struct:
			fields := fieldTypes(t)
			for i := 0; i+1 < len(n.Content); i += 2 {
				if ft, ok := fields[n.Content[i].Value]; ok {
					if p := find(n.Content[i+1], ft, want); p != nil {
						return append([]int{i + 1}, p...)
					}
				}
			}
		case reflect.Map:
			for i := 1; i < len(n.Content); i += 2 {
				if p := find(n.Content[i], t.Elem(), want); p != nil {
					return append([]int{i}, p...)
				}
			}
		}
	case yaml.SequenceNode:
		if t.Kind() == reflect.Slice || t.Kind() == reflect.Array {
			for i, c := range n.Content {
				if p := find(c, t.Elem(), want); p != nil {
					return append([]int{i}, p...)
				}
			}
		}
	}
	return nil
}

func follow(n *yaml.Node, path []int) *yaml.Node {
	for _, i := range path {
		n = n.Content[i]
	}
	return n
}

// fieldTypes maps the keys a struct decodes to their field types, merging inline structs.
func fieldTypes(t reflect.Type) map[string]reflect.Type {
	out := map[string]reflect.Type{}
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		tag := f.Tag.Get("yaml")
		if tag == "-" || (f.PkgPath != "" && !f.Anonymous) {
			continue
		}
		key, opts, _ := strings.Cut(tag, ",")
		if strings.Contains(","+opts+",", ",inline,") {
			ft := f.Type
			for ft.Kind() == reflect.Pointer {
				ft = ft.Elem()
			}
			if ft.Kind() == reflect.Struct {
				for k, v := range fieldTypes(ft) {
					out[k] = v
				}
			}
			continue
		}
		if key == "" {
			key = strings.ToLower(f.Name)
		}
		out[key] = f.Type
	}
	return out
}

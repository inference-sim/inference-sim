package validate

import (
	"fmt"
	"math/big"
	"reflect"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// yaml.v3 accepts some documents that say something other than what they decode to
// (SUB-12, blis-schemas#114, #107):
//
//   - a fraction in an integer field is truncated (seed: 1.5 loads as 1);
//   - an integer too large for a float64 field to hold exactly is rounded;
//   - .inf and .nan load as non-finite numbers;
//   - a key repeated inside a block held as a node (an engine block) is kept until someone
//     decodes the block;
//   - an anchor whose value contains an alias to itself makes a node tree a graph, and a
//     tree walk over it never ends.
//
// CheckNode refuses each of these before a document is decoded, so every loader refuses
// them the same way. Unknown keys are refused by the decoder's KnownFields, and by
// RejectUnknownKeys in custom unmarshallers.

// CheckNode reports the first construct in n that a strict loader refuses, with its line.
// target is the type n will be decoded into, or nil when it is not known; the number rules
// that depend on a field's type are applied where the type is known, and stop at a type
// with its own UnmarshalYAML, which checks its own node with DecodeNode.
func CheckNode(n *yaml.Node, target reflect.Type) error {
	if err := checkShape(n, map[*yaml.Node]bool{}); err != nil {
		return err
	}
	if target == nil {
		return checkFinite(n, "")
	}
	return checkNumbers(n, target, "", 0)
}

// checkFinite refuses a non-finite number anywhere in n. YAML spells them .inf, -.inf and
// .nan, in any case; a JSON document converted by JSONNode cannot hold one.
func checkFinite(n *yaml.Node, at string) error {
	for n != nil && n.Kind == yaml.AliasNode {
		n = n.Alias
	}
	if n == nil {
		return nil
	}
	if n.Kind == yaml.ScalarNode {
		if v := strings.ToLower(strings.TrimLeft(n.Value, "+-")); n.ShortTag() == "!!float" &&
			(v == ".inf" || v == ".nan") {
			return fmt.Errorf("line %d: %s: %s is not a finite number", n.Line, name(at), n.Value)
		}
		return nil
	}
	for i, c := range n.Content {
		sub := at
		if n.Kind == yaml.MappingNode {
			if i%2 == 0 {
				continue
			}
			sub = join(at, n.Content[i-1].Value)
		} else if n.Kind == yaml.SequenceNode {
			sub = fmt.Sprintf("%s[%d]", at, i)
		}
		if err := checkFinite(c, sub); err != nil {
			return err
		}
	}
	return nil
}

// DecodeNode is node.Decode after CheckNode. A custom UnmarshalYAML calls it for the
// value it decodes, so the number rules reach fields under it.
func DecodeNode(n *yaml.Node, dst any) error {
	if err := CheckNode(n, reflect.TypeOf(dst)); err != nil {
		return err
	}
	return n.Decode(dst)
}

// checkShape refuses recursive aliases and repeated keys anywhere in n.
// onPath holds the alias targets being expanded, so an alias to an ancestor is found.
func checkShape(n *yaml.Node, onPath map[*yaml.Node]bool) error {
	if n == nil {
		return nil
	}
	switch n.Kind {
	case yaml.AliasNode:
		if onPath[n.Alias] {
			return fmt.Errorf("line %d: the alias *%s refers to an anchor that contains it; a "+
				"recursive alias has no finite value", n.Line, n.Value)
		}
		onPath[n.Alias] = true
		defer delete(onPath, n.Alias)
		return checkShape(n.Alias, onPath)
	case yaml.MappingNode:
		seen := map[string]int{}
		for i := 0; i+1 < len(n.Content); i += 2 {
			k := n.Content[i]
			if k.Kind == yaml.ScalarNode && !(k.Value == "<<" && (k.Tag == "!!merge" || k.Tag == "")) {
				if first, dup := seen[k.Value]; dup {
					return fmt.Errorf("line %d: mapping key %q already defined at line %d", k.Line, k.Value, first)
				}
				seen[k.Value] = k.Line
			}
		}
	}
	for _, c := range n.Content {
		if err := checkShape(c, onPath); err != nil {
			return err
		}
	}
	return nil
}

var unmarshaler = reflect.TypeOf((*yaml.Unmarshaler)(nil)).Elem()

// checkNumbers applies the number rules that depend on the field type, descending as the
// decoder does. Aliases are followed; checkShape has already refused a recursive one.
func checkNumbers(n *yaml.Node, t reflect.Type, at string, depth int) error {
	if depth > 256 {
		return nil
	}
	for n != nil && n.Kind == yaml.AliasNode {
		n = n.Alias
	}
	if n != nil && n.Kind == yaml.DocumentNode && len(n.Content) == 1 {
		n = n.Content[0]
	}
	if n == nil {
		return nil
	}
	for t.Kind() == reflect.Pointer {
		if depth > 0 && (t.Implements(unmarshaler) || t.Elem().Implements(unmarshaler)) {
			return nil
		}
		t = t.Elem()
	}
	if depth > 0 && (t.Implements(unmarshaler) || reflect.PointerTo(t).Implements(unmarshaler)) {
		return nil
	}
	switch t.Kind() {
	case reflect.Interface:
		// An opaque value (map[string]any) has no field types; only the rule that holds
		// for every number applies.
		return checkFinite(n, at)
	case reflect.Struct:
		if n.Kind != yaml.MappingNode {
			return nil
		}
		fields := yamlFieldTypes(t)
		for i := 0; i+1 < len(n.Content); i += 2 {
			k, v := n.Content[i], n.Content[i+1]
			if k.Value == "<<" {
				if err := checkNumbers(v, t, at, depth+1); err != nil {
					return err
				}
				continue
			}
			if ft, ok := fields[k.Value]; ok {
				if err := checkNumbers(v, ft, join(at, k.Value), depth+1); err != nil {
					return err
				}
			}
		}
	case reflect.Map:
		if n.Kind == yaml.MappingNode {
			for i := 0; i+1 < len(n.Content); i += 2 {
				if err := checkNumbers(n.Content[i+1], t.Elem(), join(at, n.Content[i].Value), depth+1); err != nil {
					return err
				}
			}
		}
	case reflect.Slice, reflect.Array:
		if n.Kind == yaml.SequenceNode {
			for i, c := range n.Content {
				if err := checkNumbers(c, t.Elem(), fmt.Sprintf("%s[%d]", at, i), depth+1); err != nil {
					return err
				}
			}
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		if n.Kind == yaml.ScalarNode && n.ShortTag() == "!!float" {
			return fmt.Errorf("line %d: %s: %s is not an integer, and the field holds one; it is "+
				"refused rather than truncated", n.Line, name(at), n.Value)
		}
	case reflect.Float32, reflect.Float64:
		if n.Kind != yaml.ScalarNode {
			return nil
		}
		bits := t.Bits()
		if err := checkFinite(n, at); err != nil {
			return err
		}
		switch n.ShortTag() {
		case "!!int":
			if !exactInt(n, bits) {
				return fmt.Errorf("line %d: %s: %s cannot be held exactly by a %d-bit float, and "+
					"would be rounded", n.Line, name(at), n.Value, bits)
			}
		case "!!float":
			if _, err := strconv.ParseFloat(n.Value, bits); err != nil && strings.Contains(err.Error(), "range") {
				return fmt.Errorf("line %d: %s: %s is out of range for a %d-bit float", n.Line, name(at), n.Value, bits)
			}
		}
	}
	return nil
}

// exactInt reports whether an integer scalar converts to a float of the given width with no
// rounding.
func exactInt(n *yaml.Node, bits int) bool {
	var i big.Int
	var s string
	if err := n.Decode(&s); err != nil {
		return false
	}
	if _, ok := i.SetString(strings.ReplaceAll(s, "_", ""), 0); !ok {
		return false
	}
	f := new(big.Float).SetInt(&i)
	if bits == 32 {
		_, acc := f.Float32()
		return acc == big.Exact
	}
	_, acc := f.Float64()
	return acc == big.Exact
}

// yamlFieldTypes maps the keys a struct decodes to their field types, merging inline
// structs.
func yamlFieldTypes(t reflect.Type) map[string]reflect.Type {
	out := map[string]reflect.Type{}
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if f.PkgPath != "" && !f.Anonymous {
			continue
		}
		tag := f.Tag.Get("yaml")
		if tag == "-" {
			continue
		}
		key, opts, _ := strings.Cut(tag, ",")
		if strings.Contains(","+opts+",", ",inline,") {
			ft := f.Type
			for ft.Kind() == reflect.Pointer {
				ft = ft.Elem()
			}
			if ft.Kind() == reflect.Struct {
				for k, v := range yamlFieldTypes(ft) {
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

func join(at, key string) string {
	if at == "" {
		return key
	}
	return at + "." + key
}

func name(at string) string {
	if at == "" {
		return "the value"
	}
	return at
}

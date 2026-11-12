package validate

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math/big"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// yaml.v3 accepts some documents that say something other than what they decode to
// (SUB-12, blis-schemas#114, #107):
//
//   - a fraction in an integer field is truncated (seed: 1.5 loads as 1);
//   - an integer too large for a float64 field to hold exactly is rounded;
//   - .inf and .nan load as non-finite numbers (a field whose meaning includes "no limit"
//     opts in to .inf with the struct tag infinity:"allowed"; .nan and -.inf are refused
//     everywhere);
//   - a key repeated inside a block held as a node (an engine block) is kept until someone
//     decodes the block;
//   - an anchor whose value contains an alias to itself makes a node tree a graph, and a
//     tree walk over it never ends.
//
// CheckNode refuses each of these before a document is decoded, so every loader refuses
// them the same way. Unknown keys are refused by the decoder's KnownFields, and by
// RejectUnknownKeys in custom unmarshallers.

// MaxNodes bounds the nodes a document may expand to, counting each alias as the whole
// subtree it names. Aliases make a few hundred bytes expand to billions of nodes (eight
// nested anchors of ten aliases each parse in a fraction of a millisecond and expand to 10^8),
// and every pass over the document pays for the expansion. The largest document the project
// reads expands to well under a million nodes, so this is several times that and still
// refuses a bomb in under a second.
const MaxNodes = 1 << 22

// MaxDocumentBytes bounds the size of one document a strict loader reads. The largest data
// file in the catalog is about 7 MB; a file past this is not a configuration document.
const MaxDocumentBytes = 64 << 20

// MaxDepth bounds how deeply the number rules descend into a document. A document nested
// deeper than this is refused, not passed unchecked: stopping the descent quietly would
// leave every number below it unread. No document the project reads nests a tenth as deep.
const MaxDepth = 256

// CheckSize refuses a document of n bytes that is larger than MaxDocumentBytes.
func CheckSize(path string, n int) error {
	if n > MaxDocumentBytes {
		return fmt.Errorf("%s is %d bytes; a document is limited to %d bytes (MaxDocumentBytes)",
			path, n, MaxDocumentBytes)
	}
	return nil
}

// CheckNode reports the first construct in n that a strict loader refuses, with its line.
// target is the type n will be decoded into, or nil when it is not known; the number rules
// that depend on a field's type are applied where the type is known, and stop at a type
// with its own UnmarshalYAML, which checks its own node with DecodeNode.
func CheckNode(n *yaml.Node, target reflect.Type) error {
	count := 0
	if err := checkShape(n, map[*yaml.Node]bool{}, &count); err != nil {
		return err
	}
	if target == nil {
		return checkFinite(n, "")
	}
	return checkNumbers(n, target, "", 0, false)
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
		if nonFinite(n) != "" {
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

// nonFinite reports which non-finite number a scalar spells: "+inf", "-inf" or "nan", and ""
// for any other scalar. YAML spells them .inf, -.inf and .nan, in any case.
func nonFinite(n *yaml.Node) string {
	if n.Kind != yaml.ScalarNode || n.ShortTag() != "!!float" {
		return ""
	}
	v := strings.ToLower(n.Value)
	neg := strings.HasPrefix(v, "-")
	v = strings.TrimLeft(v, "+-")
	switch v {
	case ".inf":
		if neg {
			return "-inf"
		}
		return "+inf"
	case ".nan":
		return "nan"
	}
	return ""
}

// InfinityTag is the struct tag that lets a float field hold +infinity: infinity:"allowed".
// It is for a field whose meaning includes "no limit", such as a grace period that waits
// indefinitely. It admits .inf and +.inf and nothing else: -.inf and .nan stay refused (SUB-12).
const InfinityTag = "infinity"

// InfinityAllowed is the value of InfinityTag that opts a field in.
const InfinityAllowed = "allowed"

// AllowsInfinity reports whether a struct field opts in to +infinity.
func AllowsInfinity(f reflect.StructField) bool { return f.Tag.Get(InfinityTag) == InfinityAllowed }

// InfinityNodes returns the scalar nodes of n that sit under a float field opting in to
// +infinity (AllowsInfinity), found by descending as the decoder does with the type t n will
// be decoded into. A writer uses it to spell those values for an encoding that has no
// infinity, and a reader to turn that spelling back; both then agree with the strict loader
// on which fields hold one.
func InfinityNodes(n *yaml.Node, t reflect.Type) []*yaml.Node {
	var out []*yaml.Node
	collectInfinity(n, t, false, 0, &out)
	return out
}

func collectInfinity(n *yaml.Node, t reflect.Type, inf bool, depth int, out *[]*yaml.Node) {
	if depth > MaxDepth {
		return
	}
	n = resolve(n)
	if n != nil && n.Kind == yaml.DocumentNode && len(n.Content) == 1 {
		n = resolve(n.Content[0])
	}
	if n == nil {
		return
	}
	for t.Kind() == reflect.Pointer {
		if depth > 0 && (t.Implements(unmarshaler) || t.Elem().Implements(unmarshaler)) {
			return
		}
		t = t.Elem()
	}
	if depth > 0 && (t.Implements(unmarshaler) || reflect.PointerTo(t).Implements(unmarshaler)) {
		return
	}
	switch t.Kind() {
	case reflect.Struct:
		if n.Kind != yaml.MappingNode {
			return
		}
		fields := yamlFields(t)
		for i := 0; i+1 < len(n.Content); i += 2 {
			k, v := n.Content[i], n.Content[i+1]
			if k.Value == "<<" {
				members := []*yaml.Node{v}
				if m := resolve(v); m != nil && m.Kind == yaml.SequenceNode {
					members = m.Content
				}
				for _, m := range members {
					collectInfinity(m, t, inf, depth+1, out)
				}
				continue
			}
			if ft, ok := fields[k.Value]; ok {
				collectInfinity(v, ft.typ, ft.inf, depth+1, out)
			}
		}
	case reflect.Map:
		if n.Kind == yaml.MappingNode {
			for i := 0; i+1 < len(n.Content); i += 2 {
				collectInfinity(n.Content[i+1], t.Elem(), inf, depth+1, out)
			}
		}
	case reflect.Slice, reflect.Array:
		if n.Kind == yaml.SequenceNode {
			for _, c := range n.Content {
				collectInfinity(c, t.Elem(), inf, depth+1, out)
			}
		}
	case reflect.Float32, reflect.Float64:
		if inf && n.Kind == yaml.ScalarNode {
			*out = append(*out, n)
		}
	}
}

// InfinityFields lists the fields of t, and of the types it holds, that opt in to +infinity,
// as "Type.Field". A test names each one, so a field cannot opt in unnoticed.
func InfinityFields(t reflect.Type) []string {
	seen := map[reflect.Type]bool{}
	var out []string
	var walk func(t reflect.Type)
	walk = func(t reflect.Type) {
		for t.Kind() == reflect.Pointer || t.Kind() == reflect.Slice || t.Kind() == reflect.Array || t.Kind() == reflect.Map {
			t = t.Elem()
		}
		if t.Kind() != reflect.Struct || seen[t] {
			return
		}
		seen[t] = true
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if f.PkgPath != "" && !f.Anonymous || f.Tag.Get("yaml") == "-" {
				continue
			}
			if AllowsInfinity(f) {
				out = append(out, t.Name()+"."+f.Name)
			}
			walk(f.Type)
		}
	}
	walk(t)
	sort.Strings(out)
	return out
}

// DecodeNode is node.Decode after CheckNode. A custom UnmarshalYAML calls it for the
// value it decodes, so the number rules reach fields under it.
func DecodeNode(n *yaml.Node, dst any) error {
	if err := CheckNode(n, reflect.TypeOf(dst)); err != nil {
		return err
	}
	return n.Decode(dst)
}

// DecodeStrict decodes one YAML document into dst the way every loader of a BLIS document
// does (design SUB-12): it refuses a document over MaxDocumentBytes, a repeated key, a
// recursive alias, a non-finite number, a number that does not convert exactly to its field,
// an unknown key, and a second document. name is the file or source, for the message.
//
// Callers that read a BLIS document use this, or a function that ends here, so there is one
// definition of "strict" to test. An empty document is the caller's to treat; DecodeStrict
// returns the decoder's own end-of-input error for it.
func DecodeStrict(name string, raw []byte, dst any) error {
	if err := CheckSize(name, len(raw)); err != nil {
		return err
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	if err := CheckNode(&doc, reflect.TypeOf(dst)); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true)
	if err := dec.Decode(dst); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	if err := dec.Decode(&yaml.Node{}); !errors.Is(err, io.EOF) {
		return fmt.Errorf("%s: holds a second YAML document, which would be ignored; a file is one document", name)
	}
	return nil
}

// checkShape refuses recursive aliases and repeated keys anywhere in n.
// onPath holds the alias targets being expanded, so an alias to an ancestor is found.
// count is the number of nodes visited, aliases expanded; it is held to MaxNodes.
func checkShape(n *yaml.Node, onPath map[*yaml.Node]bool, count *int) error {
	if n == nil {
		return nil
	}
	if *count++; *count > MaxNodes {
		return fmt.Errorf("line %d: the document expands to more than %d nodes once aliases "+
			"are followed (MaxNodes); an alias that nests aliases grows exponentially", n.Line, MaxNodes)
	}
	switch n.Kind {
	case yaml.AliasNode:
		if onPath[n.Alias] {
			return fmt.Errorf("line %d: the alias *%s refers to an anchor that contains it; a "+
				"recursive alias has no finite value", n.Line, n.Value)
		}
		onPath[n.Alias] = true
		defer delete(onPath, n.Alias)
		return checkShape(n.Alias, onPath, count)
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
		if err := checkShape(c, onPath, count); err != nil {
			return err
		}
	}
	return nil
}

var unmarshaler = reflect.TypeOf((*yaml.Unmarshaler)(nil)).Elem()

// checkNumbers applies the number rules that depend on the field type, descending as the
// decoder does. Aliases are followed; checkShape has already refused a recursive one, so
// the descent ends, and a document deeper than MaxDepth is refused. inf is true below a field
// that opts in to +infinity (AllowsInfinity); it applies to the floats that field holds.
func checkNumbers(n *yaml.Node, t reflect.Type, at string, depth int, inf bool) error {
	if depth > MaxDepth {
		return fmt.Errorf("line %d: %s: the document nests deeper than %d levels (MaxDepth); "+
			"its numbers below this point cannot be checked", lineOf(n), name(at), MaxDepth)
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
		fields := yamlFields(t)
		for i := 0; i+1 < len(n.Content); i += 2 {
			k, v := n.Content[i], n.Content[i+1]
			if k.Value == "<<" {
				// A merge names one mapping or a sequence of them; each member is
				// read as if its keys were written here.
				members := []*yaml.Node{v}
				if m := resolve(v); m != nil && m.Kind == yaml.SequenceNode {
					members = m.Content
				}
				for _, m := range members {
					if err := checkNumbers(m, t, at, depth+1, inf); err != nil {
						return err
					}
				}
				continue
			}
			if ft, ok := fields[k.Value]; ok {
				if err := checkNumbers(v, ft.typ, join(at, k.Value), depth+1, ft.inf); err != nil {
					return err
				}
			}
		}
	case reflect.Map:
		if n.Kind == yaml.MappingNode {
			for i := 0; i+1 < len(n.Content); i += 2 {
				if err := checkNumbers(n.Content[i+1], t.Elem(), join(at, n.Content[i].Value), depth+1, inf); err != nil {
					return err
				}
			}
		}
	case reflect.Slice, reflect.Array:
		if n.Kind == yaml.SequenceNode {
			for i, c := range n.Content {
				if err := checkNumbers(c, t.Elem(), fmt.Sprintf("%s[%d]", at, i), depth+1, inf); err != nil {
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
		switch nf := nonFinite(n); {
		case nf == "+inf" && inf:
			return nil
		case nf == "+inf":
			return fmt.Errorf("line %d: %s: %s is not a finite number; the field does not allow infinity "+
				"(a field opts in with the struct tag infinity:%q)", n.Line, name(at), n.Value, InfinityAllowed)
		case nf != "":
			// -.inf and .nan: no field holds either, whatever it opts in to.
			return fmt.Errorf("line %d: %s: %s is not a finite number, and no field holds it", n.Line, name(at), n.Value)
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

// yamlField is what a struct key decodes into: the field's type, and whether it opts in to
// +infinity.
type yamlField struct {
	typ reflect.Type
	inf bool
}

// yamlFields maps the keys a struct decodes to their field types and infinity opt-ins,
// merging inline structs.
func yamlFields(t reflect.Type) map[string]yamlField {
	out := map[string]yamlField{}
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
				for k, v := range yamlFields(ft) {
					out[k] = v
				}
			}
			continue
		}
		if key == "" {
			key = strings.ToLower(f.Name)
		}
		out[key] = yamlField{typ: f.Type, inf: AllowsInfinity(f)}
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

// resolve follows aliases to the node they name.
func resolve(n *yaml.Node) *yaml.Node {
	for n != nil && n.Kind == yaml.AliasNode {
		n = n.Alias
	}
	return n
}

func lineOf(n *yaml.Node) int {
	if n == nil {
		return 0
	}
	return n.Line
}

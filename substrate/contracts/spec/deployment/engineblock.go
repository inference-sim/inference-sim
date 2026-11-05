package deployment

import (
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// EngineBlock is a block of a deployment that belongs to its engine, held unopened: a pool's
// engine: block, or an offload's connector_options. Which keys it may carry, their defaults
// and which values are valid are the engine model's to say, not this schema's: an engine
// model opens the block when it builds an engine, and a schema that knew one engine's keys
// would churn with that engine's releases. So the block passes through this package
// unchanged: decoding a deployment keeps the node as written, and encoding it writes the node
// back.
//
// The zero value is an absent block.
type EngineBlock struct {
	node *yaml.Node
}

// NewEngineBlock holds a copy of n as an engine block. A nil n, or a null node, is an absent
// block.
func NewEngineBlock(n *yaml.Node) EngineBlock {
	if n == nil || n.Kind == 0 || n.Kind == yaml.ScalarNode && n.Tag == "!!null" {
		return EngineBlock{}
	}
	return EngineBlock{node: cloneNode(n)}
}

// IsZero reports whether the pool states no engine block. yaml.v3 reads it for omitempty.
func (b EngineBlock) IsZero() bool { return b.node == nil }

// Node returns a copy of the block's node, nil for an absent block. Editing the copy leaves
// the block unchanged.
func (b EngineBlock) Node() *yaml.Node {
	if b.node == nil {
		return nil
	}
	return cloneNode(b.node)
}

// Decode decodes the block into v strictly, as the schema's own documents are decoded
// (decodeStrict): a key a struct in v has no field for is an error, reported at its line in
// the file the block was read from, in the decoder's own words. An absent block decodes
// nothing and leaves v as it was.
func (b EngineBlock) Decode(v any) error {
	if b.node == nil {
		return nil
	}
	var unknown []string
	if rv := reflect.ValueOf(v); rv.Kind() == reflect.Pointer && !rv.IsNil() {
		unknown = unknownFields(b.node, rv.Elem().Type(), 0)
	}
	err := b.node.Decode(v)
	if len(unknown) == 0 {
		return err
	}
	msgs := unknown
	var te *yaml.TypeError
	switch {
	case errors.As(err, &te):
		msgs = append(msgs, te.Errors...)
	case err != nil:
		return err
	}
	sort.SliceStable(msgs, func(i, j int) bool { return lineOf(msgs[i]) < lineOf(msgs[j]) })
	return &yaml.TypeError{Errors: msgs}
}

// Equal reports whether two blocks hold the same YAML: the same keys, values, tags and
// structure, whatever line, column or comment either was read with. Two absent blocks are
// equal.
func (b EngineBlock) Equal(o EngineBlock) bool {
	return sameNode(b.node, o.node)
}

func sameNode(a, b *yaml.Node) bool {
	if a == nil || b == nil {
		return a == b
	}
	if a.Kind != b.Kind || a.Value != b.Value || a.ShortTag() != b.ShortTag() || len(a.Content) != len(b.Content) {
		return false
	}
	if (a.Alias == nil) != (b.Alias == nil) || a.Alias != nil && !sameNode(a.Alias, b.Alias) {
		return false
	}
	for i := range a.Content {
		if !sameNode(a.Content[i], b.Content[i]) {
			return false
		}
	}
	return true
}

// UnmarshalYAML holds the node as written.
func (b *EngineBlock) UnmarshalYAML(n *yaml.Node) error {
	*b = NewEngineBlock(n)
	return nil
}

// MarshalYAML writes the node back as it was held. An absent block is written as an empty
// mapping, as a pool with no engine settings always was; a holder that omits an absent block
// tags it omitempty, which IsZero answers.
func (b EngineBlock) MarshalYAML() (any, error) {
	if b.node == nil {
		return &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}, nil
	}
	return cloneNode(b.node), nil
}

// cloneNode copies a node tree, so a holder of a block never shares nodes with its caller.
func cloneNode(n *yaml.Node) *yaml.Node {
	c := *n
	if n.Content != nil {
		c.Content = make([]*yaml.Node, len(n.Content))
		for i, child := range n.Content {
			c.Content[i] = cloneNode(child)
		}
	}
	if n.Alias != nil {
		c.Alias = cloneNode(n.Alias)
	}
	return &c
}

var unmarshalerType = reflect.TypeOf((*yaml.Unmarshaler)(nil)).Elem()

// unknownFields lists, in yaml.v3's KnownFields words, every mapping key under n that the
// struct it decodes into has no field for, descending through structs, maps, slices and
// pointers as the decoder does. A type that decodes itself (yaml.Unmarshaler) is its own to
// check.
func unknownFields(n *yaml.Node, t reflect.Type, depth int) []string {
	if depth > 64 {
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
		t = t.Elem()
	}
	if t.Implements(unmarshalerType) || reflect.PointerTo(t).Implements(unmarshalerType) {
		return nil
	}
	var out []string
	switch t.Kind() {
	case reflect.Struct:
		if n.Kind != yaml.MappingNode {
			return nil
		}
		fields, anyKey := yamlFields(t)
		for i := 0; i+1 < len(n.Content); i += 2 {
			k, val := n.Content[i], n.Content[i+1]
			if k.Kind == yaml.ScalarNode && k.Value == "<<" && (k.Tag == "!!merge" || k.Tag == "") {
				merged := []*yaml.Node{val}
				if val.Kind == yaml.SequenceNode {
					merged = val.Content
				}
				for _, m := range merged {
					out = append(out, unknownFields(m, t, depth+1)...)
				}
				continue
			}
			ft, ok := fields[k.Value]
			if !ok {
				if !anyKey {
					out = append(out, fmt.Sprintf("line %d: field %s not found in type %s", k.Line, k.Value, t))
				}
				continue
			}
			out = append(out, unknownFields(val, ft, depth+1)...)
		}
	case reflect.Map:
		if n.Kind == yaml.MappingNode {
			for i := 1; i < len(n.Content); i += 2 {
				out = append(out, unknownFields(n.Content[i], t.Elem(), depth+1)...)
			}
		}
	case reflect.Slice, reflect.Array:
		if n.Kind == yaml.SequenceNode {
			for _, c := range n.Content {
				out = append(out, unknownFields(c, t.Elem(), depth+1)...)
			}
		}
	}
	return out
}

// yamlFields maps the keys a struct decodes, as yaml.v3 names them (the tag's name, else the
// field name in lower case), to their field types, merging inline structs. anyKey is true
// when an inline map takes every other key.
func yamlFields(t reflect.Type) (fields map[string]reflect.Type, anyKey bool) {
	fields = map[string]reflect.Type{}
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if f.PkgPath != "" && !f.Anonymous {
			continue
		}
		tag := f.Tag.Get("yaml")
		if tag == "-" {
			continue
		}
		name, opts, _ := strings.Cut(tag, ",")
		if strings.Contains(","+opts+",", ",inline,") {
			ft := f.Type
			for ft.Kind() == reflect.Pointer {
				ft = ft.Elem()
			}
			switch ft.Kind() {
			case reflect.Struct:
				inner, any := yamlFields(ft)
				for k, v := range inner {
					fields[k] = v
				}
				anyKey = anyKey || any
			case reflect.Map:
				anyKey = true
			}
			continue
		}
		if name == "" {
			name = strings.ToLower(f.Name)
		}
		fields[name] = f.Type
	}
	return fields, anyKey
}

// lineOf is the line a decoder message names ("line N: ..."), or 0.
func lineOf(msg string) int {
	var n int
	if _, err := fmt.Sscanf(msg, "line %d:", &n); err != nil {
		return 0
	}
	return n
}

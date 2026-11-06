package validate

import (
	"fmt"
	"reflect"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// A custom UnmarshalYAML receives a yaml.Node, and Node.Decode does NOT honour the
// decoder's KnownFields setting. So every type that implements UnmarshalYAML loses
// strict field checking unless it re-establishes it, which is a silent loss: the type
// still decodes, and a misspelled field leaves a zero value nothing reports.
//
// RejectUnknownKeys restores the check. It compares a mapping node's keys against the
// yaml tags a target type declares, and is called by each custom unmarshaller before
// decoding.

// RejectUnknownKeys reports an error naming any key in node that target does not
// declare. Keys matching an ignore predicate are skipped, which is how the catalog's
// `_comment`-prefixed provenance narrative is allowed through.
func RejectUnknownKeys(node *yaml.Node, target any, ignore func(string) bool) error {
	if node.Kind != yaml.MappingNode {
		return nil
	}
	known := yamlFieldNames(reflect.TypeOf(target))
	var unknown []string
	for i := 0; i+1 < len(node.Content); i += 2 {
		key := node.Content[i].Value
		if ignore != nil && ignore(key) {
			continue
		}
		if !known[key] {
			unknown = append(unknown, key)
		}
	}
	if len(unknown) == 0 {
		return nil
	}
	sort.Strings(unknown)
	allowed := make([]string, 0, len(known))
	for k := range known {
		allowed = append(allowed, k)
	}
	sort.Strings(allowed)
	return fmt.Errorf("unknown field(s) %v; this type declares %v",
		unknown, allowed)
}

// IgnoreCatalogComments reports whether a key is one of the catalog's provenance
// narrative keys, which are deliberate content rather than stray fields. A key qualifies
// only if it is exactly "_comment" or a "_comment_"-prefixed suffix form (_comment_sm,
// _comment_interconnect) — the catalog's documented prose convention. The separator is
// required: a bare "_comment" prefix would also swallow "_commentary" or "_commentmfu",
// reopening the very loophole this check closes, so a key like "_commentmfu: 0.85" is a
// data field and fails the unknown-field check. Every other key — including any other
// underscore-prefixed one — is a data field too (a "_mfu" still fails), matching
// blis-catalog's deleted validate_catalog.py.
func IgnoreCatalogComments(key string) bool {
	return key == "_comment" || strings.HasPrefix(key, "_comment_")
}

// StripCatalogComments removes the catalog's `_comment`-prefixed provenance keys from a
// mapping node in place, before it is decoded, so no prose survives onto the struct. It is
// the companion to RejectUnknownKeys: that call lets the comment keys past the unknown-field
// check, this one drops them so Decode does not try to bind them to a field.
//
// It lives here rather than in spec/hardware because more than one catalog document type
// carries the convention — a chip, a fabric and a storage device do, and a workload shape
// does now too (#28) — and a nine-line node walk duplicated per package is the worse
// outcome. The predicate it strips on is IgnoreCatalogComments, so the set of keys accepted
// by RejectUnknownKeys and the set stripped here cannot drift apart.
func StripCatalogComments(node *yaml.Node) {
	if node.Kind != yaml.MappingNode {
		return
	}
	kept := make([]*yaml.Node, 0, len(node.Content))
	for i := 0; i+1 < len(node.Content); i += 2 {
		if IgnoreCatalogComments(node.Content[i].Value) {
			continue
		}
		kept = append(kept, node.Content[i], node.Content[i+1])
	}
	node.Content = kept
}

func yamlFieldNames(t reflect.Type) map[string]bool {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	out := map[string]bool{}
	if t.Kind() != reflect.Struct {
		return out
	}
	for i := 0; i < t.NumField(); i++ {
		tag := t.Field(i).Tag.Get("yaml")
		if tag == "-" {
			continue
		}
		name := strings.Split(tag, ",")[0]
		if name == "" {
			// An untagged field decodes under its lowercased name.
			name = strings.ToLower(t.Field(i).Name)
		}
		out[name] = true
	}
	return out
}

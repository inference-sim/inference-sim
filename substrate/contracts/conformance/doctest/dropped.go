package doctest

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Dropped returns the scalar values of valid that the loader drops: for each one, a copy of
// the document with that value changed loads to the same Go value as the original. SUB-12
// requires every declared field to be used or refused, never dropped; a field a custom
// unmarshaller forgets to copy, or a tag that names no key the document uses, shows up
// here. The catalog's _comment keys are prose for a reader and are skipped.
//
// decode loads a document and returns the loaded value.
func Dropped(valid []byte, decode func([]byte) (any, error)) ([]string, error) {
	base, err := decode(valid)
	if err != nil {
		return nil, fmt.Errorf("the valid document does not load: %w", err)
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(valid, &doc); err != nil {
		return nil, err
	}
	var paths [][]int
	var names []string
	walk(&doc, nil, "", &paths, &names)
	var dropped []string
	for i, p := range paths {
		var d yaml.Node
		if err := yaml.Unmarshal(valid, &d); err != nil {
			return nil, err
		}
		perturb(follow(&d, p))
		b, err := yaml.Marshal(&d)
		if err != nil {
			return nil, err
		}
		got, err := decode(b)
		if err != nil {
			continue // refused: the value is read
		}
		if reflect.DeepEqual(base, got) {
			dropped = append(dropped, names[i])
		}
	}
	return dropped, nil
}

func walk(n *yaml.Node, path []int, at string, paths *[][]int, names *[]string) {
	switch n.Kind {
	case yaml.DocumentNode:
		for i, c := range n.Content {
			walk(c, append(append([]int{}, path...), i), at, paths, names)
		}
	case yaml.MappingNode:
		for i := 0; i+1 < len(n.Content); i += 2 {
			k := n.Content[i].Value
			if k == "_comment" || strings.HasPrefix(k, "_comment_") {
				continue
			}
			walk(n.Content[i+1], append(append([]int{}, path...), i+1), join(at, k), paths, names)
		}
	case yaml.SequenceNode:
		for i, c := range n.Content {
			walk(c, append(append([]int{}, path...), i), fmt.Sprintf("%s[%d]", at, i), paths, names)
		}
	case yaml.ScalarNode:
		*paths = append(*paths, path)
		*names = append(*names, at)
	}
}

// perturb changes a scalar to another value of the same kind.
func perturb(n *yaml.Node) {
	switch n.ShortTag() {
	case "!!int":
		if v, err := strconv.ParseInt(n.Value, 0, 64); err == nil {
			n.Value = strconv.FormatInt(v+1, 10)
		}
	case "!!float":
		if v, err := strconv.ParseFloat(n.Value, 64); err == nil {
			n.Value = strconv.FormatFloat(v*1.5+1, 'g', -1, 64)
		}
	case "!!bool":
		if n.Value == "true" {
			n.Value = "false"
		} else {
			n.Value = "true"
		}
	case "!!null":
		n.Tag, n.Value = "!!str", "x"
	default:
		n.Value += "x"
	}
}

func join(at, key string) string {
	if at == "" {
		return key
	}
	return at + "." + key
}

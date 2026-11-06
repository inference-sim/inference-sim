package validate

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"gopkg.in/yaml.v3"
)

// JSONNode reads one JSON document into a YAML node tree for CheckNode, so a JSON document
// is held to the same rules as a YAML one. A number written with a fraction or an exponent
// is tagged !!float, any other number !!int; a repeated key is kept, so CheckNode reports
// it, where encoding/json would keep the last value silently. Line numbers are not known
// and are reported as 0.
func JSONNode(data []byte) (*yaml.Node, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	n, err := jsonValue(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("unexpected data after the JSON document")
	}
	return n, nil
}

func jsonValue(dec *json.Decoder) (*yaml.Node, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("decoding JSON at offset %d: %w", dec.InputOffset(), err)
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			n := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
			for dec.More() {
				k, err := dec.Token()
				if err != nil {
					return nil, err
				}
				v, err := jsonValue(dec)
				if err != nil {
					return nil, err
				}
				n.Content = append(n.Content, &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: k.(string)}, v)
			}
			_, err := dec.Token()
			return n, err
		case '[':
			n := &yaml.Node{Kind: yaml.SequenceNode, Tag: "!!seq"}
			for dec.More() {
				v, err := jsonValue(dec)
				if err != nil {
					return nil, err
				}
				n.Content = append(n.Content, v)
			}
			_, err := dec.Token()
			return n, err
		}
	case json.Number:
		tag := "!!int"
		if strings.ContainsAny(t.String(), ".eE") {
			tag = "!!float"
		}
		return &yaml.Node{Kind: yaml.ScalarNode, Tag: tag, Value: t.String()}, nil
	case string:
		return &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: t}, nil
	case bool:
		return &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!bool", Value: fmt.Sprint(t)}, nil
	case nil:
		return &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!null", Value: "null"}, nil
	}
	return nil, fmt.Errorf("unexpected JSON token %v", tok)
}

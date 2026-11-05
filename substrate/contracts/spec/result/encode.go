package result

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Format is an encoding of a Result document.
type Format string

const (
	// FormatJSON is the default encoding.
	FormatJSON Format = "json"
	// FormatYAML is the same document as YAML.
	FormatYAML Format = "yaml"
)

// Marshal writes r in the given format. Both encodings carry the same keys, the yaml tags
// of this package and of the documents a Result embeds, so a block those documents hold
// unopened (a pool's engine block) is written as it was read in either.
//
// The output is a deterministic function of r: struct fields in declaration order, map
// keys sorted, JSON indented by two spaces, one trailing newline. A non-finite number has
// no JSON form and is an error; Validate refuses one anyway. In a block held as an opaque
// map (Engine, Router, Kernel, Saturation, Input.Policy), a whole-number float such as
// 2.0 is written as 2 in either encoding, as yaml.v3 writes it, and reads back an integer.
func Marshal(r *Result, f Format) ([]byte, error) {
	switch f {
	case FormatYAML:
		var buf bytes.Buffer
		enc := yaml.NewEncoder(&buf)
		enc.SetIndent(2)
		if err := enc.Encode(r); err != nil {
			return nil, err
		}
		if err := enc.Close(); err != nil {
			return nil, err
		}
		return buf.Bytes(), nil
	case FormatJSON:
		var n yaml.Node
		if err := n.Encode(r); err != nil {
			return nil, err
		}
		var buf bytes.Buffer
		if err := writeJSON(&buf, &n, ""); err != nil {
			return nil, err
		}
		buf.WriteByte('\n')
		return buf.Bytes(), nil
	}
	return nil, fmt.Errorf("unknown result format %q: %s or %s", f, FormatJSON, FormatYAML)
}

// Unmarshal reads a Result from data, JSON or YAML, strictly: a key no field declares is
// an error naming it, in either encoding. A document whose first non-blank byte is `{` is
// read as JSON, by a JSON decoder, so every escape JSON allows reads as JSON defines it;
// anything else is read as YAML.
func Unmarshal(data []byte, r *Result) error {
	if t := bytes.TrimLeft(data, " \t\r\n"); len(t) > 0 && t[0] == '{' {
		n, err := jsonToNode(data)
		if err != nil {
			return err
		}
		// The JSON is re-expressed as YAML (every string quoted, every number tagged by its
		// JSON form) so that the one strict decoder, and the yaml tags it reads, decide both
		// encodings. Positions in an error refer to that rendering, not to the JSON text.
		if data, err = yaml.Marshal(n); err != nil {
			return err
		}
	}
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	return dec.Decode(r)
}

// writeJSON writes the YAML node n as JSON.
func writeJSON(w *bytes.Buffer, n *yaml.Node, indent string) error {
	switch n.Kind {
	case yaml.DocumentNode:
		if len(n.Content) != 1 {
			return errors.New("result: a document holds one value")
		}
		return writeJSON(w, n.Content[0], indent)
	case yaml.AliasNode:
		return writeJSON(w, n.Alias, indent)
	case yaml.MappingNode:
		if len(n.Content) == 0 {
			w.WriteString("{}")
			return nil
		}
		inner := indent + "  "
		w.WriteString("{\n")
		for i := 0; i+1 < len(n.Content); i += 2 {
			k := n.Content[i]
			if k.Kind != yaml.ScalarNode {
				return fmt.Errorf("result: a mapping key at line %d is not a scalar; JSON keys are strings", k.Line)
			}
			w.WriteString(inner)
			writeString(w, k.Value)
			w.WriteString(": ")
			if err := writeJSON(w, n.Content[i+1], inner); err != nil {
				return err
			}
			if i+2 < len(n.Content) {
				w.WriteByte(',')
			}
			w.WriteByte('\n')
		}
		w.WriteString(indent + "}")
		return nil
	case yaml.SequenceNode:
		if len(n.Content) == 0 {
			w.WriteString("[]")
			return nil
		}
		inner := indent + "  "
		w.WriteString("[\n")
		for i, c := range n.Content {
			w.WriteString(inner)
			if err := writeJSON(w, c, inner); err != nil {
				return err
			}
			if i+1 < len(n.Content) {
				w.WriteByte(',')
			}
			w.WriteByte('\n')
		}
		w.WriteString(indent + "]")
		return nil
	case yaml.ScalarNode:
		return writeScalar(w, n)
	}
	return fmt.Errorf("result: unexpected YAML node kind %d", n.Kind)
}

func writeScalar(w *bytes.Buffer, n *yaml.Node) error {
	switch n.ShortTag() {
	case "!!null":
		w.WriteString("null")
	case "!!bool":
		var b bool
		if err := n.Decode(&b); err != nil {
			return err
		}
		fmt.Fprint(w, b)
	case "!!int":
		var i int64
		if err := n.Decode(&i); err != nil {
			var u uint64
			if n.Decode(&u) != nil {
				return err
			}
			fmt.Fprint(w, u)
			return nil
		}
		fmt.Fprint(w, i)
	case "!!float":
		var v float64
		if err := n.Decode(&v); err != nil {
			return err
		}
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("result: %s has no JSON form", n.Value)
		}
		// The text as written when JSON can spell it; otherwise the shortest form that
		// reads back as the same value.
		text := n.Value
		if !json.Valid([]byte(text)) {
			text = strconv.FormatFloat(v, 'g', -1, 64)
		}
		w.WriteString(text)
	default:
		writeString(w, n.Value)
	}
	return nil
}

func writeString(w *bytes.Buffer, s string) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(s) // a string always encodes
	w.Write(bytes.TrimRight(buf.Bytes(), "\n"))
}

// jsonToNode reads one JSON value into a YAML node tree, keeping key order. Strings become
// quoted scalars, so no string reads back as another type; numbers keep their JSON text.
func jsonToNode(data []byte) (*yaml.Node, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	n, err := readJSONValue(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("result: unexpected data after the JSON document")
	}
	return n, nil
}

func readJSONValue(dec *json.Decoder) (*yaml.Node, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("decoding JSON at offset %d: %w", dec.InputOffset(), err)
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			m := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return nil, fmt.Errorf("decoding JSON at offset %d: %w", dec.InputOffset(), err)
				}
				k := kt.(string) // json.Decoder yields only strings as object keys
				v, err := readJSONValue(dec)
				if err != nil {
					return nil, err
				}
				m.Content = append(m.Content, str(k), v)
			}
			_, err := dec.Token() // '}'
			return m, err
		case '[':
			s := &yaml.Node{Kind: yaml.SequenceNode, Tag: "!!seq"}
			for dec.More() {
				v, err := readJSONValue(dec)
				if err != nil {
					return nil, err
				}
				s.Content = append(s.Content, v)
			}
			_, err := dec.Token() // ']'
			return s, err
		}
	case string:
		return str(t), nil
	case json.Number:
		tag := "!!int"
		if strings.ContainsAny(string(t), ".eE") {
			tag = "!!float"
		}
		return &yaml.Node{Kind: yaml.ScalarNode, Tag: tag, Value: string(t)}, nil
	case bool:
		return &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!bool", Value: fmt.Sprint(t)}, nil
	case nil:
		return &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!null", Value: "null"}, nil
	}
	return nil, fmt.Errorf("result: unexpected JSON token %v", tok)
}

func str(s string) *yaml.Node {
	return &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: s, Style: yaml.DoubleQuotedStyle}
}

package measurement

import (
	"bytes"

	"gopkg.in/yaml.v3"
)

// Marshal writes v in the canonical form: two-space indentation, fields in declaration
// order, map keys sorted. Two extractions of one input are byte-identical.
func Marshal(v any) ([]byte, error) {
	var b bytes.Buffer
	enc := yaml.NewEncoder(&b)
	enc.SetIndent(2)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	if err := enc.Close(); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

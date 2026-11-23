// Original to blis-sglang.

package blissglang

import (
	_ "embed"
	"fmt"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

//go:embed fields.json
var fieldsJSON []byte

// Fields is the field list of the release this build models (SGL-3): every configuration field and
// environment variable SGLang reads, with its status. engine.Engine.Fields returns it.
func Fields() (fields.FieldList, error) {
	l, err := fields.Parse(fieldsJSON)
	if err != nil {
		return fields.FieldList{}, fmt.Errorf("fields.json: %w", err)
	}
	return *l, nil
}

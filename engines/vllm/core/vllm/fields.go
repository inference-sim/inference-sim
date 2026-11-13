// Original to blis-vllm.

package vllm

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/spec/fields"

	blisvllm "github.com/inference-sim/blis-vllm"
)

// fieldList is fields.json, parsed once. The engine reports it through engine.Engine.Fields
// (design VLM-3); the file is the one list for this build, so the engine and the `fields` command
// cannot disagree.
var fieldList = mustFieldList()

func mustFieldList() fields.FieldList {
	l, err := fields.Parse(blisvllm.FieldListJSON)
	if err != nil {
		panic(fmt.Sprintf("vllm: fields.json does not parse: %v", err))
	}
	if l.Engine.Kind != blisvllm.Kind || l.Engine.Version != Version {
		panic(fmt.Sprintf("vllm: fields.json is for %s %s, and this engine models %s %s",
			l.Engine.Kind, l.Engine.Version, blisvllm.Kind, Version))
	}
	return *l
}

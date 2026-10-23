package llmd

import (
	"github.com/inference-sim/blis-schemas/embed"
	fork "github.com/llm-d/llm-d-router/pkg/embed"
)

// newRouter is the router's constructor: the fork's implementation of the blis-schemas embed
// contract, and the one identifier BLIS takes from the fork (imports_test.go).
var newRouter embed.Constructor = fork.New

package llmd

import (
	embed "github.com/llm-d/llm-d-router/blisembed"
	fork "github.com/llm-d/llm-d-router/pkg/embed"
)

// newRouter is the router's constructor: the fork's implementation of the blisembed
// contract, and the one identifier BLIS takes from the fork (imports_test.go).
var newRouter embed.Constructor = fork.New

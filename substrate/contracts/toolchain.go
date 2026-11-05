package contracts

import (
	_ "embed"
	"strings"
)

// toolchain is the one Go toolchain version every BLIS build uses (LIF-2), as the go
// command writes it, go1.N.M. Every CI job sets GOTOOLCHAIN=local and installs exactly
// this version, and no module BLIS writes declares a go version above it. The file is
// contracts/toolchain; it is the only place the version is written (X23).
//
//go:embed toolchain
var toolchain string

// Toolchain returns the Go toolchain version named in contracts/toolchain.
func Toolchain() string { return strings.TrimSpace(toolchain) }

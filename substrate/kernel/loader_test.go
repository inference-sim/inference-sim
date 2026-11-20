package latencykernel

import (
	"os"
	"path/filepath"
)

// The catalog and registry these tests read are this repository's own, beside this
// directory, so the cost model is always tested against the data it is released with.
// They are relative paths because a Go test runs with its package directory as the
// working directory. Overridable so a working copy can be tested against a candidate:
//
//	BLIS_REGISTRY=/path/to/candidate/registry go test ./kernel/...
var (
	catalogRoot  = envOr("BLIS_CATALOG", filepath.Join("..", "catalog"))
	registryRoot = envOr("BLIS_REGISTRY", filepath.Join("..", "registry"))
)

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

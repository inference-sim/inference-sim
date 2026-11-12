// Package substrate holds the catalog and the registry this release carries, compiled into
// the program with go:embed, and the file patterns that say which of their files are data.
//
// The package sits at the root of the repository because go:embed reaches only files at or
// below the package's own directory (design §3.3). The loader, package load, reads these
// files, or a directory with the same layout, with the same patterns.
package substrate

import (
	"embed"
	"io/fs"
)

// CatalogPatterns are the catalog's data files, relative to catalog/, in path.Match syntax.
// Every model's config.json stays in the repository for provenance and for the graph
// deriver, but no runtime code reads it, so it is not data (SUB-3).
//
// The ledger's published role lists (catalog/ledger/published/) join this list, and the
// embed line below, in the change that commits them, with the pattern
// "ledger/published/*.yaml". The directive refuses a pattern that matches no file, so the
// pattern cannot come before the files. TestPublishedListsAreEmbeddedWhenPresent fails a tree
// that has the lists without the pattern, and `ledger check -release` and the release tool
// refuse a release whose tree has no lists (D7, COV-2).
var CatalogPatterns = []string{
	"devices/*.yaml",
	"hardware/*.yaml",
	"measurements/sources.yaml",
	"models/*/graph.yaml",
	"models/*/model.yaml",
	"networks/*.yaml",
	"routers/*.yaml",
	"workloads/*.yaml",
}

// RegistryPatterns are the registry's data files, relative to registry/.
var RegistryPatterns = []string{
	"acceptance/*.yaml",
	"coefficients/*.yaml",
	"routing/*.yaml",
}

// The go:embed patterns must be literals, so they repeat the two lists above with their
// directory prefixed. TestEmbedPatternsEqualLoaderPatterns checks that they agree (SUB-9).
//
//go:embed catalog/devices/*.yaml catalog/hardware/*.yaml catalog/measurements/sources.yaml catalog/models/*/graph.yaml catalog/models/*/model.yaml catalog/networks/*.yaml catalog/routers/*.yaml catalog/workloads/*.yaml
//go:embed registry/acceptance/*.yaml registry/coefficients/*.yaml registry/routing/*.yaml
var embedded embed.FS

// Catalog returns the embedded catalog, rooted at catalog/.
func Catalog() fs.FS { return sub("catalog") }

// Registry returns the embedded registry, rooted at registry/.
func Registry() fs.FS { return sub("registry") }

func sub(dir string) fs.FS {
	f, err := fs.Sub(embedded, dir)
	if err != nil {
		// fs.Sub fails only for an invalid path, and dir is a constant.
		panic(err)
	}
	return f
}

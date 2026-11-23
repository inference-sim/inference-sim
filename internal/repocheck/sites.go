package repocheck

import (
	"fmt"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// EngineSites are the only files of sim/ and cmd/ whose non-test code may import an engine module
// (design X7, LIF-*): the one build site per engine. Every other file reaches an engine through
// sim.EngineDriver and contracts/engine, so adding an engine or changing a version touches one
// file. Test files are not counted: a test may link an engine to exercise the simulator against it.
var EngineSites = []string{"cmd/engine_build.go", "cmd/sglang_engine.go"}

// EngineImportSites returns the non-test Go files under sim/ and cmd/ of the module at root that
// import a package under module/engines/ and are not in sites, sorted. The files are read for their
// import declarations only.
func EngineImportSites(root, module string, sites []string) ([]Violation, error) {
	prefix := module + "/engines/"
	allowed := map[string]bool{}
	for _, s := range sites {
		allowed[s] = true
	}
	var out []Violation
	for _, top := range []string{"sim", "cmd"} {
		base := filepath.Join(root, top)
		err := filepath.WalkDir(base, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				if p == base {
					return nil // a tree without the directory has nothing to police
				}
				return err
			}
			if d.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
				return nil
			}
			rel, err := filepath.Rel(root, p)
			if err != nil {
				return err
			}
			rel = filepath.ToSlash(rel)
			f, err := parser.ParseFile(token.NewFileSet(), p, nil, parser.ImportsOnly)
			if err != nil {
				return fmt.Errorf("%s: %w", rel, err)
			}
			for _, im := range f.Imports {
				path, err := strconv.Unquote(im.Path.Value)
				if err != nil {
					return fmt.Errorf("%s: %w", rel, err)
				}
				if strings.HasPrefix(path, prefix) && !allowed[rel] {
					out = append(out, Violation{rel, fmt.Sprintf("imports %s; only %s may import an engine module",
						strings.TrimPrefix(path, module+"/"), strings.Join(sites, " and ")), "X7: one build site per engine"})
				}
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Where != out[j].Where {
			return out[i].Where < out[j].Where
		}
		return out[i].What < out[j].What
	})
	return out, nil
}

package repocheck

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const mod = "github.com/inference-sim/inference-sim"

func pkg(dir string, imports ...string) Package {
	p := Package{ImportPath: mod}
	if dir != "" {
		p.ImportPath = mod + "/" + dir
	}
	for _, i := range imports {
		if strings.Contains(i, ".") && !strings.HasPrefix(i, mod) || !strings.Contains(i, "/") {
			p.Imports = append(p.Imports, i)
		} else {
			p.Imports = append(p.Imports, mod+"/"+i)
		}
	}
	return p
}

func TestCheckImportsAcceptsTheAllowedEdges(t *testing.T) {
	pkgs := []Package{
		pkg("sim", "substrate/contracts/engine", "engines/vllm/core"),
		pkg("cmd", "substrate/kernel", "engines/sglang"),
		pkg("substrate/contracts/engine", "fmt"),
		pkg("substrate/catalog", "substrate/contracts/model"),
		pkg("substrate/registry", "substrate/contracts/model"),
		pkg("substrate/kernel", "substrate/contracts/model", "substrate/load"),
		pkg("substrate/load", "substrate/contracts/model"),
		pkg("substrate/validate", "substrate/contracts/model"),
		pkg("substrate", "substrate/contracts/model", "substrate/validate"),
		pkg("engines/vllm/core", "substrate/contracts/engine", "engines/vllm/internal/x"),
		pkg("engines/vllm/internal/x"),
		pkg("engines/sglang", "substrate/contracts/engine", "github.com/spf13/cobra"),
	}
	if got := CheckImports(mod, pkgs, Table1); len(got) != 0 {
		t.Fatalf("allowed edges reported: %v", got)
	}
}

func TestCheckImportsOnAnEmptyTreeIsClean(t *testing.T) {
	if got := CheckImports(mod, []Package{pkg("sim"), pkg("cmd", "sim")}, Table1); len(got) != 0 {
		t.Fatalf("a tree without substrate/ or engines/ reported: %v", got)
	}
}

func TestCheckImportsCatchesEachBrokenRule(t *testing.T) {
	for _, tc := range []struct {
		name string
		pkgs []Package
		rule string
		want string
	}{
		{"substrate imports an engine (X2)",
			[]Package{pkg("substrate/kernel", "engines/vllm/core")}, "X2", "substrate/kernel"},
		{"substrate root imports an engine (X2)",
			[]Package{pkg("substrate", "engines/sglang")}, "X2", "substrate"},
		{"contracts import a sibling component",
			[]Package{pkg("substrate/contracts/model", "substrate/kernel")}, "X7, Table 1", "substrate/contracts/model"},
		{"catalog imports the loader",
			[]Package{pkg("substrate/catalog", "substrate/load")}, "X7, Table 1", "substrate/catalog"},
		{"a vllm package imports sglang",
			[]Package{pkg("engines/vllm/core", "engines/sglang")}, "an engine module imports no other", "engines/vllm/core"},
		{"an sglang package imports vllm",
			[]Package{pkg("engines/sglang", "engines/vllm/core")}, "an engine module imports no other", "engines/sglang"},
		{"an engine imports a substrate component beyond contracts",
			[]Package{pkg("engines/vllm/core", "substrate/kernel")}, "X7, Table 1", "engines/vllm/core"},
		{"a directory no row owns",
			[]Package{pkg("substrate/newthing")}, "X7, Table 1", "substrate/newthing"},
		{"an engine directory no row owns",
			[]Package{pkg("engines/foo/x")}, "X7, Table 1", "engines/foo/x"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := CheckImports(mod, tc.pkgs, Table1)
			if len(got) != 1 {
				t.Fatalf("want one violation, got %v", got)
			}
			if got[0].Where != tc.want || !strings.Contains(got[0].Rule, tc.rule) {
				t.Fatalf("want %s under %q, got %v", tc.want, tc.rule, got[0])
			}
		})
	}
}

func TestCheckImportsCountsTestImports(t *testing.T) {
	p := pkg("substrate/kernel")
	p.XTestImports = []string{mod + "/engines/vllm/core"}
	if got := CheckImports(mod, []Package{p}, Table1); len(got) != 1 {
		t.Fatalf("an engine import from an external test must be reported, got %v", got)
	}
	q := pkg("substrate/kernel")
	q.TestImports = []string{mod + "/engines/vllm/core"}
	if got := CheckImports(mod, []Package{q}, Table1); len(got) != 1 {
		t.Fatalf("an engine import from an in-package test must be reported, got %v", got)
	}
}

func write(t *testing.T, root, rel, body string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestPathReferencesFixtures(t *testing.T) {
	t.Run("no substrate directory is clean", func(t *testing.T) {
		got, err := PathReferences(t.TempDir())
		if err != nil || len(got) != 0 {
			t.Fatalf("got %v, %v", got, err)
		}
	})
	t.Run("a clean substrate is clean", func(t *testing.T) {
		root := t.TempDir()
		write(t, root, "substrate/kernel/k.go", "package kernel\n// the engine contract lives in contracts/engine/\n")
		write(t, root, "substrate/registry/inputs/a.json", `{"engine":"vllm","source":"registry/inputs/x"}`)
		write(t, root, "engines/vllm/x.go", "package vllm\n// substrate/contracts\n")
		got, err := PathReferences(root)
		if err != nil || len(got) != 0 {
			t.Fatalf("got %v, %v", got, err)
		}
	})
	for _, tc := range []struct{ name, rel, body string }{
		{"a go string", "substrate/kernel/k.go", "package kernel\nvar p = \"engines/vllm/core\"\n"},
		{"a yaml value", "substrate/catalog/c.yaml", "path: ../engines/sglang/golden\n"},
		{"a markdown link", "substrate/README.md", "see [x](engines/vllm/README.md)\n"},
		{"a python open", "substrate/tools/t.py", "open('engines/sglang/data.json')\n"},
		{"a line start", "substrate/x.txt", "ok\nengines/vllm\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			write(t, root, tc.rel, tc.body)
			got, err := PathReferences(root)
			if err != nil || len(got) != 1 || !strings.HasPrefix(got[0].Where, tc.rel+":") || got[0].Rule != "X7" {
				t.Fatalf("got %v, %v", got, err)
			}
		})
	}
	t.Run("a binary file is not read as text", func(t *testing.T) {
		root := t.TempDir()
		write(t, root, "substrate/b.bin", "\x00engines/vllm/x")
		if got, _ := PathReferences(root); len(got) != 0 {
			t.Fatalf("got %v", got)
		}
	})
}

func TestNestedModulesFixture(t *testing.T) {
	root := t.TempDir()
	write(t, root, "go.mod", "module x\n")
	write(t, root, "a/b.go", "package a\n")
	if got, err := NestedModules(root); err != nil || len(got) != 0 {
		t.Fatalf("got %v, %v", got, err)
	}
	write(t, root, "engines/vllm/go.mod", "module y\n")
	got, err := NestedModules(root)
	if err != nil || len(got) != 1 || got[0] != "engines/vllm/go.mod" {
		t.Fatalf("got %v, %v", got, err)
	}
}

func TestModuleZipSizeSumsUncompressedBytes(t *testing.T) {
	root := t.TempDir()
	write(t, root, "go.mod", "module example.com/m\n\ngo 1.21\n")
	write(t, root, "a.go", "package m\n")
	write(t, root, "d/big.txt", strings.Repeat("x", 1<<20))
	got, err := ModuleZipSize(root, "example.com/m")
	if err != nil {
		t.Fatal(err)
	}
	want := int64(len("module example.com/m\n\ngo 1.21\n") + len("package m\n") + 1<<20)
	if got != want {
		t.Fatalf("size %d, want %d", got, want)
	}
}

func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func TestImportGraphFollowsTable1(t *testing.T) {
	module, pkgs, err := ListPackages(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	if module != mod {
		t.Fatalf("module %q, the fixtures and Table1 assume %q", module, mod)
	}
	for _, v := range CheckImports(module, pkgs, Table1) {
		t.Error(v)
	}
}

func TestNoPathUnderEnginesIsNamedBySubstrate(t *testing.T) {
	got, err := PathReferences(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range got {
		t.Error(v)
	}
}

func TestNoNestedModule(t *testing.T) {
	got, err := NestedModules(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	// Rows 2.5a and 2.6a import the history of blis-substrate and blis-vllm with their go.mod;
	// the combined import-rewrite PR deletes both together with this exemption.
	var nested []string
	for _, g := range got {
		if g != "substrate/go.mod" && g != "engines/vllm/go.mod" {
			nested = append(nested, g)
		}
	}
	if len(nested) != 0 {
		t.Fatalf("SUB-13: one Go module, no nested go.mod; found %v", nested)
	}
}

func TestModuleZipIsUnder100MB(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the module zip")
	}
	size, err := ModuleZipSize(repoRoot(t), mod)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("module zip uncompressed: %d bytes (limit %d)", size, ModuleZipLimit)
	if size >= ModuleZipLimit {
		t.Fatalf("SUB-13: module zip is %d bytes uncompressed, limit %d", size, ModuleZipLimit)
	}
}

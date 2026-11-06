package load

import (
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"testing/fstest"
)

// repoDir is this repository's root; a test runs in load/.
const repoDir = ".."

// SUB-9's check: the embedded copy and an exported checkout of the same commit load to the
// same data and the same hashes. The export is `git archive HEAD`, so the files compared
// are the committed ones; a test run on a tree with uncommitted data changes compares the
// embedded copy of the working tree with HEAD and fails, which is the right answer.
func TestEmbeddedAndExportedCheckoutAreTheSame(t *testing.T) {
	export := t.TempDir()
	archive := exec.Command("sh", "-c", `git -C "$1" archive HEAD catalog registry | tar -x -C "$2"`,
		"sh", repoDir, export)
	if out, err := archive.CombinedOutput(); err != nil {
		t.Fatalf("exporting HEAD: %v\n%s", err, out)
	}
	emb, err := Embedded()
	if err != nil {
		t.Fatalf("the embedded data: %v", err)
	}
	exp, err := Open(Options{Catalog: filepath.Join(export, "catalog"), Registry: filepath.Join(export, "registry")})
	if err != nil {
		t.Fatalf("the exported checkout: %v", err)
	}
	if emb.Catalog.Hash != exp.Catalog.Hash || emb.Registry.Hash != exp.Registry.Hash {
		t.Errorf("hashes differ: embedded catalog %s registry %s, exported catalog %s registry %s",
			emb.Catalog.Hash, emb.Registry.Hash, exp.Catalog.Hash, exp.Registry.Hash)
	}
	if !reflect.DeepEqual(emb.catalog, exp.catalog) {
		t.Error("the embedded catalog and the exported one parse to different data")
	}
	if !reflect.DeepEqual(emb.registry, exp.registry) {
		t.Error("the embedded registry and the exported one parse to different data")
	}
	if emb.Catalog.Source != SourceEmbedded || exp.Catalog.Source != SourcePath ||
		exp.Catalog.Path != filepath.Join(export, "catalog") {
		t.Errorf("sources: embedded %+v, exported %+v", emb.Catalog, exp.Catalog)
	}
}

// The content hash is the one design §3.3 writes down. The expected value was computed
// outside Go:
//
//	printf 'ca978112...48bb  a.yaml\n3e23e816...009d  z/b.yaml\n' | shasum -a 256
//
// where the two digests are sha256("a") and sha256("b"), listed by path in byte order.
func TestHashIsTheWrittenRule(t *testing.T) {
	got := Hash(map[string][]byte{"z/b.yaml": []byte("b"), "a.yaml": []byte("a")})
	const want = "4c155b89f48d131136ce8ecfe70957505234278cfbbac3d0c59fc64662fee2b7"
	if got != want {
		t.Errorf("Hash = %s, want %s", got, want)
	}
}

// copyTree copies this repository's catalog and registry into a temporary directory, so a
// test can break one file.
func copyTree(t *testing.T) (catalog, registry string) {
	t.Helper()
	dst := t.TempDir()
	for _, dir := range []string{"catalog", "registry"} {
		cmd := exec.Command("cp", "-R", filepath.Join(repoDir, dir), filepath.Join(dst, dir))
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("copying %s: %v\n%s", dir, err, out)
		}
	}
	return filepath.Join(dst, "catalog"), filepath.Join(dst, "registry")
}

func write(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// A path load of an unchanged copy succeeds; each thing that would make the data ambiguous
// is refused, naming the file (design §3.3).
func TestAPathIsRefusedForAmbiguousFiles(t *testing.T) {
	cat, reg := copyTree(t)
	if _, err := Open(Options{Catalog: cat, Registry: reg}); err != nil {
		t.Fatalf("an unchanged copy is refused: %v", err)
	}
	for _, c := range []struct {
		name   string
		break_ func(t *testing.T, cat, reg string)
		want   string
	}{
		{"a .yml where .yaml is expected", func(t *testing.T, cat, reg string) {
			write(t, filepath.Join(reg, "coefficients", "x.yml"), "kind: CoefficientSet\n")
		}, "coefficients/x.yml"},
		{"a file in a data directory that matches no pattern", func(t *testing.T, cat, reg string) {
			write(t, filepath.Join(cat, "hardware", "notes.txt"), "x\n")
		}, "hardware/notes.txt"},
		{"a symbolic link", func(t *testing.T, cat, reg string) {
			if err := os.Symlink("h100.yaml", filepath.Join(cat, "hardware", "h100-link.yaml")); err != nil {
				t.Fatal(err)
			}
		}, "hardware/h100-link.yaml is a symbolic link"},
		{"a carriage return", func(t *testing.T, cat, reg string) {
			p := filepath.Join(cat, "hardware", "h100.yaml")
			b, _ := os.ReadFile(p)
			write(t, p, strings.Replace(string(b), "\n", "\r\n", 1))
		}, "hardware/h100.yaml holds a carriage return at line 1"},
		{"a file that does not validate", func(t *testing.T, cat, reg string) {
			p := filepath.Join(cat, "hardware", "h100.yaml")
			b, _ := os.ReadFile(p)
			write(t, p, string(b)+"Frobnicate: 1\n")
		}, "hardware/h100.yaml"},
		{"a coefficient set whose name is not its file's", func(t *testing.T, cat, reg string) {
			p := filepath.Join(reg, "coefficients", "cost-model-fabric.yaml")
			b, _ := os.ReadFile(p)
			write(t, p, strings.Replace(string(b), "name: cost-model-fabric", "name: other", 1))
		}, "coefficients/cost-model-fabric.yaml"},
		{"a coefficient scoped to a chip the catalog does not have", func(t *testing.T, cat, reg string) {
			if err := os.Remove(filepath.Join(cat, "hardware", "l40s.yaml")); err != nil {
				t.Fatal(err)
			}
		}, `chip "l40s" is not in the catalog in use`},
		{"a model with a graph and no model.yaml", func(t *testing.T, cat, reg string) {
			if err := os.Remove(filepath.Join(cat, "models", "qwen3-14b", "model.yaml")); err != nil {
				t.Fatal(err)
			}
		}, "models/qwen3-14b/: has a graph.yaml but no model.yaml"},
	} {
		t.Run(c.name, func(t *testing.T) {
			cat, reg := copyTree(t)
			c.break_(t, cat, reg)
			_, err := Open(Options{Catalog: cat, Registry: reg})
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("want a refusal naming %q, got %v", c.want, err)
			}
		})
	}
}

// Two paths that differ only in letter case are refused. A case-insensitive file system
// cannot hold both, so the tree is in memory.
func TestTwoPathsDifferingInCaseAreRefused(t *testing.T) {
	fsys := fstest.MapFS{
		"hardware/h100.yaml": {Data: []byte("x\n")},
		"hardware/H100.yaml": {Data: []byte("x\n")},
	}
	_, err := readTree("t", fsys, catalogRules())
	if err == nil || !strings.Contains(err.Error(), "hardware/H100.yaml and hardware/h100.yaml differ only in letter case") {
		t.Fatalf("want a case refusal, got %v", err)
	}
}

// Files outside the data directories, each model's config.json, and the ledger outside
// published/ are not data: changing them changes neither the load nor the hash.
func TestFilesThatAreNotDataAreIgnored(t *testing.T) {
	cat, reg := copyTree(t)
	before, err := Open(Options{Catalog: cat, Registry: reg})
	if err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(cat, "README.md"), "notes\r\n")
	write(t, filepath.Join(cat, "scripts", "x.yml"), "x\n")
	write(t, filepath.Join(cat, "models", "qwen3-14b", "config.json"), "{\"changed\": 1}\r\n")
	write(t, filepath.Join(cat, "ledger", "units", "u.yaml"), "anything\r\n")
	write(t, filepath.Join(reg, "scripts", "y.yml"), "y\n")
	after, err := Open(Options{Catalog: cat, Registry: reg})
	if err != nil {
		t.Fatalf("a change outside the data was refused: %v", err)
	}
	if before.Catalog.Hash != after.Catalog.Hash || before.Registry.Hash != after.Registry.Hash {
		t.Error("a change outside the data changed a content hash")
	}
	// The ledger's published lists are data once they exist: a stray file there is refused.
	write(t, filepath.Join(cat, "ledger", "published", "fit.txt"), "x\n")
	if _, err := Open(Options{Catalog: cat, Registry: reg}); err == nil ||
		!strings.Contains(err.Error(), "ledger/published/fit.txt") {
		t.Errorf("a file under ledger/published/ that matches no pattern was not refused: %v", err)
	}
}

// SIM-6: a path replaces the embedded copy; it never overlays it. A registry path holding
// only one coefficient set is that set alone, and the catalog stays embedded.
func TestAPathReplacesTheEmbeddedCopy(t *testing.T) {
	_, reg := copyTree(t)
	for _, f := range []string{"cost-model-attention", "cost-model-collectives", "cost-model-collectives-inferencex",
		"cost-model-host-overheads", "cost-model-memory", "cost-model-primitives", "cost-model-recurrent"} {
		if err := os.Remove(filepath.Join(reg, "coefficients", f+".yaml")); err != nil {
			t.Fatal(err)
		}
	}
	d, err := Open(Options{Registry: reg})
	if err != nil {
		t.Fatal(err)
	}
	if d.Catalog.Source != SourceEmbedded || d.Registry.Source != SourcePath {
		t.Errorf("sources: catalog %s, registry %s", d.Catalog.Source, d.Registry.Source)
	}
	if _, err := d.CoefficientSet("cost-model-fabric"); err != nil {
		t.Error(err)
	}
	_, err = d.CoefficientSet("cost-model-memory")
	if err == nil || !strings.Contains(err.Error(), reg) || !strings.Contains(err.Error(), "coefficients/cost-model-memory.yaml") {
		t.Errorf("a set only the embedded copy has must be refused, naming the path in use and the file: %v", err)
	}
	if _, err := Open(Options{Catalog: filepath.Join(t.TempDir(), "nowhere")}); err == nil {
		t.Error("a catalog path that does not exist was accepted")
	}
}

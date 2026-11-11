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

// X12: no simulation path reads, requires or enforces a licence. A model.yaml with no licence
// record loads, as the base catalog (a BLIS_CATALOG-style root) and as a catalog overlay; a
// licence record that is present but invalid does not stop the load either. The licence rule
// is a CI check (validate-catalog); the project's own embedded catalog still passes it.
func TestTheLoaderDoesNotRequireALicence(t *testing.T) {
	const placeholder = "licence:\n  id: LicenseRef-other\n  evidence: http://example.invalid/README.md\n"
	strip := func(t *testing.T, path string, with string) {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		i := strings.Index(string(b), "licence:")
		if i < 0 {
			t.Fatalf("%s has no licence block to strip", path)
		}
		write(t, path, string(b[:i])+with)
	}
	for name, with := range map[string]string{"no licence": "", "invalid licence": placeholder} {
		t.Run("root, "+name, func(t *testing.T) {
			cat, _ := copyTree(t)
			strip(t, filepath.Join(cat, "models", "qwen3-14b", "model.yaml"), with)
			d, err := Open(Options{Catalog: cat})
			if err != nil {
				t.Fatalf("the loader refused a catalog over its licence record: %v", err)
			}
			if _, err := d.Model("qwen3-14b"); err != nil {
				t.Error(err)
			}
		})
		t.Run("overlay, "+name, func(t *testing.T) {
			ov := t.TempDir()
			for _, f := range []string{"graph.yaml", "model.yaml"} {
				b, err := os.ReadFile(filepath.Join(repoDir, "catalog", "models", "qwen3-14b", f))
				if err != nil {
					t.Fatal(err)
				}
				write(t, filepath.Join(ov, "models", "private-model", f), strings.ReplaceAll(string(b), "qwen3-14b", "private-model"))
			}
			strip(t, filepath.Join(ov, "models", "private-model", "model.yaml"), with)
			d, err := Open(Options{CatalogOverlays: []string{ov}})
			if err != nil {
				t.Fatalf("the loader refused an overlay over its licence record: %v", err)
			}
			if _, err := d.Model("private-model"); err != nil {
				t.Error(err)
			}
		})
	}
	t.Run("a licence registry with off-list terms", func(t *testing.T) {
		cat, _ := copyTree(t)
		p := filepath.Join(cat, "measurements", "sources.yaml")
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		// Make the first verified source redistributable under a licence not on the allow-list.
		out := strings.Replace(string(b), "license: Apache-2.0\n    status: verified", "license: LicenseRef-unlisted\n    status: verified", 1)
		if out == string(b) {
			t.Fatal("no verified Apache-2.0 source to edit")
		}
		write(t, p, out)
		if _, err := Open(Options{Catalog: cat}); err != nil {
			t.Fatalf("the loader judged a registry's licence terms: %v", err)
		}
	})
	t.Run("an unknown key is still refused", func(t *testing.T) {
		cat, _ := copyTree(t)
		p := filepath.Join(cat, "models", "qwen3-14b", "model.yaml")
		b, _ := os.ReadFile(p)
		write(t, p, string(b)+"licence_note: x\n")
		if _, err := Open(Options{Catalog: cat}); err == nil || !strings.Contains(err.Error(), "licence_note") {
			t.Errorf("strict decoding was lost: %v", err)
		}
	})
}

// X12 (CI side): every model of the project's own embedded catalog records a licence that
// passes the licence rule, so the requirement removed from the loader is still met by the
// data the project ships.
func TestTheEmbeddedCatalogRecordsEveryLicence(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	if len(d.catalog.Identities) == 0 {
		t.Fatal("the embedded catalog has no model identities")
	}
	for name, id := range d.catalog.Identities {
		if p := id.ValidateLicence(); !p.OK() {
			t.Errorf("models/%s/model.yaml: %s", name, p.Error())
		}
	}
}

// The embedded catalog carries the router table, and LookupRouter finds an entry by router,
// mode and session-affinity setting, and reports a router the table lacks.
func TestTheEmbeddedCatalogCarriesTheRouterTable(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	e, ok := d.LookupRouter("dynamo", "round-robin", nil)
	if !ok || e.Name != "dynamo-round-robin" || e.Policy != "none" {
		t.Fatalf("LookupRouter(dynamo, round-robin) = %+v, %v", e, ok)
	}
	if got, err := d.Router("dynamo-round-robin"); err != nil || got != e {
		t.Errorf("Router = %v, %v", got, err)
	}
	ttl := 300
	for name, c := range map[string]struct {
		router, mode string
		ttl          *int
	}{
		"another mode":        {"dynamo", "kv", nil},
		"another router":      {"vllm", "round-robin", nil},
		"an affinity setting": {"dynamo", "round-robin", &ttl},
	} {
		if e, ok := d.LookupRouter(c.router, c.mode, c.ttl); ok {
			t.Errorf("%s: found %s; the table has no such entry, so the router stays unsupported", name, e.Name)
		}
	}
	if _, err := d.Router("nope"); err == nil || !strings.Contains(err.Error(), "routers/nope.yaml") {
		t.Errorf("a missing entry: %v, want the path it belongs at", err)
	}
	if len(d.Routers()) == 0 {
		t.Error("no entries")
	}
}

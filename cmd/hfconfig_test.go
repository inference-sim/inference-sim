package cmd

import (
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// TestCatalogRootFrom_Precedence is AC-1 and AC-2 of #1731 (R1/S4) as a pure law:
// --catalog and BLIS_CATALOG both locate the catalog, --catalog wins when both are set,
// and NEITHER is refused naming both forms. catalogRootFrom takes both inputs as
// arguments, so the law is table-testable without touching globals or the environment.
func TestCatalogRootFrom_Precedence(t *testing.T) {
	tests := []struct {
		name    string
		flag    string
		env     string
		want    string
		wantErr bool
	}{
		{"flag only", "/from/flag", "", "/from/flag", false},
		{"env only", "", "/from/env", "/from/env", false},
		{"both set: flag wins", "/from/flag", "/from/env", "/from/flag", false},
		{"both set to the same path", "/same", "/same", "/same", false},
		{"neither set: refused", "", "", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := catalogRootFrom(tt.flag, tt.env)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected a refusal, got %q", got)
				}
				// AC-2: the refusal must name BOTH forms so the operator knows either works.
				if !strings.Contains(err.Error(), "--catalog") {
					t.Errorf("refusal must name --catalog, got: %v", err)
				}
				if !strings.Contains(err.Error(), catalogEnvVar) {
					t.Errorf("refusal must name %s, got: %v", catalogEnvVar, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("catalogRootFrom(%q, %q) = %q, want %q", tt.flag, tt.env, got, tt.want)
			}
		})
	}
}

// TestResolveCatalogRoot_ReadsFlagAndEnv is AC-1 through the production wrapper, which
// is the only place the --catalog package var and os.Getenv(BLIS_CATALOG) are read.
func TestResolveCatalogRoot_ReadsFlagAndEnv(t *testing.T) {
	origFlag := catalogPath
	t.Cleanup(func() { catalogPath = origFlag })

	flagDir := t.TempDir()
	envDir := t.TempDir()

	// Env only.
	catalogPath = ""
	t.Setenv(catalogEnvVar, envDir)
	got, err := resolveCatalogRoot()
	if err != nil {
		t.Fatalf("env-only: unexpected error: %v", err)
	}
	if got != envDir {
		t.Errorf("env-only: got %q, want %q", got, envDir)
	}

	// Flag wins over env.
	catalogPath = flagDir
	got, err = resolveCatalogRoot()
	if err != nil {
		t.Fatalf("flag+env: unexpected error: %v", err)
	}
	if got != flagDir {
		t.Errorf("flag+env: got %q, want %q (--catalog must win)", got, flagDir)
	}

	// Neither: refused. There is no working-directory default.
	catalogPath = ""
	t.Setenv(catalogEnvVar, "")
	if got, err = resolveCatalogRoot(); err == nil {
		t.Errorf("neither --catalog nor %s set must be refused, got %q", catalogEnvVar, got)
	}
}

// TestResolveCatalogRoot_RejectsUnusableRoot: a mistyped catalog path is refused as a
// CATALOG error naming both forms, not deferred into a per-model "not in the catalog"
// message that would blame the model (R1).
func TestResolveCatalogRoot_RejectsUnusableRoot(t *testing.T) {
	origFlag := catalogPath
	t.Cleanup(func() { catalogPath = origFlag })
	t.Setenv(catalogEnvVar, "")

	fileNotDir := filepath.Join(t.TempDir(), "catalog-is-a-file")
	if err := os.WriteFile(fileNotDir, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, tt := range []struct{ name, root string }{
		{"missing directory", filepath.Join(t.TempDir(), "no-such-catalog")},
		{"not a directory", fileNotDir},
	} {
		t.Run(tt.name, func(t *testing.T) {
			catalogPath = tt.root
			got, err := resolveCatalogRoot()
			if err == nil {
				t.Fatalf("expected refusal for %s, got %q", tt.name, got)
			}
			if !strings.Contains(err.Error(), tt.root) {
				t.Errorf("refusal must name the offending root %q, got: %v", tt.root, err)
			}
			if !strings.Contains(err.Error(), "--catalog") || !strings.Contains(err.Error(), catalogEnvVar) {
				t.Errorf("refusal must name both --catalog and %s, got: %v", catalogEnvVar, err)
			}
		})
	}
}

// TestResolveModelConfig_CatalogHit is the happy path: a catalogued model resolves to
// its catalog directory.
func TestResolveModelConfig_CatalogHit(t *testing.T) {
	tmpDir := t.TempDir()
	localDir := filepath.Join(tmpDir, "test-model")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(localDir, hfConfigFile), []byte(`{"num_hidden_layers": 32, "hidden_size": 4096}`), 0o644); err != nil {
		t.Fatal(err)
	}

	dir, err := resolveModelConfigInCatalog("test-org/test-model", tmpDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := filepath.Join(tmpDir, "test-model")
	if dir != expected {
		t.Errorf("expected %s, got %s", expected, dir)
	}
}

// writeCatalogEntry writes content as the config.json of the catalog entry for model
// short name under dir, creating dir as needed, and returns the config.json path.
func writeCatalogEntry(t *testing.T, dir, content string) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, hfConfigFile)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// minimalHFConfig is the smallest config.json isHFConfig accepts.
const minimalHFConfig = `{"num_hidden_layers": 32, "hidden_size": 4096}`

// TestResolveModelConfig_CloneRootLayout is the #1774 contract: --catalog names the
// catalog CLONE ROOT, so an entry stored the way the authoritative blis-catalog
// repository stores it — <catalog>/models/<short-name>/config.json — resolves. Before
// #1774 this failed and the operator had to pass <clone-root>/models instead.
func TestResolveModelConfig_CloneRootLayout(t *testing.T) {
	root := t.TempDir()
	entryDir := filepath.Join(root, "models", "test-model")
	writeCatalogEntry(t, entryDir, minimalHFConfig)

	dir, err := resolveModelConfigInCatalog("test-org/test-model", root)
	if err != nil {
		t.Fatalf("clone-root layout must resolve: %v", err)
	}
	if dir != entryDir {
		t.Errorf("resolved %q, want the models/ entry %q", dir, entryDir)
	}
}

// TestResolveModelConfig_FlatLayoutFallback is the transition half of #1774 (INV-6): the
// flat <catalog>/<short-name> layout — the bundled model_configs/ tree and every in-repo
// test catalog — keeps resolving unchanged until #1771 removes both the tree and this
// fallback. Without it, `export BLIS_CATALOG=$PWD/model_configs` would break instantly.
func TestResolveModelConfig_FlatLayoutFallback(t *testing.T) {
	root := t.TempDir()
	entryDir := filepath.Join(root, "test-model")
	writeCatalogEntry(t, entryDir, minimalHFConfig)

	dir, err := resolveModelConfigInCatalog("test-org/test-model", root)
	if err != nil {
		t.Fatalf("flat layout must keep resolving during the transition: %v", err)
	}
	if dir != entryDir {
		t.Errorf("resolved %q, want the flat entry %q", dir, entryDir)
	}
}

// TestResolveModelConfig_CloneRootLayoutWinsOverFlat pins the resolution ORDER where it
// is observable: with the SAME model present in both layouts, the canonical clone-root
// entry is the one used. The two configs differ in an architectural field, so the
// assertion is on which entry directory (and therefore which numbers) resolution picked —
// not merely on "some entry resolved".
func TestResolveModelConfig_CloneRootLayoutWinsOverFlat(t *testing.T) {
	root := t.TempDir()
	nested := filepath.Join(root, "models", "test-model")
	flat := filepath.Join(root, "test-model")
	writeCatalogEntry(t, nested, `{"num_hidden_layers": 32, "hidden_size": 4096}`)
	writeCatalogEntry(t, flat, `{"num_hidden_layers": 8, "hidden_size": 512}`)

	dir, err := resolveModelConfigInCatalog("test-org/test-model", root)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dir != nested {
		t.Errorf("resolved %q, want the canonical clone-root entry %q — a stale flat entry "+
			"must not shadow a catalogued one", dir, nested)
	}
}

// TestResolveModelConfig_MalformedCloneRootEntry_NotBypassedByFlatFallback is the
// fallback's boundary (R1, NS-6): only ABSENCE advances to the flat layout. A clone-root
// entry that exists but is not a HuggingFace config.json is REPORTED naming that file —
// resolution must not quietly substitute a different config for a broken one, which would
// make a typo in the catalogued entry look like a successful (but wrong-model) run.
func TestResolveModelConfig_MalformedCloneRootEntry_NotBypassedByFlatFallback(t *testing.T) {
	root := t.TempDir()
	nested := filepath.Join(root, "models", "test-model")
	flat := filepath.Join(root, "test-model")
	nestedPath := writeCatalogEntry(t, nested, `{"error": "not found"}`)
	writeCatalogEntry(t, flat, minimalHFConfig)

	dir, err := resolveModelConfigInCatalog("test-org/test-model", root)
	if err == nil {
		t.Fatalf("a malformed clone-root entry must be refused, not bypassed; got dir=%q", dir)
	}
	if !strings.Contains(err.Error(), nestedPath) {
		t.Errorf("refusal must name the offending clone-root entry (%s), got: %v", nestedPath, err)
	}
}

// TestResolveModelConfig_UnreadableCloneRootEntry_NotBypassedByFlatFallback is the same
// boundary as the malformed case for a config.json that is PRESENT but cannot be opened
// (mode 000): resolution reports it naming that path instead of quietly using the flat
// entry. Presence is what distinguishes the two dispositions — not the errno.
func TestResolveModelConfig_UnreadableCloneRootEntry_NotBypassedByFlatFallback(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root: mode 000 does not make a file unreadable")
	}
	root := t.TempDir()
	nested := filepath.Join(root, "models", "test-model")
	nestedPath := writeCatalogEntry(t, nested, minimalHFConfig)
	if err := os.Chmod(nestedPath, 0o000); err != nil {
		t.Fatal(err)
	}
	writeCatalogEntry(t, filepath.Join(root, "test-model"), minimalHFConfig)

	dir, err := resolveModelConfigInCatalog("test-org/test-model", root)
	if err == nil {
		t.Fatalf("an unreadable clone-root entry must be refused, not bypassed; got dir=%q", dir)
	}
	if !strings.Contains(err.Error(), nestedPath) {
		t.Errorf("refusal must name the unreadable entry (%s), got: %v", nestedPath, err)
	}
}

// TestResolveModelConfig_FlatCatalogHoldingAFileNamedModels keeps the models/ candidate
// from breaking a working flat catalog for an unrelated reason: a flat root that happens
// to contain a regular FILE named "models" makes the canonical candidate path traverse a
// non-directory. That means "no entry in this layout", not "broken entry", so resolution
// must fall through to the flat entry rather than hard-error on the errno.
func TestResolveModelConfig_FlatCatalogHoldingAFileNamedModels(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, catalogModelsSubdir), []byte("not a directory"), 0o644); err != nil {
		t.Fatal(err)
	}
	entryDir := filepath.Join(root, "test-model")
	writeCatalogEntry(t, entryDir, minimalHFConfig)

	dir, err := resolveModelConfigInCatalog("test-org/test-model", root)
	if err != nil {
		t.Fatalf("a flat catalog must keep resolving even with an unrelated file named %q: %v",
			catalogModelsSubdir, err)
	}
	if dir != entryDir {
		t.Errorf("resolved %q, want the flat entry %q", dir, entryDir)
	}
}

// TestResolveModelConfig_AbsentFromBothLayouts_NamesEveryPathLookedAt: when neither
// layout holds the model, the refusal must name every path resolution looked at (so the
// operator can see BOTH were tried) and the canonical path an entry belongs at.
func TestResolveModelConfig_AbsentFromBothLayouts_NamesEveryPathLookedAt(t *testing.T) {
	root := t.TempDir()

	dir, err := resolveModelConfigInCatalog("test-org/absent-model", root)
	if err == nil {
		t.Fatalf("expected refusal for a model absent from every layout, got dir=%q", dir)
	}
	canonical := filepath.Join(root, "models", "absent-model", hfConfigFile)
	flat := filepath.Join(root, "absent-model", hfConfigFile)
	for _, want := range []string{canonical, flat} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal must name the path it looked at (%s), got: %v", want, err)
		}
	}
}

// TestResolveModelConfig_RelativeCatalogIsCWDRelative pins the path semantics #1774 asks
// to state precisely: a RELATIVE --catalog / BLIS_CATALOG value is resolved against the
// process working directory. The same relative value resolves in a directory that has
// the catalog and is refused in one that does not.
func TestResolveModelConfig_RelativeCatalogIsCWDRelative(t *testing.T) {
	withCatalog := t.TempDir()
	writeCatalogEntry(t, filepath.Join(withCatalog, "cat", "models", "test-model"), minimalHFConfig)
	without := t.TempDir()

	t.Chdir(withCatalog)
	dir, err := resolveModelConfigInCatalog("test-org/test-model", "cat")
	if err != nil {
		t.Fatalf("a relative catalog must resolve against the working directory: %v", err)
	}
	if want := filepath.Join("cat", "models", "test-model"); dir != want {
		t.Errorf("resolved %q, want the relative path %q left as given", dir, want)
	}

	t.Chdir(without)
	if dir, err := resolveModelConfigInCatalog("test-org/test-model", "cat"); err == nil {
		t.Errorf("the same relative catalog must NOT resolve from a directory without it; got %q", dir)
	}
}

// TestResolveModelConfig_AbsoluteCatalogIsCWDIndependent is the other half: an ABSOLUTE
// value is used as given, so the same catalog resolves from any working directory.
func TestResolveModelConfig_AbsoluteCatalogIsCWDIndependent(t *testing.T) {
	root := t.TempDir()
	entryDir := filepath.Join(root, "models", "test-model")
	writeCatalogEntry(t, entryDir, minimalHFConfig)

	for _, cwd := range []string{root, t.TempDir()} {
		t.Chdir(cwd)
		dir, err := resolveModelConfigInCatalog("test-org/test-model", root)
		if err != nil {
			t.Fatalf("cwd %s: an absolute catalog must resolve regardless of cwd: %v", cwd, err)
		}
		if dir != entryDir {
			t.Errorf("cwd %s: resolved %q, want %q", cwd, dir, entryDir)
		}
	}
}

// TestResolveModelConfig_AbsentFromCatalog_RefusedNamingPath is BC-1 (NS-6, #1733): a
// model with no catalog entry is REFUSED, and the refusal names the path the entry
// belongs at so an operator can act on it. Before #1733 this case silently downloaded
// config.json from HuggingFace and wrote it into the catalog.
func TestResolveModelConfig_AbsentFromCatalog_RefusedNamingPath(t *testing.T) {
	tmpDir := t.TempDir()
	dir, err := resolveModelConfigInCatalog("test-org/uncatalogued-model", tmpDir)
	if err == nil {
		t.Fatalf("expected an uncatalogued model to be refused, got dir=%q", dir)
	}

	wantPath := filepath.Join(tmpDir, "uncatalogued-model", hfConfigFile)
	if !strings.Contains(err.Error(), wantPath) {
		t.Errorf("refusal must name the catalog path an entry belongs at (%s), got: %v", wantPath, err)
	}
	if !strings.Contains(err.Error(), "test-org/uncatalogued-model") {
		t.Errorf("refusal must name the model, got: %v", err)
	}
}

// TestResolveModelConfig_AbsentFromCatalog_CreatesNothing is the other half of BC-1 and
// the core of BC-3 at the resolution boundary: refusing an uncatalogued model must not
// create a directory or a file anywhere under the catalog root. This is the observable
// form of "no run adds a catalog entry as a side effect".
func TestResolveModelConfig_AbsentFromCatalog_CreatesNothing(t *testing.T) {
	tmpDir := t.TempDir()
	before := listTree(t, tmpDir)

	if _, err := resolveModelConfigInCatalog("test-org/uncatalogued-model", tmpDir); err == nil {
		t.Fatal("expected refusal for an uncatalogued model")
	}

	after := listTree(t, tmpDir)
	if len(after) != len(before) {
		t.Errorf("resolution must not create anything under the catalog root; before=%v after=%v", before, after)
	}
	entryDir := filepath.Join(tmpDir, "uncatalogued-model")
	if _, statErr := os.Stat(entryDir); statErr == nil {
		t.Errorf("resolution must not create the catalog entry directory %s", entryDir)
	}
}

// listTree returns every path under root, sorted — used to assert nothing was written.
func listTree(t *testing.T, root string) []string {
	t.Helper()
	var paths []string
	err := filepath.WalkDir(root, func(path string, _ fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		paths = append(paths, path)
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", root, err)
	}
	sort.Strings(paths)
	return paths
}

// TestResolveModelConfig_MalformedCatalogEntry_RefusedAndPreserved is BC-2: an existing
// catalog entry that is not a HuggingFace config.json is refused naming the file, and the
// file is left byte-for-byte unchanged (never overwritten by a fetch, never deleted — it
// may be an operator's hand-written entry with a fixable typo).
func TestResolveModelConfig_MalformedCatalogEntry_RefusedAndPreserved(t *testing.T) {
	tests := []struct {
		name    string
		content string
	}{
		{"not JSON at all", `<html>not json</html>`},
		{"valid JSON but not an HF config", `{"error": "not found"}`},
		{"empty JSON object", `{}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			localDir := filepath.Join(tmpDir, "test-model")
			if err := os.MkdirAll(localDir, 0o755); err != nil {
				t.Fatal(err)
			}
			entryPath := filepath.Join(localDir, hfConfigFile)
			if err := os.WriteFile(entryPath, []byte(tt.content), 0o644); err != nil {
				t.Fatal(err)
			}

			if _, err := resolveModelConfigInCatalog("test-org/test-model", tmpDir); err == nil {
				t.Fatal("expected refusal for a malformed catalog entry")
			} else if !strings.Contains(err.Error(), entryPath) {
				t.Errorf("refusal must name the offending catalog file (%s), got: %v", entryPath, err)
			}

			// The entry must survive untouched.
			got, readErr := os.ReadFile(entryPath)
			if readErr != nil {
				t.Fatalf("catalog entry must be preserved, not deleted: %v", readErr)
			}
			if string(got) != tt.content {
				t.Errorf("catalog entry must be preserved byte-for-byte; got %q want %q", string(got), tt.content)
			}
		})
	}
}

func TestResolveModelConfig_MultimodalConfig(t *testing.T) {
	tmpDir := t.TempDir()
	localDir := filepath.Join(tmpDir, "llama4-test")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Write a multimodal config (text_config structure)
	multimodalConfig := `{
		"architectures": ["Llama4ForConditionalGeneration"],
		"model_type": "llama4",
		"text_config": {
			"num_hidden_layers": 48,
			"hidden_size": 5120,
			"num_attention_heads": 40,
			"num_key_value_heads": 8
		},
		"vision_config": {
			"num_hidden_layers": 34,
			"hidden_size": 1408
		}
	}`
	if err := os.WriteFile(filepath.Join(localDir, hfConfigFile), []byte(multimodalConfig), 0o644); err != nil {
		t.Fatal(err)
	}

	dir, err := resolveModelConfigInCatalog("test-org/llama4-test", tmpDir)
	if err != nil {
		t.Fatalf("multimodal config should be recognized: %v", err)
	}
	expected := filepath.Join(tmpDir, "llama4-test")
	if dir != expected {
		t.Errorf("expected %s, got %s", expected, dir)
	}
}

func TestResolveHardwareConfig_ExplicitOverride(t *testing.T) {
	path, err := resolveHardwareConfig("/explicit/hw.json", "defaults.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if path != "/explicit/hw.json" {
		t.Errorf("expected /explicit/hw.json, got %s", path)
	}
}

func TestResolveHardwareConfig_BundledDefault(t *testing.T) {
	tmpDir := t.TempDir()
	hwPath := filepath.Join(tmpDir, "hardware_config.json")
	if err := os.WriteFile(hwPath, []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}

	defaultsFile := filepath.Join(tmpDir, "defaults.yaml")
	path, err := resolveHardwareConfig("", defaultsFile)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if path != hwPath {
		t.Errorf("expected %s, got %s", hwPath, path)
	}
}

func TestResolveHardwareConfig_Missing_ReturnsError(t *testing.T) {
	_, err := resolveHardwareConfig("", "/nonexistent/dir/defaults.yaml")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

// TestCatalogModelDir maps a model name onto its candidate catalog entry directories.
// Since #1731 the catalog ROOT is a required input (there is no working-directory
// default), so an empty root is an error rather than a relative path. Since #1774 the
// root is the catalog CLONE ROOT: the canonical candidate is <catalog>/models/<name>,
// and the flat <catalog>/<name> is the transition fallback, IN THAT ORDER.
//
// Path semantics are pinned here too: a relative root yields relative candidates (the OS
// resolves them against the process working directory at read time) and an absolute root
// yields absolute candidates — neither is rewritten.
func TestCatalogModelDir(t *testing.T) {
	tests := []struct {
		model    string
		catalog  string
		expected []string
		wantErr  bool
	}{
		{"meta-llama/llama-3.1-8b-instruct", "/base", []string{
			filepath.Join("/base", "models", "llama-3.1-8b-instruct"),
			filepath.Join("/base", "llama-3.1-8b-instruct"),
		}, false},
		{"codellama/codellama-34b-instruct-hf", "/base", []string{
			filepath.Join("/base", "models", "codellama-34b-instruct-hf"),
			filepath.Join("/base", "codellama-34b-instruct-hf"),
		}, false},
		{"simple-model", "/base", []string{
			filepath.Join("/base", "models", "simple-model"),
			filepath.Join("/base", "simple-model"),
		}, false},
		// A relative root stays relative (CWD-resolved by the OS), not absolutized.
		{"meta-llama/llama-3.1-8b-instruct", "relative/catalog", []string{
			filepath.Join("relative/catalog", "models", "llama-3.1-8b-instruct"),
			filepath.Join("relative/catalog", "llama-3.1-8b-instruct"),
		}, false},
		// The pre-#1774 invocation (--catalog pointed at the models directory) still has a
		// candidate that matches, via the fallback.
		{"meta-llama/llama-3.1-8b-instruct", "/clone/models", []string{
			filepath.Join("/clone", "models", "models", "llama-3.1-8b-instruct"),
			filepath.Join("/clone", "models", "llama-3.1-8b-instruct"),
		}, false},
		// An empty root must NOT silently resolve to a working-directory-relative path:
		// that is the retired default #1731 removed.
		{"meta-llama/llama-3.1-8b-instruct", "", nil, true},
		{"evil/../../../etc/passwd", "/base", nil, true},
		{"org/../../etc/shadow", "/base", nil, true},
	}

	for _, tt := range tests {
		got, err := catalogModelDirs(tt.model, tt.catalog)
		if tt.wantErr {
			if err == nil {
				t.Errorf("catalogModelDirs(%q, %q) expected error, got nil", tt.model, tt.catalog)
			}
			continue
		}
		if err != nil {
			t.Errorf("catalogModelDirs(%q, %q) unexpected error: %v", tt.model, tt.catalog, err)
			continue
		}
		if !reflect.DeepEqual(got, tt.expected) {
			t.Errorf("catalogModelDirs(%q, %q) = %q, want %q", tt.model, tt.catalog, got, tt.expected)
		}
	}
}

// TestCatalogModelDirs_CanonicalLayoutIsFirst is the ORDER law on its own, stated
// independently of the exact strings above (#1774): whatever the root, the canonical
// clone-root candidate precedes the flat transition fallback. Reversing them would make
// a stale flat entry shadow a fresh catalogued one — the failure this pins.
func TestCatalogModelDirs_CanonicalLayoutIsFirst(t *testing.T) {
	for _, catalog := range []string{"/abs/catalog", "rel/catalog", "."} {
		got, err := catalogModelDirs("org/some-model", catalog)
		if err != nil {
			t.Fatalf("catalogModelDirs(_, %q): %v", catalog, err)
		}
		if len(got) != 2 {
			t.Fatalf("catalogModelDirs(_, %q) = %v, want exactly 2 candidates", catalog, got)
		}
		wantPrimary := filepath.Join(catalog, catalogModelsSubdir, "some-model")
		wantFallback := filepath.Join(catalog, "some-model")
		if got[0] != wantPrimary {
			t.Errorf("catalog %q: first candidate = %q, want the clone-root layout %q", catalog, got[0], wantPrimary)
		}
		if got[1] != wantFallback {
			t.Errorf("catalog %q: second candidate = %q, want the flat fallback %q", catalog, got[1], wantFallback)
		}
	}
}

func TestGetHFRepo_ValidModel(t *testing.T) {
	// Create a minimal defaults.yaml with hf_repo
	tmpDir := t.TempDir()
	defaultsPath := filepath.Join(tmpDir, "defaults.yaml")
	content := `defaults:
  test-org/test-model:
    GPU: H100
    tensor_parallelism: 2
    hf_repo: TestOrg/Test-Model
workloads: {}
version: "0.0.1"
`
	if err := os.WriteFile(defaultsPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	repo, err := GetHFRepo("test-org/test-model", defaultsPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if repo != "TestOrg/Test-Model" {
		t.Errorf("expected TestOrg/Test-Model, got %q", repo)
	}
}

func TestGetHFRepo_ModelWithoutHFRepo(t *testing.T) {
	tmpDir := t.TempDir()
	defaultsPath := filepath.Join(tmpDir, "defaults.yaml")
	content := `defaults:
  test-org/test-model:
    GPU: H100
    tensor_parallelism: 2
workloads: {}
version: "0.0.1"
`
	if err := os.WriteFile(defaultsPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	repo, err := GetHFRepo("test-org/test-model", defaultsPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if repo != "" {
		t.Errorf("expected empty string for model without hf_repo, got %q", repo)
	}
}

func TestGetHFRepo_ModelNotFound(t *testing.T) {
	tmpDir := t.TempDir()
	defaultsPath := filepath.Join(tmpDir, "defaults.yaml")
	content := `defaults:
  other-model:
    GPU: H100
    tensor_parallelism: 2
workloads: {}
version: "0.0.1"
`
	if err := os.WriteFile(defaultsPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	repo, err := GetHFRepo("nonexistent/model", defaultsPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if repo != "" {
		t.Errorf("expected empty string for nonexistent model, got %q", repo)
	}
}

func TestGetHFRepo_NonexistentFile(t *testing.T) {
	_, err := GetHFRepo("any-model", "/nonexistent/defaults.yaml")
	if err == nil {
		t.Fatal("expected error for nonexistent file, got nil")
	}
}

func TestGetHFRepo_MalformedYAML(t *testing.T) {
	tmpDir := t.TempDir()
	defaultsPath := filepath.Join(tmpDir, "defaults.yaml")
	if err := os.WriteFile(defaultsPath, []byte(`{invalid yaml: [`), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := GetHFRepo("any-model", defaultsPath)
	if err == nil {
		t.Fatal("expected error for malformed YAML, got nil")
	}
}

// TestResolveModelConfig_ResolutionInvariant verifies the documented resolution order,
// which after #1733 (NS-6), #1731 (S4) and #1774 stays confined to ONE catalog root: the
// entry is read from <catalog>/models/<short-name>/config.json or, during the #1771
// transition, the flat <catalog>/<short-name>/config.json. Those are two LAYOUTS of the
// same root, not two sources — removing the entry from a root does not open a path
// outside it (no cross-catalog, working-directory or network fallback), it makes
// resolution fail. Two DIFFERENT catalogs holding the same model resolve independently,
// which is what makes "point --catalog at a scratch clone" the replacement for the
// retired --model-config-folder.
func TestResolveModelConfig_ResolutionInvariant(t *testing.T) {
	tmpDir := t.TempDir()
	const cfg = `{"num_hidden_layers": 32, "hidden_size": 4096}`

	catalogA := filepath.Join(tmpDir, "catalog-a")
	catalogB := filepath.Join(tmpDir, "catalog-b")
	for _, root := range []string{catalogA, catalogB} {
		entry := filepath.Join(root, "precedence-model")
		if err := os.MkdirAll(entry, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(entry, hfConfigFile), []byte(cfg), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// The resolved entry follows the catalog root it was asked for — nothing else.
	for _, root := range []string{catalogA, catalogB} {
		dir, err := resolveModelConfigInCatalog("test-org/precedence-model", root)
		if err != nil {
			t.Fatalf("catalog %s: resolution failed: %v", root, err)
		}
		if want := filepath.Join(root, "precedence-model"); dir != want {
			t.Errorf("catalog %s: expected %s, got %s", root, want, dir)
		}
	}

	// There is no second step: with catalog A's entry removed, A is refused while B
	// still resolves (no cross-catalog or working-directory fallback).
	if err := os.Remove(filepath.Join(catalogA, "precedence-model", hfConfigFile)); err != nil {
		t.Fatal(err)
	}
	if dir, err := resolveModelConfigInCatalog("test-org/precedence-model", catalogA); err == nil {
		t.Errorf("expected refusal once catalog A's entry is gone, got dir=%q", dir)
	}
	if _, err := resolveModelConfigInCatalog("test-org/precedence-model", catalogB); err != nil {
		t.Errorf("catalog B must be unaffected by catalog A's missing entry: %v", err)
	}
}

// TestResolveModelConfig_CompletenessInvariant verifies the resolution chain's
// completeness law: resolution never returns ("", nil). It must always return either a
// non-empty directory path or a non-nil error (R7: invariant test).
func TestResolveModelConfig_CompletenessInvariant(t *testing.T) {
	emptyCatalog := t.TempDir()

	// Table of inputs covering edge cases
	tests := []struct {
		name    string
		model   string
		catalog string
	}{
		{"empty model", "", emptyCatalog},
		{"org/model, empty catalog", "test-org/test-model", emptyCatalog},
		{"simple model, empty catalog", "simple-model", emptyCatalog},
		{"nonexistent catalog root", "meta-llama/llama-3.1-8b", "/no/such/catalog"},
		{"empty catalog root", "meta-llama/llama-3.1-8b", ""},
		{"path traversal model name", "evil/../../../etc/passwd", emptyCatalog},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir, err := resolveModelConfigInCatalog(tt.model, tt.catalog)
			// Completeness invariant: never ("", nil)
			if dir == "" && err == nil {
				t.Errorf("resolveModelConfigInCatalog(%q, %q) returned (\"\", nil) — "+
					"must return either a non-empty path or a non-nil error",
					tt.model, tt.catalog)
			}
		})
	}
}

// TestIsHFConfig verifies semantic validation of HuggingFace config JSON.
func TestIsHFConfig(t *testing.T) {
	tests := []struct {
		name string
		json string
		want bool
	}{
		{"valid with num_hidden_layers", `{"num_hidden_layers": 32, "hidden_size": 4096}`, true},
		{"valid with hidden_size only", `{"hidden_size": 4096}`, true},
		{"valid with num_hidden_layers only", `{"num_hidden_layers": 32}`, true},
		{"empty object", `{}`, false},
		{"error response", `{"error": "not found"}`, false},
		{"array", `[1, 2, 3]`, false},
		{"string", `"hello"`, false},
		{"invalid JSON", `not json`, false},
		{"multimodal with text_config num_hidden_layers", `{"text_config": {"num_hidden_layers": 48}}`, true},
		{"multimodal with text_config hidden_size", `{"text_config": {"hidden_size": 5120}}`, true},
		{"multimodal with both text_config fields", `{"text_config": {"num_hidden_layers": 48, "hidden_size": 5120, "num_attention_heads": 40}}`, true},
		{"multimodal without expected fields", `{"text_config": {"other_field": 123}, "vision_config": {"hidden_size": 1408}}`, false},
		{"vision_config only (no text_config)", `{"vision_config": {"num_hidden_layers": 34, "hidden_size": 1408}}`, false},
		{"text_config is not an object (string)", `{"text_config": "not_an_object"}`, false},
		{"text_config is not an object (null)", `{"text_config": null}`, false},
		{"deeply nested text_config", `{"text_config": {"text_config": {"num_hidden_layers": 48}}}`, false},
		{"zero-value num_hidden_layers at top level", `{"num_hidden_layers": 0}`, true},
		{"zero-value hidden_size at top level", `{"hidden_size": 0}`, true},
		{"zero-value num_hidden_layers in text_config", `{"text_config": {"num_hidden_layers": 0}}`, true},
		{"zero-value hidden_size in text_config", `{"text_config": {"hidden_size": 0}}`, true},
		{"mixed top-level and text_config fields", `{"num_hidden_layers": 48, "text_config": {"hidden_size": 5120}}`, true},
		// #1729 (NS-4): a config whose only layer-count evidence is the block-type array
		// is one the parser can now read, so the presence detector must accept it too —
		// otherwise the located config would be judged unusable and a fetch attempted.
		{"block-type array only at top level", `{"layers_block_type": ["attention", "mamba"]}`, true},
		{"block-type array only in text_config", `{"text_config": {"layers_block_type": ["attention", "mamba"]}}`, true},
		{"block-type array alongside other fields", `{"layers_block_type": ["attention"], "vocab_size": 32000}`, true},
		// Still rejected: the key names a value the parser cannot count.
		{"block-type array empty", `{"layers_block_type": []}`, false},
		{"block-type array wrong type", `{"layers_block_type": 52}`, false},
		{"vision_config block-type array only", `{"vision_config": {"layers_block_type": ["attention"]}}`, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isHFConfig([]byte(tt.json))
			if got != tt.want {
				t.Errorf("isHFConfig(%s) = %v, want %v", tt.json, got, tt.want)
			}
		})
	}
}

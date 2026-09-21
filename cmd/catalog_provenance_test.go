package cmd

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
)

// Catalog provenance capture tests (#1732, R1/S5). These exercise the acceptance
// criteria of the git capture itself: a clean checkout records its revision with
// dirty=false (AC-2), an uncommitted catalog edit records dirty=true (AC-3), and a
// non-git or absent catalog degrades to an unknown revision without panicking (AC-5).

// requireGit skips the test when no git binary is available. The capture is designed to
// degrade gracefully without git (covered by its own test), so the git-dependent
// assertions are simply not checkable in that environment.
func requireGit(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skipf("git not available: %v", err)
	}
}

// gitInRepo runs a git command inside dir and fails the test on error. Identity and
// signing are supplied per-invocation so the test does not depend on (or mutate) the
// machine's git configuration.
func gitInRepo(t *testing.T, dir string, args ...string) string {
	t.Helper()
	full := append([]string{
		"-C", dir,
		"-c", "user.name=blis-test",
		"-c", "user.email=blis-test@example.invalid",
		"-c", "commit.gpgsign=false",
	}, args...)
	out, err := exec.Command("git", full...).CombinedOutput()
	if err != nil {
		t.Fatalf("git %s in %s: %v\n%s", strings.Join(args, " "), dir, err, out)
	}
	return string(out)
}

// newGitCatalog creates a catalog directory holding one model entry's config.json,
// initializes it as a git repository, and commits. Returns the catalog root and the
// committed HEAD revision.
func newGitCatalog(t *testing.T) (catalogRoot, headRevision string) {
	t.Helper()
	requireGit(t)
	catalogRoot = filepath.Join(t.TempDir(), "catalog")
	entry := filepath.Join(catalogRoot, "test-model")
	if err := os.MkdirAll(entry, 0o755); err != nil {
		t.Fatalf("mkdir catalog entry: %v", err)
	}
	if err := os.WriteFile(filepath.Join(entry, "config.json"), []byte(`{"num_hidden_layers": 2}`), 0o644); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	gitInRepo(t, catalogRoot, "init", "--quiet")
	gitInRepo(t, catalogRoot, "add", ".")
	gitInRepo(t, catalogRoot, "commit", "--quiet", "-m", "catalog: initial entry")
	headRevision = strings.TrimSpace(gitInRepo(t, catalogRoot, "rev-parse", "HEAD"))
	if headRevision == "" {
		t.Fatal("fixture produced no HEAD revision")
	}
	return catalogRoot, headRevision
}

// TestCatalogProvenance_CleanCheckout_RecordsCommittedRevision is AC-2: a clean catalog
// checkout records the committed revision and dirty=false.
func TestCatalogProvenance_CleanCheckout_RecordsCommittedRevision(t *testing.T) {
	catalogRoot, head := newGitCatalog(t)

	got := captureCatalogProvenance(catalogRoot)

	if got == nil {
		t.Fatal("a resolved catalog must produce a provenance block")
	}
	if got.Path != catalogRoot {
		t.Errorf("path = %q, want %q", got.Path, catalogRoot)
	}
	if got.Revision != head {
		t.Errorf("revision = %q, want the committed HEAD %q", got.Revision, head)
	}
	if got.Dirty {
		t.Error("dirty = true for a clean checkout — a reproducible run would be reported as an experiment")
	}
}

// TestCatalogProvenance_UncommittedConfigEdit_RecordsDirty is AC-3: an uncommitted edit
// to a catalog config.json is recorded as dirty rather than silently passing off the
// committed revision as the state that produced the result.
func TestCatalogProvenance_UncommittedConfigEdit_RecordsDirty(t *testing.T) {
	catalogRoot, head := newGitCatalog(t)
	cfg := filepath.Join(catalogRoot, "test-model", "config.json")
	if err := os.WriteFile(cfg, []byte(`{"num_hidden_layers": 4}`), 0o644); err != nil {
		t.Fatalf("edit config.json: %v", err)
	}

	got := captureCatalogProvenance(catalogRoot)

	if got == nil {
		t.Fatal("a resolved catalog must produce a provenance block")
	}
	if !got.Dirty {
		t.Errorf("dirty = false after an uncommitted edit to %s — an experiment is "+
			"indistinguishable from a reproducible run at %s", cfg, head)
	}
	// The revision is still recorded: "dirty at revision X" is strictly more
	// informative than "unknown".
	if got.Revision != head {
		t.Errorf("revision = %q, want the checked-out HEAD %q even when dirty", got.Revision, head)
	}
}

// TestCatalogProvenance_UntrackedEntry_RecordsDirty pins that an uncommitted NEW catalog
// entry counts as dirty. An untracked config.json can change what a run resolves just as
// much as an edit to a tracked one, so the catalog's content differs from its committed
// revision and the result is not reproducible from that revision alone.
func TestCatalogProvenance_UntrackedEntry_RecordsDirty(t *testing.T) {
	catalogRoot, _ := newGitCatalog(t)
	newEntry := filepath.Join(catalogRoot, "scratch-model")
	if err := os.MkdirAll(newEntry, 0o755); err != nil {
		t.Fatalf("mkdir new entry: %v", err)
	}
	if err := os.WriteFile(filepath.Join(newEntry, "config.json"), []byte(`{"num_hidden_layers": 8}`), 0o644); err != nil {
		t.Fatalf("write new entry: %v", err)
	}

	got := captureCatalogProvenance(catalogRoot)

	if got == nil || !got.Dirty {
		t.Errorf("an untracked catalog entry must be recorded as dirty; got %+v", got)
	}
}

// TestCatalogProvenance_DirtyIsScopedToCatalogSubtree pins the scoping law: a catalog is
// commonly a SUBDIRECTORY of a larger checkout, and an unrelated edit elsewhere in that
// repository must not mark the run an experiment — otherwise the flag fires on nearly
// every developer run and stops distinguishing anything.
func TestCatalogProvenance_DirtyIsScopedToCatalogSubtree(t *testing.T) {
	requireGit(t)
	repo := t.TempDir()
	catalogRoot := filepath.Join(repo, "model_configs")
	entry := filepath.Join(catalogRoot, "test-model")
	if err := os.MkdirAll(entry, 0o755); err != nil {
		t.Fatalf("mkdir catalog entry: %v", err)
	}
	if err := os.WriteFile(filepath.Join(entry, "config.json"), []byte(`{"num_hidden_layers": 2}`), 0o644); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	unrelated := filepath.Join(repo, "docs", "notes.md")
	if err := os.MkdirAll(filepath.Dir(unrelated), 0o755); err != nil {
		t.Fatalf("mkdir docs: %v", err)
	}
	if err := os.WriteFile(unrelated, []byte("committed\n"), 0o644); err != nil {
		t.Fatalf("write notes.md: %v", err)
	}
	gitInRepo(t, repo, "init", "--quiet")
	gitInRepo(t, repo, "add", ".")
	gitInRepo(t, repo, "commit", "--quiet", "-m", "repo: catalog plus docs")
	head := strings.TrimSpace(gitInRepo(t, repo, "rev-parse", "HEAD"))

	// Edit a file OUTSIDE the catalog subtree.
	if err := os.WriteFile(unrelated, []byte("edited outside the catalog\n"), 0o644); err != nil {
		t.Fatalf("edit notes.md: %v", err)
	}

	got := captureCatalogProvenance(catalogRoot)
	if got == nil {
		t.Fatal("a resolved catalog must produce a provenance block")
	}
	if got.Revision != head {
		t.Errorf("revision = %q, want %q", got.Revision, head)
	}
	if got.Dirty {
		t.Error("dirty = true for an edit outside the catalog subtree — the flag must " +
			"describe the CATALOG's content, not the whole containing repository")
	}

	// Non-vacuity: the same repository DOES report dirty when the catalog itself changes,
	// so the assertion above is not passing because the check is inert.
	if err := os.WriteFile(filepath.Join(entry, "config.json"), []byte(`{"num_hidden_layers": 3}`), 0o644); err != nil {
		t.Fatalf("edit config.json: %v", err)
	}
	if inside := captureCatalogProvenance(catalogRoot); inside == nil || !inside.Dirty {
		t.Errorf("non-vacuity: an edit INSIDE the catalog subtree must be dirty; got %+v", inside)
	}
}

// TestCatalogProvenance_NonGitCatalog_DegradesGracefully is AC-5: a catalog that is not a
// git checkout, and a catalog directory that does not exist, both record the path with an
// unknown revision and no panic. Neither may invent a revision or a dirty verdict.
func TestCatalogProvenance_NonGitCatalog_DegradesGracefully(t *testing.T) {
	tmp := t.TempDir()
	plainCatalog := filepath.Join(tmp, "plain-catalog")
	if err := os.MkdirAll(filepath.Join(plainCatalog, "test-model"), 0o755); err != nil {
		t.Fatalf("mkdir plain catalog: %v", err)
	}
	absentCatalog := filepath.Join(tmp, "does-not-exist")

	for _, tc := range []struct {
		name string
		path string
	}{
		{"non-git-directory", plainCatalog},
		{"absent-directory", absentCatalog},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := captureCatalogProvenance(tc.path)
			if got == nil {
				t.Fatalf("the path must still be recorded for %s", tc.name)
			}
			if got.Path != tc.path {
				t.Errorf("path = %q, want %q", got.Path, tc.path)
			}
			if got.Revision != sim.UnknownCatalogRevision {
				t.Errorf("revision = %q, want %q — a revision must never be invented",
					got.Revision, sim.UnknownCatalogRevision)
			}
			if got.Dirty {
				t.Errorf("dirty = true with an unknown revision: there is no committed " +
					"state for the catalog to differ from")
			}
		})
	}
}

// TestCatalogProvenance_NoCatalogResolved_OmitsBlock pins that an unresolved catalog
// (empty path) yields no block at all, rather than a block claiming an empty catalog path.
func TestCatalogProvenance_NoCatalogResolved_OmitsBlock(t *testing.T) {
	if got := captureCatalogProvenance(""); got != nil {
		t.Errorf("an empty catalog root must produce no provenance block; got %+v", got)
	}
}

// TestCatalogProvenanceEmitOptions_FileOnly pins that the shared emit-option helper is a
// no-op without an output file path (provenance is file-only, so a stdout-only run must
// not pay for a git subprocess) and supplies exactly one option when a file is written.
func TestCatalogProvenanceEmitOptions_FileOnly(t *testing.T) {
	catalogRoot, _ := newGitCatalog(t)

	if opts := catalogProvenanceEmitOptions("", catalogRoot); len(opts) != 0 {
		t.Errorf("no output file path must yield no emit options, got %d", len(opts))
	}
	opts := catalogProvenanceEmitOptions(filepath.Join(t.TempDir(), "metrics.json"), catalogRoot)
	if len(opts) != 1 {
		t.Fatalf("an output file path must yield exactly one emit option, got %d", len(opts))
	}
	// The option must carry a populated provenance, not a nil one.
	m := sim.NewMetrics()
	out := m.BuildOutput("cluster")
	fpath := filepath.Join(t.TempDir(), "metrics.json")
	if err := m.EmitOutput(out, fpath, catalogProvenanceEmitOptions(fpath, catalogRoot)...); err != nil {
		t.Fatalf("EmitOutput: %v", err)
	}
	data, err := os.ReadFile(fpath)
	if err != nil {
		t.Fatalf("read metrics file: %v", err)
	}
	if !strings.Contains(string(data), catalogRoot) {
		t.Errorf("results file must record the catalog path %q:\n%s", catalogRoot, data)
	}
}

// TestCatalogProvenance_EveryFileEmitSitePassesProvenance is the R23 drift guard for
// INV-13: every production EmitOutput call in cmd/ must route through
// catalogProvenanceEmitOptions, so a new emit site cannot silently produce a results file
// with no provenance while its sibling command records one. (The per-instance
// stdout-only path goes through SaveResults, which writes no file and is not an
// EmitOutput site.)
func TestCatalogProvenance_EveryFileEmitSitePassesProvenance(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob cmd/*.go: %v", err)
	}
	sites := 0
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		src, readErr := os.ReadFile(file)
		if readErr != nil {
			t.Fatalf("read %s: %v", file, readErr)
		}
		for _, call := range emitOutputCallArgs(string(src)) {
			sites++
			if !strings.Contains(call.args, "catalogProvenanceEmitOptions") {
				t.Errorf("%s:%d: EmitOutput call must pass catalogProvenanceEmitOptions(...) "+
					"so its results file records catalog provenance (#1732, INV-13):\n\t%s",
					file, call.line, strings.Join(strings.Fields(call.args), " "))
			}
		}
	}
	// Non-vacuity: the two known emit sites (run + replay) must be found, otherwise the
	// scan is looking at nothing and would pass after a rename.
	if sites < 2 {
		t.Fatalf("non-vacuity: expected at least the `blis run` and `blis replay` EmitOutput "+
			"sites, found %d", sites)
	}
}

// emitOutputCall is one `.EmitOutput(...)` call found in a source file: its 1-based line
// number and the full argument text, which may span several lines.
type emitOutputCall struct {
	line int
	args string
}

// emitOutputCallArgs extracts every `.EmitOutput(...)` call's argument text from Go source
// by matching parentheses, so a call broken across lines is examined as one unit (a
// line-wise grep would miss an option passed on a continuation line).
func emitOutputCallArgs(src string) []emitOutputCall {
	const marker = ".EmitOutput("
	var calls []emitOutputCall
	for offset := 0; ; {
		idx := strings.Index(src[offset:], marker)
		if idx < 0 {
			return calls
		}
		open := offset + idx + len(marker) - 1 // index of '('
		depth, end := 0, -1
		for i := open; i < len(src); i++ {
			switch src[i] {
			case '(':
				depth++
			case ')':
				depth--
				if depth == 0 {
					end = i
				}
			}
			if end >= 0 {
				break
			}
		}
		if end < 0 {
			return calls // unbalanced source; nothing more to extract
		}
		calls = append(calls, emitOutputCall{
			line: strings.Count(src[:open], "\n") + 1,
			args: src[open+1 : end],
		})
		offset = end
	}
}

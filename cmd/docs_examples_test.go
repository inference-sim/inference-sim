package cmd

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// docExampleRoots is the LIVE documentation set whose `blis run` / `blis replay` examples
// must stay runnable. Paths are relative to the cmd/ test working directory.
//
// Deliberately excluded, because they are historical records rather than instructions:
// docs/plans/ (archived implementation plans, "PR history" per CLAUDE.md) and specs/
// (frozen per-feature spec artifacts). Editing a shipped plan's example would rewrite the
// record of what that PR did.
var docExampleRoots = []string{
	"../CLAUDE.md",
	"../README.md",
	"../docs",     // the mkdocs site, minus docs/plans (skipped below)
	"../examples", // YAML fixtures whose header comments are copy-paste commands
	"../scripts",
}

// docExampleSkipDirs are subtrees under docExampleRoots that hold historical records.
var docExampleSkipDirs = []string{"../docs/plans"}

// docCommandStart matches the beginning of a `blis run` / `blis replay` shell invocation
// in prose or in a YAML comment: optional indentation, an optional `# ` comment marker,
// an optional `$ ` prompt and `./` prefix, then the binary and subcommand.
var docCommandStart = regexp.MustCompile(`^\s*(?:#\s+)?(?:\$ )?(?:\./)?blis\s+(?:run|replay)\b`)

// TestDocExamplesPassDeploymentFlags is BC-6 for #1733: --hardware and --tp are REQUIRED,
// so every documented `blis run` / `blis replay` example must pass both — otherwise the
// docs hand the reader a command that now aborts.
//
// It is a behavioral test of the docs, not a style check: it reconstructs each example
// command (joining `\` line continuations, exactly as a shell would) and asserts the
// resulting command line carries both flags. A placeholder value (`--tp <N>`) satisfies it;
// an omitted flag does not.
func TestDocExamplesPassDeploymentFlags(t *testing.T) {
	files := collectDocFiles(t)
	if len(files) == 0 {
		t.Fatal("non-vacuity: found no documentation files to scan")
	}

	examples := 0
	for _, file := range files {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		lines := strings.Split(string(src), "\n")
		for i := 0; i < len(lines); i++ {
			if !docCommandStart.MatchString(lines[i]) {
				continue
			}
			// Join `\` continuations into the single command line a shell would see.
			start, command := i, lines[i]
			for strings.HasSuffix(strings.TrimRight(lines[i], " \t"), `\`) && i+1 < len(lines) {
				i++
				command += " " + lines[i]
			}
			examples++

			var missing []string
			if !hasFlag(command, "--hardware") {
				missing = append(missing, "--hardware")
			}
			if !hasFlag(command, "--tp") {
				missing = append(missing, "--tp")
			}
			if len(missing) > 0 {
				t.Errorf("%s:%d: documented example omits %s — both are required flags since "+
					"#1733 (NS-6), so this command aborts as written:\n  %s",
					file, start+1, strings.Join(missing, " and "), collapse(command))
			}
		}
	}
	// Guard against the scanner silently matching nothing (a regex edit, a moved doc tree).
	if examples < 50 {
		t.Errorf("non-vacuity: expected the live docs to contain many blis run/replay examples, found %d — "+
			"docCommandStart or docExampleRoots is probably wrong", examples)
	}
}

// hasFlag reports whether the command passes the given flag, in either `--flag value` or
// `--flag=value` form. The trailing separator check keeps `--tp` from matching a longer
// flag that starts with it.
func hasFlag(command, flag string) bool {
	for _, form := range []string{flag + " ", flag + "="} {
		if strings.Contains(command, form) {
			return true
		}
	}
	return strings.HasSuffix(strings.TrimRight(command, " \t\\"), flag)
}

// collapse squeezes whitespace so a multi-line example prints as one readable line.
func collapse(command string) string {
	return strings.Join(strings.Fields(strings.ReplaceAll(command, `\`, " ")), " ")
}

// collectDocFiles walks docExampleRoots and returns every markdown/YAML file that could
// hold a command example, skipping the historical-record subtrees.
func collectDocFiles(t *testing.T) []string {
	t.Helper()
	var files []string
	for _, root := range docExampleRoots {
		info, err := os.Stat(root)
		if err != nil {
			t.Fatalf("stat %s: %v (docExampleRoots is stale)", root, err)
		}
		if !info.IsDir() {
			files = append(files, root)
			continue
		}
		err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				for _, skip := range docExampleSkipDirs {
					if filepath.Clean(path) == filepath.Clean(skip) {
						return fs.SkipDir
					}
				}
				return nil
			}
			switch filepath.Ext(path) {
			case ".md", ".yaml", ".yml", ".sh":
				files = append(files, path)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("walk %s: %v", root, err)
		}
	}
	return files
}

// docCatalogEntryPoints are the documentation entry points a reader actually starts from.
// Since #1731 the model catalog must be located explicitly (--catalog / BLIS_CATALOG, no
// default and no search path), so each of these must tell the reader how — otherwise every
// example downstream of it aborts as written.
var docCatalogEntryPoints = []string{
	"../CLAUDE.md",
	"../README.md",
	"../docs/index.md",
	"../docs/getting-started/quickstart.md",
	"../docs/getting-started/installation.md",
	"../docs/getting-started/tutorial.md",
}

// TestDocExamplesDocumentTheCatalogEnvVar is the #1731 counterpart of
// TestDocExamplesPassDeploymentFlags, and deliberately a DIFFERENT shape.
//
// --hardware/--tp have no environment-variable form, so #1733 had to add them to every one
// of the ~108 documented examples. --catalog does have one — BLIS_CATALOG exists precisely
// so the catalog location is stated ONCE rather than repeated in every command. So the
// documented examples keep omitting the flag, and what has to hold instead is that each
// entry point a reader starts from names both forms and shows the one-time export. This
// test pins that instruction, so a future edit cannot silently delete the thing that makes
// those examples runnable.
//
// If a later PR decides to put --catalog on every example instead, delete this test and
// extend TestDocExamplesPassDeploymentFlags — but the two policies must not both be half
// applied, which is exactly what an unpinned prose instruction drifts into.
func TestDocExamplesDocumentTheCatalogEnvVar(t *testing.T) {
	for _, path := range docCatalogEntryPoints {
		src, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v (docCatalogEntryPoints is stale)", path, err)
		}
		text := string(src)
		if !strings.Contains(text, catalogEnvVar) {
			t.Errorf("%s must name the %s environment variable: the catalog has no default and no "+
				"search path since #1731, so a reader following this page needs to know how to "+
				"locate it", path, catalogEnvVar)
		}
		if !strings.Contains(text, "--catalog") {
			t.Errorf("%s must name --catalog, the explicit form that overrides %s", path, catalogEnvVar)
		}
	}

	// Non-vacuity: the export instruction itself must be present somewhere a reader
	// following the quick start will run it, not merely mentioned in passing.
	quickstart, err := os.ReadFile("../docs/getting-started/quickstart.md")
	if err != nil {
		t.Fatalf("read quickstart: %v", err)
	}
	if !strings.Contains(string(quickstart), "export "+catalogEnvVar+"=") {
		t.Errorf("docs/getting-started/quickstart.md must show the one-time `export %s=...` the "+
			"downstream examples rely on", catalogEnvVar)
	}
}

// catalogCompatibilityDoc is the CANONICAL place the compatible blis-catalog release is
// declared (#1814). Every other page's clone command must pin the same tag; this is the only
// file whose prose the guard below reads a version out of.
const catalogCompatibilityDoc = "../docs/getting-started/installation.md"

// canonicalCatalogPin extracts the declared release tag from catalogCompatibilityDoc. The
// shape is fixed deliberately: a bump edits one line, and the guard fails loudly (rather
// than silently finding nothing to compare against) if that line is reworded away.
var canonicalCatalogPin = regexp.MustCompile("(?m)^\\*\\*Compatible blis-catalog release: `([^`]+)`\\*\\*")

// catalogCloneCommand matches a documented `git clone` of the model catalog, in a fenced
// block or inline in prose. Both forms exist today (README.md states it inline).
//
// It requires the full repository URL, so it matches instructions a reader can copy and
// paste and not the abbreviated `git clone .../blis-catalog` that appears in CLAUDE.md's
// change history — a record of what a past PR said, which must not be rewritten.
var catalogCloneCommand = regexp.MustCompile(`git clone\b[^\n]*github\.com/inference-sim/blis-catalog`)

// TestDocExamplesPinTheCatalogVersion is #1814: every documented clone of blis-catalog must
// pin an explicit release tag, and they must all pin the SAME tag — the one declared in
// docs/getting-started/installation.md.
//
// Why this is a guard test rather than a docs convention: blis-catalog versions
// independently of BLIS, and BLIS parses every catalog file with KnownFields(true) (R10).
// Strict parsing is one-way — an added or renamed key in a future catalog schema is a hard
// load error, not a silently ignored field — so a bare `git clone` of `main` lets a catalog
// release turn a previously-working `blis run` into a startup failure with no change to
// BLIS. The floating clone is the bug; an edit that reintroduces one has to fail.
//
// It is the sibling of TestDocExamplesDocumentTheCatalogEnvVar (which pins that the
// location is documented at all) and scans the whole live docs set, not just the entry
// points, so a NEW page cannot introduce an unpinned clone either.
func TestDocExamplesPinTheCatalogVersion(t *testing.T) {
	declaration, err := os.ReadFile(catalogCompatibilityDoc)
	if err != nil {
		t.Fatalf("read %s: %v (catalogCompatibilityDoc is stale)", catalogCompatibilityDoc, err)
	}
	match := canonicalCatalogPin.FindStringSubmatch(string(declaration))
	if match == nil {
		t.Fatalf("%s must declare the compatible catalog release on its own line, as "+
			"**Compatible blis-catalog release: `<tag>`** — it is the single place the version "+
			"is stated, and every documented clone command is checked against it", catalogCompatibilityDoc)
	}
	tag := match[1]
	if strings.TrimSpace(tag) == "" || strings.Contains(tag, "<") {
		t.Fatalf("%s declares a placeholder catalog release %q — the pin must name a real "+
			"blis-catalog release tag", catalogCompatibilityDoc, tag)
	}
	if !strings.Contains(string(declaration), "## Catalog compatibility") {
		t.Errorf("%s must keep the `## Catalog compatibility` heading: the other pages link to "+
			"its #catalog-compatibility anchor", catalogCompatibilityDoc)
	}

	// Every documented clone of the catalog pins that exact tag.
	wantPin := "--branch " + tag
	clones := 0
	for _, file := range collectDocFiles(t) {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		for i, line := range strings.Split(string(src), "\n") {
			if !catalogCloneCommand.MatchString(line) {
				continue
			}
			clones++
			if !strings.Contains(line, "--branch ") {
				t.Errorf("%s:%d: documented blis-catalog clone is unpinned — it takes whatever "+
					"`main` is at clone time, so a catalog release can break this build (strict "+
					"parsing makes a schema change a hard load error). Pin it with `%s`:\n  %s",
					file, i+1, wantPin, strings.TrimSpace(line))
				continue
			}
			if !strings.Contains(line, wantPin) {
				t.Errorf("%s:%d: documented blis-catalog clone pins a different release than the "+
					"canonical declaration in %s (`%s`). One version, stated once:\n  %s",
					file, i+1, catalogCompatibilityDoc, tag, strings.TrimSpace(line))
			}
		}
	}
	// Non-vacuity: the entry points named by docCatalogEntryPoints each show the clone, so a
	// scan that finds fewer has a broken matcher (or lost an instruction a reader needs).
	if clones < len(docCatalogEntryPoints) {
		t.Errorf("non-vacuity: found %d documented blis-catalog clone commands, expected at least "+
			"one per entry point (%d) — catalogCloneCommand or the docs are wrong",
			clones, len(docCatalogEntryPoints))
	}
}

// TestDocCatalogEntryPointsLinkTheCompatibilityNote pins the other half of #1814's "state it
// in one canonical place, linked from the others": a page showing the pinned clone must also
// say where the version comes from. Without the link the tag reads as an arbitrary string,
// and a reader hitting a catalog incompatibility has nowhere to go.
func TestDocCatalogEntryPointsLinkTheCompatibilityNote(t *testing.T) {
	for _, path := range docCatalogEntryPoints {
		src, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v (docCatalogEntryPoints is stale)", path, err)
		}
		text := string(src)
		if !catalogCloneCommand.MatchString(text) {
			continue // this page defers the clone instruction to another entry point
		}
		if filepath.Clean(path) == filepath.Clean(catalogCompatibilityDoc) {
			continue // the canonical note itself, checked above
		}
		if !strings.Contains(text, "#catalog-compatibility") &&
			!strings.Contains(text, "Catalog compatibility") {
			t.Errorf("%s shows the pinned blis-catalog clone but never points at the canonical "+
				"Catalog compatibility note (%s#catalog-compatibility), so a reader cannot tell "+
				"where the pinned version comes from or how to move off it",
				path, catalogCompatibilityDoc)
		}
	}
}

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
//
// It is matched against a RECONSTRUCTED command (docLogicalLines below), not a physical
// line, so a clone wrapped across lines with `\` is still seen as one command.
var catalogCloneCommand = regexp.MustCompile(`git clone\b[^\n]*github\.com/inference-sim/blis-catalog`)

// catalogClonePinFlag captures the ref a clone command pins, as the WHOLE flag value. The
// value is delimited by whitespace (or the `\` of a line continuation docLogicalLines has
// joined in), which is what makes the comparison below exact: a substring test for
// "--branch 0.1.1" is satisfied by `--branch 0.1.10`, so a page left on a different release
// whose tag merely starts with the canonical one would pass unnoticed.
//
// The spellings git itself accepts are all recognized, so a clone pinned as `-b <tag>` or
// `--branch=<tag>` is compared against the canonical tag rather than reported as unpinned.
var catalogClonePinFlag = regexp.MustCompile(`(?:--branch[ =]|-b )([^\s\\]+)`)

// catalogClonePin returns the ref a documented clone command pins, and whether it pins one
// at all. Presence and value come from the same extraction, so "unpinned" can never be
// reported for a command that does carry a ref in a spelling the value check would have read.
func catalogClonePin(command string) (string, bool) {
	match := catalogClonePinFlag.FindStringSubmatch(command)
	if match == nil {
		return "", false
	}
	return match[1], true
}

// docLogicalLine is one reconstructed shell command from a doc file, with the 1-based
// physical line it starts on so a diagnostic names the line a reader would edit.
type docLogicalLine struct {
	num  int
	text string
}

// docLogicalLines joins `\` continuations into the single command line a shell would see —
// the same reconstruction TestDocExamplesPassDeploymentFlags does for `blis run` examples.
//
// Matching physical lines instead would be wrong in both directions for a wrapped clone
// command: `git clone <url> \` / `  --branch <tag>` reports a false "unpinned" (the pin is
// on the continuation line), and `git clone --branch <tag> \` / `  <url>` is missed entirely
// (no single line carries both `git clone` and the URL).
//
// A markdown hard line break is also a trailing `\`, so prose can be joined too. That is
// safe here: joining only ever ADDS text to a logical line, so it can make the pin check
// more permissive but can never turn a correctly pinned clone into a failure. The matched
// text is reported collapsed, so an over-joined diagnostic still reads as one command.
func docLogicalLines(src string) []docLogicalLine {
	lines := strings.Split(src, "\n")
	out := make([]docLogicalLine, 0, len(lines))
	for i := 0; i < len(lines); i++ {
		start, command := i, lines[i]
		for strings.HasSuffix(strings.TrimRight(lines[i], " \t"), `\`) && i+1 < len(lines) {
			i++
			command += " " + lines[i]
		}
		out = append(out, docLogicalLine{num: start + 1, text: command})
	}
	return out
}

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
// points, so a NEW page that introduces an unpinned clone fails too.
//
// Two separate things are checked, because a global count cannot establish either alone:
// every clone found anywhere pins the canonical tag, AND every page in
// docCatalogEntryPoints still carries a clone of its own (so a matcher that stops seeing
// one entry point's command cannot be covered for by a second match on another page).
//
// "Pins the canonical tag" is EQUALITY of the extracted flag value (catalogClonePin), not a
// substring test: `--branch 0.1.10` is a different release from `--branch 0.1.1` even though
// one contains the other, and both shapes must fail. TestCatalogClonePinIsExtractedExactly
// covers the extraction over those cases.
//
// Scope, stated so the guarantee is not read wider than it is: what is checked is every
// clone command catalogCloneCommand recognizes, which by design means those carrying the
// full repository URL (see that matcher). A clone written with an abbreviated URL is not
// matched and so not checked — deliberate, since that is the shape CLAUDE.md's change
// history uses and must keep. The per-entry-point count above is what keeps a *recognized*
// command from quietly becoming an unrecognized one on the pages that matter.
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
	clonesByFile := map[string]int{}
	for _, file := range collectDocFiles(t) {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		for _, command := range docLogicalLines(string(src)) {
			if !catalogCloneCommand.MatchString(command.text) {
				continue
			}
			clonesByFile[filepath.Clean(file)]++
			pinned, ok := catalogClonePin(command.text)
			if !ok {
				t.Errorf("%s:%d: documented blis-catalog clone is unpinned — it takes whatever "+
					"`main` is at clone time, so a catalog release can break this build (strict "+
					"parsing makes a schema change a hard load error). Pin it with `%s`:\n  %s",
					file, command.num, wantPin, collapse(command.text))
				continue
			}
			if pinned != tag {
				t.Errorf("%s:%d: documented blis-catalog clone pins release %q, but the canonical "+
					"declaration in %s says %q. One version, stated once:\n  %s",
					file, command.num, pinned, catalogCompatibilityDoc, tag, collapse(command.text))
			}
		}
	}

	// Non-vacuity, PER ENTRY POINT: each page a reader starts from must carry a clone command
	// of its own. A single total would let extra matches on one page cover for an entry point
	// the matcher stopped seeing — which is the case that matters, since a missed clone is
	// silently exempt from the pin check above.
	for _, entry := range docCatalogEntryPoints {
		if clonesByFile[filepath.Clean(entry)] == 0 {
			t.Errorf("non-vacuity: found no documented blis-catalog clone command in %s — either "+
				"the page lost an instruction a reader needs, or catalogCloneCommand no longer "+
				"matches how it is written (in which case its pin is unchecked)", entry)
		}
	}
}

// TestDocCatalogEntryPointsLinkTheCompatibilityNote pins the other half of #1814's "state it
// in one canonical place, linked from the others": a page showing the pinned clone must also
// say where the version comes from. Without the link the tag reads as an arbitrary string,
// and a reader hitting a catalog incompatibility has nowhere to go.
//
// Every entry point is required to link the note, unconditionally — there is deliberately no
// "this page has no clone, skip it" escape, which would make the check silently vacuous for
// exactly the page whose clone command the matcher had stopped recognizing. That every entry
// point does show a clone is the sibling test's per-entry-point non-vacuity assertion.
func TestDocCatalogEntryPointsLinkTheCompatibilityNote(t *testing.T) {
	for _, path := range docCatalogEntryPoints {
		if filepath.Clean(path) == filepath.Clean(catalogCompatibilityDoc) {
			continue // the canonical note itself, checked by the sibling test
		}
		src, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v (docCatalogEntryPoints is stale)", path, err)
		}
		// The ANCHOR is required, not the note's title: a page that merely says the words
		// "Catalog compatibility" in prose gives a reader nothing to follow, so accepting that
		// would let the link be deleted without the guard noticing. Requiring the fragment
		// admits every link form in use — `#catalog-compatibility`,
		// `installation.md#catalog-compatibility`, `getting-started/installation.md#...` — since
		// all of them end in it, and the relative prefix legitimately differs per page depth.
		if !strings.Contains(string(src), "#catalog-compatibility") {
			t.Errorf("%s never links the canonical Catalog compatibility note "+
				"(%s#catalog-compatibility): every entry point shows the pinned blis-catalog "+
				"clone, so without a followable link a reader cannot tell where the pinned version "+
				"comes from or how to move off it",
				path, catalogCompatibilityDoc)
		}
	}
}

// TestDocLogicalLinesReconstructsWrappedCloneCommands covers the guard's own matcher over the
// wrapped forms a doc author may reasonably write. A line-oriented scan mis-reads both: the
// first as an unpinned clone (the pin is on the continuation line) and the second as no clone
// at all (no single line carries both `git clone` and the URL) — so a wrapped clone would
// either fail a correct page or exempt an unpinned one from the check.
func TestDocLogicalLinesReconstructsWrappedCloneCommands(t *testing.T) {
	const url = "https://github.com/inference-sim/blis-catalog.git"
	for _, tc := range []struct {
		name     string
		src      string
		wantPin  bool // the reconstructed clone command carries --branch
		wantLine int  // physical line the command is reported at
	}{
		{
			name:     "single line pinned",
			src:      "git clone --branch 0.1.1 --depth 1 " + url + "\n",
			wantPin:  true,
			wantLine: 1,
		},
		{
			name:     "url first, pin on continuation",
			src:      "git clone " + url + " \\\n  --branch 0.1.1 --depth 1\n",
			wantPin:  true,
			wantLine: 1,
		},
		{
			name:     "pin first, url on continuation",
			src:      "git clone --branch 0.1.1 --depth 1 \\\n  " + url + "\n",
			wantPin:  true,
			wantLine: 1,
		},
		{
			name:     "wrapped and genuinely unpinned is still caught",
			src:      "git clone --depth 1 \\\n  " + url + "\n",
			wantPin:  false,
			wantLine: 1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var found []docLogicalLine
			for _, command := range docLogicalLines(tc.src) {
				if catalogCloneCommand.MatchString(command.text) {
					found = append(found, command)
				}
			}
			if len(found) != 1 {
				t.Fatalf("expected exactly 1 reconstructed catalog clone command, got %d from:\n%s",
					len(found), tc.src)
			}
			if _, got := catalogClonePin(found[0].text); got != tc.wantPin {
				t.Errorf("pin detected = %v, want %v, in reconstructed command: %s",
					got, tc.wantPin, collapse(found[0].text))
			}
			if found[0].num != tc.wantLine {
				t.Errorf("reported line = %d, want %d (a diagnostic must name the line a reader edits)",
					found[0].num, tc.wantLine)
			}
		})
	}
}

// TestCatalogClonePinIsExtractedExactly covers the guard's pin comparison over the shapes that
// decide whether a page is actually held to the canonical release.
//
// The case that motivates it: the comparison used to be a substring test for
// "--branch <tag>", which `--branch 0.1.10` satisfies when the canonical tag is `0.1.1` — so a
// page pinned to a DIFFERENT release whose tag happens to extend the canonical one passed
// silently, defeating the guard's whole purpose. The pin is therefore extracted as a whole
// flag value and compared for equality, and the prefix cases below are what keeps it that way.
func TestCatalogClonePinIsExtractedExactly(t *testing.T) {
	const url = "https://github.com/inference-sim/blis-catalog.git"
	for _, tc := range []struct {
		name     string
		command  string
		wantPin  string
		wantHas  bool
		wantSame bool // extracted pin equals the canonical tag "0.1.1"
	}{
		{
			name:     "canonical pin",
			command:  "git clone --branch 0.1.1 --depth 1 " + url,
			wantPin:  "0.1.1",
			wantHas:  true,
			wantSame: true,
		},
		{
			name:     "pin last on the line",
			command:  "git clone --depth 1 " + url + " --branch 0.1.1",
			wantPin:  "0.1.1",
			wantHas:  true,
			wantSame: true,
		},
		{
			name:     "equals form",
			command:  "git clone --branch=0.1.1 --depth 1 " + url,
			wantPin:  "0.1.1",
			wantHas:  true,
			wantSame: true,
		},
		{
			name:     "short form is a pin, not an unpinned clone",
			command:  "git clone -b 0.1.1 --depth 1 " + url,
			wantPin:  "0.1.1",
			wantHas:  true,
			wantSame: true,
		},
		{
			name:     "later release extending the canonical tag is a DIFFERENT release",
			command:  "git clone --branch 0.1.10 --depth 1 " + url,
			wantPin:  "0.1.10",
			wantHas:  true,
			wantSame: false,
		},
		{
			name:     "prerelease extending the canonical tag is a DIFFERENT release",
			command:  "git clone --branch 0.1.1-rc1 --depth 1 " + url,
			wantPin:  "0.1.1-rc1",
			wantHas:  true,
			wantSame: false,
		},
		{
			name:     "earlier release",
			command:  "git clone --branch 0.1.0 --depth 1 " + url,
			wantPin:  "0.1.0",
			wantHas:  true,
			wantSame: false,
		},
		{
			name:     "pin followed by a joined line continuation",
			command:  "git clone --branch 0.1.1 \\   " + url,
			wantPin:  "0.1.1",
			wantHas:  true,
			wantSame: true,
		},
		{
			name:    "unpinned",
			command: "git clone --depth 1 " + url,
			wantHas: false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pin, has := catalogClonePin(tc.command)
			if has != tc.wantHas {
				t.Fatalf("pinned = %v, want %v, for command: %s", has, tc.wantHas, tc.command)
			}
			if pin != tc.wantPin {
				t.Errorf("extracted pin = %q, want %q, for command: %s", pin, tc.wantPin, tc.command)
			}
			if got := has && pin == "0.1.1"; got != tc.wantSame {
				t.Errorf("matches canonical tag 0.1.1 = %v, want %v (pin %q) — a guard that "+
					"accepts a different release defeats the pin", got, tc.wantSame, pin)
			}
		})
	}
}

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
// docs/plans/ (archived implementation plans, "PR history" per CLAUDE.md), docs/blog/ (dated
// posts, which carry an editor's note instead of a rewrite) and specs/ (frozen per-feature
// spec artifacts). Editing a shipped plan's example would rewrite the
// record of what that PR did.
var docExampleRoots = []string{
	"../CLAUDE.md",
	"../README.md",
	"../docs",             // the mkdocs site, minus docs/plans (skipped below)
	"../examples",         // YAML fixtures whose header comments are copy-paste commands
	"../examplescenarios", // ready-to-run scenarios + their README's copy-paste commands
	"../scripts",
}

// docExampleSkipDirs are subtrees under docExampleRoots that hold historical records.
var docExampleSkipDirs = []string{"../docs/plans", "../docs/blog"}

// docCommandStart matches the beginning of a `blis run` / `blis replay` shell invocation
// in prose or in a YAML comment: optional indentation, an optional `# ` comment marker,
// an optional `$ ` prompt and `./` prefix, then the binary and subcommand.
var docCommandStart = regexp.MustCompile(`^\s*(?:#\s+)?(?:\$ )?(?:\./)?blis\s+(?:run|replay)\b`)

// removedRunFlags are the flags `blis run` / `blis replay` no longer register: the kernel
// scenario states the deployment and the engine the kernel prices, so a documented example
// passing one aborts with "unknown flag".
var removedRunFlags = []string{
	"--latency-model", "--alpha-coeffs", "--beta-coeffs", "--hardware-config",
	"--model", "--hardware", "--tp", "--dp", "--enable-expert-parallel", "--moe-comm-backend",
	"--prefill-tp", "--decode-tp", "--prefill-hardware", "--decode-hardware",
	"--prefill-latency-model", "--decode-latency-model", "--prefill-max-model-len", "--decode-max-model-len",
	"--prefill-moe-comm-backend", "--decode-moe-comm-backend",
	"--total-kv-blocks", "--max-num-seqs", "--max-num-running-reqs", "--max-num-batched-tokens",
	"--max-num-scheduled-tokens", "--block-size-in-tokens", "--gpu-memory-utilization", "--max-model-len",
	"--kv-cache-dtype", "--no-enable-prefix-caching", "--num-speculative-tokens", "--speculative-method",
	"--kv-transfer-bandwidth", "--kv-transfer-base-latency",
	"--pd-transfer-bandwidth", "--pd-transfer-base-latency",
	"--comm-serialization-factor", "--enforce-eager",
	"--epp-flags", "--epp-config", "--epp-manifests", "--pod-labels", "--epp-version", "--epp-cpu-tier-view",
}

// TestDocExamplesPassDeploymentFlags: the deployment comes from a kernel scenario, so every
// documented `blis run` / `blis replay` example must name one (--scenario) and must pass none
// of the removed deployment/engine flags -- otherwise the docs hand the reader a command that
// aborts as written.
//
// It is a behavioral test of the docs, not a style check: it reconstructs each example
// command (joining `\` line continuations, exactly as a shell would) and asserts on the
// resulting command line. A placeholder value (`--scenario <file>`) satisfies it.
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

			if !hasFlag(command, "--scenario") {
				t.Errorf("%s:%d: documented example names no --scenario, which every run and "+
					"replay requires, so this command aborts as written:\n  %s",
					file, start+1, collapse(command))
			}
			for _, removed := range removedRunFlags {
				if hasFlag(command, removed) {
					t.Errorf("%s:%d: documented example passes %s, which is no longer a flag (the "+
						"kernel scenario states it), so this command aborts as written:\n  %s",
						file, start+1, removed, collapse(command))
				}
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

// docCatalogEntryPoints are the documentation entry points a reader starts from. A run reads
// the catalog and registry embedded in the pinned blis-substrate unless --catalog/BLIS_CATALOG
// (or --registry/BLIS_REGISTRY) replace them (SIM-6), so each of these must tell the reader how
// to use their own.
var docCatalogEntryPoints = []string{
	"../CLAUDE.md",
	"../README.md",
	"../docs/index.md",
	"../docs/getting-started/quickstart.md",
	"../docs/getting-started/installation.md",
	"../docs/getting-started/tutorial.md",
}

// TestDocExamplesDocumentTheCatalogEnvVar: each entry point a reader starts from names both
// forms that replace the embedded catalog, and the quickstart shows the one-time export, so
// the examples downstream run as written on the embedded copy and on a reader's own catalog.
// The examples omit the flag: BLIS_CATALOG states the location once.
func TestDocExamplesDocumentTheCatalogEnvVar(t *testing.T) {
	for _, path := range docCatalogEntryPoints {
		src, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v (docCatalogEntryPoints is stale)", path, err)
		}
		text := string(src)
		if !strings.Contains(text, catalogEnvVar) {
			t.Errorf("%s must name the %s environment variable, which replaces the embedded catalog", path, catalogEnvVar)
		}
		if !strings.Contains(text, "--catalog") {
			t.Errorf("%s must name --catalog, the explicit form that overrides %s", path, catalogEnvVar)
		}
		if strings.Contains(text, "no default and no search path") {
			t.Errorf("%s still says the catalog has no default: a run reads the embedded copy", path)
		}
	}
	quickstart, err := os.ReadFile("../docs/getting-started/quickstart.md")
	if err != nil {
		t.Fatalf("read quickstart: %v", err)
	}
	if !strings.Contains(string(quickstart), "export "+catalogEnvVar+"=") {
		t.Errorf("docs/getting-started/quickstart.md must show the optional `export %s=...`", catalogEnvVar)
	}
}

// catalogCompatibilityDoc is the page whose #catalog-compatibility anchor every entry point
// links: it holds the version table rendered from stack.json.
const catalogCompatibilityDoc = "../docs/getting-started/installation.md"

// retiredDataClone matches a documented clone of the two frozen data repositories. The
// catalog and registry are blis-substrate's (design §3.3), at the version stack.json names, so
// no page may tell a reader to clone blis-catalog or blis-registry.
var retiredDataClone = regexp.MustCompile(`git clone\b[^\n]*github\.com/inference-sim/blis-(catalog|registry)\b`)

// dataCheckoutBlock is the generated block that shows the substrate checkout (internal/stack).
const dataCheckoutBlock = "<!-- BEGIN GENERATED from stack.json: data-checkout -->"

// Every documented checkout of the data is the generated one: no page clones the frozen
// blis-catalog or blis-registry, and every entry point carries the data-checkout block that
// versions-doc renders from stack.json, so the version a reader checks out is the one the
// build pins (X23). versions-doc -check in CI holds the block's text to stack.json.
func TestDocExamplesCheckOutTheSubstrateTheBuildPins(t *testing.T) {
	for _, file := range collectDocFiles(t) {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		for _, command := range docLogicalLines(string(src)) {
			if retiredDataClone.MatchString(command.text) {
				t.Errorf("%s:%d: clones a frozen data repository; the catalog and registry are "+
					"blis-substrate's, checked out as the generated data-checkout block shows:\n  %s",
					file, command.num, collapse(command.text))
			}
		}
	}
	for _, path := range docCatalogEntryPoints {
		src, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v (docCatalogEntryPoints is stale)", path, err)
		}
		if !strings.Contains(string(src), dataCheckoutBlock) {
			t.Errorf("%s carries no generated data-checkout block, so a reader starting there is "+
				"not told which substrate commit to check out", path)
		}
	}
}

// Every entry point links the page that holds the rendered version table.
func TestDocCatalogEntryPointsLinkTheCompatibilityNote(t *testing.T) {
	for _, path := range docCatalogEntryPoints {
		if filepath.Clean(path) == filepath.Clean(catalogCompatibilityDoc) {
			continue
		}
		src, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v (docCatalogEntryPoints is stale)", path, err)
		}
		if !strings.Contains(string(src), "#catalog-compatibility") {
			t.Errorf("%s never links %s#catalog-compatibility, the rendered version table", path, catalogCompatibilityDoc)
		}
	}
}

// docLogicalLine is one shell command as a reader would run it: physical lines joined across
// trailing `\` continuations, with the 1-based line number it starts on.
type docLogicalLine struct {
	num  int
	text string
}

// docLogicalLines joins `\`-continued lines into one logical command each, so a clone wrapped
// across lines is still seen as one command.
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

// The matcher sees a wrapped clone as one command, and a clone of the substrate is not a
// retired one.
func TestRetiredDataCloneMatchesWrappedCommands(t *testing.T) {
	const url = "https://github.com/inference-sim/blis-catalog.git"
	wrapped := docLogicalLines("git clone --depth 1 \\\n  " + url + "\n")
	if len(wrapped) == 0 || !retiredDataClone.MatchString(wrapped[0].text) {
		t.Errorf("a wrapped clone of blis-catalog is not seen: %+v", wrapped)
	}
	if retiredDataClone.MatchString("git clone https://github.com/inference-sim/blis-substrate.git") {
		t.Error("a clone of blis-substrate is reported as a retired data clone")
	}
	if !retiredDataClone.MatchString("git clone --branch v0.2.0 https://github.com/inference-sim/blis-registry.git") {
		t.Error("a clone of blis-registry is not seen")
	}
}

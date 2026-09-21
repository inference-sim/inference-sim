package cmd

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// This file guards the comment and documentation sweep of #1795: the references that still
// described `defaults.yaml` blocks deleted by #1768 (`defaults:`), #1769 (`workloads:`) and
// #1770 (`kv_offload_devices:`), plus the one comment that mis-attributed the workload-preset
// funnel to loadPresetWorkload.
//
// None of them changed a simulation number, which is exactly why they survived three
// consecutive removals: nothing read them, and a wrong sentence about what a config file holds
// fails no test. So the sweep's third acceptance criterion — "a grep for the deleted-field
// references returns no stale hits" — is made executable here rather than left to a reviewer.
//
// The tests are of two deliberately different strengths:
//
//   - TestDefaultsFilepathHelpDescribesOnlyDeclaredSections derives the truth from the code.
//     It reflects over cmd.Config's yaml tags (the complete set of sections defaults.yaml may
//     carry, since KnownFields(true) makes any other key a hard load error, R10) and only bans
//     a retired section's prose aliases while that section is genuinely undeclared. A future
//     removal therefore tightens it automatically, and a deliberate reinstatement relaxes it,
//     with no rewrite either way — the shape TestDefaultsSchemaDocMatchesConfigStruct
//     (cmd/docs_stale_claims_test.go) recommends for exactly this reason.
//   - The other two pin a specific retired claim out of prose. That is weaker and more
//     brittle by nature, so each targets the FALSEHOOD's assertion form and carries a
//     non-vacuity anchor, so a renamed symbol or heading surfaces as a stale-test failure
//     rather than a silently vacuous pass.

// retiredDefaultsSections maps each top-level defaults.yaml section that has been deleted to
// the prose that used to describe it in the --defaults-filepath help and in the
// defaultsFilePath declaration comment. A section is only checked while cmd.Config does NOT
// declare it, so the guard tracks the struct rather than a frozen list.
var retiredDefaultsSections = map[string][]string{
	// #1768: per-model GPU / tensor_parallelism / hf_repo.
	"defaults": {"default specs", "default GPU", "deployment defaults"},
	// #1769: the four named workload presets, now <catalog>/workloads/<name>.yaml.
	"workloads": {"workloads", "workload preset"},
	// #1770: the KV-offload storage-device table, now <catalog>/devices/storage.yaml.
	"kv_offload_devices": {"device constants", "device defaults", "storage device"},
}

// TestDefaultsFilepathHelpDescribesOnlyDeclaredSections is BC-2. The --defaults-filepath help
// and the defaultsFilePath declaration comment are the two places that tell a reader what the
// file is for; each described sections the file has not carried since #1768/#1769/#1770.
func TestDefaultsFilepathHelpDescribesOnlyDeclaredSections(t *testing.T) {
	declared := configYAMLTags()
	if len(declared) == 0 {
		t.Fatal("non-vacuity: reflection over cmd.Config found no yaml-tagged fields")
	}

	// The flag help, read off the live command so the guard checks the string an operator
	// actually sees. INV-13: run and replay share registerSimConfigFlags, so checking both
	// pins that neither can drift into describing the file differently.
	descriptions := map[string]string{}
	for _, cmdName := range []string{"run", "replay"} {
		c := findCommandByPath(t, rootCmd, []string{cmdName})
		f := c.Flags().Lookup("defaults-filepath")
		if f == nil {
			t.Fatalf("non-vacuity: `blis %s` does not register --defaults-filepath", cmdName)
		}
		descriptions["`blis "+cmdName+"` --defaults-filepath help"] = f.Usage
	}
	descriptions["cmd/root.go defaultsFilePath declaration comment"] = defaultsFilePathDeclComment(t)

	checked := 0
	for where, text := range descriptions {
		if strings.TrimSpace(text) == "" {
			t.Errorf("non-vacuity: %s is empty", where)
			continue
		}
		lowered := strings.ToLower(text)
		for section, aliases := range retiredDefaultsSections {
			if declared[section] {
				// Reinstated on purpose: describing it is now correct, so skip it.
				continue
			}
			for _, alias := range aliases {
				checked++
				if strings.Contains(lowered, strings.ToLower(alias)) {
					t.Errorf("%s says %q, but cmd.Config declares no %q section — defaults.yaml "+
						"has not carried it since it was deleted, and strict parsing (R10) now "+
						"refuses a file that does. Full text: %q",
						where, alias, section, text)
				}
			}
		}
		// The positive half: it must still say what the file DOES hold, or the correction
		// could be an empty description that passes every ban above.
		if !strings.Contains(lowered, "coefficient") {
			t.Errorf("%s must still name the coefficients the file holds; got %q", where, text)
		}
	}
	if checked == 0 {
		t.Fatal("non-vacuity: no retired section was checked (are all three declared again?)")
	}
}

// defaultsFilePathDeclComment returns the trailing line comment on the defaultsFilePath
// package variable in cmd/root.go. Parsed rather than grepped so a moved declaration is
// followed rather than silently unchecked.
func defaultsFilePathDeclComment(t *testing.T) string {
	t.Helper()
	fset := token.NewFileSet()
	parsed, err := parser.ParseFile(fset, "root.go", nil, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse cmd/root.go: %v", err)
	}
	for _, decl := range parsed.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.VAR {
			continue
		}
		for _, spec := range gen.Specs {
			vs, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			for _, name := range vs.Names {
				if name.Name != "defaultsFilePath" {
					continue
				}
				var text string
				if vs.Comment != nil {
					text += vs.Comment.Text()
				}
				if vs.Doc != nil {
					text += vs.Doc.Text()
				}
				if strings.TrimSpace(text) == "" {
					t.Fatal("cmd/root.go declares defaultsFilePath with no comment: this guard " +
						"checks that comment, so an uncommented declaration is a silent pass")
				}
				return text
			}
		}
	}
	t.Fatal("cmd/root.go declares no defaultsFilePath variable (renamed? update this guard)")
	return ""
}

// TestPresetFunnelCommentNamesTheSharedReader is BC-1. loadPresetWorkload's comment claimed to
// be the single entry point shared by all three preset consumers, but observe resolves the
// catalog root itself and calls readCatalogPresetWorkload directly (cmd/observe_cmd.go,
// buildPresetSpec). The R23 property is real — it just belongs to readCatalogPresetWorkload.
func TestPresetFunnelCommentNamesTheSharedReader(t *testing.T) {
	// Behavioral anchor first, so the comment is checked against the real call graph rather
	// than against itself: observe must NOT route through loadPresetWorkload.
	observeCallsDirectly := false
	loadPresetCallers := map[string]bool{}
	fset := token.NewFileSet()
	for _, file := range []string{"observe_cmd.go", "convert.go", "root.go"} {
		parsed, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			t.Fatalf("parse cmd/%s: %v", file, err)
		}
		ast.Inspect(parsed, func(n ast.Node) bool {
			id, ok := n.(*ast.Ident)
			if !ok {
				return true
			}
			switch id.Name {
			case "readCatalogPresetWorkload":
				if file == "observe_cmd.go" {
					observeCallsDirectly = true
				}
			case "loadPresetWorkload":
				loadPresetCallers[file] = true
			}
			return true
		})
	}
	if !observeCallsDirectly {
		t.Fatal("non-vacuity: cmd/observe_cmd.go no longer names readCatalogPresetWorkload; the " +
			"comment this guard checks describes that call, so re-check both")
	}
	if loadPresetCallers["observe_cmd.go"] {
		t.Error("cmd/observe_cmd.go calls loadPresetWorkload: if observe now goes through the " +
			"catalog-locating wrapper, loadPresetWorkload's comment should say so again")
	}
	if !loadPresetCallers["convert.go"] || !loadPresetCallers["root.go"] {
		t.Errorf("non-vacuity: loadPresetWorkload should still be called by `blis convert preset` "+
			"(convert.go) and `blis run --workload` (root.go); found callers %v", loadPresetCallers)
	}

	// The comment half.
	src, err := os.ReadFile("catalog_workloads.go")
	if err != nil {
		t.Fatalf("read cmd/catalog_workloads.go: %v", err)
	}
	comment := docCommentFor(t, string(src), "func loadPresetWorkload(")
	if !strings.Contains(comment, "readCatalogPresetWorkload") {
		t.Errorf("loadPresetWorkload's doc comment must name readCatalogPresetWorkload as the "+
			"funnel the three preset consumers share; got:\n%s", comment)
	}
	// The falsehood's assertion form: this function being the single shared entry point.
	for _, claim := range []string{"single production entry point", "all three preset consumers —"} {
		if strings.Contains(comment, claim) {
			t.Errorf("loadPresetWorkload's doc comment still claims %q, but `blis observe "+
				"--workload` bypasses it (buildPresetSpec calls readCatalogPresetWorkload "+
				"directly); got:\n%s", claim, comment)
		}
	}
}

// docCommentFor returns the contiguous run of `//` lines immediately above the line that
// starts with the given declaration prefix.
func docCommentFor(t *testing.T, src, declPrefix string) string {
	t.Helper()
	lines := strings.Split(src, "\n")
	for i, line := range lines {
		if !strings.HasPrefix(line, declPrefix) {
			continue
		}
		start := i
		for start > 0 && strings.HasPrefix(strings.TrimSpace(lines[start-1]), "//") {
			start--
		}
		if start == i {
			t.Fatalf("%q has no doc comment (this guard checks it)", declPrefix)
		}
		return strings.Join(lines[start:i], "\n")
	}
	t.Fatalf("no declaration starting with %q found (renamed? update this guard)", declPrefix)
	return ""
}

// TestProjectStructureDefaultsAnnotationIsCurrent is BC-3. The file-tree annotation for
// defaults.yaml was doubly stale: "default GPU/TP/vLLM mappings" went with #1768 and "workload
// presets" with #1769. #1771 fixed the sibling cmd/default_config.go tree entry but not this
// top-level one.
func TestProjectStructureDefaultsAnnotationIsCurrent(t *testing.T) {
	const doc = "../docs/reference/project-structure.md"
	src, err := os.ReadFile(doc)
	if err != nil {
		t.Fatalf("read %s: %v", doc, err)
	}
	// The TOP-LEVEL defaults.yaml entry specifically: the tree also has a cmd/default_config.go
	// entry whose annotation names the retired blocks on purpose, to say they are gone. This
	// guard checks the line that asserts what the file DOES hold, so it matches on the entry
	// name rather than on the substring "defaults.yaml" anywhere.
	var annotation string
	for _, line := range strings.Split(string(src), "\n") {
		entry := strings.TrimLeft(line, "│├└─ ")
		if strings.HasPrefix(entry, "defaults.yaml") && strings.Contains(entry, "#") {
			annotation = line
			break
		}
	}
	if annotation == "" {
		t.Fatalf("non-vacuity: %s has no annotated top-level defaults.yaml tree entry", doc)
	}
	lowered := strings.ToLower(annotation)
	for _, stale := range []string{"gpu/tp", "vllm mapping", "workload preset"} {
		if strings.Contains(lowered, stale) {
			t.Errorf("%s still annotates defaults.yaml with %q: per-model GPU/TP defaults went "+
				"with #1768 and the workload presets with #1769. Got: %s", doc, stale, annotation)
		}
	}
	if !strings.Contains(lowered, "coefficient") {
		t.Errorf("%s must say defaults.yaml holds coefficients; got: %s", doc, annotation)
	}
}

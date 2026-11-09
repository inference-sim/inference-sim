package cmd

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

// Contracts for #1769: the named workload presets (chatbot, summarization, contentgen,
// multidoc) are read from the CATALOG — <catalog>/workloads/<name>.yaml, the sibling of the
// models/ namespace the #1774 clone-root contract settled — and the duplicate copy in the
// bundled defaults.yaml `workloads:` block is gone.
//
//	BC-1 the catalog file IS the preset source: its values reach the synthesized workload;
//	BC-2 INV-6 — the cutover moves no number: the bundled catalog's presets equal the values
//	     the deleted defaults.yaml block declared, and a preset-driven run is deterministic;
//	BC-3 one shared reader and one shared locator behind both preset consumers
//	     (`blis run --workload`, `blis convert preset`);
//	BC-4 an absent preset is refused naming the path looked at and the presets the catalog
//	     does define;
//	BC-5 strict parsing (R10): an unknown key or a missing token distribution is refused
//	     naming the file, never a silent zero;
//	BC-6 no catalog located is refused naming both --catalog and BLIS_CATALOG;
//	BC-7 --catalog is registered on every command that resolves a preset or a model;
//	BC-8 the retired duplicate cannot come back unnoticed.

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

// retiredDefaultsPresets is the four presets EXACTLY as the deleted defaults.yaml
// `workloads:` block declared them, transcribed from that block at commit 18458b38 (the
// last commit before this cutover). It is the golden for BC-2: the catalog copy must agree
// with it field for field, which is what makes the cutover a change of source rather than a
// change of numbers.
//
// Do not "fix" a value here to match a catalog edit — that would silently redefine the
// baseline this test exists to hold. A deliberate preset change belongs in the catalog with
// this golden updated in the same commit, and the reason recorded.
var retiredDefaultsPresets = map[string]presetWorkload{
	"chatbot": {
		PrefixTokens:     0,
		PromptTokensMean: 256, PromptTokensStdev: 100, PromptTokensMin: 2, PromptTokensMax: 800,
		OutputTokensMean: 256, OutputTokensStdev: 100, OutputTokensMin: 1, OutputTokensMax: 1024,
	},
	"contentgen": {
		PrefixTokens:     0,
		PromptTokensMean: 1024, PromptTokensStdev: 150, PromptTokensMin: 10, PromptTokensMax: 2048,
		OutputTokensMean: 1024, OutputTokensStdev: 200, OutputTokensMin: 10, OutputTokensMax: 2048,
	},
	"summarization": {
		PrefixTokens:     0,
		PromptTokensMean: 4096, PromptTokensStdev: 500, PromptTokensMin: 100, PromptTokensMax: 8192,
		OutputTokensMean: 512, OutputTokensStdev: 150, OutputTokensMin: 10, OutputTokensMax: 2048,
	},
	// multidoc declared no prefix_tokens key at all, so it parsed as 0.
	"multidoc": {
		PrefixTokens:     0,
		PromptTokensMean: 10240, PromptTokensStdev: 1200, PromptTokensMin: 500, PromptTokensMax: 20480,
		OutputTokensMean: 1536, OutputTokensStdev: 300, OutputTokensMin: 50, OutputTokensMax: 4096,
	},
}

// presetYAML renders a preset back to the catalog's YAML shape (blis-schemas workload.Shape).
func presetYAML(w presetWorkload) string {
	return fmt.Sprintf(`prefix_tokens: %d
prompt:
  tokens: %d
  tokens_stdev: %d
  tokens_min: %d
  tokens_max: %d
output:
  tokens: %d
  tokens_stdev: %d
  tokens_min: %d
  tokens_max: %d
`, w.PrefixTokens, w.PromptTokensMean, w.PromptTokensStdev, w.PromptTokensMin, w.PromptTokensMax,
		w.OutputTokensMean, w.OutputTokensStdev, w.OutputTokensMin, w.OutputTokensMax)
}

// writeTestPresetCatalog builds a catalog CLONE ROOT whose workloads/ namespace holds the
// given preset bodies, and returns the root.
func writeTestPresetCatalog(t *testing.T, presets map[string]string) string {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, catalogWorkloadsSubdir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	for name, body := range presets {
		if err := os.WriteFile(filepath.Join(dir, name+presetFileExt), []byte(body), 0o600); err != nil {
			t.Fatalf("write preset %q: %v", name, err)
		}
	}
	return root
}

// readPresetFrom reads a preset from the embedded catalog with the given presets as a catalog
// overlay, through the loader as a run reads them. An error from the loader (a file the
// contracts' strict decoding or validation refuses) is returned as the reader's would be.
func readPresetFrom(t *testing.T, presets map[string]string, name string) (*presetWorkload, error) {
	t.Helper()
	d, err := dataWithCatalogOverlay(writeTestPresetCatalog(t, presets))
	if err != nil {
		return nil, err
	}
	return readCatalogPresetWorkload(name, d)
}

// withTestCatalogPath points the production locator at the given catalog for the duration
// of a test: it sets the --catalog flag var, neutralizes any ambient BLIS_CATALOG, and drops the
// run's opened data, so the test exercises the real runData. The returned function restores
// the previous values.
func withTestCatalogPath(t *testing.T, catalog string) func() {
	t.Helper()
	prev, prevData := catalogPath, openedData
	catalogPath, openedData = catalog, nil
	t.Setenv(catalogEnvVar, "")
	return func() { catalogPath, openedData = prev, prevData }
}

// ---------------------------------------------------------------------------
// BC-2: the cutover moves no number
// ---------------------------------------------------------------------------

// TestCatalogPresets_BundledCatalogMatchesRetiredDefaults is BC-2. It reads each preset
// through the production reader out of the catalog of the blis-substrate
// module go.mod selects and compares every field to the
// values the deleted defaults.yaml block declared. Since the downstream path (PresetConfig →
// SynthesizeFromPreset) is untouched by this PR, equal inputs there mean an equal workload,
// which is the INV-6 argument for the cutover.
func TestCatalogPresets_BundledCatalogMatchesRetiredDefaults(t *testing.T) {
	if len(retiredDefaultsPresets) != 4 {
		t.Fatalf("non-vacuity: expected the 4 shipped presets in the golden, got %d", len(retiredDefaultsPresets))
	}
	for name, want := range retiredDefaultsPresets {
		if want.PromptTokensMean <= 0 || want.OutputTokensMean <= 0 {
			t.Fatalf("non-vacuity: golden preset %q has no token distribution", name)
		}
		got, err := readCatalogPresetWorkload(name, testData())
		if err != nil {
			t.Fatalf("preset %q must resolve from the test catalog: %v", name, err)
		}
		if *got != want {
			t.Errorf("preset %q drifted from the retired defaults.yaml values:\n  catalog: %+v\n  golden:  %+v",
				name, *got, want)
		}
	}
}

// ---------------------------------------------------------------------------
// BC-1: the catalog file is the preset source
// ---------------------------------------------------------------------------

// TestCatalogPresets_ReaderReadsCatalogFile is BC-1 at the reader: every field of a preset
// comes from <catalog>/workloads/<name>.yaml. The fixture values are deliberately unlike any
// shipped preset, so a reader that fell back to a built-in default would fail here.
func TestCatalogPresets_ReaderReadsCatalogFile(t *testing.T) {
	want := presetWorkload{
		PrefixTokens:     7,
		PromptTokensMean: 111, PromptTokensStdev: 22, PromptTokensMin: 3, PromptTokensMax: 999,
		OutputTokensMean: 55, OutputTokensStdev: 6, OutputTokensMin: 2, OutputTokensMax: 777,
	}
	got, err := readPresetFrom(t, map[string]string{"chatbot": presetYAML(want)}, "chatbot")
	if err != nil {
		t.Fatalf("readCatalogPresetWorkload: %v", err)
	}
	if *got != want {
		t.Errorf("preset fields must come from the catalog file:\n  got  %+v\n  want %+v", *got, want)
	}

	// And the same values must survive the one PresetConfig construction (R4).
	pc := got.toPresetConfig()
	if pc.PrefixTokens != want.PrefixTokens || pc.PromptTokensMean != want.PromptTokensMean ||
		pc.PromptTokensStdev != want.PromptTokensStdev || pc.PromptTokensMin != want.PromptTokensMin ||
		pc.PromptTokensMax != want.PromptTokensMax || pc.OutputTokensMean != want.OutputTokensMean ||
		pc.OutputTokensStdev != want.OutputTokensStdev || pc.OutputTokensMin != want.OutputTokensMin ||
		pc.OutputTokensMax != want.OutputTokensMax {
		t.Errorf("toPresetConfig dropped or reordered a field: %+v from %+v", pc, want)
	}
}

// ---------------------------------------------------------------------------
// BC-4 / BC-5: refusals name the path, and never fall back
// ---------------------------------------------------------------------------

// TestCatalogPresets_UnknownPreset_NamesPathAndAvailable is BC-4. The refusal has to be
// actionable in two ways: it names where the entry belongs, and it lists what the catalog in
// use does define — a hard-coded list would go stale the moment a catalog adds a preset.
func TestCatalogPresets_UnknownPreset_NamesPathAndAvailable(t *testing.T) {
	_, err := readCatalogPresetWorkload("nosuchpreset", testData())
	if err == nil {
		t.Fatal("an uncatalogued preset must be refused, never silently substituted")
	}
	msg := err.Error()
	if want := catalogWorkloadsSubdir + "/nosuchpreset" + presetFileExt; !strings.Contains(msg, want) {
		t.Errorf("refusal must name where the entry belongs (%s), got: %q", want, msg)
	}
	for name := range retiredDefaultsPresets {
		if !strings.Contains(msg, name) {
			t.Errorf("refusal must list the available preset %q, got: %q", name, msg)
		}
	}
}

// TestCatalogPresets_StrictParsing is BC-5. A preset file is decoded with KnownFields(true)
// (R10), and a definition with no token distribution is refused when the catalog opens — the
// alternative is a plausible-but-wrong zero-token workload with no diagnostic anywhere.
func TestCatalogPresets_StrictParsing(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string // substring the refusal must contain
	}{
		{
			name: "unknown key",
			body: presetYAML(retiredDefaultsPresets["chatbot"]) + "prompt_tokens_average: 512\n",
			want: "prompt_tokens_average",
		},
		{
			name: "misspelled key that would have dropped to zero",
			body: "prompt: {tokens: 256}\noutput: {tokens: 256, tokens_stdevv: 100}\n",
			want: "tokens_stdevv",
		},
		// The catalog loader refuses these when the data opens, before a run reads the preset, and
		// names the file and the key.
		{name: "empty file", body: "", want: "workloads/chatbot.yaml"},
		{name: "zero prompt tokens", body: "prompt: {tokens: 0}\noutput: {tokens: 256}\n", want: "prompt.tokens: must be positive"},
		{name: "negative output tokens", body: "prompt: {tokens: 256}\noutput: {tokens: -1}\n", want: "output.tokens: must be positive"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := readPresetFrom(t, map[string]string{"chatbot": tc.body}, "chatbot")
			if err == nil {
				t.Fatal("a malformed preset definition must be refused, not silently zeroed")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("refusal must name %q, got: %q", tc.want, err.Error())
			}
			// Every refusal names the offending preset, so the fix is obvious.
			if !strings.Contains(err.Error(), "chatbot") {
				t.Errorf("refusal must name the preset, got: %q", err.Error())
			}
		})
	}
}

// ---------------------------------------------------------------------------
// #1793: min/max/stdev distribution-bound validation on presets
// ---------------------------------------------------------------------------

// TestCatalogPresets_DistributionBoundsValidated is the #1793 contract. Before this, the
// preset reader validated only that the token MEANS were > 0, so a malformed preset —
// min > max, a mean outside [min, max], a negative stdev, or an omitted-and-therefore-zero
// min/max — was accepted and the Gaussian sampler then silently clamped it to a wrong
// distribution. The reader now applies the SAME bound validation the CLI distribution path
// applies (validateDistributionParams, root.go), refusing the preset naming the file (R1). A
// valid preset still loads unchanged (INV-6).
//
// It is also a LAW test: over well-formed preset YAML the reader must refuse a preset iff
// validateDistributionParams refuses the same bounds, so the preset path and the
// --prompt-tokens-* / --output-tokens-* CLI path cannot drift.
func TestCatalogPresets_DistributionBoundsValidated(t *testing.T) {
	// A fully-valid chatbot baseline (prompt 256 in [2,800] stdev 100; output 256 in
	// [1,1024] stdev 100); each case mutates one aspect of it.
	base := retiredDefaultsPresets["chatbot"]
	if msg := validateDistributionParams(base.PromptTokensMin, base.PromptTokensMax,
		base.OutputTokensMin, base.OutputTokensMax, base.PromptTokensStdev, base.OutputTokensStdev,
		base.PromptTokensMean, base.OutputTokensMean); msg != "" {
		t.Fatalf("non-vacuity: the baseline preset must itself be valid, got %q", msg)
	}

	tests := []struct {
		name    string
		mutate  func(w *presetWorkload)
		refused bool
	}{
		{name: "valid full preset accepted", mutate: func(*presetWorkload) {}, refused: false},
		{name: "constant preset (min=max=mean, stdev 0) accepted", mutate: func(w *presetWorkload) {
			w.PromptTokensMean, w.PromptTokensStdev, w.PromptTokensMin, w.PromptTokensMax = 64, 0, 64, 64
			w.OutputTokensMean, w.OutputTokensStdev, w.OutputTokensMin, w.OutputTokensMax = 16, 0, 16, 16
		}, refused: false},
		{name: "prompt min greater than max is refused", mutate: func(w *presetWorkload) {
			w.PromptTokensMin, w.PromptTokensMax = 800, 2
		}, refused: true},
		{name: "prompt mean above max is refused", mutate: func(w *presetWorkload) {
			w.PromptTokensMean = 900 // > max 800
		}, refused: true},
		{name: "output mean below min is refused", mutate: func(w *presetWorkload) {
			w.OutputTokensMin, w.OutputTokensMean = 10, 5
		}, refused: true},
		{name: "negative prompt stdev is refused", mutate: func(w *presetWorkload) {
			w.PromptTokensStdev = -1
		}, refused: true},
		{name: "negative output stdev is refused", mutate: func(w *presetWorkload) {
			w.OutputTokensStdev = -1
		}, refused: true},
		{name: "prompt stdev above max is refused", mutate: func(w *presetWorkload) {
			w.PromptTokensStdev = 900 // > max 800
		}, refused: true},
		{name: "omitted prompt min and max (zero) is refused", mutate: func(w *presetWorkload) {
			// The exact silently-wrong case from the issue: a mean-only preset whose
			// GaussianSampler (min==max==0) returns a zero-token distribution.
			w.PromptTokensMin, w.PromptTokensMax = 0, 0
		}, refused: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			w := base
			tc.mutate(&w)
			_, err := readPresetFrom(t, map[string]string{"chatbot": presetYAML(w)}, "chatbot")

			if tc.refused {
				if err == nil {
					t.Fatalf("a preset with invalid distribution bounds must be refused, not silently "+
						"clamped by the sampler: %+v", w)
				}
				// Actionable (R1): the refusal names the offending preset.
				if !strings.Contains(err.Error(), "chatbot") {
					t.Errorf("refusal must name the preset, got: %q", err.Error())
				}
			} else if err != nil {
				t.Fatalf("a valid preset must still load unchanged (INV-6), got refusal: %v", err)
			}

			// Law: the reader refuses a well-formed preset iff validateDistributionParams
			// refuses the same bounds, so the preset and CLI-distribution paths cannot drift.
			cliRefused := validateDistributionParams(
				w.PromptTokensMin, w.PromptTokensMax, w.OutputTokensMin, w.OutputTokensMax,
				w.PromptTokensStdev, w.OutputTokensStdev, w.PromptTokensMean, w.OutputTokensMean,
			) != ""
			if (err != nil) != cliRefused {
				t.Errorf("preset/CLI validation drift: reader refused=%v, validateDistributionParams refused=%v (bounds %+v)",
					err != nil, cliRefused, w)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// BC-6: no catalog located
// ---------------------------------------------------------------------------

// TestCatalogPresets_NoCatalogLocated_ReadsTheEmbeddedCopy is BC-6 under design §3.3: with
// neither --catalog nor BLIS_CATALOG a preset comes from the catalog embedded in the pinned
// substrate release, never from a search path.
func TestCatalogPresets_NoCatalogLocated_ReadsTheEmbeddedCopy(t *testing.T) {
	restore := withTestCatalogPath(t, "")
	defer restore()

	wl, err := loadPresetWorkload("chatbot")
	if err != nil {
		t.Fatalf("with no catalog named, the embedded catalog's chatbot must load: %v", err)
	}
	if *wl != retiredDefaultsPresets["chatbot"] {
		t.Errorf("embedded chatbot: %+v, want %+v", *wl, retiredDefaultsPresets["chatbot"])
	}
}

// TestCatalogPresets_LocatorHonorsEnvVar shows the preset reader inherits the whole locator
// law: BLIS_CATALOG alone replaces the embedded catalog, and --catalog wins when both are set.
func TestCatalogPresets_LocatorHonorsEnvVar(t *testing.T) {
	// Two whole catalogs, each the test catalog with its own chatbot (prompt mean 100 or 200).
	catalogWith := func(promptMean int) string {
		root := copyKernelCatalog(t)
		body := presetYAML(presetWorkload{
			PromptTokensMean: promptMean, PromptTokensStdev: 10, PromptTokensMin: 1, PromptTokensMax: 800,
			OutputTokensMean: 10, OutputTokensStdev: 2, OutputTokensMin: 1, OutputTokensMax: 100,
		})
		if err := os.WriteFile(filepath.Join(root, catalogWorkloadsSubdir, "chatbot"+presetFileExt), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		return root
	}
	viaEnv, viaFlag := catalogWith(100), catalogWith(200)

	restore := withTestCatalogPath(t, "")
	defer restore()

	t.Setenv(catalogEnvVar, viaEnv)
	openedData = nil
	wl, err := loadPresetWorkload("chatbot")
	if err != nil {
		t.Fatalf("%s alone must locate the catalog: %v", catalogEnvVar, err)
	}
	if wl.PromptTokensMean != 100 {
		t.Errorf("preset read via %s: prompt_tokens=%d, want 100", catalogEnvVar, wl.PromptTokensMean)
	}

	catalogPath, openedData = viaFlag, nil
	wl, err = loadPresetWorkload("chatbot")
	if err != nil {
		t.Fatalf("--catalog must locate the catalog: %v", err)
	}
	if wl.PromptTokensMean != 200 {
		t.Errorf("--catalog must win over %s: prompt_tokens=%d, want 200", catalogEnvVar, wl.PromptTokensMean)
	}
}

// ---------------------------------------------------------------------------
// BC-7: the locator is registered wherever it is needed
// ---------------------------------------------------------------------------

// TestCatalogPresets_CatalogFlagOnEveryConsumer is BC-7. It walks the real cobra tree, so a
// command that resolves a preset (or a model config) without accepting the locator fails
// here rather than at an operator's first invocation.
func TestCatalogPresets_CatalogFlagOnEveryConsumer(t *testing.T) {
	for _, path := range [][]string{{"run"}, {"replay"}, {"convert", "preset"}} {
		cmd := findCommandByPath(t, rootCmd, path)
		if f := cmd.Flags().Lookup("catalog"); f == nil {
			t.Errorf("`blis %s` must accept --catalog: it resolves a model config or a workload preset from the catalog",
				strings.Join(path, " "))
		} else if f.DefValue != "" {
			t.Errorf("`blis %s` --catalog default: got %q, want empty (the embedded copy)",
				strings.Join(path, " "), f.DefValue)
		}
	}
}

// findCommandByPath resolves a cobra command by its subcommand path.
func findCommandByPath(t *testing.T, root *cobra.Command, path []string) *cobra.Command {
	t.Helper()
	cur := root
	for _, name := range path {
		var next *cobra.Command
		for _, c := range cur.Commands() {
			if c.Name() == name {
				next = c
				break
			}
		}
		if next == nil {
			t.Fatalf("command %q not found under %q", name, cur.Name())
		}
		cur = next
	}
	return cur
}

// ---------------------------------------------------------------------------
// BC-8: the duplicate cannot come back
// ---------------------------------------------------------------------------

// TestCatalogPresets_DuplicateSourceIsGone is BC-8. A behavioral test can only show that
// today's inputs do not read the defaults.yaml copy; this shows there is nothing left to
// read — no `workloads:` key in the shipped file, and no Go surface that could parse one.
func TestCatalogPresets_DuplicateSourceIsGone(t *testing.T) {
	bannedIdents := map[string]string{
		"Workloads": "Config.Workloads (the defaults.yaml preset map) was removed by #1769",
		"Workload":  "the defaults.yaml Workload struct was removed by #1769 — the catalog shape is presetWorkload",
	}
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob cmd/*.go: %v", err)
	}
	scanned := 0
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		scanned++
		fset := token.NewFileSet()
		parsed, parseErr := parser.ParseFile(fset, file, nil, parser.ParseComments)
		if parseErr != nil {
			t.Fatalf("parse %s: %v", file, parseErr)
		}
		// A name another package exports (blis-schemas' acceptance.Workload, the Workload
		// field of its acceptance.Request literal) is not cmd/'s surface: identifiers selected
		// from an imported package, and the field keys of that package's composite literals,
		// are skipped. Every other identifier is checked.
		imported := map[string]bool{}
		for _, imp := range parsed.Imports {
			name := filepath.Base(strings.Trim(imp.Path.Value, `"`))
			if imp.Name != nil {
				name = imp.Name.Name
			}
			imported[name] = true
		}
		qualified := func(e ast.Expr) bool {
			sel, ok := e.(*ast.SelectorExpr)
			if !ok {
				return false
			}
			x, ok := sel.X.(*ast.Ident)
			return ok && imported[x.Name]
		}
		skip := map[*ast.Ident]bool{}
		// selectorOf marks the selector identifiers that are called as methods.
		selectorOf := map[*ast.Ident]bool{}
		ast.Inspect(parsed, func(n ast.Node) bool {
			if c, ok := n.(*ast.CallExpr); ok {
				if sel, ok := c.Fun.(*ast.SelectorExpr); ok {
					selectorOf[sel.Sel] = true
				}
			}
			return true
		})
		ast.Inspect(parsed, func(n ast.Node) bool {
			switch v := n.(type) {
			case *ast.SelectorExpr:
				if qualified(v) {
					skip[v.Sel] = true
				}
			case *ast.CompositeLit:
				if qualified(v.Type) {
					for _, el := range v.Elts {
						if kv, ok := el.(*ast.KeyValueExpr); ok {
							if key, ok := kv.Key.(*ast.Ident); ok {
								skip[key] = true
							}
						}
					}
				}
			}
			return true
		})
		ast.Inspect(parsed, func(n ast.Node) bool {
			id, ok := n.(*ast.Ident)
			if !ok || skip[id] {
				return true
			}
			// A method call on a value (data.Workload(name), blis-substrate's catalog reader) is
			// not the retired type: the ban is on the identifier as a type or a field.
			if call, isSel := selectorOf[id]; isSel && call {
				return true
			}
			if why, banned := bannedIdents[id.Name]; banned {
				t.Errorf("%s:%d: %s must not appear in cmd/'s production sources — %s",
					file, fset.Position(id.Pos()).Line, id.Name, why)
			}
			return true
		})
	}
	if scanned == 0 {
		t.Fatal("non-vacuity: no production sources were scanned")
	}

	// The YAML half: the shipped defaults.yaml declares no `workloads:` block. Match a
	// declaration only, so the file's explanatory comment (which names the retired key on
	// purpose) does not trip the guard.
	data, err := os.ReadFile("../defaults.yaml")
	if err != nil {
		t.Fatalf("read defaults.yaml: %v", err)
	}
	if regexp.MustCompile(`(?m)^workloads:`).Match(data) {
		t.Error("defaults.yaml must not declare a `workloads:` block: the presets live in the " +
			"catalog (<catalog>/workloads/<name>.yaml, #1769), and a second copy has nothing " +
			"keeping it in sync")
	}
}

// TestCatalogPresets_StaleWorkloadsBlockIsRefused is the behavioral half of BC-8, and the
// one observable behavior change beyond the source of the presets: a hand-maintained
// defaults.yaml that still carries a `workloads:` block no longer loads. That is the
// intended outcome — re-declaring the field so the block parses would leave a config key
// with no consumer, exactly the silent-acceptance antipattern strict parsing (R10) exists to
// prevent.
//
// loadDefaultsConfig reports a parse error via logrus.Fatalf, so this runs in a subprocess.
func TestCatalogPresets_StaleWorkloadsBlockIsRefused(t *testing.T) {
	if os.Getenv("BLIS_STALE_WORKLOADS_SUBPROCESS") == "1" {
		// The block is the ONLY difference from a file that loads (see the negative control
		// below), so a failure can only be the `workloads` key.
		content := "workloads:\n  chatbot:\n    prompt_tokens: 256\n    output_tokens: 256\nversion: \"0.0.1\"\n"
		path := filepath.Join(t.TempDir(), "defaults.yaml")
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			os.Exit(2)
		}
		loadDefaultsConfig(path) // must Fatalf before returning
		os.Exit(0)               // reached only if the stale block was accepted
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestCatalogPresets_StaleWorkloadsBlockIsRefused$", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_STALE_WORKLOADS_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("a defaults.yaml carrying a stale `workloads:` block must be refused at load, "+
			"not silently ignored; output:\n%s", out)
	}
	if !strings.Contains(string(out), "workloads") {
		t.Errorf("the refusal must name the offending `workloads` field; output:\n%s", out)
	}

	// Negative control: the same fixture minus the block loads, so the rejection is
	// attributable to that key and nothing else about the fixture.
	path := filepath.Join(t.TempDir(), "defaults.yaml")
	if writeErr := os.WriteFile(path, []byte("version: \"0.0.1\"\n"), 0o644); writeErr != nil {
		t.Fatal(writeErr)
	}
	if cfg := loadDefaultsConfig(path); cfg.Version != "0.0.1" {
		t.Errorf("control fixture must load: got version %q, want \"0.0.1\"", cfg.Version)
	}
}

// TestCatalogPresets_RetiredLocatorFlagsAreGone pins the other half of the cutover: the
// command that only ever read defaults.yaml for its preset block no longer accepts
// --defaults-filepath. A flag whose sole consumer was deleted is a config key with no
// consumer (#1768's lesson), and leaving it would silently accept a path nothing reads.
func TestCatalogPresets_RetiredLocatorFlagsAreGone(t *testing.T) {
	for _, path := range [][]string{{"convert", "preset"}} {
		cmd := findCommandByPath(t, rootCmd, path)
		if f := cmd.Flags().Lookup("defaults-filepath"); f != nil {
			t.Errorf("`blis %s` must not accept --defaults-filepath: its only consumer was the "+
				"retired defaults.yaml `workloads:` block (#1769)", strings.Join(path, " "))
		}
	}
	// Non-vacuity / scope: run and replay DO still read defaults.yaml (trained coefficients,
	// LoRA and KV-offload device constants), so the flag must survive there.
	for _, path := range [][]string{{"run"}, {"replay"}} {
		cmd := findCommandByPath(t, rootCmd, path)
		if f := cmd.Flags().Lookup("defaults-filepath"); f == nil {
			t.Errorf("`blis %s` must keep --defaults-filepath: it reads the trained coefficients "+
				"and device constants from that file", strings.Join(path, " "))
		}
	}
}

// ---------------------------------------------------------------------------
// BC-1 / BC-2 at the CLI boundary
// ---------------------------------------------------------------------------

const (
	presetRunLegEnv     = "BLIS_PRESET_RUN_LEG"
	presetRunCatalogEnv = "BLIS_PRESET_RUN_CATALOG"
)

// newPresetRunCatalog builds a catalog holding everything a preset-driven kernel `blis run`
// needs -- a copy of the vendored catalog, so the scenario's model graph, hardware and fabric
// resolve -- with workloads/chatbot.yaml carrying the given prompt/output means.
func newPresetRunCatalog(t *testing.T, promptMean, outputMean int) string {
	t.Helper()
	root := copyKernelCatalog(t)

	preset := retiredDefaultsPresets["chatbot"]
	preset.PromptTokensMean = promptMean
	preset.OutputTokensMean = outputMean
	workloadsDir := filepath.Join(root, catalogWorkloadsSubdir)
	if err := os.MkdirAll(workloadsDir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", workloadsDir, err)
	}
	if err := os.WriteFile(filepath.Join(workloadsDir, "chatbot"+presetFileExt), []byte(presetYAML(preset)), 0o600); err != nil {
		t.Fatalf("write catalog preset: %v", err)
	}
	return root
}

// runPresetLeg re-execs this test binary as a real `blis run --workload chatbot` against the
// given catalog and returns its stdout.
func runPresetLeg(t *testing.T, catalog string) string {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestCatalogPresets_RunCLI_PresetComesFromCatalog$")
	cmd.Env = append(os.Environ(),
		presetRunLegEnv+"=1",
		presetRunCatalogEnv+"="+catalog,
		// Neutralize any ambient BLIS_CATALOG so the leg reads the catalog it was handed.
		catalogEnvVar+"=",
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("preset run leg (catalog %s) failed: %v\nstdout:\n%s\nstderr:\n%s",
			catalog, err, stdout.String(), stderr.String())
	}
	return stdout.String()
}

// TestCatalogPresets_RunCLI_PresetComesFromCatalog is BC-1 and BC-2 at the CLI boundary. It
// asserts two things a unit test on the reader cannot:
//
//	determinism — the same catalog preset produces byte-identical stdout across runs
//	              (INV-6, the property the cutover had to preserve);
//	reach       — a catalog whose preset declares different token means produces DIFFERENT
//	              stdout, so the catalog file demonstrably drives the simulation rather than
//	              being read and discarded in favour of some other default.
//
// The second half is what makes the first non-vacuous: byte-identity alone would also hold
// if the preset were ignored entirely.
func TestCatalogPresets_RunCLI_PresetComesFromCatalog(t *testing.T) {
	if os.Getenv(presetRunLegEnv) == "1" {
		rootCmd.SetArgs([]string{
			"run", "--scenario", kernelTestScenario, "--scenarios", testScenarios, "--registry", testRegistryDir(),
			"--workload", "chatbot", "--rate", "5", "--num-requests", "20",
			"--seed", "42", "--horizon", "600000000",
			"--defaults-filepath", "../defaults.yaml",
			"--catalog", os.Getenv(presetRunCatalogEnv),
		})
		if err := rootCmd.Execute(); err != nil {
			os.Exit(1)
		}
		os.Exit(0)
	}

	base := newPresetRunCatalog(t, 256, 256)
	first := runPresetLeg(t, base)
	second := runPresetLeg(t, base)
	if first == "" {
		t.Fatal("non-vacuity: a preset-driven run produced no stdout")
	}
	if withoutRuntime(t, first) != withoutRuntime(t, second) {
		t.Errorf("a preset-driven run must be byte-identical outside runtime across runs (INV-6):\n--- first ---\n%s\n--- second ---\n%s",
			first, second)
	}

	// A different preset definition in the catalog must move the numbers. The shifted means
	// stay within chatbot's [min,max] bounds (prompt [2,800], output [1,1024]) so the preset
	// passes the #1793 distribution-bound validation; 512 vs 256 still moves the simulation.
	shifted := runPresetLeg(t, newPresetRunCatalog(t, 512, 512))
	if shifted == first {
		t.Error("changing the catalog preset's token means left stdout unchanged: the preset " +
			"definition in <catalog>/workloads/chatbot.yaml is not reaching the simulation")
	}
}

// TestRemovedCommandsAreGone pins the v0.16.0 removal of blis observe and blis calibrate
// (#1901): neither is a subcommand. run, replay and convert are, so the walk is not vacuous.
func TestRemovedCommandsAreGone(t *testing.T) {
	names := map[string]bool{}
	for _, c := range rootCmd.Commands() {
		names[c.Name()] = true
	}
	for _, gone := range []string{"observe", "calibrate"} {
		if names[gone] {
			t.Errorf("`blis %s` was removed in v0.16.0 (#1901) and must not be registered", gone)
		}
	}
	for _, kept := range []string{"run", "replay", "convert"} {
		if !names[kept] {
			t.Errorf("`blis %s` must be registered", kept)
		}
	}
}

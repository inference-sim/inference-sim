package cmd

import (
	"os"
	"reflect"
	"regexp"
	"strings"
	"testing"
)

// This file guards the documentation claims corrected by #1775. Each of them described
// pre-NS-6 (pre-#1733/#1731) behavior that no longer exists, and none of them changed a
// simulation number — which is exactly why they survived so long: no test read them, and a
// wrong sentence about how BLIS resolves a deployment or a coefficient fails nothing.
//
// The tests below are deliberately of two different strengths, and the difference matters:
//
//   - TestDefaultsSchemaDocMatchesConfigStruct derives the truth from the code (reflection
//     over cmd.Config's yaml tags) and compares the documentation against it. It keeps
//     working as sections are added and removed — #1769/#1770 are expected to remove blocks
//     from this very schema — because it asserts agreement rather than a fixed list.
//   - The remaining tests pin a specific retired claim out of the prose. They are weaker and
//     more brittle by nature, so each one targets the FALSEHOOD's assertion form, not a
//     phrase that also occurs in its correction (the corrected text says "not keyed by
//     model, GPU or TP", so a bare "keyed by model" substring check would fail on the fix).

const configurationDoc = "../docs/reference/configuration.md"

// docSection returns the body of the `## <heading>` section of a markdown document: from the
// heading to the next same-level heading, exclusive. Fails the test when absent, so a renamed
// heading surfaces as a stale-test error rather than a silently vacuous pass.
func docSection(t *testing.T, path, heading string) string {
	t.Helper()
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	lines := strings.Split(string(src), "\n")
	start := -1
	for i, line := range lines {
		if strings.TrimSpace(line) == "## "+heading {
			start = i + 1
			break
		}
	}
	if start < 0 {
		t.Fatalf("%s has no `## %s` section (heading renamed? update this test)", path, heading)
	}
	for i := start; i < len(lines); i++ {
		if strings.HasPrefix(lines[i], "## ") {
			return strings.Join(lines[start:i], "\n")
		}
	}
	return strings.Join(lines[start:], "\n")
}

// fencedYAMLTopLevelKeys collects the column-0 `key:` names from every ```yaml block in the
// given markdown body — i.e. the top-level sections a reader would copy into their own file.
func fencedYAMLTopLevelKeys(body string) []string {
	topLevel := regexp.MustCompile(`^([a-zA-Z_][a-zA-Z0-9_]*):`)
	var keys []string
	inFence := false
	for _, line := range strings.Split(body, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			// Only ```yaml fences describe the file; a ```bash fence is a command.
			inFence = !inFence && strings.Contains(trimmed, "yaml")
			continue
		}
		if !inFence {
			continue
		}
		if m := topLevel.FindStringSubmatch(line); m != nil {
			keys = append(keys, m[1])
		}
	}
	return keys
}

// configYAMLTags returns the top-level yaml keys cmd.Config actually declares. Strict parsing
// (KnownFields(true), R10) means this is the complete set of keys defaults.yaml may carry: any
// other key is a hard load error, so documenting one is documenting an impossibility.
func configYAMLTags() map[string]bool {
	tags := map[string]bool{}
	ct := reflect.TypeOf(Config{})
	for i := 0; i < ct.NumField(); i++ {
		tag := ct.Field(i).Tag.Get("yaml")
		if tag == "" || tag == "-" {
			continue
		}
		tags[strings.Split(tag, ",")[0]] = true
	}
	return tags
}

// TestDefaultsSchemaDocMatchesConfigStruct is the substantive #1775 contract: the documented
// defaults.yaml schema must describe the sections cmd.Config declares, and no others.
//
// The retired claim it kills: the schema block showed a `models:` list mapping a
// model+GPU+TP triple to its own alpha/beta coefficients. cmd.Config has no Models field, so
// strict parsing would have rejected such a file outright — the documented example could not
// be loaded by the binary it documented. Coefficients are one global block
// (trained_physics_coefficients), never keyed by deployment.
//
// Written as an agreement check rather than a fixed list on purpose: #1769/#1770 are expected
// to remove blocks from this schema, and this test should then fail on the DOC being stale
// rather than have to be rewritten alongside every removal.
func TestDefaultsSchemaDocMatchesConfigStruct(t *testing.T) {
	declared := configYAMLTags()
	if len(declared) == 0 {
		t.Fatal("non-vacuity: reflection over cmd.Config found no yaml-tagged fields")
	}

	section := docSection(t, configurationDoc, "defaults.yaml")
	documented := fencedYAMLTopLevelKeys(section)
	if len(documented) == 0 {
		t.Fatal("non-vacuity: the `## defaults.yaml` section shows no ```yaml block with top-level keys")
	}

	// (a) Nothing fictional: every key the schema shows must be loadable.
	for _, key := range documented {
		if !declared[key] {
			t.Errorf("the documented defaults.yaml schema shows a top-level %q section, but cmd.Config "+
				"declares no such yaml key — strict parsing (KnownFields(true), R10) would reject a file "+
				"carrying it. Remove it from the schema, or add the field if it is meant to exist.", key)
		}
	}

	// (b) Nothing missing: a section that exists but goes undocumented is the mirror defect.
	shown := map[string]bool{}
	for _, key := range documented {
		shown[key] = true
	}
	for key := range declared {
		if !shown[key] {
			t.Errorf("cmd.Config declares the defaults.yaml key %q but the documented schema in %s "+
				"does not show it; a reader cannot discover a section that is only in the struct",
				key, configurationDoc)
		}
	}
}

// TestCoefficientResolutionDocDescribesGlobalBlock pins the corrected coefficient-resolution
// claim. The retired one said BLIS "automatically loads pre-trained coefficients from
// defaults.yaml based on the model, GPU, and TP configuration" — describing a keyed selection
// that has no implementation. resolveLatencyConfig reads one global
// trained_physics_coefficients block, and only on the trained-physics branch.
func TestCoefficientResolutionDocDescribesGlobalBlock(t *testing.T) {
	src, err := os.ReadFile(configurationDoc)
	if err != nil {
		t.Fatalf("read %s: %v", configurationDoc, err)
	}
	text := string(src)

	if !strings.Contains(text, "trained_physics_coefficients") {
		t.Errorf("%s must name `trained_physics_coefficients` — the actual defaults.yaml key the "+
			"coefficients are read from — when describing coefficient resolution", configurationDoc)
	}

	// The falsehood's assertion form: coefficients SELECTED BY the deployment. Matched as a
	// claim, so the correction's own "not keyed by model, GPU or TP" does not trip it.
	keyedClaims := []*regexp.Regexp{
		// "...coefficients ... based on the model, GPU, and TP configuration". Kept on one
		// line and within 120 characters so it reads as a claim about coefficient selection,
		// but otherwise permissive: the retired sentence had a backticked `defaults.yaml`
		// between the two halves, so a class excluding `.` never matched it.
		regexp.MustCompile(`(?i)coefficients[^\n]{0,120}based on the model`),
		regexp.MustCompile(`(?i)coefficients\s*\(keyed\s+by`),
	}
	for _, re := range keyedClaims {
		if loc := re.FindStringIndex(text); loc != nil {
			t.Errorf("%s claims coefficients are selected per model/GPU/TP (%q). No such keyed lookup "+
				"exists: resolveLatencyConfig reads the single global trained_physics_coefficients block",
				configurationDoc, text[loc[0]:loc[1]])
		}
	}
}

// TestMetricsPathDocumentedOnRunAndReplay pairs a behavioral anchor with a doc check, so the
// doc claim cannot be "corrected" into a second falsehood if the flag registration changes.
//
// Retired claim: `--metrics-path` is "blis run only — blis replay uses --results-path
// instead". #1583 put --metrics-path on replay (that is how `blis calibrate --sim-metrics`
// gets a replayed cache_hit_rate); --results-path is the replay-only PER-REQUEST path, a
// complement rather than a substitute.
func TestMetricsPathDocumentedOnRunAndReplay(t *testing.T) {
	// Behavioral anchor: the documented shape is the real one.
	if runCmd.Flags().Lookup("metrics-path") == nil {
		t.Fatal("blis run must register --metrics-path")
	}
	if replayCmd.Flags().Lookup("metrics-path") == nil {
		t.Fatal("blis replay must register --metrics-path (#1583)")
	}
	if runCmd.Flags().Lookup("results-path") != nil {
		t.Error("blis run must NOT register --results-path; the docs describe it as replay-only")
	}
	if replayCmd.Flags().Lookup("results-path") == nil {
		t.Fatal("blis replay must register --results-path")
	}

	src, err := os.ReadFile(configurationDoc)
	if err != nil {
		t.Fatalf("read %s: %v", configurationDoc, err)
	}
	text := string(src)

	runOnly := []*regexp.Regexp{
		// Stays on one line but crosses `|` freely: in the flag table the name and the
		// description are in DIFFERENT cells, so a class excluding the pipe could never see
		// the retired "blis run only" claim at all. Backticks optional — "`blis run` only"
		// is the same claim.
		regexp.MustCompile("(?i)--metrics-path[^\n]{0,200}`?blis run`? only"),
		regexp.MustCompile("(?i)--metrics-path`?\\s*\\(run only\\)"),
	}
	for _, re := range runOnly {
		if loc := re.FindStringIndex(text); loc != nil {
			t.Errorf("%s describes --metrics-path as run-only (%q), but blis replay registers it too "+
				"(#1583)", configurationDoc, text[loc[0]:loc[1]])
		}
	}
}

// TestResolveLatencyConfigSideEffectsDocComment: resolveLatencyConfig's doc comment
// enumerates the package-level vars it mutates, and a caller relies on that list to know what
// is available afterwards. resolvedCatalogRoot has been one since #1732 — set transitively by
// resolveModelConfig, and read back at the EmitOutput sites as catalog provenance — but the
// list did not mention it, so the one thing the list exists to tell you was missing.
func TestResolveLatencyConfigSideEffectsDocComment(t *testing.T) {
	src, err := os.ReadFile("root.go")
	if err != nil {
		t.Fatalf("read root.go: %v", err)
	}
	text := string(src)

	const marker = "// Side effects (package-level vars mutated):"
	idx := strings.Index(text, marker)
	if idx < 0 {
		t.Fatalf("root.go no longer carries %q above resolveLatencyConfig (comment restructured? "+
			"update this test)", marker)
	}
	end := strings.Index(text[idx:], "func resolveLatencyConfig(")
	if end < 0 {
		t.Fatal("could not find resolveLatencyConfig after its side-effect comment")
	}

	// Collect ONLY the godoc indented-block lines (`//\t...`) that form the enumeration —
	// not the whole comment down to the signature. The prose after the list explains WHY
	// resolvedCatalogRoot is a side effect, so scanning the whole block would let the name be
	// deleted from the list itself and still pass: the test would assert the explanation
	// exists rather than that the enumeration is complete.
	var list strings.Builder
	for _, line := range strings.Split(text[idx:idx+end], "\n") {
		if strings.HasPrefix(line, "//\t") {
			list.WriteString(line)
			list.WriteString("\n")
		}
	}
	if list.Len() == 0 {
		t.Fatal("non-vacuity: resolveLatencyConfig's side-effect comment has no indented `//\\t` list")
	}

	for _, v := range []string{"modelConfigDir", "resolvedCatalogRoot", "hwConfigPath", "totalKVBlocks"} {
		if !strings.Contains(list.String(), v) {
			t.Errorf("resolveLatencyConfig's documented side-effect list omits %q, which it does mutate "+
				"(directly or via resolveModelConfig). A caller reading the list would not know the value "+
				"is available.", v)
		}
	}
}

// TestModelsDocScopeClaimIsBounded pins the corrected compatibility framing on
// docs/reference/models.md. "any other model runs" / "Any other model ... will work"
// overstated it: a config with no derivable layer count is refused, a non-SwiGLU activation
// makes KV auto-sizing fatal (the run aborts unless `--total-kv-blocks` is set), and several
// modern shapes run only under documented approximations.
func TestModelsDocScopeClaimIsBounded(t *testing.T) {
	const path = "../docs/reference/models.md"
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	text := string(src)

	overstatements := []*regexp.Regexp{
		regexp.MustCompile(`(?i)any other model runs`),
		regexp.MustCompile(`(?i)any other model with a [^.\n]*config\.json[^.\n]* will work`),
	}
	for _, re := range overstatements {
		if loc := re.FindStringIndex(text); loc != nil {
			t.Errorf("%s claims unbounded model support (%q). State the boundary: some configs are "+
				"refused, some lose KV auto-sizing, and some run only under known approximations",
				path, text[loc[0]:loc[1]])
		}
	}

	// Non-vacuity: the page must still make the positive claim that onboarding needs no
	// BLIS-side code — deleting the sentence entirely would trivially satisfy the above.
	if !strings.Contains(text, "HuggingFace `config.json`") {
		t.Errorf("%s must still explain that a model is onboarded from its HuggingFace `config.json`", path)
	}
}

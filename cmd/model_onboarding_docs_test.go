package cmd

import (
	"bytes"
	"os"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// Guards the two model-onboarding documentation claims corrected by #1730.
//
// Both tests are consistency laws between a document and the code it describes,
// not golden-value checks: BC-2's expected key set is derived from the live
// schema by reflection, so it cannot encode a mistake as correct (R7/R12).

const (
	modelsDocPath       = "../docs/reference/models.md"
	contributingDocPath = "../CONTRIBUTING.md"
)

func readDoc(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		// A doc guard that skips when the file moves is a silent failure (R1).
		t.Fatalf("read %s: %v", path, err)
	}
	return string(data)
}

// preamble returns everything before a markdown document's first `## ` heading:
// the prose a reader meets before any section. Deliberately the whole preamble
// rather than the first paragraph — otherwise the claim could simply move one
// paragraph down and the guard would go quiet.
func preamble(t *testing.T, doc string) string {
	t.Helper()
	body := doc
	if end := strings.Index(body, "\n## "); end >= 0 {
		body = body[:end]
	}
	// Drop the H1 title line so only prose is examined.
	if nl := strings.Index(body, "\n"); nl >= 0 && strings.HasPrefix(body, "# ") {
		body = body[nl+1:]
	}
	if strings.TrimSpace(body) == "" {
		t.Fatal("model compatibility doc has no prose before its first section heading")
	}
	return body
}

// overclaimPattern matches the family of unqualified absence claims this page
// contradicts: "no per-model setup or calibration required", "requires no
// per-model setup", "works without per-model calibration". It targets the claim
// shape rather than one sentence, so a reworded but still-qualified intro passes.
// \b matters: an unanchored "no" also matches inside "not", "node" and "know".
var overclaimPattern = regexp.MustCompile(`(?i)\b(no|without)\b[^.]{0,40}per-model[^.]{0,40}\b(setup|calibration)\b`)

// TestOverclaimPattern_Discriminates keeps the guard below honest. A pattern that
// stopped matching the sentence #1730 removed would pass forever without
// detecting a regression, so the historical claim is pinned as a positive case
// alongside the qualified statements that must not trip it.
func TestOverclaimPattern_Discriminates(t *testing.T) {
	tests := []struct {
		name  string
		text  string
		flags bool
	}{
		{
			name:  "the pre-#1730 sentence",
			text:  "BLIS supports **any transformer model with a HuggingFace `config.json`** — no per-model setup or calibration required.",
			flags: true,
		},
		{
			name:  "reworded overclaim",
			text:  "Any model runs without per-model calibration of any kind.",
			flags: true,
		},
		{
			name:  "qualified: no per-model coefficient fit is a narrower, true claim",
			text:  "Onboarding a model needs no BLIS-side code and no per-model coefficient fit; one global coefficient set is shared.",
			flags: false,
		},
		{
			name:  "qualified: calibration is stated as required",
			text:  "An unvalidated architecture needs per-model calibration before its absolute numbers can be trusted.",
			flags: false,
		},
		{
			name:  "substring trap: 'not'/'node' must not read as 'no'",
			text:  "A node does not escape per-model calibration requirements.",
			flags: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := overclaimPattern.MatchString(tc.text); got != tc.flags {
				t.Errorf("MatchString(%q) = %v, want %v", tc.text, got, tc.flags)
			}
		})
	}
}

// TestModelsDoc_IntroDoesNotContradictCalibrationNote asserts BC-1 of #1730:
// docs/reference/models.md documents calibration inputs and known approximations,
// so its intro must not claim that none are required.
//
// The assertion targets the contradiction pattern ("no per-model
// setup/calibration [is] required"), not an exact sentence, so a future rewrite
// that keeps the qualified framing still passes.
func TestModelsDoc_IntroDoesNotContradictCalibrationNote(t *testing.T) {
	doc := readDoc(t, modelsDocPath)

	// Non-vacuity: the test is only meaningful because the page really does
	// document a calibration input and a set of known approximations. If either
	// disappears the premise changed and this guard must be revisited.
	for _, premise := range []string{"MFU Calibration", "Known approximations"} {
		if !strings.Contains(doc, premise) {
			t.Fatalf("premise %q is gone from %s: the intro's claim must be re-evaluated against the page's new content, not silently accepted",
				premise, modelsDocPath)
		}
	}

	intro := preamble(t, doc)

	if loc := overclaimPattern.FindString(intro); loc != "" {
		t.Errorf("%s preamble claims %q, but the same page documents an MFU calibration input and known per-architecture approximations (#1730 BC-1).\npreamble: %s",
			modelsDocPath, loc, intro)
	}
}

// yamlKeyInBackticks matches a YAML key as the docs write it inside backticks:
// `defaults:`, `models:`, `hf_repo: Qwen/Qwen3-14B`, `GPU: H100`. Case-insensitive
// because one real key (`GPU`) is upper-case; matches are compared lower-cased.
var yamlKeyInBackticks = regexp.MustCompile("(?i)`([a-z][a-z0-9_]*):")

// sectionBody returns the body of the markdown section introduced by heading,
// up to the next heading at the same-or-shallower level.
func sectionBody(t *testing.T, doc, heading string) string {
	t.Helper()
	start := strings.Index(doc, heading)
	if start < 0 {
		t.Fatalf("section %q not found in CONTRIBUTING.md; if it was renamed, retarget this guard rather than deleting it", heading)
	}
	body := doc[start+len(heading):]
	level := strings.Repeat("#", len(heading)-len(strings.TrimLeft(heading, "#")))
	if end := strings.Index(body, "\n"+level+" "); end >= 0 {
		body = body[:end]
	}
	return body
}

// TestContributingAddModelSection_NamesOnlyRealDefaultsKeys asserts BC-2 of
// #1730: every YAML key the "Adding a New Model to defaults.yaml" section tells a
// contributor to add must be one the real schema accepts, so that following the
// section leaves defaults.yaml parseable.
//
// defaults.yaml is decoded into the fixed cmd.Config struct with
// KnownFields(true), so an unrecognized top-level key (such as the `models:`
// list this section used to name) is a hard parse error, not an inert extra.
func TestContributingAddModelSection_NamesOnlyRealDefaultsKeys(t *testing.T) {
	section := sectionBody(t, readDoc(t, contributingDocPath), "### Adding a New Model to defaults.yaml")

	accepted := acceptedDefaultsKeys(t)

	seen := map[string]bool{}
	var offenders []string
	for _, m := range yamlKeyInBackticks.FindAllStringSubmatch(section, -1) {
		key := strings.ToLower(m[1])
		if seen[key] {
			continue
		}
		seen[key] = true
		if !accepted[key] {
			offenders = append(offenders, key)
		}
	}
	sort.Strings(offenders) // deterministic report (R2), and report all, not just the first
	if len(offenders) > 0 {
		sortedAccepted := make([]string, 0, len(accepted))
		for k := range accepted {
			sortedAccepted = append(sortedAccepted, k)
		}
		sort.Strings(sortedAccepted)
		t.Errorf("CONTRIBUTING.md \"Adding a New Model to defaults.yaml\" instructs adding YAML key(s) %v, which the real defaults.yaml schema does not accept (strict parsing rejects them, R10). Accepted: %v (#1730 BC-2)",
			offenders, sortedAccepted)
	}
}

// acceptedDefaultsKeys is the set of YAML keys a contributor may legitimately
// add while onboarding a model: the real top-level keys of defaults.yaml plus
// the fields of one per-model entry. Derived from the live file and the live
// struct, never hardcoded.
func acceptedDefaultsKeys(t *testing.T) map[string]bool {
	t.Helper()

	data, err := os.ReadFile("../defaults.yaml")
	if err != nil {
		t.Fatalf("read defaults.yaml: %v", err)
	}

	// Strict-decode into the real parse target first: this test's premise is
	// that the committed file matches cmd.Config.
	var cfg Config
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	if err := dec.Decode(&cfg); err != nil {
		t.Fatalf("defaults.yaml does not parse into cmd.Config: %v", err)
	}

	var top yaml.Node
	if err := yaml.Unmarshal(data, &top); err != nil {
		t.Fatalf("parse defaults.yaml as a node tree: %v", err)
	}

	keys := map[string]bool{}
	if len(top.Content) > 0 {
		root := top.Content[0]
		for i := 0; i+1 < len(root.Content); i += 2 {
			keys[root.Content[i].Value] = true
		}
	}
	if len(keys) == 0 {
		t.Fatal("no top-level keys found in defaults.yaml")
	}

	// Fields of one per-model `defaults:` entry (GPU, tensor_parallelism, hf_repo).
	entry := reflect.TypeOf(DefaultConfig{})
	for i := 0; i < entry.NumField(); i++ {
		tag := strings.Split(entry.Field(i).Tag.Get("yaml"), ",")[0]
		if tag != "" && tag != "-" {
			keys[strings.ToLower(tag)] = true
		}
	}
	return keys
}

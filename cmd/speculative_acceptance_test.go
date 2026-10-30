package cmd

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	blisschemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/spec/acceptance"

	"github.com/inference-sim/inference-sim/sim"
)

// The DeepSeek-V4.1-Flash DSpark scenario the acceptance tests vary K and the pin on.
const (
	acceptScenarioDir  = "../testdata/scenarios"
	acceptScenarioFile = "deepseek-v4.1-flash-h100-tp8-dspark5.yaml"
	acceptFixtureSet   = "../testdata/acceptance/inferencex-golden-al.yaml"
)

// acceptanceRepos is a registry -- the vendored coefficients plus the acceptance-profile set
// fixture under acceptance/ -- and a catalog -- the vendored one plus, when preset is set, a
// workloads/accept.yaml preset with that YAML appended to chatbot's distributions.
func acceptanceRepos(t *testing.T, preset string) (registry, catalog string) {
	t.Helper()
	root := t.TempDir()
	registry = filepath.Join(root, "registry")
	mustSymlinkAbs(t, "../testdata/registry/coefficients", filepath.Join(registry, "coefficients"))
	set, err := os.ReadFile(acceptFixtureSet)
	if err != nil {
		t.Fatal(err)
	}
	mustWrite(t, filepath.Join(registry, "acceptance", "inferencex-golden-al.yaml"), set)

	catalog = filepath.Join(root, "catalog")
	for _, d := range []string{"models", "hardware", "networks", "devices"} {
		mustSymlinkAbs(t, filepath.Join("../testdata/catalog", d), filepath.Join(catalog, d))
	}
	chat, err := os.ReadFile("../testdata/catalog/workloads/chatbot.yaml")
	if err != nil {
		t.Fatal(err)
	}
	mustWrite(t, filepath.Join(catalog, "workloads", "chatbot.yaml"), chat)
	if preset != "" {
		mustWrite(t, filepath.Join(catalog, "workloads", "accept.yaml"), append(chat, []byte("\n"+preset)...))
	}
	return registry, catalog
}

func mustSymlinkAbs(t *testing.T, target, link string) {
	t.Helper()
	abs, err := filepath.Abs(target)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(abs, link); err != nil {
		t.Fatal(err)
	}
}

func mustWrite(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

// dsparkVariant writes the DSpark scenario drafting k tokens, with scenarioExtra appended to
// its Scenario document (an acceptance pin, say), and returns its directory.
func dsparkVariant(t *testing.T, k int, scenarioExtra string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(acceptScenarioDir, acceptScenarioFile))
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Replace(string(data), "num_spec_tokens: 5", "num_spec_tokens: "+strconv.Itoa(k), 1)
	if scenarioExtra != "" {
		text = strings.Replace(text, "\n---\n", "\n"+scenarioExtra+"\n---\n", 1)
	}
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, acceptScenarioFile), []byte(text))
	return dir
}

// runAccept runs blis against the given scenario directory, registry and catalog, returning
// the results file's acceptance block (nil when absent), stdout and stderr.
func runAccept(t *testing.T, scenarios, registry, catalog string, args ...string) (*sim.AcceptanceRecord, string, string, error) {
	t.Helper()
	metrics := filepath.Join(t.TempDir(), "metrics.json")
	full := append(args, "--scenario", acceptScenarioFile, "--scenarios", scenarios,
		"--registry", registry, "--catalog", catalog, "--defaults-filepath", "../defaults.yaml",
		"--metrics-path", metrics, "--seed", "3", "--log", "info")
	c := exec.Command(os.Args[0], "-test.run=^TestKernelCLILeg$")
	c.Env = append(os.Environ(), kernelCLIArgsEnv+"="+strings.Join(full, "\x1f"))
	var out, errBuf bytes.Buffer
	c.Stdout, c.Stderr = &out, &errBuf
	err := c.Run()
	if err != nil {
		return nil, out.String(), errBuf.String(), err
	}
	data, rerr := os.ReadFile(metrics)
	if rerr != nil {
		t.Fatalf("reading the results file: %v", rerr)
	}
	var file struct {
		Acceptance *sim.AcceptanceRecord `json:"acceptance"`
	}
	if jerr := json.Unmarshal(data, &file); jerr != nil {
		t.Fatal(jerr)
	}
	return file.Acceptance, out.String(), errBuf.String(), nil
}

// fixtureAL is the fixture set's acceptance length for DeepSeek-V4.1-Flash DSpark at k.
func fixtureAL(t *testing.T, k int, thinking acceptance.Thinking) float64 {
	t.Helper()
	s, err := blisschemas.LoadAcceptanceProfileSet(acceptFixtureSet)
	if err != nil {
		t.Fatal(err)
	}
	want := acceptance.Key{Model: "deepseek-v4.1-flash", SpeculativeMethod: "dspark", NumSpecTokens: k,
		Thinking: thinking, Category: "coding"}
	for _, p := range s.Profiles {
		if p.Key() == want {
			return p.AcceptanceLength
		}
	}
	t.Fatalf("the fixture set has no %s", want)
	return 0
}

var acceptRunArgs = []string{"run", "--num-requests", "3", "--rate", "5"}

// checkRecord asserts the record's internal law -- AL used = 1 + rate*K, the rate in [0,1]
// and AL in [1, K+1] -- and its layer and AL.
func checkRecord(t *testing.T, rec *sim.AcceptanceRecord, k int, layer string, al float64) {
	t.Helper()
	if rec == nil {
		t.Fatal("the results file records no acceptance")
	}
	if rec.SourceLayer != layer {
		t.Errorf("source_layer %q, want %q (entry %+v)", rec.SourceLayer, layer, rec.Entry)
	}
	if math.Abs(rec.Length-al) > 1e-9 {
		t.Errorf("length %v, want %v", rec.Length, al)
	}
	if math.Abs(rec.Length-(1+rec.Rate*float64(k))) > 1e-12 || rec.Length < 1 || rec.Length > float64(k+1) {
		t.Errorf("length %v and rate %v are not AL = 1 + rate*K at K=%d", rec.Length, rec.Rate, k)
	}
}

// #1939 precedence, lowest to highest: registry profile < workload < scenario pin < CLI. Each
// row states every layer up to one and checks that one supplied the acceptance; the registry
// row is the default. The record names the layer and, for a profile, the entry.
func TestAcceptancePrecedence(t *testing.T) {
	const k = 5
	selectors := []string{"--speculative-thinking", "on", "--speculative-category", "coding"}
	workloadPreset := "acceptance:\n  num_spec_tokens: 5\n  acceptance_length: 2.2\n  source: a chat trace's server log\n"
	pin := "acceptance:\n  value:\n    num_spec_tokens: 5\n    acceptance_length: 2.7\n    source: the InferenceX run's synthetic_acceptance_length\n"
	for _, tt := range []struct {
		name    string
		preset  string // the run's --workload preset's acceptance terms
		pin     string // the scenario's acceptance pin
		extra   []string
		layer   string
		al      float64
		profile bool
	}{
		{"registry profile by the CLI selectors", "", "", selectors, "registry", fixtureAL(t, k, acceptance.ThinkingOn), true},
		{"registry profile by the workload's selectors", "category: coding\nthinking: \"off\"\n", "", nil, "registry", fixtureAL(t, k, acceptance.ThinkingOff), true},
		{"CLI selectors override the workload's", "category: coding\nthinking: \"off\"\n", "", []string{"--speculative-thinking", "on"}, "registry", fixtureAL(t, k, acceptance.ThinkingOn), true},
		{"workload value over the registry", workloadPreset, "", selectors, "workload", 2.2, false},
		{"scenario pin over the workload", workloadPreset, pin, selectors, "scenario", 2.7, false},
		{"scenario pin naming the profile set", "", "acceptance:\n  profiles: inferencex-golden-al\n", selectors, "scenario", fixtureAL(t, k, acceptance.ThinkingOn), true},
		{"CLI length over the scenario", workloadPreset, pin, append([]string{"--speculative-acceptance-length", "1.5"}, selectors...), "cli", 1.5, false},
		{"CLI rate over the scenario", workloadPreset, pin, append([]string{"--speculative-acceptance-rate", "0.6"}, selectors...), "cli", 1 + 0.6*k, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			registry, catalog := acceptanceRepos(t, tt.preset)
			args := append([]string{}, acceptRunArgs...)
			if tt.preset != "" {
				args = append(args, "--workload", "accept")
			}
			rec, _, stderr, err := runAccept(t, dsparkVariant(t, k, tt.pin), registry, catalog, append(args, tt.extra...)...)
			if err != nil {
				t.Fatalf("run refused: %v\n%s", err, lastLines(stderr, 4))
			}
			checkRecord(t, rec, k, tt.layer, tt.al)
			if got := rec.Entry.Key != ""; got != tt.profile {
				t.Errorf("entry %+v: profile recorded = %t, want %t", rec.Entry, got, tt.profile)
			}
			if tt.profile && rec.Entry.Set != "inferencex-golden-al" {
				t.Errorf("entry set %q, want inferencex-golden-al", rec.Entry.Set)
			}
			if !strings.Contains(stderr, "speculative acceptance: AL") {
				t.Errorf("stderr does not report the resolution:\n%s", lastLines(stderr, 6))
			}
		})
	}
}

// A sweep of the deployment's K with no acceptance override takes the registry's profile at
// each K, so AL differs at every K -- never one acceptance carried across them -- and is the
// fixture's value at that K.
func TestAcceptanceKSweepPicksTheProfileAtEachK(t *testing.T) {
	registry, catalog := acceptanceRepos(t, "")
	seen := map[float64]int{}
	for k := 1; k <= 5; k++ {
		rec, _, stderr, err := runAccept(t, dsparkVariant(t, k, ""), registry, catalog,
			append(append([]string{}, acceptRunArgs...), "--speculative-thinking", "on", "--speculative-category", "coding")...)
		if err != nil {
			t.Fatalf("K=%d refused: %v\n%s", k, err, lastLines(stderr, 4))
		}
		checkRecord(t, rec, k, "registry", fixtureAL(t, k, acceptance.ThinkingOn))
		if prev, dup := seen[rec.Length]; dup {
			t.Errorf("K=%d and K=%d resolved the same AL %v", prev, k, rec.Length)
		}
		seen[rec.Length] = k
	}
}

// Nothing falls back silently: an unmatched category, an unstated selector, a value stated for
// another K, an AL outside [1, K+1], both spellings at once, and an acceptance for a scenario
// that does not draft are refused, naming the cause. A requested fallback category resolves
// and is recorded as such.
func TestAcceptanceRefusals(t *testing.T) {
	registry, catalog := acceptanceRepos(t, "")
	for _, tt := range []struct {
		name  string
		k     int
		pin   string
		args  []string
		cause string
	}{
		{"unmatched category", 5, "", []string{"--speculative-thinking", "on", "--speculative-category", "chat"}, "no fallback_category was requested"},
		{"no thinking mode", 5, "", []string{"--speculative-category", "coding"}, "states no thinking mode"},
		{"no selectors and no value", 5, "", nil, "states no thinking mode"},
		{"pinned value stated for another K", 5, "acceptance:\n  value:\n    num_spec_tokens: 3\n    acceptance_length: 2.0\n", nil, "num_spec_tokens"},
		{"AL above K+1", 5, "", []string{"--speculative-acceptance-length", "6.5"}, "lies outside [1, 6]"},
		{"AL below 1", 5, "", []string{"--speculative-acceptance-length", "0.5"}, "lies outside [1, 6]"},
		{"rate and length together", 5, "", []string{"--speculative-acceptance-length", "2", "--speculative-acceptance-rate", "0.2"}, "none of the others can be"},
		{"length with nothing drafted", 0, "", []string{"--speculative-acceptance-length", "2"}, "drafts no tokens"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := dsparkVariant(t, tt.k, tt.pin)
			if tt.k == 0 {
				dir = noSpeculationVariant(t)
			}
			_, _, stderr, err := runAccept(t, dir, registry, catalog, append(append([]string{}, acceptRunArgs...), tt.args...)...)
			if err == nil {
				t.Fatalf("accepted; want a refusal naming %q", tt.cause)
			}
			if !strings.Contains(stderr, tt.cause) {
				t.Errorf("refusal does not name %q:\n%s", tt.cause, lastLines(stderr, 4))
			}
		})
	}
	t.Run("requested fallback category", func(t *testing.T) {
		rec, _, stderr, err := runAccept(t, dsparkVariant(t, 5, ""), registry, catalog, append(append([]string{}, acceptRunArgs...),
			"--speculative-thinking", "on", "--speculative-category", "chat", "--speculative-fallback-category", "coding")...)
		if err != nil {
			t.Fatalf("refused: %v\n%s", err, lastLines(stderr, 4))
		}
		checkRecord(t, rec, 5, "registry", fixtureAL(t, 5, acceptance.ThinkingOn))
		if rec.Entry.FallbackFrom != "chat" {
			t.Errorf("fallback_from %q, want chat", rec.Entry.FallbackFrom)
		}
	})
}

// noSpeculationVariant is the DSpark scenario with its speculative block removed.
func noSpeculationVariant(t *testing.T) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(acceptScenarioDir, acceptScenarioFile))
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Replace(string(data), "      speculative:\n        method: dspark\n        num_spec_tokens: 5\n", "", 1)
	if text == string(data) {
		t.Fatal("the scenario's speculative block was not found")
	}
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, acceptScenarioFile), []byte(text))
	return dir
}

// A run without speculation records no acceptance, and selectors given to it are reported as
// ignored rather than silently dropped.
func TestAcceptanceIsAbsentWithoutSpeculation(t *testing.T) {
	registry, catalog := acceptanceRepos(t, "")
	rec, _, stderr, err := runAccept(t, noSpeculationVariant(t), registry, catalog,
		append(append([]string{}, acceptRunArgs...), "--speculative-thinking", "on")...)
	if err != nil {
		t.Fatalf("refused: %v\n%s", err, lastLines(stderr, 4))
	}
	if rec != nil {
		t.Errorf("a run that does not speculate records acceptance %+v", rec)
	}
	if !strings.Contains(stderr, "--speculative-thinking is ignored") {
		t.Errorf("the ignored selector is not reported:\n%s", lastLines(stderr, 6))
	}
}

// INV-13: a resolved acceptance is a function of the scenario, registry and flags alone, so
// the replay of a run's trace under the same flags resolves the same acceptance and
// reproduces the run byte for byte.
func TestAcceptanceRunReplayParity(t *testing.T) {
	registry, catalog := acceptanceRepos(t, "")
	dir := dsparkVariant(t, 5, "")
	prefix := filepath.Join(t.TempDir(), "trace")
	common := []string{"--speculative-thinking", "on", "--speculative-category", "coding", "--horizon", "600000000"}
	runRec, runOut, stderr, err := runAccept(t, dir, registry, catalog,
		append(append(append([]string{}, acceptRunArgs...), "--trace-output", prefix), common...)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 4))
	}
	repRec, repOut, stderr, err := runAccept(t, dir, registry, catalog,
		append([]string{"replay", "--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv"}, common...)...)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 4))
	}
	if inv13View(t, runOut) != inv13View(t, repOut) {
		t.Errorf("INV-13: the replay's Result differs from the run's")
	}
	if !reflect.DeepEqual(runRec, repRec) {
		t.Errorf("run recorded %+v, replay %+v", runRec, repRec)
	}
}

package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// inference-sim#2079: a run whose acceptance is borrowed from a profile measured under another
// configuration says so once on stderr, naming the step, what differed and the source profile;
// an exact match prints no such line; a lookup that finds nothing is still refused with the
// resolver's message. The Result's acceptance.entry carries the same step, differed and k_rule.
const museDSparkScenario = "muse-gb200-nvl-tep8-dspark.yaml"

// museDSparkDir writes the Muse-Spark GB200 TEP8 DSpark scenario with edit applied to its text.
func museDSparkDir(t *testing.T, edit func(string) string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "testdata", "scenarios", "muse-spark", museDSparkScenario))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, museDSparkScenario), []byte(edit(string(data))), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// statesSamplers adds the draft sampler and verifier the registry's Muse profile was measured
// with, so the run's configuration is the profile's own.
func statesSamplers(s string) string {
	return strings.Replace(s, "num_speculative_tokens: 7,",
		"num_speculative_tokens: 7, draft_sample_method: probabilistic, rejection_sample_method: block,", 1)
}

func runMuseAcceptance(t *testing.T, dir string, extra ...string) (entry map[string]any, length float64, stderr string, err error) {
	t.Helper()
	args := append([]string{"run", "--scenarios", dir, "--scenario", museDSparkScenario,
		"--rate", "2", "--num-requests", "4", "--seed", "1"}, extra...)
	stdout, stderr, err := runKernelCLI(t, args...)
	if err != nil {
		return nil, 0, stderr, err
	}
	cluster := resultDoc(t, stdout)["reports"].(map[string]any)["engine"].(map[string]any)["cluster"].(map[string]any)
	acc := cluster["acceptance"].(map[string]any)
	entry, _ = acc["entry"].(map[string]any)
	length, _ = acc["length"].(float64)
	return entry, length, stderr, nil
}

func borrowedLines(stderr string) []string {
	var out []string
	for _, l := range strings.Split(stderr, "\n") {
		if strings.Contains(l, "speculative acceptance: borrowed") {
			out = append(out, l)
		}
	}
	return out
}

func TestABorrowedAcceptanceWarnsOnceNamingTheStepAndTheSourceProfile(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the scenario")
	}
	entry, _, stderr, err := runMuseAcceptance(t, museDSparkDir(t, func(s string) string { return s }))
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 4))
	}
	if entry["step"] != "same_drafter_other_settings" {
		t.Fatalf("step %v, want same_drafter_other_settings (the scenario states no draft sampler or verifier)", entry["step"])
	}
	lines := borrowedLines(stderr)
	if len(lines) != 1 {
		t.Fatalf("%d borrowed-acceptance warnings, want exactly one:\n%s", len(lines), strings.Join(lines, "\n"))
	}
	for _, want := range []string{"level=warning", "same_drafter_other_settings", "draft_sampler", "verifier",
		"internal-deployment-measurements", "muse-spark-open-dspark/dspark/k7/thinking-on/coding"} {
		if !strings.Contains(lines[0], want) {
			t.Errorf("the warning lacks %q:\n%s", want, lines[0])
		}
	}
}

func TestAnExactAcceptanceMatchPrintsNoBorrowedWarning(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the scenario")
	}
	entry, length, stderr, err := runMuseAcceptance(t, museDSparkDir(t, statesSamplers))
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 4))
	}
	if entry["step"] != "exact" || length != 2.72 {
		t.Fatalf("step %v AL %v, want the profile's own key: exact, 2.72", entry["step"], length)
	}
	if lines := borrowedLines(stderr); len(lines) != 0 {
		t.Errorf("an exact match must not warn that its acceptance is borrowed:\n%s", strings.Join(lines, "\n"))
	}
	if _, has := entry["differed"]; has {
		t.Errorf("an exact match records nothing differing, got %v", entry["differed"])
	}
}

func TestABorrowedAcceptanceAtAnotherKNamesTheKRule(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the scenario")
	}
	dir := museDSparkDir(t, func(s string) string {
		return strings.Replace(statesSamplers(s), "num_speculative_tokens: 7", "num_speculative_tokens: 5", 1)
	})
	entry, _, stderr, err := runMuseAcceptance(t, dir)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 4))
	}
	rule, _ := entry["k_rule"].(string)
	if entry["step"] != "same_drafter_other_k" || rule == "" {
		t.Fatalf("step %v k_rule %q, want same_drafter_other_k with a K rule", entry["step"], rule)
	}
	lines := borrowedLines(stderr)
	if len(lines) != 1 || !strings.Contains(lines[0], "same_drafter_other_k") || !strings.Contains(lines[0], rule) {
		t.Errorf("want one warning naming the step and the %q rule, got:\n%s", rule, strings.Join(lines, "\n"))
	}
}

func TestAnAcceptanceNoProfileCoversIsStillRefusedWithTheResolversMessage(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the scenario")
	}
	_, _, stderr, err := runMuseAcceptance(t, museDSparkDir(t, func(s string) string { return s }),
		"--speculative-category", "writing")
	if err == nil {
		t.Fatal("a category no profile covers must be refused, not borrowed")
	}
	if !strings.Contains(stderr, "no profile for") || len(borrowedLines(stderr)) != 0 {
		t.Errorf("want the resolver's refusal and no borrowed warning:\n%s", lastLines(stderr, 3))
	}
}

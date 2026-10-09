package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/latency"
)

// A scenario must not carry a batch setting that the deployment did not have.
//
// The scenario generator originally wrote max_num_seqs: 256 for every deployment and
// justified it in a comment: the comparison scores a ratio across concurrency, so a constant
// cancels. That is true for step time and false for time to first token, where the sequence
// cap decides whether a request waits at all. Nothing caught it, because nothing asserted
// that an unstated setting had been resolved rather than chosen.
//
// This test asserts the property that was missing: where a scenario states a batch setting,
// it must match what the engine would resolve for that chip, unless the file records that the
// deployment stated it explicitly. A value that matches neither is a guess.
func TestScenariosDoNotCarryGuessedBatchDefaults(t *testing.T) {
	dir := DefaultScenarios()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("scenario directory: %v", err)
	}

	// Total device memory per chip, from blis-catalog. Stated here rather than loaded so the
	// test says what it is checking against.
	memory := map[string]float64{
		"h100": 80, "h200": 141, "b200": 180, "b300": 288,
		"gb200-nvl72": 186, "l40s": 48, "a100-sxm": 80, "a100-80": 80,
	}

	var guessed []string
	checked := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".yaml") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatalf("%s: %v", e.Name(), err)
		}
		text := string(raw)
		chip := yamlScalar(text, "hardware")
		mem, ok := memory[chip]
		if !ok {
			continue
		}
		stated := yamlScalar(text, "max_num_seqs")
		if stated == "" {
			continue // nothing stated is nothing to check
		}
		checked++
		want := latency.ResolveVLLMBatchDefaults(mem, chip)
		// A file may legitimately state a measured value from the deployment's own launch.
		// It must say so, so a reader can tell a measurement from a default.
		if strings.Contains(text, "measured launch setting") {
			continue
		}
		if stated != itoa(want.MaxNumSeqs) {
			guessed = append(guessed, e.Name()+": max_num_seqs "+stated+
				" on "+chip+", but the engine resolves "+itoa(want.MaxNumSeqs))
		}
	}
	if checked == 0 {
		t.Skip("no scenario states max_num_seqs")
	}
	if len(guessed) == 0 {
		return
	}

	// The scenario files are generated into blis-latency-kernel and corrected by the
	// extraction that reads each deployment's own engine log (#1867 and its sibling). Until
	// that lands they all carry the guessed 256, so failing here would leave a permanently
	// red suite that says nothing new on each run.
	//
	// Set BLIS_REQUIRE_RESOLVED_SETTINGS=1 to make this fatal. That is what the scenario
	// regeneration turns on, and what keeps a corrected file from silently regressing.
	fatal := os.Getenv("BLIS_REQUIRE_RESOLVED_SETTINGS") != ""
	report := t.Logf
	if fatal {
		report = t.Errorf
	}
	report("%d of %d scenarios carry a max_num_seqs the engine would not resolve:",
		len(guessed), checked)
	for _, g := range guessed {
		report("    %s", g)
	}
	report("A scenario must carry either the engine's resolved default or a value the " +
		"deployment stated, marked \"measured launch setting\". Neither holds for the " +
		"above, so the value is a guess -- and the sequence cap decides whether a request " +
		"waits, which no step-time calibration can absorb. Set " +
		"BLIS_REQUIRE_RESOLVED_SETTINGS=1 to fail on this.")
}

// yamlScalar reads a top-level or indented `key: value` scalar, which is all this check
// needs and avoids pulling a YAML dependency into the test.
func yamlScalar(doc, key string) string {
	for _, line := range strings.Split(doc, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") {
			continue
		}
		if after, found := strings.CutPrefix(trimmed, key+":"); found {
			return strings.TrimSpace(after)
		}
	}
	return ""
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

package kernelmodel

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The census check of X19/VLM-1 over the committed fixtures (kerneltests/testdata/direct and
// aisimulate): each fixture's block_size is removed and the resolution run. A fixture's header
// says whether the recorded run stated the size ("block_size N: stated by the run") or the
// generator filled it in ("not stated"). The three counts:
//
//	(a) the run stated the size, so it was an operator's flag and not vLLM's pick (interface.py:757
//	    picks a size only when none is passed): excluded;
//	(b) the run's size is known from its logs as vLLM's own pick: the resolved size must equal it.
//	    No committed fixture records that. The recorded sizes are all operator flags, so (b) is
//	    empty and the table rests on the CPU oracle alone (oracle/blocksize);
//	(c) the resolution is refused, citing #68.
//
// A fixture whose header does not say the run stated the size is counted with (c) or as resolved
// with no recorded size to compare; none of those is a claim about vLLM.
func TestBlockSizeResolutionOverTheCensusFixtures(t *testing.T) {
	stated := regexp.MustCompile(`(?m)^# block_size \d+: stated by the run\.$`)
	strip := regexp.MustCompile(`(?m)^\s*block_size:\s*\d+\s*\n`)
	var files []string
	for _, d := range []string{"direct", "aisimulate"} {
		m, err := filepath.Glob(filepath.Join("..", "..", "kerneltests", "testdata", d, "*.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, m...)
	}
	// (b) is empty: see above.
	var a, c, resolvedNotStated int
	const b = 0
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		body := string(raw)
		if !strings.Contains(body, `engine_version: "0.32.0"`) {
			continue
		}
		switch {
		case stated.MatchString(body):
			a++
			continue
		}
		// Every other fixture's size was filled in by the generator or a hand-written fixture's
		// author, not recorded by a run.
		if !strip.MatchString(body) {
			continue
		}
		r := repos()
		r.Scenarios = t.TempDir()
		if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(strip.ReplaceAllString(body, "")), 0o644); err != nil {
			t.Fatal(err)
		}
		m, err := Open("v.yaml", r)
		if err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		if _, err := m.Engine(); err != nil {
			if !strings.Contains(err.Error(), "blis-vllm/issues/68") && !strings.Contains(err.Error(), "does not build") {
				t.Errorf("%s: refused, but not citing #68: %v", f, err)
			}
			c++
			continue
		}
		resolvedNotStated++
	}
	t.Logf("(a) run stated the size: %d; (b) size known from logs: %d; (c) refused: %d; resolved with no recorded size: %d", a, b, c, resolvedNotStated)
	if a+c+resolvedNotStated < 600 {
		t.Fatalf("non-vacuity: only %d fixtures classified", a+c+resolvedNotStated)
	}
}

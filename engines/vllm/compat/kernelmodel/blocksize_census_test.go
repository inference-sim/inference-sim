package kernelmodel

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
)

// The census of X19/VLM-1 over the committed fixtures (kerneltests/testdata/direct and
// aisimulate): each v0.32.0 fixture's block_size is removed and the resolution run. A fixture's
// header says whether the recorded run stated the size ("block_size N: stated by the run") or
// the generator filled it in ("not stated"). The classes, with their exact counts:
//
//	stated       the run stated the size. That is an operator's flag, not vLLM's pick (vLLM picks a
//	             size only when none is passed, vllm/platforms/interface.py:757-781@5df63a72), so it
//	             is not evidence of the default and is excluded here. The recorded-run comparison,
//	             which separates a flag from a size the engine logged, is
//	             TestResolutionAgainstTheRecordedRuns;
//	refused      no size was recorded and the resolution is refused, citing #68;
//	resolved     no size was recorded and the resolution gives one; there is nothing to compare it
//	             with, so this is not a claim about vLLM.
//
// Nine of the refused fixtures are hybrid models, which the resolution refuses because their block
// size is aligned to the recurrent page (interface.py:783-790), and which the v0.32.0 engine also
// does not build (Model.V032Pieces, hybrid KV groups, blis-vllm#7).
//
// A v0.32.0 fixture with no block_size line fails the test: the census would otherwise skip it
// without counting it. A fixture's release is read as the kernel reads it (Scenario.EngineRelease),
// so one that writes engine: {kind, version} is counted.
func TestBlockSizeResolutionOverTheCensusFixtures(t *testing.T) {
	stated := regexp.MustCompile(`(?m)^# block_size \d+: stated by the run\.$`)
	strip := regexp.MustCompile(`(?m)^\s*block_size:\s*\d+\s*\n`)
	release := regexp.MustCompile(`(?m)^(?:engine_version:\s*"?0\.32\.0"?|\s+version:\s*"?0\.32\.0"?)\s*$`)
	var files []string
	for _, d := range []string{"direct", "aisimulate"} {
		m, err := filepath.Glob(filepath.Join("..", "..", "kerneltests", "testdata", d, "*.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, m...)
	}
	// The fixtures are independent (each opens its own Data from a scenario of its own), so they run
	// as parallel subtests; the counts are summed under a mutex and asserted exactly after they all end.
	var (
		mu                                                 sync.Mutex
		nStated, nRefused, nResolved, nHybridPieces, nV032 int
	)
	t.Run("fixtures", func(t *testing.T) {
		for _, f := range files {
			raw, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			body := string(raw)
			if !release.MatchString(body) {
				continue
			}
			mu.Lock()
			nV032++
			mu.Unlock()
			if !strip.MatchString(body) {
				t.Errorf("%s: a v0.32.0 fixture states no block_size line, so the census cannot strip it", f)
				continue
			}
			if stated.MatchString(body) {
				mu.Lock()
				nStated++
				mu.Unlock()
				continue
			}
			t.Run(filepath.Base(f), func(t *testing.T) {
				t.Parallel()
				r := repos()
				r.Scenarios = t.TempDir()
				if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(strip.ReplaceAllString(body, "")), 0o644); err != nil {
					t.Fatal(err)
				}
				m, err := Open("v.yaml", r)
				if err != nil {
					t.Fatalf("%s: %v", f, err)
				}
				pieces := 0
				for _, p := range m.V032Pieces() {
					if !strings.HasPrefix(p, "hybrid KV groups (layer kind gdn_moe: recurrent state is not mapped to a KV cache spec)") {
						t.Errorf("%s: V032Pieces names %q, want only the hybrid-KV-groups piece", f, p)
					}
					pieces++
				}
				_, err = m.Engine()
				if err != nil && !strings.Contains(err.Error(), "blis-vllm/issues/68") {
					t.Errorf("%s: refused, but not citing #68: %v", f, err)
				}
				mu.Lock()
				defer mu.Unlock()
				nHybridPieces += pieces
				if err != nil {
					nRefused++
				} else {
					nResolved++
				}
			})
		}
	})
	t.Logf("v0.32.0 fixtures %d: stated by the run %d; refused %d (hybrid pieces %d); resolved with no recorded size %d",
		nV032, nStated, nRefused, nHybridPieces, nResolved)
	if nV032 != 679 || nStated != 354 || nRefused != 38 || nResolved != 287 || nHybridPieces != 9 {
		t.Errorf("the census counts changed: %d fixtures, %d stated, %d refused (%d hybrid), %d resolved; want 679, 354, 38 (9), 287",
			nV032, nStated, nRefused, nHybridPieces, nResolved)
	}
}

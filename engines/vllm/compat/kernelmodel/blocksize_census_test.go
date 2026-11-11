package kernelmodel

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// The census of X19/VLM-1 over the committed fixtures (kerneltests/testdata/direct and
// aisimulate). Each v0.32.0 fixture's block_size is removed and the resolution run. A fixture's
// header says whether the recorded run stated the size ("block_size N: stated by the run") or the
// generator filled it in ("not stated"). The classes, with their exact counts:
//
//	stated       the run stated the size. That is an OPERATOR'S FLAG, not vLLM's pick (vLLM picks a
//	             size only when none is passed, vllm/platforms/interface.py:757-781@5df63a72), so
//	             it is not evidence of vLLM's choice. The resolution is compared with it anyway,
//	             and every difference is named: reproduced / refused citing #68 / differs, and a
//	             difference not in statedDiffers fails the test. The rows whose server log shows the
//	             size vLLM chose are in TestResolutionAgainstTheRecordedRuns;
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

// statedDiffers are the stated fixtures whose resolved size differs from the stated one, keyed by
// path under kerneltests/testdata, with the reason. Every one is a minimax-m2.5 deployment on
// Blackwell that passed --block-size 32: the server logs of the same recipes (InferenceX rows
// 194852, 194862, 411043, 411048, 411049, 417629 and 417631 in recorded_block_sizes.json) show
// block_size 32 among their non-default arguments, so 32 is the operator's flag, and the table's
// pick for minimax-m2.5 on compute capability 10.x is 16.
const statedFlag32 = "the run passed --block-size 32 (the recipe's server log states it among its non-default arguments); vLLM's own pick on Blackwell is 16"

var statedDiffers = map[string]string{
	"direct/minimax-m2.5-b200-fp8-vllm-tp2-cfg1.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp2-cfg10.yaml": statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp2-cfg11.yaml": statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp2-cfg2.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp2-cfg3.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp2-cfg4.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp2-cfg5.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp2-cfg6.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp2-cfg7.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp2-cfg8.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp2-cfg9.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp2-ep2.yaml":   statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg1.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg10.yaml": statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg11.yaml": statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg12.yaml": statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg13.yaml": statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg14.yaml": statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg15.yaml": statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg2.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg3.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg4.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg5.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg6.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg7.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg8.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-cfg9.yaml":  statedFlag32,
	"direct/minimax-m2.5-b200-fp8-vllm-tp4-ep4.yaml":   statedFlag32,
	"direct/minimax-m2.5-b300-fp8-vllm-tp1-cfg1.yaml":  statedFlag32,
	"direct/minimax-m2.5-b300-fp8-vllm-tp1-cfg2.yaml":  statedFlag32,
	"direct/minimax-m2.5-b300-fp8-vllm-tp2-cfg1.yaml":  statedFlag32,
	"direct/minimax-m2.5-b300-fp8-vllm-tp2-cfg2.yaml":  statedFlag32,
	"direct/minimax-m2.5-b300-fp8-vllm-tp2-cfg3.yaml":  statedFlag32,
	"direct/minimax-m2.5-b300-fp8-vllm-tp4-cfg1.yaml":  statedFlag32,
	"direct/minimax-m2.5-b300-fp8-vllm-tp4-cfg2.yaml":  statedFlag32,
	"direct/minimax-m2.5-b300-fp8-vllm-tp4-cfg3.yaml":  statedFlag32,
}

type censusKey struct{ model, class string }

func TestBlockSizeResolutionOverTheCensusFixtures(t *testing.T) {
	stated := regexp.MustCompile(`(?m)^# block_size (\d+): stated by the run\.$`)
	strip := regexp.MustCompile(`(?m)^\s*block_size:\s*\d+\s*\n`)
	release := regexp.MustCompile(`(?m)^(?:engine_version:\s*"?0\.32\.0"?|\s+version:\s*"?0\.32\.0"?)\s*$`)
	modelOf := regexp.MustCompile(`(?m)^model: (\S+)`)
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
		mu                           sync.Mutex
		nStated, nRefused, nResolved int
		nHybridPieces, nV032         int
		statedClass                  = map[string]int{}
		coverage                     = map[censusKey]int{}
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
			statedSize := ""
			if m := stated.FindStringSubmatch(body); m != nil {
				statedSize = m[1]
			}
			key := filepath.ToSlash(filepath.Join(filepath.Base(filepath.Dir(f)), filepath.Base(f)))
			model := modelOf.FindStringSubmatch(body)[1]
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
				e, err := m.Engine()
				if err != nil && !strings.Contains(err.Error(), "blis-vllm/issues/68") {
					t.Errorf("%s: refused, but not citing #68: %v", f, err)
				}
				class := ""
				if statedSize != "" {
					switch {
					case err != nil:
						class = "refused"
					case strconv.Itoa(e.BlockSize) == statedSize:
						class = "reproduced"
					default:
						class = "differs"
						if _, ok := statedDiffers[key]; !ok {
							t.Errorf("%s: the run stated block_size %s (an operator's flag) and the table resolves %d, and no entry of statedDiffers names the fixture",
								key, statedSize, e.BlockSize)
						}
					}
				}
				mu.Lock()
				defer mu.Unlock()
				nHybridPieces += pieces
				switch {
				case statedSize != "":
					nStated++
					statedClass[class]++
					coverage[censusKey{model, class}]++
				case err != nil:
					nRefused++
				default:
					nResolved++
				}
			})
		}
	})
	for k := range statedDiffers {
		if _, err := os.Stat(filepath.Join("..", "..", "kerneltests", "testdata", filepath.FromSlash(k))); err != nil {
			t.Errorf("statedDiffers names %s: %v", k, err)
		}
	}
	// Coverage by model of the operator-flag fixtures. They are not evidence of vLLM's choice; the
	// engine-logged rows (TestResolutionAgainstTheRecordedRuns) are.
	var models []string
	seen := map[string]bool{}
	for k := range coverage {
		if !seen[k.model] {
			seen[k.model] = true
			models = append(models, k.model)
		}
	}
	sort.Strings(models)
	for _, mo := range models {
		t.Logf("operator-flag fixtures, %-20s reproduced %3d, refused %3d, differs %3d", mo,
			coverage[censusKey{mo, "reproduced"}], coverage[censusKey{mo, "refused"}], coverage[censusKey{mo, "differs"}])
	}
	t.Logf("v0.32.0 fixtures %d: stated by the run %d (reproduced %d, refused %d, differs %d); no size stated: refused %d (hybrid pieces %d), resolved %d",
		nV032, nStated, statedClass["reproduced"], statedClass["refused"], statedClass["differs"], nRefused, nHybridPieces, nResolved)
	if nV032 != 679 || nStated != 354 || nRefused != 26 || nResolved != 299 || nHybridPieces != 9 ||
		statedClass["reproduced"] != 0 || statedClass["refused"] != 318 || statedClass["differs"] != 36 || len(statedDiffers) != 36 {
		t.Errorf("the census counts changed: %d fixtures; stated %d (reproduced %d, refused %d, differs %d); no size stated: refused %d (hybrid %d), resolved %d; "+
			"want 679; 354 (0, 318, 36); 26 (9), 299", nV032, nStated, statedClass["reproduced"], statedClass["refused"], statedClass["differs"],
			nRefused, nHybridPieces, nResolved)
	}
}

// Original to blis-vllm.

package vllmengine

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// kvStartupTiers are the tolerances a group of recorded runs is held to, in order: the tightest
// one that holds today is the group's (kvStartupGroup.Tolerance), so a change that closes a gap is
// asked to say so, and one that opens it fails. 0.03 is plan Table 5's gate on A3, the KV capacity
// ratio within [0.97, 1.03].
var kvStartupTiers = []float64{0.03, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0, 4.0}

type kvStartupObservation struct {
	LoggedTokens       int64   `json:"logged_tokens"`
	LoggedAvailableGiB float64 `json:"logged_available_gib"`
	LoggedVLLM         string  `json:"logged_vllm"`
	Formula            string  `json:"formula"`
	FitUnits           int     `json:"fit_units"`
}

type kvStartupFixture struct {
	Scenario     string                 `json:"scenario"`
	Group        string                 `json:"group"`
	Observations []kvStartupObservation `json:"observations"`
}

type kvStartupGroup struct {
	EndToEnd  float64  `json:"end_to_end"`
	Pinned    *float64 `json:"pinned"`
	Scenarios int      `json:"scenarios"`
	FitUnits  int      `json:"fit_units"`
}

type kvStartupFile struct {
	Tolerance map[string]kvStartupGroup `json:"tolerance"`
	Fixtures  []kvStartupFixture        `json:"fixtures"`
}

func loadKVStartup(t *testing.T) kvStartupFile {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "kvstartup", "logged.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f kvStartupFile
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Fixtures) == 0 {
		t.Fatal("no recorded runs: this test would pass without comparing anything")
	}
	return f
}

var engineBlockStart = regexp.MustCompile(`(?m)^\s+engine:[ \t]*\n(\s+)`)

// kvStartupEngine builds the engine for a recorded run's scenario. With pinGiB > 0 the pool is
// sized from that many GiB per GPU (kv_cache_memory_bytes), the memory the run's own log says it had
// for KV, so only the pool's structure and the start-up figure's formula are compared.
func kvStartupEngine(t *testing.T, scenario string, pinGiB float64) engine.Engine {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "kvstartup", scenario))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	if pinGiB > 0 {
		loc := engineBlockStart.FindStringSubmatchIndex(body)
		if loc == nil {
			t.Fatalf("%s has no engine block", scenario)
		}
		indent := body[loc[2]:loc[3]]
		body = body[:loc[1]-len(indent)] + fmt.Sprintf("%skv_cache_memory_bytes: %d\n", indent, int64(math.Round(pinGiB*(1<<30)))) + body[loc[1]-len(indent):]
	}
	path := filepath.Join(t.TempDir(), scenario)
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	e, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: map[string]string{RootScenario: path}})
	if err != nil {
		t.Fatalf("%s: %v", scenario, err)
	}
	return e
}

func tightestTier(dev float64) float64 {
	for _, x := range kvStartupTiers {
		if dev <= x {
			return x
		}
	}
	return math.Inf(1)
}

// The KV cache size an engine core logs at start-up is asserted for every fit unit of the published
// ledger whose server log prints it (blis-vllm#73): the engine's own figure for the run's recorded
// scenario (Engine.KVCacheCapacity, vLLM's get_kv_cache_capacity) against the logged "GPU KV cache
// size: N tokens". Each group (model, chip, tensor and data parallel width) is held to the
// tightest tolerance of kvStartupTiers that holds today, in the file beside the scenarios.
//
// Two comparisons, both over the same recorded scenarios:
//
//   - end to end: the pool is sized from the kernel's own memory estimate. This is the figure a
//     simulated run has, and its gap is the memory terms' (weights, activations, graphs: SUB-14).
//   - pinned: the pool is sized from the run's logged "Available KV cache memory", so the gap that
//     is left is the pool's structure (the KV cache groups and their block bytes) and the
//     formula. A group whose logs were written by a vLLM older than 0.28 has a start-up figure
//     computed another way (get_max_concurrency_for_kv_cache_config divided by the largest group's
//     layers), so its pinned gap includes that: formula "older" in the file.
//
// The scenarios are the dashboard's own for each recorded run, from the run's recorded
// arguments; only the number of fit units behind each logged figure is kept, not their identity.
func TestEngineKVStartupFigureAgainstTheLoggedOne(t *testing.T) {
	Register()
	f := loadKVStartup(t)
	type dev struct{ e2e, pinned []float64 }
	seen := map[string]*dev{}
	var mu sync.Mutex
	for _, fx := range f.Fixtures {
		fx := fx
		if _, ok := f.Tolerance[fx.Group]; !ok {
			t.Fatalf("%s: group %s has no stated tolerance", fx.Scenario, fx.Group)
		}
		t.Run(strings.TrimSuffix(fx.Scenario, ".yaml"), func(t *testing.T) {
			t.Parallel()
			e2eTokens, _ := startupFigure(t, kvStartupEngine(t, fx.Scenario, 0))
			var e2e, pinned []float64
			for _, o := range fx.Observations {
				e2e = append(e2e, float64(e2eTokens)/float64(o.LoggedTokens)-1)
				tokens, _ := startupFigure(t, kvStartupEngine(t, fx.Scenario, o.LoggedAvailableGiB))
				pinned = append(pinned, float64(tokens)/float64(o.LoggedTokens)-1)
			}
			mu.Lock()
			d := seen[fx.Group]
			if d == nil {
				d = &dev{}
				seen[fx.Group] = d
			}
			d.e2e, d.pinned = append(d.e2e, e2e...), append(d.pinned, pinned...)
			mu.Unlock()
		})
	}
	t.Cleanup(func() {
		var groups []string
		for g := range seen {
			groups = append(groups, g)
		}
		sort.Strings(groups)
		for _, g := range groups {
			d, tol := seen[g], f.Tolerance[g]
			check := func(what string, devs []float64, stated float64) {
				worst := 0.0
				for _, x := range devs {
					worst = math.Max(worst, math.Abs(x))
				}
				switch {
				case worst > stated:
					t.Errorf("%s %s: the engine's KV figure is %.1f%% from the logged one, beyond the stated %.0f%%", g, what, worst*100, stated*100)
				case tightestTier(worst) < stated:
					t.Errorf("%s %s: the gap is %.1f%%, inside the %.0f%% tier, and the stated tolerance is the looser %.0f%%: state %.0f%%", g, what, worst*100, tightestTier(worst)*100, stated*100, tightestTier(worst)*100)
				}
			}
			check("end to end", d.e2e, tol.EndToEnd)
			if tol.Pinned != nil {
				check("pinned", d.pinned, *tol.Pinned)
			}
		}
	})
}

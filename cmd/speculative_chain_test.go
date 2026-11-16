package cmd

import (
	"path/filepath"
	"testing"
)

// chainSet is a profile set stating a chain schedule for the DSpark scenario's key, under a
// category of its own so it does not collide with the fixture's coding curve.
func chainSet(sd string) string {
	return `kind: AcceptanceProfileSet
name: chain-fixture
profiles:
- model: deepseek-v4.1-flash
  speculative_method: dspark
  num_spec_tokens: 5
  thinking: 'on'
  category: agentic
  acceptance_length: 3.2
  per_position_rates: [0.8, 0.6, 0.4, 0.25, 0.15]
  schedule: chain
  request_logit_sd: ` + sd + `
  method: assumed
  rationale: a test fixture
`
}

func chainRun(t *testing.T, sd string, extra ...string) (map[string]any, string) {
	t.Helper()
	registry, catalog := acceptanceRepos(t, "")
	mustWrite(t, filepath.Join(registry, "acceptance", "chain-fixture.yaml"), []byte(chainSet(sd)))
	args := append([]string{"run", "--num-requests", "24", "--rate", "5",
		"--speculative-thinking", "on", "--speculative-category", "agentic"}, extra...)
	rec, stdout, stderr, err := runAccept(t, dsparkVariant(t, 5, ""), registry, catalog, args...)
	if err != nil {
		t.Fatalf("run refused: %v\n%s", err, lastLines(stderr, 4))
	}
	out := map[string]any{"schedule": "", "sd": 0.0}
	if rec != nil {
		out["schedule"], out["sd"] = rec.Entry.Schedule, rec.Entry.RequestLogitSD
	}
	return out, stdout
}

// tpots is each completed request's TPOT, in row order.
func tpots(t *testing.T, stdout string) []float64 {
	t.Helper()
	rows, _ := resultDoc(t, stdout)["request_rows"].([]any)
	var out []float64
	for _, r := range rows {
		m, _ := r.(map[string]any)
		if v, ok := m["tpot_us"].(float64); ok {
			out = append(out, v)
		}
	}
	if len(out) == 0 {
		t.Fatal("the Result has no request rows with tpot_us")
	}
	return out
}

// A profile that states a chain reaches the engine: the run records the schedule and its
// spread, draws each step's acceptance, so its requests' TPOTs differ from the mean schedule's
// at the same AL, and gives the same bytes again for the same seed (INV-6).
func TestAChainProfileDrawsAcceptanceInTheRun(t *testing.T) {
	rec, chain := chainRun(t, "0.5")
	if rec["schedule"] != "chain" || rec["sd"] != 0.5 {
		t.Errorf("the record states schedule %v, request_logit_sd %v; want chain, 0.5", rec["schedule"], rec["sd"])
	}
	_, again := chainRun(t, "0.5")
	if !equalFloats(tpots(t, chain), tpots(t, again)) {
		t.Error("two runs with the same seed and a chain schedule gave different per-request TPOTs")
	}
	rec, mean := chainRun(t, "0.5", "--speculative-acceptance-length", "3.2")
	if rec["schedule"] != "" {
		t.Errorf("a command-line length recorded schedule %v, want the mean", rec["schedule"])
	}
	if equalFloats(tpots(t, chain), tpots(t, mean)) {
		t.Error("the chain run's per-request TPOTs equal the mean schedule's at the same AL: no draw reached the engine")
	}
}

func equalFloats(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

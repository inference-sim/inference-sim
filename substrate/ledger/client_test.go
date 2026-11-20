package ledger_test

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/ledger"
)

func clientSources() *ledger.Sources {
	s := apiRowsSources()
	s.Sources[0].Client = &ledger.Client{
		MeasuredPerConcurrency: 10, WarmupPerConcurrency: 2, LoggedPerConcurrency: 12,
		Evidence: []ledger.LicenceQuote{{
			URL: "https://github.com/o/r/blob/" + strings.Repeat("b", 40) + "/bench.sh", SHA256: strings.Repeat("d", 64), Lines: "7", Quote: "--num-prompts $((CONC * 10))"}},
	}
	return s
}

// A source's client multiples say how many requests a run logs per unit of concurrency, so a
// run's request count can be read against them (measurement_contaminated). The multiples are
// consistent only if the logged count is the measured plus the warm-up count.
func TestClientMultiplesAreConsistentAndEvidenced(t *testing.T) {
	if err := clientSources().Validate(); err != nil {
		t.Fatalf("consistent multiples were refused: %v", err)
	}
	for name, c := range map[string]struct {
		mut  func(c *ledger.Client)
		want string
	}{
		"logged is not measured plus warm-up": {func(c *ledger.Client) { c.LoggedPerConcurrency = 13 }, "measured plus warm-up"},
		"no measured multiple":                {func(c *ledger.Client) { c.MeasuredPerConcurrency = 0 }, "measured_per_concurrency"},
		"negative warm-up":                    {func(c *ledger.Client) { c.WarmupPerConcurrency = -2 }, "warmup_per_concurrency"},
		"no evidence":                         {func(c *ledger.Client) { c.Evidence = nil }, "evidence"},
		"evidence on a branch": {func(c *ledger.Client) {
			c.Evidence[0].URL = "https://github.com/o/r/blob/main/bench.sh"
		}, "does not name a commit"},
	} {
		s := clientSources()
		c.mut(s.Sources[0].Client)
		if err := s.Validate(); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error containing %q", name, err, c.want)
		}
	}
}

// InferenceX's client logs 12 requests per unit of concurrency: 10 measured and 2 warm-up.
func TestInferenceXClientMultiplesAreRecorded(t *testing.T) {
	src, err := ledger.LoadSources("../catalog/ledger")
	if err != nil {
		t.Fatal(err)
	}
	x := src.Lookup("inferencex")
	if x == nil || x.Client == nil {
		t.Fatal("the inferencex source records no client multiples")
	}
	c := x.Client
	if c.MeasuredPerConcurrency != 10 || c.WarmupPerConcurrency != 2 || c.LoggedPerConcurrency != 12 {
		t.Errorf("client multiples %d measured, %d warm-up, %d logged; want 10, 2, 12",
			c.MeasuredPerConcurrency, c.WarmupPerConcurrency, c.LoggedPerConcurrency)
	}
	if len(c.Evidence) < 2 {
		t.Errorf("%d evidence quotes; want the measured and the warm-up multiple each quoted", len(c.Evidence))
	}
}

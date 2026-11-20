package acceptance

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func sameRates(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !near(a[i], b[i]) {
			return false
		}
	}
	return true
}

// The expected lists are what vLLM v0.31.0's _acceptance_length_to_rates returns for the
// same arguments, including its truncation at AL = K+1.
func TestMinimumVarianceRates(t *testing.T) {
	for _, c := range []struct {
		name string
		al   float64
		k    int
		want []float64
	}{
		{"nothing accepted", 1, 3, []float64{0, 0, 0}},
		{"everything accepted truncates to K", 4, 3, []float64{1, 1, 1}},
		{"K=1 midpoint", 1.5, 1, []float64{0.5}},
		{"integer AL", 3, 5, []float64{1, 1, 0, 0, 0}},
		{"V4.1-Flash DSpark K=5 thinking off", 4.07, 5, []float64{1, 1, 1, 0.07, 0}},
		{"V4.1-Flash DSpark K=5 thinking on", 3.51, 5, []float64{1, 1, 0.51, 0, 0}},
		{"just below the cap", 8.99, 8, []float64{1, 1, 1, 1, 1, 1, 1, 0.99}},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := MinimumVarianceRates(c.al, c.k)
			if err != nil {
				t.Fatal(err)
			}
			if !sameRates(got, c.want) {
				t.Errorf("MinimumVarianceRates(%v, %d) = %v, want %v", c.al, c.k, got, c.want)
			}
		})
	}
}

func TestMinimumVarianceRatesRefuses(t *testing.T) {
	for _, c := range []struct {
		name string
		al   float64
		k    int
	}{
		{"K zero", 1, 0},
		{"below one", 0.99, 3},
		{"above K+1", 4.01, 3},
		{"NaN", math.NaN(), 3},
		{"+Inf", math.Inf(1), 3},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := MinimumVarianceRates(c.al, c.k); err == nil {
				t.Errorf("MinimumVarianceRates(%v, %d) accepted", c.al, c.k)
			}
		})
	}
}

// Every schedule the helper derives is one the schema accepts, and it sums back to AL-1:
// deriving rates and stating them must not disagree.
func TestMinimumVarianceRatesProperties(t *testing.T) {
	for k := 1; k <= 8; k++ {
		for step := 0; step <= 100; step++ {
			al := 1 + float64(k)*float64(step)/100
			rates, err := MinimumVarianceRates(al, k)
			if err != nil {
				t.Fatalf("AL %v K %d: %v", al, k, err)
			}
			if err := CheckRates(rates, k); err != nil {
				t.Errorf("AL %v K %d: derived rates rejected: %v", al, k, err)
			}
			if math.Abs(sum(rates)-(al-1)) > 1e-9 {
				t.Errorf("AL %v K %d: rates %v sum to %v, want %v", al, k, rates, sum(rates), al-1)
			}
		}
	}
}

func TestCheckRates(t *testing.T) {
	for _, c := range []struct {
		name  string
		rates []float64
		k     int
		ok    bool
	}{
		{"valid", []float64{0.9, 0.7, 0.4}, 3, true},
		{"equal neighbours", []float64{0.5, 0.5}, 2, true},
		{"wrong length", []float64{0.9, 0.7}, 3, false},
		{"rising", []float64{0.5, 0.6}, 2, false},
		{"above one", []float64{1.1}, 1, false},
		{"negative", []float64{0.5, -0.1}, 2, false},
		{"NaN", []float64{math.NaN()}, 1, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			if err := CheckRates(c.rates, c.k); (err == nil) != c.ok {
				t.Errorf("CheckRates(%v, %d) = %v, want ok=%v", c.rates, c.k, err, c.ok)
			}
		})
	}
}

// The acceptance rate is the accepted fraction (AL-1)/K, and the two conversions invert.
func TestRateAndLengthConvert(t *testing.T) {
	if got := RateFromLength(4.07, 5); !near(got, 0.614) {
		t.Errorf("RateFromLength(4.07, 5) = %v, want 0.614", got)
	}
	for _, al := range []float64{1, 2.5, 6} {
		if got := LengthFromRate(RateFromLength(al, 5), 5); !near(got, al) {
			t.Errorf("round trip of %v gave %v", al, got)
		}
	}
}

// TestMinimumVarianceScheduleMatchSource pins the helper to the engine function it
// mirrors. Without a checkout at VLLM_SOURCE it skips and says so.
func TestMinimumVarianceScheduleMatchSource(t *testing.T) {
	root := os.Getenv("VLLM_SOURCE")
	skip := t.Skipf
	if os.Getenv("VLLM_SOURCE_REQUIRED") != "" {
		skip = t.Fatalf
	}
	if root == "" {
		skip("VLLM_SOURCE is unset: MinimumVarianceRates was NOT checked against " +
			"vLLM v0.31.0's _acceptance_length_to_rates")
	}
	if _, err := os.Stat(filepath.Join(root, "vllm", "config")); err != nil {
		skip("VLLM_SOURCE=%s has no vllm/config: MinimumVarianceRates was NOT checked", root)
	}
	path := filepath.Join(root, "vllm", "config", "speculative.py")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	src := string(b)
	for _, want := range []string{
		"def _acceptance_length_to_rates(length: float, n: int) -> list[float]:",
		"num_drafts = length - 1",
		"num_full = int(num_drafts)",
		"[1.0] * num_full + [num_drafts - num_full] + [0.0] * (n - num_full - 1)",
		")[:n]",
		"if not 1.0 <= length <= float(n + 1):",
		"if not all(0.0 <= r <= 1.0 for r in rates):",
		"if any(rates[i] > rates[i - 1] for i in range(1, n)):",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("%s no longer contains %q; re-derive MinimumVarianceRates and CheckRates", path, want)
		}
	}
}

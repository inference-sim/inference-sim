package workload

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// testWindowStreams is a window's stream family for a test that calls generateRequestsForWindow
// directly: client 0's window 0 under seed.
func testWindowStreams(seed int64) sim.Streams {
	return windowStreams(clientStreams(seed, 0), 0)
}

// Each purpose is its own stream (DEV-6): a change to one client's output-length distribution
// leaves every request's arrival time and prompt length unchanged, and a change to the request
// cap leaves the requests both runs share unchanged. With one sequential stream per client, the
// first change moved every later arrival and prompt.
func TestNamedStreams_OnePurposeDoesNotMoveAnother(t *testing.T) {
	spec := func(outputMean float64) *WorkloadSpec {
		return &WorkloadSpec{
			Version: "2", Seed: 11, Category: "language", AggregateRate: 20,
			Clients: []ClientSpec{
				{ID: "a", SLOClass: "batch", RateFraction: 0.5, Arrival: ArrivalSpec{Process: "poisson"},
					InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 200, "std_dev": 50, "min": 10, "max": 600}},
					OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": outputMean}}},
				{ID: "b", SLOClass: "batch", RateFraction: 0.5, Arrival: ArrivalSpec{Process: "poisson"},
					InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 300, "std_dev": 50, "min": 10, "max": 600}},
					OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 80}}},
			},
		}
	}
	gen := func(outputMean float64, n int64) []*sim.Request {
		reqs, err := GenerateRequests(spec(outputMean), 1<<40, n)
		if err != nil {
			t.Fatal(err)
		}
		return reqs
	}
	base, moved := gen(64, 200), gen(512, 200)
	if len(base) != len(moved) {
		t.Fatalf("request counts %d and %d", len(base), len(moved))
	}
	outputsDiffer := 0
	for i := range base {
		a, b := base[i], moved[i]
		if a.ArrivalTime != b.ArrivalTime || a.ClientID != b.ClientID || a.InputLen() != b.InputLen() {
			t.Fatalf("request %d: arrival %d/%d, client %s/%s, input %d/%d: an output-length change moved another draw",
				i, a.ArrivalTime, b.ArrivalTime, a.ClientID, b.ClientID, a.InputLen(), b.InputLen())
		}
		if a.ClientID == "b" && len(a.OutputTokens) != len(b.OutputTokens) {
			t.Fatalf("request %d of client b: output %d/%d, though only client a's distribution changed", i, len(a.OutputTokens), len(b.OutputTokens))
		}
		if len(a.OutputTokens) != len(b.OutputTokens) {
			outputsDiffer++
		}
	}
	if outputsDiffer == 0 {
		t.Fatal("no output length changed: the test exercises nothing")
	}
	short := gen(64, 50)
	for i := range short {
		if short[i].ArrivalTime != base[i].ArrivalTime || short[i].InputLen() != base[i].InputLen() ||
			len(short[i].OutputTokens) != len(base[i].OutputTokens) {
			t.Fatalf("request %d differs between a 50- and a 200-request run", i)
		}
	}
}

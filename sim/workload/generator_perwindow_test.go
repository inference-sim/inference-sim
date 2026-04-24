package workload

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRescaleIATsToMatchDuration(t *testing.T) {
	t.Run("rescale to 10 second window", func(t *testing.T) {
		// IATs sum to 20 seconds (20,000,000 us)
		iats := []int64{5000000, 5000000, 5000000, 5000000} // 5s each
		targetDuration := int64(10000000)                     // 10 seconds

		rescaled := rescaleIATsToMatchDuration(iats, targetDuration)

		// Sum should equal target duration
		sum := int64(0)
		for _, iat := range rescaled {
			sum += iat
		}
		assert.Equal(t, targetDuration, sum)

		// CV should be preserved (all equal -> CV=0)
		for i := 1; i < len(rescaled); i++ {
			assert.Equal(t, rescaled[0], rescaled[i], "uniform scaling preserves equal values")
		}
	})

	t.Run("rescale preserves relative ratios", func(t *testing.T) {
		// Unequal IATs
		iats := []int64{1000000, 3000000, 6000000} // 1s, 3s, 6s (sum=10s)
		targetDuration := int64(20000000)            // 20 seconds

		rescaled := rescaleIATsToMatchDuration(iats, targetDuration)

		// Check sum
		sum := int64(0)
		for _, iat := range rescaled {
			sum += iat
		}
		assert.Equal(t, targetDuration, sum)

		// Check ratios preserved: rescaled[1]/rescaled[0] should equal iats[1]/iats[0]
		originalRatio := float64(iats[1]) / float64(iats[0])
		rescaledRatio := float64(rescaled[1]) / float64(rescaled[0])
		assert.InDelta(t, originalRatio, rescaledRatio, 0.01, "ratio 3:1 preserved")
	})

	t.Run("zero sum IATs returns zeros", func(t *testing.T) {
		iats := []int64{0, 0, 0}
		targetDuration := int64(10000000)

		rescaled := rescaleIATsToMatchDuration(iats, targetDuration)

		for _, iat := range rescaled {
			assert.Equal(t, int64(0), iat)
		}
	})

	t.Run("empty input returns nil", func(t *testing.T) {
		rescaled := rescaleIATsToMatchDuration(nil, 10000000)
		assert.Nil(t, rescaled)

		rescaled = rescaleIATsToMatchDuration([]int64{}, 10000000)
		assert.Nil(t, rescaled)
	})

	t.Run("rounding residual distributed to last element", func(t *testing.T) {
		// Choose values where float64 truncation causes rounding loss
		iats := []int64{3333333, 3333333, 3333334} // sum = 10000000
		targetDuration := int64(5000000)             // scale factor = 0.5

		rescaled := rescaleIATsToMatchDuration(iats, targetDuration)

		// Sum must exactly equal target despite int64 truncation
		sum := int64(0)
		for _, iat := range rescaled {
			sum += iat
		}
		assert.Equal(t, targetDuration, sum, "sum must exactly match target duration after rounding correction")
	})

	t.Run("single element rescales exactly", func(t *testing.T) {
		iats := []int64{7000000}
		targetDuration := int64(3000000)

		rescaled := rescaleIATsToMatchDuration(iats, targetDuration)

		assert.Equal(t, 1, len(rescaled))
		assert.Equal(t, targetDuration, rescaled[0])
	})

	t.Run("does not mutate input slice", func(t *testing.T) {
		iats := []int64{2000000, 3000000, 5000000}
		original := make([]int64, len(iats))
		copy(original, iats)
		targetDuration := int64(20000000)

		_ = rescaleIATsToMatchDuration(iats, targetDuration)

		assert.Equal(t, original, iats, "original slice must not be mutated")
	})
}

func TestResolveWindowParameters(t *testing.T) {
	clientArrival := ArrivalSpec{Process: "poisson"}
	clientInputDist := DistSpec{Type: "constant", Params: map[string]float64{"value": 100}}
	clientOutputDist := DistSpec{Type: "constant", Params: map[string]float64{"value": 50}}
	clientRateFraction := 1.0

	client := ClientSpec{
		Arrival:      clientArrival,
		InputDist:    clientInputDist,
		OutputDist:   clientOutputDist,
		RateFraction: clientRateFraction,
	}

	t.Run("window overrides all parameters", func(t *testing.T) {
		windowArrival := ArrivalSpec{Process: "gamma", Shape: ptrFloat64(2.0), Scale: ptrFloat64(1000.0)}
		windowInputDist := DistSpec{Type: "constant", Params: map[string]float64{"value": 200}}
		windowOutputDist := DistSpec{Type: "constant", Params: map[string]float64{"value": 75}}
		windowTraceRate := 15.2

		window := ActiveWindow{
			StartUs:    0,
			EndUs:      10000000,
			TraceRate:  &windowTraceRate,
			Arrival:    &windowArrival,
			InputDist:  &windowInputDist,
			OutputDist: &windowOutputDist,
		}

		arrival, input, output, traceRate := resolveWindowParameters(client, window)

		assert.Equal(t, "gamma", arrival.Process)
		assert.Equal(t, 2.0, *arrival.Shape)
		assert.Equal(t, "constant", input.Type)
		assert.Equal(t, 200.0, input.Params["value"])
		assert.Equal(t, "constant", output.Type)
		assert.Equal(t, 75.0, output.Params["value"])
		assert.Equal(t, 15.2, traceRate)
	})

	t.Run("window fallback to client parameters", func(t *testing.T) {
		window := ActiveWindow{
			StartUs: 0,
			EndUs:   10000000,
			// No overrides - should use client-level
		}

		arrival, input, output, traceRate := resolveWindowParameters(client, window)

		assert.Equal(t, "poisson", arrival.Process)
		assert.Equal(t, "constant", input.Type)
		assert.Equal(t, 100.0, input.Params["value"])
		assert.Equal(t, "constant", output.Type)
		assert.Equal(t, 50.0, output.Params["value"])
		assert.Equal(t, 1.0, traceRate)
	})

	t.Run("partial window overrides", func(t *testing.T) {
		windowTraceRate := 8.5
		windowArrival := ArrivalSpec{Process: "gamma"}

		window := ActiveWindow{
			StartUs:   0,
			EndUs:     10000000,
			TraceRate: &windowTraceRate,
			Arrival:   &windowArrival,
			// InputDist and OutputDist are nil - should use client-level
		}

		arrival, input, output, traceRate := resolveWindowParameters(client, window)

		assert.Equal(t, "gamma", arrival.Process)
		assert.Equal(t, "constant", input.Type)
		assert.Equal(t, 100.0, input.Params["value"])
		assert.Equal(t, "constant", output.Type)
		assert.Equal(t, 50.0, output.Params["value"])
		assert.Equal(t, 8.5, traceRate)
	})
}

func TestComputeProportionalRate(t *testing.T) {
	t.Run("three co-active clients with trace rates", func(t *testing.T) {
		// ServeGen scenario from spec: chunk-2 (15.2), chunk-8 (22.5), chunk-20 (5.3)
		// At timestamp 0-10s, all three overlap
		clients := []ClientSpec{
			{ID: "chunk-2", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 10000000, TraceRate: ptrFloat64(15.2)}},
			}},
			{ID: "chunk-8", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 10000000, TraceRate: ptrFloat64(22.5)}},
			}},
			{ID: "chunk-20", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 10000000, TraceRate: ptrFloat64(5.3)}},
			}},
		}

		aggregateRate := 150.0
		window := clients[1].Lifecycle.Windows[0] // chunk-8's window

		allocatedRate := computeProportionalRate(clients[1], window, clients, aggregateRate)

		// Expected: 150 * (22.5 / (15.2+22.5+5.3)) = 150 * (22.5 / 43.0) = 78.49
		assert.InDelta(t, 78.49, allocatedRate, 0.01)
	})

	t.Run("non-overlapping windows", func(t *testing.T) {
		// chunk-2 active 0-10s, chunk-8 active 20-30s
		clients := []ClientSpec{
			{ID: "chunk-2", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 10000000, TraceRate: ptrFloat64(15.2)}},
			}},
			{ID: "chunk-8", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 20000000, EndUs: 30000000, TraceRate: ptrFloat64(22.5)}},
			}},
		}

		aggregateRate := 150.0
		window := clients[1].Lifecycle.Windows[0] // chunk-8's window (20-30s)

		allocatedRate := computeProportionalRate(clients[1], window, clients, aggregateRate)

		// Expected: 150 * (22.5 / 22.5) = 150 (only chunk-8 active)
		assert.InDelta(t, 150.0, allocatedRate, 0.01)
	})

	t.Run("always-on client (no lifecycle)", func(t *testing.T) {
		// chunk-8 has lifecycle, background client always-on
		clients := []ClientSpec{
			{ID: "background", RateFraction: 10.0, Lifecycle: nil}, // Always-on
			{ID: "chunk-8", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 10000000, TraceRate: ptrFloat64(22.5)}},
			}},
		}

		aggregateRate := 150.0
		window := clients[1].Lifecycle.Windows[0]

		allocatedRate := computeProportionalRate(clients[1], window, clients, aggregateRate)

		// Expected: 150 * (22.5 / (10.0+22.5)) = 150 * (22.5 / 32.5) = 103.85
		assert.InDelta(t, 103.85, allocatedRate, 0.01)
	})

	t.Run("zero total trace rate returns zero", func(t *testing.T) {
		clients := []ClientSpec{
			{ID: "chunk-1", RateFraction: 0.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 10000000, TraceRate: ptrFloat64(0.0)}},
			}},
		}

		allocatedRate := computeProportionalRate(clients[0], clients[0].Lifecycle.Windows[0], clients, 150.0)

		assert.Equal(t, 0.0, allocatedRate)
	})

	t.Run("single client gets full rate", func(t *testing.T) {
		clients := []ClientSpec{
			{ID: "chunk-1", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 10000000, TraceRate: ptrFloat64(42.0)}},
			}},
		}

		aggregateRate := 200.0
		allocatedRate := computeProportionalRate(clients[0], clients[0].Lifecycle.Windows[0], clients, aggregateRate)

		// Single client: 200 * (42.0 / 42.0) = 200
		assert.InDelta(t, 200.0, allocatedRate, 0.01)
	})

	t.Run("multiple windows per client partial overlap", func(t *testing.T) {
		// Client A has two windows: 0-10s and 20-30s
		// Client B has one window: 5-25s (overlaps with both A windows)
		// Query: Client B's window rate when overlapping with both A windows
		clients := []ClientSpec{
			{ID: "clientA", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{
					{StartUs: 0, EndUs: 10000000, TraceRate: ptrFloat64(10.0)},
					{StartUs: 20000000, EndUs: 30000000, TraceRate: ptrFloat64(20.0)},
				},
			}},
			{ID: "clientB", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{
					{StartUs: 5000000, EndUs: 25000000, TraceRate: ptrFloat64(30.0)},
				},
			}},
		}

		aggregateRate := 100.0
		window := clients[1].Lifecycle.Windows[0] // clientB's window (5-25s)

		allocatedRate := computeProportionalRate(clients[1], window, clients, aggregateRate)

		// clientA's first window (0-10s) overlaps with clientB's (5-25s): traceRate=10.0
		// clientA's second window (20-30s) overlaps with clientB's (5-25s): traceRate=20.0
		// Only count clientA once (first overlapping window): totalRate = 10.0 + 30.0 = 40.0
		// allocatedRate = 100 * (30.0 / 40.0) = 75.0
		assert.InDelta(t, 75.0, allocatedRate, 0.01)
	})
}

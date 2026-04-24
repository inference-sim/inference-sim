package workload

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

func TestHasPerWindowParameters(t *testing.T) {
	t.Run("returns true when any window has TraceRate", func(t *testing.T) {
		traceRate := 15.2
		clients := []ClientSpec{
			{
				ID:           "client-1",
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{
						{StartUs: 0, EndUs: 10000000, TraceRate: &traceRate},
					},
				},
			},
		}
		assert.True(t, hasPerWindowParameters(clients))
	})

	t.Run("returns true when any window has InputDist", func(t *testing.T) {
		inputDist := DistSpec{Type: "constant", Params: map[string]float64{"value": 200}}
		clients := []ClientSpec{
			{
				ID:           "client-1",
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{
						{StartUs: 0, EndUs: 10000000, InputDist: &inputDist},
					},
				},
			},
		}
		assert.True(t, hasPerWindowParameters(clients))
	})

	t.Run("returns false when no per-window params exist", func(t *testing.T) {
		clients := []ClientSpec{
			{
				ID:           "client-1",
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{
						{StartUs: 0, EndUs: 10000000}, // No per-window overrides
					},
				},
			},
		}
		assert.False(t, hasPerWindowParameters(clients))
	})

	t.Run("returns false when no lifecycle", func(t *testing.T) {
		clients := []ClientSpec{
			{
				ID:           "client-1",
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			},
		}
		assert.False(t, hasPerWindowParameters(clients))
	})

	t.Run("returns false for empty clients", func(t *testing.T) {
		assert.False(t, hasPerWindowParameters(nil))
		assert.False(t, hasPerWindowParameters([]ClientSpec{}))
	})
}

func TestGenerateRequests_TimeVaryingWorkload(t *testing.T) {
	t.Run("detects per-window parameters and routes to time-varying generator", func(t *testing.T) {
		spec := &WorkloadSpec{
			Version:       "2",
			AggregateRate: 100,
			Seed:          42,
			Clients: []ClientSpec{
				{
					ID:           "client-1",
					TenantID:     "tenant-1",
					SLOClass:     "standard",
					RateFraction: 1.0,
					Streaming:    true,
					Arrival:      ArrivalSpec{Process: "poisson"},
					InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
					OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{
								StartUs:   0,
								EndUs:     10000000,
								TraceRate: ptrFloat64(100.0),
							},
						},
					},
				},
			},
		}

		requests, err := GenerateRequests(spec, 10000000, 0)
		require.NoError(t, err)

		// Should generate requests
		require.Greater(t, len(requests), 0, "should generate requests")

		// Check requests are within window
		for _, req := range requests {
			assert.GreaterOrEqual(t, req.ArrivalTime, int64(0))
			assert.Less(t, req.ArrivalTime, int64(10000000))
		}

		// Check IDs are assigned sequentially
		assert.Equal(t, "request_0", requests[0].ID)

		// Check requests are sorted by arrival time
		for i := 1; i < len(requests); i++ {
			assert.LessOrEqual(t, requests[i-1].ArrivalTime, requests[i].ArrivalTime)
		}

		// Check metadata
		assert.Equal(t, "client-1", requests[0].ClientID)
		assert.Equal(t, "tenant-1", requests[0].TenantID)
		assert.Equal(t, "standard", requests[0].SLOClass)
	})

	t.Run("falls back to static generator when no per-window params", func(t *testing.T) {
		spec := &WorkloadSpec{
			Version:       "2",
			AggregateRate: 100,
			Seed:          42,
			Clients: []ClientSpec{
				{
					ID:           "client-1",
					TenantID:     "tenant-1",
					RateFraction: 1.0,
					Arrival:      ArrivalSpec{Process: "poisson"},
					InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
					OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
					// No lifecycle windows - should use static generator
				},
			},
		}

		requests, err := GenerateRequests(spec, 10000000, 0)
		require.NoError(t, err)

		// Should generate requests using existing static generator
		assert.Greater(t, len(requests), 0)
	})

	t.Run("determinism: same seed produces same output", func(t *testing.T) {
		makeSpec := func() *WorkloadSpec {
			return &WorkloadSpec{
				Version:       "2",
				AggregateRate: 50,
				Seed:          99,
				Clients: []ClientSpec{
					{
						ID:           "client-1",
						RateFraction: 1.0,
						Arrival:      ArrivalSpec{Process: "poisson"},
						InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
						OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
						Lifecycle: &LifecycleSpec{
							Windows: []ActiveWindow{
								{
									StartUs:   0,
									EndUs:     5000000,
									TraceRate: ptrFloat64(50.0),
								},
							},
						},
					},
				},
			}
		}

		reqs1, err1 := GenerateRequests(makeSpec(), 5000000, 0)
		require.NoError(t, err1)

		reqs2, err2 := GenerateRequests(makeSpec(), 5000000, 0)
		require.NoError(t, err2)

		require.Equal(t, len(reqs1), len(reqs2), "same seed must produce same count")
		for i := range reqs1 {
			assert.Equal(t, reqs1[i].ArrivalTime, reqs2[i].ArrivalTime, "arrival times must match at index %d", i)
			assert.Equal(t, len(reqs1[i].InputTokens), len(reqs2[i].InputTokens), "input token counts must match at index %d", i)
		}
	})

	t.Run("maxRequests cap applied", func(t *testing.T) {
		spec := &WorkloadSpec{
			Version:       "2",
			AggregateRate: 100,
			Seed:          42,
			Clients: []ClientSpec{
				{
					ID:           "client-1",
					RateFraction: 1.0,
					Arrival:      ArrivalSpec{Process: "poisson"},
					InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 10}},
					OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 10}},
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{
								StartUs:   0,
								EndUs:     10000000,
								TraceRate: ptrFloat64(100.0),
							},
						},
					},
				},
			},
		}

		requests, err := GenerateRequests(spec, 10000000, 50)
		require.NoError(t, err)
		assert.LessOrEqual(t, len(requests), 50, "maxRequests cap must be respected")
	})

	t.Run("multi-window client generates requests across all windows", func(t *testing.T) {
		spec := &WorkloadSpec{
			Version:       "2",
			AggregateRate: 50,
			Seed:          42,
			Clients: []ClientSpec{
				{
					ID:           "multi-window",
					TenantID:     "tenant-1",
					RateFraction: 1.0,
					Arrival:      ArrivalSpec{Process: "poisson"},
					InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
					OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{
								StartUs:   0,
								EndUs:     5000000,
								TraceRate: ptrFloat64(30.0),
							},
							{
								StartUs:   10000000,
								EndUs:     15000000,
								TraceRate: ptrFloat64(20.0),
							},
						},
					},
				},
			},
		}

		requests, err := GenerateRequests(spec, 20000000, 0)
		require.NoError(t, err)
		require.Greater(t, len(requests), 0)

		// Count requests per window
		window1Count := 0
		window2Count := 0
		for _, req := range requests {
			if req.ArrivalTime >= 0 && req.ArrivalTime < 5000000 {
				window1Count++
			}
			if req.ArrivalTime >= 10000000 && req.ArrivalTime < 15000000 {
				window2Count++
			}
		}

		assert.Greater(t, window1Count, 0, "window 1 should have requests")
		assert.Greater(t, window2Count, 0, "window 2 should have requests")

		// No requests in the gap (5s-10s)
		for _, req := range requests {
			if req.ArrivalTime >= 5000000 && req.ArrivalTime < 10000000 {
				t.Errorf("unexpected request in gap between windows at time %d", req.ArrivalTime)
			}
		}
	})
}

func TestGenerateRequestsForWindow(t *testing.T) {
	t.Run("single window with per-window distributions", func(t *testing.T) {
		// Single client with window-level overrides for arrival, input, and output.
		// The function should use the window distributions, not client-level.
		clients := []ClientSpec{
			{
				ID:           "test-client",
				TenantID:     "tenant-1",
				SLOClass:     "standard",
				RateFraction: 1.0,
				Streaming:    true,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{
						{
							StartUs:    0,
							EndUs:      10000000, // 10 seconds
							TraceRate:  ptrFloat64(100.0),
							Arrival:    &ArrivalSpec{Process: "gamma", Shape: ptrFloat64(2.0), Scale: ptrFloat64(50000.0)},
							InputDist:  &DistSpec{Type: "constant", Params: map[string]float64{"value": 200}},
							OutputDist: &DistSpec{Type: "constant", Params: map[string]float64{"value": 75}},
						},
					},
				},
			},
		}

		aggregateRate := 100.0
		rng := rand.New(rand.NewSource(42))

		window := clients[0].Lifecycle.Windows[0]
		requests := generateRequestsForWindow(clients[0], window, clients, aggregateRate, rng)

		// Should generate requests
		require.Greater(t, len(requests), 0, "should generate requests")

		// Check window-specific distributions were used (constant 200 input, 75 output)
		for _, req := range requests {
			assert.Len(t, req.InputTokens, 200, "should use window's input dist (200 tokens)")
			assert.Len(t, req.OutputTokens, 75, "should use window's output dist (75 tokens)")
		}

		// Check all requests within window bounds
		for _, req := range requests {
			assert.GreaterOrEqual(t, req.ArrivalTime, window.StartUs,
				"request arrival must be >= window start")
			assert.Less(t, req.ArrivalTime, window.EndUs,
				"request arrival must be < window end")
		}

		// Check client metadata propagated
		assert.Equal(t, "test-client", requests[0].ClientID)
		assert.Equal(t, "tenant-1", requests[0].TenantID)
		assert.Equal(t, "standard", requests[0].SLOClass)
		assert.True(t, requests[0].Streaming)
	})

	t.Run("IAT rescaling achieves target rate", func(t *testing.T) {
		// Single client with 10s window at 50 req/s = ~500 requests expected.
		clients := []ClientSpec{
			{
				ID:           "rate-client",
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 10}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 10}},
				Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{
						{
							StartUs:   0,
							EndUs:     10000000, // 10 seconds
							TraceRate: ptrFloat64(50.0),
						},
					},
				},
			},
		}

		aggregateRate := 50.0
		rng := rand.New(rand.NewSource(123))

		window := clients[0].Lifecycle.Windows[0]
		requests := generateRequestsForWindow(clients[0], window, clients, aggregateRate, rng)

		// Check achieved rate is close to target (50 req/s for 10s = 500 requests)
		windowDurationSec := float64(window.EndUs-window.StartUs) / 1e6
		expectedRequests := int(aggregateRate * windowDurationSec)

		// Allow +/- 5% tolerance due to rounding and edge effects
		assert.InDelta(t, expectedRequests, len(requests), float64(expectedRequests)*0.05,
			"request count should be within 5%% of expected %d", expectedRequests)
	})

	t.Run("zero allocated rate produces no requests", func(t *testing.T) {
		// Two clients, both with trace_rate=0 -> allocated rate = 0
		clients := []ClientSpec{
			{
				ID:           "zero-rate",
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 10}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 10}},
				Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{
						{
							StartUs:   0,
							EndUs:     10000000,
							TraceRate: ptrFloat64(0.0),
						},
					},
				},
			},
		}

		rng := rand.New(rand.NewSource(42))
		window := clients[0].Lifecycle.Windows[0]
		requests := generateRequestsForWindow(clients[0], window, clients, 100.0, rng)

		assert.Empty(t, requests, "zero trace_rate should produce no requests")
	})

	t.Run("arrival times are monotonically increasing", func(t *testing.T) {
		clients := []ClientSpec{
			{
				ID:           "mono-client",
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 10}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 10}},
				Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{
						{
							StartUs:   5000000,
							EndUs:     15000000,
							TraceRate: ptrFloat64(20.0),
						},
					},
				},
			},
		}

		rng := rand.New(rand.NewSource(99))
		window := clients[0].Lifecycle.Windows[0]
		requests := generateRequestsForWindow(clients[0], window, clients, 20.0, rng)

		require.Greater(t, len(requests), 1, "need multiple requests for monotonicity check")

		for i := 1; i < len(requests); i++ {
			assert.LessOrEqual(t, requests[i-1].ArrivalTime, requests[i].ArrivalTime,
				"arrival times must be monotonically non-decreasing (index %d)", i)
		}
	})

	t.Run("non-zero start window offsets arrivals correctly", func(t *testing.T) {
		// Window starts at 5s, not 0.
		clients := []ClientSpec{
			{
				ID:           "offset-client",
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 10}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 10}},
				Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{
						{
							StartUs:   5000000,  // 5 seconds
							EndUs:     15000000, // 15 seconds
							TraceRate: ptrFloat64(10.0),
						},
					},
				},
			},
		}

		rng := rand.New(rand.NewSource(42))
		window := clients[0].Lifecycle.Windows[0]
		requests := generateRequestsForWindow(clients[0], window, clients, 10.0, rng)

		require.Greater(t, len(requests), 0)

		// All arrivals should be >= 5s and < 15s
		for _, req := range requests {
			assert.GreaterOrEqual(t, req.ArrivalTime, int64(5000000),
				"all arrivals must be >= window start (5s)")
			assert.Less(t, req.ArrivalTime, int64(15000000),
				"all arrivals must be < window end (15s)")
		}
	})

	t.Run("MaxOutputLen matches output token count", func(t *testing.T) {
		clients := []ClientSpec{
			{
				ID:           "maxout-client",
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 30}},
				Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{
						{
							StartUs:   0,
							EndUs:     5000000,
							TraceRate: ptrFloat64(20.0),
						},
					},
				},
			},
		}

		rng := rand.New(rand.NewSource(42))
		window := clients[0].Lifecycle.Windows[0]
		requests := generateRequestsForWindow(clients[0], window, clients, 20.0, rng)

		require.Greater(t, len(requests), 0)
		for _, req := range requests {
			assert.Equal(t, len(req.OutputTokens), req.MaxOutputLen,
				"MaxOutputLen must equal len(OutputTokens)")
		}
	})
}

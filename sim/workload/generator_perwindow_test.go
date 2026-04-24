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

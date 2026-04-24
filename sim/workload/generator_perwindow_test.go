package workload

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

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

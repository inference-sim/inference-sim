package workload

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFitLognormalFromPDF_KnownValue(t *testing.T) {
	t.Run("single bin PDF", func(t *testing.T) {
		// Single bin at 128 with probability 1.0
		// Expected: mu = ln(128), sigma = 0
		pdf := map[int]float64{128: 1.0}
		dist, err := fitLognormalFromPDF(pdf)
		require.NoError(t, err)
		assert.Equal(t, "lognormal", dist.Type)
		assert.InDelta(t, math.Log(128), dist.Params["mu"], 1e-10)
		assert.InDelta(t, 0.0, dist.Params["sigma"], 1e-10)
	})

	t.Run("two bin PDF", func(t *testing.T) {
		// Two bins: 100 and 200, each with probability 0.5
		// Expected: mu = 0.5*ln(100) + 0.5*ln(200) = ln(sqrt(20000))
		// variance = 0.5*(ln(100)^2 + ln(200)^2) - mu^2
		pdf := map[int]float64{100: 0.5, 200: 0.5}
		dist, err := fitLognormalFromPDF(pdf)
		require.NoError(t, err)

		expectedMu := 0.5*math.Log(100) + 0.5*math.Log(200)
		logX1 := math.Log(100)
		logX2 := math.Log(200)
		expectedVariance := 0.5*(logX1*logX1+logX2*logX2) - expectedMu*expectedMu
		expectedSigma := math.Sqrt(expectedVariance)

		assert.InDelta(t, expectedMu, dist.Params["mu"], 1e-10)
		assert.InDelta(t, expectedSigma, dist.Params["sigma"], 1e-10)
	})
}

func TestFitLognormalFromPDF_ErrorCases(t *testing.T) {
	t.Run("empty PDF", func(t *testing.T) {
		pdf := map[int]float64{}
		_, err := fitLognormalFromPDF(pdf)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "empty PDF")
	})

	t.Run("negative probability", func(t *testing.T) {
		pdf := map[int]float64{100: 0.5, 200: -0.3}
		_, err := fitLognormalFromPDF(pdf)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "negative probability")
	})

	t.Run("zero sum probabilities", func(t *testing.T) {
		pdf := map[int]float64{100: 0.0, 200: 0.0}
		_, err := fitLognormalFromPDF(pdf)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "sum of probabilities is zero")
	})

	t.Run("all zero/negative values filtered", func(t *testing.T) {
		pdf := map[int]float64{0: 0.5, -10: 0.5}
		_, err := fitLognormalFromPDF(pdf)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "sum of probabilities is zero")
	})
}

func TestFitLognormalFromPDF_Determinism(t *testing.T) {
	// R2 invariant: same Go map (non-deterministic iteration) must produce
	// identical mu/sigma across multiple calls. This verifies that
	// sort.Ints(keys) guards against map iteration order affecting float
	// accumulation.
	pdf := map[int]float64{
		100: 0.2,
		200: 0.3,
		300: 0.25,
		400: 0.15,
		500: 0.1,
	}

	// Call fitLognormalFromPDF multiple times
	const numTrials = 10
	var mus []float64
	var sigmas []float64

	for i := 0; i < numTrials; i++ {
		dist, err := fitLognormalFromPDF(pdf)
		require.NoError(t, err)
		mus = append(mus, dist.Params["mu"])
		sigmas = append(sigmas, dist.Params["sigma"])
	}

	// All mu values should be identical
	for i := 1; i < numTrials; i++ {
		assert.Equal(t, mus[0], mus[i], "mu must be deterministic across calls (trial %d)", i)
	}

	// All sigma values should be identical
	for i := 1; i < numTrials; i++ {
		assert.Equal(t, sigmas[0], sigmas[i], "sigma must be deterministic across calls (trial %d)", i)
	}
}

func TestFitLognormalFromPDF_ZeroProbabilityFiltering(t *testing.T) {
	// Bins with zero probability should be skipped
	pdf := map[int]float64{
		100: 0.5,
		200: 0.0, // Should be filtered
		300: 0.5,
	}

	dist, err := fitLognormalFromPDF(pdf)
	require.NoError(t, err)

	// Expected: same result as if 200 wasn't in the PDF
	pdfWithout200 := map[int]float64{100: 0.5, 300: 0.5}
	distExpected, err := fitLognormalFromPDF(pdfWithout200)
	require.NoError(t, err)

	assert.InDelta(t, distExpected.Params["mu"], dist.Params["mu"], 1e-10)
	assert.InDelta(t, distExpected.Params["sigma"], dist.Params["sigma"], 1e-10)
}

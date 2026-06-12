package latency

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// dpepTestHW returns a hardware calib usable for the DP/EP step-time tests
// (trained-physics StepTime needs TFlopsPeak + BwPeakTBs; MemoryGiB is only used
// by KV-capacity sizing, not step time).
func dpepTestHW() sim.HardwareCalib {
	return sim.HardwareCalib{TFlopsPeak: 989.0, BwPeakTBs: 3.35, MfuPrefill: 0.55, MfuDecode: 0.30}
}

// dpepMixedBatch is the fixed batch the INV BC-DP1 golden is captured against:
// 3 prefill requests (128 tokens) + 5 decode requests (256-token context).
func dpepMixedBatch() []*sim.Request {
	return append(makePrefillBatch(3, 128), makeDecodeBatch(5, 256)...)
}

// TestBetaEP_DefaultsToBeta4 verifies the β_EP (Beta[10]) default wiring (#1419):
// a caller providing <11 beta coefficients gets β_EP defaulted to β₄ (Beta[3]) — a
// derived default (both MoE comm families share the NVLink collective efficiency β₄
// encodes; the volume difference lives in the dispatch basis). A passive zero-fill
// would instead silently disable MoE dispatch comm. An explicit 11th coeff overrides.
func TestBetaEP_DefaultsToBeta4(t *testing.T) {
	mc := trainedPhysicsTestModelConfig()
	hw := dpepTestHW()
	mhw := sim.NewModelHardwareConfig(*mc, hw, "m", "H100", 1, 1, false, "", "trained-physics", 0)

	// 10-coeff caller: β_EP must default to β₄.
	c10 := &sim.LatencyCoeffs{
		AlphaCoeffs: []float64{15563.199579, 777.3455, 45.907545},
		BetaCoeffs:  []float64{0.152128, 0.0, 1.36252915, 0.752037, 32.09546717, 4.41684444, 126.024825, 481.8613888, 0.0, 1.94710771},
	}
	m10, err := NewTrainedPhysicsModel(*c10, mhw)
	require.NoError(t, err)
	require.Len(t, m10.Beta, 11, "Beta slice must grow to 11 for β_EP")
	assert.Equal(t, m10.Beta[3], m10.Beta[10], "β_EP must default to β₄ when not provided")

	// 11-coeff caller: β_EP must be the provided value (not β₄).
	c11 := &sim.LatencyCoeffs{
		AlphaCoeffs: c10.AlphaCoeffs,
		BetaCoeffs:  append(append([]float64{}, c10.BetaCoeffs...), 0.9999),
	}
	m11, err := NewTrainedPhysicsModel(*c11, mhw)
	require.NoError(t, err)
	assert.InDelta(t, 0.9999, m11.Beta[10], 1e-12, "explicit β_EP must override the β₄ default")
	assert.NotEqual(t, m11.Beta[3], m11.Beta[10], "provided β_EP differs from β₄ here")
}

// TestNewTrainedPhysicsModel_RejectsUnknownCommBackend verifies the constructor
// surfaces an unknown --moe-comm-backend as an error (R1), not a silent fallback.
func TestNewTrainedPhysicsModel_RejectsUnknownCommBackend(t *testing.T) {
	mc := trainedPhysicsTestModelConfig()
	hw := dpepTestHW()
	mhw := sim.NewModelHardwareConfig(*mc, hw, "m", "H100", 1, 1, false, "not-a-backend", "trained-physics", 0)
	_, err := NewTrainedPhysicsModel(*testCoeffs(), mhw)
	require.Error(t, err, "unknown MoE comm backend must be rejected")
	assert.Contains(t, err.Error(), "not-a-backend")
}

// TestINVBCDP1_DenseStepTimeByteIdentical is the INV BC-DP1 golden (issue #1419):
// for a DENSE model at DP=1 (EP off), the #C refactor must be byte-identical to the
// pre-#C step time across the TP matrix. The TP-comm split (monolithic tTp →
// tTpAttention + tTpDenseFFN [+ tMoEReduce]) is value-preserving for dense models:
//
//	tTpAttention + tTpDenseFFN = V(numLayers, tp) + V(numDenseLayers, tp)
//	                          = V(2·numLayers, tp)   (dense: numDenseLayers == numLayers, numMoELayers == 0)
//
// which equals today's monolithic tTp (allReduceUnits = 2·numDenseLayers + numMoELayers
// = 2·numLayers). The golden values below were captured from the pre-refactor
// implementation; any drift means the dense path changed, violating INV BC-DP1.
func TestINVBCDP1_DenseStepTimeByteIdentical(t *testing.T) {
	coeffs := testCoeffs()
	mc := trainedPhysicsTestModelConfig() // dense (NumLocalExperts == 0)
	hw := dpepTestHW()

	// Golden: pre-#C dense step time for the fixed dpepMixedBatch, per TP.
	golden := map[int]int64{
		1: 7797,
		2: 4538,
		4: 2909,
		8: 2094,
	}

	for tp, want := range golden {
		mhw := sim.NewModelHardwareConfig(*mc, hw, "m", "H100", tp, 1, false, "", "trained-physics", 0)
		m, err := NewTrainedPhysicsModel(*coeffs, mhw)
		require.NoError(t, err)
		got := m.StepTime(dpepMixedBatch())
		assert.Equalf(t, want, got,
			"INV BC-DP1: dense DP=1 step time must be byte-identical to pre-#C (tp=%d)", tp)
	}
}

// TestINVBCDP1_DenseDP1Determinism is the companion invariant to the golden above
// (CLAUDE.md BDD rule 4): rather than re-encoding the golden numbers, it asserts the
// law that makes the split safe — for a dense model the step time is invariant to the
// EnableExpertParallel flag and to the MoE comm backend (both are no-ops for dense),
// and identical across repeated calls (determinism, INV-6). This would survive a full
// rewrite of the term computation as long as the dense behavior is preserved.
func TestINVBCDP1_DenseDP1Determinism(t *testing.T) {
	coeffs := testCoeffs()
	mc := trainedPhysicsTestModelConfig()
	hw := dpepTestHW()
	batch := dpepMixedBatch()

	for _, tp := range []int{1, 2, 4, 8} {
		base := sim.NewModelHardwareConfig(*mc, hw, "m", "H100", tp, 1, false, "", "trained-physics", 0)
		mBase, err := NewTrainedPhysicsModel(*coeffs, base)
		require.NoError(t, err)
		want := mBase.StepTime(batch)

		// EP flag is a no-op for dense models.
		epOn := sim.NewModelHardwareConfig(*mc, hw, "m", "H100", tp, 1, true, "", "trained-physics", 0)
		mEP, err := NewTrainedPhysicsModel(*coeffs, epOn)
		require.NoError(t, err)
		assert.Equalf(t, want, mEP.StepTime(batch), "dense step time must ignore EP flag (tp=%d)", tp)

		// Comm backend is a no-op for dense models (no MoE dispatch).
		for _, backend := range ValidMoECommBackends {
			mhw := sim.NewModelHardwareConfig(*mc, hw, "m", "H100", tp, 1, false, backend, "trained-physics", 0)
			mB, err := NewTrainedPhysicsModel(*coeffs, mhw)
			require.NoError(t, err)
			assert.Equalf(t, want, mB.StepTime(batch),
				"dense step time must ignore comm backend %q (tp=%d)", backend, tp)
		}

		// Determinism: repeated calls identical.
		assert.Equal(t, want, mBase.StepTime(batch), "repeated StepTime must be identical (tp=%d)", tp)
	}
}

package sim

// TestRooflineConfig returns minimal valid ModelConfig and HardwareCalib for roofline tests.
// These are simplified configs that satisfy validation but may not match real hardware exactly.
func TestRooflineConfig() (ModelConfig, HardwareCalib) {
	// Minimal model config (typical 7B-class model parameters)
	mc := ModelConfig{
		NumHeads:    32,
		NumLayers:   32,
		HiddenDim:   4096,
		BytesPerParam: 2.0, // FP16
	}

	// H100 hardware calibration (simplified)
	hc := HardwareCalib{
		TFlopsPeak: 989.0,  // H100 SXM FP16 peak
		BwPeakTBs:  3.35,   // H100 SXM memory bandwidth
		MfuPrefill: 0.60,   // Typical MFU for prefill
		MfuDecode:  0.30,   // Typical MFU for decode
	}

	return mc, hc
}

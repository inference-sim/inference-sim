package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// A W4A16 checkpoint stores weights at four bits and runs the matmul at the compute dtype,
// because compressed-tensors leaves input_activations null: nothing quantizes the
// activations. So the narrow storage buys memory traffic, not FLOPs, and the compute peak
// must be BF16's -- NOT the four-bit rate a part advertises for NVFP4.
//
// Pricing int4 at a four-bit peak would understate every GEMM in the model by the ratio
// between the two peaks, which on a Blackwell part is about 2x.
func TestINT4RunsAtTheComputePeakNotTheFourBitPeak(t *testing.T) {
	// A part with native four-bit support, so the two peaks differ and the test can tell
	// them apart.
	chip := hardware.Chip{BF16Peak: 1000, FP8Peak: 2000, NVFP4Peak: 4000}

	_, int4Peak := dtypeFit(model.DTypeINT4, chip)
	_, bf16Peak := dtypeFit(model.DTypeBF16, chip)
	_, nvfp4Peak := dtypeFit(model.DTypeNVFP4, chip)

	if int4Peak != bf16Peak {
		t.Errorf("int4 should reach the BF16 peak (%v), got %v", bf16Peak, int4Peak)
	}
	if int4Peak == nvfp4Peak {
		t.Errorf("int4 and nvfp4 both reached %v; W4A16 dequantizes to the compute dtype "+
			"while NVFP4 has a native tensor-core path, so the two must differ on a part "+
			"that supports the latter", int4Peak)
	}
}

// Storage width is the other half: four bits of payload, so weight bytes are half an fp8
// checkpoint's and a quarter of a bf16 one. This is what the narrow format actually buys.
func TestINT4StoresFourBitsOfPayload(t *testing.T) {
	if got := model.DTypeINT4.Bytes(); got != 0.5 {
		t.Fatalf("int4 should be 0.5 bytes, got %v", got)
	}
	if model.DTypeINT4.Bytes() != model.DTypeNVFP4.Bytes() {
		t.Error("the 4-bit formats should share a payload width")
	}
	if model.DTypeINT4.Bytes()*2 != model.DTypeFP8.Bytes() {
		t.Error("int4 should be half of fp8")
	}
}

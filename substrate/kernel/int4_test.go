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

// An "auto" KV cache follows the model's COMPUTE dtype, not its weight storage width.
//
// A W4A16 checkpoint stores weights at four bits and computes in bf16. Following the
// storage width gave a half-byte KV element, which rounded to a per-block figure of
// zero once paged, and the kernel refused twelve Kimi-K2.5 sweeps rather than divide a
// budget by it. vLLM sizes an auto cache at the model dtype (vllm/platforms/interface.py:
// 861-862 at v0.31.0) unless the checkpoint's quantization config declares a KV algorithm,
// which the graph does not carry; see cacheDTypeBytes.
func TestAnAutoCacheFollowsTheComputeWidthNotTheStorageWidth(t *testing.T) {
	for _, tc := range []struct {
		served model.DType
		want   float64
		why    string
	}{
		{model.DTypeINT4, 2, "W4A16 computes in bf16"},
		{model.DTypeNVFP4, 2, "a 4-bit float is a storage format too"},
		{model.DTypeMXFP4, 2, "likewise"},
		{model.DTypeFP8, 2, "fp8 linears return bf16 (out_dtype=x.dtype), so the model dtype is bf16"},
		{model.DTypeBF16, 2, "unquantized passes through"},
		{model.DTypeFP32, 4, "and so does fp32"},
	} {
		if got := cacheDTypeBytes("auto", tc.served); got != tc.want {
			t.Errorf("auto cache on a %s model = %.1f bytes, want %.1f (%s)",
				tc.served, got, tc.want, tc.why)
		}
		// An empty setting means the engine's default, which is auto.
		if got := cacheDTypeBytes("", tc.served); got != tc.want {
			t.Errorf("empty cache dtype on a %s model = %.1f, want %.1f",
				tc.served, got, tc.want)
		}
	}
	// An explicitly named narrow cache is still honoured: the rule above is about what
	// "auto" resolves to, not a floor on every cache.
	if got := cacheDTypeBytes("fp8", model.DTypeINT4); got != 1 {
		t.Errorf("an explicit fp8 cache = %.1f bytes, want 1", got)
	}
	// nvfp4 carries an fp8 scale per 16 elements beside its four-bit data:
	// head_size/2 + head_size/16 bytes a head (vllm/utils/torch_utils.py:543-545).
	if got := cacheDTypeBytes("nvfp4", model.DTypeBF16); got != 0.5625 {
		t.Errorf("an explicit nvfp4 cache = %.4f bytes, want 0.5625", got)
	}
}

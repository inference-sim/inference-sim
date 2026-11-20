package units

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

func TestQuantFormatStoredBytes(t *testing.T) {
	cases := []struct {
		name  string
		f     QuantFormat
		elems int64
		want  Bytes
	}{
		{"bf16", QuantFormat{PayloadBits: 16}, 1000, 2000},
		{"fp8 unscaled", QuantFormat{PayloadBits: 8}, 1000, 1000},
		{"nvfp4: 16-element blocks, 8-bit scale", QuantFormat{PayloadBits: 4, ScaleBits: 8, BlockElements: 16}, 1600, 900},
		{"mxfp4: 32-element blocks, 8-bit scale", QuantFormat{PayloadBits: 4, ScaleBits: 8, BlockElements: 32}, 3200, 1700},
		{"rounds up once, at the end", QuantFormat{PayloadBits: 4}, 3, 2},
		{"no elements", QuantFormat{PayloadBits: 4}, 0, 0},
	}
	for _, c := range cases {
		got, err := c.f.StoredBytes(c.elems)
		if err != nil || got != c.want {
			t.Errorf("%s: StoredBytes(%d) = %d, %v; want %d", c.name, c.elems, got, err, c.want)
		}
	}
}

func TestQuantFormatRefusals(t *testing.T) {
	for name, f := range map[string]QuantFormat{
		"no payload":       {},
		"scale, no block":  {PayloadBits: 4, ScaleBits: 8},
		"block, no scale":  {PayloadBits: 4, BlockElements: 16},
		"negative scale":   {PayloadBits: 4, ScaleBits: -8, BlockElements: 16},
		"negative block":   {PayloadBits: 4, ScaleBits: 8, BlockElements: -16},
		"negative payload": {PayloadBits: -4},
	} {
		if err := f.Validate(); err == nil {
			t.Errorf("%s: accepted", name)
		}
		if _, err := f.StoredBytes(10); err == nil {
			t.Errorf("%s: StoredBytes accepted it", name)
		}
	}
	if _, err := (QuantFormat{PayloadBits: 8}).StoredBytes(-1); err == nil {
		t.Error("a negative element count was accepted")
	}
}

// The one function for stored bytes agrees with the model graph's, for every format the
// graph knows (SUB-16): same format, same bytes per element.
func TestQuantFormatAgreesWithTheModelGraph(t *testing.T) {
	formats := map[model.DType]QuantFormat{
		model.DTypeBF16:  {PayloadBits: 16},
		model.DTypeFP16:  {PayloadBits: 16},
		model.DTypeFP32:  {PayloadBits: 32},
		model.DTypeFP8:   {PayloadBits: 8},
		model.DTypeINT8:  {PayloadBits: 8},
		model.DTypeNVFP4: {PayloadBits: 4, ScaleBits: 8, BlockElements: 16},
		model.DTypeMXFP4: {PayloadBits: 4, ScaleBits: 8, BlockElements: 32},
		model.DTypeINT4:  {PayloadBits: 4},
	}
	for d := range map[model.DType]bool{
		model.DTypeBF16: true, model.DTypeFP16: true, model.DTypeFP32: true, model.DTypeFP8: true,
		model.DTypeINT8: true, model.DTypeNVFP4: true, model.DTypeMXFP4: true, model.DTypeINT4: true,
	} {
		f, ok := formats[d]
		if !ok {
			t.Fatalf("%s has no QuantFormat in this test; add it", d)
		}
		if got, want := f.BytesPerElement(), model.StoredBytes(d, nil); got != want {
			t.Errorf("%s: %v bytes per element, the graph's StoredBytes says %v", d, got, want)
		}
	}
	// A block-scaled weight the graph states explicitly: one E8M0 byte per 32 elements.
	s := model.WeightScale{Elements: 32, Bytes: 1}
	f := QuantFormat{PayloadBits: 4, ScaleBits: 8 * s.Bytes, BlockElements: s.Elements}
	if got, want := f.BytesPerElement(), model.StoredBytes(model.DTypeMXFP4, &s); got != want {
		t.Errorf("explicit scale: %v, the graph's StoredBytes says %v", got, want)
	}
}

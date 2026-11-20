package units

import (
	"fmt"
	"math"
)

// QuantFormat is a quantised number format: payload bits per element, and one scale of
// ScaleBits shared by every BlockElements elements (design SUB-16). It is the one type that
// states how many bytes a tensor of the format occupies; StoredBytes is the one function.
//
// A format with no scale has ScaleBits 0 and BlockElements 0. A per-tensor scale is
// negligible and is not counted, as the model graph does not count NVFP4's.
type QuantFormat struct {
	PayloadBits   int
	ScaleBits     int
	BlockElements int
}

// Validate refuses a format that cannot describe storage: no payload, a scale without a
// block (or a block without a scale), or a negative field.
func (f QuantFormat) Validate() error {
	switch {
	case f.PayloadBits < 1:
		return fmt.Errorf("units: payload of %d bits", f.PayloadBits)
	case f.ScaleBits < 0 || f.BlockElements < 0:
		return fmt.Errorf("units: scale of %d bits over a block of %d elements", f.ScaleBits, f.BlockElements)
	case (f.ScaleBits == 0) != (f.BlockElements == 0):
		return fmt.Errorf("units: a scale of %d bits over a block of %d elements; a format has both or neither", f.ScaleBits, f.BlockElements)
	}
	return nil
}

// BitsPerElement is the payload plus this element's share of its block's scale.
func (f QuantFormat) BitsPerElement() float64 {
	b := float64(f.PayloadBits)
	if f.BlockElements > 0 {
		b += float64(f.ScaleBits) / float64(f.BlockElements)
	}
	return b
}

// BytesPerElement is BitsPerElement in bytes. It is fractional for a 4-bit format; a caller
// multiplies by an element count and rounds once, or calls StoredBytes.
func (f QuantFormat) BytesPerElement() float64 { return f.BitsPerElement() / 8 }

// StoredBytes returns the bytes elements occupy, rounded up to a whole byte once, at the
// end. A negative count is an error.
func (f QuantFormat) StoredBytes(elements int64) (Bytes, error) {
	if err := f.Validate(); err != nil {
		return 0, err
	}
	if elements < 0 {
		return 0, fmt.Errorf("units: %d elements", elements)
	}
	return Bytes(math.Ceil(float64(elements) * f.BitsPerElement() / 8)), nil
}

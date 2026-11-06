package model

import "testing"

// INT4 must be recognized and must be four bits of payload, like the 4-bit floats. A format
// the schema does not know cannot be priced at all: the deriver refuses a config carrying it,
// which is how this gap surfaced -- Kimi-K2.5 ships W4A16 as compressed-tensors num_bits 4
// with type "int".
func TestINT4IsRecognizedAtFourBits(t *testing.T) {
	if !DTypeINT4.Valid() {
		t.Fatal("int4 is not a recognized dtype")
	}
	if got := DTypeINT4.Bytes(); got != 0.5 {
		t.Errorf("int4 should be 0.5 bytes of payload, got %v", got)
	}
}

// The three 4-bit formats share a payload width and must stay DISTINCT values. Collapsing
// them would let a checkpoint in one be priced as another, and they reach different rates on
// the same part: an integer grid dequantizes through a different path than a float one.
func TestTheThreeFourBitFormatsStayDistinct(t *testing.T) {
	four := []DType{DTypeNVFP4, DTypeMXFP4, DTypeINT4}
	seen := map[DType]bool{}
	for _, d := range four {
		if seen[d] {
			t.Errorf("%q appears twice; the 4-bit formats must be distinct values", d)
		}
		seen[d] = true
		if !d.Valid() {
			t.Errorf("%q is not recognized", d)
		}
		if got := d.Bytes(); got != 0.5 {
			t.Errorf("%q should be 0.5 bytes, got %v", d, got)
		}
	}
	if len(seen) != 3 {
		t.Errorf("want three distinct 4-bit formats, got %d", len(seen))
	}
}

// A graph whose weights are int4 must validate. The dtype changes what a weight read costs,
// not whether a model is coherent.
func TestAnINT4GraphValidates(t *testing.T) {
	g := validGraph()
	g.Global.WeightDType = DTypeINT4
	if p := g.Validate(); !p.OK() {
		t.Errorf("an int4 graph should validate:\n%s", p.Error())
	}
}

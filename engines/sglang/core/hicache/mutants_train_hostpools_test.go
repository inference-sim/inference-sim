// Original to blis-sglang.

package hicache

import "testing"

// Behavioural tests of the host pools' geometry (the mutation sweep of the M3.1 train): each fails on
// one generated mutant of hostpools.go that the golden traces do not reach.

// An index_kpool of zero is a refusal, not a division by zero.
func TestDSAIndexerHostRefusesAZeroOrNegativeIndexKPool(t *testing.T) {
	for _, k := range []int{0, -1, -4} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("index_kpool %d: panicked instead of refusing: %v", k, r)
				}
			}()
			if s, err := DSAIndexerHost(128, 128, 64, k, 3); err == nil {
				t.Errorf("index_kpool %d: accepted, shape %+v", k, s)
			}
		}()
	}
	// A positive index_kpool that divides the page is the shape the layers own.
	s, err := DSAIndexerHost(128, 128, 64, 1, 3)
	if err != nil || s.LayerNum != 3 || s.Components != 1 || s.RowBytes != 132 {
		t.Errorf("index_kpool 1: %+v, %v; want 3 layers of one component with 132-byte rows (128 keys and one fp32 scale)", s, err)
	}
}

// A page size of zero is a refusal naming the pages, not a division by zero; so is a count that is
// not whole pages. Zero tokens is a whole number of pages and prices nothing.
func TestTransferPriceRefusesAZeroPageAndPricesZeroTokensAsNothing(t *testing.T) {
	s := MHAHost(8, 128, 128, 2, 2, 1)
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("page size 0: panicked instead of refusing: %v", r)
			}
		}()
		if p, err := s.Backup("layer_first", 0, 0); err == nil {
			t.Errorf("page size 0 with 0 tokens: priced %+v", p)
		}
		if p, err := s.Backup("layer_first", 0, 64); err == nil {
			t.Errorf("page size 0 with 64 tokens: priced %+v", p)
		}
	}()
	if _, err := s.Backup("layer_first", 64, 100); err == nil {
		t.Error("100 tokens of pages of 64: priced")
	}
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("zero tokens: panicked instead of pricing nothing: %v", r)
			}
		}()
		for _, price := range []func(string, int, int) (TransferPrice, error){s.Backup, s.LoadLayer} {
			p, err := price("layer_first", 64, 0)
			if err != nil || p != (TransferPrice{}) {
				t.Errorf("zero tokens: %+v, %v; want a price of nothing", p, err)
			}
		}
	}()
}

// With K and V rows of different widths, component 0 is K's row and component 1 is V's.
func TestAsymmetricHostRowsAreKThenV(t *testing.T) {
	s := MHAHost(4, 192, 128, 2, 2, 1) // K rows of 4*192*2 = 1,536 bytes, V rows of 4*128*2 = 1,024
	if k, v := s.componentRow(0), s.componentRow(1); k != 1536 || v != 1024 {
		t.Errorf("component rows %d and %d, want K 1536 then V 1024", k, v)
	}
	if got := s.BytesPerTokenPerLayer(); got != 2560 {
		t.Errorf("a token's row of one layer is %d bytes, want 2560", got)
	}
	// The contiguous run of page_first is K's row across the layers.
	run, units, err := s.hostRun("page_first", 64, 2)
	if err != nil || run != 1536*2 || units != 64*2 {
		t.Errorf("page_first run %d bytes in %d units, %v; want %d bytes in 128", run, units, err, 1536*2)
	}
	// A symmetric pool has one width for both.
	m := MHAHost(4, 128, 128, 2, 2, 1)
	if m.componentRow(0) != 1024 || m.componentRow(1) != 1024 {
		t.Errorf("symmetric rows %d and %d, want 1024 each", m.componentRow(0), m.componentRow(1))
	}
}

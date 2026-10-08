// Not vendored: BLIS-side test for the one change to the vendored score.go.

package lorascore

import (
	"math"
	"testing"
	"time"
)

// Share must not depend on map iteration order. Rates spanning more than 2^53
// make the floating-point total order-sensitive, so a map-ordered sum would
// differ between calls; the sorted sum must give the same bits every time.
func TestShareIsIndependentOfMapOrder(t *testing.T) {
	d, err := NewDemand(time.Second)
	if err != nil {
		t.Fatal(err)
	}
	t0 := time.Unix(0, 0)
	d.rate["big"], d.last["big"] = 1e17, t0
	for _, a := range []string{"s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"} {
		d.rate[a], d.last[a] = 7, t0
	}
	first := d.Share("s1", t0)
	for i := 0; i < 500; i++ {
		if got := d.Share("s1", t0); math.Float64bits(got) != math.Float64bits(first) {
			t.Fatalf("call %d: Share = %v, first call gave %v", i, got, first)
		}
	}
}

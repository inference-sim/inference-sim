package kvtransfer

import "testing"

// An injected service time is the whole price of a transfer: the station returns exactly what
// it says for every direction, size and depth (floored at one tick), and needs no bandwidth of
// its own to be valid.
func TestInjectedServiceTimeIsTheWholePrice(t *testing.T) {
	price := func(dir Direction, bytes int64, q int) int64 { return int64(dir)*7 + bytes/1000 + int64(q) }
	st, err := New(Config{Tiers: []TierConfig{{NRead: 1, NWrite: 1, ServiceTime: price}}})
	if err != nil {
		t.Fatalf("a tier priced by an injected service time was refused: %v", err)
	}
	for _, dir := range []Direction{Read, Write} {
		for _, bytes := range []int64{0, 4096, 1 << 30} {
			for _, q := range []int{0, 1, 5} {
				want := max(1, price(dir, bytes, max(1, q)))
				if got := st.ServiceTicksAtDepth(0, dir, bytes, q); got != want {
					t.Errorf("dir=%d bytes=%d q=%d: station charged %d, the injected price is %d", dir, bytes, q, got, want)
				}
			}
		}
	}
}

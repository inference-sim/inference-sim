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

// A job of n files is n transfers one after another on its server, each of its share of the
// bytes (vLLM's fs tier opens, moves and closes each chunk's file in turn,
// tiering/fs/io.py:166-218 at v0.31.0), so each pays the tier's floor: two 4 KiB files cost
// twice one, not one 8 KiB transfer.
func TestAJobOfFilesIsPricedFileByFile(t *testing.T) {
	price := func(_ Direction, bytes int64, _ int) int64 { return 100 + bytes/1024 }
	st, err := New(Config{Tiers: []TierConfig{{NRead: 1, NWrite: 1, ServiceTime: price}}})
	if err != nil {
		t.Fatal(err)
	}
	st.Submit(TransferJob{Direction: Write, Bytes: 8192, Files: 2})
	if done := st.Poll(207); len(done) != 0 {
		t.Fatal("two 4 KiB files take 2 x 104 ticks; the job must not be done at 207")
	}
	if done := st.Poll(208); len(done) != 1 {
		t.Fatal("the two-file job must be done at 208 ticks")
	}
}

// With DeviceDepth, a tier's reads and writes share one queue depth: a read started while a
// write is in service is priced at depth 2.
func TestDeviceDepthCountsBothDirections(t *testing.T) {
	var q int
	price := func(dir Direction, _ int64, d int) int64 {
		if dir == Read {
			q = d
		}
		return 50
	}
	st, err := New(Config{Tiers: []TierConfig{{NRead: 1, NWrite: 1, ServiceTime: price, DeviceDepth: true}}})
	if err != nil {
		t.Fatal(err)
	}
	st.Submit(TransferJob{Direction: Write, Bytes: 1})
	st.Submit(TransferJob{Direction: Read, Bytes: 1})
	if q != 2 {
		t.Fatalf("read priced at depth %d with a write in service, want 2", q)
	}
}

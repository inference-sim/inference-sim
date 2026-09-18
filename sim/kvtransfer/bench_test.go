package kvtransfer

import "testing"

// BenchmarkSubmitPoll measures the steady-state cost of a submit+poll pair
// (objective P: amortized O(1), no per-block allocation). The station is kept in
// steady state — one submit and one poll per iteration against a modest server
// pool — so the reported ns/op and allocs/op reflect the per-job cost and must
// not grow with the total number of jobs processed.
func BenchmarkSubmitPoll(b *testing.B) {
	s, err := New(Config{Tiers: []TierConfig{{
		NRead: 16, NWrite: 16,
		ReadBaseTicks: 5, WriteBaseTicks: 9,
		ReadBytesPerTick: 4, WriteBytesPerTick: 2,
	}}})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	tick := int64(0)
	for i := 0; i < b.N; i++ {
		dir := Read
		if i%2 == 0 {
			dir = Write
		}
		s.Submit(TransferJob{Tier: 0, Direction: dir, Bytes: int64(i%1024) * 64, SubmitTick: tick})
		s.Poll(tick)
		tick += 3
	}
}

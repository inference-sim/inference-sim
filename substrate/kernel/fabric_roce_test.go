package latencykernel

import "testing"

// A RoCE class carries the line rate of its InfiniBand twin, and its efficiency is assumed
// equal to ib-400g's, so a handoff over roce-400g takes as long as over ib-400g and over
// roce-800g as long as over ib-800g, half as long as over roce-400g.
func TestHandoffOverRoCETakesTheTimeOfItsInfiniBandTwin(t *testing.T) {
	const tokens = 60000
	for _, tc := range []struct{ roce, ib string }{
		{"roce-400g", "ib-400g"},
		{"roce-800g", "ib-800g"},
	} {
		r, err := handoffOn(t, tc.roce, tokens)
		if err != nil {
			t.Fatalf("a scenario naming %s was refused: %v", tc.roce, err)
		}
		i, err := handoffOn(t, tc.ib, tokens)
		if err != nil {
			t.Fatal(err)
		}
		if r.Bytes != i.Bytes || r.Bytes == 0 || len(r.Links) == 0 || len(i.Links) == 0 {
			t.Fatalf("%s: payload %d vs %d bytes, links %d vs %d", tc.roce, r.Bytes, i.Bytes,
				len(r.Links), len(i.Links))
		}
		if r.Links[0].BytesPerSecond != i.Links[0].BytesPerSecond {
			t.Errorf("%s NIC rate %.4g, %s %.4g", tc.roce, r.Links[0].BytesPerSecond, tc.ib,
				i.Links[0].BytesPerSecond)
		}
		if r.AloneTime() != i.AloneTime() {
			t.Errorf("%s takes %v, %s %v", tc.roce, r.AloneTime(), tc.ib, i.AloneTime())
		}
	}
}

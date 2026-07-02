package cluster

import "blis/sim"

// RequestSource is a pull-style stream of *sim.Request values for the cluster
// arrival pump.
//
// Implementations MUST:
//   - Yield each request exactly once, in non-decreasing ArrivalTime order.
//     Run() relies on this and does not verify or re-sort it.
//   - Return (nil, false) when exhausted. Subsequent calls MUST also return
//     (nil, false) — exhaustion is sticky.
//   - Never return (nil, true). Run() dereferences the returned request and
//     would panic on a nil with ok=true.
//
// Implementations are not expected to be safe for concurrent use. The cluster's
// Run() loop consumes a source from a single goroutine.
//
// The streaming generator (issue #1438 Change A3) will add a second
// implementation that yields requests on demand without ever materializing the
// full slice. This PR introduces only the eager SliceRequestSource adapter.
type RequestSource interface {
	Next() (*sim.Request, bool)
}

// SliceRequestSource is the trivial eager adapter that wraps a pre-materialized
// []*sim.Request and serves it through the RequestSource interface. It is
// value-identity over the input slice: same elements, same order, no
// transformation, no copy.
type SliceRequestSource struct {
	reqs []*sim.Request
	i    int
}

// NewSliceRequestSource wraps reqs as a RequestSource. The caller retains
// ownership of the slice; the adapter does not copy it. A nil or empty reqs is
// valid and exhausts immediately.
func NewSliceRequestSource(reqs []*sim.Request) *SliceRequestSource {
	return &SliceRequestSource{reqs: reqs}
}

// Next returns the next request and true, or (nil, false) when exhausted.
func (s *SliceRequestSource) Next() (*sim.Request, bool) {
	if s.i >= len(s.reqs) {
		return nil, false
	}
	req := s.reqs[s.i]
	s.i++
	return req, true
}

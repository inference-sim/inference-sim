package sim

// IntSeries is a run of int64 entries kept as the segments it was built from, one per completed
// request, instead of one flat array. The inter-token series of a long replay are hundreds of
// megabytes each, and a request's own entries are already held by the request; a flat series
// copies them, and the cluster's merged series copies them again. A segment here is the very slice
// the request holds (capped at its length, so a later append to the request cannot reach it), and
// a merge of series joins segment lists.
//
// The zero value is an empty series. Entries are read in append order; Len, Flatten, All and the
// cursor give the same sequence a flat slice would.
type IntSeries struct {
	segs [][]int64
	n    int
}

// SeriesOf returns a series holding the given entries as one segment (no copy).
func SeriesOf(v ...int64) IntSeries {
	var s IntSeries
	s.Add(v)
	return s
}

// Add appends a segment without copying it; an empty segment adds nothing. The caller does not
// change the entries afterwards.
func (s *IntSeries) Add(v []int64) {
	if len(v) == 0 {
		return
	}
	s.segs = append(s.segs, v[:len(v):len(v)])
	s.n += len(v)
}

// AddAll appends the segments of o, in order, without copying their entries.
func (s *IntSeries) AddAll(o IntSeries) {
	s.segs = append(s.segs, o.segs...)
	s.n += o.n
}

// Grow makes room for n more segments.
func (s *IntSeries) Grow(n int) {
	if n > 0 {
		s.segs = append(make([][]int64, 0, len(s.segs)+n), s.segs...)
	}
}

// Len is the number of entries.
func (s IntSeries) Len() int { return s.n }

// Segments is the number of segments held.
func (s IntSeries) Segments() int { return len(s.segs) }

// Flatten returns the entries as one new slice, nil for an empty series.
func (s IntSeries) Flatten() []int64 {
	if s.n == 0 {
		return nil
	}
	out := make([]int64, 0, s.n)
	for _, g := range s.segs {
		out = append(out, g...)
	}
	return out
}

// All calls yield for each entry in order, stopping when it returns false.
func (s IntSeries) All(yield func(int64) bool) {
	for _, g := range s.segs {
		for _, v := range g {
			if !yield(v) {
				return
			}
		}
	}
}

// seriesCursor reads a series front to back in runs of a given length.
type seriesCursor struct {
	s        IntSeries
	seg, off int
}

// take calls emit with the next n entries as pieces (one per segment they span), or only skips
// them when emit is nil.
func (c *seriesCursor) take(n int, emit func([]int64)) {
	for n > 0 && c.seg < len(c.s.segs) {
		g := c.s.segs[c.seg][c.off:]
		k := min(n, len(g))
		if emit != nil {
			emit(g[:k])
		}
		n -= k
		c.off += k
		if c.off == len(c.s.segs[c.seg]) {
			c.seg, c.off = c.seg+1, 0
		}
	}
}

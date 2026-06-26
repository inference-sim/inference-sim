// SessionTokenBuffer is a session-scoped growable token buffer shared by all
// rounds of one multi-turn session. Each round's Request.InputTokens is a flat
// sub-slice (view) into the buffer's underlying array. The buffer grows by
// append, amortized O(1) per token; total per-session storage is O(R) where R
// is the round count, replacing the prior O(R²) per-round eager copy (#1445).
//
// Not safe for concurrent use; sessions are processed by a single goroutine
// (the DES event loop).

package workload

// SessionTokenBuffer holds the growable backing array for a session's
// accumulated context plus prefix.
type SessionTokenBuffer struct {
	b []int
}

// NewSessionTokenBuffer creates an empty buffer.
func NewSessionTokenBuffer() *SessionTokenBuffer {
	return &SessionTokenBuffer{}
}

// Append copies the given tokens onto the end of the buffer and returns the
// absolute [start, end) range they now occupy.
func (s *SessionTokenBuffer) Append(toks []int) (start, end int64) {
	start = int64(len(s.b))
	s.b = append(s.b, toks...)
	end = int64(len(s.b))
	return
}

// Slice returns a flat view of the buffer over the given absolute range. The
// returned slice aliases the buffer's underlying array; callers must not mutate.
func (s *SessionTokenBuffer) Slice(start, end int64) []int {
	return s.b[start:end]
}

// Len returns the number of tokens currently in the buffer.
func (s *SessionTokenBuffer) Len() int64 {
	return int64(len(s.b))
}

// Cap returns the capacity of the underlying array. Exposed for memory tests.
func (s *SessionTokenBuffer) Cap() int {
	return cap(s.b)
}

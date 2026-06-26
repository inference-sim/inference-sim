// sessionTokenBuffer is a session-scoped growable token buffer shared by all
// rounds of one multi-turn session. Each round's Request.InputTokens is a flat
// sub-slice (view) into the buffer's underlying array. The buffer grows by
// append, amortized O(1) per token; total per-session storage is O(R) where R
// is the round count, replacing the prior O(R²) per-round eager copy (#1445).
//
// Not safe for concurrent use; sessions are processed by a single goroutine
// (the DES event loop).

package workload

import "fmt"

// sessionTokenBuffer holds the growable backing array for a session's
// accumulated context plus prefix.
//
// # Reallocation hazard (intentional, bounded)
//
// When Append exceeds the buffer's capacity, Go's append allocates a new
// backing array and copies the data. Slices previously returned by Slice()
// still point to the OLD array. They remain content-correct (the old array's
// data is unchanged) but become orphaned views — extending them with append
// would not be visible through new slices.
//
// In BLIS this is safe because: (a) the DES is single-threaded; (b) once
// returned to a Request, a slice is read-only for the simulator (#1445 R5
// mutation rules); (c) total memory is bounded by Go's amortized-O(1) append
// policy (cap ≤ 2× current len), so peak per-session bytes remain O(R) as
// asserted by the memory tests in session_buffer_mem_test.go. To eliminate
// the orphaning entirely when R is known upfront (open-loop generation), use
// newSessionTokenBufferWithCapacity.
type sessionTokenBuffer struct {
	b []int
}

// newSessionTokenBuffer creates an empty buffer. Capacity grows on demand via
// Go's amortized-O(1) append policy.
func newSessionTokenBuffer() *sessionTokenBuffer {
	return &sessionTokenBuffer{}
}

// newSessionTokenBufferWithCapacity pre-allocates the buffer to the given
// capacity. Use this when the total session size is known (e.g., open-loop
// reasoning generation) to avoid the reallocation hazard described on
// sessionTokenBuffer — every Slice() result remains valid through all
// subsequent appends. Panics on a negative capacity (R3: validate library
// constructor parameters; a negative value indicates a caller calculation bug
// that should surface immediately, not be silently clamped).
func newSessionTokenBufferWithCapacity(capacity int64) *sessionTokenBuffer {
	if capacity < 0 {
		panic(fmt.Sprintf("newSessionTokenBufferWithCapacity: capacity must be non-negative, got %d", capacity))
	}
	return &sessionTokenBuffer{b: make([]int, 0, capacity)}
}

// Append copies the given tokens onto the end of the buffer and returns the
// absolute [start, end) range they now occupy.
func (s *sessionTokenBuffer) Append(toks []int) (start, end int64) {
	start = int64(len(s.b))
	s.b = append(s.b, toks...)
	end = int64(len(s.b))
	return
}

// Slice returns a flat view of the buffer over the given absolute range. The
// returned slice aliases the buffer's underlying array; callers must not mutate.
// Panics with a decorated message if [start, end) is out of bounds.
func (s *sessionTokenBuffer) Slice(start, end int64) []int {
	n := int64(len(s.b))
	if start < 0 || end < start || end > n {
		panic(fmt.Sprintf("sessionTokenBuffer.Slice: invalid range [%d, %d) for buf len=%d", start, end, n))
	}
	return s.b[start:end]
}

// Len returns the number of tokens currently in the buffer.
func (s *sessionTokenBuffer) Len() int64 {
	return int64(len(s.b))
}

// bufCap returns the capacity of the underlying array as int64 (matching the
// type signature of Len/Slice/Append). Unexported: same-package memory tests
// access it via white-box; production code has no need.
func (s *sessionTokenBuffer) bufCap() int64 {
	return int64(cap(s.b))
}

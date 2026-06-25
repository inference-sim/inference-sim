package cluster

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// Compile-time check that *SliceRequestSource satisfies RequestSource.
var _ RequestSource = (*SliceRequestSource)(nil)

func TestSliceRequestSource_YieldsRequestsInOrder(t *testing.T) {
	reqs := []*sim.Request{
		{ID: "r0", ArrivalTime: 0},
		{ID: "r1", ArrivalTime: 100},
		{ID: "r2", ArrivalTime: 200},
	}
	src := NewSliceRequestSource(reqs)

	for i, want := range reqs {
		got, ok := src.Next()
		if !ok {
			t.Fatalf("Next() at i=%d: expected ok=true, got false", i)
		}
		if got != want {
			t.Errorf("Next() at i=%d: got %p, want %p (pointer-equal expected)", i, got, want)
		}
	}
}

func TestSliceRequestSource_ExhaustionReturnsFalse(t *testing.T) {
	reqs := []*sim.Request{{ID: "r0", ArrivalTime: 0}}
	src := NewSliceRequestSource(reqs)

	if _, ok := src.Next(); !ok {
		t.Fatalf("first Next(): expected ok=true")
	}
	for i := 0; i < 3; i++ {
		got, ok := src.Next()
		if ok {
			t.Errorf("post-exhaustion Next() #%d: expected ok=false, got %v", i, got)
		}
		if got != nil {
			t.Errorf("post-exhaustion Next() #%d: expected nil request, got %v", i, got)
		}
	}
}

func TestSliceRequestSource_EmptyAndNilExhaustImmediately(t *testing.T) {
	cases := []struct {
		name string
		in   []*sim.Request
	}{
		{"nil", nil},
		{"empty", []*sim.Request{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := NewSliceRequestSource(tc.in)
			got, ok := src.Next()
			if ok {
				t.Errorf("Next() on %s: expected ok=false, got true (req=%v)", tc.name, got)
			}
			if got != nil {
				t.Errorf("Next() on %s: expected nil request, got %v", tc.name, got)
			}
		})
	}
}

// fakeCountingSource wraps a SliceRequestSource and counts emissions vs
// terminating false calls. Provided as a helper for any future test that wants
// to assert drain-to-exhaustion semantics directly.
type fakeCountingSource struct {
	inner   *SliceRequestSource
	yielded int
	exhaust int
}

func (f *fakeCountingSource) Next() (*sim.Request, bool) {
	r, ok := f.inner.Next()
	if ok {
		f.yielded++
	} else {
		f.exhaust++
	}
	return r, ok
}

func TestNewClusterSimulator_PanicsOnNilRequestSource(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic on nil RequestSource, got none")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("expected string panic value, got %T: %v", r, r)
		}
		// Caller-friendly hint matters: we want the message to point at the fix.
		if msg == "" {
			t.Errorf("panic message is empty")
		}
	}()
	NewClusterSimulator(newTestDeploymentConfig(1), nil, nil)
}

// nilTrueRequestSource violates the RequestSource contract by returning
// (nil, true) on the first call, then exhausting. Used to verify Run() fails
// fast with an actionable message rather than producing an opaque nil-deref.
type nilTrueRequestSource struct {
	fired bool
}

func (n *nilTrueRequestSource) Next() (*sim.Request, bool) {
	if n.fired {
		return nil, false
	}
	n.fired = true
	return nil, true
}

func TestClusterSimulator_PanicsOnNilTrueFromSource(t *testing.T) {
	cs := NewClusterSimulator(newTestDeploymentConfig(1), &nilTrueRequestSource{}, nil)
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic on (nil, true) from RequestSource, got none")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("expected string panic value, got %T: %v", r, r)
		}
		// The message should name the contract violation so future implementers
		// can debug it without spelunking through Run().
		if !strings.Contains(msg, "nil") {
			t.Errorf("panic message should mention nil, got: %q", msg)
		}
		if !strings.Contains(msg, "contract") {
			t.Errorf("panic message should mention contract violation, got: %q", msg)
		}
	}()
	_ = cs.Run()
}

func TestSliceRequestSource_CountingWrapperIntegrates(t *testing.T) {
	reqs := []*sim.Request{
		{ID: "r0", ArrivalTime: 0},
		{ID: "r1", ArrivalTime: 100},
	}
	src := &fakeCountingSource{inner: NewSliceRequestSource(reqs)}
	for {
		_, ok := src.Next()
		if !ok {
			break
		}
	}
	if src.yielded != 2 {
		t.Errorf("yielded=%d, want 2", src.yielded)
	}
	if src.exhaust != 1 {
		t.Errorf("exhaust=%d, want 1 (one terminating false call)", src.exhaust)
	}
}

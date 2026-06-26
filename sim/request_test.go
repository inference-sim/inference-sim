package sim

import (
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRequestState_Constants_HaveExpectedStringValues(t *testing.T) {
	// BC-6: Typed constants replace raw strings
	assert.Equal(t, RequestState("queued"), StateQueued)
	assert.Equal(t, RequestState("running"), StateRunning)
	assert.Equal(t, RequestState("completed"), StateCompleted)
}

func TestRequest_String_IncludesState(t *testing.T) {
	req := Request{ID: "test-1", State: StateQueued}
	s := req.String()
	assert.Contains(t, s, "queued")
}

// TestRequestIsMultimodal verifies the IsMultimodal derivation (GAP-4, #1264):
// the method returns true iff any per-modality token count is > 0.
func TestRequestIsMultimodal(t *testing.T) {
	cases := []struct {
		name string
		req  Request
		want bool
	}{
		{"all zero — text only", Request{}, false},
		{"text count set, no modality", Request{TextTokenCount: 100}, false},
		{"image only", Request{ImageTokenCount: 1}, true},
		{"audio only", Request{AudioTokenCount: 1}, true},
		{"video only", Request{VideoTokenCount: 1}, true},
		{"image + text mixed", Request{TextTokenCount: 50, ImageTokenCount: 10}, true},
		{"all three modalities", Request{ImageTokenCount: 1, AudioTokenCount: 1, VideoTokenCount: 1}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.req.IsMultimodal(); got != tc.want {
				t.Errorf("IsMultimodal() = %v, want %v (req=%+v)", got, tc.want, tc.req)
			}
		})
	}
}

// TestRequestInputLen verifies that InputLen returns int64(len(InputTokens))
// for both empty and populated requests (BC-3, #1445).
func TestRequestInputLenZero(t *testing.T) {
	r := &Request{}
	if got := r.InputLen(); got != 0 {
		t.Fatalf("InputLen() on empty request = %d, want 0", got)
	}
}

func TestRequestInputLenMatchesLen(t *testing.T) {
	r := &Request{InputTokens: []int{1, 2, 3, 4, 5}}
	if got, want := r.InputLen(), int64(5); got != want {
		t.Fatalf("InputLen() = %d, want %d", got, want)
	}
}

func TestRequestFullInputTokensMatchesField(t *testing.T) {
	want := []int{7, 8, 9}
	r := &Request{InputTokens: want}
	got := r.FullInputTokens()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("FullInputTokens() = %v, want %v", got, want)
	}
}

func TestRequestInputTokenSliceRange(t *testing.T) {
	r := &Request{InputTokens: []int{10, 11, 12, 13, 14}}
	got := r.InputTokenSlice(1, 4)
	want := []int{11, 12, 13}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("InputTokenSlice(1,4) = %v, want %v", got, want)
	}
}

func TestRequestInputTokenSliceFull(t *testing.T) {
	r := &Request{InputTokens: []int{10, 11, 12}}
	got := r.InputTokenSlice(0, 3)
	want := []int{10, 11, 12}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("InputTokenSlice(0,3) = %v, want %v", got, want)
	}
}

// TestRequestInputTokenSliceBounds asserts that out-of-bounds ranges panic with
// a decorated message that surfaces the request ID (CRIT-1, #1445). Without this
// guard, a malformed call site would produce a bare Go runtime panic with no
// simulation context.
func TestRequestInputTokenSliceBounds(t *testing.T) {
	r := &Request{ID: "test-req", InputTokens: []int{1, 2, 3, 4, 5}}
	cases := []struct {
		name       string
		start, end int64
	}{
		{"end > len", 0, 6},
		{"start > end", 3, 2},
		{"negative start", -1, 3},
		{"start > len", 10, 12},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				rec := recover()
				if rec == nil {
					t.Fatalf("expected panic for InputTokenSlice(%d, %d)", tc.start, tc.end)
				}
				msg, ok := rec.(string)
				if !ok {
					t.Fatalf("panic value is %T, want string", rec)
				}
				if !strings.Contains(msg, "test-req") {
					t.Errorf("panic message %q does not contain request ID", msg)
				}
			}()
			_ = r.InputTokenSlice(tc.start, tc.end)
		})
	}
}

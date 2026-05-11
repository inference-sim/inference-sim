package sim

import (
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

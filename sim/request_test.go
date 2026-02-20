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

package llmd

import (
	"errors"
	"testing"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// TestClassify_AllowList: each router outcome BLIS knows maps to exactly one INV-1 bucket, and
// anything else is reported as unclassified rather than counted as shedding.
func TestClassify_AllowList(t *testing.T) {
	dropped := func(code string, r errcommon.RequestDroppedReason) error {
		return errcommon.Error{Code: code, Msg: "m", Headers: map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(r)}}
	}
	for _, tc := range []struct {
		name string
		err  error
		want cluster.Rejection
		ok   bool
	}{
		{"no endpoints", dropped(errcommon.ServiceUnavailable, errcommon.RequestDroppedReasonNoEndpoints), cluster.RejectedRouting, true},
		{"ttl expired", dropped(errcommon.ResourceExhausted, errcommon.RequestDroppedReasonTTLExpired), cluster.RejectedGatewayExpired, true},
		{"queues at capacity", dropped(errcommon.ResourceExhausted, errcommon.RequestDroppedReasonSaturated), cluster.RejectedGatewayCapacity, true},
		{"client gave up", dropped(errcommon.ServiceUnavailable, errcommon.RequestDroppedReasonContextCancelled), cluster.RejectedClientTimeout, true},
		{"legacy shed", errcommon.Error{Code: errcommon.ResourceExhausted, Msg: "system saturated, sheddable request dropped"}, cluster.RejectedAdmission, true},
		{"admitter plugin", errcommon.Error{Code: errcommon.ResourceExhausted, Msg: "probabilistic-admitter: rejected"}, cluster.RejectedAdmission, true},
		{"unavailable", errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: "m"}, cluster.RejectedRouting, true},
		{"shutting down", dropped(errcommon.ServiceUnavailable, errcommon.RequestDroppedReasonShuttingDown), 0, false},
		{"internal flow control", dropped(errcommon.Internal, errcommon.RequestDroppedReasonInternal), 0, false},
		{"internal", errcommon.Error{Code: errcommon.Internal, Msg: "m"}, 0, false},
		{"bad request", errcommon.Error{Code: errcommon.BadRequest, Msg: "m"}, 0, false},
		{"not found", errcommon.Error{Code: errcommon.NotFound, Msg: "m"}, 0, false},
		{"untyped scheduler failure", errcommon.Error{Code: errcommon.ResourceExhausted, Msg: "failed to find target endpoint: x"}, 0, false},
		{"untyped error", errors.New("boom"), 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, _, ok := classify(tc.err)
			if ok != tc.ok || (ok && got != tc.want) {
				t.Errorf("classify = (%v, %v), want (%v, %v)", got, ok, tc.want, tc.ok)
			}
		})
	}
}

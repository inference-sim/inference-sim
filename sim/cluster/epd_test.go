package cluster

import "testing"

// TestValidatePoolTopology_EncodeEdgeCases covers BC-EPD-5 config-time
// rejection at the validation boundary (encode-without-decode).
func TestValidatePoolTopology_EncodeEdgeCases(t *testing.T) {
	// encode-only with no PD pool must be rejected.
	if err := ValidatePoolTopology(0, 0, 0, 2, 2); err == nil {
		t.Error("ValidatePoolTopology(encode-only) accepted; want error")
	}
	// encode with shared-role only: allowed.
	if err := ValidatePoolTopology(0, 0, 2, 1, 3); err != nil {
		t.Errorf("ValidatePoolTopology(shared+encode) rejected: %v", err)
	}
}

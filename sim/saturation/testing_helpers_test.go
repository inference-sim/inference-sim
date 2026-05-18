// sim/saturation/test_helpers.go
package saturation

import "testing"

// asResult is a test helper to type-assert Classify result back to Result.
// Classify returns interface{} to implement sim.BatchClassifier, but tests
// need to access Result fields.
func asResult(t *testing.T, iface interface{}) Result {
	t.Helper()
	result, ok := iface.(Result)
	if !ok {
		t.Fatalf("Classify returned non-Result type: %T", iface)
	}
	return result
}

// Moved from inference-sim/sim/engine/main_test.go at ca532abca209af41425c007974a5f3ef336f5981.

package engine

import (
	"os"
	"testing"

	"github.com/sirupsen/logrus"
)

func TestMain(m *testing.M) {
	// Suppress verbose simulation logs during tests to speed up CI
	// Set DEBUG_TESTS=1 to see full logs: DEBUG_TESTS=1 go test ./sim/... -v
	if os.Getenv("DEBUG_TESTS") == "" {
		logrus.SetLevel(logrus.WarnLevel)
	}
	os.Exit(m.Run())
}

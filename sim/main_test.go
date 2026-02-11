package sim

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

package sim

import (
	"os"
	"testing"

	"github.com/sirupsen/logrus"
)

func TestMain(m *testing.M) {
	// Suppress verbose simulation logs during tests to speed up CI
	logrus.SetLevel(logrus.WarnLevel)
	os.Exit(m.Run())
}

package cluster

import (
	"bytes"
	"strings"
	"testing"

	"github.com/sirupsen/logrus"
)

// captureLogOutput runs fn and returns the log output as a string.
func captureLogOutput(fn func()) string {
	var buf bytes.Buffer
	logrus.SetOutput(&buf)
	origLevel := logrus.GetLevel()
	logrus.SetLevel(logrus.WarnLevel)
	defer func() {
		logrus.SetOutput(nil) // reset to default (stderr)
		logrus.SetLevel(origLevel)
	}()
	fn()
	return buf.String()
}

func TestClusterSimulator_HorizonTooSmall_WarnsAtStartup(t *testing.T) {
	// GIVEN horizon (100) < admissionLatency (200) + routingLatency (300) = 500
	config := newTestDeploymentConfig(1)
	config.Horizon = 100
	config.AdmissionLatency = 200
	config.RoutingLatency = 300
	workload := newTestWorkload(10)

	// WHEN the cluster simulator is constructed
	output := captureLogOutput(func() {
		NewClusterSimulator(config, workload, "")
	})

	// THEN a warning about horizon being too small MUST be logged
	if !strings.Contains(output, "horizon") || !strings.Contains(output, "pipeline latency") {
		t.Errorf("expected warning about horizon < pipeline latency, got: %q", output)
	}
}

func TestClusterSimulator_HorizonSufficient_NoWarning(t *testing.T) {
	// GIVEN horizon (10000) >= admissionLatency (200) + routingLatency (300) = 500
	config := newTestDeploymentConfig(1)
	config.Horizon = 10000
	config.AdmissionLatency = 200
	config.RoutingLatency = 300
	workload := newTestWorkload(10)

	// WHEN the cluster simulator is constructed
	output := captureLogOutput(func() {
		NewClusterSimulator(config, workload, "")
	})

	// THEN no horizon warning MUST be logged
	if strings.Contains(output, "horizon") && strings.Contains(output, "pipeline latency") {
		t.Errorf("unexpected horizon warning for sufficient horizon, got: %q", output)
	}
}

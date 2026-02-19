package cluster

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/sirupsen/logrus"
)

// captureLogOutput runs fn and returns the log output as a string.
func captureLogOutput(fn func()) string {
	var buf bytes.Buffer
	origOutput := logrus.StandardLogger().Out
	origLevel := logrus.GetLevel()
	logrus.SetOutput(&buf)
	logrus.SetLevel(logrus.WarnLevel)
	defer func() {
		if origOutput != nil {
			logrus.SetOutput(origOutput)
		} else {
			logrus.SetOutput(os.Stderr)
		}
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

func TestClusterSimulator_AllRejected_WarnsAfterRun(t *testing.T) {
	// GIVEN a cluster with reject-all admission policy
	config := newTestDeploymentConfig(1)
	config.Horizon = 100000
	config.AdmissionPolicy = "reject-all"
	workload := newTestWorkload(5)

	cs := NewClusterSimulator(config, workload, "")

	// WHEN the simulation runs to completion
	output := captureLogOutput(func() {
		cs.Run()
	})

	// THEN a warning about all requests being rejected MUST be logged
	if !strings.Contains(output, "rejected") {
		t.Errorf("expected all-rejected warning, got: %q", output)
	}
}

func TestClusterSimulator_ZeroCompletions_WarnsAfterRun(t *testing.T) {
	// GIVEN a cluster with horizon too short for any request to finish
	config := newTestDeploymentConfig(1)
	config.Horizon = 1 // 1 tick — admits but can't finish
	workload := newTestWorkload(5)

	cs := NewClusterSimulator(config, workload, "")

	// WHEN the simulation runs to completion
	output := captureLogOutput(func() {
		cs.Run()
	})

	// THEN a warning about no completed requests MUST be logged
	if !strings.Contains(output, "no requests completed") {
		t.Errorf("expected zero-completion warning, got: %q", output)
	}
}

func TestClusterSimulator_NormalOperation_NoPostSimWarning(t *testing.T) {
	// GIVEN a properly configured cluster that will complete requests
	config := newTestDeploymentConfig(1)
	config.Horizon = 1000000
	workload := newTestWorkload(2)

	cs := NewClusterSimulator(config, workload, "")

	// WHEN the simulation runs to completion
	output := captureLogOutput(func() {
		cs.Run()
	})

	// THEN no rejection or zero-completion warnings MUST be logged
	if strings.Contains(output, "rejected") || strings.Contains(output, "no requests completed") {
		t.Errorf("unexpected warning during normal operation, got: %q", output)
	}
}

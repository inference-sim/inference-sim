package cluster

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
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

func TestClusterSimulator_AllRejected_WarnsAfterRun(t *testing.T) {
	// GIVEN a cluster whose gateway rejects every request at admission
	config := newTestDeploymentConfig(1)
	config.Horizon = 100000
	workload := newTestRequests(5)

	cs := NewClusterSimulator(config, NewSliceRequestSource(workload), nil)
	cs.SetGateway(&fakeGateway{reject: func(*sim.Request) bool { return true }})

	// WHEN the simulation runs to completion
	output := captureLogOutput(func() {
		if err := cs.Run(); err != nil {
			t.Fatalf("cs.Run: %v", err)
		}
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
	workload := newTestRequests(5)

	cs := NewClusterSimulator(config, NewSliceRequestSource(workload), nil)

	// WHEN the simulation runs to completion
	output := captureLogOutput(func() {
		if err := cs.Run(); err != nil {
			t.Fatalf("cs.Run: %v", err)
		}
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
	workload := newTestRequests(2)

	cs := NewClusterSimulator(config, NewSliceRequestSource(workload), nil)

	// WHEN the simulation runs to completion
	output := captureLogOutput(func() {
		if err := cs.Run(); err != nil {
			t.Fatalf("cs.Run: %v", err)
		}
	})

	// THEN no rejection or zero-completion warnings MUST be logged
	if strings.Contains(output, "rejected") || strings.Contains(output, "no requests completed") {
		t.Errorf("unexpected warning during normal operation, got: %q", output)
	}
}

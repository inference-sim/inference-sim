package scripts_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestDispatchWithRetry exercises the retry script with stubbed gh and sleep, verifying:
//   - success on first attempt exits 0 with no sleep
//   - success after transient failure retries and exits 0
//   - exhaustion after all failures exits 1
//   - backoff delays are 5s then 10s
//
// The stubs are tiny shell scripts placed on PATH ahead of the real binaries.
func TestDispatchWithRetry(t *testing.T) {
	scriptPath, err := filepath.Abs(filepath.Join("dispatch-with-retry.sh"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(scriptPath); err != nil {
		t.Fatalf("dispatch-with-retry.sh not found: %v", err)
	}

	cases := []struct {
		name         string
		ghStub       string // shell script body for the gh stub
		wantExit     int
		wantSleeps   []string // expected sleep arguments in order
		wantAttempts int      // expected number of gh calls
	}{
		{
			name:         "success on first attempt",
			ghStub:       `#!/bin/sh` + "\n" + `echo "gh called: $*" >> "$STUB_LOG"` + "\n" + `exit 0`,
			wantExit:     0,
			wantSleeps:   nil,
			wantAttempts: 1,
		},
		{
			name: "success on second attempt after transient failure",
			ghStub: `#!/bin/sh
echo "gh called: $*" >> "$STUB_LOG"
count=$(grep -c "gh called" "$STUB_LOG")
if [ "$count" -le 1 ]; then exit 1; fi
exit 0`,
			wantExit:     0,
			wantSleeps:   []string{"5"},
			wantAttempts: 2,
		},
		{
			name: "success on third attempt",
			ghStub: `#!/bin/sh
echo "gh called: $*" >> "$STUB_LOG"
count=$(grep -c "gh called" "$STUB_LOG")
if [ "$count" -le 2 ]; then exit 1; fi
exit 0`,
			wantExit:     0,
			wantSleeps:   []string{"5", "10"},
			wantAttempts: 3,
		},
		{
			name:         "all attempts fail exits 1",
			ghStub:       `#!/bin/sh` + "\n" + `echo "gh called: $*" >> "$STUB_LOG"` + "\n" + `exit 1`,
			wantExit:     1,
			wantSleeps:   []string{"5", "10"},
			wantAttempts: 3,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stubDir := t.TempDir()
			logFile := filepath.Join(stubDir, "stub.log")

			// Write gh stub
			ghPath := filepath.Join(stubDir, "gh")
			if err := os.WriteFile(ghPath, []byte(tc.ghStub), 0755); err != nil {
				t.Fatal(err)
			}

			// Write sleep stub that logs its argument instead of sleeping
			sleepStub := `#!/bin/sh` + "\n" + `echo "sleep $1" >> "$STUB_LOG"`
			sleepPath := filepath.Join(stubDir, "sleep")
			if err := os.WriteFile(sleepPath, []byte(sleepStub), 0755); err != nil {
				t.Fatal(err)
			}

			cmd := exec.Command("bash", scriptPath, "test-workflow.yml", "--repo", "test/repo", "--ref", "main")
			cmd.Env = append(os.Environ(),
				"PATH="+stubDir+":"+os.Getenv("PATH"),
				"STUB_LOG="+logFile,
			)
			output, err := cmd.CombinedOutput()

			// Check exit code
			exitCode := 0
			if err != nil {
				if exitErr, ok := err.(*exec.ExitError); ok {
					exitCode = exitErr.ExitCode()
				} else {
					t.Fatalf("unexpected error: %v\noutput: %s", err, output)
				}
			}
			if exitCode != tc.wantExit {
				t.Errorf("exit code = %d, want %d\noutput: %s", exitCode, tc.wantExit, output)
			}

			// Read the log
			logBytes, _ := os.ReadFile(logFile)
			logLines := strings.Split(strings.TrimSpace(string(logBytes)), "\n")
			if logLines[0] == "" {
				logLines = nil
			}

			// Count gh calls
			ghCalls := 0
			var sleeps []string
			for _, line := range logLines {
				if strings.HasPrefix(line, "gh called:") {
					ghCalls++
				}
				if strings.HasPrefix(line, "sleep ") {
					sleeps = append(sleeps, strings.TrimPrefix(line, "sleep "))
				}
			}

			if ghCalls != tc.wantAttempts {
				t.Errorf("gh called %d times, want %d\nlog:\n%s", ghCalls, tc.wantAttempts, string(logBytes))
			}

			if len(sleeps) != len(tc.wantSleeps) {
				t.Errorf("sleep called %d times %v, want %d times %v",
					len(sleeps), sleeps, len(tc.wantSleeps), tc.wantSleeps)
			} else {
				for i, want := range tc.wantSleeps {
					if sleeps[i] != want {
						t.Errorf("sleep[%d] = %q, want %q", i, sleeps[i], want)
					}
				}
			}

			// Verify success output contains "Dispatched"
			if tc.wantExit == 0 && !strings.Contains(string(output), "Dispatched") {
				t.Errorf("successful run should print 'Dispatched', got: %s", output)
			}
			// Verify failure output contains error annotation
			if tc.wantExit == 1 && !strings.Contains(string(output), "::error::") {
				t.Errorf("failed run should print ::error::, got: %s", output)
			}
		})
	}
}

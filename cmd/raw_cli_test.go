package cmd

import (
	"bytes"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// runRawCLI runs `blis args...` exactly as given, in a child process that executes the real
// root command, and returns its stdout, stderr and exit error. Unlike capRawCLI it does not
// fail the test on a refusal, so a test can assert one.
func runRawCLI(t *testing.T, args ...string) (stdout, stderr string, err error) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestKernelCLILeg$")
	cmd.Env = append(os.Environ(), kernelCLIArgsEnv+"="+strings.Join(args, "\x1f"))
	var out, errBuf bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errBuf
	err = cmd.Run()
	return out.String(), errBuf.String(), err
}

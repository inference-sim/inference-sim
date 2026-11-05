package workload

import (
	"fmt"
	"sort"
	"strings"
)

// The values of TraceHeader.Mode: which command wrote the trace.
const (
	// TraceModeGenerated is a trace blis run wrote (--trace-output). Its header records the
	// deployment BLIS ran, and replay keeps it authoritative (INV-13).
	TraceModeGenerated = "generated"
	// TraceModeReplayed is a trace blis replay wrote (--trace-output). Like a generated trace,
	// it records the deployment BLIS ran.
	TraceModeReplayed = "replayed"
	// TraceModeConverted is a trace blis convert wrote from external traffic (weka, otel). It
	// records requests only, no BLIS deployment, so replay takes the deployment from the
	// scenario and the command line (#2067).
	TraceModeConverted = "converted"
)

var traceModes = map[string]bool{
	TraceModeGenerated: true,
	TraceModeReplayed:  true,
	TraceModeConverted: true,
}

// validateTraceMode refuses a header mode outside TraceModeGenerated, TraceModeReplayed and
// TraceModeConverted, naming the field and the value. Mode "real" came from blis observe,
// removed in v0.16.0; the message says so, since that is the one value an old trace carries.
func validateTraceMode(mode string) error {
	if traceModes[mode] {
		return nil
	}
	allowed := make([]string, 0, len(traceModes))
	for m := range traceModes {
		allowed = append(allowed, m)
	}
	sort.Strings(allowed)
	if mode == "real" {
		return fmt.Errorf("trace header mode %q is not supported: it was written by blis observe, "+
			"which v0.16.0 removed (#1901); the modes are %s", mode, strings.Join(allowed, ", "))
	}
	return fmt.Errorf("trace header mode %q is not one of %s", mode, strings.Join(allowed, ", "))
}

// RecordsDeployment reports whether a trace with header h records the deployment BLIS ran:
// true for a trace blis run or blis replay wrote, false for one blis convert wrote.
func (h TraceHeader) RecordsDeployment() bool {
	return h.Mode != TraceModeConverted
}

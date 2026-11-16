// Package corpora names the roots the offline scorers under tools/ read.
package corpora

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// The roots these offline scorers read by default: the cost model's scenario fixtures, the
// catalog/ and registry/ of the blis-substrate module this build selects, and the measurement
// corpora under BLIS_MEASUREMENTS, which no BLIS repository redistributes. These scorers move
// to accuracy-dashboard (design §2.2, COV-1) and are deleted here when they have.

const measurementsEnv = "BLIS_MEASUREMENTS"

const measurementsHow = "export " + measurementsEnv + "=<directory holding the extracted corpora>"

// DefaultMeasurements is the measurement-corpus directory, or "" when BLIS_MEASUREMENTS is
// unset.
func DefaultMeasurements() string { return os.Getenv(measurementsEnv) }

// MeasurementPath is one corpus under BLIS_MEASUREMENTS, or "" when that is unset.
func MeasurementPath(name string) string {
	dir := DefaultMeasurements()
	if dir == "" {
		return ""
	}
	return filepath.Join(dir, name)
}

// RequireCorpora checks that each named corpus flag holds a path.
func RequireCorpora(flags map[string]string) error {
	var names []string
	for name, path := range flags {
		if path == "" {
			names = append(names, "-"+name)
		}
	}
	if len(names) == 0 {
		return nil
	}
	sort.Strings(names)
	return fmt.Errorf("%s not set, and %s is unset, so no default exists; either pass the paths or:\n  %s",
		strings.Join(names, ", "), measurementsEnv, measurementsHow)
}

// Package corpora names the roots the offline scorers under tools/ read.
package corpora

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/internal/kernelfixtures"
	"github.com/inference-sim/inference-sim/internal/substratedata"
)

// The roots these offline scorers read by default: the cost model's scenario fixtures, the
// catalog/ and registry/ of the blis-substrate module this build selects, and the measurement
// corpora under BLIS_MEASUREMENTS, which no BLIS repository redistributes. These scorers move
// to accuracy-dashboard (design §2.2, COV-1) and are deleted here when they have.

const measurementsEnv = "BLIS_MEASUREMENTS"

const measurementsHow = "export " + measurementsEnv + "=<directory holding the extracted corpora>"

// DefaultRepos is the scenario, catalog and registry roots.
func DefaultRepos() kernelmodel.Repos {
	return kernelmodel.Repos{
		Scenarios: kernelfixtures.AISimulate(),
		Catalog:   substratedata.MustCatalog(),
		Registry:  substratedata.MustRegistry(),
	}
}

// DefaultMeasurements is the measurement-corpus directory, or "" when BLIS_MEASUREMENTS is
// unset.
func DefaultMeasurements() string { return os.Getenv(measurementsEnv) }

// RequireMeasurements checks that the measurement-corpus directory is set and exists.
func RequireMeasurements(dir string) error {
	if dir == "" {
		return fmt.Errorf("measurements root is not set; to set it:\n  %s", measurementsHow)
	}
	if _, err := os.Stat(dir); err != nil {
		return fmt.Errorf("measurements root %q is unreadable (%v); to set it:\n  %s", dir, err, measurementsHow)
	}
	return nil
}

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

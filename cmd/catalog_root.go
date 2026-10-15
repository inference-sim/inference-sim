package cmd

import (
	"fmt"
	"os"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

const (
	// catalogEnvVar names the environment variable that locates the model catalog when
	// --catalog is not given (#1731, R1/S4). It is read exactly like HF_TOKEN was — the
	// only environment variables cmd/ consults.
	catalogEnvVar = "BLIS_CATALOG"
	// catalogModelsSubdir is the models namespace inside a catalog CLONE ROOT (#1774).
	// --catalog / BLIS_CATALOG names the clone root; blis-latency-kernel reads a model's
	// derived graph from <catalog>/models/<name>/graph.yaml. The authoritative blis-catalog
	// repository lays its entries out there, with workloads/, devices/, hardware/ and
	// networks/ as SIBLING namespaces under the same root — those siblings only compose
	// off a single root, which is why the root (not the models directory) is the
	// contract every downstream reader inherits.
	catalogModelsSubdir = "models"
	// catalogModelGraphFile is the half of a model entry a run reads: the model graph
	// blis-catalog derives from the vendor config, loaded by blis-schemas' LoadModelGraph.
	catalogModelGraphFile = "graph.yaml"
)

// catalogRootFrom picks the model-catalog root from the --catalog flag value and the
// BLIS_CATALOG environment value (#1731). The flag wins when both are set (an explicit
// CLI input beats the environment), and the override is announced on stderr so the choice
// is visible in the run's history. With NEITHER there is no default, no search path and
// no remote fetch: the caller is refused, naming both forms. The retired working-directory
// default (model_configs/ resolved against --defaults-filepath's directory) meant
// `blis run` silently worked only from the repository root, and inferring a catalog
// location is exactly what NS-6 forbids — nothing about the model is inferred.
//
// Pure with respect to process state: both inputs are supplied by the caller, so the
// precedence law is table-testable without touching globals or the environment.
func catalogRootFrom(flagValue, envValue string) (string, error) {
	if flagValue != "" {
		if envValue != "" && envValue != flagValue {
			// Warnf, not Infof: both commands default --log to warn, so an Infof precedence
			// notice would be silent under normal invocation. This choice overrides an
			// explicit BLIS_CATALOG, so the announcement must be visible in the run's
			// history at the default log level (qa-review G3, #1731).
			logrus.Warnf("--catalog %q takes precedence over %s=%q", flagValue, catalogEnvVar, envValue)
		}
		return flagValue, nil
	}
	if envValue != "" {
		logrus.Infof("model catalog located via %s=%q", catalogEnvVar, envValue)
		return envValue, nil
	}
	return "", fmt.Errorf("no model catalog was supplied: pass --catalog <path> or set the %s "+
		"environment variable (there is no default and no search path; both name the catalog "+
		"CLONE ROOT, whose model entries live at <catalog>/%s/<name>/%s. A relative path "+
		"is resolved against the current working directory)",
		catalogEnvVar, catalogModelsSubdir, catalogModelGraphFile)
}

// resolveCatalogRoot resolves the catalog root from the --catalog flag and the
// BLIS_CATALOG environment variable, then verifies that it EXISTS AND IS A DIRECTORY. A
// missing, unstatable or non-directory catalog is refused naming both forms (R1) rather than
// deferred into a per-model "not in the catalog" message that would blame the model for
// a mistyped catalog path.
//
// The check is deliberately existence + directory-ness and NOT an accessibility probe
// (#1776, which asked for one or the other and got this one). BLIS never LISTS the catalog
// — it opens the files a scenario names, at paths it derives — so a root with search-only
// permission (mode --x) is perfectly usable, and probing for read permission here would
// refuse a catalog that works. There is no portable probe for "can traverse but not list",
// so the honest thing is to state what is checked. The residual case (a root that exists,
// is a directory, but cannot be traversed) therefore still surfaces one layer down, where
// the reader of the first catalog file reports the real errno naming the path it could not
// open.
//
// Three dispositions rather than one (#1776), because "%q is not readable: no such file or
// directory" described a mistyped path as a permission problem:
//   - absent      → the path does not exist (the overwhelmingly common typo)
//   - unstatable  → the real error (EACCES on a parent, EIO, a symlink loop)
//   - not a dir   → a file where the catalog root should be
func resolveCatalogRoot() (string, error) {
	root, err := catalogRootFrom(catalogPath, os.Getenv(catalogEnvVar))
	if err != nil {
		return "", err
	}
	info, statErr := os.Stat(root)
	switch {
	case statErr != nil && os.IsNotExist(statErr):
		return "", fmt.Errorf("model catalog %q does not exist (set --catalog or %s to the catalog root; "+
			"a relative path is resolved against the current working directory)", root, catalogEnvVar)
	case statErr != nil:
		return "", fmt.Errorf("model catalog %q cannot be inspected: %w (set --catalog or %s to the catalog root)",
			root, statErr, catalogEnvVar)
	case !info.IsDir():
		return "", fmt.Errorf("model catalog %q is not a directory (set --catalog or %s to the catalog root; "+
			"it names the catalog CLONE ROOT, not a file inside it)", root, catalogEnvVar)
	}
	return root, nil
}

// registerCatalogFlag declares --catalog on the given command. It is the ONE declaration of
// the flag (R23): every command that locates the catalog registers it from here, so the four
// commands cannot drift into different help text — or, worse, different defaults — for the
// input that decides which catalog a run reads.
//
// Registered by `run` and `replay` (via registerSimConfigFlags, for the kernel's catalog) and by
// `observe` and `convert preset` (for the workload presets, #1769). The flag var is
// package-level, like every other cobra binding in cmd/, so sharing it across commands is
// safe: one command runs per process.
func registerCatalogFlag(cmd *cobra.Command) {
	cmd.Flags().StringVar(&catalogPath, "catalog", "", "Path to the catalog CLONE ROOT (#1774; clone https://github.com/inference-sim/blis-catalog). A scenario's model graph is read from <catalog>/"+catalogModelsSubdir+"/<name>/"+catalogModelGraphFile+", its chip and fabric from the hardware/ and networks/ namespaces; a named workload preset is read from <catalog>/"+catalogWorkloadsSubdir+"/<name>"+presetFileExt+" (#1769). Path semantics: a RELATIVE value is resolved against the current working directory, an ABSOLUTE value is used as given. No default and no search path — supply this flag or the "+catalogEnvVar+" environment variable (the flag wins when both are set), or the run is refused naming both. BLIS never fetches or writes a catalog file at run time: an uncatalogued model is refused naming the path its entry belongs at (NS-6, #1733)")
}

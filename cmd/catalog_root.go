package cmd

import (
	"fmt"
	"os"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-substrate/load"
)

// The data a run reads (design §3.3, SIM-6): the catalog and registry embedded in the
// blis-substrate release this build pins, unless a path replaces either one. --catalog <dir>
// (or BLIS_CATALOG) and --registry <dir> (or BLIS_REGISTRY) each name a directory with the
// layout of blis-substrate's catalog/ or registry/ and replace that half of the embedded copy;
// the two are independent. --catalog-overlay and --registry-overlay add or replace entries by
// identity, and may be repeated. A path never overlays, and an overlay never replaces the
// whole. The flag wins over the environment variable, and the choice is announced on stderr.
//
// Every command that reads data opens it once, through runData, so the scenario's model, the
// engine and the Result's data record all come from one load.Data: the loader checks it in
// full when it opens, and its selected-entry report covers every reader (VAL-6).
const (
	catalogEnvVar  = "BLIS_CATALOG"
	registryEnvVar = "BLIS_REGISTRY"

	// catalogModelsSubdir, catalogModelGraphFile and catalogHardwareSubdir name the catalog's
	// layout, for messages and for the catalog-load gate.
	catalogModelsSubdir   = "models"
	catalogModelGraphFile = "graph.yaml"
)

var (
	catalogPath      string   // --catalog
	registryPath     string   // --registry
	catalogOverlays  []string // --catalog-overlay
	registryOverlays []string // --registry-overlay

	// openedData is the run's data, set by runData; nil until a command opens it. openedFor
	// is the options it was opened with.
	openedData *load.Data
	openedFor  string
)

// pathFrom picks one directory from its flag and its environment variable: the flag wins when
// both are set, and the override is announced. Empty means the embedded copy. Pure with
// respect to process state, so the precedence law is table-testable.
func pathFrom(flag, flagValue, env, envValue string) string {
	switch {
	case flagValue != "":
		if envValue != "" && envValue != flagValue {
			logrus.Warnf("--%s %q takes precedence over %s=%q", flag, flagValue, env, envValue)
		}
		return flagValue
	case envValue != "":
		logrus.Infof("--%s not given; %s=%q replaces the embedded copy", flag, env, envValue)
		return envValue
	}
	return ""
}

// dataOptions are the loader options the command line and environment state.
func dataOptions() load.Options {
	return load.Options{
		Catalog:          pathFrom("catalog", catalogPath, catalogEnvVar, os.Getenv(catalogEnvVar)),
		Registry:         pathFrom("registry", registryPath, registryEnvVar, os.Getenv(registryEnvVar)),
		CatalogOverlays:  catalogOverlays,
		RegistryOverlays: registryOverlays,
	}
}

// runData opens the run's catalog and registry once and returns the same Data on every later
// call. The engine opens its data through kernelmodel.OpenData with the same options, which
// caches one Data per options, so the simulator and the engine read one Data.
//
// The options are read on every call, and a call with options other than the opened Data's
// opens those: one process can run several commands (the tests do), and each must read the data
// its own flags name.
func runData() (*load.Data, error) {
	o := dataOptions()
	key := fmt.Sprintf("%q", []any{o.Catalog, o.Registry, o.CatalogOverlays, o.RegistryOverlays})
	if openedData != nil && openedFor == key {
		return openedData, nil
	}
	d, err := openEngineData(o)
	if err != nil {
		return nil, fmt.Errorf("the catalog and registry: %w", err)
	}
	openedData, openedFor = d, key
	return d, nil
}

// mustRunData is runData at the CLI boundary: a refusal is fatal.
func mustRunData() *load.Data {
	d, err := runData()
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return d
}

// registerDataFlags declares the four data flags on a command. It is their one declaration
// (R23), so the commands cannot drift into different help text for the input that decides
// what data a run reads.
func registerDataFlags(cmd *cobra.Command) {
	registerCatalogFlag(cmd)
	cmd.Flags().StringVar(&registryPath, "registry", "", "A directory with the layout of blis-substrate's registry/ that replaces the embedded registry (or "+registryEnvVar+"; the flag wins). Without either the run uses the registry embedded in the blis-substrate release this build pins (design §3.3)")
	cmd.Flags().StringArrayVar(&catalogOverlays, "catalog-overlay", nil, "A directory of catalog entries (whole models/<name>/ directories, or hardware, device, network or workload files) added to the catalog in use, replacing an entry of the same identity; repeatable, applied in order. It may hold no ledger files")
	cmd.Flags().StringArrayVar(&registryOverlays, "registry-overlay", nil, "A directory of coefficient sets added to the registry in use: a coefficient with the same set, name, scope and method replaces the one there; repeatable, applied in order")
}

// registerCatalogFlag declares --catalog alone, for a command that reads only the catalog
// (convert preset, catalog resolve).
func registerCatalogFlag(cmd *cobra.Command) {
	cmd.Flags().StringVar(&catalogPath, "catalog", "", "A directory with the layout of blis-substrate's catalog/ that replaces the embedded catalog (or "+catalogEnvVar+"; the flag wins). Without either the command uses the catalog embedded in the blis-substrate release this build pins (design §3.3). A model is resolved from its models/<name>/graph.yaml; a model the catalog does not have is refused, naming the catalog's source and where the entry belongs. BLIS never fetches or writes a catalog file (NS-6)")
}

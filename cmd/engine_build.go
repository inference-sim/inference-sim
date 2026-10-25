package cmd

import (
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-vllm/compat/build"
	"github.com/inference-sim/inference-sim/sim"
)

// The one place inference-sim builds blis-vllm (#2021 B2). The engine's configuration -- the
// scenario's engine-side blocks and the command line's restatements of them -- is decoded and
// resolved by blis-vllm's compat/build, which cmd calls only from here, handing the inputs over
// unread. A refusal comes back as an error; the CLI boundary turns it into the fatal exit.

// resolveKVOffloadConfig resolves the run's KV offload: the scenario's offload block (#1910)
// and the --kv-offload-config file, which may only restate it (build.ResolveKVOffloadConfig).
// Single construction site (R4), shared by run and replay for INV-13 parity.
func resolveKVOffloadConfig(cmd *cobra.Command) sim.KVOffloadConfig {
	cfg, err := build.ResolveKVOffloadConfig(kernelScenario, scenarioOffload, kvOffloadConfigPath,
		blockSizeTokens, resolveCatalogRoot)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return cfg
}

// loadCatalogStorageDevices reads the catalog's storage-device table through the engine's own
// reader, so the catalog-load gate checks the file the way a run reads it (C6).
func loadCatalogStorageDevices(catalog string) (map[string]build.KVOffloadDevice, error) {
	return build.LoadCatalogStorageDevices(catalog)
}

// catalogStorageDevicesPath, catalogDevicesSubdir and catalogStorageDevicesFile locate the
// storage-device table inside a catalog clone root, for the catalog-load gate's diagnostics.
func catalogStorageDevicesPath(catalog string) string {
	return build.CatalogStorageDevicesPath(catalog)
}

const (
	catalogDevicesSubdir      = build.CatalogDevicesSubdir
	catalogStorageDevicesFile = build.CatalogStorageDevicesFile
)

// catalogStorageDevicesRelPath is the table's location relative to the catalog clone root,
// for diagnostics that name it.
var catalogStorageDevicesRelPath = build.CatalogStorageDevicesRelPath

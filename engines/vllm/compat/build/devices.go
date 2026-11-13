// Moved from inference-sim/cmd/catalog_devices.go at ace1528fe4bba59e510eda48e80ee9b8142ea982.

// Package build is the compatibility engine's configuration: it decodes and resolves the
// engine-side blocks of a scenario (its KV offload block, its engine and pools) and the
// command-line restatements of them, so the simulator hands them over unread (#2021 B2, B6).
// It moved here from inference-sim's cmd/ (#2021 E3) and is called from inference-sim's one
// build site.
package build

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/load"
)

// catalogEnvVar names the environment variable that names a catalog directory, for the
// diagnostics that tell an operator where to point it (inference-sim's --catalog flag and
// BLIS_CATALOG; inference-sim opens the data and passes it in).
const catalogEnvVar = "BLIS_CATALOG"

const (
	// CatalogDevicesSubdir is the devices namespace inside a catalog CLONE ROOT
	// (#1774) — a SIBLING of models/, hardware/, workloads/ and networks/, which is
	// why the flag names the clone root rather than the models directory.
	CatalogDevicesSubdir = "devices"
	// CatalogStorageDevicesFile is the KV-offload storage-device physics table inside
	// that namespace: a top-level map of device_class name -> physics.
	CatalogStorageDevicesFile = "storage.yaml"
)

// KVOffloadDevice is one device_class's physics for KV offload (H5, #1587). Shipped
// catalog data (operator input), never fitted (BC-G4). Inert: only consulted when a
// --kv-offload-config secondary tier names a device_class.
//
// #1770: this table lives in the CATALOG (<catalog>/devices/storage.yaml), not in
// defaults.yaml. It used to be duplicated between the two with nothing keeping the
// copies in sync; the catalog copy is now the single source of truth and the
// defaults.yaml kv_offload_devices: block (and its Config.KVOffloadDevices field) are
// deleted.
//
// The FILE format is blis-substrate's hardware.StorageDevice, decoded and validated by its
// loader when the catalog is opened, so BLIS holds no second parser for it. It names a tier class
// and states no rate: the read rate, the write rate and the fixed cost per transfer are registry
// coefficients (offload_tier_<tier>_read_rate, _write_rate, _base_latency, in the set
// cost-model-offload-tiers), and a file that states one is refused by the loader.
//
// UNITS. ReadBandwidth/WriteBandwidth are bytes per microsecond; BaseLatency is
// microseconds. Bytes/µs and MB/s (MB = 10^6 bytes, not MiB) are the SAME number —
// 1 MB/s = 10^6 bytes / 10^6 µs = 1 byte/µs — so the catalog's _mb_s figure is read
// unchanged. Neither is a conversion of the other; they are two spellings of one unit.
//
// #1581 adds an optional non-linear device model, all opt-in (a device that omits
// these fields resolves byte-identically to pre-#1581, INV-6):
//   - a queue-depth bandwidth ramp (saturation_queue_depth Qsat + single_transfer_
//     fraction f₁): effective bandwidth ramps from f₁·peak at in-service depth q=1
//     up to the peak (read/write_bandwidth) at q=Qsat, flat beyond;
//   - a relative latency jitter stddev (latency_jitter_stddev σ);
//   - a buffered-I/O regime (buffered_*), selected per tier by direct_io=false; any
//     absent buffered field falls back to the O_DIRECT value.
//
// Optional fields are pointers so "absent" is distinct from an explicit zero (R9).
//
// The #1581 device-model fields have no catalog source: blis-schemas' StorageDevice does
// not define them, and its strict loader refuses a key it does not define, so a catalog
// that passes its own validation cannot state them. A device read from the catalog
// therefore always resolves with the ramp and jitter off. Carrying them in the catalog
// needs those fields added to blis-schemas first.
type KVOffloadDevice struct {
	ReadBandwidth  float64
	WriteBandwidth float64
	BaseLatency    float64

	// O_DIRECT regime device model (optional; absent => no ramp / no jitter).
	SaturationQueueDepth   *int64
	SingleTransferFraction *float64
	LatencyJitterStddev    *float64

	// Buffered-I/O regime (optional; each absent field falls back to O_DIRECT).
	BufferedReadBandwidth          *float64
	BufferedWriteBandwidth         *float64
	BufferedBaseLatency            *float64
	BufferedSaturationQueueDepth   *int64
	BufferedSingleTransferFraction *float64
	BufferedLatencyJitterStddev    *float64
}

// CatalogStorageDevicesRelPath is the canonical location of the storage-device table
// relative to the catalog clone root, used in diagnostics so an operator who has not
// resolved a catalog yet still learns where the file belongs. Slash-separated, as every
// diagnostic that names it has always printed it.
const CatalogStorageDevicesRelPath = CatalogDevicesSubdir + "/" + CatalogStorageDevicesFile

// StorageDevices is the catalog's KV-offload storage-device table, read from the run's data
// (blis-substrate's loader, which parsed and validated devices/storage.yaml when it opened
// the catalog), converted to BLIS's physics records. It is only called when a secondary tier
// names a device_class, so a catalog with no devices stays usable for every other run (INV-6).
//
// A catalog with no device classes is an error rather than an empty map: an empty map would
// degrade into a "device_class is not defined (known: <none configured>)" message that blames
// the operator's device_class for a missing catalog file, and zero bandwidth would be
// plausible-but-wrong physics.
func StorageDevices(data *load.Data) (map[string]KVOffloadDevice, error) {
	if data == nil {
		return nil, fmt.Errorf("kv_offload: a secondary tier names a device_class, which is resolved from "+
			"the catalog's %s, and no catalog was given", CatalogStorageDevicesRelPath)
	}
	loaded := data.StorageDevices()
	if len(loaded) == 0 {
		return nil, fmt.Errorf("kv_offload: the catalog (%s) defines no device classes in %s, so the "+
			"device_class named by a secondary tier cannot be resolved; point --catalog or %s at a "+
			"catalog that has them", data.Catalog.Source, CatalogStorageDevicesRelPath, catalogEnvVar)
	}
	// The catalog names a tier class and states no rate for it: the read rate, the write rate
	// and the fixed cost per transfer are registry coefficients, scoped to the hardware they
	// describe, and the kernel prices every catalog-class tier from them (priceOffload installs
	// the pricer and refuses a tier whose entries the scenario does not list). A class therefore
	// resolves to a name alone; its rates stay zero here and nothing reads them.
	devices := make(map[string]KVOffloadDevice, len(loaded))
	for _, d := range loaded {
		devices[d.Name] = KVOffloadDevice{}
	}
	return devices, nil
}

// BlockNamesDeviceClass reports whether any secondary tier in a parsed kv_offload
// block names a device_class. It is what makes the catalog table a LAZY dependency:
// a config whose tiers all carry an explicit bandwidth/latency triple resolves without
// the catalog devices/ namespace existing at all, exactly as it did before #1770.
func BlockNamesDeviceClass(block *KVOffloadBlock) bool {
	if block == nil {
		return false
	}
	for _, tb := range block.SecondaryTiers {
		if tb.DeviceClass != nil {
			return true
		}
	}
	return false
}

// ResolveKVOffloadDevices resolves the device_class physics table for a parsed
// kv_offload block: nil when no tier names a device_class (no catalog read at all),
// otherwise the catalog's devices/storage.yaml, from the run's data.
//
// run and replay both reach this through the single ResolveKVOffloadConfig site, so
// neither can drift into reading a different table (INV-13).
func ResolveKVOffloadDevices(block *KVOffloadBlock, data *load.Data) (map[string]KVOffloadDevice, error) {
	if !BlockNamesDeviceClass(block) {
		return nil, nil
	}
	return StorageDevices(data)
}

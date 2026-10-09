package cmd

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"syscall"

	blisschemas "github.com/inference-sim/blis-schemas"
)

const (
	// catalogDevicesSubdir is the devices namespace inside a catalog CLONE ROOT
	// (#1774) — a SIBLING of models/, hardware/, workloads/ and networks/, which is
	// why the flag names the clone root rather than the models directory.
	catalogDevicesSubdir = "devices"
	// catalogStorageDevicesFile is the KV-offload storage-device physics table inside
	// that namespace: a top-level map of device_class name -> physics.
	catalogStorageDevicesFile = "storage.yaml"
)

// kvOffloadDevice is one device_class's physics for KV offload (H5, #1587). Shipped
// catalog data (operator input), never fitted (BC-G4). Inert: only consulted when a
// --kv-offload-config secondary tier names a device_class.
//
// #1770: this table lives in the CATALOG (<catalog>/devices/storage.yaml), not in
// defaults.yaml. It used to be duplicated between the two with nothing keeping the
// copies in sync; the catalog copy is now the single source of truth and the
// defaults.yaml kv_offload_devices: block (and its Config.KVOffloadDevices field) are
// deleted.
//
// The FILE format is blis-schemas' hardware.StorageDevice, decoded by its own strict
// LoadStorageDevices, so BLIS holds no second parser for it. Its keys carry their units
// (read_bandwidth_mb_s, write_bandwidth_mb_s, base_latency_us).
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
type kvOffloadDevice struct {
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

// catalogStorageDevicesRelPath is the canonical location of the storage-device table
// relative to the catalog clone root, used in diagnostics so an operator who has not
// resolved a catalog yet still learns where the file belongs.
var catalogStorageDevicesRelPath = filepath.Join(catalogDevicesSubdir, catalogStorageDevicesFile)

// catalogStorageDevicesPath returns <catalog>/devices/storage.yaml. Pure: the catalog
// root is an argument, so the path law is table-testable without touching globals or
// the environment (same shape as catalogModelDirs, #1774).
func catalogStorageDevicesPath(catalog string) string {
	return filepath.Join(catalog, catalogDevicesSubdir, catalogStorageDevicesFile)
}

// parseCatalogStorageDevices loads and validates the storage-device table at path through
// blis-schemas, and converts each device to BLIS's physics record. It is the one reader of
// the file: loadCatalogStorageDevices and the legacy transfer path both call it and differ
// only in their diagnostics. A filesystem failure is distinguishable with isNotReadable.
func parseCatalogStorageDevices(path string) (map[string]kvOffloadDevice, error) {
	// A directory where the file belongs opens fine and fails mid-decode inside the YAML
	// reader, which does not wrap the *fs.PathError; stat first so it reads as unreadable.
	if info, err := os.Stat(path); err != nil {
		return nil, err
	} else if info.IsDir() {
		return nil, &fs.PathError{Op: "read", Path: path, Err: syscall.EISDIR}
	}
	loaded, err := blisschemas.LoadStorageDevices(path)
	if errors.Is(err, io.EOF) {
		// A file with no YAML document at all (empty, or comments only) is a table with no
		// classes, which the callers report with an actionable message rather than "EOF".
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	devices := make(map[string]kvOffloadDevice, len(loaded))
	for _, d := range loaded {
		if problems := d.Validate(); !problems.OK() {
			return nil, fmt.Errorf("device %q: %s", d.Name, problems.Error())
		}
		devices[d.Name] = kvOffloadDevice{
			ReadBandwidth:  d.ReadBandwidthMBs,
			WriteBandwidth: d.WriteBandwidthMBs,
			BaseLatency:    d.BaseLatencyUs,
		}
	}
	return devices, nil
}

// loadCatalogStorageDevices reads and strictly parses the catalog's KV-offload
// storage-device table. It is only called when a --kv-offload-config secondary tier
// actually names a device_class, so a catalog with no devices/ namespace stays
// perfectly usable for every other run (INV-6: nothing about a run that does not name
// a device_class changes).
//
// Every failure names the path (R1). An absent, unreadable, malformed or EMPTY table is
// a hard error rather than an empty map: an empty map would degrade into a
// "device_class is not defined (known: <none configured>)" message that blames the
// operator's device_class for a missing catalog file, and zero bandwidth would be
// plausible-but-wrong physics — the silent-zero class of defect R10's strict parsing
// exists to prevent.
func loadCatalogStorageDevices(catalog string) (map[string]kvOffloadDevice, error) {
	path := catalogStorageDevicesPath(catalog)
	devices, err := parseCatalogStorageDevices(path)
	if err != nil {
		if isNotReadable(err) {
			return nil, fmt.Errorf(
				"kv_offload: a secondary tier names a device_class, but the catalog storage-device "+
					"table at %s is not readable: %w.\n"+
					"  Add it to the catalog (--catalog / %s names the catalog CLONE ROOT; the table "+
					"lives at its %s), point --catalog / %s at a catalog that has it, or give each "+
					"secondary tier an explicit read_bandwidth + write_bandwidth + base_latency triple "+
					"instead of a device_class",
				path, err, catalogEnvVar, catalogStorageDevicesRelPath, catalogEnvVar)
		}
		return nil, fmt.Errorf("kv_offload: catalog storage-device table %s is malformed: %w", path, err)
	}
	if len(devices) == 0 {
		return nil, fmt.Errorf(
			"kv_offload: catalog storage-device table %s defines no device classes, so the "+
				"device_class named by a secondary tier cannot be resolved (fix that catalog file, "+
				"or give each secondary tier an explicit read_bandwidth + write_bandwidth + "+
				"base_latency triple instead)", path)
	}
	return devices, nil
}

// blockNamesDeviceClass reports whether any secondary tier in a parsed kv_offload
// block names a device_class. It is what makes the catalog table a LAZY dependency:
// a config whose tiers all carry an explicit bandwidth/latency triple resolves without
// the catalog devices/ namespace existing at all, exactly as it did before #1770.
func blockNamesDeviceClass(block *kvOffloadBlock) bool {
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

// resolveKVOffloadDevices resolves the device_class physics table for a parsed
// kv_offload block: nil when no tier names a device_class (no catalog read at all),
// otherwise the catalog's devices/storage.yaml, located by the same
// --catalog / BLIS_CATALOG resolver the model config uses.
//
// run and replay both reach this through the single resolveKVOffloadConfig site, so
// neither can drift into reading a different table (INV-13). observe derives no
// offload config and never calls it.
func resolveKVOffloadDevices(block *kvOffloadBlock) (map[string]kvOffloadDevice, error) {
	if !blockNamesDeviceClass(block) {
		return nil, nil
	}
	catalog, err := resolveCatalogRoot()
	if err != nil {
		return nil, fmt.Errorf("kv_offload: a secondary tier names a device_class, which is "+
			"resolved from the catalog's %s: %w", catalogStorageDevicesRelPath, err)
	}
	return loadCatalogStorageDevices(catalog)
}

// isNotReadable reports whether a catalog-loader error is a filesystem failure (absent,
// unreadable, a directory where a file belongs) rather than a malformed document. The
// blis-schemas loaders wrap the *fs.PathError, so the distinction survives them.
func isNotReadable(err error) bool {
	var pathErr *fs.PathError
	return errors.As(err, &pathErr)
}

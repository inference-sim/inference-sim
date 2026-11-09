package cmd

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"

	"github.com/inference-sim/blis-vllm/compat/build"
)

// The KV offload configuration is resolved by blis-vllm's compat/build (#2021 E3), and its unit
// tests moved there. These helpers, copied unchanged, build resolved configs for the tests that
// stay here: the trace-header round trip and the replay reconciliation, which are
// inference-sim's own.

// testDevices is a fixed device map for resolver tests (independent of defaults.yaml).
func testDevices() map[string]build.KVOffloadDevice {
	return map[string]build.KVOffloadDevice{
		"nvme_gen4": {ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80},
		"sata_ssd":  {ReadBandwidth: 550, WriteBandwidth: 500, BaseLatency: 150},
	}
}

func i64p(v int64) *int64 { return &v }

func f64p(v float64) *float64 { return &v }

func strp(v string) *string { return &v }

func boolp(v bool) *bool { return &v }

// validTierBlock returns a minimal valid fs tier block using a device_class.
func validTierBlock() build.KVOffloadTierBlock {
	return build.KVOffloadTierBlock{
		Type:        strp("fs"),
		RootDir:     strp("/mnt/kv"),
		DirectIO:    boolp(true),
		DeviceClass: strp("nvme_gen4"),
	}
}

// validBlock returns a minimal valid kv_offload block with one fs tier.
func validBlock() *build.KVOffloadBlock {
	return &build.KVOffloadBlock{
		CPUBytesToUse:  i64p(17179869184),
		SecondaryTiers: []build.KVOffloadTierBlock{validTierBlock()},
	}
}

func rampBlock(deviceClass string, directIO bool) *build.KVOffloadBlock {
	return &build.KVOffloadBlock{
		CPUBytesToUse: i64p(17179869184),
		SecondaryTiers: []build.KVOffloadTierBlock{{
			Type:        strp("fs"),
			RootDir:     strp("/mnt/kv"),
			DirectIO:    boolp(directIO),
			DeviceClass: strp(deviceClass),
		}},
	}
}

// deviceModelDevices is a synthetic device map exercising the #1581 device model:
// nvme_ramp declares an O_DIRECT ramp + jitter and a fully-specified buffered
// regime; nvme_partial declares an O_DIRECT ramp but NO buffered fields (buffered
// must fall back to the O_DIRECT values).
func deviceModelDevices() map[string]build.KVOffloadDevice {
	return map[string]build.KVOffloadDevice{
		"nvme_ramp": {
			ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
			SaturationQueueDepth:           i64p(8),
			SingleTransferFraction:         f64p(0.4),
			LatencyJitterStddev:            f64p(0.10),
			BufferedReadBandwidth:          f64p(4000),
			BufferedWriteBandwidth:         f64p(3000),
			BufferedBaseLatency:            f64p(120),
			BufferedSaturationQueueDepth:   i64p(4),
			BufferedSingleTransferFraction: f64p(0.6),
			BufferedLatencyJitterStddev:    f64p(0.25),
		},
		"nvme_partial": {
			ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
			SaturationQueueDepth:   i64p(8),
			SingleTransferFraction: f64p(0.4),
			LatencyJitterStddev:    f64p(0.10),
		},
	}
}

// writeCatalogStorageDevices writes a whole catalog (a copy of the embedded one) whose
// devices/storage.yaml is body, and returns its root: a catalog named by --catalog replaces the
// embedded copy whole (design §3.3), so a test of the device table writes a complete one.
func writeCatalogStorageDevices(t *testing.T, body string) string {
	t.Helper()
	root := filepath.Join(t.TempDir(), "catalog")
	if err := os.CopyFS(root, os.DirFS(testCatalogDir())); err != nil {
		t.Fatalf("copy the catalog: %v", err)
	}
	path := filepath.Join(root, catalogDevicesSubdir, catalogStorageDevicesFile)
	if err := os.Chmod(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	_ = os.Chmod(path, 0o644)
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write storage.yaml: %v", err)
	}
	return root
}

// resolveKVOffload resolves a parsed block with the engine's resolver.
func resolveKVOffload(block *build.KVOffloadBlock, devices map[string]build.KVOffloadDevice, gpuBlockSizeTokens int64) (kvOffloadConfig, error) {
	return build.ResolveKVOffload(block, devices, gpuBlockSizeTokens)
}

// parseKVOffloadBytes strictly decodes kv_offload YAML bytes with the engine's parser.
func parseKVOffloadBytes(data []byte) (build.KVOffloadFile, error) {
	return build.ParseKVOffloadBytes(data)
}

// kvOffloadBlockFromDeployment maps a scenario's offload block with the engine's mapping.
func kvOffloadBlockFromDeployment(o *deployment.Offload) (*build.KVOffloadBlock, error) {
	return build.KVOffloadBlockFromDeployment(o)
}

const simpleConnector = build.SimpleConnector

// resolveKVOffloadDevices resolves a block's device table the way a run does: the engine's
// lookup in the run's data (runData, from --catalog / BLIS_CATALOG). A block that names no device
// class reads no data.
func resolveKVOffloadDevices(block *build.KVOffloadBlock) (map[string]build.KVOffloadDevice, error) {
	if !build.BlockNamesDeviceClass(block) {
		return build.ResolveKVOffloadDevices(block, nil)
	}
	d, err := runData()
	if err != nil {
		return nil, err
	}
	return build.ResolveKVOffloadDevices(block, d)
}

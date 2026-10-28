package cmd

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"

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

// writeCatalogStorageDevices builds a catalog root holding devices/storage.yaml with the
// given body and returns the root. Used by the device-table tests here and by the
// retargeted YAML-parse tests in kv_offload_test.go / kv_offload_device_test.go.
func writeCatalogStorageDevices(t *testing.T, body string) string {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, catalogDevicesSubdir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	if err := os.WriteFile(filepath.Join(dir, catalogStorageDevicesFile), []byte(body), 0o644); err != nil {
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
// lazy lookup with this CLI's --catalog / BLIS_CATALOG resolver (engine_build.go).
func resolveKVOffloadDevices(block *build.KVOffloadBlock) (map[string]build.KVOffloadDevice, error) {
	return build.ResolveKVOffloadDevices(block, resolveCatalogRoot)
}

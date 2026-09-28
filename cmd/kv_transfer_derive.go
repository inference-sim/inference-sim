package cmd

import (
	"fmt"
	"math"
	"os"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

// Legacy single-CPU-tier CPU↔GPU KV-transfer cost (#1819, the simulator half of R2G3b).
//
// WHAT CHANGED. The legacy transfer rate used to be an unsourced Go flag default
// (--kv-transfer-bandwidth = 100.0). R2G3b (blis-registry#4, PR #14) established that a
// blocks/tick rate is neither a bus fact nor a measured correction — it folds a
// model-dependent block size into a rate — so the registry refused to transcribe it and
// pushed the conversion here. The physical CPU↔GPU bandwidth is a CATALOG fact
// (<catalog>/devices/storage.yaml, device class cpu_dram), and the per-block tick cost is
// DERIVED from it:
//
//	ticks(one block) = per_block_bytes / bandwidth + base_latency
//
// with per_block_bytes = KVBytesPerToken(model, TP) × block_size_tokens.
//
// HOW THAT REACHES THE SIMULATOR. sim/kv.TieredKVCache charges
// `baseLatency + ceil(block_size_tokens / transferBandwidth)` per reloaded block, so its
// rate is in TOKENS per tick (the flag's "blocks per tick" help text has always been a
// misnomer — the divisor is the block's TOKEN count, not 1). Expressing the derivation in
// that unit makes block_size_tokens cancel exactly:
//
//	rate = bandwidth / KVBytesPerToken           [tokens/tick]
//	ceil(block_size_tokens / rate) = ceil(block_size_tokens × KVBytesPerToken / bandwidth)
//	                               = ceil(per_block_bytes / bandwidth)
//
// so the rate this file derives IS the formula above, charged per block by the existing
// library code. Nothing in sim/ changes.
//
// The `+ base_latency` term is NOT taken from the device. cpu_dram carries a physical
// base_latency of 1.0 µs, but --kv-transfer-base-latency is the *modelling* per-transfer
// cost, which the registry transcribed separately as `kv_transfer_base_latency` = 0 ticks,
// `method: not_charged` (coefficients/legacy-kv-transfer.yaml). The registry is explicit
// that the two are distinct quantities and must not be read as the same number, so the
// flag keeps its shipped 0 default and the device's own latency is not folded in here.
//
// INERT unless --kv-cpu-blocks > 0 (the pre-#1590 path, mutually exclusive with
// --kv-offload-config), so every run that does not enable the legacy tier is byte-identical
// (INV-6) and the catalog device table stays a LAZY dependency.
const (
	// legacyKVTransferDeviceClass is the catalog storage-device class that describes the
	// legacy single CPU tier's bus: host DRAM across the PCIe/NVLink boundary.
	legacyKVTransferDeviceClass = "cpu_dram"

	// legacyKVTransferResidual is the dimensionless efficiency residual R2G3b deferred to
	// this task ("the only bandwidth-side number that could live here is the dimensionless
	// efficiency RESIDUAL (achieved rate ÷ rated rate), and only once #1819's
	// derive-from-catalog conversion exists to define it against cpu_dram").
	//
	// It is anchored at ONE named reference deployment, because the retired default was
	// model-independent while a bus fact is not. The reference is the deployment the
	// legacy tiered example in docs/guide/kv-cache.md uses — qwen/qwen3-14b, H100, TP=1,
	// block_size 16:
	//
	//	retired flag default            100.0            tokens/tick
	//	cpu_dram rated read bandwidth   2.0e4            bytes/µs   (devices/storage.yaml)
	//	reference KVBytesPerToken       163840           bytes/token
	//	                                (40 layers × 2 (K+V) × 128 head_dim × 8 kv_heads
	//	                                 × 2 bytes bf16, ÷ TP=1)
	//	nominal rate at the reference   2.0e4 / 163840 = 0.1220703125 tokens/tick
	//	residual                        100.0 / 0.1220703125 = 819.2
	//
	// READ IT HONESTLY: a residual of 819.2 is not an efficiency ≤ 1. It records that the
	// retired default asserted ~819× cpu_dram's rated bandwidth — which is exactly why
	// R2G3b would not transcribe it as physics. It is preserved rather than corrected
	// because R2 is value-preserving: the conversion must reproduce today's cost, and a
	// physics correction is a separate, arguable change. Operators who want a faithful
	// CPU-offload cost should use --kv-offload-config, whose tiers price the catalog
	// device directly with no residual.
	//
	// Away from the reference the derived rate scales as 1/KVBytesPerToken — the physically
	// meaningful behaviour the retired constant could not express (a model with a quarter
	// the KV per token moves four times the tokens per tick over the same bus).
	legacyKVTransferResidual = 819.2
)

// deriveLegacyKVTransferRate converts a catalog storage-device bandwidth into the rate
// sim/kv.TieredKVCache consumes (tokens per tick), applying the R2G3b residual.
//
// Pure: both inputs are arguments, so the conversion law is table-testable without a
// catalog, an environment variable or a cobra command.
func deriveLegacyKVTransferRate(dev kvOffloadDevice, perTokenKVBytes float64) (float64, error) {
	if perTokenKVBytes <= 0 || math.IsNaN(perTokenKVBytes) || math.IsInf(perTokenKVBytes, 0) {
		return 0, fmt.Errorf("per-token KV bytes must be finite and > 0, got %v", perTokenKVBytes)
	}
	if dev.ReadBandwidth <= 0 || math.IsNaN(dev.ReadBandwidth) || math.IsInf(dev.ReadBandwidth, 0) {
		return 0, fmt.Errorf("catalog device %q has a non-positive or non-finite read_bandwidth (%v); "+
			"the legacy CPU↔GPU transfer cost cannot be derived from it",
			legacyKVTransferDeviceClass, dev.ReadBandwidth)
	}
	// Order matters for exactness at the reference: (bandwidth / perTokenKVBytes) is
	// exactly representable there (2.0e4 / 163840 == 125/1024), so multiplying by the
	// residual last lands on exactly 100.0 and the conversion is bit-for-bit
	// value-preserving rather than merely close (INV-6).
	rate := dev.ReadBandwidth / perTokenKVBytes * legacyKVTransferResidual
	if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
		return 0, fmt.Errorf("derived CPU↔GPU transfer rate must be finite and > 0, got %v "+
			"(read_bandwidth=%v, per-token KV bytes=%v, residual=%v)",
			rate, dev.ReadBandwidth, perTokenKVBytes, legacyKVTransferResidual)
	}
	return rate, nil
}

// loadLegacyKVTransferDevice reads the cpu_dram entry out of the catalog's storage-device
// table for the legacy transfer path.
//
// It deliberately does NOT reuse loadCatalogStorageDevices: every diagnostic that function
// emits opens with "a secondary tier names a device_class", which is the --kv-offload-config
// path and would misdirect an operator who only set --kv-cpu-blocks. The parse and the path
// law are shared (parseCatalogStorageDevices / catalogStorageDevicesPath), so the two
// readers cannot disagree about the file's location or shape — only about who to blame.
//
// Every failure names the path and the escape hatch (R1).
func loadLegacyKVTransferDevice(catalog string) (kvOffloadDevice, error) {
	path := catalogStorageDevicesPath(catalog)
	data, err := os.ReadFile(path)
	if err != nil {
		return kvOffloadDevice{}, fmt.Errorf(
			"--kv-cpu-blocks > 0 prices CPU↔GPU KV transfers from the catalog storage-device "+
				"table, but %s is not readable: %w.\n"+
				"  Add it to the catalog (--catalog / %s names the catalog CLONE ROOT; the table "+
				"lives at its %s), point --catalog / %s at a catalog that has it, or set "+
				"--kv-transfer-bandwidth explicitly to override the derivation",
			path, err, catalogEnvVar, catalogStorageDevicesRelPath, catalogEnvVar)
	}
	devices, parseErr := parseCatalogStorageDevices(data)
	if parseErr != nil {
		return kvOffloadDevice{}, fmt.Errorf(
			"--kv-cpu-blocks > 0: catalog storage-device table %s is malformed: %w", path, parseErr)
	}
	dev, ok := devices[legacyKVTransferDeviceClass]
	if !ok {
		return kvOffloadDevice{}, fmt.Errorf(
			"--kv-cpu-blocks > 0 prices CPU↔GPU KV transfers from the %q device class, which the "+
				"catalog storage-device table %s does not define (known: %s).\n"+
				"  Fix that catalog file, or set --kv-transfer-bandwidth explicitly to override "+
				"the derivation",
			legacyKVTransferDeviceClass, path, knownDeviceClasses(devices))
	}
	return dev, nil
}

// resolveLegacyKVTransferBandwidth decides the --kv-transfer-bandwidth value a run actually
// uses. It is the ONE place the legacy transfer rate is decided, called from both runCmd and
// replayCmd with the same inputs, so the two commands cannot drift (R23, INV-13). `observe`
// resolves no model config and registers neither flag, so it never reaches here.
//
// Three cases, in order:
//  1. --kv-cpu-blocks == 0 (the default): the legacy tier is disabled and the rate is
//     unused. Returned unchanged — no catalog read, no model arithmetic (INV-6).
//  2. --kv-transfer-bandwidth supplied: the operator's override wins verbatim, exactly as
//     before this change.
//  3. otherwise: DERIVED from the catalog cpu_dram device (see the file comment).
//
// CLI boundary, so failures are logrus.Fatalf (R1) rather than a returned error.
func resolveLegacyKVTransferBandwidth(cmd *cobra.Command, mc sim.ModelConfig, tp int) float64 {
	if kvCPUBlocks <= 0 {
		return kvTransferBandwidth
	}
	if cmd.Flags().Changed("kv-transfer-bandwidth") {
		logrus.Infof("--kv-transfer-bandwidth=%v overrides the value derived from the catalog %q device",
			kvTransferBandwidth, legacyKVTransferDeviceClass)
		return kvTransferBandwidth
	}
	catalog, err := resolveCatalogRoot()
	if err != nil {
		logrus.Fatalf("--kv-cpu-blocks > 0 prices CPU↔GPU KV transfers from the catalog's %s: %v",
			catalogStorageDevicesRelPath, err)
	}
	dev, devErr := loadLegacyKVTransferDevice(catalog)
	if devErr != nil {
		logrus.Fatalf("%v", devErr)
	}
	perTokenKVBytes, ptErr := latency.KVBytesPerToken(mc, tp)
	if ptErr != nil {
		logrus.Fatalf("--kv-cpu-blocks > 0: cannot derive the CPU↔GPU transfer cost from the model: %v", ptErr)
	}
	rate, rateErr := deriveLegacyKVTransferRate(dev, perTokenKVBytes)
	if rateErr != nil {
		logrus.Fatalf("--kv-cpu-blocks > 0: %v", rateErr)
	}
	logrus.Infof("Derived CPU↔GPU KV transfer rate %g tokens/tick from catalog device %q "+
		"(read_bandwidth=%g bytes/µs, KVBytesPerToken=%g, residual=%g)",
		rate, legacyKVTransferDeviceClass, dev.ReadBandwidth, perTokenKVBytes, legacyKVTransferResidual)
	return rate
}

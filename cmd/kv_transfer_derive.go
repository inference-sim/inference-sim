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
// THE `+ base_latency` TERM IS NOT TAKEN FROM THE DEVICE — a TRACKED deferral
// (inference-sim#1841), not a silent implementation choice.
//
// #1819's change list asks for both `bandwidth` and `base_latency` to be read from cpu_dram.
// Its acceptance criteria ask for something that cannot hold at the same time for the
// latency term: "the derived cost ... reproduces the pre-change cost for the same inputs",
// under a governing invariant the issue states outright — "R2: value-preserving,
// byte-identical stdout on both backends".
//
// They conflict because TieredKVCache charges `baseLatency + ceil(block_tokens / rate)` per
// reloaded block. The retired configuration charged baseLatency = 0 (the shipped
// --kv-transfer-base-latency default). cpu_dram's base_latency is 1.0 µs and 1 tick = 1 µs
// (docs/reference/configuration.md), so folding it in makes the reference block cost 2 ticks
// where it was 1 — and no choice of residual can absorb that, because the residual is
// multiplicative on the second term while the first is additive, and `ceil` of a positive
// quantity is never 0. Charging the device latency is therefore a COST CHANGE, which is
// exactly what the issue's own governing invariant forbids.
//
// So this file derives the bandwidth side only, and the physics question — SHOULD the legacy
// tier charge cpu_dram's 1.0 µs per transfer? — is deferred to a maintainer in
// inference-sim#1841 rather than decided here. The authority for the deferral is #1819's own
// acceptance criteria. blis-registry's separate `kv_transfer_base_latency` = 0 ticks,
// `method: not_charged` entry (coefficients/legacy-kv-transfer.yaml) AGREES with the outcome,
// but it is another repository's decision and cannot refine an issue here, so it is corroboration
// and not the reason.
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
	//
	// AUTHORING IT INTO THE REGISTRY is the other half R2G3b asked for, and it is a
	// cross-repository change this PR cannot make: blis-registry#17 tracks it, with the
	// derivation table above and the "not an efficiency ≤ 1" warning, so the number is not
	// left here with no filed owner (#1840 review F2).
	legacyKVTransferResidual = 819.2

	// maxLegacyKVTransferTicksPerBlock bounds the per-block tick charge the derived rate may
	// produce, because sim/kv.TieredKVCache converts it with
	// `int64(math.Ceil(block_tokens / rate))` (sim/kv/tiered.go) and Go's float→int64
	// conversion is UNDEFINED when the value does not fit: on amd64 it yields MinInt64, so a
	// catalog read_bandwidth of, say, 1e-300 — positive, finite, and accepted by every check
	// below — would inject a large NEGATIVE pending latency instead of a large positive one
	// (#1840 review F3).
	//
	// 2^52 rather than MaxInt64: beyond 2^53 a float64 cannot represent consecutive integers,
	// so `math.Ceil` stops being meaningful there anyway, and one bit of headroom below that
	// leaves room for the accumulation across reloaded blocks that pendingLatency performs.
	// As a duration it is ~4.5e15 ticks ≈ 143,000 years, so no plausible deployment is
	// refused by it — it only catches a catalog fact that is not one.
	maxLegacyKVTransferTicksPerBlock = 1 << 52
)

// deriveLegacyKVTransferRate converts a catalog storage-device bandwidth into the rate
// sim/kv.TieredKVCache consumes (tokens per tick), applying the R2G3b residual.
//
// blockSizeTokens is taken as an argument only to bound the result: the rate itself does not
// depend on it (that is the whole point of expressing the rate in tokens/tick), but the tick
// charge the library derives FROM the rate does, and that charge must fit in an int64 — see
// maxLegacyKVTransferTicksPerBlock.
//
// Pure: every input is an argument, so the conversion law is table-testable without a
// catalog, an environment variable or a cobra command.
func deriveLegacyKVTransferRate(dev kvOffloadDevice, perTokenKVBytes float64, blockSizeTokens int64) (float64, error) {
	if perTokenKVBytes <= 0 || math.IsNaN(perTokenKVBytes) || math.IsInf(perTokenKVBytes, 0) {
		return 0, fmt.Errorf("per-token KV bytes must be finite and > 0, got %v", perTokenKVBytes)
	}
	if blockSizeTokens <= 0 {
		return 0, fmt.Errorf("block size in tokens must be > 0, got %d", blockSizeTokens)
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
	// "Positive and finite" is not enough: a rate small enough makes the library's
	// ceil(block_tokens / rate) exceed int64, and Go's conversion then wraps to MinInt64
	// rather than saturating. Refuse here, where the catalog device and the model are still
	// nameable, instead of shipping a negative transfer latency into the DES.
	if ticksPerBlock := math.Ceil(float64(blockSizeTokens) / rate); ticksPerBlock > maxLegacyKVTransferTicksPerBlock {
		return 0, fmt.Errorf("derived CPU↔GPU transfer rate %v tokens/tick charges %v ticks for one "+
			"%d-token block, which does not fit the simulator's int64 tick budget (max %d): "+
			"catalog device %q read_bandwidth=%v is implausibly small for per-token KV bytes=%v "+
			"(residual=%v).\n"+
			"  Fix that catalog value, or set --kv-transfer-bandwidth explicitly to override the "+
			"derivation",
			rate, ticksPerBlock, blockSizeTokens, int64(maxLegacyKVTransferTicksPerBlock),
			legacyKVTransferDeviceClass, dev.ReadBandwidth, perTokenKVBytes, legacyKVTransferResidual)
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
			"--kv-cpu-blocks > 0: catalog storage-device table %s is malformed: %w.\n"+
				"  Fix that catalog file, or set --kv-transfer-bandwidth explicitly to override "+
				"the derivation", path, parseErr)
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
//     before this change. Derivation is selected by OMITTING the flag, not by its zero
//     registered default — a supplied 0 never reaches here, because resolvePolicies
//     range-checks supplied values and refuses it (a typo must be loud, not silently read as
//     "derive").
//  3. otherwise: DERIVED from the catalog cpu_dram device (see the file comment).
//
// Both callers reach here after resolveLatencyConfig, which refuses --block-size-in-tokens <= 0,
// so the block size the int64-budget bound is checked against is the one the run will use.
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
	rate, rateErr := deriveLegacyKVTransferRate(dev, perTokenKVBytes, blockSizeTokens)
	if rateErr != nil {
		logrus.Fatalf("--kv-cpu-blocks > 0: %v", rateErr)
	}
	logrus.Infof("Derived CPU↔GPU KV transfer rate %g tokens/tick from catalog device %q "+
		"(read_bandwidth=%g bytes/µs, KVBytesPerToken=%g, residual=%g)",
		rate, legacyKVTransferDeviceClass, dev.ReadBandwidth, perTokenKVBytes, legacyKVTransferResidual)
	return rate
}

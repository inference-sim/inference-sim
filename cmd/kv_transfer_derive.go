package cmd

import (
	"fmt"
	"math"

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
// `baseLatency + ceil(block_size_tokens / transferBandwidth)` per reloaded block. The rate
// is therefore in TOKENS per tick (the flag's old "blocks per tick" help text was a
// misnomer — the divisor is the block's TOKEN count, not 1). Expressing the bandwidth
// derivation in that unit makes block_size_tokens cancel exactly:
//
//	rate = bandwidth / KVBytesPerToken           [tokens/tick]
//	ceil(block_size_tokens / rate) = ceil(block_size_tokens × KVBytesPerToken / bandwidth)
//	                               = ceil(per_block_bytes / bandwidth)
//
// The effective base latency is independently selected: an explicit
// --kv-transfer-base-latency wins (including 0), otherwise cpu_dram.base_latency is rounded
// up to whole simulator ticks. The #1819 design refinement made this additive catalog
// latency an intentional cost change for the explicitly enabled legacy path (#1841). R2's
// byte-identity promise still covers the committed matrix, where this path is disabled.
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
	// to preserve the historical bandwidth term. The separately derived catalog base
	// latency is the intentional enabled-path cost change specified by #1819/#1841.
	// Operators who want a faithful CPU-offload cost should use --kv-offload-config, whose
	// tiers price the catalog device directly with no residual.
	//
	// Away from the reference the derived rate scales as 1/KVBytesPerToken — the physically
	// meaningful behaviour the retired constant could not express (a model with a quarter
	// the KV per token moves four times the tokens per tick over the same bus).
	//
	// AUTHORED IN THE REGISTRY — the other half R2G3b asked for. This value now lives at
	// blis-registry coefficients/legacy-kv-transfer.yaml -> kv_transfer_bandwidth_residual
	// (authored in blis-registry#18), method: assumed, units: dimensionless, scope
	// {hardware: [cpu_dram], model: [qwen3-14b], tp: [1]} — cpu_dram is the device whose
	// bandwidth this residual corrects (not the anchor's H100, which the value does not
	// depend on), and qwen3-14b is the bare catalog model identity. It carries the derivation
	// table above and the "not an efficiency ≤ 1" warning, so this constant now mirrors a
	// filed registry entry rather than a number with no owner. blis-registry#17 tracked the
	// authoring (#1840 review F2).
	legacyKVTransferResidual = 819.2

	// maxLegacyKVTransferTicksPerBlock bounds the bandwidth portion of the per-block tick
	// charge handed to sim/kv, because sim/kv.TieredKVCache converts it with
	// `int64(math.Ceil(block_tokens / rate))` (sim/kv/tiered.go) and Go's float→int64
	// conversion is IMPLEMENTATION-DEFINED when the value does not fit: amd64 wraps to MinInt64,
	// arm64 saturates to MaxInt64. So a read_bandwidth of, say, 1e-300 — positive, finite, and
	// accepted by every check below — would inject a garbage pending latency (a large NEGATIVE
	// one on amd64, a ~292,000-year one on arm64) instead of the true charge (#1840 review F3).
	//
	// The bound is a property of the RATE, not of where the rate came from, so it is enforced
	// on BOTH bandwidth paths through resolveLegacyKVTransferCost — the catalog-derived rate and a
	// supplied --kv-transfer-bandwidth override. A positive finite override like 1e-300 clears
	// resolvePolicies' range check and then overflows exactly the same way (#1840 review N1);
	// see legacyKVTransferTickBudgetError, which both paths call.
	//
	// 2^52 rather than MaxInt64: beyond 2^53 a float64 cannot represent consecutive integers,
	// so `math.Ceil` stops being meaningful there anyway. This is an input-sanity bound, not
	// an overflow proof: the combined base+transfer charge and cumulative pending latency are
	// checked independently at the shared TieredKVCache boundary. As a duration it is
	// ~4.5e15 ticks ≈ 143,000 years, so no plausible deployment is refused by it.
	maxLegacyKVTransferTicksPerBlock = 1 << 52
)

type legacyKVTransferCost struct {
	bandwidth   float64
	baseLatency int64
}

// legacyKVTransferTickBudgetError reports whether a tokens/tick transfer rate is safe for the
// int64 arithmetic sim/kv.TieredKVCache performs on it, returning nil when it is and the
// arithmetic that fails when it is not.
//
// It exists as its own function because the bound belongs to the RATE and not to its
// provenance: a derived rate and an operator-supplied --kv-transfer-bandwidth override reach
// the identical `int64(math.Ceil(block_tokens / rate))` conversion, so a check on only one of
// them is a hole rather than a partial defence (#1840 review N1). Both callers add their own
// "who to blame / what to change" context around it, which is the only thing that legitimately
// differs between the two paths.
//
// Pure: rate and block size are arguments, so the bound is table-testable independently of
// the catalog, the flags and cobra.
func legacyKVTransferTickBudgetError(rate float64, blockSizeTokens int64) error {
	ticksPerBlock := math.Ceil(float64(blockSizeTokens) / rate)
	if ticksPerBlock <= maxLegacyKVTransferTicksPerBlock {
		return nil
	}
	return fmt.Errorf("a CPU↔GPU transfer rate of %v tokens/tick charges %v ticks for one "+
		"%d-token block, which does not fit the simulator's int64 tick budget (max %d)",
		rate, ticksPerBlock, blockSizeTokens, int64(maxLegacyKVTransferTicksPerBlock))
}

// deriveLegacyKVTransferBaseLatency converts cpu_dram.base_latency from microseconds to the
// simulator's whole-microsecond ticks. Rounding up preserves a positive physical latency
// instead of silently erasing a fractional one. The strict upper bound avoids Go's
// implementation-dependent float64-to-int64 conversion when the rounded value reaches 2^63.
func deriveLegacyKVTransferBaseLatency(dev kvOffloadDevice) (int64, error) {
	if dev.BaseLatency < 0 || math.IsNaN(dev.BaseLatency) || math.IsInf(dev.BaseLatency, 0) {
		return 0, fmt.Errorf("catalog device %q has a negative or non-finite base_latency (%v); "+
			"the legacy CPU↔GPU transfer cost cannot be derived from it",
			legacyKVTransferDeviceClass, dev.BaseLatency)
	}
	ticks := math.Ceil(dev.BaseLatency)
	if ticks >= float64(math.MaxInt64) {
		return 0, fmt.Errorf("catalog device %q base_latency=%v rounds to %v ticks, which does not "+
			"fit the simulator's int64 tick budget", legacyKVTransferDeviceClass, dev.BaseLatency, ticks)
	}
	return int64(ticks), nil
}

// legacyKVTransferCombinedBudgetError catches a base-latency override that is individually
// representable but leaves no room for the positive bandwidth charge. TieredKVCache repeats
// this check at the shared library boundary and checks every cumulative addition; this CLI
// check exists to give flag/catalog users an actionable error before simulation starts.
func legacyKVTransferCombinedBudgetError(rate float64, blockSizeTokens, baseLatency int64) error {
	transferTicks := int64(math.Ceil(float64(blockSizeTokens) / rate))
	if baseLatency > math.MaxInt64-transferTicks {
		return fmt.Errorf("base latency %d plus the %d-tick bandwidth charge for one %d-token "+
			"block exceeds the simulator's int64 tick budget", baseLatency, transferTicks, blockSizeTokens)
	}
	return nil
}

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
	// residual last lands on exactly 100.0 and preserves the retired bandwidth term
	// bit-for-bit rather than merely approximately.
	rate := dev.ReadBandwidth / perTokenKVBytes * legacyKVTransferResidual
	if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
		return 0, fmt.Errorf("derived CPU↔GPU transfer rate must be finite and > 0, got %v "+
			"(read_bandwidth=%v, per-token KV bytes=%v, residual=%v)",
			rate, dev.ReadBandwidth, perTokenKVBytes, legacyKVTransferResidual)
	}
	// "Positive and finite" is not enough: a rate small enough makes the library's
	// ceil(block_tokens / rate) exceed int64, and Go's conversion of that out-of-range value is
	// implementation-defined (amd64 wraps to MinInt64, arm64 saturates to MaxInt64) — either way
	// a garbage charge. Refuse here, where the catalog device and the model are still nameable,
	// instead of shipping a bogus transfer latency into the DES.
	if budgetErr := legacyKVTransferTickBudgetError(rate, blockSizeTokens); budgetErr != nil {
		return 0, fmt.Errorf("the derivation produced %w: catalog device %q read_bandwidth=%v is "+
			"implausibly small for per-token KV bytes=%v (residual=%v).\n"+
			"  Fix that catalog value, or set --kv-transfer-bandwidth explicitly to override the "+
			"derivation",
			budgetErr, legacyKVTransferDeviceClass, dev.ReadBandwidth, perTokenKVBytes,
			legacyKVTransferResidual)
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
// Every failure names the path and the two overrides that together bypass it (R1).
func loadLegacyKVTransferDevice(catalog string) (kvOffloadDevice, error) {
	path := catalogStorageDevicesPath(catalog)
	devices, err := parseCatalogStorageDevices(path)
	if err != nil && isNotReadable(err) {
		return kvOffloadDevice{}, fmt.Errorf(
			"--kv-cpu-blocks > 0 prices CPU↔GPU KV transfers from the catalog storage-device "+
				"table, but %s is not readable: %w.\n"+
				"  Add it to the catalog (--catalog / %s names the catalog CLONE ROOT; the table "+
				"lives at its %s), point --catalog / %s at a catalog that has it, or supply "+
				"both --kv-transfer-bandwidth and --kv-transfer-base-latency to override it",
			path, err, catalogEnvVar, catalogStorageDevicesRelPath, catalogEnvVar)
	}
	if err != nil {
		return kvOffloadDevice{}, fmt.Errorf(
			"--kv-cpu-blocks > 0: catalog storage-device table %s is malformed: %w.\n"+
				"  Fix that catalog file, or supply both --kv-transfer-bandwidth and "+
				"--kv-transfer-base-latency to override it", path, err)
	}
	dev, ok := devices[legacyKVTransferDeviceClass]
	if !ok {
		return kvOffloadDevice{}, fmt.Errorf(
			"--kv-cpu-blocks > 0 prices CPU↔GPU KV transfers from the %q device class, which the "+
				"catalog storage-device table %s does not define (known: %s).\n"+
				"  Fix that catalog file, or supply both --kv-transfer-bandwidth and "+
				"--kv-transfer-base-latency to override it",
			legacyKVTransferDeviceClass, path, knownDeviceClasses(devices))
	}
	return dev, nil
}

// resolveLegacyKVTransferCost decides the bandwidth and base-latency values a run actually
// uses. It is the ONE place both legacy transfer components are decided, called from runCmd
// and replayCmd with the same inputs so the commands cannot drift (R23, INV-13). `observe`
// resolves no model config and registers neither flag, so it never reaches here.
//
// Three cases, in order:
//  1. --kv-cpu-blocks == 0 (the default): both values are unused and returned unchanged —
//     no catalog read and no model arithmetic (INV-6).
//  2. Each supplied flag independently wins. An explicit base latency of 0 is a real
//     override, while a supplied bandwidth of 0 is refused by resolvePolicies.
//  3. Any omitted component is derived from the catalog cpu_dram device. The table is read
//     once if either component is omitted and not at all when both are supplied.
//
// Both callers reach here after resolveLatencyConfig, which refuses --block-size-in-tokens <= 0,
// so the block size the int64-budget bound is checked against is the one the run will use.
//
// CLI boundary, so failures are logrus.Fatalf (R1) rather than a returned error.
func resolveLegacyKVTransferCost(cmd *cobra.Command, mc sim.ModelConfig, tp int) legacyKVTransferCost {
	cost := legacyKVTransferCost{bandwidth: kvTransferBandwidth, baseLatency: kvTransferBaseLatency}
	if kvCPUBlocks <= 0 {
		return cost
	}
	bandwidthOverridden := cmd.Flags().Changed("kv-transfer-bandwidth")
	baseLatencyOverridden := cmd.Flags().Changed("kv-transfer-base-latency")

	var dev kvOffloadDevice
	if !bandwidthOverridden || !baseLatencyOverridden {
		catalog, err := resolveCatalogRoot()
		if err != nil {
			logrus.Fatalf("--kv-cpu-blocks > 0 prices CPU↔GPU KV transfers from the catalog's %s: %v",
				catalogStorageDevicesRelPath, err)
		}
		var devErr error
		dev, devErr = loadLegacyKVTransferDevice(catalog)
		if devErr != nil {
			logrus.Fatalf("%v", devErr)
		}
	}

	if bandwidthOverridden {
		// resolvePolicies has already refused a supplied value that is <= 0, NaN or Inf — but
		// "positive and finite" is exactly the condition maxLegacyKVTransferTicksPerBlock
		// exists because it is not sufficient. An override of 1e-300 clears that range check
		// and then lands in the same int64(ceil(block_tokens / rate)) conversion the derived
		// path is bounded against, whose out-of-range result is implementation-defined (amd64
		// wraps to MinInt64 and moves the clock backwards; arm64 saturates to MaxInt64)
		// (#1840 review N1). Refuse it here, blaming the flag rather than the catalog: the
		// catalog is not at fault on this path and may not even have been read.
		if budgetErr := legacyKVTransferTickBudgetError(cost.bandwidth, blockSizeTokens); budgetErr != nil {
			logrus.Fatalf("--kv-transfer-bandwidth=%v is too small to simulate: %v.\n"+
				"  Supply a larger rate, or omit --kv-transfer-bandwidth entirely to derive it from "+
				"the catalog %q device", cost.bandwidth, budgetErr, legacyKVTransferDeviceClass)
		}
		logrus.Infof("--kv-transfer-bandwidth=%v overrides the value derived from the catalog %q device",
			cost.bandwidth, legacyKVTransferDeviceClass)
	} else {
		perTokenKVBytes, ptErr := latency.KVBytesPerToken(mc, tp)
		if ptErr != nil {
			logrus.Fatalf("--kv-cpu-blocks > 0: cannot derive the CPU↔GPU transfer cost from the model: %v", ptErr)
		}
		rate, rateErr := deriveLegacyKVTransferRate(dev, perTokenKVBytes, blockSizeTokens)
		if rateErr != nil {
			logrus.Fatalf("--kv-cpu-blocks > 0: %v", rateErr)
		}
		cost.bandwidth = rate
		logrus.Infof("Derived CPU↔GPU KV transfer rate %g tokens/tick from catalog device %q "+
			"(read_bandwidth=%g bytes/µs, KVBytesPerToken=%g, residual=%g)",
			rate, legacyKVTransferDeviceClass, dev.ReadBandwidth, perTokenKVBytes, legacyKVTransferResidual)
	}

	if baseLatencyOverridden {
		logrus.Infof("--kv-transfer-base-latency=%d overrides the value derived from the catalog %q device",
			cost.baseLatency, legacyKVTransferDeviceClass)
	} else {
		baseLatency, latencyErr := deriveLegacyKVTransferBaseLatency(dev)
		if latencyErr != nil {
			logrus.Fatalf("--kv-cpu-blocks > 0: %v.\n  Fix that catalog value, or set "+
				"--kv-transfer-base-latency explicitly to override it", latencyErr)
		}
		cost.baseLatency = baseLatency
		logrus.Infof("Derived CPU↔GPU KV transfer base latency %d ticks from catalog device %q "+
			"(base_latency=%g µs, rounded up to whole ticks)",
			baseLatency, legacyKVTransferDeviceClass, dev.BaseLatency)
	}

	if budgetErr := legacyKVTransferCombinedBudgetError(cost.bandwidth, blockSizeTokens, cost.baseLatency); budgetErr != nil {
		logrus.Fatalf("legacy CPU↔GPU transfer cost is too large to simulate: %v.\n"+
			"  Supply a smaller --kv-transfer-base-latency or a larger --kv-transfer-bandwidth",
			budgetErr)
	}
	return cost
}

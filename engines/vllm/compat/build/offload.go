// Moved from inference-sim/cmd/kv_offload.go at ace1528fe4bba59e510eda48e80ee9b8142ea982.
// Moved from inference-sim/cmd/kv_offload_scenario.go at ace1528fe4bba59e510eda48e80ee9b8142ea982.

package build

import (
	"bytes"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-vllm/compat/engine"
)

// KVOffloadFile is the on-disk shape of a --kv-offload-config YAML file: a single
// top-level kv_offload: block. Strict-parsed (R10). Mirrors the --lora-config /
// --saturation-config precedent.
type KVOffloadFile struct {
	KVOffload *KVOffloadBlock `yaml:"kv_offload"`
}

// KVOffloadBlock is the user-facing kv_offload: block. All fields are pointers so an
// absent key is distinguishable from an explicit zero (R9): the resolver applies
// vLLM's default only when the pointer is nil.
//
// NOTE store_threshold: it is captured (as *int) but is NOT a live knob on the
// multi-tier path — vLLM's TieringOffloadingSpec rejects values >= 2 (it is
// single-tier-only). The resolver rejects >= 2 loudly (BC-G1) and treats nil/0/1 as a
// no-op (matching vLLM, which accepts 0/1 and ignores them on this path).
type KVOffloadBlock struct {
	CPUBytesToUse          *int64               `yaml:"cpu_bytes_to_use"`
	BlockSize              *int64               `yaml:"block_size"`
	BlocksPerChunk         *int64               `yaml:"blocks_per_chunk"`
	TokensPerHash          *int64               `yaml:"tokens_per_hash"`
	EvictionPolicy         *string              `yaml:"eviction_policy"`
	OffloadPromptOnly      *bool                `yaml:"offload_prompt_only"`
	SelfDescribingKVEvents *bool                `yaml:"self_describing_kv_events"`
	StoreThreshold         *int                 `yaml:"store_threshold"`
	SecondaryTiers         []KVOffloadTierBlock `yaml:"secondary_tiers"`
	// Simple selects the SimpleCPUOffloadConnector; LazyOffload is its lazy_offload.
	Simple      *bool `yaml:"simple_cpu_offload"`
	LazyOffload *bool `yaml:"lazy_offload"`
}

// KVOffloadTierBlock is one user-facing secondary tier. Pointer fields (R9).
type KVOffloadTierBlock struct {
	Type           *string  `yaml:"type"`
	RootDir        *string  `yaml:"root_dir"`
	NReadThreads   *int64   `yaml:"n_read_threads"`
	NWriteThreads  *int64   `yaml:"n_write_threads"`
	Locality       *string  `yaml:"locality"`
	EnableKVEvents *bool    `yaml:"enable_kv_events"`
	DirectIO       *bool    `yaml:"direct_io"`
	DeviceClass    *string  `yaml:"device_class"`
	ReadBandwidth  *float64 `yaml:"read_bandwidth"`
	WriteBandwidth *float64 `yaml:"write_bandwidth"`
	BaseLatency    *float64 `yaml:"base_latency"`
	// CapacityBytes is the tier's size on its device (BLIS-only: vLLM's fs tier writes until
	// the filesystem is full). Absent is unbounded.
	CapacityBytes *int64 `yaml:"capacity_bytes"`
}

// LoadKVOffloadConfigFile reads and strictly parses a --kv-offload-config file.
// Returns an error (never Fatalf — so it is unit-testable/fuzzable, Deviation #9);
// the CLI wrapper turns errors into logrus.Fatalf. Unknown keys error via
// KnownFields(true) (R10).
func LoadKVOffloadConfigFile(path string) (KVOffloadFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return KVOffloadFile{}, fmt.Errorf("read --kv-offload-config file %q: %w", path, err)
	}
	f, err := ParseKVOffloadBytes(data)
	if err != nil {
		return f, fmt.Errorf("parse --kv-offload-config file %q: %w", path, err)
	}
	return f, nil
}

// ParseKVOffloadBytes strictly decodes kv_offload YAML bytes (R10). Split out so it is
// directly fuzzable without touching disk.
func ParseKVOffloadBytes(data []byte) (KVOffloadFile, error) {
	var f KVOffloadFile
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&f); err != nil {
		return f, err
	}
	return f, nil
}

// ResolveKVOffload turns a parsed kv_offload block into a resolved, validated
// engine.KVOffloadConfig. It is PURE and error-returning (Deviation #9) so every reject
// branch is unit-testable and fuzzable. It applies vLLM's defaults knob-for-knob
// (BC-G2), resolves device_class against the catalog's storage-device table supplied by
// the caller — <catalog>/devices/storage.yaml, #1770 (explicit read/write/base override
// the class; a library-only path, since the CLI refuses them via
// refuseExplicitTierPhysics), enforces the block_size XOR blocks_per_chunk
// user-input rule then derives the canonical pair, rejects store_threshold>=2 and
// non-fs tiers loudly (BC-G1), and finally runs cfg.Validate() (BC-G3).
//
// gpuBlockSizeTokens is the GPU block size (the scenario's engine.block_size); vLLM's block_size
// default equals it, and it converts between the block_size and blocks_per_chunk
// encodings of the same quantity.
func ResolveKVOffload(block *KVOffloadBlock, devices map[string]KVOffloadDevice, gpuBlockSizeTokens int64) (engine.KVOffloadConfig, error) {
	if block == nil {
		return engine.KVOffloadConfig{}, fmt.Errorf("kv_offload: the --kv-offload-config file has no top-level kv_offload: block")
	}
	if gpuBlockSizeTokens <= 0 {
		return engine.KVOffloadConfig{}, fmt.Errorf("kv_offload: GPU block size (the scenario's engine.block_size) must be > 0, got %d", gpuBlockSizeTokens)
	}
	cfg := engine.KVOffloadConfig{Enabled: true}

	// cpu_bytes_to_use: required when the block is present (vLLM marks it required).
	if block.CPUBytesToUse == nil {
		return cfg, fmt.Errorf("kv_offload: cpu_bytes_to_use is required")
	}
	cfg.CPUBytesToUse = *block.CPUBytesToUse

	// store_threshold: reject < 0 (invalid) and >= 2 (vLLM TieringOffloadingSpec
	// rejects it); 0/1 are accepted as a no-op on the multi-tier path (vLLM parity).
	if block.StoreThreshold != nil {
		if *block.StoreThreshold < 0 {
			return cfg, fmt.Errorf("kv_offload: store_threshold must be >= 0, got %d", *block.StoreThreshold)
		}
		if *block.StoreThreshold >= 2 {
			return cfg, fmt.Errorf("kv_offload: store_threshold=%d: store_threshold is not supported for TieringOffloadingSpec (values >= 2 are rejected; it is single-tier-only and fixed at 1 on the multi-tier offload path)", *block.StoreThreshold)
		}
	}

	// block_size XOR blocks_per_chunk (mutually exclusive alternate encodings of the
	// same chunk-granularity quantity). Derive one canonical pair.
	if block.BlockSize != nil && block.BlocksPerChunk != nil {
		return cfg, fmt.Errorf("kv_offload: block_size and blocks_per_chunk are mutually exclusive (they encode the same quantity: block_size = blocks_per_chunk × gpu_block_size); set at most one")
	}
	switch {
	case block.BlockSize != nil:
		bs := *block.BlockSize
		if bs <= 0 || bs%gpuBlockSizeTokens != 0 {
			return cfg, fmt.Errorf("kv_offload: block_size (%d) must be a positive multiple of the GPU block size (%d)", bs, gpuBlockSizeTokens)
		}
		cfg.BlockSize = bs
		cfg.BlocksPerChunk = bs / gpuBlockSizeTokens
	case block.BlocksPerChunk != nil:
		bpc := *block.BlocksPerChunk
		if bpc <= 0 {
			return cfg, fmt.Errorf("kv_offload: blocks_per_chunk must be > 0, got %d", bpc)
		}
		cfg.BlocksPerChunk = bpc
		cfg.BlockSize = bpc * gpuBlockSizeTokens
		if cfg.BlockSize/gpuBlockSizeTokens != bpc { // int64 overflow guard
			return cfg, fmt.Errorf("kv_offload: blocks_per_chunk (%d) is too large — block_size = blocks_per_chunk × gpu_block_size (%d) would overflow int64", bpc, gpuBlockSizeTokens)
		}
	default:
		cfg.BlocksPerChunk = 1             // vLLM default
		cfg.BlockSize = gpuBlockSizeTokens // vLLM default = GPU block size
	}

	// tokens_per_hash: vLLM has no default (required); BLIS defaults to the GPU block
	// size (natural one-hash-per-block stride).
	if block.TokensPerHash != nil {
		if *block.TokensPerHash <= 0 {
			return cfg, fmt.Errorf("kv_offload: tokens_per_hash must be > 0, got %d", *block.TokensPerHash)
		}
		cfg.TokensPerHash = *block.TokensPerHash
	} else {
		cfg.TokensPerHash = gpuBlockSizeTokens
	}

	// eviction_policy: vLLM default "lru".
	cfg.EvictionPolicy = "lru"
	if block.EvictionPolicy != nil {
		cfg.EvictionPolicy = *block.EvictionPolicy
	}
	// The offload cache simulates LRU only; vLLM's other in-tree policy (arc) is refused here,
	// at config time, rather than reaching the cache's constructor as a panic.
	if cfg.EvictionPolicy != "lru" {
		return cfg, fmt.Errorf("kv_offload: eviction_policy %q is not simulated; the offload cache "+
			"is LRU (vLLM's arc is not yet modelled)", cfg.EvictionPolicy)
	}

	// offload_prompt_only: vLLM DEFAULT TRUE (trap 1).
	cfg.OffloadPromptOnly = true
	if block.OffloadPromptOnly != nil {
		cfg.OffloadPromptOnly = *block.OffloadPromptOnly
	}

	// self_describing_kv_events: vLLM default false.
	if block.SelfDescribingKVEvents != nil {
		cfg.SelfDescribingKVEvents = *block.SelfDescribingKVEvents
	}

	// secondary_tiers: vLLM default empty.
	for i, tb := range block.SecondaryTiers {
		tier, err := resolveKVOffloadTier(i, tb, devices)
		if err != nil {
			return cfg, err
		}
		cfg.Tiers = append(cfg.Tiers, tier)
	}

	if block.Simple != nil {
		cfg.Simple = *block.Simple
	}
	if block.LazyOffload != nil {
		cfg.LazyOffload = *block.LazyOffload
	}
	if err := cfg.Validate(); err != nil {
		return cfg, err
	}
	return cfg, nil
}

// resolveKVOffloadTier resolves one secondary tier. index is used in error messages.
func resolveKVOffloadTier(index int, tb KVOffloadTierBlock, devices map[string]KVOffloadDevice) (engine.KVOffloadTier, error) {
	var tier engine.KVOffloadTier

	// type: required; only "fs" is representable (reject obj/p2p/example loudly, BC-G1).
	if tb.Type == nil {
		return tier, fmt.Errorf("kv_offload: secondary_tiers[%d].type is required", index)
	}
	tier.Type = *tb.Type
	if tier.Type != "fs" {
		return tier, fmt.Errorf("kv_offload: secondary_tiers[%d].type=%q is not supported in BLIS yet (only \"fs\" is representable; obj/p2p/example tiers have no faithful config mapping — see #1587/#1585)", index, tier.Type)
	}

	// root_dir: required for fs.
	if tb.RootDir == nil || *tb.RootDir == "" {
		return tier, fmt.Errorf("kv_offload: secondary_tiers[%d].root_dir is required for an \"fs\" tier", index)
	}
	tier.RootDir = *tb.RootDir

	// n_read_threads / n_write_threads: vLLM default 16 each.
	tier.NReadThreads = 16
	if tb.NReadThreads != nil {
		tier.NReadThreads = *tb.NReadThreads
	}
	tier.NWriteThreads = 16
	if tb.NWriteThreads != nil {
		tier.NWriteThreads = *tb.NWriteThreads
	}

	if tb.CapacityBytes != nil {
		if *tb.CapacityBytes < 0 {
			return tier, fmt.Errorf("kv_offload: secondary_tiers[%d].capacity_bytes must be >= 0, got %d", index, *tb.CapacityBytes)
		}
		tier.CapacityBytes = *tb.CapacityBytes
	}

	// locality: default unset ("").
	if tb.Locality != nil {
		tier.Locality = *tb.Locality
	}

	// enable_kv_events: vLLM default false.
	if tb.EnableKVEvents != nil {
		tier.EnableKVEvents = *tb.EnableKVEvents
	}

	// direct_io: REQUIRED — BLIS makes vLLM's runtime O_DIRECT probe an explicit
	// config axis (a simulator cannot probe the operator's disk). No silent default. A run
	// refuses direct_io: false (priceOffload): the kernel prices only O_DIRECT I/O.
	if tb.DirectIO == nil {
		return tier, fmt.Errorf("kv_offload: secondary_tiers[%d].direct_io must be set explicitly (BLIS makes vLLM's runtime O_DIRECT probe an explicit config axis rather than guess the operator's disk)", index)
	}
	tier.DirectIO = *tb.DirectIO

	// bandwidth/latency + device model: a device_class resolves the numbers for the
	// regime selected by direct_io (#1581 BC-D3); explicit per-tier bandwidth/latency
	// fields override the resolved base triple (the ramp/jitter come only from the
	// device class, DL-6). Defaults are the "ramp/jitter off" values so a device that
	// declares none is byte-identical to pre-#1581 (BC-D2/INV-6).
	tier.SaturationQueueDepth = 1
	tier.SingleTransferFraction = 1.0
	tier.LatencyJitterStddev = 0
	hasClass := tb.DeviceClass != nil
	if hasClass {
		dev, ok := devices[*tb.DeviceClass]
		if !ok {
			return tier, fmt.Errorf("kv_offload: secondary_tiers[%d].device_class=%q is not defined in the catalog storage-device table %s (known: %s)", index, *tb.DeviceClass, CatalogStorageDevicesRelPath, KnownDeviceClasses(devices))
		}
		tier.DeviceClass = *tb.DeviceClass
		rb, wb, base, qsat, f1, sigma := resolveDeviceRegime(dev, tier.DirectIO)
		tier.ReadBandwidth, tier.WriteBandwidth, tier.BaseLatency = rb, wb, base
		tier.SaturationQueueDepth, tier.SingleTransferFraction, tier.LatencyJitterStddev = qsat, f1, sigma
	}
	if tb.ReadBandwidth != nil {
		tier.ReadBandwidth = *tb.ReadBandwidth
	}
	if tb.WriteBandwidth != nil {
		tier.WriteBandwidth = *tb.WriteBandwidth
	}
	if tb.BaseLatency != nil {
		tier.BaseLatency = *tb.BaseLatency
	}
	// Every tier needs a resolvable physics triple: either a device_class or the full
	// explicit read_bandwidth + write_bandwidth + base_latency (per-(tier,direction)
	// bandwidth is required — never a single number, #1588 BC-S3).
	if !hasClass && (tb.ReadBandwidth == nil || tb.WriteBandwidth == nil || tb.BaseLatency == nil) {
		return tier, fmt.Errorf("kv_offload: secondary_tiers[%d] needs a device_class OR an explicit read_bandwidth + write_bandwidth + base_latency triple", index)
	}
	return tier, nil
}

// resolveDeviceRegime picks the physics numbers for one device_class under the
// selected I/O regime (#1581 BC-D3). It starts from the O_DIRECT set (with ramp/
// jitter defaulting to off — Qsat=1, f₁=1.0, σ=0 — when the device omits them). For
// the buffered regime (directIO=false) each buffered_* field overrides its O_DIRECT
// counterpart when present, and otherwise falls back to the O_DIRECT value. To
// activate a real ramp a device must set BOTH saturation_queue_depth ≥ 2 AND
// single_transfer_fraction < 1 (Qsat alone with f₁=1.0 is a no-op).
func resolveDeviceRegime(dev KVOffloadDevice, directIO bool) (readBW, writeBW, base float64, qsat int64, f1, sigma float64) {
	readBW, writeBW, base = dev.ReadBandwidth, dev.WriteBandwidth, dev.BaseLatency
	qsat, f1, sigma = 1, 1.0, 0
	if dev.SaturationQueueDepth != nil {
		qsat = *dev.SaturationQueueDepth
	}
	if dev.SingleTransferFraction != nil {
		f1 = *dev.SingleTransferFraction
	}
	if dev.LatencyJitterStddev != nil {
		sigma = *dev.LatencyJitterStddev
	}
	if directIO {
		return
	}
	// Buffered regime: override per field, else keep the O_DIRECT value.
	if dev.BufferedReadBandwidth != nil {
		readBW = *dev.BufferedReadBandwidth
	}
	if dev.BufferedWriteBandwidth != nil {
		writeBW = *dev.BufferedWriteBandwidth
	}
	if dev.BufferedBaseLatency != nil {
		base = *dev.BufferedBaseLatency
	}
	if dev.BufferedSaturationQueueDepth != nil {
		qsat = *dev.BufferedSaturationQueueDepth
	}
	if dev.BufferedSingleTransferFraction != nil {
		f1 = *dev.BufferedSingleTransferFraction
	}
	if dev.BufferedLatencyJitterStddev != nil {
		sigma = *dev.BufferedLatencyJitterStddev
	}
	return
}

// KnownDeviceClasses returns the sorted device_class names for a deterministic error
// message (INV-6: no map-iteration order in output).
func KnownDeviceClasses(devices map[string]KVOffloadDevice) string {
	if len(devices) == 0 {
		return "<none configured>"
	}
	names := make([]string, 0, len(devices))
	for n := range devices {
		names = append(names, n)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

// ResolveKVOffloadConfig resolves the run's KV offload from its two sources. The scenario's
// Deployment.offload block (scenarioOffload, nil when it states none) is the deployment's tier
// configuration (#1910); a --kv-offload-config file (configPath, "" when not given) may state
// the same hierarchy in vLLM's kv_offload terms, and must then resolve identically -- there is
// no precedence between two descriptions of one deployment, so a disagreement is refused,
// naming the fields. With neither, the inert zero value. scenario names the scenario in errors;
// catalog locates the catalog root, asked only when a tier names a device_class. Single
// construction site (R4), shared by run and replay for INV-13 parity.
// It returns errors; the caller (inference-sim's CLI) turns them into its fatal exit.
func ResolveKVOffloadConfig(scenario string, scenarioOffload *deployment.Offload, configPath string,
	blockSizeTokens int64, catalog CatalogLocator) (engine.KVOffloadConfig, error) {
	var fromScenario, fromFlag engine.KVOffloadConfig
	if scenarioOffload != nil {
		block, err := KVOffloadBlockFromDeployment(scenarioOffload)
		var devices map[string]KVOffloadDevice
		if err == nil {
			devices, err = ResolveKVOffloadDevices(block, catalog)
		}
		if err == nil {
			fromScenario, err = ResolveKVOffload(block, devices, blockSizeTokens)
		}
		if err != nil {
			return engine.KVOffloadConfig{}, fmt.Errorf("scenario %q: %w", scenario, err)
		}
	}
	if configPath != "" {
		var err error
		fromFlag, err = ResolveKVOffloadFile(configPath, blockSizeTokens, catalog)
		if err != nil {
			return engine.KVOffloadConfig{}, err
		}
	}
	switch {
	case scenarioOffload != nil && configPath != "":
		// blis-schemas' Offload cannot state the chunk width (proposed field:
		// offload.blocks_per_chunk), so the scenario leaves it unstated and the flag's applies.
		fromScenario.BlockSize, fromScenario.BlocksPerChunk = fromFlag.BlockSize, fromFlag.BlocksPerChunk
		if diff := OffloadConfigDiff(fromScenario, fromFlag); diff != "" {
			return engine.KVOffloadConfig{}, fmt.Errorf("--kv-offload-config %q disagrees with scenario %q's offload block (%s); the "+
				"scenario is the deployment, so the flag may only restate it -- drop the flag, or "+
				"change the scenario", configPath, scenario, diff)
		}
		return fromScenario, nil
	case scenarioOffload != nil:
		return fromScenario, nil
	default:
		return fromFlag, nil
	}
}

// ResolveKVOffloadFile loads and resolves one --kv-offload-config file.
func ResolveKVOffloadFile(path string, blockSizeTokens int64, catalog CatalogLocator) (engine.KVOffloadConfig, error) {
	f, err := LoadKVOffloadConfigFile(path)
	if err != nil {
		return engine.KVOffloadConfig{}, err
	}
	if err := RefuseExplicitTierPhysics(f.KVOffload); err != nil {
		return engine.KVOffloadConfig{}, err
	}
	// #1770: device_class physics come from the CATALOG (<catalog>/devices/storage.yaml),
	// the single source of truth, not from defaults.yaml. Read LAZILY — a config whose
	// tiers all carry explicit bandwidth/latency triples never touches the catalog
	// devices/ namespace, so it resolves exactly as it did before the cutover.
	devices, err := ResolveKVOffloadDevices(f.KVOffload, catalog)
	if err != nil {
		return engine.KVOffloadConfig{}, fmt.Errorf("--kv-offload-config %q: %w", path, err)
	}
	cfg, err := ResolveKVOffload(f.KVOffload, devices, blockSizeTokens)
	if err != nil {
		return engine.KVOffloadConfig{}, fmt.Errorf("--kv-offload-config %q: %w", path, err)
	}
	return cfg, nil
}

// RefuseExplicitTierPhysics refuses a kv_offload tier that states its own bandwidth or
// latency: the kernel prices the tier from its catalog device, and a second
// source for the same physics would need precedence rules to reconcile.
func RefuseExplicitTierPhysics(block *KVOffloadBlock) error {
	if block == nil {
		return nil
	}
	for i, t := range block.SecondaryTiers {
		if t.ReadBandwidth != nil || t.WriteBandwidth != nil || t.BaseLatency != nil {
			return fmt.Errorf("kv_offload secondary_tiers[%d] states read_bandwidth, "+
				"write_bandwidth or base_latency, but the kernel prices a tier from its catalog "+
				"device_class; name the device and drop the explicit physics",
				i)
		}
	}
	return nil
}

// The scenario's offload block (#1910). A blis-schemas Deployment states its KV offload
// hierarchy as
//
//	offload: {tiers: [{tier: <catalog device class>, bytes}], connector, spec,
//	          eviction_policy, prefetch_depth}
//
// and BLIS simulates vLLM's OffloadingConnector, whose CPU tier is TieringOffloadingSpec's
// cpu_bytes_to_use (vllm/v1/kv_offload/cpu/spec.py:97-120 and tiering/spec.py at v0.31.0).
// KVOffloadBlockFromDeployment maps the block onto that model's kv_offload terms and the
// shared resolver applies vLLM's defaults to everything the schema does not state; a field the
// model does not implement is refused by name rather than dropped (R1).
//
// The block is deployment-level and the schema has no per-pool form, while vLLM's
// cpu_bytes_to_use belongs to one engine's connector config. So every engine -- the one pool
// of a colocated run, each prefill and each decode engine of a P/D run -- gets its own tier of
// the stated bytes (kernelPools.applyOffload sizes it by that pool's bytes per block).

// OffloadCPUTier is the catalog device class that names the CPU (host DRAM) tier: the
// OffloadingConnector's primary tier, sized by cpu_bytes_to_use.
const OffloadCPUTier = "cpu_dram"

// fsTierRoot is the root_dir a scenario's secondary tier is given. blis-schemas states a tier as
// a catalog device and its bytes, not a directory; the directory is only the tier's identity in
// vLLM (vllm/v1/kv_offload/file_mapper.py:135-144 at v0.31.0), and BLIS shares a node's tier
// among that node's engines by position (sim.OffloadTierSharing), so one fixed name stands for
// the device's mount.
const fsTierRoot = "/mnt/kv-offload"

// objectStoreDevices are catalog devices that are not a filesystem: vLLM reaches them through
// its obj tier (tiering/obj), which BLIS does not simulate.
var objectStoreDevices = map[string]bool{"s3": true}

// offloadConnectors are the offload connectors the model is: vLLM's native OffloadingConnector
// (the connector `--kv-offloading-backend native` configures, vllm/config/vllm.py:1140-1148).
// SimpleCPUOffloadConnector and LMCache place and evict KV on other terms.
var offloadConnectors = map[string]bool{"": true, "OffloadingConnector": true, SimpleConnector: true}

// SimpleConnector is vLLM's SimpleCPUOffloadConnector (vllm/v1/simple_kv_offload at v0.31.0):
// one block-granular CPU tier whose stores are asynchronous copies. Its cpu_bytes_to_use is
// per engine and divided among the engine's workers (simple_cpu_offload_connector.py:85-137),
// as the OffloadingConnector's is; the scenario tier's bytes are the engine's.
const SimpleConnector = "SimpleCPUOffloadConnector"

// offloadSpecs are the OffloadingConnector specs the model is. With only a CPU tier the two
// registered specs (rules v0_31 OffloadSpecs) keep the same CPU tier.
var offloadSpecs = map[string]bool{"": true, "CPUOffloadingSpec": true, "TieringOffloadingSpec": true}

// KVOffloadBlockFromDeployment maps a scenario's offload block onto the kv_offload model, or
// names what the model cannot represent. The schema has already refused a tier of no bytes.
func KVOffloadBlockFromDeployment(o *deployment.Offload) (*KVOffloadBlock, error) {
	if !offloadConnectors[o.Connector] {
		return nil, fmt.Errorf("offload.connector %q is not simulated: BLIS models vLLM's "+
			"OffloadingConnector, so state it (or leave the field out)", o.Connector)
	}
	if o.Connector == SimpleConnector {
		return simpleOffloadBlock(o)
	}
	if !offloadSpecs[o.Spec] {
		return nil, fmt.Errorf("offload.spec %q is not simulated: BLIS models the OffloadingConnector's "+
			"CPUOffloadingSpec and TieringOffloadingSpec", o.Spec)
	}
	if o.PrefetchDepth != 0 {
		return nil, fmt.Errorf("offload.prefetch_depth %d is not simulated: a tier fetch is issued "+
			"when batch formation needs it, so leave prefetch_depth out", o.PrefetchDepth)
	}
	block := &KVOffloadBlock{}
	for i, t := range o.Tiers {
		if t.Device == OffloadCPUTier {
			b := t.Bytes
			block.CPUBytesToUse = &b
			continue
		}
		if objectStoreDevices[t.Device] {
			return nil, fmt.Errorf("offload.tiers[%d] (%q) is an object store, which vLLM reaches through "+
				"its obj secondary tier; BLIS simulates the fs tier only", i, t.Device)
		}
		if o.Spec != "TieringOffloadingSpec" {
			return nil, fmt.Errorf("offload.tiers[%d] (%q) is a secondary tier, which only the "+
				"TieringOffloadingSpec runs (vllm/v1/kv_offload/tiering/spec.py); state "+
				"offload.spec: TieringOffloadingSpec", i, t.Device)
		}
		block.SecondaryTiers = append(block.SecondaryTiers, fsTierBlock(t))
	}
	if len(block.SecondaryTiers) > 0 && (len(o.Tiers) == 0 || o.Tiers[0].Device != OffloadCPUTier) {
		return nil, fmt.Errorf("offload.tiers[0] must be the %s tier: every secondary tier sits behind "+
			"the CPU primary tier (tiering/manager.py:1-20)", OffloadCPUTier)
	}
	if block.CPUBytesToUse == nil {
		return nil, fmt.Errorf("offload states no %s tier: the OffloadingConnector's primary tier is "+
			"host DRAM (cpu_bytes_to_use, required by vLLM)", OffloadCPUTier)
	}
	if o.EvictionPolicy != "" {
		ep := o.EvictionPolicy
		block.EvictionPolicy = &ep
	}
	return block, nil
}

// OffloadConfigDiff names the fields in which two resolved offload configs differ, "" when
// they agree. PerBlockBytes is not compared: it is derived from the kernel after resolution.
func OffloadConfigDiff(a, b engine.KVOffloadConfig) string {
	a.PerBlockBytes, b.PerBlockBytes = 0, 0
	va, vb := reflect.ValueOf(a), reflect.ValueOf(b)
	var diffs []string
	for i := 0; i < va.NumField(); i++ {
		if !reflect.DeepEqual(va.Field(i).Interface(), vb.Field(i).Interface()) {
			name := va.Type().Field(i).Name
			diffs = append(diffs, fmt.Sprintf("%s: scenario %v, flag %v", name, va.Field(i).Interface(), vb.Field(i).Interface()))
		}
	}
	return strings.Join(diffs, "; ")
}

// fsTierBlock is the fs secondary tier a scenario tier states: vLLM's fs tier on the device
// (secondary_tiers [{type: fs, root_dir}], tiering/fs/manager.py:109-120 at v0.31.0) with its
// default 16 read and 16 write threads, holding up to the stated bytes. It runs O_DIRECT, which
// vLLM probes and uses where the filesystem accepts it (fs/manager.py:191-201) -- a local NVMe
// ext4 or xfs mount does -- and which is the regime the kernel prices.
func fsTierBlock(t deployment.Tier) KVOffloadTierBlock {
	fs, root, direct, class := "fs", fsTierRoot, true, t.Device
	capacity := t.Bytes
	return KVOffloadTierBlock{Type: &fs, RootDir: &root, DirectIO: &direct, DeviceClass: &class,
		CapacityBytes: &capacity}
}

// simpleOffloadBlock maps a SimpleCPUOffloadConnector block: one cpu_dram tier, block-granular,
// every block offloaded (its eager store has no prompt-only clamp, manager.py:641-951 at
// v0.31.0). The connector takes no spec; its lazy_offload mode is OffloadSpec "lazy" here,
// since blis-schemas has no field for it (eager, vLLM's default, when unstated).
func simpleOffloadBlock(o *deployment.Offload) (*KVOffloadBlock, error) {
	if o.Spec != "" && o.Spec != "lazy" && o.Spec != "eager" {
		return nil, fmt.Errorf("offload.spec %q: the SimpleCPUOffloadConnector has no offloading spec; "+
			"state \"lazy\" for lazy_offload, \"eager\" or nothing for its default", o.Spec)
	}
	if len(o.Tiers) != 1 || o.Tiers[0].Device != OffloadCPUTier {
		return nil, fmt.Errorf("offload: the SimpleCPUOffloadConnector has one %s tier (its disk backend "+
			"replaces the CPU tier and is not simulated)", OffloadCPUTier)
	}
	if o.PrefetchDepth != 0 {
		return nil, fmt.Errorf("offload.prefetch_depth %d is not simulated", o.PrefetchDepth)
	}
	b := o.Tiers[0].Bytes
	promptOnly, one, simple, lazy := false, int64(1), true, o.Spec == "lazy"
	block := &KVOffloadBlock{CPUBytesToUse: &b, OffloadPromptOnly: &promptOnly, BlocksPerChunk: &one,
		Simple: &simple, LazyOffload: &lazy}
	if o.EvictionPolicy != "" && o.EvictionPolicy != "lru" {
		return nil, fmt.Errorf("offload.eviction_policy %q: the SimpleCPUOffloadConnector's CPU pool is LRU", o.EvictionPolicy)
	}
	return block, nil
}

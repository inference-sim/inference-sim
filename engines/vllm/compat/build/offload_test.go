// Moved from inference-sim/cmd/kv_offload_test.go at ace1528fe4bba59e510eda48e80ee9b8142ea982.

package build

import (
	"strings"
	"testing"
)

// testDevices is a fixed device map for resolver tests (independent of defaults.yaml).
func testDevices() map[string]KVOffloadDevice {
	return map[string]KVOffloadDevice{
		"nvme_gen4": {ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80},
		"sata_ssd":  {ReadBandwidth: 550, WriteBandwidth: 500, BaseLatency: 150},
	}
}

func i64p(v int64) *int64 { return &v }

func intp(v int) *int { return &v }

func f64p(v float64) *float64 { return &v }

func strp(v string) *string { return &v }

func boolp(v bool) *bool { return &v }

// validTierBlock returns a minimal valid fs tier block using a device_class.
func validTierBlock() KVOffloadTierBlock {
	return KVOffloadTierBlock{
		Type:        strp("fs"),
		RootDir:     strp("/mnt/kv"),
		DirectIO:    boolp(true),
		DeviceClass: strp("nvme_gen4"),
	}
}

// validBlock returns a minimal valid kv_offload block with one fs tier.
func validBlock() *KVOffloadBlock {
	return &KVOffloadBlock{
		CPUBytesToUse:  i64p(17179869184),
		SecondaryTiers: []KVOffloadTierBlock{validTierBlock()},
	}
}

// T4 (#1770: retargeted from defaults.yaml to the catalog): a catalog
// devices/storage.yaml parses through the real reader and a device resolves to a triple.
func TestKVOffloadDevices_CatalogStorageYAMLParses(t *testing.T) {
	catalog := writeCatalogStorageDevices(t,
		"nvme_gen4: {read_bandwidth_mb_s: 7.0e3, write_bandwidth_mb_s: 5.0e3, base_latency_us: 80.0}\n")
	devices, err := LoadCatalogStorageDevices(catalog)
	if err != nil {
		t.Fatalf("catalog storage device table must load: %v", err)
	}
	dev, ok := devices["nvme_gen4"]
	if !ok {
		t.Fatalf("nvme_gen4 device class must parse from %s", CatalogStorageDevicesRelPath)
	}
	if dev.ReadBandwidth != 7000 || dev.WriteBandwidth != 5000 || dev.BaseLatency != 80 {
		t.Errorf("device triple mismatch: %+v", dev)
	}
}

// T4 (#1770): the committed catalog device table parses (catches a malformed file). Since
// #1771 deleted the bundled model_configs/ tree, the committed clone-root fixture
// testdata/catalog/ (mirroring blis-catalog) carries the devices/ namespace the tests
// resolve against.
func TestKVOffloadDevices_CommittedCatalogTableParses(t *testing.T) {
	devices, err := LoadCatalogStorageDevices("../../testdata/catalog")
	if err != nil {
		t.Fatalf("committed catalog %s must load: %v", CatalogStorageDevicesRelPath, err)
	}
	if _, ok := devices["nvme_gen4"]; !ok {
		t.Errorf("committed catalog %s should define nvme_gen4 (got %s)",
			CatalogStorageDevicesRelPath, KnownDeviceClasses(devices))
	}
}

// T5: strict loader — valid parses; unknown key errors; store_threshold is captured.
func TestParseKVOffloadBytes(t *testing.T) {
	valid := "kv_offload:\n  cpu_bytes_to_use: 1024\n  eviction_policy: lru\n"
	if _, err := ParseKVOffloadBytes([]byte(valid)); err != nil {
		t.Fatalf("valid config must parse, got %v", err)
	}

	unknown := "kv_offload:\n  cpu_bytes_to_use: 1024\n  not_a_field: 3\n"
	if _, err := ParseKVOffloadBytes([]byte(unknown)); err == nil {
		t.Fatal("unknown key must error (KnownFields, R10)")
	}

	// store_threshold is a captured field (parses fine); the resolver, not the parser,
	// rejects >= 2.
	st := "kv_offload:\n  cpu_bytes_to_use: 1024\n  store_threshold: 1\n"
	f, err := ParseKVOffloadBytes([]byte(st))
	if err != nil {
		t.Fatalf("store_threshold key must parse, got %v", err)
	}
	if f.KVOffload == nil || f.KVOffload.StoreThreshold == nil || *f.KVOffload.StoreThreshold != 1 {
		t.Errorf("store_threshold must be captured as 1")
	}
}

// T6/BC-G2: omitted knobs resolve to vLLM's defaults, knob for knob.
func TestResolveKVOffload_Defaults(t *testing.T) {
	// gpu block size 16; block with only the required cpu_bytes_to_use and one tier.
	cfg, err := ResolveKVOffload(validBlock(), testDevices(), 16)
	if err != nil {
		t.Fatalf("valid minimal config must resolve, got %v", err)
	}
	if !cfg.Enabled {
		t.Fatal("resolved config must be Enabled")
	}
	// vLLM defaults (BC-G2), knob for knob.
	if cfg.EvictionPolicy != "lru" {
		t.Errorf("eviction_policy default: got %q want lru", cfg.EvictionPolicy)
	}
	if !cfg.OffloadPromptOnly {
		t.Errorf("offload_prompt_only default must be TRUE (trap 1)")
	}
	if cfg.SelfDescribingKVEvents {
		t.Errorf("self_describing_kv_events default must be false")
	}
	if cfg.BlocksPerChunk != 1 {
		t.Errorf("blocks_per_chunk default: got %d want 1", cfg.BlocksPerChunk)
	}
	if cfg.BlockSize != 16 {
		t.Errorf("block_size default must equal GPU block size 16, got %d", cfg.BlockSize)
	}
	if cfg.TokensPerHash != 16 {
		t.Errorf("tokens_per_hash default must equal GPU block size 16, got %d", cfg.TokensPerHash)
	}
	tr := cfg.Tiers[0]
	if tr.NReadThreads != 16 || tr.NWriteThreads != 16 {
		t.Errorf("n_read/write_threads default 16: got %d/%d", tr.NReadThreads, tr.NWriteThreads)
	}
	if tr.EnableKVEvents {
		t.Errorf("enable_kv_events default must be false")
	}
	if tr.Locality != "" {
		t.Errorf("locality default must be unset, got %q", tr.Locality)
	}
	// device_class resolved to the triple.
	if tr.ReadBandwidth != 7000 || tr.WriteBandwidth != 5000 || tr.BaseLatency != 80 {
		t.Errorf("device_class did not resolve triple: %+v", tr)
	}
}

// T6: explicit bandwidth/latency overrides the device_class.
func TestResolveKVOffload_ExplicitOverridesClass(t *testing.T) {
	b := validBlock()
	b.SecondaryTiers[0].ReadBandwidth = f64p(1234)
	b.SecondaryTiers[0].BaseLatency = f64p(42)
	cfg, err := ResolveKVOffload(b, testDevices(), 16)
	if err != nil {
		t.Fatal(err)
	}
	tr := cfg.Tiers[0]
	if tr.ReadBandwidth != 1234 || tr.BaseLatency != 42 {
		t.Errorf("explicit override not applied: %+v", tr)
	}
	if tr.WriteBandwidth != 5000 {
		t.Errorf("un-overridden write_bandwidth should keep class value 5000, got %v", tr.WriteBandwidth)
	}
}

// T6: block_size supplied derives blocks_per_chunk (alternate encoding).
func TestResolveKVOffload_BlockSizeDerivesChunk(t *testing.T) {
	b := validBlock()
	b.BlockSize = i64p(64) // 4 × gpu block size 16
	cfg, err := ResolveKVOffload(b, testDevices(), 16)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.BlockSize != 64 || cfg.BlocksPerChunk != 4 {
		t.Errorf("block_size=64/gpu=16 must give blocks_per_chunk=4, got block=%d chunk=%d", cfg.BlockSize, cfg.BlocksPerChunk)
	}
}

// T6/BC-G1/BC-G3: reject branches (each assertable — pure function, no os.Exit).
func TestResolveKVOffload_Rejects(t *testing.T) {
	tests := []struct {
		name     string
		mutate   func(*KVOffloadBlock)
		wantFrag string
	}{
		{"missing cpu_bytes", func(b *KVOffloadBlock) { b.CPUBytesToUse = nil }, "cpu_bytes_to_use"},
		{"store_threshold 2", func(b *KVOffloadBlock) { b.StoreThreshold = intp(2) }, "TieringOffloadingSpec"},
		{"store_threshold 5", func(b *KVOffloadBlock) { b.StoreThreshold = intp(5) }, "TieringOffloadingSpec"},
		{"store_threshold negative", func(b *KVOffloadBlock) { b.StoreThreshold = intp(-1) }, "store_threshold must be >= 0"},
		{"both block sizes", func(b *KVOffloadBlock) { b.BlockSize = i64p(16); b.BlocksPerChunk = i64p(1) }, "mutually exclusive"},
		{"block_size not multiple", func(b *KVOffloadBlock) { b.BlockSize = i64p(20) }, "multiple of the GPU block size"},
		{"tokens_per_hash zero", func(b *KVOffloadBlock) { b.TokensPerHash = i64p(0) }, "tokens_per_hash"},
		{"tier type obj", func(b *KVOffloadBlock) { b.SecondaryTiers[0].Type = strp("obj") }, "not supported"},
		{"tier type p2p", func(b *KVOffloadBlock) { b.SecondaryTiers[0].Type = strp("p2p") }, "not supported"},
		{"tier missing type", func(b *KVOffloadBlock) { b.SecondaryTiers[0].Type = nil }, "type is required"},
		{"tier missing root_dir", func(b *KVOffloadBlock) { b.SecondaryTiers[0].RootDir = nil }, "root_dir is required"},
		{"tier missing direct_io", func(b *KVOffloadBlock) { b.SecondaryTiers[0].DirectIO = nil }, "direct_io must be set"},
		// #1770: the device table moved from defaults.yaml to the catalog, so the refusal
		// now names the catalog file. TestResolveKVOffload_UnknownDeviceClassNamesCatalogTable
		// covers the full message (known classes, sorted, and no stale defaults.yaml mention).
		{"unknown device_class", func(b *KVOffloadBlock) { b.SecondaryTiers[0].DeviceClass = strp("floppy") }, "not defined in the catalog storage-device table"},
		{"no class no triple", func(b *KVOffloadBlock) {
			b.SecondaryTiers[0].DeviceClass = nil
		}, "device_class OR an explicit"},
		{"eviction bad", func(b *KVOffloadBlock) { b.EvictionPolicy = strp("fifo") }, "eviction_policy"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b := validBlock()
			tc.mutate(b)
			_, err := ResolveKVOffload(b, testDevices(), 16)
			if err == nil {
				t.Fatalf("expected error for %q, got nil", tc.name)
			}
			if !strings.Contains(err.Error(), tc.wantFrag) {
				t.Fatalf("error %q must contain %q", err.Error(), tc.wantFrag)
			}
		})
	}
}

// T6: store_threshold 0 and 1 are accepted (no-op on the multi-tier path, vLLM parity).
func TestResolveKVOffload_StoreThresholdAllowsZeroAndOne(t *testing.T) {
	for _, v := range []int{0, 1} {
		b := validBlock()
		b.StoreThreshold = intp(v)
		if _, err := ResolveKVOffload(b, testDevices(), 16); err != nil {
			t.Errorf("store_threshold=%d must be accepted (no-op), got %v", v, err)
		}
	}
}

// T6: a nil block errors (a supplied file with no kv_offload: block).
func TestResolveKVOffload_NilBlockErrors(t *testing.T) {
	if _, err := ResolveKVOffload(nil, testDevices(), 16); err == nil {
		t.Fatal("nil block must error")
	}
}

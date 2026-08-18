package cmd

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

// testDevices is a fixed device map for resolver tests (independent of defaults.yaml).
func testDevices() map[string]KVOffloadDeviceDefaults {
	return map[string]KVOffloadDeviceDefaults{
		"nvme_gen4": {ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80},
		"sata_ssd":  {ReadBandwidth: 550, WriteBandwidth: 500, BaseLatency: 150},
	}
}

func i64p(v int64) *int64     { return &v }
func intp(v int) *int         { return &v }
func f64p(v float64) *float64 { return &v }
func strp(v string) *string   { return &v }
func boolp(v bool) *bool      { return &v }

// validTierBlock returns a minimal valid fs tier block using a device_class.
func validTierBlock() kvOffloadTierBlock {
	return kvOffloadTierBlock{
		Type:        strp("fs"),
		RootDir:     strp("/mnt/kv"),
		DirectIO:    boolp(true),
		DeviceClass: strp("nvme_gen4"),
	}
}

// validBlock returns a minimal valid kv_offload block with one fs tier.
func validBlock() *kvOffloadBlock {
	return &kvOffloadBlock{
		CPUBytesToUse:  i64p(17179869184),
		SecondaryTiers: []kvOffloadTierBlock{validTierBlock()},
	}
}

// T4: the committed defaults.yaml device block parses, and a device resolves to a triple.
func TestKVOffloadDevices_DefaultsYAMLParses(t *testing.T) {
	// Write a minimal defaults.yaml with a kv_offload_devices block and parse it via
	// the real loader to exercise Config decoding of the new field (R10 strict).
	dir := t.TempDir()
	path := filepath.Join(dir, "defaults.yaml")
	content := "version: 0.0.1\n" +
		"kv_offload_devices:\n" +
		"  nvme_gen4: {read_bandwidth: 7.0e3, write_bandwidth: 5.0e3, base_latency: 80.0}\n"
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	cfg := loadDefaultsConfig(path)
	dev, ok := cfg.KVOffloadDevices["nvme_gen4"]
	if !ok {
		t.Fatalf("nvme_gen4 device class must parse from defaults.yaml")
	}
	if dev.ReadBandwidth != 7000 || dev.WriteBandwidth != 5000 || dev.BaseLatency != 80 {
		t.Errorf("device triple mismatch: %+v", dev)
	}
}

// T4: the actual committed repo defaults.yaml parses (catches a malformed block).
func TestKVOffloadDevices_CommittedDefaultsParse(t *testing.T) {
	cfg := loadDefaultsConfig("../defaults.yaml")
	if len(cfg.KVOffloadDevices) == 0 {
		t.Fatalf("committed defaults.yaml should define kv_offload_devices")
	}
	if _, ok := cfg.KVOffloadDevices["nvme_gen4"]; !ok {
		t.Errorf("committed defaults.yaml should define nvme_gen4")
	}
}

// T5: strict loader — valid parses; unknown key errors; store_threshold is captured.
func TestParseKVOffloadBytes(t *testing.T) {
	valid := "kv_offload:\n  cpu_bytes_to_use: 1024\n  eviction_policy: lru\n"
	if _, err := parseKVOffloadBytes([]byte(valid)); err != nil {
		t.Fatalf("valid config must parse, got %v", err)
	}

	unknown := "kv_offload:\n  cpu_bytes_to_use: 1024\n  not_a_field: 3\n"
	if _, err := parseKVOffloadBytes([]byte(unknown)); err == nil {
		t.Fatal("unknown key must error (KnownFields, R10)")
	}

	// store_threshold is a captured field (parses fine); the resolver, not the parser,
	// rejects >= 2.
	st := "kv_offload:\n  cpu_bytes_to_use: 1024\n  store_threshold: 1\n"
	f, err := parseKVOffloadBytes([]byte(st))
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
	cfg, err := resolveKVOffload(validBlock(), testDevices(), 16)
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
	cfg, err := resolveKVOffload(b, testDevices(), 16)
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
	cfg, err := resolveKVOffload(b, testDevices(), 16)
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
		mutate   func(*kvOffloadBlock)
		wantFrag string
	}{
		{"missing cpu_bytes", func(b *kvOffloadBlock) { b.CPUBytesToUse = nil }, "cpu_bytes_to_use"},
		{"store_threshold 2", func(b *kvOffloadBlock) { b.StoreThreshold = intp(2) }, "TieringOffloadingSpec"},
		{"store_threshold 5", func(b *kvOffloadBlock) { b.StoreThreshold = intp(5) }, "TieringOffloadingSpec"},
		{"store_threshold negative", func(b *kvOffloadBlock) { b.StoreThreshold = intp(-1) }, "store_threshold must be >= 0"},
		{"both block sizes", func(b *kvOffloadBlock) { b.BlockSize = i64p(16); b.BlocksPerChunk = i64p(1) }, "mutually exclusive"},
		{"block_size not multiple", func(b *kvOffloadBlock) { b.BlockSize = i64p(20) }, "multiple of the GPU block size"},
		{"tokens_per_hash zero", func(b *kvOffloadBlock) { b.TokensPerHash = i64p(0) }, "tokens_per_hash"},
		{"tier type obj", func(b *kvOffloadBlock) { b.SecondaryTiers[0].Type = strp("obj") }, "not supported"},
		{"tier type p2p", func(b *kvOffloadBlock) { b.SecondaryTiers[0].Type = strp("p2p") }, "not supported"},
		{"tier missing type", func(b *kvOffloadBlock) { b.SecondaryTiers[0].Type = nil }, "type is required"},
		{"tier missing root_dir", func(b *kvOffloadBlock) { b.SecondaryTiers[0].RootDir = nil }, "root_dir is required"},
		{"tier missing direct_io", func(b *kvOffloadBlock) { b.SecondaryTiers[0].DirectIO = nil }, "direct_io must be set"},
		{"unknown device_class", func(b *kvOffloadBlock) { b.SecondaryTiers[0].DeviceClass = strp("floppy") }, "not defined in defaults.yaml"},
		{"no class no triple", func(b *kvOffloadBlock) {
			b.SecondaryTiers[0].DeviceClass = nil
		}, "device_class OR an explicit"},
		{"eviction bad", func(b *kvOffloadBlock) { b.EvictionPolicy = strp("fifo") }, "eviction_policy"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b := validBlock()
			tc.mutate(b)
			_, err := resolveKVOffload(b, testDevices(), 16)
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
		if _, err := resolveKVOffload(b, testDevices(), 16); err != nil {
			t.Errorf("store_threshold=%d must be accepted (no-op), got %v", v, err)
		}
	}
}

// T6: a nil block errors (a supplied file with no kv_offload: block).
func TestResolveKVOffload_NilBlockErrors(t *testing.T) {
	if _, err := resolveKVOffload(nil, testDevices(), 16); err == nil {
		t.Fatal("nil block must error")
	}
}

// T9 support: sim↔header conversion is a lossless inverse.
func TestKVOffloadHeaderConversion_RoundTrip(t *testing.T) {
	cfg, err := resolveKVOffload(validBlock(), testDevices(), 16)
	if err != nil {
		t.Fatal(err)
	}
	back := headerToSimOffload(simToHeaderOffload(cfg))
	if !reflect.DeepEqual(cfg, back) {
		t.Errorf("sim→header→sim not identity:\n got  %+v\n want %+v", back, cfg)
	}
	// inert converts to nil header and back to inert.
	if simToHeaderOffload(sim.KVOffloadConfig{}) != nil {
		t.Error("inert config must convert to a nil header block")
	}
	if headerToSimOffload(nil).IsEnabled() {
		t.Error("nil header must convert to an inert config")
	}
}

// T7/BC-G3 + INV-13: metamorphic round-trip fuzz. Any bytes that parse+resolve+validate
// must survive the FULL trace-header serialization path unchanged —
// sim→header→yaml.Marshal→strict-decode→header→sim (the exact primitives ExportTraceV2
// and LoadTraceV2 use). This exercises the YAML tags and omitempty rules of
// TraceKVOffloadConfig/TraceKVOffloadTier (not just in-memory converter idempotence),
// and asserts the loader/resolver never panic on arbitrary parseable YAML.
func FuzzKVOffloadRoundTrip(f *testing.F) {
	f.Add("kv_offload:\n  cpu_bytes_to_use: 17179869184\n  secondary_tiers:\n    - {type: fs, root_dir: /mnt, direct_io: true, device_class: nvme_gen4}\n")
	f.Add("kv_offload:\n  cpu_bytes_to_use: 1024\n  store_threshold: 3\n")
	f.Add("kv_offload:\n  cpu_bytes_to_use: 1024\n  block_size: 16\n  blocks_per_chunk: 1\n")
	f.Add("kv_offload:\n  cpu_bytes_to_use: 1024\n  secondary_tiers:\n    - {type: p2p, root_dir: /x, direct_io: true}\n")
	f.Add("kv_offload:\n  cpu_bytes_to_use: 1024\n  secondary_tiers:\n    - {type: fs, root_dir: /m, direct_io: false, read_bandwidth: 100.0, write_bandwidth: 50.0, base_latency: 0.0, locality: REMOTE, enable_kv_events: true}\n")
	f.Add("not yaml: [")
	devices := testDevices()
	f.Fuzz(func(t *testing.T, data string) {
		parsed, err := parseKVOffloadBytes([]byte(data))
		if err != nil {
			return // malformed YAML — loader correctly rejects
		}
		cfg, err := resolveKVOffload(parsed.KVOffload, devices, 16)
		if err != nil {
			return // invalid config — resolver correctly rejects
		}
		// Serialize through the real trace-header YAML path.
		hdr := &workload.TraceHeader{
			Version: 3, TimeUnit: "microseconds", Mode: "generated",
			KVOffload: simToHeaderOffload(cfg),
		}
		out, err := yaml.Marshal(hdr)
		if err != nil {
			t.Fatalf("marshal header for %q: %v", data, err)
		}
		dec := yaml.NewDecoder(bytes.NewReader(out))
		dec.KnownFields(true) // same strictness as LoadTraceV2
		var got workload.TraceHeader
		if err := dec.Decode(&got); err != nil {
			t.Fatalf("strict-decode of self-marshaled header failed for %q: %v\n%s", data, err, out)
		}
		back := headerToSimOffload(got.KVOffload)
		if !reflect.DeepEqual(cfg, back) {
			t.Fatalf("YAML round-trip not idempotent for input %q:\n got  %+v\n want %+v", data, back, cfg)
		}
	})
}

// resolveReplayKVOffload branch coverage (T9, BC-G6): the trace header is
// authoritative on replay; every reject branch is exercised via the pure function
// (no os.Exit needed).
func TestResolveReplayKVOffload(t *testing.T) {
	// A valid recorded config (explicit triple so it validates without a device map).
	validHeader := &workload.TraceKVOffloadConfig{
		CPUBytesToUse: 1024, BlockSize: 16, BlocksPerChunk: 1, TokensPerHash: 16,
		EvictionPolicy: "lru", OffloadPromptOnly: true,
		Tiers: []workload.TraceKVOffloadTier{{
			Type: "fs", RootDir: "/mnt", NReadThreads: 16, NWriteThreads: 16,
			DirectIO: true, ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
		}},
	}
	validSim := headerToSimOffload(validHeader)

	t.Run("header authoritative, no flag", func(t *testing.T) {
		got, err := resolveReplayKVOffload(validHeader, false, sim.KVOffloadConfig{})
		if err != nil || !reflect.DeepEqual(got, validSim) {
			t.Fatalf("header should be used verbatim: got %+v err %v", got, err)
		}
	})
	t.Run("matching flag accepted", func(t *testing.T) {
		got, err := resolveReplayKVOffload(validHeader, true, validSim)
		if err != nil || !reflect.DeepEqual(got, validSim) {
			t.Fatalf("identical flag must be accepted: got %+v err %v", got, err)
		}
	})
	t.Run("conflicting flag rejected", func(t *testing.T) {
		other := validSim
		other.CPUBytesToUse = 2048
		_, err := resolveReplayKVOffload(validHeader, true, other)
		if err == nil || !strings.Contains(err.Error(), "conflicts") {
			t.Fatalf("conflicting flag must error with 'conflicts', got %v", err)
		}
	})
	t.Run("unreproducible header rejected", func(t *testing.T) {
		// A header recording a tier type this binary cannot reconstruct.
		bad := &workload.TraceKVOffloadConfig{
			CPUBytesToUse: 1024, BlockSize: 16, BlocksPerChunk: 1, TokensPerHash: 16,
			EvictionPolicy: "lru",
			Tiers:          []workload.TraceKVOffloadTier{{Type: "p2p", RootDir: "/x"}},
		}
		_, err := resolveReplayKVOffload(bad, false, sim.KVOffloadConfig{})
		if err == nil || !strings.Contains(err.Error(), "cannot reproduce") {
			t.Fatalf("unreproducible header must fail loudly, got %v", err)
		}
	})
	t.Run("no header, no flag -> inert", func(t *testing.T) {
		got, err := resolveReplayKVOffload(nil, false, sim.KVOffloadConfig{})
		if err != nil || got.IsEnabled() {
			t.Fatalf("nil header + no flag must be inert, got %+v err %v", got, err)
		}
	})
	t.Run("flag adds to a no-offload trace -> rejected", func(t *testing.T) {
		_, err := resolveReplayKVOffload(nil, true, validSim)
		if err == nil || !strings.Contains(err.Error(), "cannot add one on replay") {
			t.Fatalf("adding offload to a no-offload trace must error, got %v", err)
		}
	})
}

// runToTraceWithOffload drives runCmd in-process with --kv-offload-config set, writing
// a TraceV2 whose header records the resolved offload config. Mirrors the parity
// harness (runSpecToTraceFiles) plus the one new flag; uses an explicit bandwidth
// triple in the config so it does not depend on the fixture defaults.yaml device map.
func runToTraceWithOffload(t *testing.T, offloadPath string, seedVal, horizon int64) (headerFile, dataFile string) {
	t.Helper()
	shape := paritySpecShapes()[0] // chatbot
	tmpDir := t.TempDir()
	tracePrefix := filepath.Join(tmpDir, "trace")
	specPath := filepath.Join(tmpDir, "workload.yaml")
	if err := os.WriteFile(specPath, []byte(shape.yaml), 0644); err != nil {
		t.Fatal(err)
	}
	mcFolder, hwPath, defaultsPath := setupTrainedPhysicsTestFixturesWithDefaults(t)

	orig := captureCmdLevelVars()
	defer orig.restore()
	origOffload := kvOffloadConfigPath
	defer func() { kvOffloadConfigPath = origOffload }()

	traceOutput = tracePrefix
	workloadSpecPath = specPath
	workloadType = ""
	simulationHorizon = horizon
	seed = seedVal
	lazyGeneration = false
	requestTimeoutSecs = 300

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "")
	testCmd.Flags().StringVar(&traceOutput, "trace-output", "", "")
	testCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "")
	args := []string{
		"--model", "qwen/qwen3-14b", "--latency-model", "trained-physics",
		"--defaults-filepath", defaultsPath, "--model-config-folder", mcFolder,
		"--hardware-config", hwPath, "--hardware", "H100", "--tp", "1",
		"--total-kv-blocks", "1000", "--seed", strconv.FormatInt(seedVal, 10),
		"--workload-spec", specPath, "--horizon", strconv.FormatInt(horizon, 10),
		"--trace-output", tracePrefix, "--kv-offload-config", offloadPath,
	}
	if err := testCmd.ParseFlags(args); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	runCmd.Run(testCmd, nil)
	return tracePrefix + ".yaml", tracePrefix + ".csv"
}

// T9/BC-G6 end-to-end: a run with a multi-tier --kv-offload-config records the resolved
// config in the trace header, and replaying that trace reproduces it (header
// authoritative) without error.
func TestKVOffload_EndToEnd_RunReplayRoundTrip(t *testing.T) {
	dir := t.TempDir()
	offloadPath := filepath.Join(dir, "offload.yaml")
	offloadYAML := "kv_offload:\n" +
		"  cpu_bytes_to_use: 17179869184\n" +
		"  eviction_policy: lru\n" +
		"  offload_prompt_only: true\n" +
		"  secondary_tiers:\n" +
		"    - type: fs\n" +
		"      root_dir: /mnt/kv\n" +
		"      direct_io: true\n" +
		"      read_bandwidth: 7000.0\n" +
		"      write_bandwidth: 5000.0\n" +
		"      base_latency: 80.0\n"
	if err := os.WriteFile(offloadPath, []byte(offloadYAML), 0644); err != nil {
		t.Fatal(err)
	}

	headerFile, dataFile := runToTraceWithOffload(t, offloadPath, 20260818, 60_000_000)

	td, err := workload.LoadTraceV2(headerFile, dataFile)
	if err != nil {
		t.Fatalf("load exported trace: %v", err)
	}
	if td.Header.KVOffload == nil {
		t.Fatal("run must record kv_offload in the trace header (BC-G6)")
	}
	h := td.Header.KVOffload
	if h.CPUBytesToUse != 17179869184 || h.EvictionPolicy != "lru" || !h.OffloadPromptOnly {
		t.Errorf("recorded scalars wrong: %+v", h)
	}
	if h.BlockSize != 16 || h.BlocksPerChunk != 1 || h.TokensPerHash != 16 {
		t.Errorf("resolved defaults not recorded: %+v", h)
	}
	if len(h.Tiers) != 1 || h.Tiers[0].ReadBandwidth != 7000 || h.Tiers[0].WriteBandwidth != 5000 || !h.Tiers[0].DirectIO {
		t.Errorf("recorded tier wrong: %+v", h.Tiers)
	}

	// Replay the trace: the header is authoritative; no flag passed. Must reproduce
	// without a Fatalf and produce completed requests.
	results := replaySpecTrace(t, headerFile, dataFile)
	if len(results) == 0 {
		t.Fatal("replay of an offload trace produced no completed requests")
	}
}

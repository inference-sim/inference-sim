package cmd

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

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
	if simToHeaderOffload(kvOffloadConfig{}) != nil {
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
			Version: workload.TraceVersion, TimeUnit: "microseconds", Mode: "generated",
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
		got, err := resolveReplayKVOffload(validHeader, false, kvOffloadConfig{}, false)
		if err != nil || !reflect.DeepEqual(got, validSim) {
			t.Fatalf("header should be used verbatim: got %+v err %v", got, err)
		}
	})
	t.Run("matching flag accepted", func(t *testing.T) {
		got, err := resolveReplayKVOffload(validHeader, true, validSim, false)
		if err != nil || !reflect.DeepEqual(got, validSim) {
			t.Fatalf("identical flag must be accepted: got %+v err %v", got, err)
		}
	})
	t.Run("conflicting flag rejected", func(t *testing.T) {
		other := validSim
		other.CPUBytesToUse = 2048
		_, err := resolveReplayKVOffload(validHeader, true, other, false)
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
		_, err := resolveReplayKVOffload(bad, false, kvOffloadConfig{}, false)
		if err == nil || !strings.Contains(err.Error(), "cannot reproduce") {
			t.Fatalf("unreproducible header must fail loudly, got %v", err)
		}
	})
	t.Run("no header, no flag -> inert", func(t *testing.T) {
		got, err := resolveReplayKVOffload(nil, false, kvOffloadConfig{}, false)
		if err != nil || got.IsEnabled() {
			t.Fatalf("nil header + no flag must be inert, got %+v err %v", got, err)
		}
	})
	t.Run("flag adds to a no-offload trace -> rejected", func(t *testing.T) {
		_, err := resolveReplayKVOffload(nil, true, validSim, false)
		if err == nil || !strings.Contains(err.Error(), "cannot add one on replay") {
			t.Fatalf("adding offload to a no-offload trace must error, got %v", err)
		}
	})
}

// runToTraceWithOffload drives runCmd in-process over the test scenario with offload, its offload
// block, inserted, writing
// a TraceV2 whose header records the resolved offload config. Mirrors the parity
// harness (runSpecToTraceFiles) plus the one new flag; uses an explicit bandwidth
// triple in the config so it does not depend on the fixture defaults.yaml device map.
func runToTraceWithOffload(t *testing.T, offload string, seedVal, horizon int64) (headerFile, dataFile string) {
	t.Helper()
	shape := paritySpecShapes()[0] // chatbot
	tmpDir := t.TempDir()
	tracePrefix := filepath.Join(tmpDir, "trace")
	specPath := filepath.Join(tmpDir, "workload.yaml")
	if err := os.WriteFile(specPath, []byte(shape.yaml), 0644); err != nil {
		t.Fatal(err)
	}
	catalogDir, defaultsPath := setupKernelTestFixturesWithDefaults(t)

	orig := captureCmdLevelVars()
	defer orig.restore()
	scenarios := writeOffloadScenario(t, kernelScenariosDir(t), kernelTestScenario, "", offload)

	traceOutput = tracePrefix
	workloadSpecPath = specPath
	workloadType = ""
	simulationHorizon = horizon
	seed = seedVal
	requestTimeoutSecs = 300

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "")
	testCmd.Flags().StringVar(&traceOutput, "trace-output", "", "")
	testCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "")
	args := []string{
		"--defaults-filepath", defaultsPath, "--catalog", catalogDir, "--scenario", kernelTestScenario, "--scenarios", scenarios, "--registry", kernelRegistryDir(t),
		"--seed", strconv.FormatInt(seedVal, 10),
		"--workload-spec", specPath, "--horizon", strconv.FormatInt(horizon, 10),
		"--trace-output", tracePrefix,
	}
	if err := testCmd.ParseFlags(args); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	runCmd.Run(testCmd, nil)
	return tracePrefix + ".yaml", tracePrefix + ".csv"
}

// T9/BC-G6 end-to-end: a run whose scenario states an offload block records the resolved
// config in the trace header, and replaying that trace reproduces it (header authoritative)
// without error. The block states the CPU tier only: the v0.32.0 engine refuses a secondary
// tier (blis-vllm#63).
func TestKVOffload_EndToEnd_RunReplayRoundTrip(t *testing.T) {
	offload := "offload:\n" +
		"  connector: OffloadingConnector\n" +
		"  spec: TieringOffloadingSpec\n" +
		"  eviction_policy: lru\n" +
		"  tiers:\n" +
		"    - tier: cpu_dram\n" +
		"      bytes: 17179869184\n"
	headerFile, dataFile := runToTraceWithOffload(t, offload, 20260818, 60_000_000)

	td, err := workload.LoadTraceV2(headerFile, dataFile)
	if err != nil {
		t.Fatalf("load exported trace: %v", err)
	}
	if td.Header.KVOffload == nil {
		t.Fatal("run must record kv_offload in the trace header (BC-G6)")
	}
	h := td.Header.KVOffload
	if h.CPUBytesToUse != 17179869184 || h.EvictionPolicy != "lru" {
		t.Errorf("recorded scalars wrong: %+v", h)
	}
	// The offload block size defaults to the GPU block size the kernel sized the run with,
	// one block per chunk and one hash per block.
	if h.BlockSize <= 0 || h.BlocksPerChunk != 1 || h.TokensPerHash != h.BlockSize {
		t.Errorf("resolved defaults not recorded: %+v", h)
	}
	if len(h.Tiers) != 0 {
		t.Errorf("a CPU-only offload block recorded secondary tiers: %+v", h.Tiers)
	}

	// Replay the trace: the header is authoritative; no flag passed. Must reproduce
	// without a Fatalf and produce completed requests.
	results := replaySpecTrace(t, headerFile, dataFile)
	if len(results) == 0 {
		t.Fatal("replay of an offload trace produced no completed requests")
	}
}

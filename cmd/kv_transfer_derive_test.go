package cmd

import (
	"bytes"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/latency"
)

// #1819 (R2G3b-sim): the legacy single-CPU-tier CPU↔GPU transfer cost is DERIVED from the
// catalog cpu_dram device fact instead of being authored as a blocks/tick flag default.
//
// The contracts pinned here:
//
//	BC-1  the derivation implements ticks = per_block_bytes / bandwidth — block_size_tokens
//	      cancels, so the rate handed to sim/kv is bandwidth ÷ KVBytesPerToken × residual
//	BC-2  VALUE-PRESERVING (R2): at the named reference deployment (qwen3-14b, TP=1,
//	      cpu_dram) the derived rate is EXACTLY the retired 100.0 default, so an enabled
//	      legacy run reproduces the pre-change cost
//	BC-3  degenerate inputs are refused, never silently resolved to zero physics (R1/R9)
//	BC-4  CLI: an enabled legacy run with no override is byte-identical to the same run
//	      passing the retired default explicitly — and NOT byte-identical to the same run
//	      at the residual-free nominal rate (the non-vacuity control)
//	BC-5  the flags still override the derivation verbatim
//	BC-6  INV-13: run and replay derive identically for the same enabled config
//	BC-7  INV-6 / lazy dependency: with --kv-cpu-blocks 0 nothing is derived and no
//	      catalog device table is required
//	BC-8  a catalog that cannot supply cpu_dram is refused naming the path AND the
//	      --kv-transfer-bandwidth escape hatch (R1)
//
// The static half — "a physics literal cannot come back into a cmd/ flag default", plus the
// LoRA defaults-vs-registry drift guard — lives in physics_literal_guard_test.go.

// ---------------------------------------------------------------------------
// Frozen goldens
// ---------------------------------------------------------------------------

// retiredKVTransferBandwidthDefault is the flag default #1819 removed, kept here as a
// FROZEN golden rather than in production code. It is the number the conversion must
// reproduce; regenerating it from whatever the derivation currently returns would make
// BC-2 vacuous.
const retiredKVTransferBandwidthDefault = 100.0

// referenceKVBytesPerToken is KVBytesPerToken(qwen3-14b, TP=1) — the anchor of
// legacyKVTransferResidual. 40 layers × 2 (K+V) × 128 head_dim × 8 kv_heads × 2 bytes
// (bf16), ÷ TP=1. Also FROZEN: the committed test catalog carries that model config, and
// TestDeriveLegacyKVTransferRate_ReferenceMatchesCommittedCatalog checks the two agree, so
// a catalog edit that moved the reference would be caught rather than silently re-anchoring
// the residual.
const referenceKVBytesPerToken = 163840.0

// referenceCPUDRAM is the cpu_dram entry the conversion reads, transcribed from
// <catalog>/devices/storage.yaml. Frozen for the same reason.
func referenceCPUDRAM() kvOffloadDevice {
	return kvOffloadDevice{ReadBandwidth: 2.0e4, WriteBandwidth: 2.0e4, BaseLatency: 1.0}
}

// ---------------------------------------------------------------------------
// BC-2: value preservation at the reference
// ---------------------------------------------------------------------------

// TestDeriveLegacyKVTransferRate_ReproducesRetiredDefault is BC-2, and it is the whole
// justification for legacyKVTransferResidual existing. EXACT equality is asserted, not a
// tolerance: the residual and the operand order in deriveLegacyKVTransferRate were chosen
// so the reference lands on 100.0 bit-for-bit, which is what makes an enabled legacy run
// byte-identical rather than merely close (INV-6).
func TestDeriveLegacyKVTransferRate_ReproducesRetiredDefault(t *testing.T) {
	got, err := deriveLegacyKVTransferRate(referenceCPUDRAM(), referenceKVBytesPerToken)
	if err != nil {
		t.Fatalf("deriveLegacyKVTransferRate at the reference: %v", err)
	}
	if got != retiredKVTransferBandwidthDefault {
		t.Errorf("derived rate at the reference deployment = %v, want exactly %v "+
			"(the retired --kv-transfer-bandwidth default); the R2G3b residual %v no longer "+
			"reproduces the shipped cost, which is a physics change and must be argued on its own",
			got, retiredKVTransferBandwidthDefault, legacyKVTransferResidual)
	}
}

// TestLegacyKVTransferResidual_IsTheReferenceRatio states the residual's DEFINITION
// independently of the constant: achieved rate ÷ rated rate at the reference. If someone
// edits the constant, this fails with the arithmetic that should have produced it.
func TestLegacyKVTransferResidual_IsTheReferenceRatio(t *testing.T) {
	nominal := referenceCPUDRAM().ReadBandwidth / referenceKVBytesPerToken // tokens/tick, rated
	want := retiredKVTransferBandwidthDefault / nominal
	if legacyKVTransferResidual != want {
		t.Errorf("legacyKVTransferResidual = %v, but the reference ratio "+
			"(retired %v tokens/tick ÷ rated %v tokens/tick) is %v",
			legacyKVTransferResidual, retiredKVTransferBandwidthDefault, nominal, want)
	}
	// Non-vacuity, and the honest reading: the retired default asserted far MORE than
	// cpu_dram's rated bandwidth, which is why R2G3b refused to transcribe it as physics.
	// A residual that had quietly become <= 1 would mean the anchor moved.
	if legacyKVTransferResidual <= 1 {
		t.Errorf("residual %v <= 1 contradicts the documented finding that the retired "+
			"blocks/tick default over-states cpu_dram", legacyKVTransferResidual)
	}
}

// TestDeriveLegacyKVTransferRate_ReferenceMatchesCommittedCatalog keeps the two frozen
// goldens honest against the committed test catalog: the cpu_dram row the conversion reads
// and the per-token KV size of the anchor model. Without this, a catalog edit would
// re-anchor the residual silently and BC-2 would still "pass" against a stale constant.
func TestDeriveLegacyKVTransferRate_ReferenceMatchesCommittedCatalog(t *testing.T) {
	catalog := filepath.Join("..", "testdata", "catalog")
	dev, err := loadLegacyKVTransferDevice(catalog)
	if err != nil {
		t.Fatalf("read committed test catalog device table: %v", err)
	}
	if dev.ReadBandwidth != referenceCPUDRAM().ReadBandwidth {
		t.Errorf("committed catalog cpu_dram read_bandwidth = %v, but the frozen reference used to "+
			"anchor legacyKVTransferResidual is %v — re-derive the residual deliberately",
			dev.ReadBandwidth, referenceCPUDRAM().ReadBandwidth)
	}

	configPath, resolveErr := resolveModelConfigInCatalog(devicesCLIModel, catalog)
	if resolveErr != nil {
		t.Fatalf("resolve %s in the committed test catalog: %v", devicesCLIModel, resolveErr)
	}
	hfConfig, parseErr := latency.ParseHFConfig(filepath.Join(configPath, hfConfigFile))
	if parseErr != nil {
		t.Fatalf("ParseHFConfig(%s): %v", configPath, parseErr)
	}
	mc, mcErr := latency.GetModelConfigFromHF(hfConfig)
	if mcErr != nil {
		t.Fatalf("GetModelConfigFromHF: %v", mcErr)
	}
	perToken, ptErr := latency.KVBytesPerToken(*mc, 1)
	if ptErr != nil {
		t.Fatalf("KVBytesPerToken(reference, TP=1): %v", ptErr)
	}
	if perToken != referenceKVBytesPerToken {
		t.Errorf("KVBytesPerToken(%s, TP=1) = %v, but the frozen anchor is %v — the residual is "+
			"defined against that number and must be re-derived if the model config moves",
			devicesCLIModel, perToken, referenceKVBytesPerToken)
	}
}

// ---------------------------------------------------------------------------
// BC-1: the derivation IS ticks = per_block_bytes / bandwidth
// ---------------------------------------------------------------------------

// TestDeriveLegacyKVTransferRate_ChargesPerBlockBytesOverBandwidth pins the law the issue
// states, as the tick cost sim/kv.TieredKVCache actually charges. The rate is expressed in
// tokens/tick precisely so block_size_tokens cancels; this checks the cancellation holds
// across block sizes and model sizes instead of trusting the algebra.
func TestDeriveLegacyKVTransferRate_ChargesPerBlockBytesOverBandwidth(t *testing.T) {
	dev := referenceCPUDRAM()
	for _, perToken := range []float64{referenceKVBytesPerToken, 4096, 1e6} {
		rate, err := deriveLegacyKVTransferRate(dev, perToken)
		if err != nil {
			t.Fatalf("derive(perToken=%v): %v", perToken, err)
		}
		for _, blockSizeTokens := range []int64{1, 16, 32, 128} {
			perBlockBytes := perToken * float64(blockSizeTokens)
			// The library's charge: ceil(block_size_tokens / rate).
			gotTicks := math.Ceil(float64(blockSizeTokens) / rate)
			// The issue's formula, at the residual-adjusted effective bandwidth.
			wantTicks := math.Ceil(perBlockBytes / (dev.ReadBandwidth * legacyKVTransferResidual))
			if gotTicks != wantTicks {
				t.Errorf("perToken=%v blockSize=%d: library charges %v ticks/block but "+
					"per_block_bytes/bandwidth is %v ticks/block", perToken, blockSizeTokens, gotTicks, wantTicks)
			}
		}
	}
}

// TestDeriveLegacyKVTransferRate_ScalesInverselyWithPerTokenBytes is the behavioural
// difference the conversion buys: the retired constant was model-independent, a bus is not.
// A model with a quarter the KV per token moves four times the tokens per tick over the
// same bus. Stated as a ratio law so it survives any change to the residual's value.
func TestDeriveLegacyKVTransferRate_ScalesInverselyWithPerTokenBytes(t *testing.T) {
	dev := referenceCPUDRAM()
	base, err := deriveLegacyKVTransferRate(dev, referenceKVBytesPerToken)
	if err != nil {
		t.Fatalf("derive(base): %v", err)
	}
	quarter, err := deriveLegacyKVTransferRate(dev, referenceKVBytesPerToken/4)
	if err != nil {
		t.Fatalf("derive(quarter): %v", err)
	}
	if quarter != base*4 {
		t.Errorf("quartering per-token KV bytes must quadruple the rate: got %v, want %v", quarter, base*4)
	}
	// And doubling the bus doubles the rate.
	faster := dev
	faster.ReadBandwidth *= 2
	doubled, err := deriveLegacyKVTransferRate(faster, referenceKVBytesPerToken)
	if err != nil {
		t.Fatalf("derive(faster bus): %v", err)
	}
	if doubled != base*2 {
		t.Errorf("doubling read_bandwidth must double the rate: got %v, want %v", doubled, base*2)
	}
}

// ---------------------------------------------------------------------------
// BC-3: degenerate inputs are refused
// ---------------------------------------------------------------------------

// TestDeriveLegacyKVTransferRate_RefusesDegenerateInputs: every route to a
// non-positive/non-finite rate errors naming the offending quantity. The alternative —
// returning 0, +Inf or NaN — reaches NewKVCacheConfig as a library panic with no hint that
// the catalog or the model shape was the cause.
func TestDeriveLegacyKVTransferRate_RefusesDegenerateInputs(t *testing.T) {
	good := referenceCPUDRAM()
	cases := []struct {
		name     string
		dev      kvOffloadDevice
		perToken float64
		wantFrag string
	}{
		{"per_token_zero", good, 0, "per-token KV bytes"},
		{"per_token_negative", good, -1, "per-token KV bytes"},
		{"per_token_nan", good, math.NaN(), "per-token KV bytes"},
		{"per_token_inf", good, math.Inf(1), "per-token KV bytes"},
		{"bandwidth_zero", kvOffloadDevice{}, referenceKVBytesPerToken, "read_bandwidth"},
		{"bandwidth_negative", kvOffloadDevice{ReadBandwidth: -1}, referenceKVBytesPerToken, "read_bandwidth"},
		{"bandwidth_nan", kvOffloadDevice{ReadBandwidth: math.NaN()}, referenceKVBytesPerToken, "read_bandwidth"},
		{"bandwidth_inf", kvOffloadDevice{ReadBandwidth: math.Inf(1)}, referenceKVBytesPerToken, "read_bandwidth"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rate, err := deriveLegacyKVTransferRate(tc.dev, tc.perToken)
			if err == nil {
				t.Fatalf("expected a refusal, got rate %v", rate)
			}
			if !strings.Contains(err.Error(), tc.wantFrag) {
				t.Errorf("refusal must name %q, got: %v", tc.wantFrag, err)
			}
		})
	}
}

// TestLoadLegacyKVTransferDevice_Diagnostics: the reader blames the legacy flag, not
// --kv-offload-config, and always names the escape hatch. Getting this wrong sends an
// operator who set only --kv-cpu-blocks hunting for a secondary tier they never configured.
func TestLoadLegacyKVTransferDevice_Diagnostics(t *testing.T) {
	cases := []struct {
		name  string
		setup func(t *testing.T) string
		frags []string
	}{
		{
			name:  "absent_table",
			setup: func(t *testing.T) string { return t.TempDir() },
			frags: []string{"kv-cpu-blocks", catalogStorageDevicesRelPath, "--kv-transfer-bandwidth"},
		},
		{
			name: "malformed_table",
			setup: func(t *testing.T) string {
				return writeCatalogStorageDevices(t, "cpu_dram: {read_bandwidth: 1.0}\n")
			},
			frags: []string{"kv-cpu-blocks", "malformed", "base_latency"},
		},
		{
			name: "class_absent",
			setup: func(t *testing.T) string {
				return writeCatalogStorageDevices(t,
					"nvme_gen4: {read_bandwidth: 7.0e3, write_bandwidth: 5.0e3, base_latency: 80.0}\n")
			},
			frags: []string{legacyKVTransferDeviceClass, "nvme_gen4", "--kv-transfer-bandwidth"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := loadLegacyKVTransferDevice(tc.setup(t))
			if err == nil {
				t.Fatal("expected a refusal")
			}
			for _, frag := range tc.frags {
				if !strings.Contains(err.Error(), frag) {
					t.Errorf("diagnostic must mention %q, got: %v", frag, err)
				}
			}
		})
	}
	// Positive control: the committed test catalog resolves.
	dev, err := loadLegacyKVTransferDevice(filepath.Join("..", "testdata", "catalog"))
	if err != nil {
		t.Fatalf("committed test catalog must resolve %q: %v", legacyKVTransferDeviceClass, err)
	}
	if dev.ReadBandwidth <= 0 {
		t.Errorf("resolved %q must carry a positive read_bandwidth, got %v", legacyKVTransferDeviceClass, dev.ReadBandwidth)
	}
}

// ---------------------------------------------------------------------------
// CLI legs (BC-4 … BC-8)
// ---------------------------------------------------------------------------

const (
	kvTransferCLILegEnv     = "BLIS_KVXFER_CLI_LEG"
	kvTransferCLICatalogEnv = "BLIS_KVXFER_CLI_CATALOG"
	kvTransferCLIBWEnv      = "BLIS_KVXFER_CLI_BANDWIDTH"
	kvTransferCLICPUEnv     = "BLIS_KVXFER_CLI_CPUBLOCKS"
	kvTransferCLITraceEnv   = "BLIS_KVXFER_CLI_TRACE"
)

// newModelOnlyCatalog builds a catalog CLONE ROOT holding the reference model entry and the
// workload preset the legs use, but NO devices/ namespace at all. It is the "catalog that
// cannot price the transfer" fixture for BC-7 (still fine when the tier is disabled) and
// BC-8 (refused when it is enabled), and it is deliberately complete in every other respect
// so a failure is attributable to the missing device table.
func newModelOnlyCatalog(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	shortName := devicesCLIModel[strings.Index(devicesCLIModel, "/")+1:]
	copyInto := func(relDir, file string) {
		src := filepath.Join("..", "testdata", "catalog", relDir, file)
		content, err := os.ReadFile(src)
		if err != nil {
			t.Fatalf("read test catalog file %s: %v", src, err)
		}
		dstDir := filepath.Join(root, relDir)
		if err := os.MkdirAll(dstDir, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", dstDir, err)
		}
		if err := os.WriteFile(filepath.Join(dstDir, file), content, 0o644); err != nil {
			t.Fatalf("write %s: %v", filepath.Join(dstDir, file), err)
		}
	}
	copyInto(filepath.Join(catalogModelsSubdir, shortName), hfConfigFile)
	copyInto(catalogWorkloadsSubdir, "chatbot.yaml")
	return root
}

// formatFloatForFlag renders a rate as a CLI flag argument without losing precision, so a
// control leg really runs at the rate the test computed.
func formatFloatForFlag(v float64) string {
	return strconv.FormatFloat(v, 'g', -1, 64)
}

// kvTransferCLIBaseArgs are the flags `run` and `replay` share. The GPU tier is
// deliberately tight so the legacy CPU tier actually carries traffic (blocks are mirrored to
// CPU and reloaded from it); without that traffic every comparison below would be vacuous,
// which is what the residual-free control leg in BC-4 proves is not the case.
func kvTransferCLIBaseArgs(catalog string) []string {
	return []string{
		"--model", devicesCLIModel,
		"--hardware", "H100", "--tp", "1",
		"--seed", "42",
		"--defaults-filepath", "../defaults.yaml",
		"--catalog", catalog,
		"--total-kv-blocks", "300",
		// INV-13 is stated for identical flags INCLUDING --horizon: run defaults the horizon
		// from the workload while replay defaults it from the trace, so an unset --horizon
		// makes run and replay differ for reasons unrelated to this change. Pinning it keeps
		// the parity leg a test of the derivation.
		"--horizon", "100000000",
	}
}

// kvTransferCLIWorkloadArgs are the `run`-only workload flags (replay takes its workload
// from the trace, and rejects these).
var kvTransferCLIWorkloadArgs = []string{"--workload", "chatbot", "--rate", "200", "--num-requests", "60"}

// kvTransferCLISubprocess executes the leg named by the environment, if any.
func kvTransferCLISubprocess() bool {
	leg := os.Getenv(kvTransferCLILegEnv)
	if leg == "" {
		return false
	}
	catalog := os.Getenv(kvTransferCLICatalogEnv)
	tracePrefix := os.Getenv(kvTransferCLITraceEnv)
	cpuBlocks := os.Getenv(kvTransferCLICPUEnv)
	if cpuBlocks == "" {
		cpuBlocks = "300"
	}

	args := kvTransferCLIBaseArgs(catalog)
	args = append(args, "--kv-cpu-blocks", cpuBlocks)
	if bw := os.Getenv(kvTransferCLIBWEnv); bw != "" {
		args = append(args, "--kv-transfer-bandwidth", bw)
	}
	switch leg {
	case "run", "run-export":
		args = append([]string{"run"}, append(args, kvTransferCLIWorkloadArgs...)...)
		if leg == "run-export" {
			args = append(args, "--trace-output", tracePrefix)
		}
	case "replay":
		args = append([]string{"replay",
			"--trace-header", tracePrefix + ".yaml",
			"--trace-data", tracePrefix + ".csv"}, args...)
	default:
		os.Exit(2)
	}

	rootCmd.SetArgs(args)
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
	os.Exit(0)
	return true
}

// runKVTransferCLILeg re-execs this test binary as the named leg.
func runKVTransferCLILeg(t *testing.T, testName, leg, catalog, bandwidth, cpuBlocks, tracePrefix string) (stdout, stderr string, err error) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^"+testName+"$")
	cmd.Env = append(os.Environ(),
		kvTransferCLILegEnv+"="+leg,
		kvTransferCLICatalogEnv+"="+catalog,
		kvTransferCLIBWEnv+"="+bandwidth,
		kvTransferCLICPUEnv+"="+cpuBlocks,
		kvTransferCLITraceEnv+"="+tracePrefix,
		// Neutralize any ambient catalog so a leg can only use the one it was handed.
		catalogEnvVar+"=",
	)
	var out, errBuf bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errBuf
	err = cmd.Run()
	return out.String(), errBuf.String(), err
}

// TestRunCmd_LegacyKVTransfer_DerivationReproducesRetiredDefault is BC-4 and BC-5 at the
// real CLI boundary — the regression pin the issue asks for, on a config that enables the
// legacy tier.
//
// Three legs over identical flags but for the transfer rate:
//
//	derived  (no --kv-transfer-bandwidth)                  ⇒ the catalog-derived rate
//	explicit (--kv-transfer-bandwidth 100.0)               ⇒ the retired default
//	nominal  (--kv-transfer-bandwidth <rated, no residual>) ⇒ the control
//
// derived == explicit is value preservation (R2). derived != nominal is what makes that
// non-vacuous: it shows this run's stdout DOES move with the transfer rate, so the first
// comparison is not two runs that both ignore it.
func TestRunCmd_LegacyKVTransfer_DerivationReproducesRetiredDefault(t *testing.T) {
	if kvTransferCLISubprocess() {
		return
	}
	const name = "TestRunCmd_LegacyKVTransfer_DerivationReproducesRetiredDefault"
	catalog := filepath.Join("..", "testdata", "catalog")

	derived, errOut, err := runKVTransferCLILeg(t, name, "run", catalog, "", "300", "")
	if err != nil {
		t.Fatalf("derived leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, derived, errOut)
	}
	if !strings.Contains(derived, "completed_requests") {
		t.Fatalf("non-vacuity: the derived leg produced no metrics:\n%s", derived)
	}

	explicit, errOut, err := runKVTransferCLILeg(t, name, "run", catalog, "100.0", "300", "")
	if err != nil {
		t.Fatalf("explicit-override leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, explicit, errOut)
	}
	if derived != explicit {
		t.Errorf("an enabled legacy run must reproduce the retired --kv-transfer-bandwidth=%v cost "+
			"when the rate is derived from the catalog (R2 value preservation)\nderived:\n%s\nexplicit:\n%s",
			retiredKVTransferBandwidthDefault, derived, explicit)
	}

	// Control: the residual-free RATED rate (cpu_dram ÷ reference KV bytes/token) must
	// produce different output, or the comparison above proves nothing.
	nominal := referenceCPUDRAM().ReadBandwidth / referenceKVBytesPerToken
	control, errOut, err := runKVTransferCLILeg(t, name, "run", catalog, formatFloatForFlag(nominal), "300", "")
	if err != nil {
		t.Fatalf("nominal-rate control leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, control, errOut)
	}
	if control == derived {
		t.Error("non-vacuity: this run's stdout does not move with --kv-transfer-bandwidth, so the " +
			"value-preservation comparison above is vacuous — tighten the GPU tier until the CPU " +
			"tier carries traffic")
	}
}

// TestReplayCmd_LegacyKVTransfer_MatchesRunDerivation is BC-6 (INV-13). The flags live in
// the shared registerSimConfigFlags and the rate is decided by the one
// resolveLegacyKVTransferBandwidth, so run and replay must derive the same number; the
// trace header does not carry it, which is exactly why a shared derivation is the only
// thing keeping them together.
func TestReplayCmd_LegacyKVTransfer_MatchesRunDerivation(t *testing.T) {
	if kvTransferCLISubprocess() {
		return
	}
	const name = "TestReplayCmd_LegacyKVTransfer_MatchesRunDerivation"
	catalog := filepath.Join("..", "testdata", "catalog")
	tracePrefix := filepath.Join(t.TempDir(), "legacy-kv")

	runOut, errOut, err := runKVTransferCLILeg(t, name, "run-export", catalog, "", "300", tracePrefix)
	if err != nil {
		t.Fatalf("run-export leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, runOut, errOut)
	}
	replayOut, errOut, err := runKVTransferCLILeg(t, name, "replay", catalog, "", "300", tracePrefix)
	if err != nil {
		t.Fatalf("replay leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, replayOut, errOut)
	}
	if !strings.Contains(replayOut, "completed_requests") {
		t.Fatalf("non-vacuity: replay produced no metrics:\n%s", replayOut)
	}
	if runOut != replayOut {
		t.Errorf("run and replay must derive the same legacy transfer rate (INV-13)\nrun:\n%s\nreplay:\n%s",
			runOut, replayOut)
	}

	// Non-vacuity for the replay side specifically: replaying the same trace at the
	// residual-free rated rate must differ, so the identity above is not two replays that
	// both ignore --kv-transfer-bandwidth.
	nominal := referenceCPUDRAM().ReadBandwidth / referenceKVBytesPerToken
	control, errOut, err := runKVTransferCLILeg(t, name, "replay", catalog, formatFloatForFlag(nominal), "300", tracePrefix)
	if err != nil {
		t.Fatalf("nominal-rate replay control leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, control, errOut)
	}
	if control == replayOut {
		t.Error("non-vacuity: replay's stdout does not move with --kv-transfer-bandwidth, so the " +
			"parity comparison above is vacuous")
	}
}

// TestRunCmd_LegacyKVTransfer_InertWithoutCPUBlocks is BC-7: the catalog device table is a
// LAZY dependency. A catalog with models/ but no devices/ namespace still runs when the
// legacy tier is disabled — which is every committed run — and its stdout is byte-identical
// to the same run against the full catalog (INV-6).
func TestRunCmd_LegacyKVTransfer_InertWithoutCPUBlocks(t *testing.T) {
	if kvTransferCLISubprocess() {
		return
	}
	const name = "TestRunCmd_LegacyKVTransfer_InertWithoutCPUBlocks"
	full := filepath.Join("..", "testdata", "catalog")
	deviceless := newModelOnlyCatalog(t)

	viaFull, errOut, err := runKVTransferCLILeg(t, name, "run", full, "", "0", "")
	if err != nil {
		t.Fatalf("disabled leg against the full catalog failed: %v\nstdout:\n%s\nstderr:\n%s", err, viaFull, errOut)
	}
	viaDeviceless, errOut, err := runKVTransferCLILeg(t, name, "run", deviceless, "", "0", "")
	if err != nil {
		t.Fatalf("a run with --kv-cpu-blocks 0 must not require a catalog device table: %v\nstdout:\n%s\nstderr:\n%s",
			err, viaDeviceless, errOut)
	}
	if !strings.Contains(viaFull, "completed_requests") {
		t.Fatalf("non-vacuity: the disabled leg produced no metrics:\n%s", viaFull)
	}
	if viaFull != viaDeviceless {
		t.Errorf("a disabled legacy tier must be byte-identical whether or not the catalog has a "+
			"device table (INV-6)\nfull:\n%s\ndeviceless:\n%s", viaFull, viaDeviceless)
	}
}

// TestRunCmd_LegacyKVTransfer_MissingDeviceTableIsRefused is BC-8: enabling the legacy tier
// against a catalog that cannot supply cpu_dram is a hard error naming the path and the
// override, never a silent fall back to the retired constant or to zero physics. The
// override leg is the paired escape hatch — the SAME catalog runs once a rate is supplied,
// so the refusal is attributable to the derivation and not to the catalog being unusable.
func TestRunCmd_LegacyKVTransfer_MissingDeviceTableIsRefused(t *testing.T) {
	if kvTransferCLISubprocess() {
		return
	}
	const name = "TestRunCmd_LegacyKVTransfer_MissingDeviceTableIsRefused"
	deviceless := newModelOnlyCatalog(t)

	out, errOut, err := runKVTransferCLILeg(t, name, "run", deviceless, "", "300", "")
	if err == nil {
		t.Fatalf("enabling --kv-cpu-blocks against a catalog with no device table must be refused;\nstdout:\n%s", out)
	}
	for _, frag := range []string{catalogStorageDevicesRelPath, "--kv-transfer-bandwidth"} {
		if !strings.Contains(errOut, frag) {
			t.Errorf("the refusal must name %q, got:\n%s", frag, errOut)
		}
	}

	overridden, errOut, err := runKVTransferCLILeg(t, name, "run", deviceless, "100.0", "300", "")
	if err != nil {
		t.Fatalf("--kv-transfer-bandwidth must make the same catalog usable (the documented escape "+
			"hatch): %v\nstdout:\n%s\nstderr:\n%s", err, overridden, errOut)
	}
	if !strings.Contains(overridden, "completed_requests") {
		t.Fatalf("non-vacuity: the override leg produced no metrics:\n%s", overridden)
	}
}

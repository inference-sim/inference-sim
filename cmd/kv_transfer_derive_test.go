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

	"github.com/spf13/pflag"

	"github.com/inference-sim/inference-sim/sim/latency"
)

// #1819 (R2G3b-sim): the legacy single-CPU-tier CPU↔GPU transfer cost is DERIVED from the
// catalog cpu_dram device fact instead of being authored as a blocks/tick flag default.
//
// The contracts pinned here:
//
//	BC-1  the derivation implements ticks = per_block_bytes / bandwidth + base_latency;
//	      block_size_tokens cancels from the bandwidth rate, while catalog base_latency is
//	      rounded up to whole simulator ticks
//	BC-2  at the named reference deployment the bandwidth term remains EXACTLY the retired
//	      100.0 rate and the newly required catalog latency adds exactly one tick per reload
//	BC-3  degenerate inputs are refused, never silently resolved to zero physics (R1/R9)
//	BC-4  CLI: the fully derived path charges both catalog components
//	BC-5  bandwidth and base-latency overrides are independent, including an explicit
//	      base-latency 0; both overrides together avoid any catalog device read
//	BC-6  INV-13: run and replay resolve all derived/mixed/fully-overridden cases identically
//	BC-7  INV-6 / lazy dependency: with --kv-cpu-blocks 0 nothing is derived and no
//	      catalog device table is required
//	BC-8  a catalog that cannot supply cpu_dram is refused naming the path and both flags;
//	      supplying both overrides is the only catalog-free enabled path (R1)
//	BC-9  derivation is selected by OMITTING --kv-transfer-bandwidth: a SUPPLIED 0 is
//	      refused, and the flag help says so rather than advertising "0 = derive"
//	BC-10 a derived rate that is positive and finite but so small that the library's
//	      int64(ceil(block_tokens / rate)) would overflow is REFUSED here, not shipped into
//	      the DES as a negative latency (#1840 review F3)
//	BC-11 that bound is a property of the RATE, not of its provenance: a SUPPLIED
//	      --kv-transfer-bandwidth override is held to it too, on both run and replay, so the
//	      override path is not a way around BC-10 (#1840 review N1)
//	BC-12 catalog base latency must be nonnegative, finite, and representable; the combined
//	      and cumulative int64 additions are protected at the shared sim/kv boundary
//
// The static half — "a physics literal cannot come back into a cmd/ flag default", plus the
// LoRA defaults-vs-registry drift guard — lives in physics_literal_guard_test.go.

// ---------------------------------------------------------------------------
// Frozen goldens
// ---------------------------------------------------------------------------

// retiredKVTransferBandwidthDefault is the flag default #1819 removed, kept here as a
// FROZEN golden rather than in production code. It is the bandwidth number the conversion
// must reproduce; the independently derived catalog base latency intentionally changes the
// total enabled-path cost. Regenerating this from the derivation would make BC-2 vacuous.
const retiredKVTransferBandwidthDefault = 100.0

// referenceKVBytesPerToken is KVBytesPerToken(qwen3-14b, TP=1) — the anchor of
// legacyKVTransferResidual. 40 layers × 2 (K+V) × 128 head_dim × 8 kv_heads × 2 bytes
// (bf16), ÷ TP=1. Also FROZEN: the committed test catalog carries that model config, and
// TestDeriveLegacyKVTransferRate_ReferenceMatchesCommittedCatalog checks the two agree, so
// a catalog edit that moved the reference would be caught rather than silently re-anchoring
// the residual.
const referenceKVBytesPerToken = 163840.0

// referenceBlockSizeTokens is --block-size-in-tokens at the reference deployment. The derived
// RATE does not depend on it (block_size_tokens cancels — BC-1); it is passed only so the
// int64 tick-budget bound BC-10 checks has a block size to check against.
const referenceBlockSizeTokens = 16

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
	got, err := deriveLegacyKVTransferRate(referenceCPUDRAM(), referenceKVBytesPerToken, referenceBlockSizeTokens)
	if err != nil {
		t.Fatalf("deriveLegacyKVTransferRate at the reference: %v", err)
	}
	if got != retiredKVTransferBandwidthDefault {
		t.Errorf("derived rate at the reference deployment = %v, want exactly %v "+
			"(the retired --kv-transfer-bandwidth default); the R2G3b residual %v no longer "+
			"reproduces the retired bandwidth term, which is a physics change and must be argued on its own",
			got, retiredKVTransferBandwidthDefault, legacyKVTransferResidual)
	}
}

func TestDeriveLegacyKVTransferBaseLatency_UsesWholeTicks(t *testing.T) {
	for _, tc := range []struct {
		name    string
		latency float64
		want    int64
		wantErr bool
	}{
		{"zero", 0, 0, false},
		{"whole_tick", 1, 1, false},
		{"fraction_rounds_up", 1.01, 2, false},
		{"negative", -1, 0, true},
		{"nan", math.NaN(), 0, true},
		{"positive_inf", math.Inf(1), 0, true},
		{"int64_overflow", float64(math.MaxInt64), 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dev := referenceCPUDRAM()
			dev.BaseLatency = tc.latency
			got, err := deriveLegacyKVTransferBaseLatency(dev)
			if tc.wantErr {
				if err == nil || !strings.Contains(err.Error(), "base_latency") {
					t.Fatalf("derive base_latency=%v: want a base_latency refusal, got %d, %v", tc.latency, got, err)
				}
				return
			}
			if err != nil || got != tc.want {
				t.Fatalf("derive base_latency=%v: got %d, %v; want %d", tc.latency, got, err, tc.want)
			}
		})
	}
}

func TestLegacyKVTransferReferenceCost_AddsCatalogLatency(t *testing.T) {
	rate, err := deriveLegacyKVTransferRate(referenceCPUDRAM(), referenceKVBytesPerToken, referenceBlockSizeTokens)
	if err != nil {
		t.Fatal(err)
	}
	baseLatency, err := deriveLegacyKVTransferBaseLatency(referenceCPUDRAM())
	if err != nil {
		t.Fatal(err)
	}
	bandwidthTicks := int64(math.Ceil(float64(referenceBlockSizeTokens) / rate))
	if bandwidthTicks != 1 || baseLatency != 1 || bandwidthTicks+baseLatency != 2 {
		t.Fatalf("reference charge = %d bandwidth + %d base = %d ticks; want 1 + 1 = 2",
			bandwidthTicks, baseLatency, bandwidthTicks+baseLatency)
	}
}

func TestLegacyKVTransferCombinedBudget_RejectsAdditionOverflow(t *testing.T) {
	if err := legacyKVTransferCombinedBudgetError(16, 16, math.MaxInt64-1); err != nil {
		t.Fatalf("a combined charge equal to MaxInt64 must fit: %v", err)
	}
	if err := legacyKVTransferCombinedBudgetError(16, 16, math.MaxInt64); err == nil {
		t.Fatal("MaxInt64 base latency plus a positive transfer charge must be refused")
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
	if dev.BaseLatency != referenceCPUDRAM().BaseLatency {
		t.Errorf("committed catalog cpu_dram base_latency = %v, but the frozen reference is %v",
			dev.BaseLatency, referenceCPUDRAM().BaseLatency)
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
		for _, blockSizeTokens := range []int64{1, 16, 32, 128} {
			rate, err := deriveLegacyKVTransferRate(dev, perToken, blockSizeTokens)
			if err != nil {
				t.Fatalf("derive(perToken=%v, blockSize=%d): %v", perToken, blockSizeTokens, err)
			}
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
	base, err := deriveLegacyKVTransferRate(dev, referenceKVBytesPerToken, referenceBlockSizeTokens)
	if err != nil {
		t.Fatalf("derive(base): %v", err)
	}
	quarter, err := deriveLegacyKVTransferRate(dev, referenceKVBytesPerToken/4, referenceBlockSizeTokens)
	if err != nil {
		t.Fatalf("derive(quarter): %v", err)
	}
	if quarter != base*4 {
		t.Errorf("quartering per-token KV bytes must quadruple the rate: got %v, want %v", quarter, base*4)
	}
	// And doubling the bus doubles the rate.
	faster := dev
	faster.ReadBandwidth *= 2
	doubled, err := deriveLegacyKVTransferRate(faster, referenceKVBytesPerToken, referenceBlockSizeTokens)
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
		name      string
		dev       kvOffloadDevice
		perToken  float64
		blockSize int64
		wantFrag  string
	}{
		{"per_token_zero", good, 0, referenceBlockSizeTokens, "per-token KV bytes"},
		{"per_token_negative", good, -1, referenceBlockSizeTokens, "per-token KV bytes"},
		{"per_token_nan", good, math.NaN(), referenceBlockSizeTokens, "per-token KV bytes"},
		{"per_token_inf", good, math.Inf(1), referenceBlockSizeTokens, "per-token KV bytes"},
		{"bandwidth_zero", kvOffloadDevice{}, referenceKVBytesPerToken, referenceBlockSizeTokens, "read_bandwidth"},
		{"bandwidth_negative", kvOffloadDevice{ReadBandwidth: -1}, referenceKVBytesPerToken, referenceBlockSizeTokens, "read_bandwidth"},
		{"bandwidth_nan", kvOffloadDevice{ReadBandwidth: math.NaN()}, referenceKVBytesPerToken, referenceBlockSizeTokens, "read_bandwidth"},
		{"bandwidth_inf", kvOffloadDevice{ReadBandwidth: math.Inf(1)}, referenceKVBytesPerToken, referenceBlockSizeTokens, "read_bandwidth"},
		{"block_size_zero", good, referenceKVBytesPerToken, 0, "block size in tokens"},
		{"block_size_negative", good, referenceKVBytesPerToken, -16, "block size in tokens"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rate, err := deriveLegacyKVTransferRate(tc.dev, tc.perToken, tc.blockSize)
			if err == nil {
				t.Fatalf("expected a refusal, got rate %v", rate)
			}
			if !strings.Contains(err.Error(), tc.wantFrag) {
				t.Errorf("refusal must name %q, got: %v", tc.wantFrag, err)
			}
		})
	}
}

// TestDeriveLegacyKVTransferRate_RefusesRatesThatOverflowTheTickBudget is BC-10, and it is a
// simulation-INTEGRITY contract rather than a tidiness one.
//
// "Positive and finite" is not a sufficient condition on the derived rate. sim/kv.TieredKVCache
// charges `int64(math.Ceil(block_tokens / rate))` per reloaded block, and a Go float→int64
// conversion whose value does not fit is IMPLEMENTATION-DEFINED: amd64 wraps it to MinInt64 (a
// large NEGATIVE latency that moves the clock the wrong way), arm64 saturates it to MaxInt64 (a
// ~292,000-year latency) — either way a garbage charge, not the true one. So a catalog
// read_bandwidth of 1e-300 — positive, finite, and accepted by every other check — yields a
// rate around 5e-303 that passes the pre-existing guards and then overflows this conversion.
// The test asserts the refusal AND demonstrates the arithmetic it prevents; the demonstration is
// stated in the FLOAT domain (the ceil'd charge exceeds the int64 range) so it holds on both
// architectures rather than relying on how one of them realises the undefined conversion — so the
// bound cannot later be deleted as "defensive".
func TestDeriveLegacyKVTransferRate_RefusesRatesThatOverflowTheTickBudget(t *testing.T) {
	for _, tc := range []struct {
		name      string
		bandwidth float64
		blockSize int64
	}{
		{"denormal_bandwidth", 1e-300, referenceBlockSizeTokens},
		{"tiny_bandwidth", 1e-20, referenceBlockSizeTokens},
		{"large_block_size", 1e-15, 128},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dev := referenceCPUDRAM()
			dev.ReadBandwidth = tc.bandwidth

			// The rate the unbounded conversion would have returned is positive and finite,
			// so the pre-existing guards do NOT catch this — that is the whole finding.
			nominal := dev.ReadBandwidth / referenceKVBytesPerToken * legacyKVTransferResidual
			if nominal <= 0 || math.IsNaN(nominal) || math.IsInf(nominal, 0) {
				t.Fatalf("non-vacuity: this case must reach the new bound with a positive finite "+
					"rate, got %v", nominal)
			}

			rate, err := deriveLegacyKVTransferRate(dev, referenceKVBytesPerToken, tc.blockSize)
			if err == nil {
				t.Fatalf("a rate of %v tokens/tick must be refused: it charges %v ticks for one "+
					"%d-token block", rate, math.Ceil(float64(tc.blockSize)/rate), tc.blockSize)
			}
			for _, frag := range []string{"int64", "--kv-transfer-bandwidth", legacyKVTransferDeviceClass} {
				if !strings.Contains(err.Error(), frag) {
					t.Errorf("refusal must mention %q, got: %v", frag, err)
				}
			}

			// What the refusal prevents: the library's own int64(math.Ceil(...)) conversion is
			// undefined here because the ceil'd charge does not fit int64. Assert that
			// out-of-range condition in the FLOAT domain — performing the conversion and
			// checking its sign would be non-portable (amd64 wraps to MinInt64, arm64 saturates
			// to MaxInt64), which is the very undefinedness the bound exists to refuse.
			overflowTicks := math.Ceil(float64(tc.blockSize) / nominal)
			if overflowTicks <= float64(math.MaxInt64) {
				t.Fatalf("non-vacuity: the unguarded charge must exceed the int64 range (an "+
					"overflow), got %v — this case no longer demonstrates the failure it guards",
					overflowTicks)
			}
		})
	}

	// The control: the reference deployment, and a generously slow but plausible bus, are NOT
	// refused. A bound that rejects real catalogs would be removed rather than fixed.
	for _, bandwidth := range []float64{2.0e4, 1.0, 1e-6} {
		dev := referenceCPUDRAM()
		dev.ReadBandwidth = bandwidth
		if _, err := deriveLegacyKVTransferRate(dev, referenceKVBytesPerToken, 128); err != nil {
			t.Errorf("read_bandwidth=%v must still be accepted: %v", bandwidth, err)
		}
	}
}

// TestLegacyKVTransferTickBudget_IsAPropertyOfTheRateNotItsProvenance is the unit half of
// BC-11, and it states the law BC-10's implementation originally got half right.
//
// The int64 tick budget is a constraint on the NUMBER handed to sim/kv.TieredKVCache. Where
// that number came from — the catalog conversion or an operator's --kv-transfer-bandwidth —
// cannot change whether `int64(math.Ceil(block_tokens / rate))` fits. So the bound is asserted
// on the shared checker directly, and the same rate is shown to be refused whether it arrives
// via the derivation or verbatim from the flag: a test that only exercised one path is how the
// override hole survived three reviews (#1840 review N1).
func TestLegacyKVTransferTickBudget_IsAPropertyOfTheRateNotItsProvenance(t *testing.T) {
	for _, tc := range []struct {
		name      string
		rate      float64
		blockSize int64
		wantErr   bool
	}{
		// Refused: positive, finite, and small enough that the library's conversion wraps.
		{"denormal_rate", 1e-300, referenceBlockSizeTokens, true},
		{"tiny_rate", 1e-20, referenceBlockSizeTokens, true},
		{"large_block_size", 1e-15, 128, true},
		// Accepted: the reference, and rates far slower than any real bus.
		{"reference_rate", retiredKVTransferBandwidthDefault, referenceBlockSizeTokens, false},
		{"nominal_rate", referenceCPUDRAM().ReadBandwidth / referenceKVBytesPerToken, referenceBlockSizeTokens, false},
		{"very_slow_but_representable", 1e-12, referenceBlockSizeTokens, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := legacyKVTransferTickBudgetError(tc.rate, tc.blockSize)
			if tc.wantErr && err == nil {
				t.Fatalf("rate %v with a %d-token block charges %v ticks and must be refused",
					tc.rate, tc.blockSize, math.Ceil(float64(tc.blockSize)/tc.rate))
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("rate %v with a %d-token block is representable and must be accepted: %v",
					tc.rate, tc.blockSize, err)
			}
			if !tc.wantErr {
				// Non-vacuity for the accepted half: the library's own conversion really does
				// produce a positive charge, so "accepted" is not hiding the same overflow.
				if charged := int64(math.Ceil(float64(tc.blockSize) / tc.rate)); charged <= 0 {
					t.Fatalf("an accepted rate must charge a positive number of ticks, got %d", charged)
				}
				return
			}
			if !strings.Contains(err.Error(), "int64") {
				t.Errorf("the refusal must name the int64 budget it protects, got: %v", err)
			}
			// The derived path must reach the SAME refusal for a catalog bandwidth that
			// produces this rate — the two paths are checked against one bound, not two.
			dev := referenceCPUDRAM()
			dev.ReadBandwidth = tc.rate * referenceKVBytesPerToken / legacyKVTransferResidual
			if dev.ReadBandwidth > 0 && !math.IsInf(dev.ReadBandwidth, 0) {
				if _, derivedErr := deriveLegacyKVTransferRate(dev, referenceKVBytesPerToken, tc.blockSize); derivedErr == nil {
					t.Errorf("the derived path must refuse the same rate %v that the bound refuses", tc.rate)
				}
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
			frags: []string{"kv-cpu-blocks", catalogStorageDevicesRelPath, "--kv-transfer-bandwidth", "--kv-transfer-base-latency"},
		},
		{
			name: "malformed_table",
			setup: func(t *testing.T) string {
				return writeCatalogStorageDevices(t, "cpu_dram: {read_bandwidth: 1.0}\n")
			},
			// The escape hatch too: the function contract promises EVERY failure names it,
			// and this branch used to be the one that did not (#1840 review F5).
			frags: []string{"kv-cpu-blocks", "malformed", "base_latency", "--kv-transfer-bandwidth", "--kv-transfer-base-latency"},
		},
		{
			name: "class_absent",
			setup: func(t *testing.T) string {
				return writeCatalogStorageDevices(t,
					"nvme_gen4: {read_bandwidth: 7.0e3, write_bandwidth: 5.0e3, base_latency: 80.0}\n")
			},
			frags: []string{legacyKVTransferDeviceClass, "nvme_gen4", "--kv-transfer-bandwidth", "--kv-transfer-base-latency"},
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
	kvTransferCLIBaseLatEnv = "BLIS_KVXFER_CLI_BASE_LATENCY"
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

// newLegacyKVTransferCatalog adds a caller-selected cpu_dram row to the otherwise complete
// model/workload fixture. It lets CLI tests vary device physics without mutating the committed
// catalog or accidentally testing a catalog that cannot resolve the model.
func newLegacyKVTransferCatalog(t *testing.T, cpuDRAM string) string {
	t.Helper()
	root := newModelOnlyCatalog(t)
	devicesDir := filepath.Join(root, catalogDevicesSubdir)
	if err := os.MkdirAll(devicesDir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", devicesDir, err)
	}
	body := "cpu_dram: " + cpuDRAM + "\n"
	path := filepath.Join(devicesDir, catalogStorageDevicesFile)
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
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
	if baseLatency := os.Getenv(kvTransferCLIBaseLatEnv); baseLatency != "" {
		args = append(args, "--kv-transfer-base-latency", baseLatency)
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
func runKVTransferCLILeg(t *testing.T, testName, leg, catalog, bandwidth, baseLatency, cpuBlocks, tracePrefix string) (stdout, stderr string, err error) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^"+testName+"$")
	cmd.Env = append(os.Environ(),
		kvTransferCLILegEnv+"="+leg,
		kvTransferCLICatalogEnv+"="+catalog,
		kvTransferCLIBWEnv+"="+bandwidth,
		kvTransferCLIBaseLatEnv+"="+baseLatency,
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

// TestRunCmd_LegacyKVTransfer_IndependentOverrides is BC-4/BC-5 at the real CLI boundary. At the
// reference catalog, derived bandwidth=100 and derived base latency=1. Supplying either one
// alone must leave the other derived, supplying both reproduces the same cost, and explicitly
// supplying base latency 0 must disable the additive catalog latency rather than mean "derive".
func TestRunCmd_LegacyKVTransfer_IndependentOverrides(t *testing.T) {
	if kvTransferCLISubprocess() {
		return
	}
	const name = "TestRunCmd_LegacyKVTransfer_IndependentOverrides"
	catalog := filepath.Join("..", "testdata", "catalog")

	derived, errOut, err := runKVTransferCLILeg(t, name, "run", catalog, "", "", "300", "")
	if err != nil {
		t.Fatalf("derived leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, derived, errOut)
	}
	if !strings.Contains(derived, "completed_requests") {
		t.Fatalf("non-vacuity: the derived leg produced no metrics:\n%s", derived)
	}

	for _, tc := range []struct {
		name, bandwidth, baseLatency string
	}{
		{"bandwidth_only", "100", ""},
		{"base_latency_only", "", "1"},
		{"both", "100", "1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, errOut, err := runKVTransferCLILeg(t, name, "run", catalog,
				tc.bandwidth, tc.baseLatency, "300", "")
			if err != nil {
				t.Fatalf("override leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, got, errOut)
			}
			if got != derived {
				t.Errorf("independent overrides matching the catalog must preserve output\nderived:\n%s\noverridden:\n%s",
					derived, got)
			}
		})
	}

	// An explicit zero is a real base-latency override. This is also the expected +1 tick
	// behavior delta from the retired enabled configuration at the reference deployment.
	withoutBaseLatency, errOut, err := runKVTransferCLILeg(t, name, "run", catalog, "100", "0", "300", "")
	if err != nil {
		t.Fatalf("explicit-zero base-latency leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, withoutBaseLatency, errOut)
	}
	if withoutBaseLatency == derived {
		t.Error("explicit --kv-transfer-base-latency=0 must differ from the catalog-derived 1-tick cost; " +
			"otherwise the override or the test traffic is inert")
	}
}

// TestReplayCmd_LegacyKVTransfer_MatchesRunResolution is BC-6 (INV-13) across derived,
// mixed-override, and fully overridden configurations. The trace header carries neither
// component, so sharing one resolver is what keeps run and replay together.
func TestReplayCmd_LegacyKVTransfer_MatchesRunResolution(t *testing.T) {
	if kvTransferCLISubprocess() {
		return
	}
	const name = "TestReplayCmd_LegacyKVTransfer_MatchesRunResolution"
	catalog := filepath.Join("..", "testdata", "catalog")
	for _, tc := range []struct {
		name, bandwidth, baseLatency string
	}{
		{"derived", "", ""},
		{"bandwidth_only", "100", ""},
		{"base_latency_only", "", "1"},
		{"both", "100", "1"},
		{"explicit_zero_latency", "100", "0"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tracePrefix := filepath.Join(t.TempDir(), "legacy-kv")
			runOut, errOut, err := runKVTransferCLILeg(t, name, "run-export", catalog,
				tc.bandwidth, tc.baseLatency, "300", tracePrefix)
			if err != nil {
				t.Fatalf("run-export leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, runOut, errOut)
			}
			replayOut, errOut, err := runKVTransferCLILeg(t, name, "replay", catalog,
				tc.bandwidth, tc.baseLatency, "300", tracePrefix)
			if err != nil {
				t.Fatalf("replay leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, replayOut, errOut)
			}
			if !strings.Contains(replayOut, "completed_requests") {
				t.Fatalf("non-vacuity: replay produced no metrics:\n%s", replayOut)
			}
			if runOut != replayOut {
				t.Errorf("run and replay must resolve the same legacy transfer cost (INV-13)\nrun:\n%s\nreplay:\n%s",
					runOut, replayOut)
			}
		})
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

	viaFull, errOut, err := runKVTransferCLILeg(t, name, "run", full, "", "", "0", "")
	if err != nil {
		t.Fatalf("disabled leg against the full catalog failed: %v\nstdout:\n%s\nstderr:\n%s", err, viaFull, errOut)
	}
	viaDeviceless, errOut, err := runKVTransferCLILeg(t, name, "run", deviceless, "", "", "0", "")
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

// TestRunCmd_LegacyKVTransfer_MissingDeviceTableIsRefused is BC-8: each omitted component
// requires cpu_dram. Supplying only one override still needs the table for the other; supplying
// both is the escape hatch and performs no device-table read.
func TestRunCmd_LegacyKVTransfer_MissingDeviceTableIsRefused(t *testing.T) {
	if kvTransferCLISubprocess() {
		return
	}
	const name = "TestRunCmd_LegacyKVTransfer_MissingDeviceTableIsRefused"
	deviceless := newModelOnlyCatalog(t)

	for _, tc := range []struct {
		name, bandwidth, baseLatency string
	}{
		{"neither", "", ""},
		{"bandwidth_only", "100", ""},
		{"base_latency_only", "", "1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, errOut, err := runKVTransferCLILeg(t, name, "run", deviceless,
				tc.bandwidth, tc.baseLatency, "300", "")
			if err == nil {
				t.Fatalf("an omitted transfer component must require the device table; stdout:\n%s", out)
			}
			for _, frag := range []string{catalogStorageDevicesRelPath, "--kv-transfer-bandwidth", "--kv-transfer-base-latency"} {
				if !strings.Contains(errOut, frag) {
					t.Errorf("the refusal must name %q, got:\n%s", frag, errOut)
				}
			}
		})
	}

	overridden, errOut, err := runKVTransferCLILeg(t, name, "run", deviceless, "100.0", "1", "300", "")
	if err != nil {
		t.Fatalf("supplying both transfer overrides must make the same catalog usable without a "+
			"device-table read: %v\nstdout:\n%s\nstderr:\n%s", err, overridden, errOut)
	}
	if !strings.Contains(overridden, "completed_requests") {
		t.Fatalf("non-vacuity: the override leg produced no metrics:\n%s", overridden)
	}
}

// TestRunCmd_LegacyKVTransfer_SuppliedZeroIsRefused pins the distinction the flag help and the
// guides have to state precisely: DERIVATION IS SELECTED BY OMITTING THE FLAG, not by passing
// 0. Zero is only the REGISTERED default — it has to be, because a physics number must not
// live in a flag table (BC-G1) — but a SUPPLIED zero is an operator error, and swallowing it
// as "derive" would make a typo silently change the transfer cost.
//
// Both legs run against the SAME catalog and differ only in whether the flag is present, so
// the refusal is attributable to the supplied value and not to anything about the deployment.
// It is this behaviour the documentation wording is checked against (#1840 correction round 1,
// qa-review F2/G3).
func TestRunCmd_LegacyKVTransfer_SuppliedZeroIsRefused(t *testing.T) {
	if kvTransferCLISubprocess() {
		return
	}
	const name = "TestRunCmd_LegacyKVTransfer_SuppliedZeroIsRefused"
	catalog := filepath.Join("..", "testdata", "catalog")

	out, errOut, err := runKVTransferCLILeg(t, name, "run", catalog, "0", "", "300", "")
	if err == nil {
		t.Fatalf("--kv-transfer-bandwidth 0 must be refused as a supplied out-of-range override, "+
			"never read as a request to derive;\nstdout:\n%s", out)
	}
	for _, frag := range []string{"--kv-transfer-bandwidth", "when supplied"} {
		if !strings.Contains(errOut, frag) {
			t.Errorf("the refusal must name %q, got:\n%s", frag, errOut)
		}
	}

	// The paired control: omitting the flag on the same catalog derives and runs. Without it
	// the refusal above could just as well be a catalog or deployment failure.
	derived, errOut, err := runKVTransferCLILeg(t, name, "run", catalog, "", "", "300", "")
	if err != nil {
		t.Fatalf("omitting --kv-transfer-bandwidth must derive and run: %v\nstdout:\n%s\nstderr:\n%s",
			err, derived, errOut)
	}
	if !strings.Contains(derived, "completed_requests") {
		t.Fatalf("non-vacuity: the derived control leg produced no metrics:\n%s", derived)
	}
}

// TestLegacyKVTransfer_OverrideThatOverflowsTheTickBudgetIsRefused is the CLI half of BC-11,
// on BOTH run and replay.
//
// The finding it pins: BC-10 bounded the DERIVED rate, and resolvePolicies range-checks a
// supplied --kv-transfer-bandwidth for <= 0 / NaN / Inf — so `--kv-transfer-bandwidth 1e-300`
// satisfied every check on the books and was returned verbatim into sim/kv, where
// int64(ceil(16 / 1e-300)) wraps to MinInt64 and subtracts ~292 years from the clock. The
// override path is the operator-facing one and the escape hatch every other diagnostic in this
// file recommends, so a hole there is reachable from the documented happy path.
//
// Each refusal leg is paired with a leg that differs ONLY in the flag's value and completes,
// so the refusal is attributable to the rate and not to the deployment or the trace.
func TestLegacyKVTransfer_OverrideThatOverflowsTheTickBudgetIsRefused(t *testing.T) {
	if kvTransferCLISubprocess() {
		return
	}
	const name = "TestLegacyKVTransfer_OverrideThatOverflowsTheTickBudgetIsRefused"
	catalog := filepath.Join("..", "testdata", "catalog")
	tracePrefix := filepath.Join(t.TempDir(), "legacy-kv-overflow")

	// Non-vacuity, stated before the refusals: this really is a value resolvePolicies lets
	// through, and the library conversion really does overflow on it. If either stops being true
	// the test below is checking something else. The overflow is stated in the FLOAT domain (the
	// ceil'd charge exceeds the int64 range) rather than by converting and checking the sign,
	// which is implementation-defined (amd64 wraps to MinInt64, arm64 saturates to MaxInt64).
	const overflowing = "1e-300"
	if bw := 1e-300; bw <= 0 || math.IsNaN(bw) || math.IsInf(bw, 0) {
		t.Fatalf("non-vacuity: %s must pass resolvePolicies' range check", overflowing)
	}
	if overflowTicks := math.Ceil(float64(referenceBlockSizeTokens) / 1e-300); overflowTicks <= float64(math.MaxInt64) {
		t.Fatalf("non-vacuity: the unguarded charge for %s must exceed the int64 range (an overflow), got %v",
			overflowing, overflowTicks)
	}

	// The trace the replay leg needs, exported at the derived rate.
	exported, errOut, err := runKVTransferCLILeg(t, name, "run-export", catalog, "", "", "300", tracePrefix)
	if err != nil {
		t.Fatalf("run-export leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, exported, errOut)
	}

	for _, leg := range []string{"run", "replay"} {
		t.Run(leg, func(t *testing.T) {
			out, errOut, err := runKVTransferCLILeg(t, name, leg, catalog, overflowing, "0", "300", tracePrefix)
			if err == nil {
				t.Fatalf("%s: --kv-transfer-bandwidth %s must be refused — it overflows the int64 "+
					"tick budget in sim/kv and injects a negative transfer latency;\nstdout:\n%s",
					leg, overflowing, out)
			}
			// The diagnostic must blame the flag the operator set (the catalog is not at fault
			// here and may not even have been read) and name the constraint it violated.
			for _, frag := range []string{"--kv-transfer-bandwidth", "int64"} {
				if !strings.Contains(errOut, frag) {
					t.Errorf("%s: the refusal must name %q, got:\n%s", leg, frag, errOut)
				}
			}
			if strings.Contains(errOut, "Fix that catalog value") {
				t.Errorf("%s: the refusal must not blame the catalog for a value the operator "+
					"supplied, got:\n%s", leg, errOut)
			}

			// The paired control: the same leg, same catalog, same trace, a representable rate.
			ok, errOut, err := runKVTransferCLILeg(t, name, leg, catalog,
				formatFloatForFlag(retiredKVTransferBandwidthDefault), "0", "300", tracePrefix)
			if err != nil {
				t.Fatalf("%s: a representable override must still run: %v\nstdout:\n%s\nstderr:\n%s",
					leg, err, ok, errOut)
			}
			if !strings.Contains(ok, "completed_requests") {
				t.Fatalf("%s: non-vacuity: the control leg produced no metrics:\n%s", leg, ok)
			}
		})
	}
}

func TestLegacyKVTransfer_CatalogBaseLatencyValidation(t *testing.T) {
	if kvTransferCLISubprocess() {
		return
	}
	const name = "TestLegacyKVTransfer_CatalogBaseLatencyValidation"
	for _, tc := range []struct {
		name, baseLatency string
	}{
		{"negative", "-1"},
		{"nan", ".nan"},
		{"infinite", ".inf"},
		{"int64_overflow", "9.223372036854776e18"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			catalog := newLegacyKVTransferCatalog(t, "{read_bandwidth: 2.0e4, write_bandwidth: 2.0e4, base_latency: "+tc.baseLatency+"}")
			out, errOut, err := runKVTransferCLILeg(t, name, "run", catalog, "100", "", "300", "")
			if err == nil {
				t.Fatalf("catalog base_latency=%s must be refused; stdout:\n%s", tc.baseLatency, out)
			}
			for _, frag := range []string{"base_latency", "--kv-transfer-base-latency"} {
				if !strings.Contains(errOut, frag) {
					t.Errorf("refusal must mention %q, got:\n%s", frag, errOut)
				}
			}
		})
	}
}

func TestLegacyKVTransfer_MaxBaseLatencyOverrideIsRefused(t *testing.T) {
	if kvTransferCLISubprocess() {
		return
	}
	const name = "TestLegacyKVTransfer_MaxBaseLatencyOverrideIsRefused"
	catalog := filepath.Join("..", "testdata", "catalog")
	tracePrefix := filepath.Join(t.TempDir(), "legacy-kv-base-overflow")
	exported, errOut, err := runKVTransferCLILeg(t, name, "run-export", catalog, "100", "1", "300", tracePrefix)
	if err != nil {
		t.Fatalf("run-export control failed: %v\nstdout:\n%s\nstderr:\n%s", err, exported, errOut)
	}

	const maxInt64 = "9223372036854775807"
	for _, leg := range []string{"run", "replay"} {
		t.Run(leg, func(t *testing.T) {
			out, errOut, err := runKVTransferCLILeg(t, name, leg, catalog, "100", maxInt64, "300", tracePrefix)
			if err == nil {
				t.Fatalf("%s: MaxInt64 base latency plus a positive transfer charge must be refused; stdout:\n%s", leg, out)
			}
			for _, frag := range []string{"--kv-transfer-base-latency", "int64"} {
				if !strings.Contains(errOut, frag) {
					t.Errorf("%s: refusal must mention %q, got:\n%s", leg, frag, errOut)
				}
			}
		})
	}
}

// TestKVTransferBandwidthFlagHelp_DoesNotPromiseZeroDerives keeps the user-facing wording
// honest about what TestRunCmd_LegacyKVTransfer_SuppliedZeroIsRefused proves. The original
// help text read "Unset (0) = derived", which reads as an invitation to pass 0 — the exact
// mis-description qa-review F2 reported. The help must instead say derivation comes from
// leaving the flag out.
func TestKVTransferBandwidthFlagHelp_DoesNotPromiseZeroDerives(t *testing.T) {
	for cmdName, flags := range map[string]*pflag.FlagSet{
		"run": runCmd.Flags(), "replay": replayCmd.Flags(),
	} {
		f := flags.Lookup("kv-transfer-bandwidth")
		if f == nil {
			t.Fatalf("%s must register --kv-transfer-bandwidth", cmdName)
		}
		if strings.Contains(f.Usage, "(0) = derived") || strings.Contains(f.Usage, "0 = derive") {
			t.Errorf("%s --kv-transfer-bandwidth help presents 0 as selecting derivation, but a "+
				"supplied 0 is refused — say derivation comes from leaving the flag UNSET (#1819)\n"+
				"  usage: %s", cmdName, f.Usage)
		}
		if !strings.Contains(strings.ToUpper(f.Usage), "UNSET") {
			t.Errorf("%s --kv-transfer-bandwidth help must tell the operator that leaving the flag "+
				"unset is what derives the rate\n  usage: %s", cmdName, f.Usage)
		}

		base := flags.Lookup("kv-transfer-base-latency")
		if base == nil {
			t.Fatalf("%s must register --kv-transfer-base-latency", cmdName)
		}
		for _, want := range []string{"UNSET", "explicitly supplied 0"} {
			if !strings.Contains(strings.ToUpper(base.Usage), strings.ToUpper(want)) {
				t.Errorf("%s --kv-transfer-base-latency help must mention %q to distinguish omission from zero\n  usage: %s",
					cmdName, want, base.Usage)
			}
		}
	}
}

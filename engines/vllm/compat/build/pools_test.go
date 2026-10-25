// Original to blis-vllm.

package build

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/blis-vllm/compat/kvtransfer"
)

func readScenario(t *testing.T, repos kernelmodel.Repos) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(repos.Scenarios, "variant.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func writeScenario(t *testing.T, repos kernelmodel.Repos, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(repos.Scenarios, "variant.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func noFatal(t *testing.T) Fatal {
	return func(format string, args ...any) { t.Fatalf("unexpected run-time failure: "+format, args...) }
}

// A disaggregated run opens both pools of its scenario; a colocated run or one with no
// kernel opens none; counts that do not add up, and shared or encode instances, are refused.
func TestOpenPools_OpensBothPoolsOrSaysWhyNot(t *testing.T) {
	repos := pdVariant(t, "", "")
	p, _, err := OpenPools("variant.yaml", repos, true, Topology{NumInstances: 4, Prefill: 3, Decode: 1})
	if err != nil || p == nil || p.Prefill.Role() != deployment.RolePrefill || p.Decode.Role() != deployment.RoleDecode {
		t.Fatalf("3P1D: %+v, %v", p, err)
	}
	if p.Connector != "NixlConnector" || !p.HoldsPrefillKV() {
		t.Errorf("connector %q, holds %v; want NixlConnector holding the prefill KV", p.Connector, p.HoldsPrefillKV())
	}
	if p, _, err := OpenPools("variant.yaml", repos, true, Topology{NumInstances: 4}); p != nil || err != nil {
		t.Errorf("a colocated topology: %+v, %v; want no pools", p, err)
	}
	if p, _, err := OpenPools("variant.yaml", repos, false, Topology{NumInstances: 4, Prefill: 3, Decode: 1}); p != nil || err != nil {
		t.Errorf("no kernel open: %+v, %v; want no pools", p, err)
	}
	if _, _, err := OpenPools("variant.yaml", repos, true, Topology{NumInstances: 5, Prefill: 3, Decode: 1}); err == nil ||
		!strings.Contains(err.Error(), "must add up") {
		t.Errorf("counts that do not add up: %v", err)
	}
	if _, _, err := OpenPools("variant.yaml", repos, true, Topology{NumInstances: 4, SharedOrEncode: true}); err == nil ||
		!strings.Contains(err.Error(), "--prefill-decode-instances and --encode-instances are not") {
		t.Errorf("shared or encode instances: %v", err)
	}
	var none *Pools
	if none.HoldsPrefillKV() {
		t.Error("a colocated run's pools hold no prefill KV")
	}
}

// The MultiConnector assumption is a warning the run states however opening the pools ends: a
// refusal found after it (here, instance counts that do not add up) still returns it, so the
// operator sees the assumption and the refusal together, as before the move from the CLI.
func TestOpenPools_KeepsTheMultiConnectorWarningWhenRefused(t *testing.T) {
	const multi = "deepseek-v4.1-flash-h100-1p1d-dspark5-cpu1t.yaml"
	repos := vendoredRepos(t)
	p, warnings, err := OpenPools(multi, repos, true, Topology{NumInstances: 2, Prefill: 1, Decode: 1})
	if err != nil || p == nil || len(warnings) != 1 || !strings.Contains(warnings[0], "MultiConnector") {
		t.Fatalf("an accepted MultiConnector run: %+v, %q, %v; want its one warning", p, warnings, err)
	}
	p, warnings, err = OpenPools(multi, repos, true, Topology{NumInstances: 3, Prefill: 1, Decode: 1})
	if err == nil || !strings.Contains(err.Error(), "must add up") {
		t.Fatalf("counts that do not add up: %v", err)
	}
	if p != nil {
		t.Errorf("a refused run must return no pools to run, got %+v", p)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "MultiConnector") {
		t.Errorf("a refused MultiConnector run: warnings %q; want the warning returned with the refusal", warnings)
	}
	// A plain NIXL scenario has nothing to warn about, accepted or refused.
	for _, top := range []Topology{{NumInstances: 4, Prefill: 3, Decode: 1}, {NumInstances: 5, Prefill: 3, Decode: 1}} {
		if _, warnings, _ := OpenPools("variant.yaml", pdVariant(t, "", ""), true, top); len(warnings) != 0 {
			t.Errorf("NixlConnector %+v: warnings %q, want none", top, warnings)
		}
	}
}

// A pricing failure found while the run is going is reported through the caller's Fatal hook,
// which never returns. One that does return would leave the call with no price to give, so it
// panics rather than price a free transfer: zero ticks, or an empty flow.
func TestRunTimeFailuresNeverPriceAFreeTransfer(t *testing.T) {
	repos := pdVariant(t, "", "")
	p, _, err := OpenPools("variant.yaml", repos, true, Topology{NumInstances: 4, Prefill: 3, Decode: 1})
	if err != nil {
		t.Fatal(err)
	}
	reported := ""
	returning := func(format string, args ...any) { reported = fmt.Sprintf(format, args...) }
	mustPanic := func(what string, f func()) {
		t.Helper()
		reported = ""
		defer func() {
			if recover() == nil {
				t.Errorf("%s: a Fatal hook that returns must not let the call return a price", what)
			}
			if reported == "" {
				t.Errorf("%s: the failure was not reported through the Fatal hook", what)
			}
		}()
		f()
	}
	mustPanic("an unknown tier", func() { TierTicks(p.Prefill, "no-such-tier", true, 4096, 1, returning) })
	if got := TierTicks(p.Prefill, KernelCPUTier, true, 4096, 1, noFatal(t)); got <= 0 {
		t.Errorf("a known tier: %d ticks, want a positive price", got)
	}
	// The handoff demand's run-time failure: a demand the kernel could not give.
	broken := func(int64, kernel.Placement, kernel.Placement) (kvtransfer.Flow, error) {
		return kvtransfer.Flow{}, fmt.Errorf("a malformed demand")
	}
	mustPanic("a malformed handoff demand", func() { flowOrFail(broken, 64, kernel.Placement{}, kernel.Placement{}, returning) })
	fine := func(int64, kernel.Placement, kernel.Placement) (kvtransfer.Flow, error) {
		return kvtransfer.Flow{Bytes: 1}, nil
	}
	if f := flowOrFail(fine, 64, kernel.Placement{}, kernel.Placement{}, noFatal(t)); f.Bytes != 1 {
		t.Errorf("a well-formed demand: flow %+v, want the demand's own", f)
	}
}

// Pools that page, cache, replicate or draft differently cannot hand KV off and are refused,
// naming what differs.
func TestOpenPools_RefusesPoolsThatCannotHandOff(t *testing.T) {
	for _, c := range []struct{ name, prefill, decode, want string }{
		{"cache dtype", "", "      mamba_cache_dtype: float32\n", "mamba_cache_dtype"},
	} {
		repos := pdVariant(t, c.prefill, c.decode)
		_, _, err := OpenPools("variant.yaml", repos, true, Topology{NumInstances: 4, Prefill: 3, Decode: 1})
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want a refusal naming %q", c.name, err, c.want)
		}
	}
	if err := RequireNIXLHandoff("s.yaml", "MooncakeConnector"); err == nil || !strings.Contains(err.Error(), "MooncakeConnector") {
		t.Errorf("a non-NIXL connector: %v", err)
	}
	for c := range nixlConnectors {
		if err := RequireNIXLHandoff("s.yaml", c); err != nil {
			t.Errorf("%s is NIXL's handoff: %v", c, err)
		}
	}
}

// Every placed pair is priced up front; the price between two instances is the prefill
// kernel's between their ranks' placements; contention needs a disaggregated run, and its
// demand costs exactly the price alone (the blis-schemas law).
func TestHandoffPricing_IsThePlacedPairsPrice(t *testing.T) {
	repos := pdVariant(t, "", "")
	top := Topology{NumInstances: 4, Prefill: 3, Decode: 1}
	p, _, err := OpenPools("variant.yaml", repos, true, top)
	if err != nil {
		t.Fatal(err)
	}
	price, demand, err := p.HandoffPricing("variant.yaml", true, top, 64, noFatal(t))
	if err != nil || price == nil || demand == nil {
		t.Fatalf("pricing: %v", err)
	}
	for i := 0; i < 3; i++ {
		from := "instance_" + string(rune('0'+i))
		want := p.Prefill.PDTransferTicks(1024, p.Prefill.PlacementOf(i), p.Decode.PlacementOf(0))
		if got := price(1024, from, "instance_3"); got != want || got <= 0 {
			t.Errorf("%s -> instance_3: %d ticks, want the kernel's %d", from, got, want)
		}
	}
	if _, _, err := (*Pools)(nil).HandoffPricing("s.yaml", true, top, 64, noFatal(t)); err == nil ||
		!strings.Contains(err.Error(), "--pd-transfer-contention needs a disaggregated run") {
		t.Errorf("contention without pools: %v", err)
	}
	if pr, de, err := (*Pools)(nil).HandoffPricing("s.yaml", false, top, 64, noFatal(t)); pr != nil || de != nil || err != nil {
		t.Errorf("a colocated run prices no handoff: %v", err)
	}
}

// An instance's rank is its number in the prefill pool and its offset from the first decode
// instance in the decode pool; a malformed ID panics.
func TestRankInPool(t *testing.T) {
	for _, c := range []struct {
		id          string
		first, want int
	}{{"instance_0", 3, 0}, {"instance_2", 3, 2}, {"instance_3", 3, 0}, {"instance_5", 3, 2}, {"instance_7", 1 << 30, 7}} {
		if got := RankInPool(c.id, c.first); got != c.want {
			t.Errorf("RankInPool(%s, %d) = %d, want %d", c.id, c.first, got, c.want)
		}
	}
	defer func() {
		if recover() == nil {
			t.Error("a malformed instance ID must panic")
		}
	}()
	RankInPool("node-1", 0)
}

// A run that asks for more instances than a pool's layout holds is refused naming the pool;
// one that fits passes.
func TestRequireCapacity(t *testing.T) {
	repos := pdVariant(t, "", "")
	top := Topology{NumInstances: 4, Prefill: 3, Decode: 1}
	p, _, err := OpenPools("variant.yaml", repos, true, top)
	if err != nil {
		t.Fatal(err)
	}
	if err := RequireCapacity("variant.yaml", p.Prefill, p, top); err != nil {
		t.Errorf("a fitting topology: %v", err)
	}
	big := Topology{NumInstances: 4, Prefill: 3, Decode: p.Decode.RankCapacity() + 1}
	if err := RequireCapacity("variant.yaml", p.Prefill, p, big); err == nil || !strings.Contains(err.Error(), "decode instance(s) requested") {
		t.Errorf("too many decode instances: %v", err)
	}
	if err := RequireCapacity("variant.yaml", nil, nil, big); err != nil {
		t.Errorf("no kernel: %v", err)
	}
}

// The legacy CPU tier's reload is priced by the kernel's cpu_dram TierTime in whole ticks;
// an enabled offload on a block of no bytes is refused; a tier with no device class is
// refused.
func TestPriceColocatedOffload(t *testing.T) {
	m, err := kernelmodel.Open("glm-5-h200-tp8-mtp3.yaml", vendoredRepos(t))
	if err != nil {
		t.Fatal(err)
	}
	var none engine.KVOffloadConfig
	ticks, err := PriceColocatedOffload(m, &none, OffloadPricing{BlockSizeTokens: 64, KVCPUBlocks: 10}, noFatal(t))
	want, terr := tierTicks(m, KernelCPUTier, false, m.Kernel().SequenceVariableBytes(64), 1)
	if err != nil || terr != nil || ticks != want || ticks <= 0 {
		t.Errorf("legacy CPU tier: %d ticks (%v), want the kernel's %d (%v)", ticks, err, want, terr)
	}
	if ticks, err := PriceColocatedOffload(nil, &none, OffloadPricing{BlockSizeTokens: 64, KVCPUBlocks: 10}, noFatal(t)); ticks != 0 || err != nil {
		t.Errorf("no kernel: %d, %v", ticks, err)
	}
	noClass := engine.KVOffloadConfig{Enabled: true, CPUBytesToUse: 1 << 32, BlockSize: 64, BlocksPerChunk: 1,
		TokensPerHash: 64, EvictionPolicy: "lru", PerBlockBytes: 1, Tiers: []engine.KVOffloadTier{{DirectIO: true}}}
	if _, err := PriceColocatedOffload(m, &noClass, OffloadPricing{BlockSizeTokens: 64}, noFatal(t)); err == nil ||
		!strings.Contains(err.Error(), "names no device_class") {
		t.Errorf("a tier with no device class: %v", err)
	}
}

// A P/D run's offload needs a MultiConnector; with one, each pool is priced by its own
// kernel and the run's description is not mutated.
func TestPoolsPriceOffload(t *testing.T) {
	repos := pdVariant(t, "", "")
	top := Topology{NumInstances: 4, Prefill: 3, Decode: 1}
	p, _, err := OpenPools("variant.yaml", repos, true, top)
	if err != nil {
		t.Fatal(err)
	}
	run := engine.KVOffloadConfig{}
	if _, _, ok, err := p.PriceOffload("variant.yaml", run, OffloadPricing{BlockSizeTokens: 64}, top, noFatal(t)); ok || err != nil {
		t.Errorf("no offload: ok %v, %v", ok, err)
	}
	if _, _, _, err := p.PriceOffload("variant.yaml", run, OffloadPricing{BlockSizeTokens: 64, KVCPUBlocks: 10}, top, noFatal(t)); err == nil ||
		!strings.Contains(err.Error(), "MultiConnector") {
		t.Errorf("offload under a plain NIXL connector: %v", err)
	}
	p.Connector = "MultiConnector"
	pre, dec, ok, err := p.PriceOffload("variant.yaml", run, OffloadPricing{BlockSizeTokens: 64, KVCPUBlocks: 10}, top, noFatal(t))
	if err != nil || !ok || pre.TicksPerBlock <= 0 || dec.TicksPerBlock <= 0 {
		t.Fatalf("legacy tier under a MultiConnector: %+v %+v %v %v", pre, dec, ok, err)
	}
	wantPre, _ := tierTicks(p.Prefill, KernelCPUTier, false, p.Prefill.Kernel().SequenceVariableBytes(64), 1)
	if pre.TicksPerBlock != wantPre {
		t.Errorf("prefill reload %d ticks, want its own kernel's %d", pre.TicksPerBlock, wantPre)
	}
}

// The steady per-sequence hold is reported for a model with bounded state and is silent for
// one without; a window of zero is refused.
func TestSequenceFixedChargeAndWindow(t *testing.T) {
	if note := SequenceFixedCharge("pool", kernelmodel.Settings{}); note != "" {
		t.Errorf("a model with no bounded state: %q, want silence", note)
	}
	if err := RequireWindowFits("s.yaml", "first pool", kernelmodel.Settings{}); err == nil ||
		!strings.Contains(err.Error(), "states no engine.max_model_len") {
		t.Errorf("no window: %v", err)
	}
	if err := RequireWindowFits("s.yaml", "first pool", kernelmodel.Settings{MaxModelLen: 4096}); err != nil {
		t.Errorf("a window: %v", err)
	}
}

// Endpoint placements put each instance on its pool's rank; an instance past a pool's
// capacity is refused.
func TestEndpointPlacements(t *testing.T) {
	repos := pdVariant(t, "", "")
	top := Topology{NumInstances: 4, Prefill: 3, Decode: 1}
	p, _, err := OpenPools("variant.yaml", repos, true, top)
	if err != nil {
		t.Fatal(err)
	}
	got, err := EndpointPlacements(p.Prefill, p, 4, 3, 1)
	if err != nil || len(got) != 4 {
		t.Fatalf("placements: %v, %v", got, err)
	}
	for id, pl := range got {
		if pl.NodeCount < 1 || (pl.NodeCount == 1 && pl.Hostname == "") {
			t.Errorf("%s: %+v, want a placed endpoint", id, pl)
		}
	}
	if got["instance_0"].Hostname == got["instance_3"].Hostname {
		t.Error("prefill and decode pools must be on different nodes")
	}
	if _, err := EndpointPlacements(p.Prefill, p, 5, 3, 1); err == nil || !strings.Contains(err.Error(), "no kernel pool placement") {
		t.Errorf("an instance in no pool: %v", err)
	}
	if got, err := EndpointPlacements(nil, nil, 4, 0, 0); got != nil || err != nil {
		t.Errorf("no kernel: %v %v", got, err)
	}
}

// A tier transfer is not done until its last byte lands, so its price is the kernel's time
// rounded up to a whole tick, never down: a transfer with a fractional last microsecond costs
// the next tick.
func TestTierTicks_RoundUpToAWholeTick(t *testing.T) {
	m, err := kernelmodel.Open("glm-5-h200-tp8-mtp3.yaml", vendoredRepos(t))
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, bytes := range []int64{1, 1000, 4097, 65537, 1 << 20, 3<<20 + 7} {
		d := m.Kernel().TierTime(KernelCPUTier, kernel.DirectionFromTier, bytes, 1)
		got, err := tierTicks(m, KernelCPUTier, false, bytes, 1)
		if err != nil {
			t.Fatal(err)
		}
		want := max(1, (d.Nanoseconds()+999)/1000)
		if got != want {
			t.Errorf("%d bytes: %d ticks, want %d (%v rounded up)", bytes, got, want, d)
		}
		if d.Nanoseconds()%1000 != 0 {
			found = true
		}
	}
	if !found {
		t.Fatal("no probe has a fractional microsecond; the rounding is untested")
	}
}

// Each pool of a disaggregated run sizes the run's one offload by its own kernel: pools that
// cache in different formats hold different bytes per block, and each pool's tier sizing and
// reload charge follow its own.
func TestPoolsPriceOffload_EachPoolByItsOwnKernel(t *testing.T) {
	repos := pdVariant(t, "", "")
	top := Topology{NumInstances: 4, Prefill: 3, Decode: 1}
	p, _, err := OpenPools("variant.yaml", repos, true, top)
	if err != nil {
		t.Fatal(err)
	}
	// Open the decode pool from a variant caching in bf16 (the handoff check is not in play:
	// the pools are priced, not opened together).
	bf16 := pdVariant(t, "", "")
	raw := strings.Replace(readScenario(t, bf16), "      cache_dtype: fp8\n", "      cache_dtype: auto\n", 2)
	writeScenario(t, bf16, raw)
	dec, err := kernelmodel.OpenRole("variant.yaml", bf16, deployment.RoleDecode)
	if err != nil {
		t.Fatal(err)
	}
	p.Decode, p.Connector = dec, "MultiConnector"
	if p.Prefill.Kernel().SequenceVariableBytes(64) == p.Decode.Kernel().SequenceVariableBytes(64) {
		t.Fatal("fixture lost its point: both pools hold the same bytes per block")
	}
	run := engine.KVOffloadConfig{Enabled: true, CPUBytesToUse: 1 << 32, BlockSize: 64, BlocksPerChunk: 1,
		TokensPerHash: 64, EvictionPolicy: "lru", PerBlockBytes: 1}
	pre, de, ok, err := p.PriceOffload("variant.yaml", run, OffloadPricing{BlockSizeTokens: 64}, top, noFatal(t))
	if err != nil || !ok {
		t.Fatalf("pricing: %v %v", ok, err)
	}
	if pre.Config.PerBlockBytes != p.Prefill.Kernel().SequenceVariableBytes(64) ||
		de.Config.PerBlockBytes != p.Decode.Kernel().SequenceVariableBytes(64) {
		t.Errorf("per-block bytes %d / %d, want each pool's own %d / %d", pre.Config.PerBlockBytes,
			de.Config.PerBlockBytes, p.Prefill.Kernel().SequenceVariableBytes(64), p.Decode.Kernel().SequenceVariableBytes(64))
	}
	if run.PerBlockBytes != 1 {
		t.Error("PriceOffload mutated the run's description")
	}
}

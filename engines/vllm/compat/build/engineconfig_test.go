// Original to blis-vllm.

package build

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/engine"
)

func writeLoRAFile(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "lora.yaml")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// The three LoRA sources compose in increasing precedence: defaults.yaml's coefficients fill
// only what the file leaves unset, and a flag that is set wins over both; an unset flag never
// clobbers them (R18). With no file and no flags, the coefficients are there but no adapter
// is, so the subsystem is inert.
func TestResolveLoRAConfig_DefaultsThenFileThenFlags(t *testing.T) {
	defs := &LoRADefaults{LoadBaseLatencyUs: 1500, LoadBandwidthBytesUs: 2e6, FootprintBytesPerRank: 2e6,
		StepOverheadTiers: map[int][2]float64{8: {1, 2}}}
	cfg, err := ResolveLoRAConfig("", defs, LoRAFlags{})
	if err != nil {
		t.Fatal(err)
	}
	if *cfg.LoadBaseLatencyUs != 1500 || *cfg.LoadBandwidthBytesUs != 2e6 || *cfg.FootprintBytesPerRank != 2e6 ||
		*cfg.StepOverheadTiers[8].K6 != 1 || *cfg.StepOverheadTiers[8].K7 != 2 {
		t.Errorf("defaults only: %+v", cfg)
	}
	if cfg.HasAdapters() || cfg.AdapterCapacity != nil {
		t.Errorf("no file: want no adapters and no capacity, got %+v", cfg)
	}

	path := writeLoRAFile(t, "lora:\n  adapter_capacity: 4\n  load_base_latency_us: 700\n  step_overhead_tiers:\n    16: {k6: 5, k7: 6}\n  adapters:\n    - id: a0\n      rank: 8\n    - id: a1\n      rank: 16\n")
	cfg, err = ResolveLoRAConfig(path, defs, LoRAFlags{})
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Adapters) != 2 || *cfg.AdapterCapacity != 4 {
		t.Errorf("file: want its 2 adapters and capacity 4, got %+v", cfg)
	}
	if *cfg.LoadBaseLatencyUs != 700 || *cfg.LoadBandwidthBytesUs != 2e6 {
		t.Errorf("file over defaults: base %v, bandwidth %v; want the file's 700 and the default 2e6",
			*cfg.LoadBaseLatencyUs, *cfg.LoadBandwidthBytesUs)
	}
	if _, ok := cfg.StepOverheadTiers[8]; ok || *cfg.StepOverheadTiers[16].K6 != 5 {
		t.Errorf("a file that states tiers keeps its own, without the defaults': %+v", cfg.StepOverheadTiers)
	}

	cfg, err = ResolveLoRAConfig(path, defs, LoRAFlags{
		AdapterCapacity: 9, AdapterCapacitySet: true,
		LoadBaseLatencyUs: 9999, LoadBaseLatencyUsSet: true,
		LoadBandwidthBytesUs: 3e6, LoadBandwidthBytesUsSet: true,
		FootprintBytesPerRank: 4e6, FootprintBytesPerRankSet: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if *cfg.AdapterCapacity != 9 || *cfg.LoadBaseLatencyUs != 9999 || *cfg.LoadBandwidthBytesUs != 3e6 || *cfg.FootprintBytesPerRank != 4e6 {
		t.Errorf("set flags must win over file and defaults: %+v", cfg)
	}
	cfg, err = ResolveLoRAConfig(path, defs, LoRAFlags{AdapterCapacity: 9, LoadBaseLatencyUs: 9999})
	if err != nil {
		t.Fatal(err)
	}
	if *cfg.AdapterCapacity != 4 || *cfg.LoadBaseLatencyUs != 700 {
		t.Errorf("unset flags must not clobber the file: capacity %v, base %v", *cfg.AdapterCapacity, *cfg.LoadBaseLatencyUs)
	}

	if cfg, err = ResolveLoRAConfig("", nil, LoRAFlags{}); err != nil || cfg.LoadBaseLatencyUs != nil {
		t.Errorf("no defaults, no file, no flags: %+v, %v; want the zero config", cfg, err)
	}
}

// Every way the LoRA configuration can be wrong is refused with the CLI's text: a file that
// cannot be read, one that does not parse or misspells a key (strict, R10), and a config that
// does not validate (adapters with no slot to load into).
func TestResolveLoRAConfig_RefusesWhatIsWrong(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "absent.yaml")
	for _, c := range []struct {
		name, path, want string
	}{
		{"missing file", missing, `Failed to read --lora-config file "` + missing + `"`},
		{"typo", writeLoRAFile(t, "lora:\n  adapter_capacty: 4\n"), "Failed to parse --lora-config file"},
		{"not yaml", writeLoRAFile(t, "lora: [\n"), "Failed to parse --lora-config file"},
		{"zero capacity with adapters", writeLoRAFile(t, "lora:\n  adapter_capacity: 0\n  adapters:\n    - id: a0\n      rank: 8\n"),
			"Invalid LoRA configuration: "},
	} {
		_, err := ResolveLoRAConfig(c.path, nil, LoRAFlags{})
		if err == nil || !strings.HasPrefix(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error starting %q", c.name, err, c.want)
		}
	}
	_, err := ResolveLoRAConfig(writeLoRAFile(t, "lora:\n  adapter_capacity: 0\n  adapters:\n    - id: a0\n      rank: 8\n"), nil, LoRAFlags{})
	if err == nil || !strings.Contains(err.Error(), "adapter_capacity") {
		t.Errorf("the validation refusal must name adapter_capacity: %v", err)
	}
}

// A scheduler or preemption policy the engine does not run is refused with the CLI's text and
// the valid names, sorted; the scheduler is checked first; every name the engine runs passes.
func TestCheckPolicyNames(t *testing.T) {
	for name := range engine.SchedulerNameTable() {
		if err := CheckPolicyNames(name, "fcfs"); err != nil {
			t.Errorf("scheduler %q: %v", name, err)
		}
	}
	for name := range engine.PreemptionPolicyNameTable() {
		if err := CheckPolicyNames("fcfs", name); err != nil {
			t.Errorf("preemption policy %q: %v", name, err)
		}
	}
	err := CheckPolicyNames("lottery", "fcfs")
	if want := `Unknown scheduler "lottery". Valid: fcfs, priority-fcfs, reverse-priority, sjf`; err == nil || err.Error() != want {
		t.Errorf("unknown scheduler: %v, want %q", err, want)
	}
	err = CheckPolicyNames("fcfs", "random")
	if want := `Unknown preemption policy "random". Valid: fcfs, priority`; err == nil || err.Error() != want {
		t.Errorf("unknown preemption policy: %v, want %q", err, want)
	}
	if err = CheckPolicyNames("lottery", "random"); err == nil || !strings.HasPrefix(err.Error(), "Unknown scheduler") {
		t.Errorf("both unknown: %v, want the scheduler named first", err)
	}
}

// EngineConfigOf hands every setting to the sub-config it belongs to, unchanged.
func TestEngineConfigOf_PlacesEverySetting(t *testing.T) {
	capacity := 2
	s := EngineSettings{
		TotalKVBlocks: 100, BlockSizeTokens: 16, KVCPUBlocks: 7,
		KVOffloadThreshold: 0.5, KVTransferBandwidth: 3.5, KVTransferBaseLatency: 11,
		KVOffload:               engine.KVOffloadConfig{CPUBytesToUse: 1 << 20},
		KVTransferTicksPerBlock: 13,
		SequenceState:           engine.KVSequenceState{Hold: func(int64, int64) int64 { return 4 }, Gate: func(int64) int64 { return 5 }},
		PrefixHit:               engine.KVPrefixHit{FullEagle: true},
		MaxNumSeqs:              64, MaxNumBatchedTokens: 4096, LongPrefillTokenThreshold: 256, PrefixCachingDisabled: true,
		Model:     engine.ModelConfig{NumLocalExperts: 8, NumExpertsPerTok: 2},
		ModelName: "m", GPU: "g", TP: 2, DP: 1, EnableExpertParallel: true, MaxModelLen: 8192,
		Scheduler: "priority-fcfs", PreemptionPolicy: "priority",
		LoRA:        engine.LoRAConfig{AdapterCapacity: &capacity},
		Speculative: engine.SpeculativeConfig{K: 3},
	}
	c := EngineConfigOf(s)
	kv := c.KVCache
	if kv.TotalKVBlocks != 100 || kv.BlockSizeTokens != 16 || kv.KVCPUBlocks != 7 || kv.KVOffloadThreshold != 0.5 ||
		kv.KVTransferBandwidth != 3.5 || kv.KVTransferBaseLatency != 11 || kv.KVTransferTicksPerBlock != 13 || kv.Offload.CPUBytesToUse != 1<<20 ||
		kv.SequenceState.Gate == nil || kv.SequenceState.Gate(0) != 5 || kv.SequenceState.Hold(0, 0) != 4 || !kv.PrefixHit.FullEagle {
		t.Errorf("KV cache: %+v", kv)
	}
	if b := c.Batch; b.MaxNumSeqs != 64 || b.MaxNumBatchedTokens != 4096 || b.LongPrefillTokenThreshold != 256 || !b.PrefixCachingDisabled {
		t.Errorf("batch: %+v", b)
	}
	if m := c.ModelHardware; m.ModelConfig.NumLocalExperts != 8 || m.Model != "m" || m.GPU != "g" || m.TP != 2 || m.DP != 1 ||
		!m.EnableExpertParallel || m.MaxModelLen != 8192 {
		t.Errorf("model and hardware: %+v", m)
	}
	if p := c.Policy; p.Scheduler != "priority-fcfs" || p.PreemptionPolicy != "priority" {
		t.Errorf("policy: %+v", p)
	}
	if c.LoRA.AdapterCapacity == nil || *c.LoRA.AdapterCapacity != 2 || c.Speculative.K != 3 {
		t.Errorf("LoRA %+v, speculative %+v: want them as given", c.LoRA, c.Speculative)
	}
}

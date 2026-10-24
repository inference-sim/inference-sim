package sim

import (
	"strings"
	"testing"
)

// NewEngine is the one place an instance's engine is assembled: KV store, latency model and
// Simulator. A config it cannot build from is an error naming the instance, never a silent
// default (R1), and a buildable one yields an engine whose KV store is the configured one.
func TestNewEngine_BuildsFromTheConfigOrSaysWhyNot(t *testing.T) {
	cfg := newTestSimConfig()
	cfg.LatencyModel = newFakeLatency()
	s, err := NewEngine("inst-a", cfg)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	if got := s.KVCache.TotalCapacity(); got != cfg.TotalKVBlocks {
		t.Errorf("engine KV capacity %d, want the configured %d", got, cfg.TotalKVBlocks)
	}
	if got := s.KVCache.BlockSize(); got != cfg.BlockSizeTokens {
		t.Errorf("engine block size %d, want the configured %d", got, cfg.BlockSizeTokens)
	}

	noModel := newTestSimConfig()
	if _, err := NewEngine("inst-b", noModel); err == nil || !strings.Contains(err.Error(), "inst-b") ||
		!strings.Contains(err.Error(), "LatencyModel is nil") {
		t.Errorf("a config without a latency model: err %v, want one naming inst-b and the missing model", err)
	}

	bad := newTestSimConfig()
	bad.LatencyModel = newFakeLatency()
	bad.MaxNumSeqs = 0
	if _, err := NewEngine("inst-c", bad); err == nil || !strings.Contains(err.Error(), "inst-c") {
		t.Errorf("a config NewSimulator refuses: err %v, want one naming inst-c", err)
	}

	saved := NewKVStoreFromConfig
	NewKVStoreFromConfig = nil
	defer func() { NewKVStoreFromConfig = saved }()
	if _, err := NewEngine("inst-d", cfg); err == nil || !strings.Contains(err.Error(), "import sim/kv") {
		t.Errorf("no registered KV constructor: err %v, want one saying to import sim/kv", err)
	}
}

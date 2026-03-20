package kv

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestNewKVStore_TieredMode_ThresholdOutOfRange_Panics(t *testing.T) {
	tests := []struct {
		name      string
		threshold float64
	}{
		{"negative", -0.1},
		{"above_one", 1.1},
		{"NaN", math.NaN()},
		{"pos_inf", math.Inf(1)},
		{"neg_inf", math.Inf(-1)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for threshold=%v", tc.threshold)
				}
				msg := fmt.Sprintf("%v", r)
				if !strings.Contains(msg, "KVOffloadThreshold") {
					t.Errorf("panic message should mention KVOffloadThreshold, got: %s", msg)
				}
			}()
			cfg := sim.NewKVCacheConfig(10, 2, 5, tc.threshold, 100.0, 0)
			NewKVStore(cfg)
		})
	}
}

func TestNewKVStore_TieredMode_InvalidBandwidth_Panics(t *testing.T) {
	tests := []struct {
		name      string
		bandwidth float64
	}{
		{"zero", 0},
		{"negative", -1.0},
		{"NaN", math.NaN()},
		{"pos_inf", math.Inf(1)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for bandwidth=%v", tc.bandwidth)
				}
				msg := fmt.Sprintf("%v", r)
				if !strings.Contains(msg, "KVTransferBandwidth") {
					t.Errorf("panic message should mention KVTransferBandwidth, got: %s", msg)
				}
			}()
			cfg := sim.NewKVCacheConfig(10, 2, 5, 0.5, tc.bandwidth, 0)
			NewKVStore(cfg)
		})
	}
}

func TestNewKVStore_TieredMode_ValidEdgeCases(t *testing.T) {
	// Threshold=0 and threshold=1 are both valid (threshold is deprecated but still validated)
	tests := []struct {
		name      string
		threshold float64
	}{
		{"threshold_zero", 0.0},
		{"threshold_one", 1.0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := sim.NewKVCacheConfig(10, 2, 5, tc.threshold, 100.0, 0)
			store := NewKVStore(cfg)
			if store == nil {
				t.Fatal("NewKVStore should return non-nil for valid config")
			}
		})
	}
}

func TestNewKVStore_SingleTier_SkipsValidation(t *testing.T) {
	// When KVCPUBlocks <= 0, tiered-mode validation does not apply
	cfg := sim.NewKVCacheConfig(10, 2, 0, -999.0, -999.0, 0)
	store := NewKVStore(cfg)
	if store == nil {
		t.Fatal("NewKVStore should return non-nil for single-tier mode")
	}
}

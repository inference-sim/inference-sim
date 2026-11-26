// Original to blis-sglang.

package sglangengine

import (
	"testing"
)

// A pool's memory report is SGLang's own sizing: the device as nvidia-smi states it, the free
// memory before the load as the host states it, the kernel's fixed occupancy, and the KV pool the
// remainder was cut into. The budget's arithmetic is _profile_available_bytes
// (kv_cache_configurator.py:2990-3060@297e54fd): the free memory after the weights, less the
// static memory fraction's slack.
func TestMemoryOfReportsSGLangsOwnSizing(t *testing.T) {
	const (
		preModelLoadGiB = 78.0      // fakeHost's opening
		memFraction     = 0.8       // the block's mem_fraction_static
		cellBytes       = 100 << 10 // fakeKernel.SequenceVariableBytes: 100 KiB a token
	)
	h := newHost()
	s := specOf(t, "model_path: m\nmem_fraction_static: 0.8\npage_size: 1\n")
	m, err := MemoryOf(h, s)
	if err != nil {
		t.Fatal(err)
	}

	// The device is what nvidia-smi reports, not the catalog's GiB: SGLang reads the former
	// (get_device_memory_capacity, utils/common.py@297e54fd).
	if want := int64(81559) << 20; m.DeviceBytes != want {
		t.Errorf("device %d B, want nvidia-smi's 81,559 MiB = %d B", m.DeviceBytes, want)
	}
	if want := int64(preModelLoadGiB * float64(gib)); m.BudgetBytes != want {
		t.Errorf("budget %d B, want the host's %.0f GiB = %d B", m.BudgetBytes, preModelLoadGiB, want)
	}

	// The fake kernel's resident weights are 1 GiB (fakeKernel.FixedBytes); the sizing subtracts
	// them from the pre-load memory and the static fraction's slack from what is left.
	const weights = 1.0
	avail := preModelLoadGiB - weights
	slack := preModelLoadGiB * (1 - memFraction)
	if want := int64((avail - slack) * float64(gib)); m.AllocatableBytes != want {
		t.Errorf("allocatable %d B, want (%.1f - %.1f - %v) GiB = %d B",
			m.AllocatableBytes, avail, weights, slack, want)
	}
	if m.Fixed.Weights != gib {
		t.Errorf("fixed weights %d B, want the kernel's %d B", m.Fixed.Weights, int64(gib))
	}

	// The pool is the budget over the per-token cell, in whole pages: one token a page here.
	if m.BlockTokens != 1 || m.PerTokenBytes != cellBytes {
		t.Errorf("block is %d tokens of %d B, want 1 token of %d B", m.BlockTokens, m.BlockBytes, cellBytes)
	}
	if want := m.AllocatableBytes / cellBytes; m.TokensPerRank != want {
		t.Errorf("tokens %d, want the budget over the cell = %d", m.TokensPerRank, want)
	}
	if m.BlocksPerRank != m.TokensPerRank || m.BytesPerRank != m.BlocksPerRank*m.BlockBytes {
		t.Errorf("pool is %d blocks / %d tokens / %d B; want blocks x block bytes", m.BlocksPerRank, m.TokensPerRank, m.BytesPerRank)
	}
	if m.BytesPerRank > m.AllocatableBytes {
		t.Errorf("the pool (%d B) is larger than the memory left for it (%d B)", m.BytesPerRank, m.AllocatableBytes)
	}
	// SGLang refuses data parallelism above one, so a pool is one rank's.
	if m.DataParallel != 1 || m.AggregateBytes != m.BytesPerRank {
		t.Errorf("DP %d, aggregate %d B; want one rank", m.DataParallel, m.AggregateBytes)
	}
}

// The report and the engine are one resolution: reporting an opened pool is reporting the pool.
func TestMemoryOfAnOpenedPoolMatchesOpeningIt(t *testing.T) {
	h := newHost()
	s := specOf(t, "model_path: m\nmem_fraction_static: 0.8\npage_size: 1\n")
	opened, err := h.Open(s)
	if err != nil {
		t.Fatal(err)
	}
	fromPool, err := MemoryOfPool(opened, s)
	if err != nil {
		t.Fatal(err)
	}
	fromHost, err := MemoryOf(h, s)
	if err != nil {
		t.Fatal(err)
	}
	if fromPool != fromHost {
		t.Errorf("the opened pool reports %+v, opening it reports %+v", fromPool, fromHost)
	}
}

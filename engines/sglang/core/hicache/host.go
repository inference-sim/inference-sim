// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/pool_host/base.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package hicache

import (
	"errors"
	"fmt"
	"math"
)

// HostSizing is what HostKVCache.__init__ sizes a host pool from (pool_host/base.py:165-237
// @297e54fd), after the concrete pool has worked out its bytes per token.
type HostSizing struct {
	DeviceCapacityTokens int
	SizePerToken         int64
	HostToDeviceRatio    float64
	// HostSizeGB is --hicache-size: a fixed host pool, in GB, or 0 to size by the ratio.
	HostSizeGB        float64
	PageSize          int
	Layout            string
	HostTokenCapacity *int
}

// HostPool is HostKVCache's base class (pool_host/base.py:157-577@297e54fd): the host pool's
// size and its free list. DCP is not modelled (dcp_size 1), so logical and physical slots are
// the same. The copies are the transfer engine's to price; here they move no bytes.
type HostPool struct {
	size, pageSize, pageNum int
	sizePerToken            int64
	layout                  string
	freeSlots               []int64
	releaseSlots            [][]int64
	numReleaseSlots         int
	slotUsed                []bool
}

// NewHostPool sizes a host pool as HostKVCache.__init__ does: a page-aligned token capacity when
// one is given; otherwise --hicache-size GB over the bytes per token, or the device pool times
// the ratio, plus one padding page (base.py:204-226). A fixed size is not synced across
// pipeline stages here: one stage is modelled.
func NewHostPool(s HostSizing) (*HostPool, error) {
	if s.PageSize <= 0 {
		return nil, fmt.Errorf("hicache: host page size %d", s.PageSize)
	}
	p := &HostPool{pageSize: s.PageSize, sizePerToken: s.SizePerToken, layout: s.Layout}
	switch {
	case s.HostTokenCapacity != nil:
		c := *s.HostTokenCapacity
		if c <= 0 || c%p.pageSize != 0 {
			return nil, fmt.Errorf("hicache: host token capacity must be positive and page-aligned, got %d for page_size=%d", c, p.pageSize)
		}
		p.size, p.pageNum = c, c/p.pageSize
	default:
		if s.HostSizeGB > 0 {
			p.size = int(math.Floor(s.HostSizeGB * 1e9 / float64(s.SizePerToken)))
		} else {
			p.size = int(float64(s.DeviceCapacityTokens) * s.HostToDeviceRatio)
		}
		p.pageNum = p.size/p.pageSize + 1
		p.size = p.pageNum * p.pageSize
	}
	p.Clear()
	return p, nil
}

// Size is the host pool's slots; PageNum its pages.
func (p *HostPool) Size() int { return p.size }

// PageNum is the host pool's pages.
func (p *HostPool) PageNum() int { return p.pageNum }

// PageSize is the host pool's page size, in tokens.
func (p *HostPool) PageSize() int { return p.pageSize }

// SizePerToken is the bytes one token takes on host.
func (p *HostPool) SizePerToken() int64 { return p.sizePerToken }

// Layout is the host layout (layer_first, page_first, ...).
func (p *HostPool) Layout() string { return p.layout }

// Clear frees every slot (base.py:509-521).
func (p *HostPool) Clear() {
	p.freeSlots = make([]int64, p.size)
	for i := range p.freeSlots {
		p.freeSlots[i] = int64(i)
	}
	p.releaseSlots = nil
	p.numReleaseSlots = 0
	p.slotUsed = make([]bool, p.size)
}

// AvailableSize is the free slots, released ones included (base.py:523-524).
func (p *HostPool) AvailableSize() int { return len(p.freeSlots) + p.numReleaseSlots }

// mergeReleaseSlots moves the released slots to the end of the free list (base.py:526-536).
func (p *HostPool) mergeReleaseSlots() {
	if p.numReleaseSlots == 0 {
		return
	}
	if len(p.freeSlots) == 0 && len(p.releaseSlots) == 1 {
		p.freeSlots = p.releaseSlots[0]
	} else {
		free := append([]int64{}, p.freeSlots...)
		for _, r := range p.releaseSlots {
			free = append(free, r...)
		}
		p.freeSlots = free
	}
	p.releaseSlots = nil
	p.numReleaseSlots = 0
}

// Alloc takes needSize slots from the front of the free list, merging the released slots in
// only when the free list alone is too short (base.py:548-568); false when too few are free.
// needSize must be whole pages.
func (p *HostPool) Alloc(needSize int) ([]int64, bool) {
	if needSize%p.pageSize != 0 {
		panic("The requested size should be a multiple of the page size.")
	}
	if needSize > p.AvailableSize() {
		return nil, false
	}
	if needSize > len(p.freeSlots) {
		p.mergeReleaseSlots()
	}
	sel := append([]int64{}, p.freeSlots[:needSize]...)
	p.freeSlots = p.freeSlots[needSize:]
	for _, s := range sel {
		if p.slotUsed[s] {
			panic(fmt.Sprintf("Double-alloc detected: slot %d already allocated", s))
		}
	}
	for _, s := range sel {
		p.slotUsed[s] = true
	}
	return sel, true
}

// ErrDoubleFree is SGLang's double-free assertion (base.py:575-578): a free of a slot that is
// not allocated. Nothing is freed.
var ErrDoubleFree = errors.New("hicache: double-free detected")

// Free returns indices to the pool as one released run (base.py:570-583): they are handed out
// again only after the free list runs dry.
func (p *HostPool) Free(indices []int64) (int, error) {
	if len(indices) == 0 {
		return 0, nil
	}
	for _, s := range indices {
		if !p.slotUsed[s] {
			return 0, ErrDoubleFree
		}
	}
	for _, s := range indices {
		p.slotUsed[s] = false
	}
	p.releaseSlots = append(p.releaseSlots, append([]int64{}, indices...))
	p.numReleaseSlots += len(indices)
	return len(indices), nil
}

// PrepareTransferIndices resolves the indices one submission uses; a host pool with its own
// slots uses them as they are (base.py:425-438).
func (p *HostPool) PrepareTransferIndices(host, device []int64, ioBackend string) ([]int64, []int64) {
	return host, device
}

// BackupFromDeviceAllLayerPhysical copies device KV to host. The port moves no bytes.
func (p *HostPool) BackupFromDeviceAllLayerPhysical(host, device []int64, ioBackend string) {}

// LoadToDevicePerLayerPhysical copies one layer of host KV to the device. The port moves no bytes.
func (p *HostPool) LoadToDevicePerLayerPhysical(host, device []int64, layerID int, ioBackend string) {
}

// GetDataPage is the page of KV that starts at host index index; the port's page is that index.
func (p *HostPool) GetDataPage(index int64, flat bool) int64 { return index }

// GetDummyFlatDataPage is a page buffer to read a page from storage into; the port's is no page.
func (p *HostPool) GetDummyFlatDataPage() int64 { return -1 }

// SetFromFlatDataPage writes a page read from storage at host index index. The port moves no bytes.
func (p *HostPool) SetFromFlatDataPage(index int64, page int64) {}

// Destroy releases the host buffer (base.py:283-302); the port holds none.
func (p *HostPool) Destroy() {}

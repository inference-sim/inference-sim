// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/memory_pool.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package cache

import (
	"fmt"
	"slices"
)

// RequestPool is ReqToTokenPool (memory_pool.py:376-492@297e54fd): size rows, one per running
// request, each holding the KV slot of every position of that request. Row 0 is a padding row
// that is never handed out (memory_pool.py:400-402). The auxiliary cache that Mamba and HiSparse
// attach is not modelled; those pools are refused (refuse.go), so the aux hooks do nothing, as
// they do in SGLang with no aux cache attached.
type RequestPool struct {
	size, maxContextLen int
	// free is free_slots: rows are taken from its tail and returned to its tail.
	free []int
	// rows holds each row's positions as written; a position never written reads as slot 0, as
	// the zeroed tensor does.
	rows map[int][]int64
	// generation is req_generation: bumped each time a row is handed out (memory_pool.py:448).
	generation []int64
}

// NewRequestPool is ReqToTokenPool.__init__ (memory_pool.py:388-410@297e54fd).
func NewRequestPool(size, maxContextLen int) *RequestPool {
	p := &RequestPool{size: size, maxContextLen: maxContextLen, rows: map[int][]int64{},
		generation: make([]int64, size+1)}
	p.Clear()
	return p
}

func (p *RequestPool) Size() int          { return p.size }
func (p *RequestPool) MaxContextLen() int { return p.maxContextLen }

// AvailableSize is memory_pool.py:417-418@297e54fd.
func (p *RequestPool) AvailableSize() int { return len(p.free) }

// Alloc is memory_pool.py:420-436@297e54fd: a request already holding a row (a chunked prefill
// continuing) keeps it; each other request gets a new row, in order.
func (p *RequestPool) Alloc(reqs []*Req) ([]int, bool) {
	need := 0
	for _, r := range reqs {
		if r.KV.ReqPoolIdx == nil {
			need++
		} else if r.KV.KVAllocatedLen <= 0 {
			panic("a reused row must carry allocated KV")
		}
	}
	rows, ok := p.allocRows(need)
	if !ok {
		return nil, false
	}
	out := make([]int, len(reqs))
	k := 0
	for i, r := range reqs {
		if r.KV.ReqPoolIdx == nil {
			row := rows[k]
			k++
			r.KV.ReqPoolIdx = &row
		}
		out[i] = *r.KV.ReqPoolIdx
	}
	return out, true
}

// allocRows is memory_pool.py:438-453@297e54fd: the last need rows of the free list, in order.
func (p *RequestPool) allocRows(need int) ([]int, bool) {
	if need > len(p.free) {
		return nil, false
	}
	if need == 0 {
		return []int{}, true
	}
	rows := slices.Clone(p.free[len(p.free)-need:])
	p.free = p.free[:len(p.free)-need]
	for _, r := range rows {
		p.generation[r]++
	}
	return rows, true
}

// Free is memory_pool.py:463-466@297e54fd.
func (p *RequestPool) Free(req *Req) {
	if req.KV.ReqPoolIdx == nil {
		panic("request must have req_pool_idx")
	}
	p.free = append(p.free, *req.KV.ReqPoolIdx)
	req.KV.ReqPoolIdx = nil
}

// Clear is memory_pool.py:468-475@297e54fd. The generations stay: readers detect a row's reuse
// by them.
func (p *RequestPool) Clear() {
	p.free = make([]int, p.size)
	for i := range p.free {
		p.free[i] = i + 1
	}
}

// Generation is how many times row has been handed out.
func (p *RequestPool) Generation(row int) int64 { return p.generation[row] }

// Write is write (memory_pool.py:414-415@297e54fd) on one run of one row.
func (p *RequestPool) Write(row, start int, values []int64) {
	if row < 0 || row > p.size || start < 0 || start+len(values) > p.maxContextLen {
		panic(fmt.Sprintf("write of %d slots at row %d position %d outside a %dx%d pool", len(values), row,
			start, p.size+1, p.maxContextLen))
	}
	r := p.rows[row]
	if end := start + len(values); end > len(r) {
		r = append(r, make([]int64, end-len(r))...)
	}
	copy(r[start:], values)
	p.rows[row] = r
}

// Row is the slots written to row, from position 0 to the last position written.
func (p *RequestPool) Row(row int) []int64 { return slices.Clone(p.rows[row]) }

// SchedulableTokenCapacity is memory_pool.py:485-488@297e54fd: with no aux cache, the physical
// capacity.
func (p *RequestPool) SchedulableTokenCapacity(physicalCapacity int) int { return physicalCapacity }

// AllocAuxToLengths is memory_pool.py:490-500@297e54fd: nothing with no aux cache.
func (p *RequestPool) AllocAuxToLengths(reqPoolIndices, targetSeqLens []int) {}

// FreeRows is free_slots: the rows not held, in the order they will be taken from the tail.
func (p *RequestPool) FreeRows() []int { return slices.Clone(p.free) }

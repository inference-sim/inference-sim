// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/memory_pool.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Also derived from, at the same release: python/sglang/srt/configs/mamba_utils.py,
// python/sglang/srt/configs/qwen3_next.py, python/sglang/srt/mem_cache/allocator/mamba.py,
// python/sglang/srt/mem_cache/allocation_sizing.py and
// python/sglang/srt/mem_cache/kv_cache_configurator.py (_handle_max_mamba_cache,
// _calculate_mamba_ratio, resolve_max_num_reqs).

package cache

import (
	"fmt"
	"slices"
)

// LinearAttnConfig is the part of a Gated-DeltaNet hybrid model's text configuration that sizes
// its recurrent state (configs/qwen3_next.py:272-313@297e54fd): Qwen3.5, Qwen3.8's attention
// layers and Qwen3-Next. Every FullAttentionInterval-th layer is full attention, the others carry
// the recurrent and convolution state (layers_block_type, qwen3_next.py:262-270).
type LinearAttnConfig struct {
	NumHiddenLayers       int
	FullAttentionInterval int
	KeyHeadDim            int // linear_key_head_dim
	NumKeyHeads           int // linear_num_key_heads
	ValueHeadDim          int // linear_value_head_dim
	NumValueHeads         int // linear_num_value_heads
	ConvKernel            int // linear_conv_kernel_dim
}

// LinearLayerIDs is linear_layer_ids (qwen3_next.py:272-278@297e54fd).
func (c LinearAttnConfig) LinearLayerIDs() []int { return c.layerIDs(false) }

// FullAttentionLayerIDs is full_attention_layer_ids (qwen3_next.py:280-286@297e54fd).
func (c LinearAttnConfig) FullAttentionLayerIDs() []int { return c.layerIDs(true) }

func (c LinearAttnConfig) layerIDs(full bool) []int {
	var ids []int
	for l := 0; l < c.NumHiddenLayers; l++ {
		if ((l+1)%c.FullAttentionInterval == 0) == full {
			ids = append(ids, l)
		}
	}
	return ids
}

// LinearStateParams is Mamba2CacheParams (configs/mamba_utils.py:294-296@297e54fd) for a
// Gated-DeltaNet model: the shape of one request's convolution and recurrent state in one layer,
// their dtypes, and the layers that hold it.
type LinearStateParams struct {
	// Conv is the convolution state shapes of one layer, each (dim, kernel-1).
	Conv [][2]int
	// Temporal is the recurrent state shape of one layer: (value heads, value head dim, key head dim).
	Temporal                   [3]int
	ConvDType, TemporalDType   DType
	Layers                     []int
	NumKHeadsPerTP, ConvKernel int
}

// MambaStateDType is mamba2_state_dtype (mamba_utils.py:48-110@297e54fd) with the environment
// variable it reads set from --mamba-ssm-dtype (arg_groups/serving_hook.py:439): the flag
// overrides the model configuration's mamba_ssm_dtype, which overrides float32. The convolution
// state is bfloat16 (SGLANG_MAMBA_CONV_DTYPE is not an input). An unknown name is rejected by
// the parser; here it is an error.
func MambaStateDType(flag *string, config string) (conv, temporal DType, err error) {
	known := map[string]DType{"float32": Float32, "bfloat16": BFloat16, "float16": Float16}
	temporal = Float32
	if d, ok := known[config]; ok {
		temporal = d
	}
	if flag != nil {
		d, ok := known[*flag]
		if !ok {
			return "", "", fmt.Errorf("--mamba-ssm-dtype %q: choose from float32, bfloat16, float16", *flag)
		}
		temporal = d
	}
	return BFloat16, temporal, nil
}

// NewLinearStateParams is Qwen3NextConfig.mamba2_cache_params (qwen3_next.py:288-313@297e54fd)
// through Mamba2StateShape.create (mamba_utils.py:150-201): tp is linear_attn_tp_size, the
// attention tensor-parallel size.
func NewLinearStateParams(c LinearAttnConfig, tp int, convDType, ssmDType DType) (LinearStateParams, error) {
	if tp < 1 || c.FullAttentionInterval < 1 || c.ConvKernel < 2 {
		return LinearStateParams{}, fmt.Errorf("linear attention state: tp %d, interval %d, kernel %d",
			tp, c.FullAttentionInterval, c.ConvKernel)
	}
	nGroups := c.NumKeyHeads
	intermediate := c.ValueHeadDim * c.NumValueHeads
	kHeads := nGroups / tp
	if nGroups%tp != 0 {
		kHeads = (nGroups + tp - 1) / tp
		nGroups += tp - nGroups // extra_groups_for_head_shards
	}
	convDim := intermediate + 2*nGroups*c.KeyHeadDim
	if convDim%tp != 0 || c.NumValueHeads%tp != 0 {
		return LinearStateParams{}, fmt.Errorf("conv dim %d and %d value heads must divide by tp %d", convDim,
			c.NumValueHeads, tp)
	}
	return LinearStateParams{
		Conv:           [][2]int{{convDim / tp, c.ConvKernel - 1}},
		Temporal:       [3]int{c.NumValueHeads / tp, c.ValueHeadDim, c.KeyHeadDim},
		ConvDType:      convDType,
		TemporalDType:  ssmDType,
		Layers:         c.LinearLayerIDs(),
		NumKHeadsPerTP: kHeads,
		ConvKernel:     c.ConvKernel,
	}, nil
}

func (p LinearStateParams) convNumel() int64 {
	var n int64
	for _, s := range p.Conv {
		n += int64(s[0]) * int64(s[1])
	}
	return n
}

func (p LinearStateParams) temporalNumel() int64 {
	return int64(p.Temporal[0]) * int64(p.Temporal[1]) * int64(p.Temporal[2])
}

// PerReqBytes is mamba_cache_per_req (mamba_utils.py:116-124@297e54fd): one request's state over
// every linear layer.
func (p LinearStateParams) PerReqBytes() int64 {
	return (p.convNumel()*int64(p.ConvDType.ItemSize()) + p.temporalNumel()*int64(p.TemporalDType.ItemSize())) *
		int64(len(p.Layers))
}

// ReplayRingBytesPerReq is replayssm_ring_bytes_per_req for the Gated-DeltaNet gate
// (mamba_utils.py:126-160@297e54fd): the compact d, k and scalar-g rings, with the low parts when
// the convolution dtype is not float32.
func (p LinearStateParams) ReplayRingBytesPerReq(recordLen int) int64 {
	hv, v, k := int64(p.Temporal[0]), int64(p.Temporal[1]), int64(p.Temporal[2])
	hk, conv, l := int64(p.NumKHeadsPerTP), int64(p.ConvDType.ItemSize()), int64(recordLen)
	per := hv*l*v*conv + hk*l*k*conv + hv*l*4
	if p.ConvDType != Float32 {
		per += hv*l*v*conv + hk*l*k*conv
	}
	return per * int64(len(p.Layers))
}

// MambaPoolParams is MambaPool's constructor arguments that size it (memory_pool.py:671-690
// @297e54fd). SpeculativeDraftTokens zero is unset (None). The envelope layout (page-major KV) and
// the Kimi-linear gate are refused.
type MambaPoolParams struct {
	Size, SpecStateSize    int
	State                  LinearStateParams
	SpeculativeDraftTokens int
	EagleTopK              int
	ReplayDecode           bool // enable_linear_replayssm
	ReplaySpec             bool // enable_linear_replayssm_spec
	ReplayCacheLen         int  // linear_replayssm_cache_len
}

// DedupConvWindow is conv_window_dedup_enabled (memory_pool.py:135-150@297e54fd) on CUDA for a
// non-KDA gate: linear draft chains only, so tree verify (top-k above one) keeps dense windows.
func DedupConvWindow(eagleTopK int) bool { return eagleTopK <= 1 }

// MambaPoolBytes is MambaPool.mem_usage_bytes (memory_pool.py:1086-1099@297e54fd) on CUDA: the
// convolution and recurrent state of size+1 slots over every linear layer, the ReplaySSM rings,
// and, under speculation, the per-draft intermediate state and conv windows, the deduplicated
// sliding-window buffer charged at its physical size (not its over-reporting view).
func MambaPoolBytes(p MambaPoolParams) (int64, error) {
	st := p.State
	if p.Size < 0 || p.SpecStateSize < 0 || len(st.Layers) == 0 {
		return 0, fmt.Errorf("MambaPool: size %d, spec state %d, %d layers", p.Size, p.SpecStateSize, len(st.Layers))
	}
	if p.ReplayDecode && p.ReplaySpec {
		return 0, fmt.Errorf("--enable-linear-replayssm and --enable-linear-replayssm-spec are mutually exclusive (attention_hook.py:487)")
	}
	layers := int64(len(st.Layers))
	slots := int64(p.Size) + 1
	specSlots := int64(p.SpecStateSize) + 1
	convB, ssmB := int64(st.ConvDType.ItemSize()), int64(st.TemporalDType.ItemSize())
	total := layers*slots*st.convNumel()*convB + layers*slots*st.temporalNumel()*ssmB

	if p.ReplayDecode || p.ReplaySpec {
		hv, v, k := int64(st.Temporal[0]), int64(st.Temporal[1]), int64(st.Temporal[2])
		hk, l := int64(st.NumKHeadsPerTP), int64(p.ReplayCacheLen)
		num := slots
		ring := ssmB
		if p.ReplaySpec {
			num, ring = specSlots, convB
		}
		total += layers * num * hv * l * v * ring // replayssm_d
		total += layers * num * hk * l * k * ring // replayssm_k
		total += layers * num * hv * l * 4        // replayssm_g (float32 scalar gate)
		if p.ReplaySpec && ring != 4 {
			total += layers * num * hv * l * v * convB // replayssm_rawv
			total += layers * num * hk * l * k * convB // replayssm_rawk
		}
	}
	if d := int64(p.SpeculativeDraftTokens); d > 0 {
		if !p.ReplaySpec {
			total += layers * specSlots * d * st.temporalNumel() * ssmB // intermediate_ssm
		}
		for _, s := range st.Conv {
			if s[1] != st.ConvKernel-1 {
				return 0, fmt.Errorf("conv state shape %v has no window axis of length %d", s, st.ConvKernel-1)
			}
			if DedupConvWindow(p.EagleTopK) {
				total += layers * specSlots * int64(s[0]) * (d + int64(s[1]) - 1) * convB
			} else {
				total += layers * specSlots * d * int64(s[0]) * int64(s[1]) * convB
			}
		}
	}
	return total, nil
}

// MambaTrackingSlots is get_mamba_tracking_slots (allocation_sizing.py:11-19@297e54fd).
func MambaTrackingSlots(extraBuffer, overlap, lazy bool) int {
	switch {
	case !extraBuffer:
		return 0
	case lazy:
		return 1
	case overlap:
		return 2
	}
	return 1
}

// MambaSlotAllocator is MambaSlotAllocator (allocator/mamba.py:30-90@297e54fd): the free list of
// state slots, one per request. Slot 0 is the padding slot and is never handed out.
type MambaSlotAllocator struct {
	size int
	free []int64
	// group is the slots alloc_group_begin took ahead; alloc(1) drains it.
	group  []int64
	inside bool
}

// NewMambaSlotAllocator is MambaSlotAllocator.__init__ (mamba.py:34-40).
func NewMambaSlotAllocator(size int) *MambaSlotAllocator {
	a := &MambaSlotAllocator{size: size}
	a.Clear()
	return a
}

// AvailableSize is mamba.py:42-43.
func (a *MambaSlotAllocator) AvailableSize() int { return len(a.free) }

// Free returns the free list in order.
func (a *MambaSlotAllocator) FreeSlots() []int64 { return slices.Clone(a.free) }

// AllocGroupBegin is mamba.py:49-56: take numReqs slots ahead, if that many are free.
func (a *MambaSlotAllocator) AllocGroupBegin(numReqs int) {
	a.group, a.inside = nil, false
	if numReqs > 0 {
		if got, ok := a.doAlloc(numReqs); ok {
			a.group, a.inside = got, true
		}
	}
}

// AllocGroupEnd is mamba.py:58-64: free the slots the group did not use.
func (a *MambaSlotAllocator) AllocGroupEnd() {
	if a.inside && len(a.group) > 0 {
		a.Free(a.group)
	}
	a.group, a.inside = nil, false
}

// Alloc is mamba.py:66-71.
func (a *MambaSlotAllocator) Alloc(need int) ([]int64, bool) {
	if a.inside && need == 1 && len(a.group) > 0 {
		s := a.group[0]
		a.group = a.group[1:]
		return []int64{s}, true
	}
	return a.doAlloc(need)
}

func (a *MambaSlotAllocator) doAlloc(need int) ([]int64, bool) {
	if need > len(a.free) {
		return nil, false
	}
	got := slices.Clone(a.free[:need])
	a.free = a.free[need:]
	return got, true
}

// Free is mamba.py:78-82: the slots go to the tail.
func (a *MambaSlotAllocator) Free(slots []int64) { a.free = append(a.free, slots...) }

// Clear is mamba.py:84-89.
func (a *MambaSlotAllocator) Clear() {
	a.free = make([]int64, a.size)
	for i := range a.free {
		a.free[i] = int64(i + 1)
	}
}

// MambaKV is the mamba state a request holds (ReqKvInfo, schedule_batch.py:965-990@297e54fd).
// PoolIdx and the ping-pong entries are mamba slots; -1 in the track buffer is an empty entry.
type MambaKV struct {
	PoolIdx         *int64
	PingPongBuffer  []int64
	PingPongEmpty   []bool
	NextTrackIdx    *int
	LastTrackIdx    *int
	LastTrackSeqlen *int
	PrevTrackSeqlen *int
	NeedsClear      bool
	CowSrcIndex     *int64
}

// HoldsMamba is ReqKvInfo.holds_mamba (schedule_batch.py:1007).
func (m *MambaKV) HoldsMamba() bool { return m != nil && m.PoolIdx != nil }

// HybridRequestPool is HybridReqToTokenPool (memory_pool.py:1355-1880@297e54fd): the request pool
// with a mamba slot per request, and, with the extra buffer, tracking slots (ping-pong) that the
// radix cache takes over. The state tensors are not modelled; their slots are.
type HybridRequestPool struct {
	*RequestPool
	Mamba             *MambaSlotAllocator
	extraBuffer, lazy bool
	pingPongSize      int
	initialTracking   int
	reqToMamba        []int64
	pingPongMap       [][]int64
}

// NewHybridRequestPool is HybridReqToTokenPool.__init__ (memory_pool.py:1380-1425).
func NewHybridRequestPool(size, mambaSize, maxContextLen int, extraBuffer, lazy, overlap bool) *HybridRequestPool {
	p := &HybridRequestPool{
		RequestPool: NewRequestPool(size, maxContextLen), Mamba: NewMambaSlotAllocator(mambaSize),
		extraBuffer: extraBuffer, lazy: lazy,
		pingPongSize:    MambaTrackingSlots(true, overlap, false),
		initialTracking: MambaTrackingSlots(extraBuffer, overlap, lazy),
		reqToMamba:      make([]int64, size+1),
	}
	if extraBuffer {
		p.pingPongMap = make([][]int64, size+1)
		for i := range p.pingPongMap {
			p.pingPongMap[i] = make([]int64, p.pingPongSize)
		}
	}
	return p
}

// PingPongSize is mamba_ping_pong_track_buffer_size.
func (p *HybridRequestPool) PingPongSize() int { return p.pingPongSize }

// InitialTrackingSlots is mamba_initial_tracking_slots.
func (p *HybridRequestPool) InitialTrackingSlots() int { return p.initialTracking }

// ReqToMamba is req_index_to_mamba_index_mapping.
func (p *HybridRequestPool) ReqToMamba() []int64 { return slices.Clone(p.reqToMamba) }

// PingPongMap is req_index_to_mamba_ping_pong_track_buffer_mapping.
func (p *HybridRequestPool) PingPongMap() [][]int64 {
	out := make([][]int64, len(p.pingPongMap))
	for i, r := range p.pingPongMap {
		out[i] = slices.Clone(r)
	}
	return out
}

// ErrMambaExhausted is StateSlotsExhausted, which StateStorage.acquire raises when the one
// allocation of every slot the batch needs finds too few (kv/state_storage.py:155-161@c4af57ff).
var ErrMambaExhausted = fmt.Errorf("not enough space for mamba cache or its ping pong idx, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size")

// Alloc is acquire_rows (static_kv_storage.py:98-127@c4af57ff) over HybridReqToTokenPool: rows as
// the request pool gives them, then every slot the batch needs from one allocation (state.acquire,
// kv/state_storage.py:121-185). The first slots are the working slots of the requests that hold
// none, in request order; the rest are the tracking slots of the requests with no ping-pong
// buffer, each its initial count in turn. If the slots run out, the rows taken go back before
// the exhaustion surfaces, so a failure leaves no request bound.
func (p *HybridRequestPool) Alloc(reqs []*Req) ([]int, error) {
	fresh := make([]bool, len(reqs))
	for i, r := range reqs {
		fresh[i] = r.KV.ReqPoolIdx == nil
	}
	rows, ok := p.RequestPool.Alloc(reqs)
	if !ok {
		return nil, nil
	}
	var needing, untracked []*Req
	for _, r := range reqs {
		if r.KV.Mamba == nil {
			r.KV.Mamba = &MambaKV{}
		}
		if !r.KV.Mamba.HoldsMamba() {
			needing = append(needing, r)
		}
		if p.extraBuffer && r.KV.Mamba.PingPongBuffer == nil {
			untracked = append(untracked, r)
		}
	}
	per := 0
	if len(untracked) > 0 {
		per = p.initialTracking
	}
	if count := len(needing) + per*len(untracked); count > 0 {
		slots, ok := p.Mamba.Alloc(count)
		if !ok {
			for i, r := range reqs {
				if fresh[i] {
					p.RequestPool.Free(r)
				}
			}
			return nil, ErrMambaExhausted
		}
		for i, r := range needing {
			idx := slots[i]
			r.KV.Mamba.PoolIdx = &idx
			r.KV.Mamba.NeedsClear = true
		}
		rest := slots[len(needing):]
		for i, r := range untracked {
			p.handOutTracking(r.KV.Mamba, rest[i*per:(i+1)*per])
		}
	}
	for _, r := range reqs {
		row := *r.KV.ReqPoolIdx
		p.reqToMamba[row] = *r.KV.Mamba.PoolIdx
		if p.extraBuffer {
			copy(p.pingPongMap[row], r.KV.Mamba.PingPongBuffer)
		}
	}
	return rows, nil
}

// handOutTracking is _hand_out_tracking (kv/state_storage.py:235-258@c4af57ff): the request's
// buffer is its slots, the rest of the buffer -1; the last tracked index is the other one unless
// lazy.
func (p *HybridRequestPool) handOutTracking(m *MambaKV, slots []int64) {
	buf := make([]int64, p.pingPongSize)
	for i := range buf {
		buf[i] = -1
	}
	copy(buf, slots)
	m.PingPongBuffer = buf
	m.PingPongEmpty = make([]bool, p.pingPongSize)
	for i := len(slots); i < p.pingPongSize; i++ {
		m.PingPongEmpty[i] = true
	}
	m.PrevTrackSeqlen = nil
	zero := 0
	m.NextTrackIdx = &zero
	last := p.PingPongOtherIdx(0)
	if p.lazy {
		last = 0
	}
	m.LastTrackIdx = &last
}

// PingPongOtherIdx is get_mamba_ping_pong_other_idx (memory_pool.py:1743-1747).
func (p *HybridRequestPool) PingPongOtherIdx(next int) int {
	if p.pingPongSize == 2 {
		return 1 - next
	}
	return next
}

// PingPongKeepIdx is get_mamba_ping_pong_keep_idx (memory_pool.py:1749-1751).
func (p *HybridRequestPool) PingPongKeepIdx(m *MambaKV) int { return *m.LastTrackIdx }

// SetPingPongSlot is set_mamba_ping_pong_slot (memory_pool.py:1784-1801): a slot fills the entry,
// -1 empties it, and the row's map follows.
func (p *HybridRequestPool) SetPingPongSlot(row int, m *MambaKV, idx int, value int64) {
	m.PingPongBuffer[idx] = value
	copy(p.pingPongMap[row], m.PingPongBuffer)
	m.PingPongEmpty[idx] = value == -1
}

// DonatePingPongSlot is donate_mamba_ping_pong_slot (memory_pool.py:1803-1824): the kept
// tracking slot goes to the radix cache and newSlot takes its place.
func (p *HybridRequestPool) DonatePingPongSlot(row int, m *MambaKV, newSlot int64) int64 {
	idx := p.PingPongKeepIdx(m)
	donated := m.PingPongBuffer[idx]
	p.SetPingPongSlot(row, m, idx, newSlot)
	return donated
}

// FreeMambaCache is free_mamba_cache (memory_pool.py:1826-1861): the request's slot goes back,
// and with the extra buffer so do its tracking slots except keep (nil frees all).
func (p *HybridRequestPool) FreeMambaCache(row int, m *MambaKV, keep *int) error {
	if !m.HoldsMamba() {
		return fmt.Errorf("double free? mamba_index is None")
	}
	p.Mamba.Free([]int64{*m.PoolIdx})
	m.PoolIdx = nil
	if !p.extraBuffer {
		return nil
	}
	if keep != nil && *keep != 0 && (*keep != 1 || p.pingPongSize != 2) {
		return fmt.Errorf("cannot keep ping-pong entry %d of %d", *keep, p.pingPongSize)
	}
	var positions []int
	for i := 0; i < p.pingPongSize; i++ {
		if (keep == nil || i != *keep) && !m.PingPongEmpty[i] {
			positions = append(positions, i)
		}
	}
	mapRow := p.pingPongMap[row]
	if len(positions) == p.pingPongSize {
		p.Mamba.Free(slices.Clone(mapRow))
	} else {
		for _, i := range positions {
			p.Mamba.Free([]int64{mapRow[i]})
		}
	}
	m.PingPongBuffer, m.PingPongEmpty, m.NextTrackIdx, m.LastTrackIdx = nil, nil, nil, nil
	m.LastTrackSeqlen, m.PrevTrackSeqlen, m.CowSrcIndex = nil, nil, nil
	m.NeedsClear = false
	return nil
}

// Clear is HybridReqToTokenPool.clear (memory_pool.py:1863-1880).
func (p *HybridRequestPool) Clear() {
	p.RequestPool.Clear()
	p.Mamba.Clear()
	clear(p.reqToMamba)
	for _, r := range p.pingPongMap {
		clear(r)
	}
}

// HybridKVParams is HybridLinearKVPool's constructor arguments that size an MHA full-attention
// side (memory_pool.py:4228-4300@297e54fd). The MLA and DSA sides are refused with their pools.
type HybridKVParams struct {
	Size, PageSize      int
	DType               DType
	HeadNum, HeadDim    int
	FullAttentionLayers []int
}

// HybridKVPool is HybridLinearKVPool (memory_pool.py:4228@297e54fd) as sizing: an MHA pool dense
// over the full-attention layers only; the linear layers' state is the request pool's MambaPool.
type HybridKVPool struct {
	*MHAPool
	layerIDs []int
}

// NewHybridKVPool is HybridLinearKVPool.__init__ (memory_pool.py:4259-4345) on the MHA path.
func NewHybridKVPool(p HybridKVParams) (*HybridKVPool, error) {
	if len(p.FullAttentionLayers) == 0 {
		return nil, fmt.Errorf("HybridLinearKVPool: a model with no full-attention layer needs no KV pool")
	}
	mha, err := NewMHAPool(MHAParams{Size: p.Size, PageSize: p.PageSize, DType: p.DType, HeadNum: p.HeadNum,
		HeadDim: p.HeadDim, LayerNum: len(p.FullAttentionLayers)})
	if err != nil {
		return nil, err
	}
	return &HybridKVPool{MHAPool: mha, layerIDs: slices.Clone(p.FullAttentionLayers)}, nil
}

// FullAttentionLayerIDs is the model layers the pool holds, in order.
func (h *HybridKVPool) FullAttentionLayerIDs() []int { return slices.Clone(h.layerIDs) }

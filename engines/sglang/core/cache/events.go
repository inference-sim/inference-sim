// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/events.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Also derived from, at the same release: python/sglang/srt/mem_cache/utils.py (the node hashes),
// python/sglang/srt/mem_cache/cpp_utils/hash_binding.cpp (the page hash) and
// python/sglang/srt/disaggregation/kv_events.py (the event namespace).

package cache

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
)

// pageHashes is get_native_hash with a page size (cpp_utils/hash_binding.cpp:224-275
// @297e54fd): the chained SHA-256 of each page of k, each page's digest the prior of the next.
// A page is its tokens as little-endian uint32, or, when a token does not fit in 32 bits, the
// byte 0xff then its tokens as little-endian uint64. A bigram page is its pairs of tokens.
func pageHashes(k Key, prior []byte, pageSize int) []string {
	n := k.Len()
	var out []string
	for start := 0; start < n; start += pageSize {
		units := min(pageSize, n-start)
		var words []uint64
		if k.Bigram {
			for i := 0; i < units; i++ {
				words = append(words, uint64(k.Tokens[start+i]), uint64(k.Tokens[start+i+1]))
			}
		} else {
			for i := 0; i < units; i++ {
				words = append(words, uint64(k.Tokens[start+i]))
			}
		}
		prior = hashPage(prior, words, k.Tokens, start, units, k.Bigram)
		out = append(out, hex.EncodeToString(prior))
	}
	return out
}

func hashPage(prior []byte, words []uint64, raw []int64, start, units int, bigram bool) []byte {
	rawCount := units
	if bigram {
		rawCount = units + 1
	}
	wide := false
	for _, t := range raw[start : start+rawCount] {
		wide = wide || uint64(t) > math.MaxUint32
	}
	h := sha256.New()
	h.Write(prior)
	if wide {
		// hash_wide_page (:101-116): the tag byte, then 64-bit words.
		h.Write([]byte{0xff})
		b := make([]byte, 8*len(words))
		for i, w := range words {
			binary.LittleEndian.PutUint64(b[8*i:], w)
		}
		h.Write(b)
	} else {
		b := make([]byte, 4*len(words))
		for i, w := range words {
			binary.LittleEndian.PutUint32(b[4*i:], uint32(w))
		}
		h.Write(b)
	}
	return h.Sum(nil)
}

// namespaceSeed is storage_namespace_seed (mem_cache/utils.py:125-139@297e54fd), and with
// eventTag kv_event_namespace_seed (disaggregation/kv_events.py:312-326@297e54fd).
func namespaceSeed(tag string, extraKey, cacheSalt *string) []byte {
	if extraKey == nil && cacheSalt == nil {
		return nil
	}
	h := sha256.New()
	h.Write([]byte(tag))
	for _, part := range []*string{extraKey, cacheSalt} {
		if part == nil {
			h.Write([]byte{0})
			continue
		}
		var n [8]byte
		binary.LittleEndian.PutUint64(n[:], uint64(len(*part)))
		h.Write([]byte{1})
		h.Write(n[:])
		h.Write([]byte(*part))
	}
	return h.Sum(nil)
}

const (
	storageNamespaceTag = "sglang-cache-namespace-v1"
	eventNamespaceTag   = "sglang-kv-event-namespace-v1"
)

// hashStrToInt64 is hash_str_to_int64 (mem_cache/utils.py:155-163@297e54fd): the first 64 bits
// of a hex digest as a signed integer.
func hashStrToInt64(h string) int64 {
	b, _ := hex.DecodeString(h[:16])
	return int64(binary.BigEndian.Uint64(b))
}

// namespacedBlockHash is namespaced_block_hash (disaggregation/kv_events.py:329-340@297e54fd).
func namespacedBlockHash(h int64, seed []byte) int64 {
	if seed == nil {
		return h
	}
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], uint64(h))
	d := sha256.Sum256(append(append([]byte{}, seed...), b[:]...))
	return int64(binary.BigEndian.Uint64(d[:8]))
}

func sameNamespace(a, b Key) bool {
	return eqStr(a.ExtraKey, b.ExtraKey) && eqStr(a.CacheSalt, b.CacheSalt)
}

func eqStr(a, b *string) bool {
	return (a == nil && b == nil) || (a != nil && b != nil && *a == *b)
}

// nodeHashValues is compute_node_hash_values (mem_cache/utils.py:166-175@297e54fd), cached on
// the node as SGLang caches it.
func (t *treeCore) nodeHashValues(n *node) []string {
	if n.hashValue != nil {
		return n.hashValue
	}
	var prior []byte
	if p := n.parent; p != nil && p.hashValue != nil && p.key.Len() > 0 && len(p.hashValue) > 0 {
		prior, _ = hex.DecodeString(p.hashValue[len(p.hashValue)-1])
	} else if s := namespaceSeed(storageNamespaceTag, n.key.ExtraKey, n.key.CacheSalt); s != nil {
		// get_storage_hash_str (:142-152) seeds a chain with the namespace: the seed's hex digest
		// is the prior hash.
		prior = s
	}
	n.hashValue = pageHashes(n.key, prior, t.pageSize)
	return n.hashValue
}

// nodeEventHashValues is compute_node_event_hash_values (mem_cache/utils.py:178-214@297e54fd).
// For a namespaced node SGLang's _node_event_hash_values (events.py:172-179) also computes and
// caches the storage hash_value, which only a host tier or storage reads; the FULL device port
// has neither, so it computes the storage hashes when it first needs them instead.
func (t *treeCore) nodeEventHashValues(n *node) []string {
	if n.key.ExtraKey == nil && n.key.CacheSalt == nil {
		return t.nodeHashValues(n)
	}
	if n.eventHashValue != nil {
		return n.eventHashValue
	}
	var missing []*node
	cur := n
	for cur != nil && cur.key.Len() > 0 {
		if !sameNamespace(cur.key, n.key) {
			panic("cache: radix path contains mismatched cache namespaces")
		}
		if cur.eventHashValue != nil {
			break
		}
		missing = append(missing, cur)
		cur = cur.parent
	}
	var prior []byte
	switch {
	case cur != nil && len(cur.eventHashValue) > 0:
		prior, _ = hex.DecodeString(cur.eventHashValue[len(cur.eventHashValue)-1])
	case n.key.CacheSalt != nil:
		d := sha256.Sum256(append([]byte("sglang-cache-salt-v1\x00"), []byte(*n.key.CacheSalt)...))
		prior = d[:]
	}
	for i := len(missing) - 1; i >= 0; i-- {
		hv := pageHashes(missing[i].key, prior, t.pageSize)
		missing[i].eventHashValue = hv
		if len(hv) > 0 {
			prior, _ = hex.DecodeString(hv[len(hv)-1])
		}
	}
	return n.eventHashValue
}

// eventRecorder is KVCacheEventRecorder (events.py:120-320@297e54fd) with no LoRA adapters: one
// BlockStored per page of a node, merged into the queue's last event as enqueue merges it.
type eventRecorder struct {
	enabled      bool
	pageSize     int
	dynamoFormat bool
	queue        []Event
}

// enqueue is KVCacheEventRecorder.enqueue (events.py:140-170@297e54fd), SGL-5.
func (r *eventRecorder) enqueue(e Event) {
	if len(r.queue) > 0 {
		tail := &r.queue[len(r.queue)-1]
		switch {
		case tail.Type == "BlockRemoved" && e.Type == "BlockRemoved":
			if tail.Medium == e.Medium {
				tail.BlockHashes = append(tail.BlockHashes, e.BlockHashes...)
				return
			}
		case tail.Type == "BlockStored" && e.Type == "BlockStored":
			if tail.Medium == e.Medium && eqStr(tail.LoraID, e.LoraID) && tail.BlockSize == e.BlockSize &&
				eqStr(tail.CacheSalt, e.CacheSalt) && eqStr(tail.SessionID, e.SessionID) &&
				eqStr(tail.LoraName, e.LoraName) && len(tail.BlockHashes) > 0 && e.ParentBlockHash != nil &&
				*e.ParentBlockHash == tail.BlockHashes[len(tail.BlockHashes)-1] {
				tail.BlockHashes = append(tail.BlockHashes, e.BlockHashes...)
				tail.TokenIDs = append(tail.TokenIDs, e.TokenIDs...)
				return
			}
		}
	}
	r.queue = append(r.queue, e)
}

func (t *treeCore) eventSeed(n *node) []byte {
	if !t.events.dynamoFormat {
		// Without a LoRA name table entry, extra_key does not namespace a published hash
		// (events.py:198-208); cache_salt never does in this format.
		return nil
	}
	return namespaceSeed(eventNamespaceTag, n.key.ExtraKey, n.key.CacheSalt)
}

// recordStore is KVCacheEventRecorder.record_store (events.py:213-271@297e54fd).
func (t *treeCore) recordStore(n *node, sessionID *string) {
	r := &t.events
	if !r.enabled {
		return
	}
	hv := t.nodeEventHashValues(n)
	seed := t.eventSeed(n)
	var parent *int64
	if p := n.parent; p != nil && p.parent != nil {
		phv := p.hashValue
		if n.key.ExtraKey != nil || n.key.CacheSalt != nil {
			phv = p.eventHashValue
		}
		if len(phv) > 0 {
			h := namespacedBlockHash(hashStrToInt64(phv[len(phv)-1]), seed)
			parent = &h
		}
	}
	page := 0
	for start := 0; start < n.key.Len(); start += r.pageSize {
		end := min(start+r.pageSize, n.key.Len())
		var toks []int64
		if n.key.Bigram {
			// bigram pages publish their pairs (events.py:243-246); flattened here.
			for j := start; j < end; j++ {
				toks = append(toks, n.key.Tokens[j], n.key.Tokens[j+1])
			}
		} else {
			toks = append(toks, n.key.Tokens[start:end]...)
		}
		h := namespacedBlockHash(hashStrToInt64(hv[page]), seed)
		r.enqueue(Event{Type: "BlockStored", BlockHashes: []int64{h}, ParentBlockHash: parent, TokenIDs: toks,
			BlockSize: end - start, Medium: "GPU", CacheSalt: n.key.CacheSalt, SessionID: sessionID})
		hh := h
		parent = &hh
		page++
	}
}

// recordRemove is KVCacheEventRecorder.record_remove (events.py:273-302@297e54fd).
func (t *treeCore) recordRemove(n *node) {
	r := &t.events
	if !r.enabled {
		return
	}
	hv := t.nodeEventHashValues(n)
	seed := t.eventSeed(n)
	var hashes []int64
	page := 0
	for start := 0; start < n.key.Len(); start += r.pageSize {
		hashes = append(hashes, namespacedBlockHash(hashStrToInt64(hv[page]), seed))
		page++
	}
	if len(hashes) > 0 {
		r.enqueue(Event{Type: "BlockRemoved", BlockHashes: hashes, Medium: "GPU"})
	}
}

func (r *eventRecorder) take() []Event {
	if !r.enabled {
		return nil
	}
	q := r.queue
	r.queue = nil
	return q
}

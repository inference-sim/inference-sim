// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_utils.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/utils/hashing.py (sha256). Hashes byte-identical to vLLM's, checked against
// hashes vLLM itself computed (testdata/block_hashes.json).

// Package kv is vLLM's KV-cache bookkeeping: the block pool, its free queue, prefix-cache
// block hashing and the per-request block tables, for full-attention models.
package kv

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
)

// BlockHash is the hash of one full KV block chained over its prefix: vLLM's BlockHash
// (kv_cache_utils.py:62), 32 bytes of SHA-256.
type BlockHash []byte

// ExtraKey is one extra component of a block's hash key, a tuple of strings: a LoRA adapter
// is ("lora", name, path) (kv_cache_utils.py:592-595), a cache salt ("cache_salt", salt).
type ExtraKey []string

// BlockKeys are a request's per-block extra keys.
type BlockKeys struct {
	// Extra applies to every block (the LoRA key).
	Extra []ExtraKey
	// CacheSalt applies to the first block only (kv_cache_utils.py:647-651).
	CacheSalt string
}

// noneHashSeed is DEFAULT_NONE_HASH_SEED, used for SHA-256 when PYTHONHASHSEED is unset
// (kv_cache_utils.py:122, 133-146).
const noneHashSeed = "vllm-none-hash"

var noneHash = sha256Pickle(pickleString(noneHashSeed))

// NoneHash is vLLM's NONE_HASH: the parent of every prefix's first block,
// sha256(pickle(seed)) (init_none_hash, kv_cache_utils.py:162-175).
func NoneHash() BlockHash { return append(BlockHash(nil), noneHash...) }

// HashBlock is hash_block_tokens (kv_cache_utils.py:695-725): sha256 of the pickled tuple
// (parent hash, the block's tokens as a tuple, extra keys or None). A nil parent is the
// NONE_HASH.
func HashBlock(parent BlockHash, tokens []int64, extra []ExtraKey) BlockHash {
	if len(parent) == 0 {
		parent = noneHash
	}
	var p pickler
	p.frame(func() {
		p.bytes(parent)
		p.intTuple(tokens)
		if len(extra) == 0 {
			p.none()
		} else {
			p.markTuple(func() {
				for _, k := range extra {
					p.stringTuple(k)
				}
			}, len(extra))
		}
		p.out = append(p.out, 0x87) // TUPLE3: (parent, tokens, extra)
		p.memoize()
	})
	return sha256Pickle(p.out)
}

// BlockHashes returns the chained hashes of every full block of tokens, as vLLM's
// request_block_hasher does (kv_cache_utils.py:941-1000) for a request with no multimodal
// input or prompt embeddings.
func BlockHashes(tokens []int64, blockSize int, keys BlockKeys) []BlockHash {
	if blockSize < 1 {
		panic("kv: block size must be positive")
	}
	n := len(tokens) / blockSize
	out := make([]BlockHash, 0, n)
	var parent BlockHash
	for i := 0; i < n; i++ {
		h := HashBlock(parent, tokens[i*blockSize:(i+1)*blockSize], keys.ExtraFor(i))
		out = append(out, h)
		parent = h
	}
	return out
}

// ExtraFor is generate_block_hash_extra_keys for block i: LoRA keys, then the cache salt
// on the first block (kv_cache_utils.py:646-658).
func (k BlockKeys) ExtraFor(i int) []ExtraKey {
	if k.CacheSalt == "" || i != 0 {
		return k.Extra
	}
	return append(append([]ExtraKey(nil), k.Extra...), ExtraKey{"cache_salt", k.CacheSalt})
}

func sha256Pickle(b []byte) BlockHash {
	s := sha256.Sum256(b)
	return s[:]
}

// pickler writes the subset of Python's pickle protocol 5 that block keys use, exactly as
// CPython's C pickler (pickle.dumps(obj, protocol=5)) writes it: PROTO 5, one FRAME, then
// the object, with MEMOIZE after each tuple, str and bytes, and STOP. vLLM hashes these
// bytes (vllm/utils/hashing.py sha256), so a single differing opcode changes every hash.
type pickler struct{ out []byte }

// frame writes PROTO 5, then the body and STOP inside one FRAME, as CPython's framer does
// for a pickle of at least 4 bytes (the frame length is the body plus STOP).
func (p *pickler) frame(body func()) {
	outer := p.out
	p.out = nil
	body()
	payload := append(p.out, '.') // STOP
	p.out = append(outer, 0x80, 0x05, 0x95)
	var l [8]byte
	binary.LittleEndian.PutUint64(l[:], uint64(len(payload)))
	p.out = append(p.out, l[:]...)
	p.out = append(p.out, payload...)
}

func (p *pickler) memoize() { p.out = append(p.out, 0x94) }
func (p *pickler) none()    { p.out = append(p.out, 'N') }

// bytes is SHORT_BINBYTES (C) below 256 bytes, else BINBYTES (B), then MEMOIZE.
func (p *pickler) bytes(b []byte) {
	if len(b) < 256 {
		p.out = append(p.out, 'C', byte(len(b)))
	} else {
		var l [4]byte
		binary.LittleEndian.PutUint32(l[:], uint32(len(b)))
		p.out = append(append(p.out, 'B'), l[:]...)
	}
	p.out = append(p.out, b...)
	p.memoize()
}

// str is SHORT_BINUNICODE (0x8c) below 256 UTF-8 bytes, else BINUNICODE (X), then MEMOIZE.
func (p *pickler) str(s string) {
	if len(s) < 256 {
		p.out = append(p.out, 0x8c, byte(len(s)))
	} else {
		var l [4]byte
		binary.LittleEndian.PutUint32(l[:], uint32(len(s)))
		p.out = append(append(p.out, 'X'), l[:]...)
	}
	p.out = append(p.out, s...)
	p.memoize()
}

// int is BININT1 (K) for 0..255, BININT2 (M) for 256..65535, BININT (J) for the rest of
// int32, and LONG1 beyond, as CPython's save_long does.
func (p *pickler) int(v int64) {
	switch {
	case v >= 0 && v <= 0xff:
		p.out = append(p.out, 'K', byte(v))
	case v >= 0 && v <= 0xffff:
		p.out = append(p.out, 'M', byte(v), byte(v>>8))
	case v >= math.MinInt32 && v <= math.MaxInt32:
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], uint32(int32(v)))
		p.out = append(append(p.out, 'J'), b[:]...)
	default:
		// LONG1: little-endian two's complement, minimal length.
		var b []byte
		x := v
		for {
			b = append(b, byte(x))
			x >>= 8
			if (x == 0 && b[len(b)-1]&0x80 == 0) || (x == -1 && b[len(b)-1]&0x80 != 0) {
				break
			}
		}
		p.out = append(p.out, 0x8a, byte(len(b)))
		p.out = append(p.out, b...)
	}
}

// intTuple writes a tuple of ints: TUPLE1/2/3 for up to three items, else MARK ... TUPLE;
// an empty tuple is EMPTY_TUPLE, which CPython does not memoize.
func (p *pickler) intTuple(vs []int64) {
	if len(vs) == 0 {
		p.out = append(p.out, ')')
		return
	}
	p.markTuple(func() {
		for _, v := range vs {
			p.int(v)
		}
	}, len(vs))
}

func (p *pickler) stringTuple(vs []string) {
	if len(vs) == 0 {
		p.out = append(p.out, ')')
		return
	}
	p.markTuple(func() {
		for _, v := range vs {
			p.str(v)
		}
	}, len(vs))
}

// markTuple writes n items and closes the tuple as CPython's save_tuple does.
func (p *pickler) markTuple(items func(), n int) {
	if n > 3 {
		p.out = append(p.out, '(') // MARK
	}
	items()
	switch n {
	case 1:
		p.out = append(p.out, 0x85) // TUPLE1
	case 2:
		p.out = append(p.out, 0x86) // TUPLE2
	case 3:
		p.out = append(p.out, 0x87) // TUPLE3
	default:
		p.out = append(p.out, 't') // TUPLE
	}
	p.memoize()
}

func pickleString(s string) []byte {
	var p pickler
	p.frame(func() { p.str(s) })
	return p.out
}

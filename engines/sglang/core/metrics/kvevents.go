// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/disaggregation/kv_events.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package metrics

import (
	"encoding/binary"
	"math"

	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
)

// EncodeKVEventBatch is the payload ZmqEventPublisher's thread sends for one batch:
// msgspec.msgpack.Encoder().encode(KVEventBatch(ts, events, attn_dp_rank)) (kv_events.py:233-240
// and :593-615@297e54fd). The batch is an array [ts, events, attn_dp_rank]. Each event is a
// tagged map, the tag first under "type", then the fields in their declared order
// (:243-310); a field left at its default None is omitted (omit_defaults), and so are the fields
// that have no default only when they are set: parent_block_hash and lora_id are always written.
// Integers take the smallest msgpack form, as msgspec writes them.
func EncodeKVEventBatch(ts float64, events []cache.Event, attnDPRank int) []byte {
	var b []byte
	b = append(b, 0x93) // fixarray of 3
	b = appendFloat(b, ts)
	b = appendArrayHeader(b, len(events))
	for _, e := range events {
		b = appendEvent(b, e)
	}
	return appendInt(b, int64(attnDPRank))
}

func appendEvent(b []byte, e cache.Event) []byte {
	switch e.Type {
	case "BlockStored":
		if e.LoraID != nil {
			panic("a BlockStored event with a LoRA adapter is not modelled (inference-sim/blis-sglang#80)")
		}
		n := 6 // type, block_hashes, parent_block_hash, token_ids, block_size, lora_id
		for _, set := range []bool{e.Medium != "", e.CacheSalt != nil, e.SessionID != nil, e.LoraName != nil} {
			if set {
				n++
			}
		}
		b = appendMapHeader(b, n)
		b = appendString(b, "type")
		b = appendString(b, "BlockStored")
		b = appendString(b, "block_hashes")
		b = appendIntArray(b, e.BlockHashes)
		b = appendString(b, "parent_block_hash")
		if e.ParentBlockHash == nil {
			b = append(b, 0xc0)
		} else {
			b = appendInt(b, *e.ParentBlockHash)
		}
		b = appendString(b, "token_ids")
		b = appendIntArray(b, e.TokenIDs)
		b = appendString(b, "block_size")
		b = appendInt(b, int64(e.BlockSize))
		b = appendString(b, "lora_id")
		b = append(b, 0xc0)
		if e.Medium != "" {
			b = appendString(b, "medium")
			b = appendString(b, e.Medium)
		}
		if e.CacheSalt != nil {
			b = appendString(b, "cache_salt")
			b = appendString(b, *e.CacheSalt)
		}
		if e.SessionID != nil {
			b = appendString(b, "session_id")
			b = appendString(b, *e.SessionID)
		}
		if e.LoraName != nil {
			b = appendString(b, "lora_name")
			b = appendString(b, *e.LoraName)
		}
	case "BlockRemoved":
		n := 2
		if e.Medium != "" {
			n++
		}
		b = appendMapHeader(b, n)
		b = appendString(b, "type")
		b = appendString(b, "BlockRemoved")
		b = appendString(b, "block_hashes")
		b = appendIntArray(b, e.BlockHashes)
		if e.Medium != "" {
			b = appendString(b, "medium")
			b = appendString(b, e.Medium)
		}
	case "AllBlocksCleared":
		b = appendMapHeader(b, 1)
		b = appendString(b, "type")
		b = appendString(b, "AllBlocksCleared")
	default:
		panic("unknown KV event " + e.Type)
	}
	return b
}

func appendFloat(b []byte, f float64) []byte {
	b = append(b, 0xcb)
	return binary.BigEndian.AppendUint64(b, math.Float64bits(f))
}

func appendArrayHeader(b []byte, n int) []byte {
	switch {
	case n < 16:
		return append(b, 0x90|byte(n))
	case n < 1<<16:
		return binary.BigEndian.AppendUint16(append(b, 0xdc), uint16(n))
	default:
		return binary.BigEndian.AppendUint32(append(b, 0xdd), uint32(n))
	}
}

func appendMapHeader(b []byte, n int) []byte {
	if n < 16 {
		return append(b, 0x80|byte(n))
	}
	return binary.BigEndian.AppendUint16(append(b, 0xde), uint16(n))
}

func appendString(b []byte, s string) []byte {
	n := len(s)
	switch {
	case n < 32:
		b = append(b, 0xa0|byte(n))
	case n < 1<<8:
		b = append(b, 0xd9, byte(n))
	case n < 1<<16:
		b = binary.BigEndian.AppendUint16(append(b, 0xda), uint16(n))
	default:
		b = binary.BigEndian.AppendUint32(append(b, 0xdb), uint32(n))
	}
	return append(b, s...)
}

func appendIntArray(b []byte, xs []int64) []byte {
	b = appendArrayHeader(b, len(xs))
	for _, x := range xs {
		b = appendInt(b, x)
	}
	return b
}

// appendInt writes v in the smallest msgpack integer form, unsigned for v >= 0.
func appendInt(b []byte, v int64) []byte {
	switch {
	case v >= 0 && v < 1<<7:
		return append(b, byte(v))
	case v >= 0 && v < 1<<8:
		return append(b, 0xcc, byte(v))
	case v >= 0 && v < 1<<16:
		return binary.BigEndian.AppendUint16(append(b, 0xcd), uint16(v))
	case v >= 0 && v < 1<<32:
		return binary.BigEndian.AppendUint32(append(b, 0xce), uint32(v))
	case v >= 0:
		return binary.BigEndian.AppendUint64(append(b, 0xcf), uint64(v))
	case v >= -32:
		return append(b, byte(v))
	case v >= -(1 << 7):
		return append(b, 0xd0, byte(v))
	case v >= -(1 << 15):
		return binary.BigEndian.AppendUint16(append(b, 0xd1), uint16(v))
	case v >= -(1 << 31):
		return binary.BigEndian.AppendUint32(append(b, 0xd2), uint32(v))
	default:
		return binary.BigEndian.AppendUint64(append(b, 0xd3), uint64(v))
	}
}

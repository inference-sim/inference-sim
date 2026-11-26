package kv

import (
	"bytes"
	"crypto/sha256"
	"testing"
)

// hashBlockUnderOracle is hashBlockUnder as it was before the pickler was reused: a fresh pickler
// for every block.
func hashBlockUnderOracle(root, parent BlockHash, tokens []int64, extra []ExtraKey) BlockHash {
	if len(parent) == 0 {
		parent = root
	}
	var p pickler
	p.frame(func() {
		p.markTuple(func() {
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
		}, 3)
	})
	s := sha256.Sum256(p.out)
	return s[:]
}

func FuzzHashBlockReusesScratch(f *testing.F) {
	f.Add([]byte{1, 2, 3}, []byte("ab"), uint8(2))
	f.Add(bytes.Repeat([]byte{200, 7, 255, 1}, 9000), []byte(""), uint8(0)) // a block larger than one pickle frame
	f.Fuzz(func(t *testing.T, raw []byte, parent []byte, nextra uint8) {
		var toks []int64
		for i := 0; i+2 <= len(raw); i += 2 {
			v := int64(int16(uint16(raw[i]) | uint16(raw[i+1])<<8))
			if raw[i]%7 == 0 {
				v *= 1 << 20 // larger than 16 bits, up to beyond int32 for LONG1
			}
			toks = append(toks, v)
		}
		var extra []ExtraKey
		for i := 0; i < int(nextra%3); i++ {
			extra = append(extra, ExtraKey{"lora", string(parent)})
		}
		root := NoneHashFor(nil)
		// hash twice in a row so the second call runs on a reused (dirty) pickler
		for round := 0; round < 3; round++ {
			if got, want := hashBlockUnder(root, BlockHash(parent), toks, extra), hashBlockUnderOracle(root, BlockHash(parent), toks, extra); !bytes.Equal(got, want) {
				t.Fatalf("round %d: hash %x, oracle %x", round, got, want)
			}
		}
	})
}

package llmd

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"slices"
	"strconv"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// The tests in this file pin the lane-J speed-ups to the code they replaced: each compares the new
// implementation with the old one, kept here as the oracle, on random inputs.

// hashBlockOracle is hashBlock as it was before appendDec: the digest streamed token by token.
func hashBlockOracle(prevHash string, tokens []sim.TokenID) string {
	h := sha256.New()
	h.Write([]byte(prevHash))
	for _, t := range tokens {
		h.Write([]byte(strconv.Itoa(int(t)) + "|"))
	}
	return hex.EncodeToString(h.Sum(nil))
}

func FuzzAppendDec(f *testing.F) {
	for _, v := range []uint32{0, 9, 10, 99, 100, 127999, 4294967295, 1000000000} {
		f.Add(v, []byte("xy"))
	}
	f.Fuzz(func(t *testing.T, v uint32, prefix []byte) {
		want := strconv.AppendUint(slices.Clone(prefix), uint64(v), 10)
		for _, spare := range []int{0, 3, 20} { // the capacity the caller leaves decides the code path taken
			b := append(make([]byte, 0, len(prefix)+spare), prefix...)
			if got := appendDec(b, v); string(got) != string(want) {
				t.Fatalf("appendDec(%q, %d) with %d spare = %q, want %q", prefix, v, spare, got, want)
			}
		}
	})
}

func FuzzHashBlock(f *testing.F) {
	f.Add("", []byte{1, 2, 3, 0, 0, 0, 255, 255, 255, 255})
	f.Fuzz(func(t *testing.T, prev string, raw []byte) {
		var toks []sim.TokenID
		for i := 0; i+4 <= len(raw); i += 4 { // any int32, negative ones included
			toks = append(toks, sim.TokenID(int32(uint32(raw[i])|uint32(raw[i+1])<<8|uint32(raw[i+2])<<16|uint32(raw[i+3])<<24)))
		}
		if got, want := hashBlock(prev, toks), hashBlockOracle(prev, toks); got != want {
			t.Fatalf("hashBlock = %s, oracle %s", got, want)
		}
	})
}

// canonState is everything observable about a promptCanon: the next id, each live chain's id and pins,
// and the order of the LRU.
func canonState(c *promptCanon) (uint64, map[string][2]uint64, []string) {
	ch := map[string][2]uint64{}
	for k, v := range c.chains {
		ch[k] = [2]uint64{v.id, uint64(v.pins)}
	}
	var order []string
	for e := c.lru.Front(); e != nil; e = e.Next() {
		order = append(order, e.Value.(string))
	}
	return c.next, ch, order
}

// FuzzCanonicalSessionMatchesCanonicalUnder drives two canonicalizers with the same operations, one
// through canonicalUnder (the oracle) and one through canonicalSession, with small chain and memo
// capacities so that chains and memos are evicted, and checks that every prompt and the whole state
// agree after every operation.
func FuzzCanonicalSessionMatchesCanonicalUnder(f *testing.F) {
	f.Add([]byte{1, 5, 1, 7, 2, 9, 0, 3, 1, 5, 2, 8, 3, 4, 1, 6})
	f.Add([]byte("session prompts that extend each other and sometimes diverge"))
	f.Fuzz(func(t *testing.T, ops []byte) {
		const bs = 4
		capacity := 3 + int(len(ops)%13)
		a, b := newPromptCanon(7, bs, capacity), newPromptCanon(7, bs, capacity)
		b.memoCap = 40 + int(len(ops)%50)
		sessions := []string{"", "s1", "s2", "s3"}
		var last [4][]sim.TokenID
		var pinned []string
		for i := 0; i+2 < len(ops); i += 3 {
			kind, x, y := ops[i]%8, ops[i+1], ops[i+2]
			s := int(x) % len(sessions)
			switch {
			case kind < 5:
				// a prompt: extend the session's last one, or cut it and diverge, or start afresh
				var p []sim.TokenID
				switch kind {
				case 0:
					p = nil
				case 1, 2:
					p = append(slices.Clone(last[s]), tokens(int(y)%11, int(x))...)
				case 3:
					p = slices.Clone(last[s][:len(last[s])*int(y%5)/4])
					p = append(p, tokens(int(y)%7, int(y))...)
				default:
					p = tokens(int(y)%23, int(x))
				}
				root := ""
				if y%9 == 0 {
					root = identityRoot("lora", nil)
				}
				if got, want := b.canonicalSession(root, sessions[s], p), a.canonicalUnder(root, p); !slices.Equal(got, want) {
					t.Fatalf("op %d: canonicalSession differs from canonicalUnder", i/3)
				}
				last[s] = p
			case kind == 5 && len(a.chains) > 0:
				keys := make([]string, 0, len(a.chains))
				for k := range a.chains {
					keys = append(keys, k)
				}
				slices.Sort(keys)
				k := keys[int(y)%len(keys)]
				a.pin(k)
				b.pin(k)
				pinned = append(pinned, k)
			case kind == 6 && len(pinned) > 0:
				k := pinned[int(y)%len(pinned)]
				a.unpin(k)
				b.unpin(k)
			}
			n1, c1, o1 := canonState(a)
			n2, c2, o2 := canonState(b)
			if n1 != n2 || !equalMaps(c1, c2) || !slices.Equal(o1, o2) {
				t.Fatalf("op %d: chain table or LRU order differs", i/3)
			}
		}
		if b.memoTokens > b.memoCap || b.memoLRU.Len() != len(b.memos) {
			t.Fatalf("memo accounting: %d tokens over cap %d, lru %d memos %d", b.memoTokens, b.memoCap, b.memoLRU.Len(), len(b.memos))
		}
		_ = list.New
	})
}

func equalMaps(a, b map[string][2]uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

// tokens is n tokens drawn from a small alphabet, so prompts share blocks.
func tokens(n, seed int) []sim.TokenID {
	out := make([]sim.TokenID, n)
	for i := range out {
		out[i] = sim.TokenID((seed + i) % 5)
	}
	return out
}

// A session that grows its prompt turn by turn gets, from the memo, the same canonical prompts and
// the same chain ids as a router that sees no sessions; and the memo does reuse work (the chain
// hashes of the shared blocks are not computed again).
func TestSessionMemoReusesSharedBlocks(t *testing.T) {
	a, b := newPromptCanon(3, 4, 1000), newPromptCanon(3, 4, 1000)
	p := tokens(40, 1)
	for turn := 0; turn < 5; turn++ {
		p = append(p, tokens(9, turn)...)
		if !slices.Equal(b.canonicalSession("", "chat", p), a.canonicalUnder("", p)) {
			t.Fatalf("turn %d: canonical prompts differ", turn)
		}
	}
	if m := b.memos["chat"]; m == nil || len(m.tokens) != len(p) {
		t.Fatal("the session's last prompt is not remembered")
	}
}

func TestSessionMemoIsBounded(t *testing.T) {
	c := newPromptCanon(3, 4, 1000)
	c.memoCap = 100
	for i := 0; i < 50; i++ {
		c.canonicalSession("", "s"+strconv.Itoa(i), tokens(30, i))
	}
	if c.memoTokens > c.memoCap || len(c.memos) > 3 {
		t.Fatalf("memos hold %d tokens in %d sessions over a cap of %d", c.memoTokens, len(c.memos), c.memoCap)
	}
}

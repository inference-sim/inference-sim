package workload

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// The parsed-corpus cache (--corpus-cache): a binary snapshot of a rebuilt AgentXCorpus,
// so a sweep that replays the same corpus at many concurrencies parses its JSON once.
// A snapshot is keyed by everything the rebuilt corpus depends on -- the corpus digest,
// the max-context filter, chain splitting, this schema version and the build that wrote
// it -- and carries the sha256 of its own payload. Any mismatch or damage makes the
// reader ignore the file and parse afresh, so a cached and an uncached run are
// byte-identical (INV-6). Writes go to a temporary file in the cache directory, then
// rename onto the final name, so a concurrent reader sees either nothing or a whole file.

// agxCacheSchema versions the snapshot layout and the rebuild it records; bump it when
// either changes.
const agxCacheSchema = 1

var agxCacheMagic = []byte("BLISAGXC")

// AgentXCacheResult says what a cached load did.
type AgentXCacheResult struct {
	Path string
	Hit  bool
	// Ignored is why an existing snapshot was not used ("" when there was none or it was
	// used); WriteErr why a fresh snapshot could not be written.
	Ignored  string
	WriteErr error
}

// LoadAgentXCorpusCached is LoadAgentXCorpus with a snapshot cache in cacheDir. buildKey
// identifies the build (e.g. a digest of the executable): a snapshot written by another
// build is ignored.
func LoadAgentXCorpusCached(path, wantSHA256 string, maxContextLength int, split bool, cacheDir, buildKey string) (*AgentXCorpus, AgentXCacheResult, error) {
	var res AgentXCacheResult
	sum, err := fileSHA256(path)
	if err != nil {
		return nil, res, err
	}
	if wantSHA256 != "" && !strings.EqualFold(wantSHA256, sum) {
		return nil, res, fmt.Errorf("%s: sha256 %s does not match --corpus-sha256 %s", path, sum, wantSHA256)
	}
	key := fmt.Sprintf("schema=%d corpus=%s max_context_length=%d split=%t build=%s", agxCacheSchema, sum, maxContextLength, split, buildKey)
	k := sha256.Sum256([]byte(key))
	res.Path = filepath.Join(cacheDir, "agentx-corpus-"+hex.EncodeToString(k[:8])+".bin")
	if b, rerr := os.ReadFile(res.Path); rerr == nil {
		c, derr := decodeAgentXSnapshot(b, key)
		if derr == nil {
			res.Hit = true
			return c, res, nil
		}
		res.Ignored = derr.Error()
	} else if !errors.Is(rerr, os.ErrNotExist) {
		res.Ignored = rerr.Error()
	}
	c, err := LoadAgentXCorpus(path, sum, maxContextLength, split)
	if err != nil {
		return nil, res, err
	}
	res.WriteErr = writeFileAtomic(res.Path, encodeAgentXSnapshot(c, key))
	return c, res, nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func writeFileAtomic(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".agentx-corpus-*.tmp")
	if err != nil {
		return err
	}
	_, werr := tmp.Write(data)
	if werr == nil {
		werr = tmp.Sync()
	}
	if cerr := tmp.Close(); werr == nil {
		werr = cerr
	}
	if werr == nil {
		werr = os.Chmod(tmp.Name(), 0o644)
	}
	if werr == nil {
		werr = os.Rename(tmp.Name(), path)
	}
	if werr != nil {
		_ = os.Remove(tmp.Name())
	}
	return werr
}

// --- encoding: magic, key, payload length, payload, sha256(payload) ---

type agxEnc struct{ b bytes.Buffer }

func (e *agxEnc) u64(v uint64) {
	var t [8]byte
	binary.LittleEndian.PutUint64(t[:], v)
	e.b.Write(t[:])
}
func (e *agxEnc) i64(v int64)   { e.u64(uint64(v)) }
func (e *agxEnc) f64(v float64) { e.u64(math.Float64bits(v)) }
func (e *agxEnc) str(s string)  { e.i64(int64(len(s))); e.b.WriteString(s) }
func (e *agxEnc) boolean(v bool) {
	if v {
		e.b.WriteByte(1)
	} else {
		e.b.WriteByte(0)
	}
}

// n writes a slice length, -1 for nil, so a decoded corpus is identical to the parsed one.
func (e *agxEnc) n(isNil bool, l int) {
	if isNil {
		e.i64(-1)
		return
	}
	e.i64(int64(l))
}

func (e *agxEnc) i32s(v []int32) {
	e.n(v == nil, len(v))
	for _, x := range v {
		e.i64(int64(x))
	}
}

func encodeAgentXSnapshot(c *AgentXCorpus, key string) []byte {
	var p agxEnc
	p.str(c.SHA256)
	p.i64(int64(c.Rows))
	p.i64(int64(c.FilteredByMaxCtx))
	p.i64(int64(c.MaxContextLength))
	p.n(c.Traces == nil, len(c.Traces))
	for _, t := range c.Traces {
		p.str(t.ID)
		p.i64(int64(t.Peak))
		p.f64(t.minMs)
		p.f64(t.maxMs)
		p.n(t.convs == nil, len(t.convs))
		for _, cv := range t.convs {
			p.str(cv.sessionID)
			p.i64(int64(cv.scope))
			p.n(cv.turns == nil, len(cv.turns))
			for _, tn := range cv.turns {
				p.f64(tn.tsMs)
				p.f64(tn.delayMs)
				p.boolean(tn.hasDelay)
				p.f64(tn.apiMs)
				p.i64(int64(tn.in))
				p.i64(int64(tn.maxTokens))
				p.i64(int64(tn.hashNode))
				p.i64(int64(tn.nHash))
				p.i32s(tn.spawns)
				p.i32s(tn.gates)
				p.n(tn.preds == nil, len(tn.preds))
				for _, r := range tn.preds {
					p.i64(int64(r.conv))
					p.i64(int64(r.turn))
				}
			}
		}
		p.n(t.branches == nil, len(t.branches))
		for _, b := range t.branches {
			p.str(b.id)
			p.i32s(b.children)
			p.boolean(b.background)
			p.f64(b.startMs)
			p.i64(int64(b.spawnTurn))
			p.i64(int64(b.joinTurn))
		}
		p.i32s(t.trieParent)
		p.n(t.trieHash == nil, len(t.trieHash))
		for _, h := range t.trieHash {
			p.i64(h)
		}
	}
	payload := p.b.Bytes()
	var out agxEnc
	out.b.Write(agxCacheMagic)
	out.str(key)
	out.i64(int64(len(payload)))
	out.b.Write(payload)
	sum := sha256.Sum256(payload)
	out.b.Write(sum[:])
	return out.b.Bytes()
}

type agxDec struct {
	r   *bufio.Reader
	err error
	// left bounds every length read, so a damaged length cannot make the decoder allocate
	// past the snapshot's size.
	left int64
}

func (d *agxDec) u64() uint64 {
	if d.err != nil {
		return 0
	}
	var t [8]byte
	if _, err := io.ReadFull(d.r, t[:]); err != nil {
		d.err = err
		return 0
	}
	d.left -= 8
	return binary.LittleEndian.Uint64(t[:])
}
func (d *agxDec) i64() int64   { return int64(d.u64()) }
func (d *agxDec) f64() float64 { return math.Float64frombits(d.u64()) }
func (d *agxDec) length() int {
	l := d.i64()
	if d.err == nil && (l < -1 || l > d.left) {
		d.err = fmt.Errorf("damaged snapshot: length %d", l)
	}
	if d.err != nil {
		return -1
	}
	return int(l)
}
func (d *agxDec) str() string {
	l := d.length()
	if l <= 0 {
		return ""
	}
	b := make([]byte, l)
	if _, err := io.ReadFull(d.r, b); err != nil {
		d.err = err
		return ""
	}
	d.left -= int64(l)
	return string(b)
}
func (d *agxDec) boolean() bool {
	if d.err != nil {
		return false
	}
	c, err := d.r.ReadByte()
	if err != nil {
		d.err = err
	}
	d.left--
	return c == 1
}
func (d *agxDec) i32s() []int32 {
	l := d.length()
	if l < 0 {
		return nil
	}
	v := make([]int32, l)
	for i := range v {
		v[i] = int32(d.i64())
	}
	return v
}

func decodeAgentXSnapshot(b []byte, wantKey string) (*AgentXCorpus, error) {
	if !bytes.HasPrefix(b, agxCacheMagic) {
		return nil, errors.New("not a corpus snapshot")
	}
	head := &agxDec{r: bufio.NewReader(bytes.NewReader(b[len(agxCacheMagic):])), left: int64(len(b))}
	key := head.str()
	plen := head.i64()
	if head.err != nil {
		return nil, fmt.Errorf("damaged snapshot header: %w", head.err)
	}
	if key != wantKey {
		return nil, fmt.Errorf("snapshot key %q does not match %q", key, wantKey)
	}
	start := len(agxCacheMagic) + 8 + len(key) + 8
	if plen < 0 || int64(len(b)) != int64(start)+plen+sha256.Size {
		return nil, errors.New("snapshot is truncated or oversized")
	}
	payload := b[start : start+int(plen)]
	if sum := sha256.Sum256(payload); !bytes.Equal(sum[:], b[start+int(plen):]) {
		return nil, errors.New("snapshot payload digest does not match")
	}
	d := &agxDec{r: bufio.NewReader(bytes.NewReader(payload)), left: plen}
	c := &AgentXCorpus{SHA256: d.str(), Rows: int(d.i64()), FilteredByMaxCtx: int(d.i64()), MaxContextLength: int(d.i64())}
	if nt := d.length(); nt >= 0 {
		c.Traces = make([]*AgentXTrace, nt)
	}
	for i := range c.Traces {
		t := &AgentXTrace{ID: d.str(), Peak: int(d.i64()), minMs: d.f64(), maxMs: d.f64()}
		if nc := d.length(); nc >= 0 {
			t.convs = make([]agxConv, nc)
		}
		for ci := range t.convs {
			cv := &t.convs[ci]
			cv.sessionID = d.str()
			cv.scope = int32(d.i64())
			if nn := d.length(); nn >= 0 {
				cv.turns = make([]agxTurn, nn)
			}
			for k := range cv.turns {
				tn := &cv.turns[k]
				tn.tsMs, tn.delayMs, tn.hasDelay, tn.apiMs = d.f64(), d.f64(), d.boolean(), d.f64()
				tn.in, tn.maxTokens, tn.hashNode, tn.nHash = int32(d.i64()), int32(d.i64()), int32(d.i64()), int32(d.i64())
				tn.spawns, tn.gates = d.i32s(), d.i32s()
				if np := d.length(); np >= 0 {
					tn.preds = make([]agxRef, np)
				}
				for j := range tn.preds {
					tn.preds[j] = agxRef{int32(d.i64()), int32(d.i64())}
				}
			}
		}
		if nb := d.length(); nb >= 0 {
			t.branches = make([]agxBranch, nb)
		}
		for bi := range t.branches {
			b := &t.branches[bi]
			b.id, b.children, b.background, b.startMs = d.str(), d.i32s(), d.boolean(), d.f64()
			b.spawnTurn, b.joinTurn = int32(d.i64()), int32(d.i64())
		}
		t.trieParent = d.i32s()
		if nh := d.length(); nh >= 0 {
			t.trieHash = make([]int64, nh)
		}
		for j := range t.trieHash {
			t.trieHash[j] = d.i64()
		}
		c.Traces[i] = t
	}
	if d.err != nil {
		return nil, fmt.Errorf("damaged snapshot: %w", d.err)
	}
	if d.left != 0 {
		return nil, errors.New("snapshot has trailing bytes")
	}
	return c, nil
}

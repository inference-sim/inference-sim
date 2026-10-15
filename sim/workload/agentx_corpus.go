package workload

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// AgentXCorpus is a Weka v7 corpus rebuilt for agentic replay: the root pool, in row
// order, of every trace whose peak context fits the --max-context-length filter.
type AgentXCorpus struct {
	Traces           []*AgentXTrace
	SHA256           string // digest of the file as read
	Rows             int
	FilteredByMaxCtx int
	MaxContextLength int // 0 = no filter
}

// LoadAgentXCorpus streams a Weka v7 JSONL corpus (one trace per line), hashing the bytes
// as it reads them, and rebuilds each kept trace as aiperf's session tree. Only one
// trace's JSON is alive at a time; what is kept is the compact tree (hash ids interned
// in a per-trace prefix trie). wantSHA256, when non-empty, must match the file's digest.
// maxContextLength > 0 drops every trace whose peak in + max(out, 1) exceeds it
// (dataset/loader/selection.py filter_then_cap). An empty pool is an error, as in aiperf.
func LoadAgentXCorpus(path, wantSHA256 string, maxContextLength int, split bool) (*AgentXCorpus, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	c, err := ReadAgentXCorpus(f, maxContextLength, split)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if wantSHA256 != "" && !strings.EqualFold(wantSHA256, c.SHA256) {
		return nil, fmt.Errorf("%s: sha256 %s does not match --corpus-sha256 %s", path, c.SHA256, wantSHA256)
	}
	if len(c.Traces) == 0 {
		return nil, fmt.Errorf("%s: no trace fits max_context_length %d (%d rows scanned)", path, maxContextLength, c.Rows)
	}
	return c, nil
}

// ReadAgentXCorpus is LoadAgentXCorpus over a reader, without the digest and pool checks.
func ReadAgentXCorpus(r io.Reader, maxContextLength int, split bool) (*AgentXCorpus, error) {
	h := sha256.New()
	br := bufio.NewReaderSize(io.TeeReader(r, h), 1<<20)
	c := &AgentXCorpus{MaxContextLength: maxContextLength}
	seen := map[string]bool{}
	var line []byte
	for lineNo := 1; ; lineNo++ {
		line = line[:0]
		var err error
		for {
			var frag []byte
			frag, err = br.ReadSlice('\n')
			line = append(line, frag...)
			if err != bufio.ErrBufferFull {
				break
			}
		}
		if err != nil && err != io.EOF {
			return nil, err
		}
		if trimmed := bytes.TrimSpace(line); len(trimmed) > 0 {
			var s wekaSession
			if jerr := json.Unmarshal(trimmed, &s); jerr != nil {
				return nil, fmt.Errorf("line %d: %w", lineNo, jerr)
			}
			c.Rows++
			if seen[s.ID] {
				return nil, fmt.Errorf("line %d: duplicate trace id %q", lineNo, s.ID)
			}
			seen[s.ID] = true
			if maxContextLength > 0 && wekaPeak(&s) > maxContextLength {
				c.FilteredByMaxCtx++
			} else {
				t, berr := BuildAgentXTrace(&s, split)
				if berr != nil {
					return nil, fmt.Errorf("line %d: %w", lineNo, berr)
				}
				c.Traces = append(c.Traces, t)
			}
		}
		if err == io.EOF {
			break
		}
	}
	c.SHA256 = hex.EncodeToString(h.Sum(nil))
	return c, nil
}

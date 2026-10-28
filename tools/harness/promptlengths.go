package harness

// promptlengths.go reads the evidence that the prompt lengths the benchmark client SENT are the
// targets the fixture replays.
//
// The client re-encodes each prompt up to ten times to land on its target and sends whatever it
// reached; the difference is printed only as a mean. scripts/inferencex_prompt_lengths.py
// reproduces that construction offline with each point's tokenizer at pinned Hugging Face
// revisions (the one current at the run date, and today's), and writes, per point, the largest
// |sent - target| and the tokens a server adds, with every tokenizer file's sha256, to
// testdata/inferencex-workloads/prompt-lengths.json. A point is verified when every pinned
// revision gives zero on both; a point the file does not hold is not verified, and the coverage
// report says so.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sync"
)

// PromptLengthEvidence is the evidence file.
type PromptLengthEvidence struct {
	Generator   string              `json:"generator"`
	Client      map[string]any      `json:"client"`
	Environment map[string]string   `json:"environment"`
	Revisions   []TokenizerRevision `json:"revisions"`
	Points      []PromptLengthPoint `json:"points"`

	path  string
	byKey map[string]*PromptLengthPoint
}

// TokenizerRevision is one pinned revision of a tokenizer repository and the sha256 of every file
// the construction read from it.
type TokenizerRevision struct {
	Repo      string            `json:"repo"`
	Revision  string            `json:"revision"`
	Committed string            `json:"committed"`
	Roles     []string          `json:"roles"`
	Files     map[string]string `json:"files"`
}

// PromptLengthPoint is one scored point's evidence: one entry per pinned revision, nil where the
// repository did not exist at the run date.
type PromptLengthPoint struct {
	Scenario          string              `json:"scenario"`
	Label             string              `json:"label"`
	Concurrency       int                 `json:"concurrency"`
	ServedModel       string              `json:"served_model"`
	Repo              string              `json:"repo"`
	RunDate           string              `json:"run_date"`
	ISL               int                 `json:"isl"`
	OSL               int                 `json:"osl"`
	ClientPath        string              `json:"client_path"`
	Evidence          []*RevisionEvidence `json:"evidence"`
	MaxAbsMismatch    *int                `json:"max_abs_mismatch"`
	ServerAddedTokens []int               `json:"server_added_tokens"`
}

// RevisionEvidence is the construction's result at one revision.
type RevisionEvidence struct {
	Revision           string  `json:"revision"`
	Requests           int     `json:"requests"`
	MaxAbsMismatch     int     `json:"max_abs_mismatch"`
	MeanMismatch       float64 `json:"mean_mismatch"`
	RequestsMismatched int     `json:"requests_mismatched"`
	ServerAddedTokens  []int   `json:"server_added_tokens"`
}

// DefaultPromptLengthsPath is the committed evidence file.
func DefaultPromptLengthsPath() string {
	return filepath.Join(repoRoot(), "testdata", "inferencex-workloads", "prompt-lengths.json")
}

var (
	gitRevision = regexp.MustCompile(`^[0-9a-f]{40}$`)
	sha256Hex   = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

// LoadPromptLengthEvidence reads the evidence file and refuses one without its provenance: a
// generator, the environment, a pinned 40-hex revision with a role and a sha256 for every file
// for each revision a point cites, and a point that cites a revision the header lacks.
func LoadPromptLengthEvidence(path string) (*PromptLengthEvidence, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("prompt-length evidence: %w", err)
	}
	var e PromptLengthEvidence
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("prompt-length evidence %s: %w", path, err)
	}
	if e.Generator == "" || len(e.Environment) == 0 || len(e.Revisions) == 0 || len(e.Points) == 0 {
		return nil, fmt.Errorf("prompt-length evidence %s lacks its generator, environment, revisions or points", path)
	}
	known := map[string]bool{}
	for _, r := range e.Revisions {
		if r.Repo == "" || !gitRevision.MatchString(r.Revision) || len(r.Roles) == 0 || len(r.Files) == 0 {
			return nil, fmt.Errorf("prompt-length evidence %s: revision %+v is not pinned with its roles and files", path, r)
		}
		for f, h := range r.Files {
			if !sha256Hex.MatchString(h) {
				return nil, fmt.Errorf("prompt-length evidence %s: %s@%s %s has no sha256", path, r.Repo, r.Revision, f)
			}
		}
		known[r.Repo+"@"+r.Revision] = true
	}
	e.path = path
	e.byKey = make(map[string]*PromptLengthPoint, len(e.Points))
	for i := range e.Points {
		p := &e.Points[i]
		for _, ev := range p.Evidence {
			if ev != nil && !known[p.Repo+"@"+ev.Revision] {
				return nil, fmt.Errorf("prompt-length evidence %s: %s %s c=%d cites %s@%s, which the header does not pin",
					path, p.Scenario, p.Label, p.Concurrency, p.Repo, ev.Revision)
			}
		}
		e.byKey[promptKey(p.Scenario, p.Label, p.Concurrency)] = p
	}
	return &e, nil
}

func promptKey(scenario, label string, concurrency int) string {
	return fmt.Sprintf("%s\x00%s\x00%d", scenario, label, concurrency)
}

// Verified reports whether a point's sent prompt lengths are its targets: the file holds the
// point, and at every pinned revision the construction gave zero mismatch and the server added
// no token. detail says why not.
func (e *PromptLengthEvidence) Verified(sw Sweep, concurrency int) (bool, string) {
	if e == nil {
		return false, "no prompt-length evidence loaded"
	}
	p := e.byKey[promptKey(sw.Scenario, sw.Label, concurrency)]
	if p == nil {
		return false, fmt.Sprintf("%s holds no evidence for the point (no engine log, so no run date or client "+
			"version; or a gated tokenizer)", filepath.Base(e.path))
	}
	n := 0
	for _, ev := range p.Evidence {
		if ev == nil {
			continue
		}
		n++
		if ev.MaxAbsMismatch != 0 {
			return false, fmt.Sprintf("at %s@%.8s the client sent up to %d tokens away from the target",
				p.Repo, ev.Revision, ev.MaxAbsMismatch)
		}
		for _, t := range ev.ServerAddedTokens {
			if t != 0 {
				return false, fmt.Sprintf("at %s@%.8s the server adds %d special token(s)", p.Repo, ev.Revision, t)
			}
		}
	}
	if n == 0 {
		return false, "no pinned revision of the point's tokenizer existed to check"
	}
	return true, ""
}

var (
	defaultPromptLengthsOnce sync.Once
	defaultPromptLengths     *PromptLengthEvidence
	defaultPromptLengthsErr  error
)

// promptLengths is cfg's evidence, or the committed file.
func (cfg Config) promptLengths() (*PromptLengthEvidence, error) {
	if cfg.PromptLengths != nil {
		return cfg.PromptLengths, nil
	}
	defaultPromptLengthsOnce.Do(func() {
		defaultPromptLengths, defaultPromptLengthsErr = LoadPromptLengthEvidence(DefaultPromptLengthsPath())
	})
	return defaultPromptLengths, defaultPromptLengthsErr
}

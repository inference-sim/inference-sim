package ledger

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	validate "github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// The committed files under catalog/ledger/ (design §10.2). No other file may sit there.
const (
	FileLedger    = "ledger.yaml"     // partition version, seed hash, cut-points
	FileSources   = "sources.yaml"    // sources, pinned revisions, licences
	FileLegacyMap = "legacy-map.yaml" // the legacy table
	FileReadme    = "README.md"
	DirUnits      = "units"     // units/<source>.jsonl.gz
	DirReports    = "reports"   // reports/<source>.yaml, the ingest report
	DirAliases    = "aliases"   // aliases/<source>-<n>.yaml
	DirPublished  = "published" // fit.yaml, anchor.yaml, report.yaml, written by the release job
)

// Config is ledger.yaml.
type Config struct {
	Kind             string                `yaml:"kind"` // "Ledger"
	PartitionVersion int                   `yaml:"partition_version"`
	Seed             SeedRecord            `yaml:"seed"`
	Rules            map[string]SourceRule `yaml:"rules"`
}

// SeedRecord is what the ledger commits about the seed: its hash, and who generated it, when,
// and who owns its rotation (design §10.2). Every field is empty until a ledger owner creates
// the seed; until then no role can be drawn.
type SeedRecord struct {
	SHA256        string `yaml:"sha256"`
	GeneratedBy   string `yaml:"generated_by"`
	GeneratedOn   string `yaml:"generated_on"`
	RotationOwner string `yaml:"rotation_owner"`
}

// Created reports whether a ledger owner has recorded a seed.
func (s SeedRecord) Created() bool { return s.SHA256 != "" }

// SourceRule is one source's cut-points: for units on no legacy side, and for each legacy side
// the one cut-point that divides it, fit from select or held from anchor. The same HMAC value
// is read against whichever applies (design §10.2).
type SourceRule struct {
	CutPoints CutPoints          `yaml:"cut_points"`
	Legacy    map[Side]CutPoints `yaml:"legacy,omitempty"`
}

var hex64 = regexp.MustCompile(`^[0-9a-f]{64}$`)

// Validate checks ledger.yaml on its own.
func (c *Config) Validate() error {
	var errs []error
	if c.Kind != "Ledger" {
		errs = append(errs, fmt.Errorf("kind: %q, want \"Ledger\"", c.Kind))
	}
	if c.PartitionVersion < 1 {
		errs = append(errs, fmt.Errorf("partition_version: %d, want 1 or more", c.PartitionVersion))
	}
	s := c.Seed
	if s.Created() {
		if !hex64.MatchString(s.SHA256) {
			errs = append(errs, fmt.Errorf("seed.sha256: %q is not 64 lowercase hex digits", s.SHA256))
		}
		for name, v := range map[string]string{"generated_by": s.GeneratedBy, "generated_on": s.GeneratedOn, "rotation_owner": s.RotationOwner} {
			if v == "" {
				errs = append(errs, fmt.Errorf("seed.%s: required once seed.sha256 is recorded", name))
			}
		}
	} else if s.GeneratedBy != "" || s.GeneratedOn != "" || s.RotationOwner != "" {
		errs = append(errs, errors.New("seed: who, when and rotation owner are recorded with seed.sha256, not without it"))
	}
	if len(c.Rules) == 0 {
		errs = append(errs, errors.New("rules: no source has cut-points"))
	}
	for _, src := range sortedKeys(c.Rules) {
		r := c.Rules[src]
		if err := r.CutPoints.check("rules."+src+".cut_points", Roles); err != nil {
			errs = append(errs, err)
		}
		for side, cp := range r.Legacy {
			at := fmt.Sprintf("rules.%s.legacy.%s", src, side)
			if !side.Valid() {
				errs = append(errs, fmt.Errorf("%s: %q is not FIT or HELD", at, side))
				continue
			}
			if err := cp.check(at, side.roles()); err != nil {
				errs = append(errs, err)
			} else if len(cp) != 2 || cp[0].Role != side.roles()[0] || cp[1].Role != side.roles()[1] {
				errs = append(errs, fmt.Errorf("%s: one cut-point divides the side: %s below it, %s from it", at, side.roles()[0], side.roles()[1]))
			}
		}
	}
	return errors.Join(errs...)
}

// Sources is sources.yaml.
type Sources struct {
	Kind    string   `yaml:"kind"` // "LedgerSources"
	Sources []Source `yaml:"sources"`
}

// Source is one data source, pinned to an exact revision (D3), with its licence (X12, D6).
type Source struct {
	ID    string `yaml:"id"`
	Title string `yaml:"title"`
	// Revision pins what the units were generated from: a commit, a content hash or a run id.
	Revision Revision `yaml:"revision"`
	// Units is the unit list's path under catalog/ledger/, empty while the source has no
	// ingest tool. Status says why.
	Units   string  `yaml:"units,omitempty"`
	Status  string  `yaml:"status,omitempty"`
	Licence Licence `yaml:"licence"`
}

// Revision is a pinned source revision.
type Revision struct {
	// Kind is git_commit, hf_dataset_revision or release_assets.
	Kind string `yaml:"kind"`
	// URL locates the source at the revision.
	URL string `yaml:"url"`
	// Ref is the commit, dataset revision or release tag.
	Ref string `yaml:"ref"`
	// Commit is the full commit a release tag points at; required for release_assets.
	Commit string `yaml:"commit,omitempty"`
	// Assets lists each release asset by name and SHA-256.
	Assets []Asset `yaml:"assets,omitempty"`
}

// Asset is one downloaded file, pinned by content hash.
type Asset struct {
	Name   string `yaml:"name"`
	SHA256 string `yaml:"sha256"`
}

// Licence is a source's licence, with the evidence and a ledger owner's sign-off (X12, D6).
type Licence struct {
	// SPDX is the identifier the source's own licence file states.
	SPDX     string         `yaml:"spdx"`
	Evidence []LicenceQuote `yaml:"evidence"`
	// Restrictions quotes any statement by the source that narrows what the licence file
	// grants. A source with a restriction is not redistributable whatever its licence.
	Restrictions []LicenceQuote `yaml:"restrictions,omitempty"`
	Note         string         `yaml:"note,omitempty"`
	SignOff      SignOff        `yaml:"sign_off"`
}

// LicenceQuote is a quotation from a file at a pinned revision.
type LicenceQuote struct {
	// URL is the file at a commit or revision, never at a branch.
	URL    string `yaml:"url"`
	SHA256 string `yaml:"sha256"`
	Lines  string `yaml:"lines"`
	Quote  string `yaml:"quote"`
}

// SignOff is filled by a ledger owner who has read the evidence. An agent never fills it.
type SignOff struct {
	By string `yaml:"by"`
	On string `yaml:"on"`
}

// Redistributable reports whether the source's artefacts may be committed: its licence is on
// the allow-list, it states no restriction, and a ledger owner has signed it off (D6, X12).
func (l Licence) Redistributable() bool {
	return vocab.LicenceRedistributable(l.SPDX) && len(l.Restrictions) == 0 && l.SignOff.By != "" && l.SignOff.On != ""
}

// Unpinned names the revisions D3 refuses, compared without regard to case.
var Unpinned = []string{"latest", "main", "unknown"}

var (
	sourceID  = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)
	gitCommit = regexp.MustCompile(`^[0-9a-f]{40}$`)
	isoDate   = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)
)

// Validate checks sources.yaml on its own, including D3.
func (s *Sources) Validate() error {
	var errs []error
	if s.Kind != "LedgerSources" {
		errs = append(errs, fmt.Errorf("kind: %q, want \"LedgerSources\"", s.Kind))
	}
	seen := map[string]bool{}
	for i, src := range s.Sources {
		at := fmt.Sprintf("sources[%d]", i)
		if !sourceID.MatchString(src.ID) {
			errs = append(errs, fmt.Errorf("%s.id: %q is not a lowercase slug", at, src.ID))
		}
		if seen[src.ID] {
			errs = append(errs, fmt.Errorf("%s.id: duplicate source %q", at, src.ID))
		}
		seen[src.ID] = true
		if src.Title == "" {
			errs = append(errs, fmt.Errorf("%s.title: required", at))
		}
		if err := src.Revision.check(at + ".revision"); err != nil {
			errs = append(errs, err)
		}
		if src.Units == "" && src.Status == "" {
			errs = append(errs, fmt.Errorf("%s: a source with no unit list states why in status", at))
		}
		if src.Units != "" && src.Units != DirUnits+"/"+src.ID+".jsonl.gz" {
			errs = append(errs, fmt.Errorf("%s.units: %q, want %q", at, src.Units, DirUnits+"/"+src.ID+".jsonl.gz"))
		}
		if err := src.Licence.check(at + ".licence"); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// Lookup returns the source with id, or nil.
func (s *Sources) Lookup(id string) *Source {
	for i := range s.Sources {
		if s.Sources[i].ID == id {
			return &s.Sources[i]
		}
	}
	return nil
}

// CheckPinned refuses a revision D3 does not allow: empty, or "latest", "main" or "unknown".
func CheckPinned(at, ref string) error {
	if strings.TrimSpace(ref) == "" {
		return fmt.Errorf("%s: empty; a source revision is pinned by commit, hash or run id (D3)", at)
	}
	for _, u := range Unpinned {
		if strings.EqualFold(strings.TrimSpace(ref), u) {
			return fmt.Errorf("%s: %q is not a pinned revision; pin a commit, hash or run id (D3)", at, ref)
		}
	}
	return nil
}

func (r Revision) check(at string) error {
	var errs []error
	if err := CheckPinned(at+".ref", r.Ref); err != nil {
		errs = append(errs, err)
	}
	if !strings.HasPrefix(r.URL, "https://") {
		errs = append(errs, fmt.Errorf("%s.url: %q is not an https URL", at, r.URL))
	}
	switch r.Kind {
	case "git_commit", "hf_dataset_revision":
		if !gitCommit.MatchString(r.Ref) {
			errs = append(errs, fmt.Errorf("%s.ref: %q is not a full 40-digit commit", at, r.Ref))
		}
	case "release_assets":
		if !gitCommit.MatchString(r.Commit) {
			errs = append(errs, fmt.Errorf("%s.commit: %q is not the full 40-digit commit the tag points at", at, r.Commit))
		}
		if len(r.Assets) == 0 {
			errs = append(errs, fmt.Errorf("%s.assets: a release is pinned by the SHA-256 of each asset", at))
		}
		for j, a := range r.Assets {
			if a.Name == "" || !hex64.MatchString(a.SHA256) {
				errs = append(errs, fmt.Errorf("%s.assets[%d]: needs a name and a 64-digit sha256", at, j))
			}
		}
	default:
		errs = append(errs, fmt.Errorf("%s.kind: %q is not git_commit, hf_dataset_revision or release_assets", at, r.Kind))
	}
	return errors.Join(errs...)
}

func (l Licence) check(at string) error {
	var errs []error
	if l.SPDX == "" {
		errs = append(errs, fmt.Errorf("%s.spdx: required: the identifier the source's own licence file states", at))
	}
	if len(l.Evidence) == 0 {
		errs = append(errs, fmt.Errorf("%s.evidence: quote the source's own licence file", at))
	}
	for i, q := range append(append([]LicenceQuote{}, l.Evidence...), l.Restrictions...) {
		qat := fmt.Sprintf("%s.evidence[%d]", at, i)
		if i >= len(l.Evidence) {
			qat = fmt.Sprintf("%s.restrictions[%d]", at, i-len(l.Evidence))
		}
		if err := q.check(qat); err != nil {
			errs = append(errs, err)
		}
	}
	if (l.SignOff.By == "") != (l.SignOff.On == "") {
		errs = append(errs, fmt.Errorf("%s.sign_off: by and on are filled together", at))
	}
	if l.SignOff.On != "" && !isoDate.MatchString(l.SignOff.On) {
		errs = append(errs, fmt.Errorf("%s.sign_off.on: %q is not YYYY-MM-DD", at, l.SignOff.On))
	}
	return errors.Join(errs...)
}

// pinnedURL matches a URL that names a commit or a dataset revision: a 40-digit hex segment.
var pinnedURL = regexp.MustCompile(`^https://[^\s]*/[0-9a-f]{40}(/|$)`)

func (q LicenceQuote) check(at string) error {
	var errs []error
	if !pinnedURL.MatchString(q.URL) {
		errs = append(errs, fmt.Errorf("%s.url: %q does not name a commit; quote the file at a pinned revision (D3)", at, q.URL))
	}
	if !hex64.MatchString(q.SHA256) {
		errs = append(errs, fmt.Errorf("%s.sha256: the SHA-256 of the quoted file, 64 hex digits", at))
	}
	if q.Lines == "" || q.Quote == "" {
		errs = append(errs, fmt.Errorf("%s: lines and quote are required", at))
	}
	return errors.Join(errs...)
}

// decodeStrictFile decodes one YAML document, refusing unknown keys (SUB-12) and a second
// document.
func decodeStrictFile(path string, dst any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return decodeStrict(path, raw, dst)
}

// decodeStrict is the ledger's YAML decoder: the contracts' strict one (SUB-12).
func decodeStrict(name string, raw []byte, dst any) error {
	return validate.DecodeStrict(name, raw, dst)
}

// LoadConfig reads and validates ledger.yaml in dir.
func LoadConfig(dir string) (*Config, error) {
	var c Config
	path := filepath.Join(dir, FileLedger)
	if err := decodeStrictFile(path, &c); err != nil {
		return nil, err
	}
	if err := c.Validate(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &c, nil
}

// LoadSources reads and validates sources.yaml in dir.
func LoadSources(dir string) (*Sources, error) {
	var s Sources
	path := filepath.Join(dir, FileSources)
	if err := decodeStrictFile(path, &s); err != nil {
		return nil, err
	}
	if err := s.Validate(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &s, nil
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

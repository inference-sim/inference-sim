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
	Units  string `yaml:"units,omitempty"`
	Status string `yaml:"status,omitempty"`
	// Artifacts pins each workflow artifact a tool reads from this source's publisher, by run,
	// attempt and SHA-256 (D3). It holds data no unit list yet carries.
	Artifacts []Artifact `yaml:"artifacts,omitempty"`
	// Supplements pin records the source published after Revision, so a unit can exist for a
	// measurement the pinned revision does not hold. The ingest tool fetches them and adds
	// their units to the source's one unit list; none of their values is committed (X12).
	Supplements []Supplement `yaml:"supplements,omitempty"`
	Licence     Licence      `yaml:"licence"`
}

// Supplement is a set of records a source published after its pinned revision, pinned by
// record id and by the SHA-256 of each record's content (D3). Kind is api_records: the
// records are served by the source's API at URL, which is not itself a revision, so the
// content hash is what pins them.
type Supplement struct {
	ID   string `yaml:"id"`
	Kind string `yaml:"kind"`
	// URL serves the pinned records, among others; the ingest tool reads only the pinned ones.
	URL string `yaml:"url"`
	// Run is the upstream run that produced the records, including its attempt.
	Run string `yaml:"run"`
	// Records are the pinned records in increasing id order. Each id is above every id of the
	// pinned revision: a record the revision holds is not a supplement.
	Records []PinnedRecord `yaml:"records"`
}

// PinnedRecord names one source record and the SHA-256 of its content: the canonical JSON of
// the whole record as the API serves it, numbers kept as printed (MeasuredID).
type PinnedRecord struct {
	ID     int64  `yaml:"id"`
	SHA256 string `yaml:"sha256"`
}

// SupplementKindAPI is the one kind of supplement.
const SupplementKindAPI = "api_records"

func (s Supplement) check(at string) error {
	var errs []error
	if !sourceID.MatchString(s.ID) {
		errs = append(errs, fmt.Errorf("%s.id: %q is not a lowercase slug", at, s.ID))
	}
	if s.Kind != SupplementKindAPI {
		errs = append(errs, fmt.Errorf("%s.kind: %q, want %q", at, s.Kind, SupplementKindAPI))
	}
	if !strings.HasPrefix(s.URL, "https://") {
		errs = append(errs, fmt.Errorf("%s.url: %q is not an https URL", at, s.URL))
	}
	if !strings.HasPrefix(s.Run, "https://") || !runURL.MatchString(s.Run) {
		errs = append(errs, fmt.Errorf("%s.run: %q is not the https URL of a workflow run and attempt (.../actions/runs/N/attempts/M)", at, s.Run))
	}
	if len(s.Records) == 0 {
		errs = append(errs, fmt.Errorf("%s.records: pin at least one record by id and SHA-256", at))
	}
	for i, r := range s.Records {
		rat := fmt.Sprintf("%s.records[%d]", at, i)
		if r.ID < 1 {
			errs = append(errs, fmt.Errorf("%s.id: %d, want a record id of 1 or more", rat, r.ID))
		}
		if i > 0 && r.ID <= s.Records[i-1].ID {
			errs = append(errs, fmt.Errorf("%s.id: %d is not above the record before it; list the records in increasing id order, each once", rat, r.ID))
		}
		if !hex64.MatchString(r.SHA256) {
			errs = append(errs, fmt.Errorf("%s.sha256: %q is not 64 lowercase hex digits", rat, r.SHA256))
		}
	}
	return errors.Join(errs...)
}

// Artifact is one workflow artifact of a source, pinned to an exact run. Revision.Kind is
// workflow_artifact.
type Artifact struct {
	ID string `yaml:"id"`
	// Used by says which tool reads it and for what.
	UsedBy   string   `yaml:"used_by"`
	Revision Revision `yaml:"revision"`
}

// Revision is a pinned source revision.
type Revision struct {
	// Kind is git_commit, hf_dataset_revision, release_assets or workflow_artifact.
	Kind string `yaml:"kind"`
	// URL locates the source at the revision.
	URL string `yaml:"url"`
	// Ref is the commit, dataset revision or release tag, or, for workflow_artifact, the
	// workflow run's numeric id.
	Ref string `yaml:"ref"`
	// Commit is the full commit a release tag points at, required for release_assets, or the
	// commit the run evaluated, required for workflow_artifact.
	Commit string `yaml:"commit,omitempty"`
	// Artifact names the uploaded artifact of a workflow run; required for workflow_artifact.
	Artifact string `yaml:"artifact,omitempty"`
	// RunAttempt is the run's attempt number; required for workflow_artifact.
	RunAttempt int `yaml:"run_attempt,omitempty"`
	// Assets lists each release asset, or each file of a workflow artifact (and the artifact's
	// zip, by the name <artifact>.zip), by name and SHA-256.
	Assets []Asset `yaml:"assets,omitempty"`
}

// Asset is one downloaded file, pinned by content hash.
type Asset struct {
	Name   string `yaml:"name"`
	SHA256 string `yaml:"sha256"`
}

// Licence is a source's licence, with the evidence that decides whether its data may be committed (X12, D6).
type Licence struct {
	// SPDX is the identifier the source's own licence file states.
	SPDX     string         `yaml:"spdx"`
	Evidence []LicenceQuote `yaml:"evidence"`
	// Restrictions quotes any statement by the source that narrows what the licence file
	// grants. A source with a restriction is not redistributable whatever its licence.
	Restrictions []LicenceQuote `yaml:"restrictions,omitempty"`
	Note         string         `yaml:"note,omitempty"`
}

// LicenceQuote is a quotation from a file at a pinned revision.
type LicenceQuote struct {
	// URL is the file at a commit or revision, never at a branch.
	URL    string `yaml:"url"`
	SHA256 string `yaml:"sha256"`
	Lines  string `yaml:"lines"`
	Quote  string `yaml:"quote"`
}

// Redistributable reports whether the source's artefacts may be committed: its licence is on
// the allow-list, and it states no restriction (D6, X12). NOASSERTION, the identifier for a source whose licence
// states no terms for its data, is not on the allow-list.
func (l Licence) Redistributable() bool {
	return vocab.LicenceRedistributable(l.SPDX) && len(l.Restrictions) == 0
}

// Unpinned names the revisions D3 refuses, compared without regard to case.
var Unpinned = []string{"latest", "main", "unknown"}

var (
	sourceID  = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)
	gitCommit = regexp.MustCompile(`^[0-9a-f]{40}$`)
	runID     = regexp.MustCompile(`^[0-9]{1,20}$`)
	isoDate   = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)
	runURL    = regexp.MustCompile(`/actions/runs/[0-9]+/attempts/[0-9]+$`)
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
		seenArt := map[string]bool{}
		for j, a := range src.Artifacts {
			aat := fmt.Sprintf("%s.artifacts[%d]", at, j)
			if !sourceID.MatchString(a.ID) {
				errs = append(errs, fmt.Errorf("%s.id: %q is not a lowercase slug", aat, a.ID))
			}
			if seenArt[a.ID] {
				errs = append(errs, fmt.Errorf("%s.id: duplicate artifact %q", aat, a.ID))
			}
			seenArt[a.ID] = true
			if a.UsedBy == "" {
				errs = append(errs, fmt.Errorf("%s.used_by: name the tool that reads the artifact", aat))
			}
			if a.Revision.Kind != "workflow_artifact" {
				errs = append(errs, fmt.Errorf("%s.revision.kind: %q, want workflow_artifact", aat, a.Revision.Kind))
			}
			if err := a.Revision.check(aat + ".revision"); err != nil {
				errs = append(errs, err)
			}
		}
		seenSup := map[string]bool{}
		seenRec := map[int64]string{}
		for j, sup := range src.Supplements {
			sat := fmt.Sprintf("%s.supplements[%d]", at, j)
			if seenSup[sup.ID] {
				errs = append(errs, fmt.Errorf("%s.id: duplicate supplement %q", sat, sup.ID))
			}
			seenSup[sup.ID] = true
			if err := sup.check(sat); err != nil {
				errs = append(errs, err)
			}
			for _, r := range sup.Records {
				if other, dup := seenRec[r.ID]; dup && other != sup.ID {
					errs = append(errs, fmt.Errorf("%s: record %d is also pinned by supplement %q; a record is pinned once", sat, r.ID, other))
				}
				seenRec[r.ID] = sup.ID
			}
			if src.Units == "" {
				errs = append(errs, fmt.Errorf("%s: a source with no unit list has nowhere to put a supplement's units", sat))
			}
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
	case "workflow_artifact":
		errs = append(errs, r.checkArtifact(at)...)
	default:
		errs = append(errs, fmt.Errorf("%s.kind: %q is not git_commit, hf_dataset_revision, release_assets or workflow_artifact", at, r.Kind))
	}
	return errors.Join(errs...)
}

// checkArtifact is D3 for a workflow artifact: the run is named by its numeric id and attempt,
// never by "latest" or a branch, the commit it evaluated is a full commit, and the artifact's
// zip and the one file the extractor reads are each pinned by SHA-256. An artifact is
// reproducible only if a reader can tell the bytes they downloaded are the ones pinned.
func (r Revision) checkArtifact(at string) []error {
	var errs []error
	if !runID.MatchString(r.Ref) {
		errs = append(errs, fmt.Errorf("%s.ref: %q is not a workflow run id (decimal digits)", at, r.Ref))
	}
	if r.RunAttempt < 1 {
		errs = append(errs, fmt.Errorf("%s.run_attempt: %d; name the attempt of the run", at, r.RunAttempt))
	}
	if err := CheckPinned(at+".artifact", r.Artifact); err != nil {
		errs = append(errs, err)
	}
	if !gitCommit.MatchString(r.Commit) {
		errs = append(errs, fmt.Errorf("%s.commit: %q is not the full 40-digit commit the run evaluated", at, r.Commit))
	}
	if !strings.Contains(r.URL, "/actions/runs/"+r.Ref) {
		errs = append(errs, fmt.Errorf("%s.url: %q is not the page of run %s", at, r.URL, r.Ref))
	}
	want := map[string]bool{r.Artifact + ".zip": false}
	for j, a := range r.Assets {
		if a.Name == "" || !hex64.MatchString(a.SHA256) {
			errs = append(errs, fmt.Errorf("%s.assets[%d]: needs a name and a 64-digit sha256", at, j))
		}
		if _, ok := want[a.Name]; ok {
			want[a.Name] = true
		}
	}
	for name, found := range want {
		if !found {
			errs = append(errs, fmt.Errorf("%s.assets: the artifact's zip %q is not pinned by sha256", at, name))
		}
	}
	if len(r.Assets) < 2 {
		errs = append(errs, fmt.Errorf("%s.assets: pin the artifact's zip and each file read from it", at))
	}
	return errs
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
